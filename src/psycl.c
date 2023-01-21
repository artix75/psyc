/*
 * Copyright (C) 2016-2022 Fabio Nicotra <artix2 at gmail dot com>.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the copyright holder. The name of the
 * copyright holder may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <assert.h>
#include "psyc.h"
#include "config.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "optimization.h"
#include "activation.h"
#include "mnist.h"
#include "cifar.h"
#include "log.h"
#include "debug.h"

#ifdef HAS_MAGICK
#include "image_data.h"
#endif

#define PROGRAM_NAME        "PsyC CLI"
#define NETWORK_NAME        "CLI Network"

#define CONV_FEATURE_COUNT  20
#define CONV_REGION_SIZE    5
#define POOL_REGION_SIZE    2

#define MAX_FILENAME_LEN    255
#define EPOCHS              30
#define LEARNING_RATE       1.5
#define BATCH_SIZE          10

#define MNIST_TRAIN_IMAGES  0
#define MNIST_TRAIN_LABELS  1
#define MNIST_TEST_IMAGES   2
#define MNIST_TEST_LABELS   3

#define CONFIG_MAX_LINE     1024
#define CONFIG_MAX_TOKENS   500

#define CMD_MAX_LEN        4096

#define TRAIN_EVENT_BATCH   1
#define TRAIN_EVENT_EPOCH   2

#define UNUSED(V) ((void) V)

static char* MNIST_FILE_NAMES[4] = {
    "resources/train-images-idx3-ubyte.gz",
    "resources/train-labels-idx1-ubyte.gz",
    "resources/t10k-images-idx3-ubyte.gz",
    "resources/t10k-labels-idx1-ubyte.gz"
};

static char MNISTDataFiles[4][PATH_MAX + 1] = {
    "\x0", "\x0", "\x0", "\x0"
};

/* Globals */

PSFloat *training_data = NULL;
PSFloat *test_data = NULL;
PSFloat *validation_data = NULL;
int testlen = 0;
int datalen = 0;
int valdlen = 0;
int train_dataset_len = 0;
int eval_dataset_len = 0;
int epochs = EPOCHS;
PSFloat learning_rate = LEARNING_RATE;
PSFloat l1_decay = 0.0;
PSFloat l2_decay = 0.0;
PSFloat momentum = 0.0;
PSOptimization optimization = PSDefaultOptimization;
int validate_every = 0;
int batch_size = BATCH_SIZE;
char outputFile[PATH_MAX];
int training_flags = 0;
char *on_batch_trained = NULL;
char *on_epoch_trained = NULL;
int batch_script_every = 1;
#ifdef HAS_MAGICK
char *image_filename = NULL;
char *image_dump_filename = NULL;
char *image_bgcolor = "white";
int image_invert = 0;
int image_grayscale = 0;
#endif
PSNeuralNetwork *network = NULL;

/* Forward declarations */
void printHelp(const char* program_path);
int parseOptionsFromFile(const char *filename);
PSLossFunction getLossFunctionByName(char *name);
void onBatchTrained(PSNeuralNetwork *network, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data);
void onEpochTrained(PSNeuralNetwork *network, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data);
static void cleanup(void);


/* Helper Functions */
static void toLowerCase(char *str) {
    if (str == NULL) return;
    char *p = str;
    int idx = 0;
    while (*p != 0) {
        char c = (char) tolower(str[idx]);
        str[idx++] = c;
        p++;
    }
}

static char *getPsycPath(char *executable) {
    static char path[PATH_MAX + 1] = "\x0";
    char _realpath[PATH_MAX + 1];
    if (path[0]) return path;
    _realpath[0] = 0;
    if (realpath(executable, _realpath) != NULL) {
        char *dir = dirname(_realpath);
        if (dir == NULL) return NULL;
        dir = dirname(dir);
        if (dir == NULL) return NULL;
        int len = strlen((const char*) dir);
        if (len >= PATH_MAX) {
            fprintf(stderr, "WARN: getPsycPath(): dirname length > %d",
                    PATH_MAX);
            return NULL;
        }
        memcpy(path, dir, len);
        path[len] = 0;
        return path;
    } else {
        char *syspath = getenv("PATH");
        if (syspath == NULL) return NULL;
        int execlen = strlen(executable);
        char *p = syspath;
        while ((p = strchr(p, ':'))) {
            size_t len = p - syspath;
            if (len > 0) {
                if (len > PATH_MAX) {
                    fprintf(stderr, "WARN: ENV['PATH'] path length > %d",
                            PATH_MAX);
                    return NULL;
                }
                char spath[PATH_MAX + 1] = "\x0";
                char *s = spath;
                memcpy(spath, syspath, len);
                if (spath[len - 1] != '/') spath[len++] = '/';
                s += len;
                len = len + execlen;
                if (len > PATH_MAX) {
                    fprintf(stderr, "WARN: path length > %d",
                            PATH_MAX);
                    return NULL;
                }
                memcpy(s, executable, execlen);
                spath[len] = 0;
                struct stat file_stat;
                int exists = lstat(spath, &file_stat);
                if (exists >= 0) {
                    _realpath[0] = 0;
                    if (realpath(spath, _realpath) != NULL) {
                        char *dir = dirname(_realpath);
                        if (dir == NULL) return NULL;
                        dir = dirname(dir);
                        if (dir == NULL) return NULL;
                        int len = strlen((const char*) dir);
                        if (len >= PATH_MAX) {
                            fprintf(stderr, "WARN: getPsycPath(): dirname "
                                    "length > %d",
                                    PATH_MAX);
                            return NULL;
                        }
                        memcpy(path, dir, len);
                        path[len] = 0;
                        return path;
                    }
                }
            }
            p++;
            syspath = p;
        }
        return NULL;
    }
}

static int resolveMNISTDataFiles(char *path) {
    if (!path) return 0;
    if (MNISTDataFiles[0][0]) return 1;
    int pathlen = strlen(path), i;
    for (i = 0; i < 4; i++) {
        char *mnist_fname = MNIST_FILE_NAMES[i];
        char *mnist_path = MNISTDataFiles[i];
        strcpy(mnist_path, path);
        if (mnist_path[pathlen - 1] != '/')
            strcat(mnist_path, "/");
        strcat(mnist_path, mnist_fname);
        /* printf("[%d] %s\n", i, mnist_path); */
    }
    return 1;
}

static PSLayerType getLayerType(char *name, int *is_cifar) {
    if (strcasecmp("fully_connected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("Fully Connected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("FullyConnected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("fc", name) == 0)
        return FullyConnected;
    else if (strcasecmp("input", name) == 0)
        return FullyConnected;
    else if (strcasecmp("SoftMax", name) == 0)
        return SoftMax;
    else if (strcasecmp("convolutional", name) == 0)
        return Convolutional;
    else if (strcasecmp("pooling", name) == 0)
        return Pooling;
    else if (strcasecmp("softmax", name) == 0)
        return SoftMax;
    else if (strcasecmp("recurrent", name) == 0)
        return Recurrent;
    else if (strcasecmp("lstm", name) == 0)
        return LSTM;
    else if (strcasecmp("cifar", name) == 0) {
        *is_cifar = 1;
        return FullyConnected;
    } else {
        fprintf(stderr, "Unkown layer type '%s'\n", name);
        cleanup();
        exit(1);
    }
}

static void getTempFileName(const char *prefix, char *buffer) {
    FILE *urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x.data",
            prefix,
            (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3]);
    fclose(urand);
}

static int loadMNISTData(int data_type, int argc, char **argv, int *arg_idx) {
    int i = *arg_idx, image_data_index, label_data_index;
    int *len = NULL;
    PSFloat **data = NULL;
    char *descr = NULL;
    assert(data_type == DATA_TYPE_TRAINING || data_type == DATA_TYPE_TEST);
    if (data_type == DATA_TYPE_TRAINING) {
        image_data_index = MNIST_TRAIN_IMAGES;
        label_data_index = MNIST_TRAIN_LABELS;
        len = &datalen;
        data = &training_data;
        descr = "training";
    } else {
        image_data_index = MNIST_TEST_IMAGES;
        label_data_index = MNIST_TRAIN_LABELS;
        len = &testlen;
        data = &test_data;
        descr = "test";
    }
    char *imgfile = NULL;
    char *lblfile = NULL;
    if ((i + 2) < argc) {
        imgfile = argv[++i];
        lblfile = argv[++i];
        if (imgfile[0] == '-') {
            imgfile = NULL;
            lblfile = NULL;
            i -= 2;
        } else *arg_idx = i;
    }
    if (imgfile == NULL) {
        if (!resolveMNISTDataFiles(getPsycPath(argv[0]))) {
            fprintf(stderr, "Missing MNIST %s data files\n", descr);
            return 0;
        }
        imgfile = MNISTDataFiles[image_data_index];
        lblfile = MNISTDataFiles[label_data_index];
    }
    if (imgfile != NULL && lblfile != NULL)
        *len = PSLoadMNISTData(data_type, imgfile, lblfile, data);
    if (*len == 0 || *data == NULL) {
        fprintf(stderr, "Could not load %s data!\n", descr);
        return 0;
    }
    return 1;
}

static int loadCIFARData(int data_type, int classes, int argc, char **argv,
                         int *arg_idx)
{
    assert(data_type == DATA_TYPE_TRAINING || data_type == DATA_TYPE_TEST);
    int i = *arg_idx;
    int *len = NULL, *dataset_len = NULL;
    PSFloat **data = NULL;
    if (data_type == DATA_TYPE_TRAINING) {
        len = &datalen;
        data = &training_data;
        dataset_len = &train_dataset_len;
    } else {
        len = &testlen;
        data = &test_data;
    }
    char datapath[PATH_MAX + 1];
    datapath[0] = '\x0';
    int max_images = 0, max_files = 0;
    for (; i < argc; i++) {
        char *arg = argv[i];
        int last_arg = (i == (argc - 1));
        int *max_p = NULL;
        if (strcmp("--max-images", arg) == 0 && !last_arg)
            max_p = &max_images;
        else if (strcmp("--max-files", arg) == 0 && !last_arg)
            max_p = &max_files;
        else {
            struct stat file_stat;
            int exists = lstat(arg, &file_stat);
            if (exists >= 0) {
                if (strlen(arg) >= PATH_MAX) {
                    fprintf(stderr, "File path exceeds maximum length\n");
                    return 0;
                }
                strcpy(datapath, arg);
                *arg_idx = i;
                continue;
            } else break;
        }
        if (max_p == NULL) break;
        char *next = argv[++i];
        if (next == NULL) break;
        if (next[0] == '-') {
            fprintf(stderr, "ERROR: %s requires an integer argument\n",
                    arg);
            cleanup();
            exit(1);
        }
        *max_p = atoi(next);
        if (*max_p < 0) *max_p = 0;
        *arg_idx = i;
    }
    if (datapath[0] == 0) {
        char *psych_path = getPsycPath(argv[0]);
        if (psych_path != NULL) {
            char default_dirname[PATH_MAX + 1];
            int pathlen = strlen(psych_path);
            strcpy(datapath, psych_path);
            if (datapath[pathlen - 1] != '/') strcat(datapath, "/");
            sprintf(default_dirname, "resources/cifar-%d-batches-bin", classes);
            strcat(datapath, default_dirname);
            struct stat file_stat;
            int exists = lstat(datapath, &file_stat);
            if (exists < 0) {
                fprintf(stderr, "CIFAR data not found at: '%s'\n", datapath);
                return 0;
            }
        }
    }
    int datasize = PSLoadCIFARData(
        data_type, classes, datapath, data, max_files, max_images
    );
    *len = datasize / sizeof(PSFloat);
    if (*len == 0 || *data == NULL) {
        fprintf(stderr, "Failed to load CIFAR data\n");
        return 0;
    }
    if (dataset_len != NULL) *dataset_len = (*len / CIFAR_IMAGE_SIZE);
    return 1;
}

static int loadData(int data_type, int argc, char **argv, int *arg_idx) {
    int mnist = 0, cifar = 0, i = *arg_idx;
    if (strcmp("--mnist", argv[i]) == 0) {
        mnist = 1;
        *arg_idx = ++i;
    } else if (strcmp("--cifar", argv[i]) == 0) {
        *arg_idx = ++i;
        if (i < argc) {
            int matched = sscanf(argv[i], "%d", &cifar);
            if (matched) ++i;
        }
        if (cifar <= 0) cifar = 10;
        if (cifar != 10 && cifar != 100) {
            fprintf(
                stderr, "Invalid cifar classes: 10 or 100 allowed\n"
            );
            return 0;
        }
        *arg_idx = i;
    }
    if (!mnist && cifar == 0) {
        fprintf(
            stderr, "Only MNIST or CIFAR data supported for "
            "%s ATM :(\n",
            (data_type == DATA_TYPE_TRAINING ? "training" : "testing")
        );
        return 0;
    } else {
        if (data_type == DATA_TYPE_TRAINING) {
            if (mnist) {
                train_dataset_len = 50000;
                eval_dataset_len = 10000;
            } else {
                train_dataset_len = 40000;
                eval_dataset_len = 10000;
            }
        }
    }
    if (mnist) return loadMNISTData(data_type, argc, argv, arg_idx);
    else return loadCIFARData(data_type, cifar, argc, argv, arg_idx);
}

static void cleanup(void) {
    if (on_epoch_trained != NULL) free(on_epoch_trained);
    if (on_batch_trained != NULL) free(on_batch_trained);
#ifdef HAS_MAGICK
    if (image_filename != NULL) free(image_filename);
#endif
    if (network != NULL) {
        if (network->name != (char *)NETWORK_NAME && network->name != NULL)
            free((void *)network->name);
        PSDeleteNetwork(network);
        network = NULL;
    }
}

static void printLogLevels(FILE *out) {
    if (out == NULL) out = stdout;
    int count = PSGetMaxLogLevel(), i;
    for (i = 0; i < count; i++) {
        char *lvlname = strdup(PSLogLevelName(i));
        toLowerCase(lvlname);
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "%s", lvlname);
        free(lvlname);
    }
}

void parseOptions(int argc, char **argv) {
    int i, j;
    for (i = 1; i < argc; i++) {
        /* printf("ARG[%d]: %s\n", i, argv[i]); */
        int is_last = (i == (argc - 1));
        char *arg = argv[i];
        if ((strcmp("--config", arg) == 0 || strcmp("-c", arg) == 0) &&
            !is_last)
        {
            char *configfile = argv[++i];
            if (!parseOptionsFromFile(configfile)) {
                fprintf(stderr,
                    "FATAL: failed to load configuration file '%s'\n",
                    configfile
                );
                goto err;
            }
        } else if (strcmp("--load", arg) == 0 && !is_last) {
            char *file = argv[++i];
            int loaded = PSLoadNetwork(network, file);
            if (!loaded) {
                fprintf(stderr, "Could not load pretrained network %s\n", file);
                goto err;
            }
        } else if (strcmp("--save", arg) == 0 && !is_last) {
            char *file = argv[++i];
            if (strlen(file) > PATH_MAX) {
                fprintf(
                    stderr, "--save filename length must be <= %d\n",
                    PATH_MAX
                );
                goto err;
            } else {
                snprintf(outputFile, PATH_MAX, "%s", file);
            }
        } else if (strcmp("--name", arg) == 0 && !is_last) {
            char *name = (char*) argv[++i];
            network->name = strdup(name);
        } else if (strcmp("--onehot", arg) == 0) {
            if (network->size == 0) network->flags |= FLAG_ONEHOT;
            else network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
        } else if (strcmp("--layer", arg) == 0 && !is_last) {
            char *type = argv[++i];
            int is_cifar = 0, lidx = network->size;
            PSLayerType ltype = getLayerType(type, &is_cifar);
            if ((i + 1) >= argc) break;
            PSLayer *layer = NULL;
            PSLayerDef ldef = {0};
            if (is_cifar) {
                layer = PSAddCIFARInputLayer(network);
                continue;
            }
            j = i + 1;

            for (; j < argc; j++) {
                char *carg = argv[j];
                if ((strcmp("--feature-count", carg) == 0 || /* Legacy */
                     strcmp("--output-depth", carg) == 0) && ++j < argc)
                {
                    int depth = 0;
                    char *fcstr = argv[j];
                    int matched = sscanf(fcstr, "%d", &depth);
                    if (!matched) {
                        fprintf(
                            stderr, "Invalid %s %s\n", carg, fcstr
                        );
                        goto err;
                    }
                    i = j;
                    ldef.output_depth = depth;
                } else if (strcmp("--output-width", carg) == 0 && ++j < argc) {
                    char *szstr = argv[j];
                    int matched = sscanf(
                        szstr, "%d", &(ldef.output_columns)
                    );
                    if (!matched) {
                        fprintf(
                            stderr, "Invalid %s %s\n", carg, szstr
                        );
                        goto err;
                    }
                    i = j;
                } else if (strcmp("--output-height", carg) == 0 && ++j < argc) {
                    char *szstr = argv[j];
                    int matched = sscanf(
                        szstr, "%d", &(ldef.output_rows)
                    );
                    if (!matched) {
                        fprintf(
                            stderr, "Invalid %s %s\n", carg, szstr
                        );
                        goto err;
                    }
                    i = j;
                } else if (strcmp("--activation", carg) == 0 && ++j < argc) {
                    char *actvname = argv[j];
                    if (strcasecmp("sigmoid", actvname) == 0)
                        ldef.activation = PSSigmoid;
                    else if (strcasecmp("tanh", actvname) == 0)
                        ldef.activation = PSTanhActivation;
                    else if (strcasecmp("relu", actvname) == 0)
                        ldef.activation = PSRelu;
                    else {
                        fprintf(stderr, "Invalid activation '%s'", actvname);
                        goto err;
                    }
                } else if ((strcmp("--region-size", carg) == 0 ||
                            strcmp("--filter-width", carg) == 0) &&
                            ++j<argc)
                {
                    int filter_w = 0;
                    char *rsstr = argv[j];
                    int matched = sscanf(rsstr, "%d", &filter_w);
                    if (!matched) {
                        fprintf(stderr, "Invalid %s %s\n", carg, rsstr);
                        goto err;
                    }
                    i = j;
                    ldef.filter_width = filter_w;
                } else if (strcmp("--filter-height", carg)==0 && ++j<argc) {
                    int filter_h = 0;
                    char *rsstr = argv[j];
                    int matched = sscanf(rsstr, "%d", &filter_h);
                    if (!matched) {
                        fprintf(stderr, "Invalid %s %s\n", carg, rsstr);
                        goto err;
                    }
                    i = j;
                    ldef.filter_height = filter_h;
                } else if (strcmp("--stride", carg) == 0 && ++j < argc) {
                    int stride = 0;
                    char *ststr = argv[j];
                    int matched = sscanf(ststr, "%d", &stride);
                    if (!matched) {
                        fprintf(stderr, "Invalid stride %s\n", ststr);
                        goto err;
                    }
                    i = j;
                    ldef.stride = stride;
                } else if (strcmp("--padding", carg) == 0 && ++j < argc) {
                    int padding = 0;
                    char *padstr = argv[j];
                    int matched = sscanf(padstr, "%d", &padding);
                    if (!matched) {
                        fprintf(stderr, "Invalid padding %s\n", padstr);
                        goto err;
                    }
                    i = j;
                    ldef.padding = padding;
                } else if (strcmp("--use-relu", carg) == 0) {
                    i = j;
                    ldef.activation = PSRelu;
                } else if (strcmp("--dropout", carg) == 0 && ++j < argc) {
                    char *dropout = argv[j];
                    int matched = sscanf(
                        dropout, PSFLOAT_FORMAT, &(ldef.dropout)
                    );
                    if (!matched) {
                        fprintf(
                            stderr, "Invalid dropout %s\n", dropout
                        );
                        goto err;
                    }
                    i = j;
                } else if (strcmp("--recurrent-layer", carg) == 0) {
                    ldef.flags |= FLAG_RECURRENT;
                } else break;
            }
            int size = 0;
            if (ltype != Convolutional && ltype != Pooling) {
                char *sizestr = argv[++i];
                int matched = sscanf(sizestr, "%d", &size);
                if (!matched) {
                    fprintf(stderr, "Invalid size %s\n", sizestr);
                    goto err;
                }
                j = i + 1;
            }
            layer = PSAddLayer(network, ltype, size, &ldef);
            if (layer == NULL) {
                fprintf(
                    stderr, "FATAL: Failed to create layer %d\n", lidx
                );
                goto err;
            }
            continue;
        } else if (strcmp("--train", arg) == 0 && ++i < argc) {
            if (!loadData(DATA_TYPE_TRAINING, argc, argv, &i)) goto err;
        } else if (strcmp("--test", arg) == 0 && ++i < argc) {
            if (!loadData(DATA_TYPE_TEST, argc, argv, &i)) goto err;
        }
#ifdef HAS_MAGICK
        else if (strcmp("--classify-image", arg) == 0 && ++i < argc) {
            image_filename = strdup(argv[i]);
            /* printf("Classifying %s...\n", image_filename); */
            int j = i;
            while (++j < argc) {
                char *imgarg = argv[j];
                if (strcmp("--grayscale", imgarg) == 0) image_grayscale = 1;
                else if (strcmp("--invert", imgarg) == 0) image_invert = 1;
                else if (strcmp("--background-color",imgarg) == 0 && ++j<argc){
                    image_bgcolor = argv[j];
                }
                else if (strcmp("--dump-image",imgarg) == 0 && ++j<argc) {
                    image_dump_filename = argv[j];
                } else  break;
            }
        }
#endif
        else if (strcmp("--training-datalen", arg) == 0 && ++i < argc) {
            char *len_s = argv[i];
            int matched = sscanf(len_s, "%d", &train_dataset_len);
            if (!matched) {
                fprintf(stderr, "Invalid train. data len. %s\n", len_s);
                goto err;
            }
        } else if (strcmp("--validation-datalen", arg) == 0 && ++i < argc) {
            char *len_s = argv[i];
            int matched = sscanf(len_s, "%d", &eval_dataset_len);
            if (!matched) {
                fprintf(stderr, "Invalid valid. data len. %s\n", len_s);
                goto err;
            }
        } else if (strcmp("--epochs", arg) == 0 && ++i < argc) {
            char *len_s = argv[i];
            int matched = sscanf(len_s, "%d", &epochs);
            if (!matched) {
                fprintf(stderr, "Invalid epochs %s\n", len_s);
                goto err;
            }
        } else if (strcmp("--batch-size", arg) == 0 && ++i < argc) {
            char *len_s = argv[i];
            int matched = sscanf(len_s, "%d", &batch_size);
            if (!matched) {
                fprintf(stderr, "Invalid batch size %s\n", len_s);
                goto err;
            }
        } else if (strcmp("--learning-rate", arg) == 0 && ++i < argc) {
            char *lr = argv[i];
            int matched = sscanf(lr, PSFLOAT_FORMAT, &learning_rate);
            if (!matched) {
                fprintf(stderr, "Invalid learning rate %s\n", lr);
                goto err;
            }
        } else if (strcmp("--l1-decay", arg) == 0 && ++i < argc) {
            char *l1d = argv[i];
            int matched = sscanf(l1d, PSFLOAT_FORMAT, &l1_decay);
            if (!matched) {
                fprintf(stderr, "Invalid l1 decay %s\n", l1d);
                goto err;
            }
        } else if (strcmp("--l2-decay", arg) == 0 && ++i < argc) {
            char *l2d = argv[i];
            int matched = sscanf(l2d, PSFLOAT_FORMAT, &l2_decay);
            if (!matched) {
                fprintf(stderr, "Invalid l2 decay %s\n", l2d);
                goto err;
            }
        } else if (strcmp("--weight-decay", arg) == 0) {
            training_flags |= TRAINING_WEIGHT_DECAY;
        } else if (strcmp("--momentum", arg) == 0 && ++i < argc) {
            char *momentumstr = argv[i];
            int matched = sscanf(momentumstr, PSFLOAT_FORMAT, &momentum);
            if (!matched) {
                fprintf(stderr, "Invalid momentum %s\n", momentumstr);
                goto err;
            }
        } else if (strcmp("--optimization", arg) == 0 && !is_last) {
            char *optname = argv[++i];
            if (strcmp("adam", optname) == 0)
                optimization = PSAdamOptimization;
            else if (strcmp("adagrad", optname) == 0)
                optimization = PSAdaGradOptimization;
            else if (strcmp("adadelta", optname) == 0)
                optimization = PSAdaDeltaOptimization;
            else if (strcmp("windowgrad", optname) == 0)
                optimization = PSWindowGradOptimization;
            else if (strcmp("nesterov", optname) == 0)
                optimization = PSNesterovOptimization;
            else {
                fprintf(stderr, "Invalid optmization `%s`\n", optname);
                fprintf(
                    stderr, "Valid values: adam, adagrad, adadelta, "
                    "windowgrad, nesterov\n"
                );
                goto err;
            }
        } else if (strcmp("--loss-function", arg) == 0 && !is_last) {
            char *funcname = argv[++i];
            PSLossFunction func = getLossFunctionByName(funcname);
            if (func == NULL) {
                fprintf(stderr, "ERROR: invalid loss function name '%s', "
                        "use --help to see valid function names\n", funcname);
                goto err;
            }
            network->loss = func;
        } else if (strcmp("--validate-every", arg) == 0 && !is_last) {
            char *everystr = argv[++i];
            int matched = sscanf(everystr, "%d", &validate_every);
            if (!matched) {
                fprintf(stderr, "Invalid --validate-every %s\n", everystr);
                goto err;
            }
        } else if (strcmp("--training-no-shuffle", arg) == 0) {
            training_flags |= TRAINING_NO_SHUFFLE;
        } else if (strcmp("--training-adjust-rate", arg) == 0) {
            training_flags |= TRAINING_ADJUST_RATE;
        } else if (strcmp("--disable-avx", arg) == 0) {
            PSDisableAcceleration(&network->acceleration, PSAcceleration_AVX);
        } else if (strcmp("--disable-accelerate", arg) == 0 ||
                   strcmp("--disable-acf", arg) == 0)
        {
            PSDisableAcceleration(&network->acceleration, PSAcceleration_ACF);
        } else if (strcmp("--disable-blas", arg) == 0) {
            PSDisableAcceleration(&network->acceleration, PSAcceleration_BLAS);
        } else if (strcmp("--enable-colors", arg) == 0) {
            PSGlobalFlags |= FLAG_LOG_COLORS;
        } else if (strcmp("--quiet", arg) == 0) {
            PSLogLevel = PSLOGLEVEL_ERROR;
        } else if (strcmp("--verbose", arg) == 0) {
            PSLogLevel = PSLOGLEVEL_DEBUG;
        } else if (strcmp("--loglevel", arg) == 0 && !is_last) {
            char *lvlname = argv[++i];
            int level = PSLogLevelByName(lvlname);
            if (level < 0) {
                fprintf(stderr, "Invalid level: '%s'\n", lvlname);
                fprintf(stderr, "Available levels: ");
                printLogLevels(stderr);
                fprintf(stderr, "\n");
                goto err;
            }
            PSLogLevel = level;
        } else if (strcmp("--on-batch-trained", arg) == 0 && !is_last) {
            on_batch_trained = strdup(argv[++i]);
            if (strlen(on_batch_trained) > 0)
                network->onBatchTrained = onBatchTrained;
        } else if (strcmp("--on-epoch-trained", arg) == 0 && !is_last) {
            on_epoch_trained =strdup( argv[++i]);
            if (strlen(on_epoch_trained) > 0)
                network->onEpochTrained = onEpochTrained;
        } else if (strcmp("--batch-script-every", arg) == 0 && !is_last) {
            char *every = argv[++i];
            int matched = sscanf(every, "%d", &batch_script_every);
            if (!matched) {
                fprintf(
                    stderr, "ERROR: invalid value for --batch-script-every: "
                    "'%s'\n", every
                );
                goto err;
            }
        } else if (strcmp("-v", arg) == 0 || strcmp("--version", arg) == 0) {
            printf("%s v%s (AVX=", PROGRAM_NAME, PSYC_VERSION);
#ifdef USE_AVX
            printf("on");
#else
            printf("off");
#endif
            printf(",AccelerateFramework=");
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
            printf("on");
#else
            printf("off");
#endif
            int has_double_precision = (sizeof(PSFloat) == sizeof(double));
            printf(
                ",DOUBLE_PRECISION=%s)\n", (has_double_precision ? "on" : "off")
            );
            cleanup();
            exit(0);
        } else if (strcmp("-h", arg) == 0 || strcmp("--help", arg) == 0) {
            printHelp(argv[0]);
            cleanup();
            exit(1);
        } else {
            fprintf(stderr, "ERROR: invalid argument %s\n", arg);
            goto err;
        }
    }
    return;
err:
    cleanup();
    exit(1);
}

int parseOptionsFromFile(const char *filename) {
    FILE *f;
    if (filename[0] == '-' || filename[0] == '\0') f = stdin;
    else {
        f = fopen(filename, "r");
        if (f == NULL) {
            fprintf(stderr, "Failed to open config file: '%s'\n", filename);
            return 0;
        }
    }
    int buflen = CONFIG_MAX_LINE + 1;
    char buf[buflen];
    int argc = 1, i = 0, linenum = 0, success = 1;
    char **argv = malloc(sizeof(char *));
    success = (argv != NULL);
    if (!success) goto cleanup;
    /* Insert an empty string since parseOptions always starts from index 1 */
    argv[0] = strdup("");
    while (fgets(buf, buflen, f) != NULL) {
        linenum++;
        int numtokens = 0, len = strlen(buf);
        char *tokens[CONFIG_MAX_TOKENS] = {0};
        char *p = buf, *line = NULL, *token = NULL;
        while (*p == ' ' || *p == '\r' || *p == '\n' || *p == '\t') {
            if (len-- <= 0) break;
            p++;
        }
        line = p;
        /* Search for comments */
        char *comment_start = strchr(line, '#');
        if (comment_start != NULL) *comment_start = '\0';
        if (strlen(line) == 0) goto next_line;
        token = strtok(line, " \r\t");
        while (token != NULL) {
            if (numtokens >= CONFIG_MAX_TOKENS) break;
            char *blank = strpbrk(token, " \r\t\n");
            if (blank != NULL) *blank = '\x0';
            tokens[numtokens++] = strdup(token);
            token = strtok(NULL,  " \r\t");
        }
        if (numtokens == 0) goto next_line;
        toLowerCase(tokens[0]);
        /* Ignore single char options (ie. 'p' for '-p')*/
        int first_token_len = strlen(tokens[0]);
        if (first_token_len <= 1) goto next_line;
        int handled = 0;
        if (strcmp("include", tokens[0]) == 0) {
            success = numtokens > 1;
            if (!success) {
                fprintf(stderr, "Error in config file '%s', at line %d:\n"
                        "Mandatory FILENAME argument for "
                        "'include' directive\n", filename, linenum);
                goto next_line;
            }
            char *configfile = tokens[1];
            char relpath[PATH_MAX + 1];
            if (configfile[0] != '/') {
                relpath[0] = 0;
                char *dir = dirname((char *) filename);
                if (dir != NULL) {
                    strcpy(relpath, dir);
                    int pthlen = strlen(relpath);
                    if (relpath[pthlen - 1] != '/') strcat(relpath, "/");
                    strcat(relpath, configfile);
                    configfile = relpath;
                }
            }
            success = parseOptionsFromFile(configfile);
            if (!success) goto next_line;
            handled = 1;
        } else if (strcmp("help", tokens[0]) == 0) goto next_line;
        if (handled) goto next_line;
        int first_arg_len = 2 + first_token_len;
        size_t first_arg_size = (size_t) first_arg_len + 1;
        int from = argc;
        char *arg = malloc(sizeof(char) * first_arg_size);
        snprintf(arg, first_arg_size, "--%s", tokens[0]);
        if (numtokens > 1) {
            int yesno = 0;
            if (strcasecmp("yes", tokens[1]) == 0) {
                argc += 1;
                char **argvdup = argv;
                argv = realloc(argv, argc * sizeof(char *));
                if (argv == NULL) {
                    free(argvdup);
                    PSPrintMemoryErrorMsg();
                    exit(1);
                }
                argv[from] = arg;
                yesno = 1;
            } else if (strcasecmp("no", tokens[1]) == 0) yesno = 1;
            if (yesno) goto next_line;
        }
        argc += numtokens;
        char **argvdup = argv;
        argv = realloc(argv, argc * sizeof(char *));
        if (argv == NULL) {
            free(argvdup);
            PSPrintMemoryErrorMsg();
            exit(1);
        }
        argv[from] = arg;
        for (i = 1; i < numtokens; i++) {
            char *token = tokens[i];
            argv[from + i] = strdup(token);
        }
next_line:
        for (i = 0; i < numtokens; i++) {
            if (tokens[i] != NULL) free(tokens[i]);
        }
        if (!success) goto cleanup;
    }
    if (argc > 1) parseOptions(argc, argv);
cleanup:
    if (f != stdin) fclose(f);
    if (argv != NULL) {
        for (i = 0; i < argc; i++) free(argv[i]);
        free(argv);
    }
    return success;
}

/* Event functions */

void onTrainEvent(int event_type, PSNeuralNetwork *network, int epoch,
                  int epochs, PSFloat avg_loss, PSFloat current_loss,
                  float accuracy, PSFloat *rate)
{
    char *script = NULL, *event = NULL;
    if (event_type == TRAIN_EVENT_BATCH) {
        script = on_batch_trained;
        event = "batch";
    } else if (event_type == TRAIN_EVENT_EPOCH) {
        script = on_epoch_trained;
        event = "epoch";
    }
    if (script == NULL) return;
    char cmd[CMD_MAX_LEN];
    cmd[0] = 0;
    int written = snprintf(
        cmd, CMD_MAX_LEN,
        "%s --event %s-trained --name '%s' --epoch %d --epochs %d "
        "--average-loss %g --current-loss %g --accuracy %g --learning-rate %g",
        on_batch_trained, event, network->name, epoch, epochs, avg_loss,
        current_loss, accuracy, *rate
    );
    if (written >= CMD_MAX_LEN) {
        fprintf(stderr, "\nWARN: onBatchTrained command is too big!\n");
        return;
    }
    PSTrainingInfo *info = network->training;
    if (info != NULL) {
        char *p = cmd + written;
        written += snprintf(
            p, CMD_MAX_LEN,
            " --batch %d --element %d",
            info->current_batch, info->current_element
        );
    }
    if (written >= CMD_MAX_LEN) {
        fprintf(stderr, "\nWARN: onBatchTrained command is too big!\n");
        return;
    }
    int status = system(cmd);
    if (status != 0) {
        fprintf(
            stderr, "\nWARN: onBatchTrained script exited with status %d\n",
            status
        );
    }
}

void onBatchTrained(PSNeuralNetwork *network, int epoch, int epochs,
                    PSFloat avg_loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{

    UNUSED(training_data);
    if (on_batch_trained == NULL) return;
    PSTrainingInfo *info = network->training;
    if (batch_script_every > 0 && info) {
        if ((info->current_batch % batch_script_every) != 0) return;
    }
    onTrainEvent(
        TRAIN_EVENT_BATCH, network, epoch, epochs, avg_loss, current_loss,
        accuracy, rate
    );
}

void onEpochTrained(PSNeuralNetwork *network, int epoch, int epochs,
                    PSFloat avg_loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{
    UNUSED(training_data);
    if (on_epoch_trained == NULL) return;
    onTrainEvent(
        TRAIN_EVENT_EPOCH, network, epoch, epochs, avg_loss, current_loss,
        accuracy, rate
    );
}

int main(int argc, char **argv) {
    PSHandleSignals(NULL);
    network = PSCreateNetwork(NETWORK_NAME);
    if (network == NULL) {
        fprintf(stderr, "Failed to create network");
        return 1;
    }
    outputFile[0] = 0;
    parseOptions(argc, argv);
    if (PSLogLevel <= PSLOGLEVEL_INFO) PSPrintNetworkInfo(network);
    if (training_data != NULL) {
        int element_size = network->input_size + network->output_size;
        int element_count = datalen / element_size;
        if (element_count < train_dataset_len) {
            fprintf(stderr, "Loaded dataset elements %d < %d\n", element_count,
                   train_dataset_len);
            cleanup();
            return 1;
        } else {
            int remaining = element_count - train_dataset_len;
            if (remaining < eval_dataset_len && eval_dataset_len > 0) {
                fprintf(stderr, "WARNING: eval. dataset cannot be > %d!\n",
                        remaining);
                eval_dataset_len = remaining;
            }
            if (remaining == 0) {
                fprintf(stderr,
                        "WARNING: no dataset remaining for evaluation!\n");
                eval_dataset_len = remaining;
            }
            datalen = train_dataset_len *element_size;
            if (eval_dataset_len == 0) validation_data = NULL;
            else {
                validation_data = training_data + datalen;
                valdlen = eval_dataset_len *element_size;
            }
        }

        PSTrainingOptions options = {
            .flags = training_flags,
            .l1_decay = (PSFloat) l1_decay,
            .l2_decay = (PSFloat) l2_decay,
            .momentum = (PSFloat) momentum,
            .optimization = optimization,
            .validate_every_batches = validate_every
        };
        PSTrain(network, training_data, datalen, epochs, learning_rate,
                batch_size, &options, validation_data, valdlen);
        free(training_data);
    }
    if (test_data != NULL) {
        PSTest(network, test_data, testlen);
        free(test_data);
    }

#ifdef HAS_MAGICK
    if (image_filename != NULL) {
        int res = PSClassifyImage(network, image_filename, image_grayscale,
                                  image_invert, image_bgcolor,
                                  image_dump_filename);
        if (res >= 0) {
            printf("Classify result: %d\n", res);
        }
    }
#endif

    int outfile_len = strlen(outputFile);
    if (training_data != NULL || outfile_len) {
        if (!outfile_len) {
            getTempFileName("saved-network", outputFile);
        }
        int saved = PSSaveNetwork(network, outputFile);
        if (!saved) {
            fprintf(stderr, "Could not save network to %s\n", outputFile);
        } else {
            printf("Network saved to %s\n", outputFile);
        }
    }
    cleanup();
    return 0;
}

void printLossFunctionName(const char *name, PSLossFunction func) {
    UNUSED(func);
    printf("        %s\n", name);
}

PSLossFunction getLossFunctionByName(char *name) {
    if (strcasecmp("quadratic", name) == 0) return PSQuadraticLoss;
    else if (strcasecmp("cross-entropy", name) == 0) return PSCrossEntropyLoss;
    else if (strcasecmp("cross_entropy", name) == 0) return PSCrossEntropyLoss;
    else if (strcasecmp("cross entropy", name) == 0) return PSCrossEntropyLoss;
    else if (strcasecmp("crossentropy", name) == 0) return PSCrossEntropyLoss;
    return NULL;
}

void printHelp(const char* program_path) {
    printf("Usage: %s [OPTIONS]\n\n", program_path);
    printf("OPTIONS:\n\n");
    printf("    -c, --config FILE               Load options from FILE\n");
    printf("        --load PRETRAINED           Load a pretrained network\n");
    printf("        --save FILE                 Save network\n");
    printf("        --name NAME                 Network name\n");
    printf("        --layer TYPE SIZE|OPTIONS   Add layer\n");
    printf("        --onehot                    "
           "Sets one-hot-vector flag for input\n");
    printf("                                    "
           "(if before 1st layer) or desired output\n");
    printf("                                    (if after output layer)\n");
    printf("        --train [OPT] TRAIN_DATASET Train network\n");
    printf("        --test [OPT] TEST_DATASET   Perform tests\n");
#ifdef HAS_MAGICK
    printf("        --classify-image FILE [OPT] Perform tests\n");
#endif
    printf("        --training-datalen LEN      Training data length\n");
    printf("        --validation-datalen LEN    Validation data length\n");
    printf("        --epochs EPOCHS             Training epochs (def. %d)\n",
           EPOCHS);
    printf("        --batch-size SIZE           Train. batch size (def. %d)\n",
           BATCH_SIZE);
    printf("        --learning-rate SIZE        Train. learn rate (def. %f)\n",
           LEARNING_RATE);
    printf("        --momentum MOMENTUM         Momentum (def. 0)\n");
    printf("        --l1-decay SIZE             L1 Decay (def. 0)\n");
    printf("        --l2-decay SIZE             L2 Decay (def. 0)\n");
    printf("        --weight-decay              Enable L1/L2 weight decay\n"
           "                                    instead of L1/L2 "
           "regularization\n");
    printf("        --optimization              Training Optimization\n"
           "                                    (adagrad,adadelta,adam,\n"
           "                                     windowgrad,nesterov)\n"
    );
    printf("        --training-no-shuffle       Prevent dataset shuffle\n");
    printf("        --training-adjust-rate      Auto-adjust learn rate\n");
    printf("        --loss-function FUNC        Loss Function\n");
    printf("        --validate-every BATCH_NUM  Validate inside epochs\n");
    printf("        --on-batch-trained SCRIPT   Execute script after every\n"
           "                                    batch is trained.\n"
           "                                    (See \"SCRIPTS\" section for\n"
           "                                    more info)\n");
    printf("        --on-epoch-trained SCRIPT   Execute script after every\n"
           "                                    epoch is trained\n"
           "                                    (See \"SCRIPTS\" section for\n"
           "                                    more info)\n");
    printf("        --batch-script-every NUM    Execute script specified by\n"
           "                                    --on-batch-trained every\n"
           "                                    NUM batches\n");
    printf("        --disable-avx               Disable AVX\n");
    printf("        --disable-accelerate,\n"
           "        --disable-acf               Disable Accelerate "
           "Framework\n");
    printf("        --disable-blas              Disable BLAS\n");
    printf("        --loglevel LEVEL            Set log level "
           "(see \"LOG LEVELS\" section)\n");
    printf("        --quiet                     Quiet output (loglevel ERROR)"
           "\n");
    printf("        --verbose                   Verbose output (loglevel DEBUG)"
           "\n");
    printf("        --enable-colors             Colorized output\n");
    printf("    -v, --version                   Print version\n");
    printf("    -h, --help                      Print this help\n");
    printf("\n");
    printf("LAYER TYPES:\n\n");
    int i;
    for (i = 0; i < LAYER_TYPES; i++) {
        PSLayerType type = (PSLayerType) i;
        printf("        %s\n", PSGetLabelForType(type));
    }
    printf("\n");
    printf("LOSS FUNCTIONS:\n\n");
    PSIterateLossFunctions(printLossFunctionName);
    printf("\n");
    printf("LAYER OPTIONS:\n\n");
    printf("        --activation FUNC         Activation Function:\n"
           "                                  (sigmoid,tanh,relu)\n");
    printf("        --dropout DROPOUT         Layer Dropout (float)\n");
    printf("        --recurrent-layer         Recurrent layer mode\n");
    printf("        --output-width WIDTH      Output Width\n");
    printf("        --output-height HEIGHT    Output Height\n");
    printf("        --output-depth DEPTH      Output Depth\n");
    /*       " (def. %d)\n", CONV_FEATURE_COUNT);*/
    printf("        --filter-width WIDTH      Convolutional filter width"
           " (def. %d)\n", CONV_REGION_SIZE);
    printf("        --filter-height HEIGHT    Convolutional filter height"
           " (def. %d)\n", CONV_REGION_SIZE);
    printf("        --stride STRIDE           Convolutional stride"
           " (def. 1)\n");
    printf("        --padding PADDING         Convolutional padding"
           " (def. 0)\n");
    /*printf("        --use-relu                Use ReLU activation (for "
           "Convolutional Layers)\n");*/
    printf("\n");
    printf("LOG LEVELS:\n\n");
    printf("        "); printLogLevels(stdout); printf("\n\n");
    printf("TRAIN|TEST OPTIONS:\n\n");
    printf("        --mnist                   Dataset format is MNIST\n");
    printf("        --cifar [CLASSES]         Dataset format is CIFAR\n"
           "                                  (classes: 10 or 100, default\n"
           "                                   is 10)\n"
    );
    printf("        --max-images              Max images to load (CIFAR)\n");
    printf("        --max-files               Max files to load (CIFAR)\n");
#ifdef HAS_MAGICK
    printf("\n");
    printf("IMAGE OPTIONS:\n\n");
    printf("        --grayscale              Convert image to grayscale\n");
    printf("        --invert                 Invert image pixels\n");
    printf("        --background-color COLOR Padding background color\n"
           "                                 (ie. none, white, ...),\n"
           "                                 default: white\n");
    printf("        --dump-image FILE        Save image to file\n");
#endif
    printf("\n");
    printf("SCRIPTS:\n\n");
    printf(
        "  Using options such as `--on-batch-trained` and `--on-epoch-trained`"
        "\n"
        "  it's possible to execute an arbitrary external script when such\n"
        "  events happen.The scripts will eventually receive the following\n"
        "  arguments:\n"
        "    --event TYPE, --name NETWORK_NAME --epoch CURRENT_EPOC\n"
        "    --epochs TOT_EPOCHS --average-loss AVERAGE_LOSS --current-loss\n"
        "    CURRENT_LOSS --accuracy CURRENT_ACCURACY --learning-rate RATE\n"
    );
    printf("\n");
}

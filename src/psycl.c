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
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "mnist.h"
#include "cifar.h"
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
float learning_rate = LEARNING_RATE;
float l1_decay = 0.0;
float l2_decay = 0.0;
float momentum = 0.0;
PSTrainingOptimization optimization = NoTrainingOptimization;
int validate_every = 0;
int batch_size = BATCH_SIZE;
char outputFile[255];
int training_flags = 0;
#ifdef HAS_MAGICK
char *image_filename = NULL;
char *image_dump_filename = NULL;
char *image_bgcolor = "white";
int image_invert = 0;
int image_grayscale = 0;
#endif
PSNeuralNetwork *network = NULL;

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

static PSLayerType getLayerType(char *name, PSNeuralNetwork *network,
                                int *is_cifar)
{
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
        fprintf(stderr, "Unkown layer type %s\n", name);
        PSDeleteNetwork(network);
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
    int *len = NULL;
    PSFloat **data = NULL;
    if (data_type == DATA_TYPE_TRAINING) {
        len = &datalen;
        data = &training_data;
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
            if (network != NULL) PSDeleteNetwork(network);
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

/* Forward declarations */
void printHelp(const char* program_path);
int parseOptionsFromFile(const char *filename);
PSLossFunction getLossFunctionByName(char *name);

/* Helper Functions */
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
            if (strlen(file) > 254) {
                fprintf(stderr, "--save filename length must be <= 254");
                goto err;
            } else {
                sprintf(outputFile, "%s", file);
            }
        } else if (strcmp("--name", arg) == 0 && !is_last) {
            char *name = (char*) argv[++i];
            network->name = name;
        } else if (strcmp("--onehot", arg) == 0) {
            if (network->size == 0) network->flags |= FLAG_ONEHOT;
            else network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
        } else if (strcmp("--layer", arg) == 0 && !is_last) {
            char *type = argv[++i];
            int is_cifar = 0;
            PSLayerType ltype = getLayerType(type, network, &is_cifar);
            if ((i + 1) >= argc) break;
            if (Convolutional == ltype) {
                PSHyperParameters *params = NULL;
                params = PSCreateConvolutionalParameters(CONV_FEATURE_COUNT,
                                                         CONV_REGION_SIZE,
                                                         1, 0, 0);
                PSFloat *lparams = params->parameters;
                for (j = i + 1; j < argc; j++) {
                    char *carg = argv[j];
                    if (strcmp("--feature-count", carg) == 0 && ++j < argc) {
                        int fcount = 0;
                        char *fcstr = argv[j];
                        int matched = sscanf(fcstr, "%d", &fcount);
                        if (!matched) {
                            fprintf(
                                stderr, "Invalid feature count %s\n",fcstr
                            );
                            goto err;
                        }
                        i = j;
                        lparams[PARAM_FEATURE_COUNT] = (PSFloat) fcount;
                    } else if (strcmp("--region-size", carg) == 0 && ++j<argc) {
                        int rsize = 0;
                        char *rsstr = argv[j];
                        int matched = sscanf(rsstr, "%d", &rsize);
                        if (!matched) {
                            fprintf(stderr, "Invalid region size %s\n", rsstr);
                            goto err;
                        }
                        i = j;
                        lparams[PARAM_REGION_SIZE] = (PSFloat) rsize;
                    } else if (strcmp("--stride", carg) == 0 && ++j < argc) {
                        int stride = 0;
                        char *ststr = argv[j];
                        int matched = sscanf(ststr, "%d", &stride);
                        if (!matched) {
                            fprintf(stderr, "Invalid stride %s\n", ststr);
                            goto err;
                        }
                        i = j;
                        lparams[PARAM_STRIDE] = (PSFloat) stride;
                    } else if (strcmp("--padding", carg) == 0 && ++j < argc) {
                        int padding = 0;
                        char *padstr = argv[j];
                        int matched = sscanf(padstr, "%d", &padding);
                        if (!matched) {
                            fprintf(stderr, "Invalid padding %s\n", padstr);
                            goto err;
                        }
                        i = j;
                        lparams[PARAM_PADDING] = (PSFloat) padding;
                    } else if (strcmp("--use-relu", carg) == 0) {
                        i = j;
                        lparams[PARAM_USE_RELU] = 1.0;
                    } else break;
                }
                PSAddConvolutionalLayer(network, params);
            } else if (Pooling == ltype) {
                PSHyperParameters *params = NULL;
                params = PSCreateConvolutionalParameters(0, POOL_REGION_SIZE,
                                                         POOL_REGION_SIZE,
                                                         0, 0);
                PSFloat *lparams = params->parameters;
                for (j = i + 1; j < argc; j++) {
                    char *carg = argv[j];
                    if (strcmp("--region-size", carg) == 0 && ++j < argc) {
                        int rsize = 0;
                        char *rsstr = argv[j];
                        int matched = sscanf(rsstr, "%d", &rsize);
                        if (!matched) {
                            fprintf(stderr, "Invalid region size %s\n", rsstr);
                            goto err;
                        }
                        lparams[PARAM_REGION_SIZE] = (PSFloat) rsize;
                        i = j;
                    } else break;
                }
                PSAddPoolingLayer(network, params);
            } else if (is_cifar) {
                PSAddCIFARInputLayer(network);
            } else {
                int size = 0;
                char *sizestr = argv[++i];
                int matched = sscanf(sizestr, "%d", &size);
                if (!matched) {
                    fprintf(stderr, "Invalid size %s\n", sizestr);
                    goto err;
                }
                PSHyperParameters *params = NULL;
                if (FullyConnected == ltype && (i + 1) < argc) {
                    int feature_count = 0, output_w = 0, output_h = 0;
                    for (j = i + 1; j < argc; j++) {
                        char *carg = argv[j];
                        if (strcmp("--feature-count",carg) == 0 && ++j < argc) {
                            char *fcstr = argv[j];
                            int matched = sscanf(fcstr, "%d", &feature_count);
                            if (!matched) {
                                fprintf(
                                    stderr, "Invalid feature count %s\n",fcstr
                                );
                                goto err;
                            }
                            i = j;
                        } else if (strcmp("--output-width", carg) == 0 &&
                                   ++j < argc)
                        {
                            char *szstr = argv[j];
                            int matched = sscanf(szstr, "%d", &output_w);
                            if (!matched) {
                                fprintf(
                                    stderr, "Invalid %s %s\n", carg, szstr
                                );
                                goto err;
                            }
                            i = j;
                        } else if (strcmp("--output-height", carg) == 0 &&
                                   ++j < argc)
                        {
                            char *szstr = argv[j];
                            int matched = sscanf(szstr, "%d", &output_h);
                            if (!matched) {
                                fprintf(
                                    stderr, "Invalid %s %s\n", carg, szstr
                                );
                                goto err;
                            }
                            i = j;
                        } else break;
                    }
                    if (feature_count > 0) {
                        if (output_h == 0) output_h = output_w;
                        params = PSCreateConvolutionalParameters(
                            (PSFloat) feature_count, 0, 0, 0, 0
                        );
                        params->parameters[PARAM_OUTPUT_WIDTH] =
                            (PSFloat) output_w;
                        params->parameters[PARAM_OUTPUT_HEIGHT] =
                            (PSFloat) output_h;
                    }
                }
                PSAddLayer(network, ltype, size, params);
            }
            continue;
        } else if (strcmp("--train", arg) == 0 && ++i < argc) {
            if (!loadData(DATA_TYPE_TRAINING, argc, argv, &i)) goto err;
        } else if (strcmp("--test", arg) == 0 && ++i < argc) {
            if (!loadData(DATA_TYPE_TEST, argc, argv, &i)) goto err;
        }
#ifdef HAS_MAGICK
        else if (strcmp("--classify-image", arg) == 0 && ++i < argc) {
            image_filename = argv[i];
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
            int matched = sscanf(lr, "%f", &learning_rate);
            if (!matched) {
                fprintf(stderr, "Invalid learning rate %s\n", lr);
                goto err;
            }
        } else if (strcmp("--l1-decay", arg) == 0 && ++i < argc) {
            char *l1d = argv[i];
            int matched = sscanf(l1d, "%f", &l1_decay);
            if (!matched) {
                fprintf(stderr, "Invalid l1 decay %s\n", l1d);
                goto err;
            }
        } else if (strcmp("--l2-decay", arg) == 0 && ++i < argc) {
            char *l2d = argv[i];
            int matched = sscanf(l2d, "%f", &l2_decay);
            if (!matched) {
                fprintf(stderr, "Invalid l2 decay %s\n", l2d);
                goto err;
            }
        } else if (strcmp("--weight-decay", arg) == 0) {
            training_flags |= TRAINING_WEIGHT_DECAY;
        } else if (strcmp("--momentum", arg) == 0 && ++i < argc) {
            char *momentumstr = argv[i];
            int matched = sscanf(momentumstr, "%f", &momentum);
            if (!matched) {
                fprintf(stderr, "Invalid momentum %s\n", momentumstr);
                goto err;
            }
        } else if (strcmp("--optimization", arg) == 0 && !is_last) {
            char *optname = argv[++i];
            if (strcmp("adam", optname) == 0) optimization = Adam;
            else if (strcmp("adagrad", optname) == 0) optimization = AdaGrad;
            else if (strcmp("adadelta", optname) == 0) optimization = AdaDelta;
            else if (strcmp("windowgrad", optname) == 0)
                optimization = WindowGrad;
            else if (strcmp("nesterov", optname) == 0) optimization = Nesterov;
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
            network->flags |= FLAG_AVX_DISABLED;
        } else if (strcmp("--enable-colors", arg) == 0) {
            PSGlobalFlags |= FLAG_LOG_COLORS;
        } else if (strcmp("-v", arg) == 0 || strcmp("--version", arg) == 0) {
            printf("%s v%s (AVX=", PROGRAM_NAME, PSYC_VERSION);
#ifdef USE_AVX
            printf("on");
#else
            printf("off");
#endif
            int has_double_precision = (sizeof(PSFloat) == sizeof(double));
            printf(
                ",DOUBLE_PRECISION=%s)\n", (has_double_precision ? "on" : "off")
            );
            if (network != NULL) PSDeleteNetwork(network);
            exit(0);
        } else if (strcmp("-h", arg) == 0 || strcmp("--help", arg) == 0) {
            printHelp(argv[0]);
            if (network != NULL) PSDeleteNetwork(network);
            exit(1);
        } else {
            fprintf(stderr, "ERROR: invalid argument %s\n", arg);
            goto err;
        }
    }
    return;
err:
        if (network != NULL) PSDeleteNetwork(network);
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
                goto cleanup;
            }
            char *configfile = tokens[1];
            if (configfile[0] != '/') {
                char relpath[PATH_MAX + 1];
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
            if (!success) goto cleanup;
            handled = 1;
        } else if (strcmp("help", tokens[0]) == 0) goto next_line;
        if (handled) goto next_line;
        int first_arg_len = 2 + first_token_len;
        size_t first_arg_size = (size_t) first_arg_len + 1;
        int from = argc;
        char *arg = malloc(first_arg_size * sizeof(char));
        snprintf(arg, first_arg_size, "--%s", tokens[0]);
        if (numtokens > 1) {
            int yesno = 0;
            if (strcasecmp("yes", tokens[1]) == 0) {
                argc += 1;
                argv = realloc(argv, argc * sizeof(char *));
                argv[from] = arg;
                yesno = 1;
            } else if (strcasecmp("no", tokens[1]) == 0) yesno = 1;
            if (yesno) goto next_line;
        }
        argc += numtokens;
        argv = realloc(argv, argc * sizeof(char *));
        argv[from] = arg;
        for (i = 1; i < numtokens; i++) {
            char *token = tokens[i];
            argv[from + i] = strdup(token);
        }
next_line:
        for (i = 0; i < numtokens; i++) {
            if (tokens[i] != NULL) free(tokens[i]);
        }
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

int main(int argc, char **argv) {
    PSHandleSignals(NULL);
    network = PSCreateNetwork(NETWORK_NAME);
    if (network == NULL) {
        fprintf(stderr, "Failed to create network");
        return 1;
    }
    outputFile[0] = 0;
    parseOptions(argc, argv);
    if (training_data != NULL) {
        int element_size = network->input_size + network->output_size;
        int element_count = datalen / element_size;
        if (element_count < train_dataset_len) {
            fprintf(stderr, "Loaded dataset elements %d < %d\n", element_count,
                   train_dataset_len);
            PSDeleteNetwork(network);
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

    PSDeleteNetwork(network);
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
    printf("OPTIONS:\n");
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
    printf("        --disable-avx               Disable AVX\n");
    printf("        --enable-colors             Colorized output\n");
    printf("    -v, --version                   Print version\n");
    printf("    -h, --help                      Print this help\n");
    printf("\n");
    printf("LAYER TYPES:\n");
    int i;
    for (i = 0; i < LAYER_TYPES; i++) {
        PSLayerType type = (PSLayerType) i;
        printf("        %s\n", PSGetLabelForType(type));
    }
    printf("\n");
    printf("LOSS FUNCTIONS:\n");
    PSIterateLossFunctions(printLossFunctionName);
    printf("\n");
    printf("LAYER OPTIONS:\n");
    printf("        --feature-count COUNT     Convolutional features"
           " (def. %d)\n", CONV_FEATURE_COUNT);
    printf("        --region-size SIZE        Convolutional region size"
           " (def. %d)\n", CONV_REGION_SIZE);
    printf("        --stride STRIDE           Convolutional region stride"
           " (def. 1)\n");
    printf("        --padding PADDING         Convolutional padding"
           " (def. 0)\n");
    printf("        --use-relu                Use ReLU activation (for "
           "Convolutional Layers)\n");
    printf("\n");
    printf("TRAIN|TEST OPTIONS:\n");
    printf("        --mnist                   Dataset format is MNIST\n");
    printf("        --cifar [CLASSES]         Dataset format is CIFAR\n"
           "                                  (classes: 10 or 100, default\n"
           "                                   is 10)\n"
    );
    printf("        --max-images              Max images to load (CIFAR)\n");
    printf("        --max-files               Max files to load (CIFAR)\n");
#ifdef HAS_MAGICK
    printf("\n");
    printf("IMAGE OPTIONS:\n");
    printf("        --grayscale              Convert image to grayscale\n");
    printf("        --invert                 Invert image pixels\n");
    printf("        --background-color COLOR Padding background color "
           "                                 (ie. none, white, ...),\n"
           "                                  default: white\n");
    printf("        --dump-image FILE        Save image to file\n");
#endif
    printf("\n");
}

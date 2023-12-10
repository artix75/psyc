/*
 * Copyright (C) 2016-2023 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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
#include <sys/utsname.h>
#include "psyc.h"
#include "config.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "attention.h"
#include "operator-layer.h"
#include "optimization.h"
#include "activation.h"
#include "dataset.h"
#include "log.h"
#include "debug.h"
#include "buildinfo.h"

#ifdef HAS_MAGICK
#include "image-data.h"
#endif

#define PROGRAM_NAME        PSYC_NAME " CLI"
#define MODEL_NAME          "CLI Model"

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

#define MAX_PROVIDERS       1024

#define UNUSED(V) ((void) V)

static char* MNIST_FILE_NAMES[4] = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
};

static char *MNISTDataFiles[4] = {NULL, NULL, NULL, NULL};

typedef struct {
    PSLayerType type;
    char **names;
    char *descr;
} PsyclCmdLayerType;

char *fcnames[] = {"fully-connected", "fc", "dense", NULL};
char *rnnnames[] = {"rnn", NULL};
char *oplayernames[] = {"operator", "op", NULL};
char *posencnames[] = {"positional-encoding", NULL};

static PsyclCmdLayerType CmdLayerTypes[] = {
    {FullyConnected, fcnames, "Fully Connected (dense) layer"},
    {Convolutional, NULL},
    {Pooling, NULL},
    {RNNLayer, rnnnames, "Basic Recurrent Layer"},
    {LSTM, NULL},
    {SoftMax, NULL},
    {GRU, NULL},
    {Dropout, NULL},
    {Embedding, NULL},
    {Normalization, NULL},
    {Attention, NULL},
    {OperatorLayer, oplayernames, "Operator Layer (add,concatenate,mul)"},
    {Linear, NULL},
    {PositionalEncoding, posencnames, "Position Encoding "
     "Layer"},
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
PSModel *model = NULL;

/* Forward declarations */
void printHelp(const char* program_path);
int parseOptionsFromFile(const char *filename);
PSLossFunction getLossFunctionByName(char *name);
void onBatchTrained(PSModel *model, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data);
void onEpochTrained(PSModel *model, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);
int PSIsLayerPlaceholder(PSLayer *layer);
static void cleanup(void);


/* Helper Functions */

static char *getPsycPath(char *executable) {
    static char path[PATH_MAX + 1] = {0};
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

static void printLayerTypeHelp(PSLayerType type) {
    assert(type < PS_LAYER_TYPES);
    PsyclCmdLayerType *type_info = CmdLayerTypes;//[type];
    type_info += type;
    assert(type_info != NULL);
    assert(type_info->type == type);
    char *lbl = PSGetLabelForType(type);
    assert(lbl != NULL);
    char optnames[512] = {0};
    char descr[512] = {0};
    int nameslen = 0, descrlen = 0;
    if (type_info->names != NULL) {
        char **name = type_info->names;
        int idx = 0, remaining = 512;
        char *p = optnames;
        while (*name != NULL) {
            int len = 0;
            if (idx > 0) {
                len += snprintf(p, remaining, ", ");
                p += len;
                remaining -= len;
                if (remaining <= 0) break;
                nameslen += len;
            }
            len += snprintf(p, remaining, "%s", *name);
            p += len;
            remaining -= len;
            nameslen += len;
            if (remaining <= 0) break;
            name++;
            idx++;
        }
    } else {
        nameslen = snprintf(optnames, 512, "%s", lbl);
        char *p = optnames;
        while (*p != 0) {
            char c = *p;
            if (isalpha(c)) *p = tolower(c);
            else if (isspace(c)) *p = '-';
            p++;
        }
    }
    if (type_info->descr != NULL)
        descrlen = snprintf(descr, 512, "%s", type_info->descr);
    else
        descrlen = snprintf(descr, 512, "%s Layer", lbl);
    char *sep = "  ", *descrindent = "";
    if ((2 + nameslen) > (40 - 2)) {
        sep = "\n" ;
        descrindent = "                                        ";
    }
    printf("  %-38s%s%s%s\n", optnames, sep, descrindent, descr);
}

static char *downloadMNISTDataset(char *path) {
    static char *host = "http://dia.fi.upm.es/~lbaumela/PracRF11";
    static char *dirname = "mnist";
    char url[PATH_MAX] = {0};
    char fpath[PATH_MAX] = {0};
    char *datasets_path = NULL;
    int provided_path = (path != NULL), success = 1;
    if (!provided_path) {
        const char *wdir = PSWorkingDirectory();
        if (wdir == NULL) return NULL;
        char *datasets_path = PSPathJoin(2, wdir, "datasets");
        if (datasets_path == NULL) return NULL;
        if (!PSFileExists(datasets_path))
            if (!PSMakeDir(datasets_path, 1)) goto final;
        path = PSPathJoin(2, datasets_path, dirname);
        if (path == NULL) goto final;
        if (!PSFileExists(path)) {
            success = PSMakeDir(path, 1);
            if (!success) goto final;
        }
    }
    size_t numfiles = sizeof(MNIST_FILE_NAMES) / sizeof(char *), i;
    for (i = 0; i < numfiles; i++) {
        snprintf(url, PATH_MAX, "%s/%s", host, MNIST_FILE_NAMES[i]);
        snprintf(fpath, PATH_MAX, "%s/%s", path, MNIST_FILE_NAMES[i]);
        PSNotice("Downloading MNIST file '%s'", MNIST_FILE_NAMES[i]);
        success = PSDownloadFile(url, path);
        if (!success) goto final;
    }
final:
    free(datasets_path);
    if (!success) {
        if (!provided_path) free(path);
        path = NULL;
    }
    return path;
}

static int findMNISTFiles(char **mnist_files, int *found) {
    const char *wdir = PSWorkingDirectory();
    if (wdir == NULL) return 0;
    int ok = 1, i;
    char *dataset_path = PSPathJoin(2, wdir, "datasets/mnist");
    if (dataset_path == NULL) return 0;
    char *fpath = NULL;
    *found = 0;
    for (i = 0; i < 4; i++) {
        fpath = PSPathJoin(2, dataset_path, MNIST_FILE_NAMES[i]);
        ok = (fpath != NULL);
        if (!ok) break;
        if (PSFileExists(fpath)) {
            mnist_files[i] = fpath;
            *found += 1;
        } else {
            free(fpath);
            fpath = NULL;
            ok = 0;
        }
    }
    free(dataset_path);
    return ok;
}

static PSLayerType getLayerType(char *name, int *is_cifar, PSLayerDef *ldef) {
    if (strcasecmp("fully_connected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("fully-connected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("Fully Connected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("FullyConnected", name) == 0)
        return FullyConnected;
    else if (strcasecmp("fc", name) == 0)
        return FullyConnected;
    else if (strcasecmp("dense", name) == 0)
        return FullyConnected;
    else if (strcasecmp("input", name) == 0)
        return FullyConnected;
    else if (strcasecmp("SoftMax", name) == 0)
        return SoftMax;
    else if (strcasecmp("softmax", name) == 0)
        return SoftMax;
    else if (strcasecmp("convolutional", name) == 0)
        return Convolutional;
    else if (strcasecmp("pooling", name) == 0)
        return Pooling;
    else if (strcasecmp("recurrent", name) == 0)
        return RNNLayer;
    else if (strcasecmp("rnn", name) == 0)
        return RNNLayer;
    else if (strcasecmp("lstm", name) == 0)
        return LSTM;
    else if (strcasecmp("gru", name) == 0)
        return GRU;
    else if (strcasecmp("dropout", name) == 0)
        return Dropout;
    else if (strcasecmp("embedding", name) == 0)
        return Embedding;
    else if (strcasecmp("normalization", name) == 0)
        return Normalization;
    else if (strcasecmp("attention", name) == 0)
        return Attention;
    else if (strcasecmp("operator", name) == 0)
        return OperatorLayer;
    else if (strcasecmp("op", name) == 0)
        return OperatorLayer;
    else if (strcasecmp("Positional Encoding", name) == 0)
        return PositionalEncoding;
    else if (strcasecmp("PositionalEncoding", name) == 0)
        return PositionalEncoding;
    else if (strcasecmp("positional_encoding", name) == 0)
        return PositionalEncoding;
    else if (strcasecmp("positional-encoding", name) == 0)
        return PositionalEncoding;
    else if (strcasecmp("positional", name) == 0)
        return PositionalEncoding;
    else if (strcasecmp("add", name) == 0) {
        ldef->operator = PSAddOperator;
        return OperatorLayer;
    } else if (strcasecmp("concatenate", name) == 0) {
        ldef->operator = PSConcatenateOperator;
        return OperatorLayer;
    } else if (strcasecmp("concat", name) == 0) {
        ldef->operator = PSConcatenateOperator;
        return OperatorLayer;
    } else if (strcasecmp("cifar", name) == 0) {
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
    int i = *arg_idx, ok = 1, image_data_index, label_data_index;
    int *len = NULL;
    PSFloat **data = NULL;
    char *descr = NULL;
    assert(
        data_type == PS_DATA_TYPE_TRAINING || data_type == PS_DATA_TYPE_TEST
    );
    if (data_type == PS_DATA_TYPE_TRAINING) {
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
        int found_files = 0;
        int found = findMNISTFiles(MNISTDataFiles, &found_files);
        if (!found) {
            PSNotice("could not find MNIST dataset files, trying to "
                     "download them...");
            char *dataset_path = downloadMNISTDataset(NULL);
            ok = (dataset_path != NULL);
            if (!ok) {
                PSErr(NULL, "could not download MNIST dataset files, please "
                      "provide their path (use --help for more info)");
                free(dataset_path);
                goto final;
            }
            found = findMNISTFiles(MNISTDataFiles, &found_files);
            free(dataset_path);
        }
        imgfile = MNISTDataFiles[image_data_index];
        lblfile = MNISTDataFiles[label_data_index];
        ok = (imgfile != NULL);
        if (!ok) {
            PSErr(NULL, "could not find MNIST image file '%s'\n", imgfile);
            goto final;
        }
        ok = (lblfile != NULL);
        if (!ok) {
            PSErr(NULL, "could not find MNIST label file '%s'\n", lblfile);
            goto final;
        }
    }
    if (imgfile != NULL && lblfile != NULL)
        *len = PSLoadMNISTData(data_type, imgfile, lblfile, data);
    if (*len == 0 || *data == NULL) {
        PSErr(NULL, "could not load %s data!\n", descr);
        ok = 0;
        goto final;
    }
final:
    for (i = 0; i < 4; i++) free(MNISTDataFiles[i]);
    return ok;
}

static char *downloadCIFARDataset(int classes, char *dest_dir) {
    if (classes != 10 && classes != 100) classes = 10;
    int success = 1, provided_dest = (dest_dir != NULL);
    char *path = NULL, *tarpath = NULL;
    char dirname[NAME_MAX] = {0};
    char url[PATH_MAX] = {0};
    char tarfname[NAME_MAX] = {0};
    char cmd[PATH_MAX * 3];
    if (!provided_dest) {
        const char *wdir = PSWorkingDirectory();
        if (wdir == NULL) return NULL;
        dest_dir = PSPathJoin(2, wdir, "datasets");
        if (dest_dir == NULL) return NULL;
        if (!PSFileExists(dest_dir))
            if (!PSMakeDir(dest_dir, 1)) goto final;
    }
    snprintf(dirname, NAME_MAX, "cifar-%d-batches-bin", classes);
    path = PSPathJoin(2, dest_dir, dirname);
    if (path == NULL) goto final;
    if (PSFileExists(path)) goto final;
    snprintf(tarfname, NAME_MAX, "cifar-%d-binary.tar.gz", classes);
    snprintf(
        url, PATH_MAX,"http://www.cs.toronto.edu/~kriz/%s", tarfname
    );
    PSNotice("Downloading CIFAR dataset (%d classes)", classes);
    success = PSDownloadFile(url, dest_dir);
    if (!success) goto final;
    tarpath = PSPathJoin(2, dest_dir, tarfname);
    success = (tarpath != NULL);
    if (!success) goto final;
    snprintf(
        cmd, PATH_MAX * 3, "tar xvzf \"%s\" -C \"%s\"", tarpath, dest_dir
    );
    PSNotice("Extracting CIFAR dataset (%d classes)", classes);
    int status = system(cmd);
    success = (status == 0);
    if (!success) goto final;
final:
    if (tarpath != NULL && PSFileExists(tarpath)) unlink(tarpath);
    if (!provided_dest) free(dest_dir);
    free(tarpath);
    if (!success) {
        free(path);
        path = NULL;
    }
    return path;
}

static int loadCIFARData(int data_type, int classes, int argc, char **argv,
                         int *arg_idx)
{
    assert(
        data_type == PS_DATA_TYPE_TRAINING || data_type == PS_DATA_TYPE_TEST
    );
    int i = *arg_idx;
    int *len = NULL, *dataset_len = NULL;
    PSFloat **data = NULL;
    if (data_type == PS_DATA_TYPE_TRAINING) {
        len = &datalen;
        data = &training_data;
        dataset_len = &train_dataset_len;
    } else {
        len = &testlen;
        data = &test_data;
    }
    char datapath[PATH_MAX];
    datapath[0] = '\0';
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
        char *download_path = downloadCIFARDataset(classes, NULL);
        if (download_path == NULL) {
            PSErr(NULL, "could not download CIFAR dataset");
            return 0;
        }
        if (strlen(download_path) >= PATH_MAX) {
            PSErr(NULL, "data path length exceeds max length");
            free(download_path);
            return 0;
        }
        strncpy(datapath, download_path, PATH_MAX);
        free(download_path);
    }
    *len = PSLoadCIFARData(
        data_type, classes, datapath, data, max_files, max_images
    );
    if (*len == 0 || *data == NULL) {
        fprintf(stderr, "Failed to load CIFAR data\n");
        return 0;
    }
    if (dataset_len != NULL)
        *dataset_len = (*len / (PS_CIFAR_IMAGE_SIZE + classes));
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
        if (argv[i][0] == '-') {
            fprintf(stderr, "ERROR: invalid argument '%s'", argv[i]);
            return 0;
        }
        int *len = NULL;
        PSFloat **data = NULL;
        char *descr = NULL;
        assert(
            data_type == PS_DATA_TYPE_TRAINING ||
            data_type == PS_DATA_TYPE_TEST
        );
        if (data_type == PS_DATA_TYPE_TRAINING) {
            len = &datalen;
            data = &training_data;
            descr = "training";
        } else {
            len = &testlen;
            data = &test_data;
            descr = "test";
        }
        uint64_t dlen = 0;
        printf("Loading %s dataset from file: '%s'\n", descr, argv[i]);
        *data = PSLoadDataFromFile(argv[i], &dlen);
        *len = (int) dlen;
        if (*data == NULL || datalen == 0) {
            PSErr(NULL, "could not load dataset at '%s'", argv[i]);
            free(*data);
            *len = 0;
            *data = NULL;
            return 0;
        }
        if (datalen > INT_MAX) {
            PSErr(NULL, "dataset length is too big");
            *len = 0;
            free(*data);
            return 0;
        }
        printf("Loaded %s dataset of length: %d\n", descr, *len);
        return 1;
    } else {
        if (data_type == PS_DATA_TYPE_TRAINING) {
            if (mnist) {
                train_dataset_len = 50000;
                eval_dataset_len = 10000;
            } else {
                train_dataset_len = 40000;
                eval_dataset_len = 10000;
            }
        }
        if (mnist) return loadMNISTData(data_type, argc, argv, arg_idx);
        else return loadCIFARData(data_type, cifar, argc, argv, arg_idx);
    }
}

static void cleanup(void) {
    if (on_epoch_trained != NULL) free(on_epoch_trained);
    if (on_batch_trained != NULL) free(on_batch_trained);
#ifdef HAS_MAGICK
    if (image_filename != NULL) free(image_filename);
#endif
    if (model != NULL) {
        PSModelFree(model);
        model = NULL;
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

static void printAccelerationInfo(PSAcceleration acceleration) {
    char *label = NULL, *notes = NULL, *prop = NULL;
    if (!PSIsAccelerationAvailable(acceleration)) return;
    if (acceleration == PSAcceleration_AVX) {
        label = "AVX";
        prop = "--avx";
    } else if (acceleration == PSAcceleration_ACF) {
        label = "Accelerate Framework";
        prop = "--accelerate-framework";
    } else if (acceleration == PSAcceleration_BLAS) {
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        if (PSIsAccelerationAvailable(PSAcceleration_ACF))
            notes = "(Accelerate Framework)";
#elif defined(HAS_GSL_CBLAS)
        notes = "(GNU Scientific Library)";
#endif
        if (notes == NULL) notes = "(native)";
        label = "BLAS";
        prop = "--blas";
    } else return;
    if (notes == NULL) notes = "";
    printf("%-25s %-25s %s\n", label, prop, notes);
}

static void printAvailableAccelerations(void) {
    printf("%-25s %-25s %s\n", "NAME", "OPTION", "NOTES");
    printf("-----------------------------------------------------------------"
           "-----\n");
    printAccelerationInfo(PSAcceleration_AVX);
    printAccelerationInfo(PSAcceleration_ACF);
    printAccelerationInfo(PSAcceleration_BLAS);
}

static void printInfo(void) {
    printf("Version:                %s\n", PSYC_VERSION);
    printf("Git SHA:                %s\n", PSYC_GIT_SHA);
    printf("Git Dirty:              %s\n", PSYC_GIT_DIRTY);
    printf("Git Branch:             %s\n", PSYC_GIT_BRANCH);
    printf("Arch.:                  %dbit\n", (sizeof(long) == 8 ? 64 : 32));
    printf("Double Precision:       %s\n",
            (sizeof(PSFloat) > sizeof(float) ? "yes" : "no"));
    printf("Code Optimization:      %d\n", PSGetCodeOptimizationLevel());
    printf("Available Acceleration(s):\n");
    if (PSIsAccelerationAvailable(PSAcceleration_ACF))
        printf("    Accelerate Framework\n");
    if (PSIsAccelerationAvailable(PSAcceleration_BLAS))
        printf("    BLAS\n");
    if (PSIsAccelerationAvailable(PSAcceleration_AVX))
        printf("    AVX\n");
}

static int parseParamInitMode(int param_type, char *arg, PSLayerDef *ldef,
                              char *mode)
{
    int *modeptr = NULL;
    if (param_type == PS_PARAM_BIAS) modeptr = &(ldef->bias_init_mode);
    else if (param_type == PS_PARAM_BIAS) modeptr = &(ldef->weight_init_mode);
    else return 0;
    if (strcmp("auto", mode) == 0) *modeptr = PS_INIT_MODE_AUTO;
    else if (strcmp("random", mode) == 0) *modeptr = PS_INIT_MODE_RAND;
    else if (strcmp("zero", mode) == 0) *modeptr = PS_INIT_MODE_ZERO;
    else if (strcmp("0", mode) == 0) *modeptr = PS_INIT_MODE_ZERO;
    else {
        fprintf(stderr, "ERROR: Invalid %s: '%s'.", arg, mode);
        fprintf(stderr, " Valid modes: auto, random, zero\n");
        return 0;
    }
    return 1;
}

static int parseLayerCoordinates(char *coords, int *nidx, int *lidx) {
    *nidx = -1;
    *lidx = -1;
    if (coords == NULL) return 0;
    char *sep = strchr(coords, ':');
    if (sep != NULL) {
        char *n = coords, *l = sep + 1;
        *sep = '\0';
        *nidx = atoi(n);
        *lidx = atoi(l);
        if (*nidx < 0) return 0;
    } else *lidx = atoi(coords);
    if (*lidx < 0) return 0;
    return 1;
}

static PSLayer *getLayerFromCoordinates(char *coords, int model_idx,
                                        PSModel *current,
                                        int allow_future)
{
    int nidx = -1, lidx = -1;
    PSLayer *layer = NULL;
    if (!parseLayerCoordinates(coords, &nidx, &lidx)) {
        fprintf(stderr, "ERROR: Invalid layer coordinates %s\n",
                coords);
        return NULL;
    }
    if (nidx < 0) nidx = model_idx;
    else if (nidx > model_idx) {
        fprintf(stderr, "ERROR: Invalid layer coordinates %s"
                ": model index %d > current: %d\n",
                coords, nidx, model_idx);
        return NULL;
    }
    if (nidx == model_idx) layer = current->layers[lidx];
    else layer = PSGetLayerByIndex(model, lidx, nidx);
    if (layer == NULL && allow_future)
        layer = PSMakeLayerPlaceholder(lidx, nidx);
    if (layer == NULL) {
        fprintf(stderr, "ERROR: Invalid layer coordinates %s"
                ": layer not found\n", coords);
        return NULL;
    }
    return layer;
}

static int openHTMLDoc(char *executable) {
    static char *gui_programs[] = {
        "xdg-open", "sensible-browser", "x-www-browser", "gnome-open",
    };
    static char *txt_programs[] = {
        "lynx", "sensible-browser", "w3m"
    };
    char *index_path = PSPathJoin(
        2, PS_PREFIX, "share/psyc/doc/html/index.html"
    );
    if (index_path == NULL) return 0;
    int success = 1;
    if (!PSFileExists(index_path)) {
        PSWarn("PsyC or PsyC's documentation is not installed");
        free(index_path);
        index_path = NULL;
        char *exec_path = getPsycPath(executable);
        if (exec_path == NULL) return 0;
        index_path = PSPathJoin(2, exec_path, "doc/html/index.html");
        if (index_path == NULL) return 0;
        success = PSFileExists(index_path);
        if (!success) {
            PSErr(NULL, "could not find any documentation");
            goto final;
        }
    }
    struct utsname sysinfo;
    uname(&sysinfo);
    size_t n_programs, i;
    char cmd[PATH_MAX];
    char *program = NULL;
    char **programs = NULL;
    if (strcmp("Darwin", sysinfo.sysname) == 0) program = "open";
    else {
        int has_gui = (system("test -n \"$DISPLAY\"") == 0);
        if (has_gui) {
            programs = gui_programs;
            n_programs =  sizeof(gui_programs) / sizeof(char *);
        } else {
            programs = txt_programs;
            n_programs =  sizeof(txt_programs) / sizeof(char *);
        }
        for (i = 0; i < n_programs; i++) {
            sprintf(cmd, "which %s", *(programs + i));
            if (system(cmd) == 0) {
                program = *(programs + i);
                break;
            }
        }
        if (program == NULL) {
            PSErr(NULL, "failed to find a program to open document web page:\n"
                  "%s\nTry to install one of the following programs:\n",
                  index_path);
            for (i = 0; i < n_programs; i++)
                PSPrint(PSLOGLEVEL_WARN, "%s\n", *(programs + i));
            success = 0;
            goto final;
        }
    }
    if (program == NULL) {
    }
    snprintf(cmd, PATH_MAX, "%s '%s'", program, index_path);
    int exit_status = system(cmd);
    success = exit_status == 0;
final:
    free(index_path);
    return success;
}

void parseOptions(int argc, char **argv) {
    int model_idx = 0, i, j;
    PSModel *current = model, *previous = NULL;
    PSModelLink *link = NULL;
    PSModelLink curlink = {0};
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
            int loaded = PSModelLoad(current, file);
            if (!loaded) {
                fprintf(stderr, "ERROR: Could not load pretrained model "
                        "%s\n", file);
                goto err;
            }
        } else if (strcmp("--network", arg) == 0 ||
                   strcmp("--model", arg) == 0) {
            if (previous != NULL) {
                if (!PSAddModel(previous, current, link)) {
                    fprintf(stderr, "ERROR: Could not add model\n");
                    goto err;
                }
            }
            previous = current;
            current = PSModelCreate(NULL);
            if (current == NULL) {
                fprintf(stderr, "ERROR: Could not create model\n");
                goto err;
            }
            model_idx++;
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
            PSModelSetName(current, name);
        } else if (strcmp("--onehot", arg) == 0) {
            if (current->size == 0) current->flags |= PS_FLAG_ONEHOT;
            else current->layers[current->size - 1]->flags |= PS_FLAG_ONEHOT;
        } else if ((strcmp("--layer", arg) == 0 ||
                    strcmp("-l", arg) == 0) && !is_last)
        {
            char *type = argv[++i];
            int is_cifar = 0, lidx = current->size;
            PSLayer *link_to = NULL, *query_provider_to = NULL;
            PSLayer *providers[MAX_PROVIDERS] = {0};
            int providers_count = 0;
            PSLayerDef ldef = {0};
            PSLayerType ltype = getLayerType(type, &is_cifar, &ldef);
            if ((i + 1) >= argc) break;
            PSLayer *layer = NULL;
            if (is_cifar) {
                layer = PSAddCIFARInputLayer(current);
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
                            stderr, "ERROR: Invalid %s %s\n", carg, fcstr
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
                            stderr, "ERROR: Invalid %s %s\n", carg, szstr
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
                            stderr, "ERROR: Invalid %s %s\n", carg, szstr
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
                    else if (strcasecmp("gelu", actvname) == 0)
                        ldef.activation = PSGelu;
                    else {
                        fprintf(stderr, "ERROR: Invalid activation '%s'",
                                actvname);
                        goto err;
                    }
                    i = j;
                } else if ((strcmp("--region-size", carg) == 0 ||
                            strcmp("--filter-width", carg) == 0) &&
                            ++j<argc)
                {
                    int filter_w = 0;
                    char *rsstr = argv[j];
                    int matched = sscanf(rsstr, "%d", &filter_w);
                    if (!matched) {
                        fprintf(stderr, "ERROR: Invalid %s %s\n", carg, rsstr);
                        goto err;
                    }
                    i = j;
                    ldef.filter_width = filter_w;
                } else if (strcmp("--filter-height", carg)==0 && ++j<argc) {
                    int filter_h = 0;
                    char *rsstr = argv[j];
                    int matched = sscanf(rsstr, "%d", &filter_h);
                    if (!matched) {
                        fprintf(stderr, "ERROR: Invalid %s %s\n", carg, rsstr);
                        goto err;
                    }
                    i = j;
                    ldef.filter_height = filter_h;
                } else if (strcmp("--stride", carg) == 0 && ++j < argc) {
                    int stride = 0;
                    char *ststr = argv[j];
                    int matched = sscanf(ststr, "%d", &stride);
                    if (!matched) {
                        fprintf(stderr, "ERROR: Invalid stride %s\n", ststr);
                        goto err;
                    }
                    i = j;
                    ldef.stride = stride;
                } else if (strcmp("--padding", carg) == 0 && ++j < argc) {
                    int padding = 0;
                    char *padstr = argv[j];
                    int matched = sscanf(padstr, "%d", &padding);
                    if (!matched) {
                        fprintf(stderr, "ERROR: Invalid padding %s\n", padstr);
                        goto err;
                    }
                    i = j;
                    ldef.padding = padding;
                } else if (strcmp("--use-relu", carg) == 0) {
                    i = j;
                    ldef.activation = PSRelu;
                } else if (strcmp("--pretrained", carg) == 0) {
                    i = j;
                    ldef.pretrained = 1;
                } else if (strcmp("--load-layer", carg) == 0 && ++j < argc) {
                    i = j;
                    ldef.load_from = argv[j];
                } else if (strcmp("--dropout", carg) == 0 && ++j < argc) {
                    char *dropout = argv[j];
                    int matched = sscanf(
                        dropout, PSFLOAT_FORMAT, &(ldef.dropout)
                    );
                    if (!matched) {
                        fprintf(
                            stderr, "ERROR: Invalid dropout %s\n", dropout
                        );
                        goto err;
                    }
                    i = j;
                } else if (strcmp("--operator", carg) == 0 && ++j < argc) {
                    char *opstr = argv[j];
                    PSOperatorType op;
                    if (strcmp("add", opstr) == 0) op = PSAddOperator;
                    else if (strcmp("concatenate", opstr) == 0)
                        op = PSConcatenateOperator;
                    else if (strcmp("mul", opstr) == 0)
                        op = PSMultiplyOperator;
                    else if (strcmp("multiply", opstr) == 0)
                        op = PSMultiplyOperator;
                    else {
                        fprintf(stderr, "ERROR: Invalid %s: valid values are "
                                "add|concatenate\n", opstr);
                        goto err;
                    }
                    i = j;
                    ldef.operator = op;
                } else if (strcmp("--provider", carg) == 0 && ++j < argc) {
                    if (ltype != OperatorLayer) {
                        fprintf(stderr, "ERROR: %s is only available to "
                               "operator layers\n", carg);
                        goto err;
                    }
                    if (providers_count >= MAX_PROVIDERS) {
                        fprintf(stderr, "ERROR: max providers is %d\n",
                                MAX_PROVIDERS);
                        goto err;
                    }
                    PSLayer *provider = getLayerFromCoordinates(
                        argv[j], model_idx, current, 0
                    );
                    if (provider == NULL) goto err;
                    i = j;
                    providers[providers_count++] = provider;
                    ldef.providers_count = providers_count;
                } else if (strcmp("--attention-type", carg) == 0 && ++j<argc) {
                    char *typestr = argv[j];
                    PSAttentionType type;
                    if (strcasecmp("dot", typestr) == 0) type = PSDotAttention;
                    else if (strcasecmp("add", typestr) == 0)
                        type = PSAdditiveAttention;
                    else if (strcasecmp("additive", typestr) == 0)
                        type = PSAdditiveAttention;
                    else {
                        fprintf(stderr, "ERROR: Invalid --attention-type "
                                "'%s'\n", typestr);
                        fprintf(stderr, "Valid types: dot|additive\n");
                        goto err;
                    }
                    i = j;
                    ldef.attention_type = type;
                } else if (strcmp("--causal", carg) == 0) {
                    i = j;
                    ldef.causal_attention = 1;
                } else if (strcmp("--attention-heads", carg) == 0 && ++j<argc) {
                    int heads = atoi(argv[j]);
                    if (heads < 0) heads = 0;
                    i = j;
                    ldef.attention_heads = heads;
                } else if (strcmp("--attention-scale", carg) == 0 && ++j<argc) {
                    PSFloat scale = atof(argv[j]);
                    if (scale < 0) scale = 0;
                    i = j;
                    ldef.attention_scale = scale;
                } else if (strcmp("--query-provider", carg) == 0 && ++j<argc) {
                    PSLayer *provider = getLayerFromCoordinates(
                        argv[j], model_idx, current, 1
                    );
                    if (provider == NULL) goto err;
                    if (ltype != Attention) {
                        if (provider->type != Attention) {
                            fprintf(stderr, "ERROR: current layer type is "
                                    "not attention and provider type is not "
                                    "attention\n");
                            goto err;
                        }
                        query_provider_to = provider;
                    } else ldef.query_provider = provider;
                    i = j;
                } else if (strcmp("--key-provider", carg) == 0 && ++j<argc) {
                    if (ltype != Attention) {
                        fprintf(stderr, "ERROR: %s is only available to "
                                "attention layers\n", carg);
                        goto err;
                    }
                    PSLayer *provider = getLayerFromCoordinates(
                        argv[j], model_idx, current, 0
                    );
                    if (provider == NULL) goto err;
                    i = j;
                    ldef.keys_provider = provider;
                } else if (strcmp("--value-provider", carg) == 0 && ++j<argc) {
                    if (ltype != Attention) {
                        fprintf(stderr, "ERROR: %s is only available to "
                                "attention layers\n", carg);
                        goto err;
                    }
                    PSLayer *provider = getLayerFromCoordinates(
                        argv[j], model_idx, current, 0
                    );
                    if (provider == NULL) goto err;
                    i = j;
                    ldef.values_provider = provider;
                } else if (strcmp("--link", carg) == 0 && ++j < argc) {
                    link_to = getLayerFromCoordinates(
                        argv[j], model_idx, current, 0
                    );
                    i = j;
                    if (link_to == NULL) goto err;
                } else if (strcmp("--whole-sequence", carg) == 0) {
                    ldef.flags &= ~((unsigned) PS_FLAG_RECURRENT);
                    ldef.flags |= PS_FLAG_USE_SEQUENCES;
                } else if (strcmp("--recurrent-layer", carg) == 0) {
                    ldef.flags |= PS_FLAG_RECURRENT;
                } else if (strcmp("--disable-biases", carg) == 0) {
                    ldef.flags |= PS_FLAG_NO_BIAS;
                } else if (strcmp("--weight-init-mode", carg)==0 && ++j<argc) {
                    char *modestr = argv[j];
                    int ok = parseParamInitMode(
                        PS_PARAM_WEIGHT, carg, &ldef, modestr
                    );
                    if (!ok) goto err;
                    i = j;
                } else if (strcmp("--bias-init-mode", carg)==0 && ++j<argc) {
                    char *modestr = argv[j];
                    int ok = parseParamInitMode(
                        PS_PARAM_BIAS, carg, &ldef, modestr
                    );
                    if (!ok) goto err;
                    i = j;
                } else if (strcmp("--init-range", carg)==0 && ++j<argc) {
                    char *rangestr = argv[j];
                    PSFloat range = 0.0;
                    int matched = sscanf(rangestr, PSFLOAT_FORMAT, &range);
                    if (!matched || range < 0) {
                        fprintf(stderr, "ERROR: Invalid %s\n", carg);
                        goto err;
                    }
                    ldef.init_range = range;
                    i = j;
                } else if (strcmp("--init-scale", carg)==0 && ++j<argc) {
                    char *scalestr = argv[j];
                    PSFloat scale = 0.0;
                    int matched = sscanf(scalestr, PSFLOAT_FORMAT, &scale);
                    if (!matched || scale < 0) {
                        fprintf(stderr, "ERROR: Invalid %s\n", carg);
                        goto err;
                    }
                    ldef.init_scale = scale;
                    i = j;
                } else break;
            }
            int size = 0;
            int need_size = (
                ltype != Convolutional && ltype != Pooling &&
                ltype != Dropout && ltype != Normalization &&
                ltype != Attention && ltype != OperatorLayer
            );
            if (need_size) {
                if (i >= argc) {
                    fprintf(
                        stderr, "ERROR: missing layer size for layer %d\n",
                        lidx
                    );
                }
                char *sizestr = argv[++i];
                int matched = sscanf(sizestr, "%d", &size);
                if (!matched) {
                    fprintf(stderr, "ERROR: Invalid size %s\n", sizestr);
                    goto err;
                }
                j = i + 1;
            }
            if (OperatorLayer == ltype && providers_count > 0) {
                ldef.providers_count = providers_count;
                ldef.providers = providers;
            }
            layer = PSAddLayer(current, ltype, size, &ldef);
            if (layer == NULL) {
                fprintf(
                    stderr, "FATAL: Failed to create layer %d\n", lidx
                );
                goto err;
            }
            if (link_to != NULL) {
                curlink.layer = layer;
                curlink.previous_layer = link_to;
                link = &curlink;
            }
            if (query_provider_to != NULL) {
                if (!PSSetAttentionQueryProvider(query_provider_to, layer)) {
                    fprintf(
                        stderr, "ERROR: could not set layer %d:%d (%s) as "
                        "query provider for layer %d:%d\n",
                        model_idx, layer->index, PSGetLabelForType(ltype),
                        query_provider_to->model->index,
                        query_provider_to->index
                    );
                }
            }
            continue;
        } else if (strcmp("--train", arg) == 0 && ++i < argc) {
            if (!loadData(PS_DATA_TYPE_TRAINING, argc, argv, &i)) goto err;
        } else if (strcmp("--test", arg) == 0 && ++i < argc) {
            if (!loadData(PS_DATA_TYPE_TEST, argc, argv, &i)) goto err;
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
            training_flags |= PS_TRAINING_WEIGHT_DECAY;
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
            else if (strcmp("rmsprop", optname) == 0)
                optimization = PSRMSPropOptimization;
            else if (strcmp("windowgrad", optname) == 0)
                optimization = PSWindowGradOptimization;
            else if (strcmp("nesterov", optname) == 0)
                optimization = PSNesterovOptimization;
            else if (strcmp("default", optname) == 0)
                optimization = PSDefaultOptimization;
            else if (strcmp("none", optname) == 0)
                optimization = PSDefaultOptimization;
            else {
                fprintf(stderr, "Invalid optmization `%s`\n", optname);
                fprintf(
                    stderr, "Valid values: adam, adagrad, adadelta, "
                    "windowgrad, nesterov, default\n"
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
            current->loss = func;
        } else if (strcmp("--validate-every", arg) == 0 && !is_last) {
            char *everystr = argv[++i];
            int matched = sscanf(everystr, "%d", &validate_every);
            if (!matched) {
                fprintf(stderr, "Invalid --validate-every %s\n", everystr);
                goto err;
            }
        } else if (strcmp("--training-no-shuffle", arg) == 0) {
            training_flags |= PS_TRAINING_NO_SHUFFLE;
        } else if (strcmp("--training-adjust-rate", arg) == 0) {
            training_flags |= PS_TRAINING_ADJUST_RATE;
        } else if (strcmp("--disable-avx", arg) == 0) {
            PSDisableAcceleration(&current->acceleration, PSAcceleration_AVX);
        } else if (strcmp("--disable-accelerate", arg) == 0 ||
                   strcmp("--disable-acf", arg) == 0)
        {
            PSDisableAcceleration(&current->acceleration, PSAcceleration_ACF);
        } else if (strcmp("--disable-blas", arg) == 0) {
            PSDisableAcceleration(&current->acceleration, PSAcceleration_BLAS);
        } else if (strcmp("--enable-colors", arg) == 0) {
            PSGlobalFlags |= PS_FLAG_LOG_COLORS;
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
                current->onBatchTrained = onBatchTrained;
        } else if (strcmp("--on-epoch-trained", arg) == 0 && !is_last) {
            on_epoch_trained =strdup( argv[++i]);
            if (strlen(on_epoch_trained) > 0)
                current->onEpochTrained = onEpochTrained;
        } else if (strcmp("--download-mnist", arg) == 0) {
            char *mnist_download_path = NULL;
            if (!is_last && argv[i + 1][0] != '-')
                mnist_download_path = argv[++i];
            char *path = downloadMNISTDataset(mnist_download_path);
            int downloaded = (path != NULL);
            if (!downloaded) PSErr(NULL, "failed to download MNIST dataset");
            else {
                PSPrint(
                    PSLOGLEVEL_SUCCESS, "MNIST dataset downloaded at: '%s'\n",
                    path
                );
            }
            free(path);
            exit(!downloaded);
        } else if (strcmp("--download-cifar", arg) == 0) {
            char *cifar_download_path = NULL;
            int classes = 10, cifar_argc = 0;
            for (j = i + 1; j < argc; j++) {
                if (cifar_argc >= 2) break;
                char *next = argv[j];
                if (next[0] == '-') break;
                if (isdigit(next[0])) {
                    classes = atoi(next);
                    if (classes != 10 && classes != 100) {
                        fprintf(stderr, "ERROR: invalid CIFAR classes %d: "
                                "only 10 or 100 allowed\n", classes);
                        exit(1);
                    }
                    cifar_argc++;
                } else {
                    cifar_download_path = next;
                    cifar_argc++;
                }
            }
            char *path = downloadCIFARDataset(classes, cifar_download_path);
            int downloaded = (path != NULL);
            if (!downloaded) PSErr(NULL, "failed to download CIFAR dataset");
            else {
                PSPrint(
                    PSLOGLEVEL_SUCCESS, "CIFAR dataset downloaded at: '%s'\n",
                    path
                );
            }
            free(path);
            exit(!downloaded);
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
        } else if (strcmp("--available-accelerations", arg) == 0) {
            printAvailableAccelerations();
            exit(0);
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
        } else if (strcmp("--info", arg) == 0) {
            printInfo();
            cleanup();
            exit(0);
        } else if (strcmp("--doc", arg) == 0) {
            int ok = openHTMLDoc(argv[0]);
            exit(ok ? 0 : 1);
        } else if (strcmp("-h", arg) == 0 || strcmp("--help", arg) == 0) {
            printHelp(argv[0]);
            cleanup();
            exit(1);
        } else {
            fprintf(stderr, "ERROR: invalid argument %s\n", arg);
            goto err;
        }
    }
    if (current != model && !PSModelChainContains(model, current)) {
        if (!PSAddModel(model, current, link)) {
            fprintf(stderr, "ERROR: Could not add model\n");
            goto err;
        }
    }
    return;
err:
    if (current != model && !PSModelChainContains(model, current))
        PSModelFree(current);
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
        if (numtokens == 0 || tokens[0] == NULL) goto next_line;
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

void onTrainEvent(int event_type, PSModel *model, int epoch,
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
        on_batch_trained, event, model->name, epoch, epochs, avg_loss,
        current_loss, accuracy, *rate
    );
    if (written >= CMD_MAX_LEN) {
        fprintf(stderr, "\nWARN: onBatchTrained command is too big!\n");
        return;
    }
    PSTrainingInfo *info = model->training;
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

void onBatchTrained(PSModel *model, int epoch, int epochs,
                    PSFloat avg_loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{

    UNUSED(training_data);
    if (on_batch_trained == NULL) return;
    PSTrainingInfo *info = model->training;
    if (batch_script_every > 0 && info) {
        if ((info->current_batch % batch_script_every) != 0) return;
    }
    onTrainEvent(
        TRAIN_EVENT_BATCH, model, epoch, epochs, avg_loss, current_loss,
        accuracy, rate
    );
}

void onEpochTrained(PSModel *model, int epoch, int epochs,
                    PSFloat avg_loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{
    UNUSED(training_data);
    if (on_epoch_trained == NULL) return;
    onTrainEvent(
        TRAIN_EVENT_EPOCH, model, epoch, epochs, avg_loss, current_loss,
        accuracy, rate
    );
}

int main(int argc, char **argv) {
    PSHandleSignals(NULL);
    model = PSModelCreate(NULL);
    if (model == NULL) {
        fprintf(stderr, "Failed to create model");
        return 1;
    }
    model->name = MODEL_NAME;
    outputFile[0] = 0;
    parseOptions(argc, argv);
    if (PSLogLevel <= PSLOGLEVEL_INFO) PSModelPrintInfo(model);
    if (training_data != NULL) {
        if (datalen == 0) {
            PSErr(NULL, "empty dataset");
            cleanup();
            return 1;
        }
        int element_size = model->input_size + model->output_size;
        int element_count = datalen / element_size;
        if (train_dataset_len == 0) train_dataset_len = element_count;
        if (element_count < train_dataset_len) {
            PSErr(NULL, "loaded dataset elements %d < %d\n", element_count,
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
            datalen = train_dataset_len * element_size;
            if (eval_dataset_len == 0) validation_data = NULL;
            else {
                validation_data = training_data + datalen;
                valdlen = eval_dataset_len * element_size;
            }
        }

        PSTrainingOptions options = {
            .epochs = epochs,
            .batch_size = batch_size,
            .learning_rate = learning_rate,
            .flags = training_flags,
            .l1_decay = (PSFloat) l1_decay,
            .l2_decay = (PSFloat) l2_decay,
            .momentum = (PSFloat) momentum,
            .optimization = optimization,
            .validate_every_batches = validate_every
        };
        PSTrain(model, training_data, datalen, validation_data, valdlen,
                &options);
        free(training_data);
    }
    if (test_data != NULL) {
        PSTest(model, test_data, testlen, NULL);
        free(test_data);
    }

#ifdef HAS_MAGICK
    if (image_filename != NULL) {
        int res = PSClassifyImage(model, image_filename, image_grayscale,
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
            getTempFileName("saved-model", outputFile);
        }
        int saved = PSModelSave(model, outputFile);
        if (!saved) {
            fprintf(stderr, "Could not save model to %s\n", outputFile);
        } else {
            printf("model saved to %s\n", outputFile);
        }
    }
    cleanup();
    return 0;
}

void printLossFunctionName(const char *name, PSLossFunction func) {
    UNUSED(func);
    printf("  %s\n", name);
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
    printf("\nUsage: %s [OPTIONS]\n\n", program_path);
    printf("DESCRIPTION:\n\n");
    printf("  psycl is a command-line interface to %s, an open-source C "
           "library that\n"
           "  allows building neural networks ( %s ).\n",
           PSYC_NAME, PSYC_SITE);
    printf("  The utility can be used to build, load, save and train models "
           "without\n"
           "  the need to write a C program, albeit with limitations.\n");
    printf("\n");
    printf("OPTIONS:\n\n");
    printf("  --available-accelerations    List available "
           "accelerations.\n");
    printf("  --batch-script-every NUM     Interval of NUM batches after "
           "which to run the\n"
           "                               script defined by the "
           "`--on-batch-trained`\n"
           "                               option (if set).\n");
    printf("  --batch-size SIZE            Training batch size (default: %d)\n",
           BATCH_SIZE);
#ifdef HAS_MAGICK
    printf("  --classify-image FILE [OPTIONS...]\n"
           "                               Classify the image located at path "
           "FILE with the\n"
           "                               current model (see 'IMAGE OPTIONS'"
           " section).\n");
#endif
    printf("  -c, --config FILE            Load options from FILE (see the '"
           "CONFIG FILES'\n"
           "                               section).\n");
    printf("  --disable-accelerate         Disable Accelerate "
           "Framework.\n");
    printf("  --disable-avx                Disable AVX.\n");
    printf("  --disable-blas               Disable BLAS.\n");
    printf("  --download-cifar [CLASSES] [DEST_DIR]\n"
           "                               Download CIFAR dataset and "
           "exit. If no DEST_DIR\n"
           "                               is provided, the dataset will be "
           "saved into\n"
           "                               " PSYC_NAME " working directory.\n"
           "                               Optional CLASSES can be 10 or "
           "100.\n"
           "                               (default: 10).\n");
    printf("  --download-mnist [DEST_DIR]  Download MNIST dataset and "
           "exit. If no DEST_DIR\n"
           "                               is provided, the dataset will be "
           "saved into\n"
           "                               " PSYC_NAME " working directory.\n");
    printf("  --enable-colors              Colorized output.\n");
    printf("  --epochs EPOCHS              Training epochs (default: %d).\n",
           EPOCHS);
    printf("  --info                       Print " PSYC_NAME " info.\n");
    printf("  --l1-decay SIZE              L1 Decay (default: 0).\n");
    printf("  --l2-decay SIZE              L2 Decay (default: 0).\n");
    printf("  -l, --layer TYPE (SIZE | OPTIONS...)\n"
           "                               Add layer (see 'LAYER TYPES' and "
           "'LAYER OPTIONS'\n"
           "                               sections).\n");
    printf("  --learning-rate SIZE         Training learning rate (default: "
           "%.2f).\n", LEARNING_RATE);
    printf("  --load PRETRAINED            Load a pretrained model.\n");
    printf("  --loglevel LEVEL             Set log level "
           "(see 'LOG LEVELS' section).\n");
    printf("  --loss-function FUNC         Loss Function (see 'LOSS "
           "FUNCTIONS' section).\n");
    printf("  --model                      Start new model (it can be used\n"
           "                               multiple times to create chained "
           "models\n"
           "                               composed of multiple neural "
           "networks).\n");
    printf("  --momentum MOMENTUM          Momentum (default: 0).\n");
    printf("  --name NAME                  Model name.\n");
    printf("  --on-batch-trained SCRIPT    Execute script after every "
           "batch is trained.\n"
           "                               (See \"SCRIPTS\" section for "
           "more info).\n");
    printf("  --on-epoch-trained SCRIPT    Execute script after every "
           "epoch is trained.\n"
           "                               (See \"SCRIPTS\" section for "
           "more info).\n");
    printf("  --onehot                     Sets one-hot-vector flag for "
           "input layer (if\n"
           "                               before 1st layer) or target "
           "dataset (if after\n"
           "                               output layer).\n");
    printf("  --optimization NAME          Training optimization, NAME can "
           "be:\n"
           "                               (adagrad | adadelta | adam | "
           "rmsprop |\n"
           "                               windowgrad | nesterov | default)"
           ".\n");
    printf("  --quiet                      Quiet output (loglevel ERROR)."
           "\n");
    printf("  --save FILE                  Save model to FILE.\n");
    printf("  --test [OPTIONS] TEST_DATASET\n"
           "                               Test model against TEST_DATASET.\n"
           "                               (see 'TRAIN|TEST OPTIONS' "
           "section).\n");
    printf("  --train [OPTIONS] TRAIN_DATASET\n"
           "                               Train model with TRAIN_DATASET.\n"
           "                               (see 'TRAIN|TEST OPTIONS' section"
           ").\n");
    printf("  --training-adjust-rate       Auto-adjust learn rate.\n");
    printf("  --training-datalen LEN       Training data length.\n");
    printf("  --training-no-shuffle        Prevent dataset shuffle.\n");
    printf("  --validate-every BATCH_NUM   Validate inside epochs.\n");
    printf("  --validation-datalen LEN     Validation data length.\n");
    printf("  --verbose                    Verbose output (loglevel DEBUG)."
           "\n");
    printf("  -v, --version                Print version.\n");
    printf("  --weight-decay               Enable L1/L2 weight decay\n"
           "                               instead of L1/L2 "
           "regularization.\n");
    printf("    -h, --help                 Print this help.\n");
    printf("\n");
    printf("LAYER TYPES:\n\n");
    int i;
    for (i = 0; i < PS_LAYER_TYPES; i++) {
        PSLayerType type = (PSLayerType) i;
        /*printf("  %s\n", PSGetLabelForType(type));*/
        printLayerTypeHelp(type);
    }
    printf("\n");
    printf("LOSS FUNCTIONS:\n\n");
    PSIterateLossFunctions(printLossFunctionName);
    printf("\n");
    printf("LAYER OPTIONS:\n\n");
    printf("  --activation FUNC            Activation Function: "
           "(sigmoid,tanh,relu,gelu).\n");
    printf("  --attention-heads NUM        Multi-Head Attention layer "
        "heads.\n");
    printf("  --attention-scale SCALE      Attention layer scale.\n");
    printf("  --attention-type TYPE        Attention layer type: "
           "(dot|additive).\n");
    printf("  --bias-init-mode MODE        Bias initialization mode:\n"
           "                               (auto|random|zero) "
           "(default: auto).\n");
    printf("  --causal                     Causal Attention.\n");
    printf("  --disable-biases             Disable biases.\n");
    printf("  --dropout DROPOUT            Layer Dropout (float).\n");
    printf("  --filter-width WIDTH         Convolutional filter width"
           " (default: %d).\n", CONV_REGION_SIZE);
    printf("  --filter-height HEIGHT       Convolutional filter height"
           " (default: %d).\n", CONV_REGION_SIZE);
    printf("  --init-range RANGE           Weight|Bias initialization "
           "range.\n"
           "                               (for 'random' init mode)\n");
    printf("  --init-scale SCALE           Weight|Bias initialization "
           "scale.\n"
           "                               (for 'random' init mode)\n");
    printf("  --key-provider COORDS        Attention keys provider.\n"
           "                               (See 'LAYER COORDINATES' section "
           "for details\n"
           "                               about COORDS).\n"
    );
    printf("  --link COORDS                Link layer to previous model.\n"
           "                               (See 'LAYER COORDINATES' section "
           "for details\n"
           "                               about COORDS).\n"
    );
    printf("  --load-layer PATH            Load layer parameters from "
           "PATH.\n");
    printf("  --operator OP                Operator layer operator: "
           "(add|mul|concatenate).\n");
    printf("  --output-width WIDTH         Output Width.\n");
    printf("  --output-height HEIGHT       Output Height.\n");
    printf("  --output-depth DEPTH         Output Depth.\n");
    printf("  --padding PADDING            Convolutional padding"
           " (default: 0).\n");
    printf("  --pretrained                 Pretrained layer.\n");
    printf("  --provider COORDS            Operator layer provider.\n"
           "                               (See 'LAYER COORDINATES' section "
           "for details\n"
           "                               about COORDS).\n"
    );
    printf("  --query-provider COORDS      Attention query provider. "
           "If current layer is\n"
           "                               not an attention layer, current "
           "layer\n"
           "                               will be set as provider of the "
           "layer defined by\n"
           "                               COORDS. (See 'LAYER COORDINATES' "
           "section for\n"
           "                               details about COORDS).\n"
    );
    printf("  --recurrent-layer            Recurrent layer mode.\n");
    printf("  --stride STRIDE              Convolutional stride"
           " (default: 1).\n");
    printf("  --value-provider COORDS      Attention values provider.\n"
           "                               (See 'LAYER COORDINATES' section\n"
           "                               for details about COORDS).\n"
    );
    printf("  --weight-init-mode MODE      Weight initialization mode:\n"
           "                               auto,random,zero (default: auto)."
           "\n");
    printf("  --whole-sequence             Whole sequence mode.\n");
    printf("\n");
    printf("LAYER COORDINATES:\n\n");
    printf("  Format: [model_index:]layer_index\n");
    printf("  Examples:\n");
    printf("      1:2     - Third layer (2) of second model(1)\n");
    printf("      3       - Fourth layer (3) of current model\n");
    printf("\n");
    printf("LOG LEVELS:\n\n");
    printf("  "); printLogLevels(stdout); printf("\n\n");
    printf("TRAIN|TEST OPTIONS:\n\n");
    printf("  --cifar [CLASSES             Dataset format is CIFAR.\n"
           "                               (classes: 10 or 100, default: "
           "10).\n"
    );
    printf("  --max-images                 Max images to load (CIFAR).\n");
    printf("  --max-files                  Max files to load (CIFAR).\n");
    printf("  --mnist                      Dataset format is MNIST.\n");
#ifdef HAS_MAGICK
    printf("\n");
    printf("IMAGE OPTIONS:\n\n");
    printf("  --background-color COLOR     Padding background color "
           "(ie. none, white, ...),\n"
           "                               default: white\n");
    printf("  --dump-image FILE            Save image to file.\n");
    printf("  --grayscale                  Convert image to grayscale.\n");
    printf("  --invert                     Invert image pixels.\n");
#endif
    printf("\n");
    printf("CONFIG FILES:\n\n");
    printf("  Configuration files can be loaded via the `-c` option (see "
           "above). Every\n"
           "  option that can be passed to the command line can also be used "
           "inside\n"
           "  configuration files by removing the dash prefix ('-' or '--'), "
           "for example:\n"
           "  `layer` instead of `--layer` or `learning-rate` instead of "
           "`--learning-rate`.\n"
           "  Option arguments can follow the option name by separating them "
           "with spaces\n"
           "  and every option should be written in a separate line.\n"
           "  The special `include` directive has the same effect of the `-c` "
           "option,\n"
           "  and it loads another configuration file (ie. `include "
           "/path/to/config`).\n\n");
    printf("SCRIPTS:\n\n");
    printf(
        "  By using options like `--on-batch-trained` or "
        "`--on-epoch-trained` it's\n"
        "  possible to execute an arbitrary external script when such events "
        "happen.\n"
        "  The scripts will eventually receive the following arguments:\n"
        "    --event TYPE, --name MODEL_NAME --epoch CURRENT_EPOCH\n"
        "    --epochs TOT_EPOCHS --average-loss AVERAGE_LOSS --current-loss\n"
        "    CURRENT_LOSS --accuracy CURRENT_ACCURACY --learning-rate RATE\n"
    );
    printf("\n");
}

/*
 * Copyright (C) 2016-2023 Fabio Nicotra <artix2 at gmail dot com>.
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
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "../psyc.h"
#include "../dataset.h"
#include "../debug.h"
#include "../log.h"
#include "../utils.h"

#define EPOCHS 30
#define HIDDEN_SIZE 30
#define LEARNING_RATE 3.0
#define OPTIMIZATION PSDefaultOptimization

char *getOptimizationName(PSOptimization optimization);

/* Globals */

static char *load_from = NULL;
int use_softmax = 0;
int hidden_size = HIDDEN_SIZE;
int epochs = EPOCHS;
PSFloat learning_rate = LEARNING_RATE;
PSOptimization optimization = OPTIMIZATION;

/**** Utils ****/

static char *getExecutablePath(char *executable) {
    static char path[PATH_MAX + 1] = {0};
    char _realpath[PATH_MAX + 1];
    if (path[0]) return path;
    _realpath[0] = 0;
    if (realpath(executable, _realpath) != NULL) {
        char *dir = dirname(_realpath);
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
    }
    return NULL;
}

static char *getExecutableRootPath(char *executable) {
    static char path[PATH_MAX + 1] = {0};
    char *execpath = getExecutablePath(executable);
    if (execpath == NULL) {
        PSErr(NULL, "could not determine executable path");
        return NULL;
    }
    strncpy(path, execpath, PATH_MAX);
    char *p = strstr(path, "/bin");
    if (p != NULL) *p = '\0';
    return path;
}

static int findMNISTFiles(char *executable, char **mnist_files, int *found) {
    static char *fnames[] = {
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    };
    int ok = 1, i;
    char *root_path = getExecutableRootPath(executable);
    if (root_path == NULL) return 0;
    char *resources_path = PSPathJoin(2, root_path, "resources");
    if (resources_path == NULL) return 0;
    char *fpath = NULL;
    *found = 0;
    for (i = 0; i < 4; i++) {
        char *mnist_file = mnist_files[i];
        if (mnist_file != NULL) continue;
        fpath = PSPathJoin(2, resources_path, fnames[i]);
        ok = (fpath != NULL);
        if (!ok) break;
        if (PSFileExists(fpath)) {
            mnist_files[i] = fpath;
            *found += 1;
        } else {
            free(fpath);
            fpath = NULL;
        }
    }
    free(resources_path);
    return ok;
}

void printHelp(char *executable) {
    char optimization_name[255] = {0};
    char *uc_optimization_name = getOptimizationName(OPTIMIZATION);
    int namelen = strlen(uc_optimization_name), i;
    for (i = 0; i < namelen; i++)
        optimization_name[i] = tolower(uc_optimization_name[i]);
    printf("Usage: %s [OPTIONS] [TRAIN_IMAGES [TRAIN_LABELS [TEST_IMAGES "
           "[TEST_LABELS]]]]]\n", executable);
    printf("\nOPTIONS:\n\n");
    printf("        -l, --load MODEL_FILE           Load model\n");
    printf("        --hidden-size SIZE              Hidden Layer size "
          "(def. %d)\n", HIDDEN_SIZE);
    printf("        --softmax                       Softmax Output\n");
    printf("        --optimization                  Training Optimization \n"
          "                                        "
          "(adagrad,adadelta,adam,windowgrad,\n"
          "                                         "
          "nesterov, rmsprop, none)\n"
          "                                        "
          "Default: %s\n", optimization_name
    );
    printf("        --epochs EPOCHS                 Epochs (def. %d)\n",
           EPOCHS);
    printf("        --learning-rate RATE            Learning Rate "
        "(def. %g)\n", LEARNING_RATE);
    printf("        --colors                        Enable colorized output\n");
    printf("    -h, --help                          Print this help\n");
}

int parseOptions(int argc, char **argv) {
    int i, last_arg_idx = argc - 1, is_last;
    for (i = 1; i < argc; i++) {
        is_last = (i == last_arg_idx);
        char *arg = argv[i];
        if (strcmp("--load", arg) == 0 && !is_last) {
            load_from = argv[++i];
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
            else if (strcmp("rmsprop", optname) == 0)
                optimization = PSRMSPropOptimization;
            else if (strcmp("none", optname) == 0)
                optimization = PSDefaultOptimization;
            else {
                fprintf(stderr, "Invalid optimization `%s`\n", optname);
                fprintf(
                    stderr, "Valid values: adam, adagrad, adadelta, "
                    "windowgrad, nesterov\n"
                );
                exit(1);
            }
        } else if (strcmp("--hidden-size", arg) == 0 && !is_last) {
            hidden_size = atoi(argv[++i]);
            if (hidden_size <= 0) {
                PSErr(NULL, "invalid --hidden-size");
                exit(1);
            }
        } else if (strcmp("--epochs", arg) == 0 && !is_last) {
            epochs = atoi(argv[++i]);
            if (epochs <= 0) {
                PSErr(NULL, "invalid --epochs");
                exit(1);
            }
        } else if (strcmp("--softmax", arg) == 0) {
            use_softmax = 1;
        } else if (strcmp("--learning-rate", arg) == 0 && !is_last) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp("--colors", arg) == 0) {
            PSLogEnableColor();
        } else if (strcmp("--help", arg) == 0 || strcmp("-h", arg) == 0) {
            printHelp(argv[0]);
            exit(1);
        } else {
            if (arg[0] == '-') {
                PSErr(NULL, "invalid argument '%s'", arg);
                exit(1);
            } else break;
        }
    }
    return i;
}

int main(int argc, char** argv) {
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(/*FE_INVALID | */FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    PSNeuralNetwork *network = NULL;
    PSFloat *training_data = NULL;
    PSFloat *test_data = NULL;
    char *mnist_files[] = {NULL, NULL, NULL, NULL};
    int arg_idx = parseOptions(argc, argv), arg_file_count = 0,
        found_files_count = 0, success = 1, i;
    while (arg_idx < argc && arg_file_count < 4) {
        char *fpath = argv[arg_idx];
        if (!PSFileExists(fpath)) {
            PSErr(NULL, "file not found: '%s'", fpath);
            return 1;
        }
        mnist_files[arg_file_count++] = fpath;
        arg_idx++;
    }
    if (arg_file_count < 4) {
        if (!findMNISTFiles(argv[0], mnist_files, &found_files_count)) {
            PSErr(NULL, "could not find MNIST dataset files, please provide "
                  "their path (use --help for more info)");
            success = 0;
            goto final;
        }
        if ((arg_file_count + found_files_count) < 2) {
            PSErr(NULL, "at least train images and train labels files are "
                  "required, please provide their paths");
            success = 0;
            goto final;
        }
    }
    for (i = 0; i < 4; i++) {
        char *dataset_type = (i < 2 ? "training" : "testing");
        char *datatype = ((i % 2) == 0 ? "images" : "labels");
        char *fpath = mnist_files[i];
        if (fpath == NULL) continue;
        printf("%s %s:\n", dataset_type, datatype);
        printf(" - %s\n", fpath);
    }
    int testlen = 0;
    int datalen = 0;
    int loaded = 0;
    network = PSCreateNetwork("MNIST Demo");
    success = network != NULL;
    if (!success) {
        PSErr(NULL, "Could not create network!");
        goto final;
    }

    if (load_from != NULL) {
        loaded = PSLoadNetwork(network, load_from);
        success = loaded;
        if (!success) {
            PSErr(NULL, "Could not load pretrained network!");
            goto final;
        }
        success = network->size > 0;
        if (!success) {
            PSErr(NULL, "Invalid pretrained network");
            goto final;
        }
    } else {
        PSLayerType output_type = (use_softmax ? SoftMax : FullyConnected);
        PSAddLayer(network, FullyConnected, MNIST_INPUT_SIZE, NULL);
        PSAddLayer(network, FullyConnected, hidden_size, NULL);
        PSAddLayer(network, output_type, 10, NULL);

        success = network->size > 0;
        if (network->size < 1) {
            PSErr(NULL, "Could not add all layers");
            goto final;
        }
        datalen = PSLoadMNISTData(
            DATA_TYPE_TRAINING, mnist_files[0], mnist_files[1],
            &training_data
        );
        success = (datalen > 0 && training_data != NULL);
        if (!success) {
            PSErr(NULL, "Could not load training data");
            goto final;
        }
    }
    if (mnist_files[2] && mnist_files[3]) {
        testlen = PSLoadMNISTData(
            DATA_TYPE_TEST, mnist_files[2], mnist_files[3], &test_data
        );
    };

    printf("Data len: %d\n", datalen);
    PSPrintNetworkInfo(network);

    if (!loaded) PSTrain(network, training_data, datalen, NULL, 0, PSTRAINOPT(
        .optimization = optimization,
        .learning_rate = learning_rate,
        .batch_size = 10,
        .epochs = epochs
    ));
    success = (network->status != STATUS_ERROR);
    if (!success) goto final;

    if (testlen > 0 && test_data != NULL) {
        printf("Test Data len: %d\n", testlen);
        PSTest(network, test_data, testlen, NULL);
    }
final:
    if (found_files_count > 0) {
        for (i = arg_file_count; i < 4; i++) {
            if (mnist_files[i] != NULL) free(mnist_files[i]);
        }
    }
    free(training_data);
    free(test_data);
    PSDeleteNetwork(network);
    return success ? 0 : 1;
}

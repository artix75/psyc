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
#include <stdlib.h>
#include <string.h>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <fenv.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include "../psyc.h"
#include "../optimization.h"
#include "../activation.h"
#include "w2v_training_data.h"

#define BATCHES 1
#define EPOCHS 30
#define LEARNING_RATE   0.025
#define MOMENTUM        0.0
#define L1              0.0
#define L2              0.0

#define DEFAULT_OUTPUT_FILE "/tmp/pretrained.rnn.psmodel"

#define UNUSED(V) ((void) V)

PSNeuralNetwork *network = NULL;
char *output_path = DEFAULT_OUTPUT_FILE;
int pause_requested = 0;

void print_help(char *progname) {
    printf("Usage %s OPTIONS\n", progname);
    printf("    OPTIONS:\n");
    printf("        -l, --load TRAINED_DT_FILE      Load pretrained model\n");
    printf("        -s, --save TRAINED_DT_FILE      Save trained model\n"
           "                                        "
           "(default: %s)\n", DEFAULT_OUTPUT_FILE);
    printf("        --learning-rate RATE            Learnig Rate "
        "(def. %g)\n", LEARNING_RATE);
    printf("        --momentum MOMENTUM             Momentum "
        "(def. %g)\n", MOMENTUM);
    printf("        --l1-decay DECAY                L1 Weight Decay "
        "(def. %g)\n", L1);
    printf("        --l2-decay DECAY                L2 Weight Decay "
        "(def. %g)\n", L2);
    printf("        --optimization                  Training Optimization \n"
          "                                        "
          "(adagrad,adadelta,adam,windowgrad,\n"
          "                                         "
          "nesterov)\n");
    printf("        --epochs EPOCHS                 Epochs (def. %d)\n",
        EPOCHS);
    printf("        --batch-size SIZE               Batch size (def. %d)\n",
        BATCHES);
#ifdef USE_AVX
    printf("        --disable-avx                   Disable AVX\n");
#endif
    printf("        --shuffle                       Shuffle data\n");
    printf("        -h, --help                      Print this help\n");
}

void handler(int sig) {
    UNUSED(sig);
    if (network != NULL) {
        if (!pause_requested) {
            PSPauseTraining(network);
            pause_requested = 1;
        } else {
            PSAbortTraining(network);
            exit(1);
        }
    }
}

int main(int argc, char** argv) {

    /*signal(SIGSEGV, handler);
    signal(8, handler);*/
    PSHandleSignals(handler);
#if defined(__x86_64__) || defined(__i386__)
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

    const char *pretrained_file = NULL;
    int i;
    int epochs = EPOCHS;
    int batch_size = BATCHES;
    int disable_avx = 0;
    int shuffle = 0;
    PSOptimization optimization = PSDefaultOptimization;
    PSFloat learning_rate = LEARNING_RATE;
    PSFloat momentum = MOMENTUM;
    PSFloat l1_decay = L1;
    PSFloat l2_decay = L2;
    int validate_every = 0;
    UNUSED(disable_avx); /* Actually not used if USE_AVX macro not defined */

    network = PSCreateNetwork("RNN Demo");
    if (network == NULL) {
        fprintf(stderr, "Could not create network!\n");
        return 1;
    }
    network->flags |= FLAG_ONEHOT;

    for (i = 1; i < argc; i++) {
        char *arg = argv[i];
        int is_last = (i == (argc - 1));
        if ((strcmp("--load", arg) == 0 || strcmp("-l", arg) == 0) &&
                   !is_last)
        {
            pretrained_file = argv[++i];
        } else if ((strcmp("--save", arg) == 0 || strcmp("-s", arg) == 0) &&
                   !is_last)
        {
             output_path = argv[++i];
        } else if (strcmp("--epochs", arg) == 0 && (i + 1) < argc) {
            epochs = atoi(argv[++i]);
            if (epochs < 1) {
                fprintf(stderr, "Invalid epochs: at least 1 required\n");
                return 1;
            }
        } else if (strcmp("--learning-rate", arg) == 0 && (i + 1) < argc) {
            learning_rate = (PSFloat) atof(argv[++i]);
            if (learning_rate <= 0.0) {
                fprintf(stderr, "Learning rate must be > 0\n");
                return 1;
            }
        } else if (strcmp("--momentum", arg) == 0 && (i + 1) < argc) {
            momentum = (PSFloat) atof(argv[++i]);
        } else if (strcmp("--l1-decay", arg) == 0 && (i + 1) < argc) {
            l1_decay = (PSFloat) atof(argv[++i]);
        } else if (strcmp("--l2-decay", arg) == 0 && (i + 1) < argc) {
            l2_decay = (PSFloat) atof(argv[++i]);
        } else if (strcmp("--batch-size", arg) == 0 && (i + 1) < argc) {
            batch_size = atoi(argv[++i]);
            if (batch_size < 2) {
                fprintf(stderr, "Batch size must be >= 2\n");
                return 1;
            }
        } else if (strcmp("--validate-every", arg) == 0 && (i + 1) < argc) {
            validate_every = atoi(argv[++i]);
            if (validate_every < 0)  validate_every = 0;
#ifdef USE_AVX
        } else if (strcmp("--disable-avx", arg) == 0) {
            disable_avx = 1;
#endif
        } else if (strcmp("--shuffle", arg) == 0) {
            shuffle = 1;
        } else if (strcmp("--optimization", arg) == 0 && !is_last) {
            char *optname = argv[++i];
            if (strcmp("adam", optname) == 0) optimization = PSAdamOptimization;
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
                return 1;
            }
        } else if (strcmp("--help", arg) == 0 || strcmp("-h", arg) == 0) {
            print_help(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            fprintf(stderr, "Invalid argument %s\n", arg);
            return 1;
        }
    }

    if (pretrained_file == NULL) {
        PSAddLayer(network, FullyConnected, VOCABULARY_SIZE, NULL);
        PSAddLayer(network, Recurrent, 60, NULL);
        PSAddLayer(network, SoftMax, VOCABULARY_SIZE, NULL);
        network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            return 1;
        }
    } else {
        int loaded = PSLoadNetwork(network, pretrained_file);
        if (!loaded) {
            printf("Could not load pretrained data %s\n", pretrained_file);
            PSDeleteNetwork(network);
            return 1;
        }
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            return 1;
        }
    }
#ifdef USE_AVX
    if (disable_avx)
        PSDisableAcceleration(&network->acceleration, PSAcceleration_AVX);
    if (PSAVXEnabled(network->acceleration)) printf("on\n");
    else printf("off\n");
#else
    printf("off\n");
#endif
    PSPrintNetworkInfo(network);
    int flags = TRAINING_ADJUST_RATE;
    if (!shuffle) flags |= TRAINING_NO_SHUFFLE;

    PSTrainingOptions options = {
        .flags = flags,
        .l1_decay = l1_decay,
        .l2_decay = l2_decay,
        .momentum = momentum,
        .optimization = optimization
    };
    PSTrain(network, training_data, TRAIN_DATA_LEN, epochs, learning_rate,
            batch_size, &options, validation_data, EVAL_DATA_LEN);

    if (TEST_DATA_LEN > 0) {
        printf("Test Data len: %d\n", TEST_DATA_LEN);
        PSTest(network, test_data, TEST_DATA_LEN);
    }
    if (output_path != NULL)
        PSSaveNetwork(network, output_path);
    PSDeleteNetwork(network);
    /* free(training_data); */
    /* if (TEST_DATA_LEN) free(test_data); */
    return 0;
}

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
#include <signal.h>
#include <strings.h>

#include <execinfo.h>
#include <fenv.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include "../psyc.h"
#include "../convolutional.h"
#include "../cifar.h"
#include "../debug.h"

#define EPOCHS 200
/* #define BATCH_SIZE 32 */
#define BATCH_SIZE 4
#define FEATURES_COUNT 32
#define REGION_SIZE 5
#define PADDING 2
#define POOL_SIZE 2
#define TRAIN_DATASET_LEN 40000
#define EVAL_DATASET_LEN 10000
#define RELU_ENABLED 1
#define LEARNING_RATE 0.01
#define MOMENTUM 0.0
#define ADDITIONAL_LAYERS 2
#define FC_PREOUTPUT_SIZE 20
#define SOFTMAX_OUTPUT 1
#define L1  0.0
#define L2  0.0001
#define DUMP_ACTIVATIONS_EVERY 4
#define DEFAULT_OUTPUT_FILE "/tmp/pretrained.cnn.psmodel"

#define UNUSED(V) ((void) V)

PSNeuralNetwork *network = NULL;
char *output_path = DEFAULT_OUTPUT_FILE;
int pause_requested;

void print_help(char *progname) {
    printf("Usage %s OPTIONS\n", progname);
    printf("    OPTIONS:\n");
    printf("        -d, --data DATASET_DIR          Dataset directory\n");
    printf("        -l, --load TRAINED_DT_FILE      Load pretrained model\n");
    printf("        -s, --save TRAINED_DT_FILE      Save trained model\n"
           "                                        "
           "(default: %s)\n", DEFAULT_OUTPUT_FILE);
    printf("        --classes CLASSES               10 or 100 (def. 10)\n");
    printf("        --padding PADDING               Padding (def. %d)\n",
        PADDING);
    printf("        --region-size SIZE              Region Size (def. %d)\n",
        REGION_SIZE);
    printf("        --use-relu ONE_OR_ZERO          "
        "Enable/Disable ReLU (def. %d)\n", RELU_ENABLED);
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
        BATCH_SIZE);
    printf("        --additional-layers NUM         Additional Convolutional "
        "Layers (def. %d)\n", ADDITIONAL_LAYERS);
    printf("        --add-fully-connected [SIZE]    "
        "Additional Fully Connected Layer\n"
        "                                        "
        "before Output (def. size %d)\n", FC_PREOUTPUT_SIZE);
    printf("        --softmax-output ENABLED        Softmax Output Layer "
           "(def. %d)\n", SOFTMAX_OUTPUT);
#ifdef USE_AVX
    printf("        --disable-avx                   Disable AVX\n");
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    printf("        --disable-vdsp                  Disable vDSP\n");
#endif
    printf("        --disable-blas                  Disable BLAS\n");
    printf("        --debug-dump-to FILE            Debug training to FILE\n"
           "                                        "
           "(pass 'stdout' for STDOUT)\n");
    printf("        --dump-activations-to FILE      Dump Activations\n");
    printf("        --dump-activations-every NUM    Dump Activations every \n"
            "                                        "
           "NUM batches (def. %d)\n", DUMP_ACTIVATIONS_EVERY);
    printf("        --dump-pretrained-to FILE       Dump Pretrained Network\n");
    printf("        --no-shuffle                    Do not shuffle data\n");
    printf("        --max-batches MAX               Max batches (for debug)\n");
    printf("        --max-images MAX                Max images (for debug)\n");
    printf("        --validate-every BATCH_NUM      Validate inside epochs\n");
    printf("        -h, --help              Print this help\n");
}

FILE *dump_activations_to = NULL;
char *dump_activations_str = NULL;
int dump_activations_every = DUMP_ACTIVATIONS_EVERY;
int max_batches = 0;

void handler(int sig) {
    UNUSED(sig);
    if (network != NULL) {
        if (!pause_requested) {
            PSPauseTraining(network);
            pause_requested = 1;
        } else PSAbortTraining(network);
        /*printf("\n");
        PSSaveNetwork(network, "/tmp/pretrained.cnn.psmodel");
        printf("Deleting network...\n");
        PSDeleteNetwork(network);
        printf("Exiting...\n");*/
    }
}

void onBatchTrained(PSNeuralNetwork *network, int epoch, int epochs,
                    PSFloat loss, PSFloat batch_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{
    UNUSED(epoch);
    UNUSED(epochs);
    UNUSED(loss);
    UNUSED(batch_loss);
    UNUSED(accuracy);
    UNUSED(rate);
    if (dump_activations_str == NULL && max_batches <= 0) return;
    if (network == NULL) return;
    if (network->training == NULL) return;
    int batch = network->training->current_batch;
    if (max_batches > 0 && batch >= max_batches) {
        raise(SIGINT);
        raise(SIGINT);
        return;
    } else if (max_batches > 0 && dump_activations_str == NULL) return;
    if ((batch % dump_activations_every) != 0) return;
    char fname[1024];
    snprintf(fname, 1023, "%s/psyc-activations-batch-%d.dump",
             dump_activations_str, batch);
    PSDumpNetworkActivations(network, fname);
    snprintf(fname, 1023, "%s/psyc-deltas-batch-%d.dump",
             dump_activations_str, batch);
    PSDumpNetworkDeltas(network, fname);
    PSFloat *labels = training_data + CIFAR_IMAGE_SIZE;
    snprintf(fname, 1023, "%s/psyc-labels-batch-%d.dump",
             dump_activations_str, batch);
    FILE *lblfile = fopen(fname, "w");
    if (lblfile == NULL)
        fprintf(stderr, "\nCould not open %s for writing!\n", fname);
    else {
        int i = 0;
        for (; i < 9; i++) {
            char *fmt = (i > 0 ? ",%g" : "%g");
            fprintf(lblfile, fmt, labels[i]);
        }
        fclose(lblfile);
    }
}

int main(int argc, char** argv) {
#if defined(__x86_64__) || defined(__i386__)
    _MM_SET_EXCEPTION_MASK( _MM_GET_EXCEPTION_MASK()
           & ~( _MM_EXCEPT_INVALID |
                _MM_EXCEPT_DENORM |
                _MM_EXCEPT_DIV_ZERO |
                _MM_EXCEPT_OVERFLOW |
                _MM_EXCEPT_UNDERFLOW |
                _MM_EXCEPT_INEXACT ) );
#endif
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(/*FE_INVALID | */FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(handler);
    PSFloat *training_data = NULL;
    PSFloat *test_data = NULL;
    PSFloat *validation_data = NULL;
    const char *pretrained_file = NULL;
    const char *dataset_path = NULL;
    int testsize = 0;
    int datasize = 0;
    int testlen = 0;
    int datalen = 0;
    int valdlen = 0;
    int epochs = EPOCHS;
    int classes = 10, i;
    int use_relu = RELU_ENABLED;
    int padding = PADDING;
    int additional_layers = ADDITIONAL_LAYERS;
    int batch_size = BATCH_SIZE;
    int region_size = REGION_SIZE;
    int add_fully_connected = 0;
    int fc_preoutput_size = FC_PREOUTPUT_SIZE;
    int softmax_output = SOFTMAX_OUTPUT;
    int disable_avx = 0;
    int disable_vdsp = 0;
    int disable_blas = 0;
    int max_images = 0;
    int no_shuffle = 0;
    PSTrainingOptimization optimization = NoTrainingOptimization;
    PSFloat learning_rate = LEARNING_RATE;
    PSFloat momentum = MOMENTUM;
    PSFloat l1_decay = L1;
    PSFloat l2_decay = L2;
    int validate_every = 0;
    UNUSED(disable_avx); /* Actually not used if USE_AVX macro not defined */

    FILE *debug_dump_to = NULL;
    char *debug_output_str = NULL;

    char *dump_pretrained_fname = NULL;

    for (i = 1; i < argc; i++) {
        char *arg = argv[i];
        int is_last = (i == (argc - 1));
        if (strcmp("--classes", arg) == 0 && !is_last) {
            char *next = argv[++i];
            int matched = sscanf(next, "%d", &classes);
            if (!matched) fputs("Invalid classes", stderr);
            if (classes != 10 && classes != 100) {
                fputs("Invalid classes", stderr);
                return 1;
            }
        } else if ((strcmp("--load", arg) == 0 || strcmp("-l", arg) == 0) &&
                   !is_last)
        {
            pretrained_file = argv[++i];
        } else if ((strcmp("--data", arg) == 0 || strcmp("-d", arg) == 0) &&
                   !is_last)
        {
            dataset_path = argv[++i];
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
        } else if (strcmp("--use-relu", arg) == 0 && (i + 1) < argc) {
            use_relu = atoi(argv[++i]);
        } else if (strcmp("--padding", arg) == 0 && (i + 1) < argc) {
            padding = atoi(argv[++i]);
            if (padding < 0) padding = 0;
        } else if (strcmp("--region-size", arg) == 0 && (i + 1) < argc) {
            region_size = atoi(argv[++i]);
            if (region_size < 2) {
                fprintf(stderr, "Region size must be >= 2\n");
                return 1;
            }
            if (region_size > 32) {
                fprintf(stderr, "Region size must be < 32\n");
                return 1;
            }
        } else if (strcmp("--additional-layers", arg) == 0 && (i + 1) < argc) {
            additional_layers = atoi(argv[++i]);
            if (additional_layers < 0) additional_layers = 0;
        } else if (strcmp("--softmax-output", arg) == 0 && (i + 1) < argc) {
            softmax_output =  atoi(argv[++i]);
            if (softmax_output < 0) softmax_output = 0;
        } else if (strcmp("--batch-size", arg) == 0 && (i + 1) < argc) {
            batch_size = atoi(argv[++i]);
            if (batch_size < 2) {
                fprintf(stderr, "Batch size must be >= 2\n");
                return 1;
            }
        } else if (strcmp("--max-batches", arg) == 0 && (i + 1) < argc) {
            max_batches = atoi(argv[++i]);
            if (max_batches < 0)  max_batches = 0;
        } else if (strcmp("--max-images", arg) == 0 && (i + 1) < argc) {
            max_images = atoi(argv[++i]);
            if (max_images < 0)  max_images = 0;
        } else if (strcmp("--validate-every", arg) == 0 && (i + 1) < argc) {
            validate_every = atoi(argv[++i]);
            if (validate_every < 0)  validate_every = 0;
        } else if (strcmp("--add-fully-connected", arg) == 0) {
            add_fully_connected = 1;
            if ((i + 1) < argc && argv[i + 1][0] != '-') {
                fc_preoutput_size = atoi(argv[++i]);
                if (fc_preoutput_size < 10) {
                    fprintf(
                        stderr,
                        "Last fully connected layer's size must be >= 10"
                    );
                    return 1;
                }
            }
        } else if (strcmp("--debug-dump-to", arg) == 0 && (i + 1) < argc) {
            debug_output_str = argv[++i];
            continue;
        } else if (strcmp("--dump-activations-to", arg) == 0 && (i+1) < argc) {
            dump_activations_str = argv[++i];
            continue;
        } else if (strcmp("--dump-activations-every", arg) == 0 && (i+1)<argc) {
            dump_activations_every = atoi(argv[++i]);
            if (dump_activations_every <= 0) dump_activations_every = 1;
        } else if (strcmp("--dump-pretrained-to", arg) == 0 && (i+1) < argc) {
            dump_pretrained_fname = argv[++i];
            continue;
#ifdef USE_AVX
        } else if (strcmp("--disable-avx", arg) == 0) {
            disable_avx = 1;
#endif
        } else if (strcmp("--disable-vdsp", arg) == 0) {
            disable_vdsp = 1;
        } else if (strcmp("--disable-blas", arg) == 0) {
            disable_blas = 1;
        } else if (strcmp("--no-shuffle", arg) == 0) {
            no_shuffle = 1;
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

    if (pretrained_file == NULL && dataset_path == NULL) {
        print_help(argv[0]);
        fprintf(stderr, "FATAL: at least --data or --load needed!\n");
        return 1;
    }

    if (debug_output_str != NULL) {
        if (strcasecmp("stdout", debug_output_str) == 0) debug_dump_to = stdout;
        else {
            debug_dump_to = fopen(debug_output_str, "w");
            if (debug_dump_to == NULL) {
                fprintf(
                    stderr, "FATAL: Could not open '%s' for writing\n",
                    debug_output_str
                );
                return 1;
            }
        }
    }

    int train_dataset_len = TRAIN_DATASET_LEN;
    int eval_dataset_len = EVAL_DATASET_LEN;
    if (max_images) {
        train_dataset_len = max_images;
        eval_dataset_len = 0;
    }

    if (dataset_path != NULL) {
        datasize = PSLoadCIFARData(DATA_TYPE_TRAINING, classes, dataset_path,
                                   &training_data, 0, max_images);
        if (datasize == 0 || training_data == NULL) {
            printf("Could not load training data!\n");
            return 1;
        }
        datalen = datasize / sizeof(PSFloat);
        printf("Loaded training dataset (len: %d, size: %d)\n",
            datalen, datasize);
        if (!max_images) {
            testsize = PSLoadCIFARData(DATA_TYPE_TEST, classes, dataset_path,
                                       &test_data, 0, 0);
            if (testsize == 0 || test_data == NULL) {
                printf("Could not load test data!\n");
                return 1;
            }
            testlen = testsize / sizeof(PSFloat);
            printf("Loaded test dataset (len: %d, size: %d)\n", testlen,
                   testsize);
        }
    }

    network = PSCreateNetwork("CNN CIFAR Demo");
    if (network == NULL) {
        fprintf(stderr, "Could not create network!\n");
        if (training_data != NULL) free(training_data);
        if (test_data != NULL) free(test_data);
        return 1;
    }
    if (dump_activations_str != NULL || max_batches > 0)
        network->onBatchTrained = onBatchTrained;
    printf("Network created!\n");
    printf("AVX: ");
#ifdef USE_AVX
    if (disable_avx)
        PSDisableAcceleration(&network->acceleration, PSAcceleration_AVX);
    if (PSAVXEnabled(network->acceleration)) printf("on\n");
    else printf("off\n");
#else
    printf("off\n");
#endif
    if (disable_vdsp)
        PSDisableAcceleration(&(network->acceleration), PSAcceleration_vDSP);
    if (disable_blas)
        PSDisableAcceleration(&(network->acceleration), PSAcceleration_BLAS);
    printf("vDSP: ");
    if (PSDSPEnabled(network->acceleration)) printf("on\n");
    else printf("off\n");
    printf("BLAS: ");
    if (PSBLASEnabled(network->acceleration)) printf("on\n");
    else printf("off\n");
    printf("Size of PSFloat: %d\n", (int) sizeof(PSFloat));

    if (pretrained_file == NULL) {
        PSHyperParameters *iparams; /* Input layer parameters */
        PSHyperParameters *cparams; /* Convloutional layer parameters */
        PSHyperParameters *pparams; /* Pooling layer parameters */
        iparams = PSCreateConvolutionalParameters(3, 0, 0, 0, 0);
        iparams->parameters[PARAM_OUTPUT_WIDTH] = 32.0;
        iparams->parameters[PARAM_OUTPUT_HEIGHT] = 32.0;
        /*cparams = PSCreateConvolutionalParameters(33, 3, 1, 0, use_relu);
        pparams = PSCreateConvolutionalParameters(33, 2, 0, 0, use_relu);*/

        cparams = PSCreateConvolutionalParameters(16, region_size, 1, padding,
            use_relu);
        pparams = PSCreateConvolutionalParameters(16, 2, 0, 0, use_relu);

        if (cparams == NULL || pparams == NULL) {
            fprintf(stderr, "Could not create layer params!\n");
            PSDeleteNetwork(network);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            return 1;
        }

        PSAddLayer(network, FullyConnected, CIFAR_IMAGE_SIZE, iparams);
        PSAddConvolutionalLayer(network, cparams);
        PSAddPoolingLayer(network, pparams);

        /*cparams = PSCreateConvolutionalParameters(66, 3, 1, 0, use_relu);
        pparams = PSCreateConvolutionalParameters(66, 2, 0, 0, use_relu);
        if (cparams == NULL || pparams == NULL) {
            fprintf(stderr, "Could not create layer params!\n");
            PSDeleteNetwork(network);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            return 1;
        }
        PSAddConvolutionalLayer(network, cparams);
        PSAddPoolingLayer(network, pparams);*/

        for (i = 0; i < additional_layers; i++) {
            cparams = PSCreateConvolutionalParameters(20, region_size, 1,
                padding, use_relu);
            pparams = PSCreateConvolutionalParameters(20, 2, 0, 0,
                use_relu);
            if (cparams == NULL || pparams == NULL) {
                fprintf(stderr, "Could not create layer params!\n");
                PSDeleteNetwork(network);
                if (training_data != NULL) free(training_data);
                if (test_data != NULL) free(test_data);
                return 1;
            }
            PSAddConvolutionalLayer(network, cparams);
            PSAddPoolingLayer(network, pparams);
        }

        /* PSAddLayer(network, FullyConnected, 512, NULL); */
        if (add_fully_connected && fc_preoutput_size >= 10)
            PSAddLayer(network, FullyConnected, fc_preoutput_size, NULL);
        if (softmax_output) PSAddLayer(network, SoftMax, classes, NULL);
        else PSAddLayer(network, FullyConnected, classes, NULL);
        network->loss = PSCrossEntropyLoss;

        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            return 1;
        }

        int element_size = network->input_size + network->output_size;
        printf("Element Size = %d (%d + %d)\n", element_size,
            network->input_size, network->output_size);
        int element_count = datalen / element_size;
        printf("Training elements (initial): %d\n", element_count);
        if (element_count < train_dataset_len) {
            printf("Loaded dataset elements %d < %d\n", element_count,
                   TRAIN_DATASET_LEN);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            PSDeleteNetwork(network);
            return 1;
        } else {
            int remaining = element_count - train_dataset_len;
            if (remaining < eval_dataset_len && eval_dataset_len > 0) {
                printf("WARNING: eval. dataset cannot be > %d!\n", remaining);
                eval_dataset_len = remaining;
            }
            if (remaining == 0) {
                printf("WARNING: no dataset remained for evaluation!\n");
                eval_dataset_len = remaining;
            }
            printf("Evaluation dataset length: %d\n", eval_dataset_len);
            datalen = train_dataset_len *element_size;
            if (eval_dataset_len == 0) validation_data = NULL;
            else {
                validation_data = training_data + datalen;
                valdlen = eval_dataset_len *element_size;
                int validation_elements_count = valdlen / element_size;
                element_count = datalen / element_size;
                printf("Evaluation elements: %d\n", validation_elements_count);
                printf("Training elements: %d\n", element_count);
            }
            if (testlen > 0 && test_data != NULL) {
                int test_elements_count = testlen / element_size;
                printf("Test elements: %d\n", test_elements_count);
            }
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
    PSPrintNetworkInfo(network);
    if (datalen > 0) {
        /*signal(SIGINT, handler);*/
        if (dump_pretrained_fname != NULL)
            PSSaveNetwork(network, dump_pretrained_fname);
        int flags = 0;
        if (no_shuffle) flags |= TRAINING_NO_SHUFFLE;
        PSTrainingOptions train_opts = {
            .flags = flags, .l1_decay = l1_decay, .l2_decay = l2_decay,
            .momentum = momentum, .debug_dump_to = debug_dump_to,
            .validate_every_batches = validate_every
        };
        if (optimization != NoTrainingOptimization)
            train_opts.optimization = optimization;
        PSTrain(network, training_data, datalen, epochs, learning_rate,
                batch_size, &train_opts, validation_data, valdlen);
    }
    if (network->status == STATUS_ERROR) {
        PSDeleteNetwork(network);
        if (training_data != NULL) free(training_data);
        if (test_data != NULL) free(test_data);
        return 1;
    }
    if (testlen > 0 && test_data != NULL && network->status == STATUS_TRAINED) {
        printf("Test Data len: %d\n", testlen);
        PSTest(network, test_data, testlen);
    }
    /* if (pretrained_file == NULL) */
    PSSaveNetwork(network, output_path);
    /* printf("Network saved to: /tmp/pretrained.cnn.psmodel\n"); */
    PSDeleteNetwork(network);
    if (training_data != NULL) free(training_data);
    if (test_data != NULL) free(test_data);
    return 0;
}

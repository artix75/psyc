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
#include <strings.h>
#include <inttypes.h>
#include <signal.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>
#include <sys/time.h>

#include <execinfo.h>
#include <fenv.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include "../platform.h"
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#include <Accelerate/Accelerate.h>
#endif

#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../lstm.h"
#include "../gru.h"
#include "../utils.h"
#include "../debug.h"
#include "../log.h"
#include "../config.h"
#include "../optimization.h"
#include "../activation.h"
/* Data taken from some paragraphs of Wikipedia's article about planet Saturn
 * (https://en.wikipedia.org/wiki/Saturn). */
#include "char_training_data.h"

#define LEARNING_RATE 0.1
#define EPOCHS 5000
#define BATCH_SIZE 1
#define SAMPLE_EVERY_EPOCHS 10
#define SAMPLE_LEN 200
#define CLIP 5.0
#define OPTIMIZATION PSAdaGradOptimization
#define HIDDEN_SIZE 100
#define DEFAULT_OUTPUT_FILE "/tmp/pretrained.char_rnn.psmodel"

#define UNUSED(V) ((void) V)

int hidden_size = HIDDEN_SIZE;
PSFloat learning_rate = LEARNING_RATE;
int epochs = EPOCHS;
PSFloat clip = CLIP;
PSOptimization optimization = OPTIMIZATION;
int sample_every = SAMPLE_EVERY_EPOCHS;
int use_random_choice = 1;
int num_elements = 0;
int do_validate = 1;
PSFloat smooth_loss = 0.0;
PSFloat last_batch_loss = 0.0;
PSLayerType recurrent_ltype = Recurrent;
int catch_fpe = 0;
int log_sequences = 0;
char *load_model_file = NULL;
char *output_path = DEFAULT_OUTPUT_FILE;

void printSample(PSModel *model, int input_idx, int len);
char *getOptimizationName(PSOptimization optimization);

int compareFloats(const void *p1, const void *p2) {
    PSFloat f1 = *(const PSFloat *) p1;
    PSFloat f2 = *(const PSFloat *) p2;
    return (f1 > f2) - (f1 < f2);
}

int randomChoice(PSFloat *weights, int count) {
    int i;
    PSFloat r = PSNormalizedRandom();
    for (i = 0; i < count; i++) {
        PSFloat w = weights[i];
        if (w > r) return i;
    }
    return -1;
}

void onBatchTrained(PSModel *model, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{
    UNUSED(epoch);
    UNUSED(loss);
    UNUSED(accuracy);
    UNUSED(rate);
    UNUSED(model);
    UNUSED(epochs);
    UNUSED(training_data);
    last_batch_loss = current_loss;
    if (log_sequences) {
        int seqlen = (int) training_data[0], i;
        PSFloat *seq = training_data + 1,
                *y = seq + seqlen;
        printf("\nBatch[%d] Seq(len = %d):\n         \"",
            model->training->current_batch, seqlen);
        for (i = 0; i < seqlen; i++) {
            char c = characters[(int) seq[i]];
            if (c == '\n') c = '-';
            printf("%c", c);
        }
        printf("\"\n");
        printf("\n         Y(len = %d):\n         \"", seqlen);
        for (i = 0; i < seqlen; i++) {
            char c = characters[(int) y[i]];
            if (c == '\n') c = '-';
            printf("%c", c);
        }
        printf("\"\n");
    }
}

void onEpochTrained(PSModel *model, int epoch, int epochs,
                    PSFloat loss, PSFloat current_loss, float accuracy,
                    PSFloat *rate, PSFloat *training_data)
{
    UNUSED(epoch);
    UNUSED(epochs);
    UNUSED(loss);
    UNUSED(current_loss);
    UNUSED(accuracy);
    UNUSED(rate);
    UNUSED(training_data);
    if (sample_every > 0 && (epoch % sample_every) != 0) return;
    int iteration =
        (epoch * num_elements) + model->training->current_element;
    printSample(model, 0, SAMPLE_LEN);
    if (last_batch_loss != 0) {
        PSFloat curloss = last_batch_loss * 25;
        smooth_loss = smooth_loss * 0.999 + curloss * 0.001;
        printf("** Smooth loss: %g, Curr. Loss = %g, Iteration = %d\n",
               smooth_loss, curloss, iteration);
    }
    fflush(stdout);
}

void printSample(PSModel *model, int input_idx, int len) {
    int index = 2 + input_idx;
    if (index >= TRAIN_DATA_LEN) {
        fprintf(stderr, "ERROR (%s): Invalid input %d\n", __func__, input_idx);
        return;
    }
    if (!PSResetModelStateSequences(model, 0, 0)) {
        fprintf(stderr, "ERROR (%s): Failed to reset states\n", __func__);
        return;
    }
    PSLayer *out = model->layers[model->size - 1];
    PSFloat word_idx = training_data[index];
    PSFloat data[2];
    data[0] = 1.0;
    data[1] = word_idx;
    int c = len;
    int oldstatus = model->status;
    model->status = STATUS_PAUSED;
    if (PSLogColorEnabled()) printf(PSCOLOR_BOLD);
    printf("\n\n==== SAMPLE ====\n\n");
    if (PSLogColorEnabled()) printf(PSCOLOR_RESET);
    fflush(stdout);
    char character = characters[(unsigned) word_idx];
    printf("%c", character);
    while (c-- >= 0) {
        int ok = PSForward(model, data);
        if (!ok) {
            model->status = oldstatus;
            fprintf(
                stderr, "ERROR (%s): Failed to feed data at t=%d",
                __func__, c + 1
            );
            return;
        }
        int max_idx = 0;
        int t = ((unsigned int) data[0]) - 1;
        if (!use_random_choice) {
            if (!PSFindLayerMaxState(out, NULL, &max_idx, t)) {
                model->status = oldstatus;
                PSErr(__func__, "Failed to find neuron with max value");
                return;
            }
        } else {
            PSFloat *states = PSGetStates(out, t);
            max_idx = randomChoice(states, out->size);
            if (max_idx < 0) {
                if (!PSFindLayerMaxState(out, NULL, &max_idx, t)) {
                    model->status = oldstatus;
                    PSErr(__func__, "Failed to find neuron with max value");
                    return;
                }
            }
        }
        assert(max_idx < VOCABULARY_SIZE);
        character = characters[max_idx];
        printf("%c", character);
        data[1] = (PSFloat) max_idx;
    }
    model->status = oldstatus;
    printf("\n\n");
    fflush(stdout);
}

void printHelp(char *executable) {
    char optimization_name[255] = {0};
    char *uc_optimization_name = getOptimizationName(OPTIMIZATION);
    int namelen = strlen(uc_optimization_name), i;
    for (i = 0; i < namelen; i++)
        optimization_name[i] = tolower(uc_optimization_name[i]);
    printf("Usage %s [OPTIONS]\n", executable);
    printf("    OPTIONS:\n");
    printf("        -l, --load MODEL_FILE           Load model\n");
    printf("        -s, --load MODEL_FILE           Save model\n");
    printf("        --learning-rate RATE            Learnig Rate "
        "(def. %g)\n", LEARNING_RATE);
    printf("        --lstm                          Use LSTM instead of RNN\n");
    printf("        --gru                           Use GRU instead of RNN\n");
    printf("        --clip CLIP                     Gradient clip "
        "(def. %g)\n", CLIP);
    printf("        --optimization                  Training Optimization \n"
          "                                        "
          "(adagrad,adadelta,adam,windowgrad,\n"
          "                                         "
          "nesterov,rmsprop,none)\n"
          "                                        "
          "Default: %s\n", optimization_name
    );
    printf("        --epochs EPOCHS                 Epochs (def. %d)\n",
        EPOCHS);
    printf("        --hidden-size NUM               Hidden layer size "
        "(def. %d)\n", HIDDEN_SIZE);
    printf("        --print-sample-every EPOCHS     Print sample epoch interval"
        "(def. %d)\n", SAMPLE_EVERY_EPOCHS);
    printf("        --no-sample-randomization       Disable sample text "
           "randomization\n");
    printf("        --no-validation                 Do not validate\n");
    /*printf("        --catch-fpe                     Catch floating-point "
        "exceptions\n");
      printf("        --log-sequences                 Log X and Y sequences\n");
    */
    printf("        --colors                        Enable colorized output\n");
    printf("        -h, --help                      Print this help\n");
}

void parseOptions(int argc, char **argv) {
    int i, last_arg, last_idx = argc - 1;
    for (i = 1; i < argc; i++) {
        last_arg = (i == last_idx);
        char *arg = argv[i];
        if (strcmp("--learning-rate", arg) == 0 && !last_arg) {
            learning_rate = atof(argv[++i]);
            if (learning_rate <= 0.0) {
                fprintf(stderr, "ERROR: learning rate must > 0\n");
                exit(1);
            }
        } else if (strcmp("--clip", arg) == 0 && !last_arg) {
            clip = atof(argv[++i]);
            if (clip < 0.0) clip *= -1;
        } else if (strcmp("--epochs", arg) == 0 && !last_arg) {
            epochs = atoi(argv[++i]);
            if (epochs <= 0) {
                fprintf(stderr, "ERROR: epochs must > 0\n");
                exit(1);
            }
        } else if (strcmp("--print-sample-every", arg) == 0 && !last_arg) {
            sample_every = atoi(argv[++i]);
            if (sample_every < 0) sample_every = 0;
        } else if (strcmp("--hidden-size", arg) == 0 && !last_arg) {
            hidden_size = atoi(argv[++i]);
            if (hidden_size < 2) {
                fprintf(stderr, "ERROR: --hidden-size must be >= 2\n");
                exit(1);
            }
        } else if (strcmp("--optimization", arg) == 0 && !last_arg) {
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
                fprintf(stderr, "Invalid optmization `%s`\n", optname);
                fprintf(
                    stderr, "Valid values: adam, adagrad, adadelta, "
                    "windowgrad, nesterov\n"
                );
                exit(1);
            }
        } else if (strcmp("--no-sample-randomization", arg) == 0) {
            use_random_choice = 0;
        } else if (strcmp("--no-validation", arg) == 0) {
            do_validate = 0;
        } else if (strcmp("--catch-fpe", arg) == 0) {
            catch_fpe = 1;
        } else if (strcmp("--lstm", arg) == 0) {
            recurrent_ltype = LSTM;
        } else if (strcmp("--gru", arg) == 0) {
            recurrent_ltype = GRU;
        } else if (strcmp("--log-sequences", arg) == 0) {
            log_sequences = 1;
        } else if (strcmp("--load", arg) == 0 || strcmp("-l", arg) == 0) {
            if (last_arg) {
                fprintf(stderr, "ERROR: missing model file\n");
                exit(1);
            }
            load_model_file = argv[++i];
            continue;
        } else if ((strcmp("--save", arg) == 0 || strcmp("-s", arg) == 0) &&
                   !last_arg)
        {
             output_path = argv[++i];
        } else if (strcmp("--colors", arg) == 0) {
            PSLogEnableColor();
        } else if (strcmp("--help", arg) == 0 || strcmp("-h", arg) == 0) {
            printHelp(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "ERROR: Invalid option `%s`\n", arg);
            exit(1);
        }
    }
}

void initLSTMGRUParams(PSModel *model) {
    PSLayer *layer = model->layers[1];
    if (layer->type != LSTM && layer->type != GRU) {
        PSErr(__func__, "Layer[1] is not LSTM nor GRU");
        PSModelFree(model);
        exit(1);
    }
    PSMathOpts opts = {.acceleration = model->acceleration};
    for (int i = 0; i < layer->weight_types_count; i++) {
        PSMatrix weights = layer->weights[i];
        if (weights == NULL) continue;
        uint64_t wlen = PSMatrixLength(weights);
        if (wlen == 0) continue;
        /*printf("Scaling LSTM/GRU weights[%d] (len = %llu)\n", i, wlen);*/
        PSMultiplyVectorScalar(weights, 0.01, weights, wlen, &opts);
    }
    uint64_t bias_count = PSGetLayerParametersCount(layer, PARAM_TYPE_BIAS);
    if (bias_count > 0) {
        printf(
            "Setting LSTM/GRU biases to zero (len = %" PRIu64 ")\n", bias_count
        );
        memset(layer->biases, 0, bias_count * sizeof(PSFloat));
    }
}

int main(int argc, char **argv) {
    parseOptions(argc, argv);
#if defined(__x86_64__) || defined(__i386__)
    _MM_SET_EXCEPTION_MASK( _MM_GET_EXCEPTION_MASK()
           & ~( _MM_EXCEPT_INVALID |
                _MM_EXCEPT_DENORM |
                _MM_EXCEPT_DIV_ZERO |
                _MM_EXCEPT_OVERFLOW |
                _MM_EXCEPT_UNDERFLOW |
                _MM_EXCEPT_INEXACT ) );
#endif
    if (catch_fpe) PSCatchFloatingPointExceptions(
        /*FE_INVALID | */FE_OVERFLOW | FE_DIVBYZERO
    );
    PSHandleSignals(NULL);
    /*PSFloat *test_data = NULL;
    PSFloat *validation_data = NULL;*/
    int seq_length = 25;
    int ok = 1;
    smooth_loss =
        -PSMathLog(1.0 / (PSFloat) VOCABULARY_SIZE)*(PSFloat)seq_length;
    PSModel *model = PSModelCreate("Char RNN");
    if (model == NULL) {
        fprintf(stderr, "FATAL: Could not create model\n");
        return 1;
    }
    if (load_model_file == NULL) {
        model->flags |= FLAG_ONEHOT;
        PSAddLayer(model, FullyConnected, VOCABULARY_SIZE, NULL);
        PSAddLayer(model, recurrent_ltype, hidden_size, NULL);
        PSAddLayer(model, SoftMax, VOCABULARY_SIZE, NULL);
        model->layers[model->size - 1]->flags |= FLAG_ONEHOT;
        if (LSTM == recurrent_ltype || GRU == recurrent_ltype)
            initLSTMGRUParams(model);
    } else {
        ok = PSModelLoad(model, load_model_file);
        if (!ok) {
            PSErr(NULL, "Could not load model file");
            PSModelFree(model);
            return 1;
        }
    }

    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "Could not build model!\n");
            PSModelFree(model);
            return 1;
        }
    }
    model->acceleration = PSGlobalAcceleration;
    model->loss = PSCrossEntropyLoss;
    PSModelPrintInfo(model);
    model->onEpochTrained = onEpochTrained;
    model->onBatchTrained = onBatchTrained;

    uint32_t flags = (TRAINING_NO_SHUFFLE | TRAINING_EPOCH_AS_SEQUENCE);
    PSTrainingOptions opts = {
        .epochs = epochs,
        .batch_size = BATCH_SIZE,
        .learning_rate = learning_rate,
        .bptt_truncate = 0,
        .flags = flags,
        .clip = clip,
        .optimization = optimization
    };
    num_elements = (int)training_data[0];
    PSFloat *test_data = training_data;
    int test_data_len = TRAIN_DATA_LEN;
    if (!do_validate) {
        test_data = NULL;
        test_data_len = 0;
    }
    if (PSLogColorEnabled()) printf(PSCOLOR_DIM);
    printf("*** NOTE ***\nTraining data taken from some paragraphs of "
           "Wikipedia's article about planet\nSaturn: "
           "(https://en.wikipedia.org/wiki/Saturn).\n\n");
    if (PSLogColorEnabled()) printf(PSCOLOR_RESET);
    PSTrain(model, training_data, TRAIN_DATA_LEN, test_data, test_data_len,
            &opts);
    if (output_path != NULL) PSModelSave(model, output_path);
final:
    PSModelFree(model);
    return (ok ? 0 : 1);
}

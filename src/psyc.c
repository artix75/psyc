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
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <signal.h>
#include <sys/time.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "platform.h"
#include "psyc.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "debug.h"

#define STATUS_ERROR_LOSS -999.00

#define applyGradientOnBias(opts, grad, val, mg, xg, r, i) \
    applyGradientOnParameter(PARAM_TYPE_BIAS, opts, grad, val, mg, xg, r, i, 0)
#define applyGradientOnWeight(opts, grad, val, mg, xg, r, i, widx) \
    applyGradientOnParameter(PARAM_TYPE_WEIGHT, opts, grad, val, mg, xg, r, \
    i, widx)

#ifdef BACKTRACE_AVAILABLE
void segvHandler(int sig, siginfo_t *info, void *secret);
#endif

#define UNUSED(V) ((void) V)

int PSGlobalFlags = 0;

typedef PSFloat (*PSGetDeltaFunction)(PSNeuron* n, PSLayer* l, PSLayer* next,
                                     PSFloat *last_d);

static PSLossFunction loss_functions[] = {
    NULL,
    PSQuadraticLoss,
    PSCrossEntropyLoss
};

static size_t loss_functions_count = sizeof(loss_functions) /
                                     sizeof(PSLossFunction);

/* Function Prototypes */

void PSDeleteLayerGradients(PSGradient *lgradients, int size);
void PSDeleteGradients(PSGradient **gradients, PSNeuralNetwork *network);
char *getLossFunctionName(PSLossFunction function);
char *getNetworkStatusLabel(PSNeuralNetwork *network);

/* Miscellaneous functions */

static void shutdownHandler(int sig) {
    char *msg = NULL;
    switch (sig) {
    case SIGINT:
        msg = "Received SIGINT";
        break;
    case SIGTERM:
        msg = "Received SIGTERM";
        break;
    };
    if (msg != NULL) printf("%s\n", msg);
    else printf("Received shutdown signal %d\n", sig);
    exit(0);
}

void PSHandleSignals(PSSignalHandler shutdown_handler) {
    if (shutdown_handler == NULL) shutdown_handler = shutdownHandler;
    struct sigaction act;

    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    act.sa_handler = shutdown_handler;
    sigaction(SIGTERM, &act, NULL);
    sigaction(SIGINT, &act, NULL);
#ifdef BACKTRACE_AVAILABLE
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_NODEFER | SA_RESETHAND | SA_SIGINFO;
    act.sa_sigaction = segvHandler;
    sigaction(SIGSEGV, &act, NULL);
    sigaction(SIGBUS, &act, NULL);
    sigaction(SIGFPE, &act, NULL);
    sigaction(SIGILL, &act, NULL);
#endif
}

int PSLogTrainingProgress(PSNeuralNetwork *network, int epochs, int batches,
                          int do_clear, char *msg, ...)
{
    if (network->training == NULL) return 0;
    int batch_num = network->training->current_batch + 1;
    int percent =
        (int) roundf(((float) batch_num / (float) batches) * 100.0f);
    if (0 && batch_num >= batches) {
        printf("\r");
        PSFillWithBlank(0);
    }
    int llen = printf(
        "\rEpoch %d/%d: batch %d/%d (%d%%)",
       network->training->current_epoch + 1, epochs,
       batch_num, batches, percent
    );
    if (msg != NULL) {
        va_list ap;
        va_start(ap, msg);
        llen += printf(", ");
        llen += vprintf(msg, ap);
        va_end(ap);
    }
    fflush(stdout);
    if (do_clear) PSFillWithBlank(llen);
    return llen;
}

/* Feedforward Functions */

float validate(PSNeuralNetwork *network, PSFloat *test_data, int data_size,
               int log);

static int fullFeedforward(PSNeuralNetwork *network, PSLayer *layer, ...) {
    int size = layer->size;
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer *previous = network->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int do_dump = PSShouldDebugDump(network);
    int i, j, previous_size = previous->size;
    int is_recurrent = (network->flags & FLAG_RECURRENT), times, t;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat sum = 0.0;
        j = 0;
#ifdef USE_AVX
        if (!avx_disabled) {
            AVXIterativeDotProduct(
                previous_size, previous->avx_activation_cache,
                neuron->weights, sum, j, is_recurrent, t
            );
        }
#endif
        for (; j < previous_size; j++) {
            PSNeuron *prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                PSErr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                      layer->index, j);
                return 0;
            }
            if (do_dump) PSTrainingDebugDumpStep(
                network, TRAINING_PHASE_FEEDFORWARD, "fullFeedforward",
                layer, neuron, "previous_neuron=%d-%d,weight_index=%d\n",
                previous->index, i, j
            );
#ifdef PS_DEBUG_MODE
            PSAddContextualDebug(network, layer, neuron, prev_neuron, NULL, 0);
#endif
            PSFloat a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
#ifdef PS_DEBUG_MODE
        PSAddContextualDebug(network, layer, neuron, NULL, "Sum", sum);
#endif
        neuron->z_value = sum + neuron->bias;
        neuron->activation = layer->activate(neuron->z_value);
#ifdef USE_AVX
        if (!is_recurrent && !avx_disabled)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            PSAddRecurrentState(network, neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                PSErr(__func__, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
#ifdef PS_DEBUG_MODE
        PSResetDebugInfo();
#endif
    }
    return 1;
}

static int softmaxFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
    int size = layer->size;
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer *previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, previous_size = previous->size;
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#endif
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    PSFloat max = 0.0, esum = 0.0;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat sum = 0;
        j = 0;
#ifdef USE_AVX
        if (!avx_disabled) {
            AVXIterativeDotProduct(
                previous_size, previous->avx_activation_cache,
                neuron->weights, sum, j, is_recurrent, t
            );
        }
#endif
        for (; j < previous_size; j++) {
            PSNeuron *prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                PSErr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                      layer->index, j);
                return 0;
            }
            PSFloat a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        if (i == 0)
            max = neuron->z_value;
        else if (neuron->z_value > max)
            max = neuron->z_value;
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat z = neuron->z_value;
        PSFloat e = PSExp(z - max);
        esum += e;
        neuron->activation = e;
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        neuron->activation /= esum;
#ifdef USE_AVX
        if (!is_recurrent && !avx_disabled)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            PSAddRecurrentState(net, neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                PSErr(__func__, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
    }
    return 1;
}

/* Utils */

static PSFloat norm(PSFloat* matrix, int size) {
    PSFloat r = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        PSFloat v = matrix[i];
        r += (v * v);
    }
    /*assert(!isnan(PSSqrt(r)));*/
    PSFloat norm = PSSqrt(r);
    if (isnan(norm)) {
        fprintf(stderr, "\n\nPSSqrt(%g) is nan!\n", r);
        for (i = 0; i < size; i++) {
            PSFloat v = matrix[i];
            fprintf(stderr, " -> matrix[%d] = %g\n", i, v);
        }
        assert(!isnan(norm));
    }
    return norm;
}

static void shuffle(PSFloat *array, int size, int element_size) {
    srand ( time(NULL) );
    int byte_size = element_size * sizeof(PSFloat);
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        /* printf("Shuffle cycle %d: random is %d\n", i, j); */
        PSFloat tmp_a[element_size];
        PSFloat tmp_b[element_size];
        int idx_a = i * element_size;
        int idx_b = j * element_size;
        /* printf("-> idx_a: %d\n", idx_a); */
        /* printf("-> idx_b: %d\n", idx_b); */
        memcpy(tmp_a, array + idx_a, byte_size);
        memcpy(tmp_b, array + idx_b, byte_size);
        memcpy(array + idx_a, tmp_b, byte_size);
        memcpy(array + idx_b, tmp_a, byte_size);
    }
}

static void shuffleSeries(PSFloat **series, int size) {
    srand ( time(NULL) );
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        /* printf("Shuffle cycle %d: random is %d\n", i, j); */
        PSFloat *tmp_a = series[i];
        PSFloat *tmp_b = series[j];
        series[i] = tmp_b;
        series[j] = tmp_a;
    }
}

static PSFloat **getRecurrentSeries(PSFloat *array, int series_count,
                                    int x_size, int y_size)
{
    PSFloat **series = malloc(series_count *sizeof(PSFloat**));
    if (series == NULL) {
        PSErr(NULL, "Could not allocate memory for recurrent series!");
        return NULL;
    }
    int i;
    PSFloat *p = array;
    for (i = 0; i < series_count; i++) {
        int series_size = (int) *p;
        if (!series_size) {
            PSErr(NULL, "Invalid series size 0 at %d", (int) (p - array));
            free(series);
            return NULL;
        }
        series[i] = p++;
        p += ((series_size *x_size) + (series_size *y_size));
    }
    return series;
}

static int arrayMaxIndex(PSFloat *array, int len) {
    int i;
    PSFloat max = 0;
    int max_idx = 0;
    for (i = 0; i < len; i++) {
        PSFloat v = array[i];
        if (v > max) {
            max = v;
            max_idx = i;
        }
    }
    return max_idx;
}

static void fetchRecurrentOutputState(PSLayer *out, PSFloat *outputs,
                                      int i, int onehot)
{
    int t = (onehot ? i : i % out->size), j;
    int max_idx = 0;
    PSFloat max = 0.0;
    for (j = 0; j < out->size; j++) {
        PSNeuron *neuron = out->neurons[j];
        PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
        PSFloat s = cell->states[t];
        if (onehot) {
            if (s > max) {
                max = s;
                max_idx = j;
            }
        } else {
            outputs[i] = s;
        }
    }
    if (onehot) outputs[i] = max_idx;
}

static int compareVersion(const char* vers1, const char* vers2) {
    int major1 = 0, minor1 = 0, patch1 = 0;
    int major2 = 0, minor2 = 0, patch2 = 0;
    sscanf(vers1, "%d.%d.%d", &major1, &minor1, &patch1);
    sscanf(vers2, "%d.%d.%d", &major2, &minor2, &patch2);
    if (major1 < major2) return -1;
    if (major1 > major2) return 1;
    if (minor1 < minor2) return -1;
    if (minor1 > minor2) return 1;
    if (patch1 < patch2) return -1;
    if (patch1 > patch2) return 1;
    return 0;
}

char *PSGetLabelForType(PSLayerType type) {
    switch (type) {
        case FullyConnected:
            return "Fully Connected";
        case Convolutional:
            return "Convolutional";
        case Pooling:
            return "Pooling";
        case Recurrent:
            return "Recurrent";
        case LSTM:
            return "LSTM";
        case SoftMax:
            return "Softmax";
    }
    return "UNKOWN";
}

char *PSGetLayerTypeLabel(PSLayer *layer) {
    return PSGetLabelForType(layer->type);
}

char *getLossFunctionName(PSLossFunction function) {
    if (function == NULL) return "null";
    if (function == PSQuadraticLoss) return "quadratic";
    else if (function == PSCrossEntropyLoss) return "cross_entropy";
    return "UNKOWN";
}

char *getNetworkStatusLabel(PSNeuralNetwork *network) {
    if (network == NULL) return "";
    int status = network->status;
    switch (status) {
    case STATUS_UNTRAINED:
        return "untrained";
    case STATUS_TRAINING:
        return "training";
    case STATUS_TRAINED:
        return "trained";
    case STATUS_ERROR:
        return "error";
    case STATUS_PAUSED:
        return "paused";
    case STATUS_ABORTED:
        return "aborted";
    }
    return "UNKOWN";
}

char *getOptimizationName(PSTrainingOptimization optimization) {
    switch (optimization) {
    case NoTrainingOptimization:
        return "None";
    case Adam:
        return "Adam";
    case AdaGrad:
        return "AdaGrad";
    case AdaDelta:
        return "AdaDelta";
    case Nesterov:
        return "Nesterov";
    case WindowGrad:
        return "WindowGrad";
    }
    return "UNKOWN";
}

static void printLayerInfo(PSLayer *layer) {
    if (layer == NULL) return;
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    PSLayerParameters *lparams = layer->parameters;
    char onehot_info[50];
    onehot_info[0] = 0;
    if (layer->index == 0 && layer->flags & FLAG_ONEHOT) {
        PSLayerParameters *params = layer->parameters;
        int onehot_sz = (int) (params->parameters[0]);
        sprintf(onehot_info, " (vector size: %d)", onehot_sz);
    }
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if (onehot_info[0]) printf(" %s", onehot_info);
    if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
        PSFloat *params = lparams->parameters;
        int fcount = (int) (params[PARAM_FEATURE_COUNT]);
        int rsize = (int) (params[PARAM_REGION_SIZE]);
        int input_w = (int) (params[PARAM_INPUT_WIDTH]);
        int input_h = (int) (params[PARAM_INPUT_HEIGHT]);
        int output_w = (int) (params[PARAM_OUTPUT_WIDTH]);
        int output_h = (int) (params[PARAM_OUTPUT_HEIGHT]);
        int stride = (int) (params[PARAM_STRIDE]);
        int use_relu = (int) (params[PARAM_USE_RELU]);
        if (stride <= 0 && ltype == Pooling) stride = rsize;
        printf(", input size = %dx%d, output_size = %dx%d, features = %d",
            input_w, input_h, output_w, output_h, fcount);
        printf(", region = %dx%d, stride = %d", rsize, rsize, stride);
        if (ltype == Convolutional) {
            char *actv = (use_relu ? "PSRelu" : "PSSigmoid");
            int padding = (int) (params[PARAM_PADDING]);
            if (padding < 0) padding = 0;
            printf(", padding = %d, activation = %s\n", padding, actv);
        } else printf("\n");
    } else if (lparams != NULL && ltype == FullyConnected) {
        PSFloat *params = lparams->parameters;
        int fcount = (int) (params[PARAM_FEATURE_COUNT]);
        if (fcount > 1) printf(", features = %d\n", fcount);
        else printf("\n");
    } else printf("\n");
}

void PSPrintNetworkInfo(PSNeuralNetwork *network) {
    if (network == NULL) return;
    const char *name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    printf("Network: %s\n", name);
    printf("Size: %d\nLayers:\n", network->size);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        printLayerInfo(layer);
    }
    char *loss_name = getLossFunctionName(network->loss);
    printf("Loss Function: %s\n", loss_name);
    printf("Status: %s\n", getNetworkStatusLabel(network));
    printf("AVX: %s\n", (PSIsAVXDisabled(network) ? "no" : "yes"));
}

/* Loss Functions */

PSFloat PSQuadraticLoss(PSFloat *outputs, PSFloat *desired, int size,
                        int onehot_size)
{
    PSFloat *_diffs;
    PSFloat diffs[size];
    if (!onehot_size) {
        int i;
        for (i = 0; i < size; i++) {
            PSFloat d = outputs[i] - desired[i];
            diffs[i] = d;
            if (isnan(d)) {
                fprintf(stderr,
                    "\n\nPSQuadraticLoss: diffs[%d] is nan!\n"
                    " -> output=%g, desired=%g\n",
                    i, outputs[i], desired[i]
                );
                assert(!isnan(d));
            }
        }
        _diffs = diffs;
    } else _diffs = outputs;
    PSFloat n = norm(_diffs, size);
    PSFloat loss = 0.5 * (n * n);
    if (onehot_size) loss /= (PSFloat) onehot_size;
    return loss;
}

PSFloat PSCrossEntropyLoss(PSFloat *outputs, PSFloat *desired, int size,
                        int onehot_size)
{
    PSFloat loss = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        PSFloat o = outputs[i];
        if (o == 0.0) continue;
        if (onehot_size) loss += (PSMathLog(o));
        else {
            if (o == 1) continue; /*  log(1 - 1) would be NaN */
            PSFloat y = desired[i];
            loss += (y * PSMathLog(o) + (1 - y) * PSMathLog(1 - o));
        }
    }
    loss *= -1;
    if (onehot_size)
        loss = (loss / (PSFloat) size);/*  / log((PSFloat) onehot_size); */
    return loss;
}

/* Neural Network Functions */

PSNeuralNetwork *PSCreateNetwork(const char* name) {
    PSNeuralNetwork *network = (malloc(sizeof(PSNeuralNetwork)));
    if (network == NULL) {
        PSErr("PSCreateNetwork", "Could not allocate memory for Network!");
        return NULL;
    }
    network->name = name;
    network->size = 0;
    network->layers = NULL;
    network->input_size = 0;
    network->output_size = 0;
    network->status = STATUS_UNTRAINED;
    network->flags = FLAG_NONE;
    network->loss = PSQuadraticLoss;
    network->training = NULL;
    network->onEpochTrained = NULL;
    network->onBatchTrained = NULL;
    return network;
}

PSNeuralNetwork *PSCloneNetwork(PSNeuralNetwork *network, int layout_only) {
    if (network == NULL) return NULL;
    PSNeuralNetwork *clone = PSCreateNetwork(NULL);
    if (clone == NULL) return NULL;
    if (!layout_only) {
        clone->status = network->status;
        if (network->training != NULL) {
            clone->training = malloc(sizeof(PSTrainingInfo));
            clone->training->current_epoch = network->training->current_epoch;
            clone->training->current_batch = network->training->current_batch;
            clone->training->current_element =
                network->training->current_element;
            clone->training->batch_size = network->training->batch_size;
            clone->training->started_at = network->training->started_at;
            clone->training->ended_at = network->training->ended_at;
            clone->training->debug_dump_to = NULL;
        }
    }
    clone->flags = network->flags;
    clone->loss = network->loss;

    int i, j, k, w;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType type = layer->type;
        PSLayerParameters *oparams = layer->parameters;
        PSLayerParameters *cparams = NULL;
        if (oparams) {
            cparams = malloc(sizeof(PSLayerParameters));
            if (cparams == NULL) {
                PSErr(
                    __func__, "Layer[%d]: Could not allocate layer params!", i
                );
                PSDeleteNetwork(clone);
                return NULL;
            }
            cparams->count = oparams->count;
            cparams->parameters = malloc(cparams->count * sizeof(PSFloat));
            if (cparams->parameters == NULL) {
                PSErr(
                    __func__, "Layer[%d]: Could not allocate layer params!", i
                );
                PSDeleteNetwork(clone);
                return NULL;
            }
            for (j = 0; j < cparams->count; j++)
                cparams->parameters[j] = oparams->parameters[j];
        }
        PSLayer *cloned_layer = PSAddLayer(clone, type, layer->size, cparams);
        if (cloned_layer == NULL) {
            PSDeleteNetwork(clone);
            return NULL;
        }
        cloned_layer->flags = layer->flags;
        if (!layout_only) {
            void *extra = layer->extra;
            if (Convolutional == type && extra) {
                PSSharedParams *oshared = PSGetConvSharedParams(layer);
                PSSharedParams *cshared = PSGetConvSharedParams(cloned_layer);
                cshared->feature_count = oshared->feature_count;
                cshared->weights_size = oshared->weights_size;
                for (k = 0; k < cshared->feature_count; k++) {
                    cshared->biases[k] = oshared->biases[k];
                    for (w = 0; w < cshared->weights_size; w++)
                        cshared->weights[k][w] = oshared->weights[k][w];
                }
            }
            for (j = 0; j < layer->size; j++) {
                PSNeuron *orig_n = layer->neurons[j];
                PSNeuron *clone_n = cloned_layer->neurons[j];
                clone_n->activation = orig_n->activation;
                clone_n->z_value = orig_n->z_value;
                /* if (Pooling == type) continue; */
                clone_n->bias = orig_n->bias;

                if (Convolutional != type && Pooling != type) {
                    PSFloat *oweights = orig_n->weights;
                    PSFloat *cweights = clone_n->weights;
                    for (w = 0; w < orig_n->weights_size; w++)
                        cweights[w] = oweights[w];
                }
                if (layer->flags & FLAG_RECURRENT) {
                    PSRecurrentCell *ocell = PSGetRecurrentCell(orig_n);
                    PSRecurrentCell *ccell = PSGetRecurrentCell(clone_n);
                    int sc = ocell->states_count;
                    ccell->states_count = sc;
                    if (sc > 0) {
                        ccell->states = malloc(sc * sizeof(PSFloat));
                        if (ccell->states == NULL) {
                            PSPrintMemoryErrorMsg();
                            PSDeleteNetwork(clone);
                            return NULL;
                        }
                        for (k = 0; k < sc; k++)
                            ccell->states[k] = ocell->states[k];
                    }
                }
                if (layer->type == LSTM) {
                    PSLSTMCell *ocell = PSGetLSTMCell(orig_n);
                    PSLSTMCell *ccell = PSGetLSTMCell(clone_n);
                    ccell->candidate_bias = ocell->candidate_bias;
                    ccell->input_bias = ocell->input_bias;
                    ccell->output_bias = ocell->output_bias;
                    ccell->forget_bias = ocell->forget_bias;
                }
            }
        }
    }
    return clone;
}

int PSLoadNetwork(PSNeuralNetwork *network, const char* filename) {
    if (network == NULL) return 0;
    FILE *f = fopen(filename, "r");
    printf("Loading network from %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s!\n", filename);
        return 0;
    }
    int netsize, i, j, k;
    int empty = (network->size == 0);
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int epochs = 0, batch_count = 0, elements = 0, status = STATUS_UNTRAINED,
        batch_size = 0;
    int matched = fscanf(f, "--v%d.%d.%d", &v0, &v1, &v2);
    if (matched) {
        sprintf(vers, "%d.%d.%d", v0, v1, v2);
        printf("File version is %s (current: %s).\n", vers, PSYC_VERSION);
        int idx = 0, val = 0;
        PSLossFunction loss = NULL;
        while ((matched = fscanf(f, ",%d", &val))) {
            switch (idx++) {
                case 0:
                    network->flags |= val; break;
                case 1:
                    if ((size_t) val < loss_functions_count) {
                        loss = loss_functions[val];
                        network->loss = loss;
                        printf("Loss Function: %s\n",getLossFunctionName(loss));
                    }
                    break;
                case 2: epochs = val; break;
                case 3: batch_count = val; break;
                case 4: status = val; break;
                case 5: elements = val; break;
                case 6: batch_size = val; break;
                default:
                    break;
            }
        }
        fscanf(f, "\n");
        network->status = status;
        if (status != STATUS_UNTRAINED) {
            if (network->training == NULL) {
                network->training = malloc(sizeof(PSTrainingInfo));
                network->training->requested_action = ACTION_NONE;
                network->training->debug_dump_to = NULL;
            }
            network->training->current_epoch = epochs;
            network->training->current_batch = batch_count;
            network->training->current_element = elements;
            network->training->batch_size = batch_size;
        }
    }
    matched = fscanf(f, "%d:", &netsize);
    if (!matched) {
        PSErr(__func__, "Invalid file %s!", filename);
        fclose(f);
        return 0;
    }
    if (!empty && network->size != netsize) {
        PSErr(__func__, "Network size differs!");
        fclose(f);
        return 0;
    }
    char sep[] = ",";
    char eol[] = "\n";
    int min_argc = (compareVersion(vers, "0.0.0") == 1 ? 2 : 1);
    PSLayer *layer = NULL;
    for (i = 0; i < netsize; i++) {
        int lsize = 0;
        int lflags = 0;
        PSLayerType ltype = FullyConnected;
        int args[20];
        int argc = 0, aidx = 0;
        char *last = (i == (netsize - 1) ? eol : sep);
        char fmt[50];
        char buff[255];
        sprintf(fmt, "%%d%s", last);
        /* fputs(fmt, stderr); */
        matched = fscanf(f, fmt, &lsize);
        if (!matched) {
            int type = 0, arg = 0;
            argc = 0;
            matched = fscanf(f, "[%d,%d", &type, &argc);
            if (!matched) {
                PSErr(__func__, "Invalid header: layer[%d], col. %ld!",
                      i, ftell(f));
                fclose(f);
                return 0;
            }
            if (argc == 0) {
                PSErr(__func__, "Layer must have at least 1 argument (size)");
                fclose(f);
                return 0;
            }
            ltype = (PSLayerType) type;
            for (aidx = 0; aidx < argc; aidx++) {
                matched = fscanf(f, ",%d", &arg);
                if (!matched) {
                    PSErr(__func__, "Invalid header: l%d, arg. %d, col. %ld!",
                          i, aidx, ftell(f));
                    fclose(f);
                    return 0;
                }
                if (aidx == 0) lsize = arg;
                else if (min_argc > 1 && aidx == 1) lflags = arg;
                else args[aidx - min_argc] = arg;
            }
            argc -= min_argc;
            sprintf(fmt, "]%s", last);
            fscanf(f, fmt, buff);
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                PSErr(__func__, "Layer %d size %d differs from %d!", i,
                      layer->size, lsize);
                fclose(f);
                return 0;
            }
            if (ltype != layer->type) {
                PSErr(__func__, "Layer %d type %d differs from %d!", i,
                      (int) (layer->type), (int) ltype);
                fclose(f);
                return 0;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                PSLayerParameters *params = layer->parameters;
                if (params == NULL) {
                    PSErr(__func__, "Layer %d params are NULL!", i);
                    fclose(f);
                    return 0;
                }
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= params->count) break;
                    int arg = args[aidx];
                    PSFloat val = params->parameters[aidx];
                    if (arg != (int) val) {
                        PSErr(__func__, "Layer %d arg[%d] %d diff. from %d!",
                              i, aidx,(int) val, arg);
                        fclose(f);
                        return 0;
                    }
                }
            }
        } else {
            layer = NULL;
            PSLayerParameters *params = NULL;
            if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                params = PSCreateLayerParamenters(param_c);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= param_c) break;
                    int arg = args[aidx];
                    params->parameters[aidx] = (PSFloat) arg;
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            } else {
                if (network->size == 0 && (lflags & FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    network->flags |= FLAG_ONEHOT;
                } else if (argc > 0) {
                    params = PSCreateLayerParamenters(argc);
                    for (aidx = 0; aidx < argc; aidx++) {
                        int arg = args[aidx];
                        params->parameters[aidx] = (PSFloat) arg;
                    }
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            }
            if (layer == NULL) {
                PSErr(__func__, "Could not create layer %d", i);
                fclose(f);
                return 0;
            }
            layer->flags |= lflags;
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = 0;
        PSSharedParams *shared = NULL;
        if (layer->type == Convolutional) {
            shared = PSGetConvSharedParams(layer);
            if (shared == NULL) {
                PSErr(__func__, "Layer %d, missing shared params!", i);
                fclose(f);
                return 0;
            }
            lsize = shared->feature_count;
        } else if (layer->type == Pooling) {
            continue;
        } else lsize = layer->size;
        int is_lstm = (LSTM == layer->type);
        char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
            "," PSFLOAT_FORMAT "|";
        for (j = 0; j < lsize; j++) {
            PSFloat bias = 0;
            int wsize = 0;
            PSFloat *weights = NULL;
            /* LSTM biases */
            PSFloat cb = 0.0, ib = 0.0, ob = 0.0, fb = 0.0;
            if (!is_lstm) matched = fscanf(f, PSFLOAT_FORMAT "|", &bias);
            else matched = fscanf(f, lstm_fmt, &cb, &ib, &ob, &fb);
            if (!matched) {
                PSErr(__func__, "Layer %d, neuron %d: invalid bias!", i, j);
                fclose(f);
                return 0;
            }
            if (shared == NULL) {
                PSNeuron *neuron = layer->neurons[j];
                wsize = neuron->weights_size;
                neuron->bias = bias;
                weights = neuron->weights;
                if (is_lstm) {
                    PSLSTMCell *cell = PSGetLSTMCell(neuron);
                    assert(cell != NULL);
                    cell->candidate_bias = cb;
                    cell->input_bias = ib;
                    cell->output_bias = ob;
                    cell->forget_bias = fb;
                }
            } else {
                shared->biases[j] = bias;
                wsize = shared->weights_size;
                weights = shared->weights[j];
            }
            for (k = 0; k < wsize; k++) {
                PSFloat w = 0;
                char *last = (k == (wsize - 1) ? eol : sep);
                char fmt[5];
                sprintf(fmt, "%s%s", PSFLOAT_FORMAT, last);
                matched = fscanf(f, fmt, &w);
                if (!matched) {
                    PSErr(__func__,"Layer %d neuron %d: invalid weight[%d]",
                          i, j, k);
                    fclose(f);
                    return 0;
                }
                weights[k] = w;
                printf("\rLoading layer %d, neuron %d   ", i, j);
                fflush(stdout);
            }
        }
    }
    printf("\n");
    fclose(f);
    return 1;
}

int PSSaveNetwork(PSNeuralNetwork *network, const char* filename) {
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    printf("Saving network to %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    int i, j, k, loss_function = 0;
    /*  Header */
    fprintf(f, "--v%s", PSYC_VERSION);
    for (i = 0; i < (int) loss_functions_count; i++) {
        if (network->loss == loss_functions[i]) {
            loss_function = i;
            break;
        }
    }
    int current_epoch = 0, current_batch = 0, current_element = 0,
        batch_size = 0;
    if (network->training != NULL) {
        current_epoch = network->training->current_epoch;
        current_batch = network->training->current_batch;
        current_element = network->training->current_element;
        batch_size = network->training->batch_size;
    }
    fprintf(f, ",%d,%d,%d,%d,%d,%d,%d\n", network->flags, loss_function,
            current_epoch, current_batch, network->status, current_element,
            batch_size);
    fprintf(f, "%d:", network->size);
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        if (i > 0) fprintf(f, ",");
        int flags = layer->flags;
        PSLayerParameters *params = layer->parameters;
        if (FullyConnected == ltype && !flags && !params)
            fprintf(f, "%d", layer->size);
        else if (params) {
            int argc = params->count;
            fprintf(f, "[%d,%d,%d,%d", (int) ltype, 2 + argc, layer->size,
                    layer->flags);
            for (j = 0; j < argc; j++) {
                fprintf(f, ",%d", (int) (params->parameters[j]));
            }
            fprintf(f, "]");
        } else {
            fprintf(f, "[%d,2,%d,%d]", (int) ltype, layer->size, flags);
        }
    }
    fprintf(f, "\n");
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        int lsize = layer->size;
        if (Convolutional == ltype) {
            PSSharedParams *shared = PSGetConvSharedParams(layer);
            if (shared == NULL) {
                PSErr(__func__, "Layer[%d]: shared params are NULL!", i);
                fclose(f);
                return 0;
            }
            int feature_count = shared->feature_count;
            if (feature_count < 1) {
                PSErr(__func__, "Layer[%d]: feature count must be >= 1!", i);
                fclose(f);
                return 0;
            }
            for (j = 0; j < feature_count; j++) {
                PSFloat bias = shared->biases[j];
                PSFloat *weights = shared->weights[j];
                fprintf(f, "%.15e|", bias);
                for (k = 0; k < shared->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    PSFloat w = weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        }
        else if (Pooling == ltype) continue;
        else {
            int is_lstm = (LSTM == ltype);
            for (j = 0; j < lsize; j++) {
                PSNeuron *neuron = layer->neurons[j];
                if (!is_lstm)
                    fprintf(f, "%.15e|", neuron->bias);
                else {
                    PSLSTMCell *cell = PSGetLSTMCell(neuron);
                    assert(cell != NULL);
                    fprintf(f, "%.15e,%.15e,%.15e,%.15e|",
                            cell->candidate_bias,
                            cell->input_bias,
                            cell->output_bias,
                            cell->forget_bias);
                }
                for (k = 0; k < neuron->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    PSFloat w = neuron->weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        }
    }
    fclose(f);
    return 1;
}

static void DumpNetworkHeader(PSNeuralNetwork *network, FILE *dump_file) {
    fprintf(dump_file, "psyc:version=%s\n", PSYC_VERSION);
    const char *name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    fprintf(
        dump_file, "network:name=%s,size=%d,status=%s\n", name, network->size,
        getNetworkStatusLabel(network)
    );
    if (network->training != NULL) {
        fprintf(dump_file,
            "training:started_at=%ld,current_epoch=%d,current_batch=%d,"
            "current_element=%d,batch_size=%d\n",
            network->training->started_at, network->training->current_epoch,
            network->training->current_batch,
            network->training->current_element, network->training->batch_size
        );
    }
}

static void DumpLayerInfo(PSLayer *layer, FILE *dump_file, int add_new_line) {
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    PSLayerParameters *lparams = layer->parameters;
    fprintf(dump_file, "layer:index=%d,type=%s,size=%d", layer->index,
        type_name, layer->size);
    if (layer->index == 0 && layer->flags & FLAG_ONEHOT) {
        PSLayerParameters *params = layer->parameters;
        int onehot_sz = (int) (params->parameters[0]);
        fprintf(dump_file, ",vector_size=%d", onehot_sz);
    }
    int fcount = 1;
    if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
        PSFloat *params = lparams->parameters;
        fcount = (int) (params[PARAM_FEATURE_COUNT]);
        int rsize = (int) (params[PARAM_REGION_SIZE]);
        int input_w = (int) (params[PARAM_INPUT_WIDTH]);
        int input_h = (int) (params[PARAM_INPUT_HEIGHT]);
        int output_w = (int) (params[PARAM_OUTPUT_WIDTH]);
        int output_h = (int) (params[PARAM_OUTPUT_HEIGHT]);
        int stride = (int) (params[PARAM_STRIDE]);
        int use_relu = (int) (params[PARAM_USE_RELU]);
        if (stride <= 0 && ltype == Pooling) stride = rsize;
        fprintf(
            dump_file, ",input_size=%dx%d,output_size=%dx%d,features=%d"
            ",region=%dx%d,stride=%d",
            input_w, input_h, output_w, output_h, fcount,
            rsize, rsize, stride
        );
        if (ltype == Convolutional) {
            char *actv = (use_relu ? "PSRelu" : "PSSigmoid");
            int padding = (int) (params[PARAM_PADDING]);
            if (padding < 0) padding = 0;
            fprintf(dump_file, ",padding=%d,activation=%s", padding, actv);
        }
    } else if (lparams != NULL && ltype == FullyConnected) {
        PSFloat *params = lparams->parameters;
        fcount = (int) (params[PARAM_FEATURE_COUNT]);
        if (fcount > 1) fprintf(dump_file, ",features=%d", fcount);
    }
    if (add_new_line) fprintf(dump_file, "\n");
}

int PSDumpNetworkActivations(PSNeuralNetwork *network, const char* filename) {
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    DumpNetworkHeader(network, f);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        PSLayerParameters *lparams = layer->parameters;
        int fcount = 1, nidx = 0;
        if ((ltype == Convolutional || ltype == Pooling ||
            ltype == FullyConnected) && lparams != NULL)
        {
            PSFloat *params = lparams->parameters;
            fcount = (int) (params[PARAM_FEATURE_COUNT]);
        }
        if (fcount < 0) fcount = 1;
        DumpLayerInfo(layer, f, 0);
        fprintf(f, ",activations=(");
        for(; nidx < layer->size; nidx++) {
            char *fmt = (nidx > 0 ? ",%.15e": "%.15e");
            PSNeuron *n = layer->neurons[nidx];
            fprintf(f, fmt, n->activation);
        }
        fprintf(f, ")\n");
    }
    fclose(f);
    return 1;
}

int PSDumpNetworkDeltas(PSNeuralNetwork *network, const char* filename) {
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    DumpNetworkHeader(network, f);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        PSLayerParameters *lparams = layer->parameters;
        int fcount = 1, nidx = 0;
        if ((ltype == Convolutional || ltype == Pooling ||
            ltype == FullyConnected) && lparams != NULL)
        {
            PSFloat *params = lparams->parameters;
            fcount = (int) (params[PARAM_FEATURE_COUNT]);
        }
        if (fcount < 0) fcount = 1;
        DumpLayerInfo(layer, f, 0);
        PSFloat *delta = layer->delta;
        if (delta == NULL) {
            fprintf(f, ",deltas=()\n");
            continue;
        }
        fprintf(f, ",deltas=(");
        for(; nidx < layer->size; nidx++) {
            char *fmt = (nidx > 0 ? ",%.15e": "%.15e");
            PSFloat d = delta[nidx];
            fprintf(f, fmt, d);
        }
        fprintf(f, ")\n");
    }
    fclose(f);
    return 1;
}

void PSDeleteNetwork(PSNeuralNetwork *network) {
    int size = network->size;
    int i, is_recurrent = (network->flags & FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        PSLayer *layer = network->layers[i];
        if (is_recurrent) layer->flags |= FLAG_RECURRENT;
        PSDeleteLayer(layer);
    }
    free(network->layers);
    if (network->training != NULL) free(network->training);
    free(network);
}

void PSDeleteNeuron(PSNeuron *neuron, PSLayer *layer) {
    if (neuron->weights != NULL) free(neuron->weights);
    if (neuron->extra != NULL) {
        if (layer->flags & FLAG_RECURRENT) {
            if (layer->type == LSTM)
                PSDeleteLSTMCell(PSGetLSTMCell(neuron));
            else {
                PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
                if (cell->states != NULL) free(cell->states);
                free(cell);
            }
        } else free(neuron->extra);
    }
    free(neuron);
}

PSLayer *PSAddLayer(PSNeuralNetwork *network, PSLayerType type, int size,
                     PSLayerParameters* params) {
    if (network == NULL) return NULL;
    if (network->size == 0 && type != FullyConnected) {
        PSErr(__func__, "First layer type must be FullyConnected");
        return NULL;
    }
    PSLayer *layer = malloc(sizeof(PSLayer));
    if (layer == NULL) {
        PSErr(__func__, "Could not allocate layer %d!", network->size);
        return NULL;
    }
    layer->network = network;
    layer->index = network->size++;
    layer->type = type;
    layer->size = size;
    layer->parameters = params;
    layer->extra = NULL;
    layer->flags = FLAG_NONE;
    layer->delta = NULL;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
    layer->avx_activation_cache = NULL;
#endif
    PSLayer *previous = NULL;
    int previous_size = 0;
    int initialized = 0;
    /* printf("Adding layer %d\n", layer->index); */
    if (network->layers == NULL) {
        network->layers = malloc(sizeof(PSLayer*));
        if (network->layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Could not allocate network layers!");
            return NULL;
        }
        if ((network->flags & FLAG_ONEHOT) && params == NULL) {
            layer->flags |= FLAG_ONEHOT;
            PSLayerParameters *params;
            params = PSCreateLayerParamenters(1, (PSFloat) size);
            layer->parameters = params;
            size = 1;
            layer->size = 1;
        }
        network->input_size = size;
    } else {
        network->layers = realloc(network->layers,
                                  sizeof(PSLayer*) * network->size);
        if (network->layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Could not reallocate network layers!");
            return NULL;
        }
        previous = network->layers[layer->index - 1];
        if (previous == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Previous layer is NULL!");
            return NULL;
        }
        previous_size = previous->size;
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT) {
            PSLayerParameters *params = previous->parameters;
            if (params == NULL) {
                PSAbortLayer(network, layer);
                PSErr(__func__, "Missing layer params on onehot layer[0]!");
                return NULL;
            }
            previous_size = (int) (params->parameters[0]);
        }
        network->output_size = size;
    }
    if (previous && previous->type == Convolutional && type != Pooling) {
        PSErr(__func__, "Layer[%d]: only Pooling type is allowd after a "
              "Convolutional layer (type = %s)", layer->index,
              PSGetLabelForType(type));
        PSAbortLayer(network, layer);
        return NULL;
    }
    if (type == FullyConnected || type == SoftMax) {
        layer->neurons = malloc(sizeof(PSNeuron*) * size);
        if (layer->neurons == NULL) {
            PSErr(
                __func__, "Layer[%d]: could not allocate neurons!",
                layer->index
            );
            PSAbortLayer(network, layer);
            return NULL;
        }
#ifdef USE_AVX
        if (!avx_disabled) {
            layer->avx_activation_cache = calloc(size, sizeof(PSFloat));
            if (layer->avx_activation_cache == NULL) {
                PSPrintMemoryErrorMsg();
                PSAbortLayer(network, layer);
                return NULL;
            }
        }
#endif
        int i, j;
        for (i = 0; i < size; i++) {
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSAbortLayer(network, layer);
                PSErr(__func__, "Could not allocate neuron!");
                return NULL;
            }
            neuron->index = i;
            neuron->extra = NULL;
            if (layer->index > 0) {
                neuron->weights_size = previous_size;
                neuron->bias = PSGaussianRandom(0, 1);
                neuron->weights = malloc(sizeof(PSFloat) * previous_size);
                for (j = 0; j < previous_size; j++) {
                    neuron->weights[j] = PSGaussianRandom(0, 1);
                }
            } else {
                neuron->bias = 0;
                neuron->weights_size = 0;
                neuron->weights = NULL;
            }
            neuron->activation = 0;
            neuron->z_value = 0;
            neuron->layer = layer;
            layer->neurons[i] = neuron;
        }
        if (type != SoftMax) {
            layer->activate = PSSigmoid;
            layer->derivative = PSSigmoidDerivative;
            layer->feedforward = fullFeedforward;
        } else {
            layer->activate = NULL;
            layer->derivative = NULL;
            layer->feedforward = softmaxFeedforward;
            /* network->loss = PSCrossEntropyLoss; */
        }
        initialized = 1;
    } else if (type == Convolutional) {
        initialized = PSInitConvolutionalLayer(network, layer, params);
        /* TODO: Make PSCrossEntropyLoss default also for convolutional? */
    } else if (type == Pooling) {
        initialized = PSInitPoolingLayer(network, layer, params);
    } else if (type == Recurrent) {
        initialized = PSInitRecurrentLayer(network, layer, size, previous_size);
        if (initialized) network->loss = PSCrossEntropyLoss;
    } else if (type == LSTM) {
        initialized = PSInitLSTMLayer(network, layer, size, previous_size);
        if (initialized) network->loss = PSCrossEntropyLoss;
    }
    if (!initialized) {
        PSAbortLayer(network, layer);
        PSErr(__func__, "Could not initialize layer %d!", network->size + 1);
        return NULL;
    }
    if (layer->index > 0) {
        int dsize = layer->size;
        if (type == LSTM) dsize *= 2;
        layer->delta = calloc(dsize, sizeof(PSFloat));
        if (layer->delta == NULL) {
            PSAbortLayer(network, layer);
            PSPrintMemoryErrorMsg();
            PSErr(
                __func__, "Could not initialize layer %d!",
                network->size + 1
            );
            return NULL;
        }
    }
    network->layers[layer->index] = layer;
    printLayerInfo(layer);
    return layer;
}

PSLayer *PSAddConvolutionalLayer(PSNeuralNetwork *network,
                                  PSLayerParameters* params)
{
    return PSAddLayer(network, Convolutional, 0, params);
}

PSLayer *PSAddPoolingLayer(PSNeuralNetwork *network,
                            PSLayerParameters* params)
{
    return PSAddLayer(network, Pooling, 0, params);
}

void PSDeleteLayer(PSLayer* layer) {
    int size = layer->size;
    int i;
    for (i = 0; i < size; i++) {
        PSNeuron* neuron = layer->neurons[i];
        if (layer->type != Convolutional)
            PSDeleteNeuron(neuron, layer);
        else
            free(neuron);
    }
    free(layer->neurons);
    PSLayerParameters *params = layer->parameters;
    if (params != NULL) PSDeleteLayerParamenters(params);
    void *extra = layer->extra;
    if (extra != NULL) {
        if (layer->type == Convolutional) {
            PSSharedParams *shared = (PSSharedParams*) extra;
            int fc = shared->feature_count;
            /* int ws = shared->weights_size; */
            if (shared->biases != NULL) free(shared->biases);
            if (shared->weights != NULL) {
                int i;
                for (i = 0; i < fc; i++) free(shared->weights[i]);
                free(shared->weights);
            }
            free(extra);
        } else free(extra);
    }
#ifdef USE_AVX
    if (layer->avx_activation_cache != NULL) free(layer->avx_activation_cache);
#endif
    if (layer->delta != NULL) free(layer->delta);
    free(layer);
}

PSLayerParameters *PSCreateLayerParamenters(int count, ...) {
    PSLayerParameters *params = malloc(sizeof(PSLayerParameters));
    if (params == NULL) {
        PSErr(NULL, "Could not allocate Layer Parameters!");
        return NULL;
    }
    params->count = count;
    if (count == 0) params->parameters = NULL;
    else {
        params->parameters = malloc(sizeof(PSFloat) *count);
        if (params->parameters == NULL) {
            PSErr(NULL, "Could not allocate Layer Parameters!");
            free(params);
            return NULL;
        }
        va_list args;
        va_start(args, count);
        int i;
        for (i = 0; i < count; i++)
            params->parameters[i] = (PSFloat) (va_arg(args, double));
        va_end(args);
    }
    return params;
}

PSLayerParameters *PSCreateConvolutionalParameters(PSFloat feature_count,
                                                    PSFloat region_size,
                                                    int stride,
                                                    int padding,
                                                    int use_relu)
{
    return PSCreateLayerParamenters(CONV_PARAMETER_COUNT, feature_count,
                                    region_size, (PSFloat) stride,
                                    0.0f, 0.0f, 0.0f, 0.0f,
                                    (PSFloat) padding, (PSFloat) use_relu);
}

int PSSetLayerParameter(PSLayerParameters *params, int param, PSFloat value) {
    if (params->parameters == NULL) {
        int len = param + 1;
        params->parameters = malloc(sizeof(PSFloat) * len);
        if (params->parameters == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memset(params->parameters, 0.0f, sizeof(PSFloat) * len);
        params->count = len;
    } else if (param >= params->count) {
        int len = params->count;
        int new_len = param + 1;
        PSFloat *old_params = params->parameters;
        size_t size = sizeof(PSFloat) * new_len;
        params->parameters = malloc(sizeof(PSFloat) * size);
        if (params->parameters == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memset(params->parameters, 0.0f, sizeof(PSFloat) * size);
        memcpy(params->parameters, old_params, len * sizeof(PSFloat));
        free(old_params);
    }
    params->parameters[param] = value;
    return 1;
}

int PSAddLayerParameter(PSLayerParameters *params, PSFloat val) {
    return PSSetLayerParameter(params, params->count + 1, val);
}

void PSDeleteLayerParamenters(PSLayerParameters *params) {
    if (params == NULL) return;
    if (params->parameters != NULL) free(params->parameters);
    free(params);
}

int feedforwardThroughTime(PSNeuralNetwork *network, PSFloat *values,
                           int times)
{
    if (network == NULL) return 0;
    PSLayer *first = network->layers[0];
    int input_size = first->size;
    int i, t;
    for (t = 0; t < times; t++) {
        for (i = 0; i < input_size; i++) {
            PSNeuron *neuron = first->neurons[i];
            neuron->activation = values[i];
            PSAddRecurrentState(network, neuron, values[i], times, t);
            if (neuron->extra == NULL) {
                PSErr(__func__, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
        for (i = 1; i < network->size; i++) {
            PSLayer *layer = network->layers[i];
            if (layer == NULL) {
                PSErr(__func__, "Layer %d is NULL", i);
                return 0;
            }
            if (layer->feedforward == NULL) {
                PSErr(__func__, "Layer %d feedforward function is NULL", i);
                return 0;
            }
            int ok = layer->feedforward(network, layer, times, t);
            if (!ok) return 0;
        }
        values += input_size;
    }
    return 1;
}

int PSFeedforward(PSNeuralNetwork *network, PSFloat *values) {
    if (network == NULL) return 0;
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    if (network->flags & FLAG_RECURRENT) {
        int times = (int) values[0];
        if (times <= 0) {
            PSErr(__func__, "Recurrent times must be > 0 (found %d)", times);
            return 0;
        }
        return feedforwardThroughTime(network, values + 1, times);
    }
    PSLayer *first = network->layers[0];
    int input_size = first->size, i;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    for (i = 0; i < input_size; i++) {
        first->neurons[i]->activation = values[i];
#ifdef USE_AVX
        if (!avx_disabled) first->avx_activation_cache[i] = values[i];
#endif
    }
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) {
            PSErr(__func__, "Layer %d is NULL!", i);
            return 0;
        }
        if (layer->feedforward == NULL) {
            PSErr(__func__, "Layer %d feedforward function is NULL", i);
            return 0;
        }
        int success = layer->feedforward(network, layer);
        if (!success) return 0;
    }
    return 1;
}

PSGradient *createLayerGradients(PSLayer *layer) {
    if (layer == NULL) return NULL;
    PSGradient *gradients;
    PSLayerType ltype = layer->type;
    if (ltype == Pooling) return NULL;
    int size = layer->size;
    PSLayerParameters *parameters = NULL;
    if (ltype == Convolutional) {
        parameters = layer->parameters;
        if (parameters == NULL) {
            PSErr(__func__, "Layer %d parameters are NULL!", layer->index);
            return NULL;
        }
        size = (int) (parameters->parameters[PARAM_FEATURE_COUNT]);
    }
    gradients = malloc(sizeof(PSGradient) * size);
    if (gradients == NULL) {
        PSErr(__func__, "Could not allocate memory!");
        return NULL;
    }
    int i, ws = 0;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        if (ltype == Convolutional) {
            if (!ws) {
                int region_size =
                    (int) (parameters->parameters[PARAM_REGION_SIZE]);
                ws = region_size *region_size;
                PSLayer *prev_layer = NULL;
                if (layer->index >= 1) {
                    PSNeuralNetwork *net = (PSNeuralNetwork*) layer->network;
                    prev_layer = net->layers[layer->index - 1];
                    PSLayerParameters *prev_params = prev_layer->parameters;
                    if (prev_params != NULL) {
                        int prev_feats =
                            (int)(prev_params->parameters[PARAM_FEATURE_COUNT]);
                        if (prev_feats == 0) prev_feats = 1;
                        ws *= prev_feats;
                    }
                }
            }
        } else {
            ws = neuron->weights_size;
            if (ltype == LSTM) ws += 4; /*  Make room for LSTM biases */
        }
        gradients[i].bias = 0;
        int memsize = sizeof(PSFloat) * ws;
        gradients[i].weights = malloc(memsize);
        if (gradients[i].weights == NULL) {
            PSErr(__func__, "Could not allocate memory!");
            PSDeleteLayerGradients(gradients, size);
            return NULL;
        }
        memset(gradients[i].weights, 0, memsize);
    }
    return gradients;
}

int PSClassify(PSNeuralNetwork *network, PSFloat *values) {
    int ok = PSFeedforward(network, values);
    if (!ok) {
        PSErr("PSClassify", "Feedforward failed");
        return -1;
    };
    int netsize = network->size, outsize = network->output_size, i;
    PSLayer *out = network->layers[netsize - 1];
    PSFloat max = 0.0;
    int max_idx = 0;
    for (i = 0; i < outsize; i++) {
        PSFloat a = out->neurons[i]->activation;
        if (a > max) {
            max = a;
            max_idx = i;
        }
    }
    return max_idx;
}

PSGradient **createGradients(PSNeuralNetwork *network) {
    if (network == NULL) return NULL;
    PSGradient **gradients = malloc(sizeof(PSGradient*) * network->size - 1);
    if (gradients == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int i;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int idx = i - 1;
        gradients[idx] = createLayerGradients(layer);
        if (gradients[idx] == NULL && layer->type != Pooling) {
            PSPrintMemoryErrorMsg();
            PSDeleteGradients(gradients, network);
            return NULL;
        }
    }
    return gradients;
}

void PSDeleteLayerGradients(PSGradient *gradient, int size) {
    int i;
    for (i = 0; i < size; i++) {
        PSGradient g = gradient[i];
        free(g.weights);
    }
    free(gradient);
}

void PSDeleteGradients(PSGradient **gradients, PSNeuralNetwork *network) {
    int i;
    for (i = 1; i < network->size; i++) {
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        PSLayer *layer = network->layers[i];
        int lsize;
        if (layer->type == Convolutional) {
            PSLayerParameters *params = layer->parameters;
            lsize = (int) (params->parameters[PARAM_FEATURE_COUNT]);
        } else lsize = layer->size;
        PSDeleteLayerGradients(lgradients, lsize);
    }
    free(gradients);
}

static void resetDeltas(PSNeuralNetwork *network) {
    if (network->layers == NULL) return;
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer->delta != NULL) {
            int dsize = layer->size;
            if (layer->type == LSTM) dsize *= 2;
            memset(layer->delta, 0, (size_t) dsize * sizeof(PSFloat));
        }
    }
}

PSGradient **backprop(PSNeuralNetwork *network, PSFloat *x, PSFloat *y) {
    if (network == NULL) return NULL;
    PSGradient **gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    PSLayer *outputLayer = network->layers[netsize - 1];
    int osize = outputLayer->size;
    PSGradient *lgradients = gradients[netsize - 2]; /* No gradient for
                                                        inputs */
    PSLayer *previousLayer = network->layers[outputLayer->index - 1];
    resetDeltas(network);
    PSFloat *delta = outputLayer->delta;

    int i, o, w, j, ok = 1;
    int avx_disabled = 1;
#ifdef USE_AVX
    avx_disabled = PSIsAVXDisabled(network);
#endif
    if (x != NULL) {
        ok = PSFeedforward(network, x);
        if (!ok) {
            PSDeleteGradients(gradients, network);
            return NULL;
        }
    }
    int apply_derivative = PSShouldApplyDerivative(network);
    PSFloat softmax_sum = 0.0;
    for (o = 0; o < osize; o++) {
        PSNeuron *neuron = outputLayer->neurons[o];
        PSFloat o_val = neuron->activation;
        PSFloat y_val = y[o];
        PSFloat d = 0.0;
        if (outputLayer->type != SoftMax) {
            d = o_val - y_val;
            if (apply_derivative)
                d *= outputLayer->derivative(neuron->activation);
        } else {
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
        }
        delta[o] = d;
        if (outputLayer->type != SoftMax) {
            PSGradient *gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeMultiplyValue(
                    wsize, previousLayer->avx_activation_cache, d,
                    gradient->weights, w, 0, 0, 0
                );
            }
#endif
            for (; w < wsize; w++) {
                PSFloat prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
                if (avx_disabled)
                    previousLayer->delta[w] += (d * neuron->weights[w]);
            }
            if (!avx_disabled) {
                for (w = 0; w < wsize; w++) {
                    previousLayer->delta[w] += (d * neuron->weights[w]);
                }
            }
        }
    }
    if (outputLayer->type == SoftMax) {
        for (o = 0; o < osize; o++) {
            PSNeuron *neuron = outputLayer->neurons[o];
            PSFloat o_val = neuron->activation;
            if (apply_derivative) delta[o] -= (o_val *softmax_sum);
            PSFloat d = delta[o];
            PSGradient *gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeMultiplyValue(
                    wsize, previousLayer->avx_activation_cache,
                    d, gradient->weights, w, 0, 0, 0
                );
            }
#endif
            for (; w < wsize; w++) {
                PSFloat prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
                if (avx_disabled)
                    previousLayer->delta[w] += (d * neuron->weights[w]);
            }
            if (!avx_disabled) {
                for (w = 0; w < wsize; w++) {
                    previousLayer->delta[w] += (d * neuron->weights[w]);
                }
            }
        }
    }
    for (i = previousLayer->index; i > 0; i--) {
        PSLayer *layer = network->layers[i];
        previousLayer = network->layers[i - 1];
        lgradients = gradients[i - 1];
        int lsize = layer->size;
        PSLayerType ltype = layer->type;
        PSLayerType prev_ltype = previousLayer->type;
        if (FullyConnected == ltype) {
            delta = layer->delta;
            for (j = 0; j < lsize; j++) {
                PSNeuron *neuron = layer->neurons[j];
                PSFloat d = delta[j];
                if (layer->derivative != NULL) {
                    d *= layer->derivative(neuron->activation);
                    delta[j] = d;
                }
                PSGradient *gradient = &(lgradients[j]);
                gradient->bias = d;
                w = 0;
                int wsize = neuron->weights_size;
#ifdef USE_AVX
                if (!avx_disabled) {
                    AVXIterativeMultiplyValue(
                        wsize, previousLayer->avx_activation_cache,
                        d, gradient->weights, w, 0, 0, 0
                    );
                }
#endif
                for (; w < wsize; w++) {
                    PSFloat prev_a = previousLayer->neurons[w]->activation;
                    gradient->weights[w] = d * prev_a;
                    if (avx_disabled && previousLayer->delta != NULL)
                        previousLayer->delta[w] += (d * neuron->weights[w]);
                }
                if (!avx_disabled && previousLayer->delta != NULL) {
                    for (w = 0; w < wsize; w++) {
                        previousLayer->delta[w] += (d * neuron->weights[w]);
                    }
                }
            }
        } else if (Pooling == ltype && Convolutional == prev_ltype) {
            delta = layer->delta;
            PSPoolingBackprop(layer, previousLayer, delta);
        } else if (Convolutional == ltype) {
            PSConvolutionalBackprop(layer, previousLayer, lgradients);
        } else {
            fprintf(stderr, "Backprop from %s to %s not suported!\n",
                    PSGetLayerTypeLabel(layer),
                    PSGetLayerTypeLabel(previousLayer));
            PSDeleteGradients(gradients, network);
            return NULL;
        }
    }
    return gradients;
}

PSGradient **backpropThroughTime(PSNeuralNetwork *network, PSFloat *x,
                                  PSFloat *y, int times)
{
    if (network == NULL) return NULL;
    PSGradient **gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    PSLayer *outputLayer = network->layers[netsize - 1];
    if (outputLayer->type != SoftMax) {
        PSErr("backpropThroughTime",
              "Recurrent networks require a Softmax output layer, "
              "current one is of type %s.", PSGetLayerTypeLabel(outputLayer));
        PSDeleteGradients(gradients, network);
        return NULL;
    }
    int onehot = (outputLayer->flags & FLAG_ONEHOT);
    int osize = outputLayer->size;
    int bptt_truncate = BPTT_TRUNCATE;

    int i, o, w, j, k, t;
    int ok = feedforwardThroughTime(network, x, times);
    if (!ok) {
        PSDeleteGradients(gradients, network);
        return NULL;
    }
    int last_t = times - 1;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    PSFloat *delta;
    PSFloat *last_delta;
    for (t = last_t; t >= 0; t--) {
        int lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        PSLayer *previousLayer = NULL;
        PSLayer *nextLayer = NULL;
        int ysize = (onehot ? 1 : osize);
        int time_offset = t * ysize;
        PSFloat *time_y = y + time_offset;

        PSGradient *lgradients =
            gradients[netsize - 2];/* No gradients for inputs*/
        previousLayer = network->layers[outputLayer->index - 1];
        nextLayer = NULL;

        delta = outputLayer->delta;
        last_delta = delta;

        PSFloat softmax_sum = 0.0;
        int apply_derivative = PSShouldApplyDerivative(network);
        /*  Calculate output deltas, output layer must be Softmax */
        for (o = 0; o < osize; o++) {
            PSNeuron *neuron = outputLayer->neurons[o];
            PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
            PSFloat o_val = cell->states[t];
            PSFloat y_val;
            if (onehot)
                y_val = ((int) *(time_y) == o);
            else
                y_val = time_y[o];
            PSFloat d = 0.0;
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
            delta[o] = d;
        }
        /*  Update gradients for output layer */
        for (o = 0; o < osize; o++) {
            PSNeuron *neuron = outputLayer->neurons[o];
            PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
            PSFloat o_val = cell->states[t];
            if (apply_derivative) delta[o] -= (o_val *softmax_sum);
            PSFloat d = delta[o];
            PSGradient *gradient = &(lgradients[o]);
            gradient->bias = d;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeMultiplyValue(neuron->weights_size,
                    previousLayer->avx_activation_cache, d,
                    gradient->weights, w, 1, t, AVX_STORE_MODE_ADD
                );
            }
#endif
            for (; w < neuron->weights_size; w++) {
                PSNeuron *prev_neuron = previousLayer->neurons[w];
                PSRecurrentCell *prev_cell = PSGetRecurrentCell(prev_neuron);
                PSFloat prev_a = prev_cell->states[t];
                gradient->weights[w] += (d * prev_a);
            }
        }

        /*  Cycle through other layers */
        for (i = previousLayer->index; i > 0; i--) {
            PSLayer *layer = network->layers[i];
            previousLayer = network->layers[i - 1];
            nextLayer = network->layers[i + 1];
            lgradients = gradients[i - 1];
            int lsize = layer->size;
            PSLayerType ltype = layer->type;
            int is_recurrent = (Recurrent == ltype);
            int is_lstm = (LSTM == ltype);
            if (!is_recurrent && !is_lstm) continue;
            /* PSLayerType prev_ltype = previousLayer->type; */

            delta = layer->delta;
            /*  Calculate layer deltas */
            for (j = 0; j < lsize; j++) {
                PSNeuron *neuron = layer->neurons[j];
                PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
                PSFloat sum = 0;
                for (k = 0; k < nextLayer->size; k++) {
                    PSNeuron *nextNeuron = nextLayer->neurons[k];
                    PSFloat weight = nextNeuron->weights[j];
                    PSFloat d = last_delta[k];
                    sum += (d * weight);
                }
                PSFloat dv = sum *layer->derivative(cell->states[t]);
                if (!is_lstm)
                    delta[j] = dv;
                else
                    delta[j] += dv;

                if (!is_recurrent && !is_lstm) {
                    PSGradient *gradient = &(lgradients[i]);
                    gradient->bias += dv;
                    int wsize = neuron->weights_size;
                    if (previousLayer->flags & FLAG_ONEHOT) {
                        PSLayerParameters *params = previousLayer->parameters;
                        if (params == NULL) {
                            fprintf(stderr, "Layer %d params are NULL!\n",
                                    previousLayer->index);
                            return NULL;
                        }
                        int vector_size = (int) params->parameters[0];
                        assert(vector_size > 0);
                        PSNeuron *prev_n = previousLayer->neurons[0];
                        PSRecurrentCell *prev_c = PSGetRecurrentCell(prev_n);
                        PSFloat prev_a = prev_c->states[t];
                        assert(prev_a < vector_size);
                        w = (int) prev_a;
                        gradient->weights[w] += dv;
                    } else {
                        for (w = 0; w < wsize; w++) {
                            PSNeuron *prev_n = previousLayer->neurons[w];
                            PSRecurrentCell *prev_c =
                                PSGetRecurrentCell(prev_n);
                            PSFloat prev_a = prev_c->states[t];
                            gradient->weights[w] += (dv *prev_a);
                        }
                    }
                }
            }
            int ok = 1;
            if (is_recurrent)
                ok = PSRecurrentBackprop(layer, previousLayer, lowest_t,
                                         lgradients, t);
            else if (is_lstm)
                ok = PSLSTMBackprop(layer, previousLayer, lgradients, t);
            if (!ok) return NULL;
            last_delta = layer->delta;
        }
    }
    return gradients;
}

PSFloat applyGradientOnParameter(
    int param_type, PSTrainingOptions *options, PSFloat grad, PSFloat param,
    PSGradient *mg, PSGradient *xg, PSFloat rate, int iteration,
    int param_index
)
{
    PSTrainingOptimization optimization = NoTrainingOptimization;
    PSFloat dx, eps, rho, beta1, beta2, momentum = 0;
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    eps = options->eps;
    rho = options->rho;
    beta1 = options->beta1;
    beta2 = options->beta2;
    momentum = options->momentum;
    optimization = options->optimization;
    PSFloat *mptr = NULL, *xptr= NULL;
    if (param_type == PARAM_TYPE_BIAS) {
        if (mg != NULL) mptr = &(mg->bias);
        if (xg != NULL) xptr = &(xg->bias);
    } else if (param_type == PARAM_TYPE_WEIGHT) {
        if (mg != NULL) mptr = mg->weights + param_index;
        if (xg != NULL) xptr = xg->weights + param_index;
    }
    if (optimization == Adam) {
        assert(mg != NULL);
        assert(xg != NULL);
        assert(beta1 != 0);
        assert(beta2 != 0);
        PSFloat correct1, correct2;
        *mptr = *mptr *beta1 + (1- beta1) * grad;
        *xptr = *xptr *beta2 + (1- beta2) * grad *grad;
        correct1 = *mptr * (1 - PSPow(beta1, iteration));
        correct2 = *xptr * (1 - PSPow(beta2, iteration));
        PSAssertWithMessage(
            correct2 != 0, "Adam optimization at iteration %d, "
            "param_type = %d, beta1=%g, beta2=%g, grad=%g, "
            "*mptr=%g, *xptr=%g, param_index=%d\n", iteration,
            param_type, beta1, beta2, grad, *mptr, *xptr, param_index
        );
        dx =  - rate *correct1 / (PSSqrt(correct2) + eps);
        return param + dx;
    } else if (optimization == AdaGrad) {
        assert(mg != NULL);
        *mptr = *mptr + grad *grad;
        dx = - rate / PSSqrt(*mptr + eps) * grad;
        return param + dx;
    } else if (optimization == WindowGrad) {
        assert(mg != NULL);
        *mptr = rho * *mptr + (1 - rho) * grad *grad;
        dx = - rate / PSSqrt(*mptr + eps) * grad;
        return param + dx;
    } else if (optimization == AdaDelta) {
        assert(mg != NULL);
        assert(xg != NULL);
        *mptr = rho * *mptr + (1 - rho) * grad *grad;
        dx = - PSSqrt((*xptr + eps) / (*mptr + eps)) * grad;
        *xptr = rho * *xptr + (1 - rho) * dx *dx;
        return param + dx;
    } else if (optimization == Nesterov) {
        assert(mg != NULL);
        dx = *mptr;
        *mptr = *mptr *momentum + rate *grad;
        dx = momentum *dx - (1.0 + momentum) * *mptr;
        return param + dx;
    } else {
        /* No Optimization */
        if (momentum != 0) {
            assert(mg != NULL);
            PSFloat dg = momentum * *mptr - rate *grad;
            *mptr = dg;
            return param + dg;
        } else return param - rate *grad;
    }
}

/* Iterate over a single batch of training elements (`training_data`) and
 * obtain  batch's gradients by back-propagation on each element of the batch
 * itself (by calling the `backprop` function).
 * Then, gradients are applied on network's parameters (weights and biases)
 * with the specified learing rate and eventually optimized by weight decay
 * (L1, L2, Weight decay) and the specified optimization method (ie. AdaGrad,
 * AdaDelta, etc.).
 * The function will return the calculated error (loss). */
PSFloat updateNetworkParameters(PSNeuralNetwork *network,
                                PSFloat *training_data,
                                int batch_size, int elements_count,
                                PSTrainingOptions* opts, PSFloat rate,
                                PSGradient **momentum_gradients,
                                PSGradient **aux_gradients, ...)
{
    int i, j, k, w, netsize = network->size, gsize = netsize - 1, times,
        iteration = 0, avx_disabled = 1;
#ifdef USE_AVX
    avx_disabled = PSIsAVXDisabled(network);
#else
    UNUSED(avx_disabled);
#endif
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    /* Create gradients for the current batch. */
    PSGradient **gradients = createGradients(network);
    if (gradients == NULL) {
        network->status = STATUS_ERROR;
        return STATUS_ERROR_LOSS;
    }
    PSGradient **bp_gradients = NULL;
    PSFloat **series = NULL;
    int is_recurrent = network->flags & FLAG_RECURRENT;
    if (is_recurrent) {
        va_list args;
        va_start(args, aux_gradients);
        series = va_arg(args, PSFloat**);
        va_end(args);
        if (series == NULL) {
            PSErr(__func__, "Series is NULL");
            network->status = STATUS_ERROR;
            goto final;
        }
    }
    int do_dump = (
        network->training != NULL &&
        network->training->debug_dump_to != NULL &&
        network->training->current_batch == 0 &&
        network->training->current_epoch == 0
    );
    PSFloat *x; /* Training element */
    PSFloat *y; /* Labels */
    /* Iterate elements of the batch and, for each element, get gradients
     * from the backpropagation of the error. Then, sum the backpropagation
     * gradients to the batch's gradients. */
    for (i = 0; i < batch_size; i++) {
        if (network->training != NULL) {
            network->training->current_element =
                (network->training->current_batch *batch_size) + i;
            iteration = network->training->current_element + 1;
        }
        /* Backpropagate the error through the network layers and get
         * gradients for the current element. */
        if (series == NULL) {
            /* Non-recurrent network */
            int element_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += element_size;
            bp_gradients = backprop(network, x, y);
        } else {
            /* Recurrent network */
            x = series[i];
            times = (int) *(x++);
            if (times == 0) {
                PSErr(__func__, "Series len must b > 0. (batch = %d)", i);
                /*PSDeleteGradients(gradients, network);
                return STATUS_ERROR_LOSS;*/
                network->status = STATUS_ERROR;
                goto final;
            }
            y = x + (times *training_data_size);
            bp_gradients = backpropThroughTime(network, x, y, times);
        }
        if (bp_gradients == NULL) {
            network->status = STATUS_ERROR;
            goto final;
        }
        /* Add current element's gradients obtained through the
         * backpropagation to the batch's gradients. */
        for (j = 0; j < gsize; j++) {
            PSLayer *layer = network->layers[j + 1];
            PSGradient *lgradients_bp = bp_gradients[j];
            PSGradient *lgradients = gradients[j];
            if (lgradients == NULL) continue;
            int lsize = layer->size;
            int wsize = 0;
            if (layer->type == Convolutional) {
                /* Number of gradients for Convolutional layers is determined
                 * on the number of their feature maps/filters and not on
                 * the number of their nodes (neurons). */
                PSLayerParameters *params = layer->parameters;
                lsize = (int) (params->parameters[PARAM_FEATURE_COUNT]);
                int rsize = (int) (params->parameters[PARAM_REGION_SIZE]);
                wsize = rsize *rsize;
                PSLayer *prev_layer = network->layers[j];
                PSLayerParameters *prev_params = prev_layer->parameters;
                if (prev_params != NULL) {
                    int prev_feat_count = (int) (
                        prev_params->parameters[PARAM_FEATURE_COUNT]
                    );
                    if (prev_feat_count > 1) wsize *= prev_feat_count;
                }
            }
            for (k = 0; k < lsize; k++) {
                if (wsize == 0) {
                    PSNeuron *neuron = layer->neurons[k];
                    wsize = neuron->weights_size;
                    if (layer->type == LSTM) wsize += 4; /*  LSTM biases */
                }
                PSGradient *gradient_bp = &(lgradients_bp[k]);
                PSGradient *gradient = &(lgradients[k]);
                gradient->bias += gradient_bp->bias;
                w = 0;
#ifdef USE_AVX
                if (!avx_disabled) {
                    if (do_dump) PSTrainingDebugDumpGradient(
                        network, DEBUG_PHASE_UPDATE_GRADS, __func__,
                        layer, k, wsize, w, 1, AVXGetStepLen(wsize)
                    );
                    AVXIterativeSum(
                        wsize, gradient->weights, gradient_bp->weights,
                        gradient->weights, w, 0
                    );
                }
#endif
                if (do_dump && w < wsize) PSTrainingDebugDumpGradient(
                    network, DEBUG_PHASE_UPDATE_GRADS, __func__,
                    layer, k, wsize, w, 0, 0
                );
                for (; w < wsize; w++)
                    gradient->weights[w] += gradient_bp->weights[w];
            }
        }
        PSDeleteGradients(bp_gradients, network);
        if (network->status == STATUS_PAUSED) break;
    }

    UNUSED(elements_count); /* TODO: remove elements_count arg if not needed */
    PSFloat l1 = 0.0, l2 = 0.0, l1_loss = 0.0, l2_loss = 0.0, momentum = 0.0;
    PSTrainingOptimization optimization = NoTrainingOptimization;
    int use_weight_decay = 0;
    if (opts != NULL) {
        /* If weight decay is enabled (TRAINING_WEIGHT_DECAY flag), l2_decay
         * will be used to directly update weights and not gradients.
         * Furthermore, l1_loss and l2_loss won't be computed nor used in
         * loss calculation.
         * If disabled (default), L1/L2 regularization will be used so l2_decay
         * and l1_decaywill be applied on gradients and L1/L2 loss will be
         * computed and taken into account by final loss. */
        if (opts->l2_decay != 0.0) {
            use_weight_decay = (opts->flags & TRAINING_WEIGHT_DECAY);
            if (use_weight_decay) {
                l2 = opts->l2_decay / batch_size;
                l2 = (1 - (rate *l2));
            } else l2 = opts->l2_decay;
        }
        if (opts->l1_decay != 0.0) {
            if (opts->l2_decay == 0.0)
                use_weight_decay = (opts->flags & TRAINING_WEIGHT_DECAY);
            if (use_weight_decay) {
                l1 = opts->l1_decay / batch_size;
                l1 = (1 - (rate *l1));
            } else l1 = opts->l1_decay;
            /* For the momenti, disable AVX if L1 is used since it would add
             * more complexity in AVX computations.
             * TODO: allow L1 and AVX in the futuer. */
            avx_disabled = 1;
        }
        momentum = opts->momentum;
        optimization = opts->optimization;
    }
    int apply_momentum = (momentum > 0.0);
    int use_optimization = (optimization != NoTrainingOptimization);
    if (apply_momentum || use_optimization) {
        if (momentum_gradients == NULL) {
            network->status = STATUS_ERROR;
            goto final;
        }
        if (optimization == AdaDelta || optimization == Adam) {
            if (aux_gradients == NULL) {
                network->status = STATUS_ERROR;
                goto final;
            }
        }
        avx_disabled = 1;
    }

    /* Update network paramenters (biases, weights, etc.) by apply
     * batch gradients. */
    for (i = 0; i < gsize; i++) {
        /* Get layer gradients */
        PSGradient *lgradients = gradients[i], *mgradients = NULL,
                   *xgradients = NULL;
        if (lgradients == NULL) continue;
        if (momentum_gradients != NULL) mgradients = momentum_gradients[i];
        if (aux_gradients != NULL) xgradients = aux_gradients[i];
        PSLayer *layer = network->layers[i + 1];
        PSLayerType ltype = layer->type;
        int l_size;
        PSSharedParams *shared = NULL;
        /* Layer gradients size for Convolutional layers is determined on
         * layer's feature maps/filters count. Otherwise, layer size
         * is taken into account. */
        if (ltype == Convolutional) {
            PSLayerParameters *params = layer->parameters;
            l_size = (int) (params->parameters[PARAM_FEATURE_COUNT]);
            shared = PSGetConvSharedParams(layer);
        } else l_size = layer->size;
        int is_lstm = ltype == LSTM;
        /* Iterate over layer gradients. */
        for (j = 0; j < l_size; j++) {
            PSGradient *g = &(lgradients[j]), *mg = NULL, *xg = NULL;
            if (mgradients != NULL) mg = &(mgradients[j]);
            if (xgradients != NULL) xg = &(xgradients[j]);

            PSFloat *bias_ptr = NULL, *weights = NULL;
            PSFloat bias;
            int wsize;
            PSNeuron *neuron = NULL;
            if (shared == NULL || is_lstm) {
                neuron = layer->neurons[j];
                bias = neuron->bias;
                bias_ptr = &(neuron->bias);
                weights = neuron->weights;
                wsize = neuron->weights_size;
            } else {
                bias = shared->biases[j];
                bias_ptr = shared->biases + j;
                weights = shared->weights[j];
                wsize = shared->weights_size;
            }

            /* Update Bias */
            PSFloat gbias = g->bias / (PSFloat) batch_size;
            *bias_ptr = applyGradientOnBias(
                opts, gbias, bias,
                mg, xg, rate, iteration
            );
            if (is_lstm) PSUpdateLSTMBiases(
                neuron, g, mg, xg, rate, opts, iteration
            );

            /* Update Weights */

            k = 0;
            /* TODO: implement momentum for AVX too? */
#ifdef USE_AVX
            if (!avx_disabled) {
                PSFloat r = rate / batch_size;
                if (do_dump) PSTrainingDebugDumpGradient(
                    network, DEBUG_PHASE_UPDATE_WEIGHTS, __func__,
                    layer, j, wsize, k, 1, AVXGetStepLen(wsize)
                );
                if (l2 != 0.0) {
                    /* Apply L2 */
                    int kk = 0;
                    if (use_weight_decay) {
                        AVXIterativeMultiplyValues(
                            wsize, weights, l2,
                            g->weights, r, weights,
                            k, 0, 0,
                            AVX_STORE_MODE_NORM,
                            AVX_STORE_MODE_SUB
                        );
                    } else {
                        PSFloat *l2_grads = PSCopyFloats(
                            weights, (size_t) wsize
                        );
                        if (l2_grads == NULL) goto end_avx_weights;
                        AVXIterativeMultiplyValue(
                            wsize, l2_grads, l2, l2_grads, k,
                            0, 0, AVX_STORE_MODE_NORM
                        );
                        k = 0;
                        AVXIterativeSum(
                            wsize, l2_grads, g->weights, l2_grads, k,
                            AVX_STORE_MODE_NORM
                        );
                        k = 0;
                        AVXIterativeMultiplyValue(
                            wsize, l2_grads, r, weights, k, 0, 0,
                            AVX_STORE_MODE_SUB
                        );
                        free(l2_grads);
                        l2_grads = NULL;
                        AVXIterativeDotSquare(
                            wsize, weights, l2_loss, kk, 0, 0
                        );
                    }
                    if (kk < k) { /* AVX Step Length could differ */
                        for (; kk < k; kk++) {
                            PSFloat w = weights[kk];
                            l2_loss += (w * w);
                        }
                    }
                } else {
                    AVXIterativeMultiplyValue(
                        wsize, g->weights, r, weights, k, 0, 0,
                        AVX_STORE_MODE_SUB
                    );
                }
            }
end_avx_weights:
#endif
            if (do_dump && k < wsize) PSTrainingDebugDumpGradient(
                network, DEBUG_PHASE_UPDATE_WEIGHTS, __func__,
                layer, j, wsize, k, 0, 0
            );
            for (; k < wsize; k++) {
                PSFloat grad_w, l1_grad = 0.0, l2_grad = 0.0, w = weights[k];
                if (l1 != 0.0) {
                    if (use_weight_decay) weights[k] *= l1;
                    else {
                        l1_grad = l1 * (w > 0 ? 1 : -1);
                        l1_loss += PSAbs(w);
                    }
                }
                if (l2 != 0.0) {
                    if (use_weight_decay) weights[k] *= l2;
                    else {
                        l2_grad = l2 *w;
                        l2_loss += (w * w);
                    }
                }
                grad_w = (l1_grad + l2_grad + g->weights[k]) /
                         (PSFloat) batch_size;
                weights[k] = applyGradientOnWeight(
                    opts, grad_w, w, mg, xg, rate, iteration, k
                );
            }
        }
    }
final:
    PSDeleteGradients(gradients, network);
    if (network->status == STATUS_ERROR) return STATUS_ERROR_LOSS;
    PSLayer *out = network->layers[netsize - 1];
    int onehot = out->flags & FLAG_ONEHOT;
    if (onehot) label_data_size = 1;
    if (is_recurrent) label_data_size *= times;
    PSFloat outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (!is_recurrent)
            outputs[i] = out->neurons[i]->activation;
        else {
            if (onehot) {
                int idx = (int) *(y + i);
                PSNeuron *n = out->neurons[idx];
                PSRecurrentCell *cell = PSGetRecurrentCell(n);
                outputs[i] = cell->states[i];
            } else fetchRecurrentOutputState(out, outputs, i, 0);
        }
    }
    if (l1 != 0.0) l1_loss *= (opts->l1_decay /batch_size);
    if (l2 != 0.0) l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    int onehot_s = (onehot ? out->size : 0);
    return network->loss(outputs, y, label_data_size, onehot_s) +
           l1_loss + l2_loss;
}

/* Iterate training data for the entire epoch. Unless the training flag
 * TRAINING_NO_SHUFFLE is set, training data is randomly shuffled
 * (Stochastic Gradient Descent). Training data is divided into batches
 * depending on `batch_size` and, for each batch, gradients are generated
 * and applied on network's by the `updateNetworkParameters` */
PSFloat gradientDescent(PSNeuralNetwork *network,
                       PSFloat *training_data,
                       int element_size,
                       int elements_count,
                       PSFloat learning_rate,
                       int batch_size,
                       PSTrainingOptions *options,
                       int epochs,
                       PSFloat *test_data,
                       int test_size) {
    int batches_count = elements_count / batch_size;
    PSFloat **series = NULL;
    int flags = 0, do_validate = 0;
    if (options != NULL) flags = options->flags;
    if (network->flags & FLAG_RECURRENT) {
        if (series == NULL) {
            PSLayer *out = network->layers[network->size - 1];
            int o_size = (out->flags & FLAG_ONEHOT ? 1 : network->output_size);
            series = getRecurrentSeries(training_data,
                                        elements_count,
                                        network->input_size,
                                        o_size);
            if (series == NULL) {
                network->status = STATUS_ERROR;
                return STATUS_ERROR_LOSS;
            }
        }
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffleSeries(series, elements_count);
    } else {
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffle(training_data, elements_count, element_size);
    }
    PSFloat err = 0.0, previous_err = 0.0, avg_err = 0.0,
           acc = 0.0, tot_acc = 0.0, avg_acc = 0.0;
    long tot_t = 0, avg_t, elapsed_t, test_data_size, validations = 0;
    int offset = (element_size *batch_size), validate_every = 0, i;
    PSGradient **momentum_gradients = NULL, **aux_gradients = NULL;
    if (options != NULL) {
        PSTrainingOptimization optimization = options->optimization;
        if (options->momentum != 0 || optimization != NoTrainingOptimization) {
            momentum_gradients = createGradients(network);
            if (momentum_gradients == NULL) {
                network->status = STATUS_ERROR;
                goto final;
            }
        }
        if (optimization == AdaDelta || optimization == Adam) {
            aux_gradients = createGradients(network);
            if (aux_gradients == NULL) {
                network->status = STATUS_ERROR;
                goto final;
            }
        }
        validate_every = options->validate_every_batches;
        if (validate_every > 0 && test_data != NULL) {
            if (options->max_validation_elements <= 0) {
                int max_test_elements = (int) (0.05 * elements_count);
                options->max_validation_elements =
                    (batch_size *options->validate_every_batches) / 2;
                if (options->max_validation_elements > max_test_elements)
                    options->max_validation_elements = max_test_elements;
            }
            test_data_size = options->max_validation_elements *element_size;
            if (test_data_size > test_size) test_data_size = test_size * 0.1;
            do_validate = 1;
        }
    }
    for (i = 0; i < batches_count; i++) {
        network->training->current_batch = i;
        int batch_num = i + 1;
        struct timeval st, et;
        gettimeofday(&st, NULL);
        err += updateNetworkParameters(
            network, training_data, batch_size, elements_count, options,
            learning_rate, momentum_gradients, aux_gradients, series
        );
        gettimeofday(&et, NULL);
        elapsed_t = PSGetElapsedTimeUS(st, et);
        tot_t += elapsed_t;
        avg_t = (tot_t / batch_num);
        if (batch_num < batches_count) {
            char *time_unit = "us";
            if (avg_t >= 1000) {
                avg_t /= 1000;
                time_unit = "ms";
            }
            avg_err = err / (PSFloat) batch_num;
            if (do_validate) {
                if (i > 0 && (batch_num % validate_every) == 0) {
                    PSLogTrainingProgress(network, epochs, batches_count, 1,
                        "validating %d element(s)...",
                        test_data_size / element_size);
                    acc = validate(network, test_data, test_data_size, 0);
                    tot_acc += acc;
                    avg_acc = tot_acc / (PSFloat) ++validations;
                }
                PSLogTrainingProgress(network, epochs, batches_count, 1,
                    "loss = %.2lf, acc. = %.2lf, avg. time = %ld%s",
                    avg_err, avg_acc, avg_t, time_unit);
            } else {
                PSLogTrainingProgress(network, epochs, batches_count, 1,
                    "loss = %.2lf, avg. time = %ld%s",
                    avg_err, avg_t, time_unit);
            }
        } else PSLogTrainingProgress(network, epochs, batches_count, 1, NULL);
        if (network->status == STATUS_ERROR) {
            if (series != NULL) free(series - (i * batches_count));
            return STATUS_ERROR_LOSS;
        }
        if (network->onBatchTrained != NULL) {
            network->onBatchTrained(
                network, network->training->current_epoch,
                epochs, avg_err, previous_err, 0, &learning_rate,
                training_data
            );
        }
        previous_err = avg_err;
        if (series == NULL) training_data += offset;
        else series += batch_size;
        int action = network->training->requested_action;
        if (action == ACTION_ABORT) {
            network->status = action;
            break;
        }
    }
final:
    if (momentum_gradients != NULL)
        PSDeleteGradients(momentum_gradients, network);
    if (aux_gradients != NULL)
        PSDeleteGradients(aux_gradients, network);
    if (series != NULL) free(series - (batch_size *batches_count));
    return err / (PSFloat) batches_count;
}

float validate(PSNeuralNetwork *network, PSFloat *test_data, int data_size,
               int log) {
    int i, j;
    float accuracy = 0.0f;
    int correct_results = 0;
    float correct_amount = 0.0f;
    PSLayer *output_layer = network->layers[network->size - 1];
    int input_size = network->input_size;
    int output_size = network->output_size;
    int y_size = output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int element_size = input_size + output_size;
    int elements_count;
    PSFloat **series = NULL;
    if (network->flags & FLAG_RECURRENT) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(test_data++);
        data_size--;
        if (onehot) y_size = 1;
        series = getRecurrentSeries(test_data,
                                    elements_count,
                                    input_size,
                                    y_size);
        if (series == NULL) {
            network->status = STATUS_ERROR;
            return STATUS_ERROR_LOSS;
        }
    } else elements_count = data_size / element_size;
    /* PSFloat outputs[output_size]; */
    if (log) printf("Test data elements: %d\n", elements_count);
    time_t start_t, end_t;
    char timestr[80];
    struct tm *tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    if (log) printf("Testing started at %s\n", timestr);
    for (i = 0; i < elements_count; i++) {
        if (log) printf("\rTesting %d/%d", i + 1, elements_count);
        fflush(stdout);
        PSFloat *inputs = NULL;
        PSFloat *expected = NULL;
        int times = 0;
        if (series == NULL) {
            /*  Not Recurrent */
            inputs = test_data;
            test_data += input_size;
            expected = test_data;

            int ok = PSFeedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return STATUS_ERROR_LOSS;
            }

            PSFloat max = 0.0;
            int omax = 0;
            int emax = 0;
            for (j = 0; j < output_size; j++) {
                PSNeuron *neuron = output_layer->neurons[j];
                if (neuron->activation > max) {
                    max = neuron->activation;
                    omax = j;
                }
            }
            if (!onehot)
                emax = arrayMaxIndex(expected, output_size);
            else
                emax = *(expected + (times - 1));
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            /*  Recurrent */
            inputs = series[i];
            times = (int) (*inputs);
            if (times == 0) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return STATUS_ERROR_LOSS;
            }
            expected = inputs + 1 + (times *input_size);

            int ok = PSFeedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return STATUS_ERROR_LOSS;
            }

            int label_data_size = y_size *times;
            int correct_states = 0;
            PSFloat outputs[label_data_size];
            for (j = 0; j < label_data_size; j++) {
                fetchRecurrentOutputState(output_layer, outputs, j, onehot);
                if (onehot && (outputs[j] == expected[j])) correct_states++;
                else if (!onehot && j > 0 && (j % y_size) == 0) {
                    int t = (j / y_size) - 1;
                    int omax = arrayMaxIndex(outputs + (t * y_size), y_size);
                    int emax = arrayMaxIndex(expected + (t * y_size), y_size);
                    if (emax == omax) correct_states++;
                }
            }
            correct_amount += (float) correct_states / (float) times;
        }
    }
    if (log) {
        printf("\n");
        fflush(stdout);
    }
    time(&end_t);
    if (log) printf("\nCompleted in %ld sec.\n", end_t - start_t);
    if (series == NULL) {
        accuracy = (float) correct_results / (float) elements_count;
        if (log) printf("Accuracy (%d/%d): %.2f\n",
                        correct_results, elements_count,accuracy);
    } else {
        accuracy = correct_amount / (float) elements_count;
        free(series);
        if (log) printf("Accuracy: %.2f\n", accuracy);
    }
    return accuracy;
}

static void checkTrainingOptions(PSTrainingOptions *options) {
    if (options->optimization != NoTrainingOptimization) {
        if (options->eps == 0) options->eps = DEFAULT_EPS;
        if (options->rho == 0) options->rho = DEFAULT_RHO;
        if (options->beta1 == 0) options->beta1 = DEFAULT_BETA1;
        if (options->beta2 == 0) options->beta2 = DEFAULT_BETA2;
    }
}

void PSSetDefaultTrainingOptions(PSTrainingOptions *options) {
    options->rho = DEFAULT_RHO;
    options->eps = DEFAULT_EPS;
    options->beta1 = DEFAULT_BETA1;
    options->beta2 = DEFAULT_BETA2;
}

void PSTrain(PSNeuralNetwork *network,
             PSFloat *training_data,
             int data_size,
             int epochs,
             PSFloat learning_rate,
             int batch_size,
             PSTrainingOptions *options,
             PSFloat *test_data,
             int test_size) {
    int i, elements_count;
    int element_size = network->input_size + network->output_size;
    int valid = PSVerifyNetwork(network);
    if (!valid) {
        network->status = STATUS_ERROR;
        return;
    }
    if (network->flags & FLAG_RECURRENT) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(training_data++);
        data_size--;
    } else elements_count = data_size / element_size;
    const char *name = network->name != NULL ? network->name : "UNNAMED";
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(BOLD);
    printf("Training network \"%s\"\n", name);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(RESET);
    printf("Training data elements: %d\n", elements_count);
    printf("Batch Size: %d\n", batch_size);
    printf("Learning Rate: %g\n", learning_rate);
    if (options != NULL) {
        checkTrainingOptions(options);
        if (options->validate_every_batches > 0) {
            printf("Validate Every: %d batch(es)\n",
                options->validate_every_batches);
        }
        int use_weight_decay = (
            (options->l1_decay != 0 || options->l2_decay != 0) &&
            (options->flags & TRAINING_WEIGHT_DECAY)
        );
        printf("L1 Decay: %g\n", options->l1_decay);
        printf("L2 Decay: %g\n", options->l2_decay);
        printf("Weight Decay: %s\n", (use_weight_decay ? "yes" : "no"));
        printf("Momentum: %g\n", options->momentum);
        printf("Optimization: %s\n",
            getOptimizationName(options->optimization));
    }
    int was_paused = (network->status == STATUS_PAUSED);
    network->status = STATUS_TRAINING;
    time_t start_t, end_t, epoch_t;
    char timestr[80];
    struct tm *tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(CYAN);
    printf("Training started at %s\n", timestr);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(WHITE);
    epoch_t = start_t;
    time_t e_t = epoch_t;
    PSFloat prev_err = 0.0;
    float acc = -999.99f;
    int adjust_rate = 0;
    if (options != NULL) adjust_rate = (options->flags & TRAINING_ADJUST_RATE);
    int first_epoch = 0;
    if (network->training != NULL) {
        if (was_paused) first_epoch = network->training->current_epoch;
    } else {
        network->training = malloc(sizeof(PSTrainingInfo));
        network->training->started_at = start_t;
        network->training->ended_at = (time_t) 0;
        if (options != NULL)
            network->training->debug_dump_to = options->debug_dump_to;
        else network->training->debug_dump_to = NULL;
        if (network->training->debug_dump_to) PSTrainingDebugDumpHeader(
            network, data_size, test_size, epochs, learning_rate, batch_size
        );
    }
    network->training->batch_size = batch_size;
    network->training->requested_action = ACTION_NONE;
    for (i = first_epoch; i < epochs; i++) {
        network->training->current_epoch = i;
        PSFloat err = gradientDescent(network, training_data, element_size,
                                     elements_count, learning_rate,
                                     batch_size, options, epochs,
                                     test_data, test_size);
        if (network->status == STATUS_ERROR) {
            fprintf(stderr, "\nAn error occurred while training, aborting!\n");
            return;
        }
        char accuracy_msg[255] = "";
        int batches_count = elements_count / batch_size;
        if (test_data != NULL && network->status == STATUS_TRAINING) {
            PSLogTrainingProgress(network, epochs, batches_count, 1,
                "validating...");
            acc = validate(network, test_data, test_size, 0);
            sprintf(accuracy_msg, ", acc = %.2f,", acc);
        }
        time(&epoch_t);
        time_t elapsed_t = epoch_t - e_t;
        e_t = epoch_t;
        if (i > 0 && err > prev_err && adjust_rate)
            learning_rate *= 0.5;
        if (network->onEpochTrained != NULL)
            network->onEpochTrained(network, i, epochs, err, prev_err,
                                    acc, &learning_rate, NULL);
        prev_err = err;
        PSLogTrainingProgress(network, epochs, batches_count, 1,
            "loss = %.2lf%s (%ld sec.)\n", err, accuracy_msg, elapsed_t
        );
        fflush(stdout);
        int action = network->training->requested_action;
        if (action == ACTION_ABORT || action == ACTION_PAUSE) {
            network->status = action;
            break;
        }
    }
    time(&end_t);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(GREEN);
    printf("Completed in %ld sec.\n", end_t - start_t);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(WHITE);
    network->training->ended_at = end_t;
    if (network->status == STATUS_TRAINING) network->status = STATUS_TRAINED;
}

float PSTest(PSNeuralNetwork *network, PSFloat *test_data, int data_size) {
    return validate(network, test_data, data_size, 1);
}

void PSPauseTraining(PSNeuralNetwork *network) {
    if (network->training != NULL) {
        printf("\nPause requested, "
               "training will stop after current epoch will be completed.\n");
        network->training->requested_action = ACTION_PAUSE;
    }
}

void PSAbortTraining(PSNeuralNetwork *network) {
    if (network->training != NULL) {
        printf("\nAborting...\n");
        network->training->requested_action = ACTION_ABORT;
    }
}

int PSVerifyNetwork(PSNeuralNetwork *network) {
    if (network == NULL) {
        PSErr(__func__, "Network is NULL");
        return 0;
    }
    int size = network->size, i;
    PSLayer *previous = NULL;
    int onehot_input = 0;
    for (i = 0; i < size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) {
            PSErr(__func__, "Layer[%d] is NULL", i);
            return 0;
        }
        int ltype = layer->type;
        if (i == 0) {
            if (ltype != FullyConnected) {
                PSErr(__func__, "Layer[%d] type must be '%s'",
                      i, PSGetLabelForType(FullyConnected));
                return 0;
            }
            if (layer->flags & FLAG_ONEHOT) {
                PSLayerParameters *params = layer->parameters;
                onehot_input = 1;
                if (params == NULL) {
                    PSErr(
                        __func__,
                        "Layer[%d] uses a onehot vector index as input, "
                        "but it has no parameters", i
                    );
                    return 0;
                }
                if (params->count < 1) {
                    PSErr(
                        __func__,
                        "Layer[%d] uses a onehot vector index as input, "
                        "but parameters count is < 1", i
                    );
                    return 0;
                }
            }
        }
        if (ltype == Convolutional) {
            if (onehot_input) {
                PSErr(__func__, "ONEHOT input Layer is not supported on "
                      "Convolutional netowrks");
                return 0;
            }
            if (network->flags & FLAG_RECURRENT) {
                PSErr(
                    __func__,
                    "Sorry, Convolutional layers aren't yet supported "
                    "on recurrent networks :("
                );
                return 0;
            }
        }
        if (ltype == Pooling && previous && previous->type != Convolutional) {
            PSErr(__func__, "Layer[%d] type is Pooling, "
                  "but previous type is not Convolutional", i);
            return 0;
        }
        if (ltype != Pooling && previous &&  previous->type == Convolutional) {
            PSErr(__func__, "Layer[%d] previous type is "
                  "Convolutional, but type is not Pooling", i);
            return 0;
        }
        if (layer->activate == PSSigmoid &&
            layer->derivative != PSSigmoidDerivative) {
            PSErr(__func__,
                  "Layer[%d] activate function is PSSigmoid, "
                  "but derivative function is not PSSigmoidDerivative", i);
            return 0;
        }
        if (layer->activate == PSRelu &&
            layer->derivative != PSReluDerivative)
        {
            PSErr(__func__,
                  "Layer[%d] activate function is PSRelu, "
                  "but derivative function is not PSReluDerivative", i);
            return 0;
        }
        if (layer->activate == PSTanhActivation &&
            layer->derivative != PSTanhDerivative)
        {
            PSErr(__func__,
                  "Layer[%d] activate function is PSTanhActivation, "
                  "but derivative function is not PSTanhDerivative", i);
            return 0;
        }
        previous = layer;
    }
    if (network->flags & FLAG_RECURRENT) {
        PSLayer *output = network->layers[size - 1];
        if (output->type != SoftMax) {
            PSErr(__func__,
                  "Recurrent networks require a Softmax output layer, "
                  "current one is of type %s.",
                  PSGetLabelForType(output->type));
            return 0;
        }
    }
    return 1;
}

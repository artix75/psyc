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
#include <float.h>
#include <assert.h>
#include <signal.h>
#include <sys/time.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "platform.h"
#include "buildinfo.h"
#include "psyc.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "debug.h"

#define STATUS_ERROR_LOSS ((PSFloat) FLT_MIN)
#define BPTT_TRUNCATE   4

#define applyGradientOnBias(opts, grad, val, mg, xg, r, i) \
    applyGradientOnParameter(PARAM_TYPE_BIAS, opts, grad, val, mg, xg, r, i, 0)
#define applyGradientOnWeight(opts, grad, val, mg, xg, r, i, widx) \
    applyGradientOnParameter(PARAM_TYPE_WEIGHT, opts, grad, val, mg, xg, r, \
    i, widx)
#define outputDerivativeNeeded(network) (network->loss != PSCrossEntropyLoss)
#define getNetworkContext(network) ((PSNetworkContext *) network->context)
#define setNetworkContext(network, member, val) (\
    ((PSNetworkContext *) network->context)->member = val)

#ifdef BACKTRACE_AVAILABLE
void segvHandler(int sig, siginfo_t *info, void *secret);
#endif

#define UNUSED(V) ((void) V)

int PSGlobalFlags = 0;

typedef struct {
    PSTrainingOptions options;
    PSGradient **momentum_gradients;
    PSGradient **aux_gradients;
} PSTrainingContext;

typedef struct {
    int                 built;
    PSLayer             *first_recurrent_layer;
    PSLayer             *last_recurrent_layer;
    PSTrainingContext   *training_context;
} PSNetworkContext;
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
float validate(PSNeuralNetwork *network, PSFloat *test_data, int data_size,
               int log);
PSFloat applyDropout(PSLayer *layer, PSFloat value, uint8_t *dropped_p);
uint8_t isDroppedOut(PSNeuron *neuron, ...);
int PSInitConvolutionalLayer(PSNeuralNetwork *network, PSLayer *layer,
                             PSHyperParameters *parameters);
int PSInitPoolingLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSHyperParameters *parameters);
int fullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *layer_gradients, ...);
int PSInitRecurrentLayer(PSNeuralNetwork *network, PSLayer *layer,
                         int size,int ws);
int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws);
int PSInitLSTMStates(PSLayer *layer, uint32_t steps, int retain_previous);
int PSResizeLSTMStates(PSLayer *layer, uint32_t steps);
void PSDeleteLSTMStates(PSLSTMStates *states);
int PSDumpGradients(PSNeuralNetwork *network, PSGradient **gradients,
                    const char* filename, PSTrainingOptions *opts);
PSGradient **cloneGradients(PSGradient **gradients, PSNeuralNetwork *network);
static void deleteTrainingContext(PSTrainingContext *training_ctx,
                                  PSNeuralNetwork *network);
static void deleteNetworkContext(PSNetworkContext *ctx,
                                 PSNeuralNetwork *network);
int writeSerializedFloat(FILE *out, PSFloat fnum, int opts);

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
    int is_recurrent = PSIsRecurrent(layer), tsteps = 0, t = 0;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        tsteps = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
        UNUSED(tsteps);
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat sum = 0.0;
        j = 0;
#ifdef USE_AVX
        if (!avx_disabled) {
            AVXIterativeDotProduct(
                previous_size, previous->activations,
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
            PSFloat a = PSGetActivation(previous, j, t);
            sum += (a * neuron->weights[j]);
        }
#ifdef PS_DEBUG_MODE
        PSAddContextualDebug(network, layer, neuron, NULL, "Sum", sum);
#endif
        neuron->z_value = sum;
        if (use_bias) neuron->z_value += neuron->bias;
        PSFloat activation = layer->activate(neuron->z_value);
        if (!PSSetActivation(layer, activation, i, t)) return 0;
#ifdef PS_DEBUG_MODE
        PSResetDebugInfo();
#endif
    }
    return 1;
}

static int softmaxFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
    int size = layer->size;
    if (layer->neurons == NULL || layer->size <= 0) {
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
    int is_recurrent = PSIsRecurrent(layer), tsteps = 0, t = 0;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#endif
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        tsteps = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
        UNUSED(tsteps);
    }
    PSFloat max = 0.0, esum = 0.0;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat sum = 0;
        j = 0;
#ifdef USE_AVX
        if (!avx_disabled) {
            AVXIterativeDotProduct(
                previous_size, previous->activations,
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
            PSFloat a = PSGetActivation(previous, j, t);
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum;
        if (use_bias) neuron->z_value += neuron->bias;
        if (i == 0)
            max = neuron->z_value;
        else if (neuron->z_value > max)
            max = neuron->z_value;
    }
    PSFloat exponentials[size];
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat z = neuron->z_value;
        PSFloat e = PSExp(z - max);
        esum += e;
        exponentials[i] = e;
    }
    for (i = 0; i < size; i++) {
        PSFloat activation = exponentials[i] / esum;
        if (!PSSetActivation(layer, activation, i, t)) return 0;
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

static PSFloat **getRecurrentSeries(PSNeuralNetwork *network, PSFloat *data,
                                    int series_count, int xsize, int ysize)
{
    int recurrent_input = PSIsRecurrent(network->layers[0]),
        recurrent_output = PSIsRecurrent(network->layers[network->size-1]), i;
    if (!recurrent_input && !recurrent_output) {
        PSErr(NULL, "Netowork has no recurrent input nor recurrent output");
        return NULL;
    }
    PSFloat **series = malloc(sizeof(PSFloat *) * series_count);
    if (series == NULL) {
        PSErr(NULL, "Could not allocate memory for recurrent series!");
        return NULL;
    }
    PSFloat *p = data;
    for (i = 0; i < series_count; i++) {
        PSFloat *size_p = (recurrent_input ? p : p + xsize);
        int series_size = (int) *size_p;
        if (series_size <= 0) {
            PSErr(
                NULL, "Sequence[%d] Invalid length %d at data offset %d",
                i, series_size, (int) (size_p - data)
            );
            free(series);
            return NULL;
        }
        int xmul = (recurrent_input ? series_size : 1),
            ymul = (recurrent_output ? series_size : 1);
        series[i] = p++;
        p += ((xmul * xsize) + (ymul * ysize));
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

/* Find max activation value and the relative neuron index for layer `layer`,
 * and store them into `max_p` pointer (max activation) and `index_p` pointer
 * (index of neuron having maximum activation value).
 * At least `max_p` or `index_p` must be provided.
 * If layer is recurrent, an extra argument for timestep must be provided
 * as a variadic argument (as int).
 * If timestep is negative, it will be used to read activations in a reverse
 * order (ie. -1 is last timestep, -2 is last timestep - 1, etc.).
 * Timestep must be always in range of processed timesteps (hidden states),
 * otherwise the function will fail.
 * Return value: 1 in case of success, 0 in case of error. */
int PSFindLayerMaxActivation(PSLayer *layer, PSFloat *max_p, int *index_p, ...)
{
    if (max_p == NULL && index_p == NULL) return 0;
    int tstep = 0, i;
    int is_recurrent = PSIsRecurrent(layer);
    if (is_recurrent) {
        va_list args;
        va_start(args, index_p);
        tstep = va_arg(args, int);
        va_end(args);
    }
    PSFloat max = 0.0;
    int max_idx = -1;
    for (i = 0; i < layer->size; i++) {
        PSFloat activation = PSGetActivation(layer, i, tstep);
        if (activation > max) {
            max = activation;
            max_idx = i;
        }
    }
    if (max_idx < 0) return 0;
    if (max_p != NULL) *max_p = max;
    if (index_p != NULL) *index_p = max_idx;
    return 1;
}

static int fetchRecurrentOutputState(PSLayer *out, PSFloat *outputs,
                                     int i, int onehot)
{
    int t = (onehot ? i : i / out->size), j;
    int max_idx = 0, oidx = 0;
    PSFloat max = 0.0;
    for (j = 0; j < out->size; j++) {
        PSFloat s = PSGetActivation(out, j, t);
        if (onehot) {
            if (s > max) {
                max = s;
                max_idx = j;
            }
        } else {
            oidx = (t * out->size) + j;
            outputs[oidx] = s;
        }
    }
    if (onehot) {
        outputs[i] = max_idx;
        oidx = i;
    }
    return oidx;
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
    else if (function == PSCrossEntropyLoss) return "cross-entropy";
    return "UNKOWN";
}

int getLossFunctionIndex(PSLossFunction function) {
    if (function == NULL) return 0;
    int i;
    for (i = 1; i < (int) loss_functions_count; i++) {
        PSLossFunction func = loss_functions[i];
        if (func == function) return i;
    }
    return 0;
}

PSLossFunction getLossFunctionAtIndex(int index) {
    if (index >= (int) loss_functions_count) return NULL;
    return loss_functions[index];
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
    case STATUS_VALIDATING:
        return "validating";
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

int PSGetLayerParametersCount(PSLayer *layer) {
    if (Pooling == layer->type || layer->index == 0 || layer->size == 0)
        return 0;
    if (Convolutional == layer->type) {
        PSSharedParams *shared = PSGetConvSharedParams(layer);
        if (shared == NULL) return 0;
        return (shared->weights_size * shared->feature_count) +
               shared->feature_count;
    } else {
        PSNeuron *neuron = layer->neurons[0];
        if (neuron == NULL) return 0;
        return (neuron->weights_size * layer->size) + layer->size;
    }
}

int PSGetNetworkParametersCount(PSNeuralNetwork *network) {
    int tot = 0, i;
    if (network->layers == NULL || network->size == 0) return 0;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        tot += PSGetLayerParametersCount(layer);
    }
    return tot;
}

void PSPrintLayerInfo(PSLayer *layer) {
    if (layer == NULL) return;
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    PSHyperParameters *lparams = layer->hyper_parameters;
    char onehot_info[50];
    onehot_info[0] = 0;
    int onehot_input = (layer->index == 0 && layer->flags & FLAG_ONEHOT);
    if (onehot_input) {
        PSHyperParameters *params = layer->hyper_parameters;
        int onehot_sz = (int) (params->parameters[0]);
        sprintf(onehot_info, " (vector size: %d)", onehot_sz);
    }
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if (layer->dropout > 0.0) printf(", dropout = %g", layer->dropout);
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
    } else if (lparams != NULL && ltype == FullyConnected && !onehot_input) {
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
        printf("  ");
        PSPrintLayerInfo(layer);
    }
    int is_recurrent = PSIsRecurrent(network);
    PSRecurrentNetworkMode mode = PSGetRecurrentNetworkMode(network);
    PSNetworkContext *ctx = network->context;
    assert(ctx != NULL);
    PSLayer *first_recurrent_layer = NULL, *last_recurrent_layer = NULL;
    if (is_recurrent || mode != NonRecurrent) {
        first_recurrent_layer = PSGetFirstRecurrentLayer(network);
        last_recurrent_layer = PSGetLastRecurrentLayer(network);
        printf(
            "Recurrent Network Mode: %s\n",
            PSGetRecurrentModeLabel(mode)
        );
        if (ManyToOne == mode) {
            if (last_recurrent_layer != NULL) {
                printf(
                    "Last Recurrent Layer: %d\n", last_recurrent_layer->index
                );
            }
        } else if (OneToMany == mode) {
            if (first_recurrent_layer != NULL) {
                printf(
                    "First Recurrent Layer: %d\n", first_recurrent_layer->index
                );
            }
        }
    }
    printf("Total (trainable) parameters: %d\n",
        PSGetNetworkParametersCount(network));
    char *loss_name = getLossFunctionName(network->loss);
    if (loss_name != NULL) printf("Loss Function: %s\n", loss_name);
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
    return loss;
}

/* Neural Network Functions */

PSFloat *initRecurrentStates(PSLayer *layer, uint32_t steps,
                            int retain_previous,
                            PSFloat *current, PSFloat **previous)
{
    assert(previous != NULL);
    int len = steps + (retain_previous ? 1 : 0);
    PSFloat *states = calloc(len * layer->size, sizeof(PSFloat));
    if (states == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    if (layer->recurrent_states_count <= 0) retain_previous = 0;
    if (retain_previous && current != NULL) {
        int last_step = layer->recurrent_states_count - 1;
        PSFloat *last = current + (last_step * layer->size);
        PSFloat *prev = states + ((len - 1) * layer->size);
        memcpy(prev, last, ((size_t) layer->size) * sizeof(PSFloat));
        *previous = prev;
    } else *previous = NULL;
    return states;
}

PSFloat *resizeRecurrentStates(PSLayer *layer, uint32_t steps,
                               PSFloat *current, PSFloat **previous)
{
    assert(previous != NULL);
    int len = steps;
    if (layer->previous_activations != NULL) len += 1;
    size_t size = (size_t) (len * layer->size) * sizeof(PSFloat);
    PSFloat *states = realloc(current, size);
    if (states == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSFloat *last = states + (layer->recurrent_states_count * layer->size);
    if (*previous != NULL) {
        PSFloat *prev = states + (steps * layer->size);
        memcpy(prev, last, ((size_t) layer->size) * sizeof(PSFloat));
        *previous = prev;
    }
    int diff = steps - layer->recurrent_states_count;
    memset(last, 0, (size_t) diff * sizeof(PSFloat));
    return states;
}

int PSInitRecurrentHiddenStates(PSLayer *layer, uint32_t steps,
                                int retain_previous)
{
    PSFloat *activations = layer->activations;
    PSFloat *states = NULL;
    if (layer->recurrent_states_count <= 0 || activations == NULL)
        retain_previous = 0;
    if (steps == 0 && !retain_previous) {
        layer->recurrent_states_count = 0;
        layer->activations = NULL;
        free(activations);
        if (layer->dropped_out != NULL) free(layer->dropped_out);
        layer->dropped_out = NULL;
        layer->previous_activations = NULL;
        if (LSTM == layer->type) {
            if (!PSInitLSTMStates(layer, 0, 0)) goto err;
        }
        return 1;
    }
    states = initRecurrentStates(
        layer, steps, retain_previous, activations,
        &layer->previous_activations
    );
    if (states == NULL) return 0;
    if (PSShouldApplyDropout(layer) && steps > 0) {
        if (layer->dropped_out != NULL) free(layer->dropped_out);
        layer->dropped_out = calloc(layer->size * steps, sizeof(uint8_t));
        if (layer->dropped_out == NULL) {
            PSPrintMemoryErrorMsg();
            goto err;
        }
    } else if (layer->dropped_out != NULL) {
        free(layer->dropped_out);
        layer->dropped_out = NULL;
    }
    layer->activations = states;
    layer->recurrent_states_count = steps;
    free(activations);
    if (LSTM == layer->type) {
        if (!PSInitLSTMStates(layer, steps, retain_previous))
            goto err;
    }
    return 1;
err:
    if (states != NULL) free(states);
    if (layer->activations != NULL) free(layer->activations);
    layer->recurrent_states_count = 0;
    layer->activations = NULL;
    if (layer->dropped_out != NULL) free(layer->dropped_out);
    layer->dropped_out = NULL;
    layer->network->status = STATUS_ERROR;
    return 0;
}

int PSResizeRecurrentHiddenStates(PSLayer *layer, uint32_t steps) {
    PSFloat *activations = layer->activations;
    if (activations == NULL || steps == 0 || layer->recurrent_states_count == 0)
        return PSInitRecurrentHiddenStates(layer, steps, 1);
    else if (steps == layer->recurrent_states_count) return 1;
    else if (steps < layer->recurrent_states_count) {
        PSErr(
            NULL, "Layer[%d]: Cannot resize hidden state to a smaller size: %d"
            " (current recurrent_states_count = %d )",
            layer->index, steps, layer->recurrent_states_count
        );
        return 0;
    }
    PSFloat *states = resizeRecurrentStates(
        layer, steps, activations, &layer->previous_activations
    );
    if (states == NULL) {
        free(activations);
        layer->activations = NULL;
        layer->recurrent_states_count = 0;
        layer->previous_activations = NULL;
        if (layer->dropped_out != NULL) free(layer->dropped_out);
        layer->dropped_out = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    layer->recurrent_states_count = steps;
    layer->activations = states;
    if (PSShouldApplyDropout(layer)) {
        uint8_t *dropped_out = realloc(layer->dropped_out, steps);
        if (dropped_out == NULL) {
            free(layer->dropped_out);
            layer->dropped_out = NULL;
            PSPrintMemoryErrorMsg();
            layer->network->status = STATUS_ERROR;
            return 0;
        }
        int diff = steps - layer->recurrent_states_count;
        uint8_t *new_segment =
            dropped_out + (layer->recurrent_states_count * layer->size);
        memset(new_segment, 0, (size_t) diff);
        layer->dropped_out = dropped_out;
    }
    if (LSTM == layer->type) {
        if (!PSResizeLSTMStates(layer, steps)) return 0;
    }
    return 1;
}

int PSResetLayerRecurrentStates(PSLayer *layer, uint32_t steps,
                                int retain_previous)
{
    if (layer == NULL) return 0;
    if (!PSIsRecurrent(layer)) return 0;
    return PSInitRecurrentHiddenStates(layer, steps, retain_previous);
}

int PSResetNetworkRecurrentStates(PSNeuralNetwork *network, uint32_t steps,
                                int retain_previous)
{
    if (network == NULL) return 0;
    if (!PSIsNetworkBuilt(network)) {
        PSErr(__func__, "Network is not build");
        return 0;
    }
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) continue;
        if (!PSResetLayerRecurrentStates(layer, steps, retain_previous)) {
            PSErr(__func__, "Failed to reset recurrent states on layer %d", i);
            return 0;
        }
    }
    return 1;
}

int PSSetActivation(PSLayer *layer, PSFloat activation, int index, ...) {
    if (layer->activations == NULL) {
        PSErr(__func__, "Layer %d: null activations!", layer->index);
        return 0;
    }
    if (index >= layer->size) {
        PSErr(
            __func__, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0;
    }
    int apply_dropout = PSShouldApplyDropout(layer);
    int t = 0;
    if (PSIsRecurrent(layer)) {
        va_list args;
        va_start(args, index);
        t = va_arg(args, int);
        va_end(args);
        if (t >= (int) layer->recurrent_states_count) {
            if (!PSResizeRecurrentHiddenStates(layer, t + 1)) {
                if (layer->network) layer->network->status = STATUS_ERROR;
                PSErr(
                    __func__, "Could not resize recurrent hidden states for "
                    "layer %d", layer->index
                );
                return 0;
            } else if (layer->activations == NULL) return 0;
        } else if (t < 0) {
            PSErr(
                __func__,
                "Invalid recurrent step %d for layer %d, neuron %d",
                layer->index, index
            );
            return 0;
        }
        index = (t * layer->size) + index;
    } else if (apply_dropout && layer->dropped_out == NULL) {
        layer->dropped_out = calloc(layer->size, sizeof(uint8_t));
        if (layer->dropped_out == NULL) {
            PSPrintMemoryErrorMsg();
            if (layer->network) layer->network->status = STATUS_ERROR;
            return 0;
        }
    }
    uint8_t dropped = 0;
    if (apply_dropout) {
        assert(layer->dropped_out != NULL);
        activation = applyDropout(layer, activation, &dropped);
        layer->dropped_out[index] = dropped;
    }
    layer->activations[index] = activation;
    return 1;
}

int PSSetNeuronActivation(PSNeuron *neuron, double activation, ...) {
    if (neuron->layer == NULL) return 0.0;
    PSFloat a = (PSFloat) activation;
    if (PSIsRecurrent(neuron->layer)) {
        va_list args;
        va_start(args, activation);
        int t = va_arg(args, int);
        va_end(args);
        return PSSetActivation(neuron->layer, a, neuron->index, t);
    }
    return PSSetActivation(neuron->layer, a, neuron->index);
}

PSFloat PSGetActivation(PSLayer *layer, int index, ...) {
    if (layer->activations == NULL) return 0.0;
    if (index >= layer->size) {
        PSErr(
            __func__, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0.0;
    }
    if (PSIsRecurrent(layer)) {
        va_list args;
        va_start(args, index);
        int t = va_arg(args, int);
        va_end(args);
        /* If t < 0, retrieve previous activation, if any. */
        if (t < 0) {
            if (layer->previous_activations == NULL) return 0.0;
            else return layer->previous_activations[index];
        } else {
            if (t >= (int) layer->recurrent_states_count) {
                PSErr(
                    __func__, "State %d is out-of-range: layer %d only "
                    "has %d recurrent hidden states",
                    "of size %d", t, layer->index,
                    layer->recurrent_states_count
                );
                return 0.0;
            }
            index = (t * layer->size) + index;
        }
    }
    return layer->activations[index];
}

PSFloat PSGetNeuronActivation(PSNeuron *neuron, ...) {
    if (neuron->layer == NULL) return 0.0;
    if (PSIsRecurrent(neuron->layer)) {
        va_list args;
        va_start(args, neuron);
        int t = va_arg(args, int);
        va_end(args);
        return PSGetActivation(neuron->layer, neuron->index, t);
    }
    return PSGetActivation(neuron->layer, neuron->index);
}

PSLayer *PSGetFirstRecurrentLayer(PSNeuralNetwork *network) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) return NULL;
    return ctx->first_recurrent_layer;
}

PSLayer *PSGetLastRecurrentLayer(PSNeuralNetwork *network) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) return NULL;
    return ctx->last_recurrent_layer;
}

int PSIsNetworkBuilt(PSNeuralNetwork *network) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) return 0;
    return ctx->built;
}

static PSRecurrentNetworkOptions *createDefaultRNNOptions(PSNeuralNetwork *net)
{
    if (net->rnn_options != NULL) return net->rnn_options;
    net->rnn_options = calloc(1, sizeof(PSRecurrentNetworkOptions));
    if (net->rnn_options == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSSetDefaultRNNOptions(net->rnn_options);
    return net->rnn_options;
}

void PSSetDefaultRNNOptions(PSRecurrentNetworkOptions *opts) {
    opts->sequence_stop_criterion.max_steps = MAX_RECURRENT_OUTPUT_STEPS;
    opts->sequence_stop_criterion.eos = -1;
}

PSFloat applyDropout(PSLayer *layer, PSFloat value, uint8_t *dropped_p) {
    if (layer->dropout <= 0) return value;
    PSNeuralNetwork *network = layer->network;
    if (layer->dropout > 1.0) layer->dropout = 1.0;
    if (network->status == STATUS_TRAINING) {
        assert(dropped_p != NULL);
        PSFloat r = PSNormalizedRandom();
        if (r < layer->dropout) {
            *dropped_p = 1;
            return 0.0;
        } else {
            *dropped_p = 0;
            return value;
        }
    } else return value * layer->dropout;
}

uint8_t isDroppedOut(PSNeuron *neuron, ...) {
    if (neuron->layer->dropped_out == NULL) return 0;
    int index = neuron->index;
    if (!PSIsRecurrent(neuron->layer)) {
        int t = 0;
        va_list ap;
        va_start(ap, neuron);
        t = va_arg(ap, int);
        va_end(ap);
        index += (neuron->layer->size * t);
    }
    return neuron->layer->dropped_out[index];
}

static void updateNetworkForRecurrentMode(PSNeuralNetwork *network,
                                          PSRecurrentNetworkMode mode)
{
    PSNetworkContext *ctx = getNetworkContext(network);
    assert(ctx != NULL);
    if (mode == ManyToOne || mode == OneToMany) {
        ctx->first_recurrent_layer = NULL;
        ctx->last_recurrent_layer = NULL;
    }
    int recurrent_input = (mode == ManyToMany || mode == ManyToOne),
        recurrent_output = (mode == ManyToMany || mode == OneToMany),
        last_layer_idx = (network->size - 1), i, j;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType type = layer->type;
        if (i == 0 && recurrent_input) {
            layer->flags |= FLAG_RECURRENT;
            ctx->first_recurrent_layer = layer;
        } else if (i == last_layer_idx && recurrent_output) {
            layer->flags |= FLAG_RECURRENT;
            ctx->last_recurrent_layer = layer;
        } else if (ManyToMany == mode) {
            layer->flags |= FLAG_RECURRENT;
        } else if (ManyToOne == mode) {
            if (i < last_layer_idx) {
                if (PSIsRecurrent(layer)) {
                    ctx->last_recurrent_layer = layer;
                    for (j = 1; j < layer->index; j++)
                        network->layers[j]->flags |= FLAG_RECURRENT;
                }
            } else if (Recurrent != type && LSTM != type) {
                network->layers[i]->flags &= (unsigned) (~FLAG_RECURRENT);
            }
        } else if (OneToMany == mode) {
            if (PSIsRecurrent(layer)) ctx->first_recurrent_layer = layer;
            else if (ctx->first_recurrent_layer != NULL) {
                if (layer->index > ctx->first_recurrent_layer->index)
                    layer->flags |= FLAG_RECURRENT;
            }
        }
    }
}

int PSBuildNetwork(PSNeuralNetwork *network) {
    if (network == NULL) {
        PSErr(__func__, "Network is null!");
        return 0;
    }
    if (network->size == 0) {
        PSErr(__func__, "Network is empty!");
        return 0;
    }
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) {
        ctx = network->context = calloc(1, sizeof(PSNetworkContext));
        if (ctx == NULL) {
            PSPrintMemoryErrorMsg();
            PSErr(__func__, "Cannot build network");
            return 0;
        }
    } else if (ctx->built) return 1;
    int is_recurrent = PSIsRecurrent(network);
    PSRecurrentNetworkMode mode = PSGetRecurrentNetworkMode(network);
    PSLayer *input_layer = network->layers[0],
            *output_layer = network->layers[network->size - 1];
    if (is_recurrent) {
        if (mode != NonRecurrent) updateNetworkForRecurrentMode(network, mode);
        else {
            int recurrent_input = PSIsRecurrent(input_layer),
                recurrent_output = PSIsRecurrent(output_layer);
            if (recurrent_input && recurrent_output)
                mode = ManyToMany;
            else if (recurrent_input && !recurrent_output)
                mode = ManyToOne;
            else if (!recurrent_input && recurrent_output)
                mode = OneToMany;
            else {
                PSErr(
                    __func__, "Recurrent network must have at least recurrent "
                    "input or output"
                );
                return 0;
            }
            updateNetworkForRecurrentMode(network, mode);
        }
    }
    if (network->loss == NULL) {
        if (output_layer->type == SoftMax) network->loss = PSCrossEntropyLoss;
        else network->loss = PSQuadraticLoss;
    }
    ctx->built = 1;
    return 1;
}

int PSRebuildNetwork(PSNeuralNetwork *network) {
    if (network == NULL) {
        PSErr(__func__, "Network is null!");
        return 0;
    }
    if (PSIsNetworkBuilt(network)) {
        PSNetworkContext *ctx = getNetworkContext(network);
        ctx->built = 0;
    }
    return PSBuildNetwork(network);
}

char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode) {
    switch (mode) {
        case NonRecurrent: return "Non-Recurrent";
        case ManyToMany: return "Many-to-Many";
        case ManyToOne: return "Many-to-One";
        case OneToMany: return "One-to-Many";
    }
    return "UNKOWN";
}

PSRecurrentNetworkMode PSGetRecurrentNetworkMode(PSNeuralNetwork *network) {
    PSRecurrentNetworkOptions *opts = network->rnn_options;
    if (opts == NULL) return NonRecurrent;
    return opts->mode;
}

int PSSetRecurrentNetworkMode(PSNeuralNetwork *network,
                              PSRecurrentNetworkMode mode)
{
    if (network == NULL) {
        PSErr(__func__, "Network is null!");
        return 0;
    }
    if (network->size == 0) goto final;
    if (mode == NonRecurrent && PSIsRecurrent(network)) {
        PSErr(
            __func__, "Invalid mode NonRecurrent: network is already "
            "recurrent"
        );
        return 0;
    }
    if (network->rnn_options == NULL)
        if (createDefaultRNNOptions(network) == NULL) return 0;
    updateNetworkForRecurrentMode(network, mode);
final:
    if (network->rnn_options == NULL)
        if (createDefaultRNNOptions(network) == NULL) return 0;
    network->rnn_options->mode = mode;
    if (mode != NonRecurrent) network->flags |= FLAG_RECURRENT;
    return 1;
}

PSNeuralNetwork *PSCreateNetwork(const char* name) {
    PSNeuralNetwork *network = (malloc(sizeof(PSNeuralNetwork)));
    if (network == NULL) {
        return NULL;
    }
    network->context = calloc(1, sizeof(PSNetworkContext));
    if (network->context == NULL) goto memory_err;
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
    network->rnn_options = NULL;
    return network;
memory_err:
    if (network != NULL) PSDeleteNetwork(network);
    PSErr(__func__, "Could not allocate memory for Network!");
    return NULL;
}

PSNeuralNetwork *PSCloneNetwork(PSNeuralNetwork *network, int layout_only) {
    if (network == NULL) return NULL;
    PSNeuralNetwork *clone = PSCreateNetwork(NULL);
    if (clone == NULL) return NULL;
    if (!layout_only) {
        clone->status = network->status;
        if (network->training != NULL) {
            clone->training = malloc(sizeof(PSTrainingInfo));
            if (clone->training == NULL) goto err;
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
    if (network->rnn_options != NULL) {
        clone->rnn_options = malloc(sizeof(PSRecurrentNetworkOptions));
        if (clone->rnn_options == NULL) goto err;
        memcpy(
            clone->rnn_options, network->rnn_options,
            sizeof(PSRecurrentNetworkOptions)
        );
    }

    int i, j, k, w;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType type = layer->type;
        PSHyperParameters *oparams = layer->hyper_parameters;
        PSHyperParameters *cparams = NULL;
        if (oparams) {
            cparams = malloc(sizeof(PSHyperParameters));
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
                free(cparams);
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
        cloned_layer->dropout = layer->dropout;
        if (!layout_only) {
            cloned_layer->recurrent_states_count =
                layer->recurrent_states_count;
            if (cloned_layer->activations != NULL) {
                free(cloned_layer->activations);
                cloned_layer->activations = NULL;
            }
            if (cloned_layer->dropped_out != NULL) {
                free(cloned_layer->dropped_out);
                cloned_layer->dropped_out = NULL;
            }
            if (layer->activations != NULL) {
                int len;
                if (PSIsRecurrent(layer)) {
                    len = layer->recurrent_states_count;
                    if (layer->previous_activations != NULL) len += 1;
                } else len = 1;
                if (len < 1) len = 1;
                len *= layer->size;
                size_t size = (size_t) len * sizeof(PSFloat);
                cloned_layer->activations = malloc(size);
                if (cloned_layer->activations == NULL) {
                    PSPrintMemoryErrorMsg();
                    PSDeleteNetwork(clone);
                    return NULL;
                }
                memcpy(cloned_layer->activations, layer->activations, size);
                if (layer->previous_activations != NULL) {
                    int diff = layer->previous_activations -
                               layer->activations;
                    cloned_layer->previous_activations =
                        cloned_layer->previous_activations + diff;
                }
            } else {
                cloned_layer->activations = NULL;
                cloned_layer->previous_activations = NULL;
            }
            if (layer->dropped_out != NULL) {
                int len;
                if (PSIsRecurrent(layer)) len = layer->recurrent_states_count;
                else len = 1;
                if (len < 1) len = 1;
                len *= layer->size;
                size_t size = (size_t) len * sizeof(PSFloat);
                cloned_layer->dropped_out = malloc(size);
                if (cloned_layer->dropped_out == NULL) {
                    PSPrintMemoryErrorMsg();
                    PSDeleteNetwork(clone);
                    return NULL;
                }
                memcpy(cloned_layer->dropped_out, layer->dropped_out, size);
            } else cloned_layer->dropped_out = NULL;
            void *extra = layer->extra;
            if (Convolutional == type && extra != NULL) {
                /* Copy Convolutional shared parameters. */
                PSSharedParams *oshared = PSGetConvSharedParams(layer);
                PSSharedParams *cshared = PSGetConvSharedParams(cloned_layer);
                cshared->feature_count = oshared->feature_count;
                cshared->weights_size = oshared->weights_size;
                for (k = 0; k < cshared->feature_count; k++) {
                    cshared->biases[k] = oshared->biases[k];
                    for (w = 0; w < cshared->weights_size; w++)
                        cshared->weights[k][w] = oshared->weights[k][w];
                }
            } else if (LSTM == type && extra != NULL) {
                /* Copy Convolutional shared parameters. */
                PSLSTMStates *ostates = (PSLSTMStates *) extra;
                PSLSTMStates *cstates = calloc(1, sizeof(PSLSTMStates));
                if (cstates == NULL) {
                    PSPrintMemoryErrorMsg();
                    PSDeleteNetwork(clone);
                    return NULL;
                }
                int len = layer->recurrent_states_count,
                    has_prev = (layer->previous_activations != NULL);
                if (has_prev) len += 1;
                len *= layer->size;
                PSFloat *ostate_ptrs[] = {
                    ostates->candidates, ostates->previous_candidates,
                    ostates->input_gates, ostates->previous_input_gates,
                    ostates->output_gates, ostates->previous_output_gates,
                    ostates->forget_gates, ostates->previous_forget_gates,
                    ostates->z_values, ostates->previous_z_values
                };
                PSFloat **cstate_ptrs[] = {
                    &cstates->candidates, &cstates->previous_candidates,
                    &cstates->input_gates, &cstates->previous_input_gates,
                    &cstates->output_gates, &cstates->previous_output_gates,
                    &cstates->forget_gates, &cstates->previous_forget_gates,
                    &cstates->z_values, &cstates->previous_z_values
                };
                size_t size = len * sizeof(PSFloat),
                       ptr_len = (sizeof(ostate_ptrs) / sizeof(PSFloat*));
                for (j = 0; j < (int) ptr_len; j += 2) {
                    if (len == 0) break;
                    PSFloat *o_ptr = ostate_ptrs[j];
                    PSFloat *o_prev_ptr = ostate_ptrs[j + 1];
                    if (o_ptr == NULL) continue;
                    if (has_prev && o_prev_ptr == NULL) {
                        PSErr(
                            __func__, "LSTM Layer %d has no previous "
                            "recurrent LSTM states altough layer has "
                            "previous activation.", i
                        );
                        free(cstates);
                        PSDeleteNetwork(clone);
                        return NULL;
                    }
                    PSFloat *s = malloc(size);
                    if (s == NULL) {
                        PSPrintMemoryErrorMsg();
                        free(cstates);
                        PSDeleteNetwork(clone);
                        return NULL;
                    }
                    memcpy(s, o_ptr, size);
                    PSFloat **c_ptr = cstate_ptrs[j];
                    PSFloat **c_prev_ptr = cstate_ptrs[j + 1];
                    *c_ptr = s;
                    if (has_prev) *c_prev_ptr = s + (o_prev_ptr - o_ptr);
                }
                cloned_layer->extra = cstates;
            }
            for (j = 0; j < layer->size; j++) {
                PSNeuron *orig_n = layer->neurons[j];
                PSNeuron *clone_n = cloned_layer->neurons[j];
                clone_n->z_value = orig_n->z_value;
                /* if (Pooling == type) continue; */
                clone_n->bias = orig_n->bias;

                if (Convolutional != type && Pooling != type) {
                    PSFloat *oweights = orig_n->weights;
                    PSFloat *cweights = clone_n->weights;
                    for (w = 0; w < orig_n->weights_size; w++)
                        cweights[w] = oweights[w];
                }
                if (PSIsRecurrent(layer)) {
                    PSRecurrentCell *ocell = PSGetRecurrentCell(orig_n);
                    if (ocell != NULL) {
                        PSRecurrentCell *ccell = PSGetRecurrentCell(clone_n);
                        if (ccell == NULL) {
                            if (LSTM == type || Recurrent == type) {
                                PSDeleteNetwork(clone);
                                return  NULL;
                            }
                            ccell = PSCreateRecurrentCell(clone_n, 0);
                            if (ccell == NULL) {
                                PSDeleteNetwork(clone);
                                PSPrintMemoryErrorMsg();
                                return NULL;
                            }
                        }
                    }
                }
                if (layer->type == LSTM) {
                    PSLSTMCell *ocell = PSGetLSTMCell(orig_n);
                    PSLSTMCell *ccell = PSGetLSTMCell(clone_n);
                    if (ocell != NULL && ccell != NULL) {
                        ccell->candidate_bias = ocell->candidate_bias;
                        ccell->input_bias = ocell->input_bias;
                        ccell->output_bias = ocell->output_bias;
                        ccell->forget_bias = ocell->forget_bias;
                    }
                }
            }
        }
    }
    if (clone->layers == NULL) {
        if (network->layers == NULL) return clone;
        else goto err;
    }
    if (network->context != NULL) {
        memcpy(clone->context, network->context, sizeof(PSNetworkContext));
        PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network);
        PSLayer *last_recurrent = PSGetLastRecurrentLayer(network);
        if (first_recurrent != NULL) {
            setNetworkContext(
                clone, first_recurrent_layer,
                clone->layers[first_recurrent->index]
            );
        } else setNetworkContext(clone, first_recurrent_layer, NULL);
        if (last_recurrent != NULL) {
            setNetworkContext(
                clone, last_recurrent_layer,
                clone->layers[last_recurrent->index]
            );
        } else setNetworkContext(clone, last_recurrent_layer, NULL);
        PSNetworkContext *ctx = network->context, *clone_ctx = clone->context;
        PSTrainingContext *training_ctx = ctx->training_context;
        if (training_ctx != NULL) {
            clone_ctx->training_context = calloc(1, sizeof(PSTrainingContext));
            if (clone_ctx->training_context == NULL) {
                PSPrintMemoryErrorMsg();
                goto err;
            }
            PSTrainingContext *clone_training_ctx=clone_ctx->training_context;
            clone_training_ctx->options = training_ctx->options;
            clone_training_ctx->options.debug_dump_to = NULL;
            PSGradient **mgradients = training_ctx->momentum_gradients;
            PSGradient **xgradients = training_ctx->aux_gradients;
            if (mgradients != NULL) {
                clone_training_ctx->momentum_gradients = cloneGradients(
                    mgradients, network
                );
                if (clone_training_ctx->momentum_gradients == NULL) goto err;
            } else {
                if (clone_training_ctx->momentum_gradients) {
                    PSDeleteGradients(clone_training_ctx->momentum_gradients,
                        network);
                }
                clone_training_ctx->momentum_gradients = NULL;
            }
            if (xgradients != NULL) {
                clone_training_ctx->aux_gradients = cloneGradients(
                    xgradients, network
                );
                if (clone_training_ctx->aux_gradients == NULL) goto err;
            } else {
                if (clone_training_ctx->aux_gradients) {
                    PSDeleteGradients(clone_training_ctx->aux_gradients,
                        network);
                }
                clone_training_ctx->aux_gradients = NULL;
            }
        } else {
            if (clone_ctx->training_context != NULL)
                deleteTrainingContext(clone_ctx->training_context, clone);
            clone_ctx->training_context = NULL;
        }
    } else {
        if (clone->context != NULL)
            deleteNetworkContext(clone->context, clone);
        clone->context = NULL;
    }
    return clone;
err:
    if (clone != NULL) PSDeleteNetwork(clone);
    return NULL;
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

void DumpLayerInfo(PSLayer *layer, FILE *dump_file, int add_new_line) {
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    PSHyperParameters *lparams = layer->hyper_parameters;
    fprintf(dump_file, "layer:index=%d,type=%s,size=%d", layer->index,
        type_name, layer->size);
    int onehot_input = (layer->index == 0 && layer->flags & FLAG_ONEHOT);
    if (onehot_input) {
        PSHyperParameters *params = layer->hyper_parameters;
        int onehot_sz = (int) (params->parameters[0]);
        fprintf(dump_file, ",vector_size=%d", onehot_sz);
    }
    if (PSIsRecurrent(layer) && PSIsRecurrent(layer->network))
        fprintf(dump_file, ",recurrent=1");
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
    } else if (!onehot_input && lparams != NULL && ltype == FullyConnected) {
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
        PSErr(__func__, "Cannot open %s for writing!", filename);
        return 0;
    }
    int opts = 0;
    DumpNetworkHeader(network, f);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int is_recurrent = PSIsRecurrent(layer), timesteps = 0;
        int nidx = 0, t = 0;
        DumpLayerInfo(layer, f, 0);
        if (is_recurrent) {
            timesteps = layer->recurrent_states_count;
            fprintf(f, ",timesteps=%d", timesteps);
        }
        fprintf(f, ",activations=(");
        for(; nidx < layer->size; nidx++) {
            PSNeuron *n = layer->neurons[nidx];
            if (n == NULL) {
                PSErr(__func__, "Layer[%d] Neuron[%s] is null", i, nidx);
                return 0;
            }
            if (!is_recurrent) {
                if (nidx > 0) fprintf(f, ",");
                writeSerializedFloat(f, PSGetActivation(layer, nidx), opts);
            } else {
                for (t = 0; t < timesteps; t++) {
                    if (nidx > 0 || t > 0) fprintf(f, ",");
                    writeSerializedFloat(
                        f, PSGetActivation(layer, nidx, t), opts
                    );
                }
            }
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
    int opts = 0;
    DumpNetworkHeader(network, f);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int nidx = 0;
        DumpLayerInfo(layer, f, 0);
        PSFloat *delta = layer->delta;
        if (delta == NULL) {
            fprintf(f, ",deltas=()\n");
            continue;
        }
        fprintf(f, ",deltas=(");
        for(; nidx < layer->size; nidx++) {
            if (nidx > 0) fprintf(f, ",");
            PSFloat d = delta[nidx];
            writeSerializedFloat(f, d, opts);
        }
        fprintf(f, ")\n");
    }
    fclose(f);
    return 1;
}

static void deleteTrainingContext(PSTrainingContext *training_ctx,
                                  PSNeuralNetwork *network)
{
    if (training_ctx->momentum_gradients != NULL)
        PSDeleteGradients(training_ctx->momentum_gradients, network);
    if (training_ctx->aux_gradients != NULL)
        PSDeleteGradients(training_ctx->aux_gradients, network);
    free(training_ctx);
}

static void deleteNetworkContext(PSNetworkContext *ctx,
                                 PSNeuralNetwork *network)
{
    PSTrainingContext *training_ctx = ctx->training_context;
    if (training_ctx != NULL) deleteTrainingContext(training_ctx, network);
    free(ctx);
}

void PSDeleteNetwork(PSNeuralNetwork *network) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx != NULL) deleteNetworkContext(network->context, network);
    int size = network->size;
    int i, is_recurrent = (network->flags & FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        PSLayer *layer = NULL;
        if (network->layers != NULL) layer = network->layers[i];
        if (layer == NULL) continue;
        if (is_recurrent) layer->flags |= FLAG_RECURRENT;
        PSDeleteLayer(layer);
    }
    free(network->layers);
    if (network->training != NULL) free(network->training);
    if (network->rnn_options != NULL) free(network->rnn_options);
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
                free(cell);
            }
        } else free(neuron->extra);
    }
    free(neuron);
}

PSLayer *PSAddLayer(PSNeuralNetwork *network, PSLayerType type, int size,
                     PSHyperParameters* params)
{
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
    layer->hyper_parameters = params;
    layer->extra = NULL;
    layer->flags = FLAG_NONE;
    layer->delta = NULL;
    layer->activations = NULL;
    layer->previous_activations = NULL;
    layer->dropout = 0.0;
    layer->dropped_out = NULL;
    layer->recurrent_states_count = 0;
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
            PSHyperParameters *params;
            params = PSCreateHyperParamenters(1, (PSFloat) size);
            layer->hyper_parameters = params;
            size = 1;
            layer->size = 1;
        }
        network->input_size = size;
    } else {
        PSLayer **layers = realloc(network->layers,
                                   sizeof(PSLayer*) * network->size);
        if (layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Could not reallocate network layers!");
            return NULL;
        }
        network->layers = layers;
        previous = network->layers[layer->index - 1];
        if (previous == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Previous layer is NULL!");
            return NULL;
        }
        previous_size = previous->size;
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT) {
            PSHyperParameters *params = previous->hyper_parameters;
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
        layer->activations = calloc(size, sizeof(PSFloat));
        if (layer->activations == NULL) {
            PSPrintMemoryErrorMsg();
            PSAbortLayer(network, layer);
            return NULL;
        }
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
            if (layer->index > 0 && previous_size > 0) {
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
            neuron->z_value = 0;
            neuron->layer = layer;
            layer->neurons[i] = neuron;
        }
        if (type != SoftMax) {
            layer->activate = PSSigmoid;
            layer->derivative = PSSigmoidDerivative;
            layer->feedforward = fullFeedforward;
            layer->backprop = fullBackprop;
        } else {
            layer->activate = NULL;
            layer->derivative = NULL;
            layer->feedforward = softmaxFeedforward;
            layer->backprop = NULL; /* Softmax layer should always be output
                                     * layer. */
            network->loss = PSCrossEntropyLoss;
        }
        initialized = 1;
    } else if (type == Convolutional) {
        initialized = PSInitConvolutionalLayer(network, layer, params);
        /* TODO: Make PSCrossEntropyLoss default also for convolutional? */
    } else if (type == Pooling) {
        initialized = PSInitPoolingLayer(network, layer, params);
    } else if (type == Recurrent) {
        initialized = PSInitRecurrentLayer(network, layer, size, previous_size);
    } else if (type == LSTM) {
        initialized = PSInitLSTMLayer(network, layer, size, previous_size);
    }
    int layer_idx = layer->index;
    if (!initialized) {
        PSAbortLayer(network, layer);
        PSErr(__func__, "Could not initialize layer %d!", layer_idx);
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
                layer_idx
            );
            return NULL;
        }
    }
    network->layers[layer->index] = layer;
    if (PSIsRecurrent(network) || PSIsRecurrent(layer)) {
        int ok = 1;
        PSRecurrentNetworkMode rnn_mode = PSGetRecurrentNetworkMode(network);
        if (rnn_mode == NonRecurrent)
            ok = PSSetRecurrentNetworkMode(network, DEFAULT_RECURRENT_MODE);
        else updateNetworkForRecurrentMode(network, rnn_mode);
        if (!ok) {
            PSAbortLayer(network, layer);
            PSErr(
                __func__, "Could not set default recurrent mode for  layer %d!",
                layer_idx
            );
            return NULL;
        }
    }
    /*PSPrintLayerInfo(layer);*/ /*TODO:Enable it after implementing log-levels*/
    return layer;
}

PSLayer *PSAddConvolutionalLayer(PSNeuralNetwork *network,
                                  PSHyperParameters* params)
{
    return PSAddLayer(network, Convolutional, 0, params);
}

PSLayer *PSAddPoolingLayer(PSNeuralNetwork *network,
                            PSHyperParameters* params)
{
    return PSAddLayer(network, Pooling, 0, params);
}

void PSDeleteLayer(PSLayer* layer) {
    int size = layer->size;
    int i;
    if (layer->neurons == NULL) size = 0;
    for (i = 0; i < size; i++) {
        PSNeuron* neuron = layer->neurons[i];
        if (layer->type != Convolutional) PSDeleteNeuron(neuron, layer);
        else free(neuron);
    }
    if (layer->neurons != NULL) free(layer->neurons);
    PSHyperParameters *params = layer->hyper_parameters;
    if (params != NULL) PSDeleteHyperParamenters(params);
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
        } else if (LSTM == layer->type) {
            PSLSTMStates *states = (PSLSTMStates *) extra;
            PSDeleteLSTMStates(states);
        } else free(extra);
    }
    if (layer->delta != NULL) free(layer->delta);
    if (layer->activations != NULL) free(layer->activations);
    if (layer->dropped_out != NULL) free(layer->dropped_out);
    free(layer);
}

PSHyperParameters *PSCreateHyperParamenters(int count, ...) {
    PSHyperParameters *params = malloc(sizeof(PSHyperParameters));
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

PSHyperParameters *PSCreateConvolutionalParameters(PSFloat feature_count,
                                                    PSFloat region_size,
                                                    int stride,
                                                    int padding,
                                                    int use_relu)
{
    return PSCreateHyperParamenters(CONV_PARAMETER_COUNT, feature_count,
                                    region_size, (PSFloat) stride,
                                    0.0f, 0.0f, 0.0f, 0.0f,
                                    (PSFloat) padding, (PSFloat) use_relu);
}

int PSSetHyperParameter(PSHyperParameters *params, int param, PSFloat value) {
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

int PSAddHyperParameter(PSHyperParameters *params, PSFloat val) {
    return PSSetHyperParameter(params, params->count + 1, val);
}

void PSDeleteHyperParamenters(PSHyperParameters *params) {
    if (params == NULL) return;
    if (params->parameters != NULL) free(params->parameters);
    free(params);
}

int inputLayerFeedforward(PSNeuralNetwork *network, PSFloat *values, ...) {
    PSLayer *first = network->layers[0];
    int input_size = first->size, i, timesteps = 0, t = 0;

    /* TODO: After implementing Recurrent network types
     * (many-to-many,one-to-many, etc.), remove check for
     * `PSIsRecurrent(network)` since check will be made only on
     * layer itself. */
    int is_recurrent = PSIsRecurrent(first);
    if (is_recurrent) {
        va_list ap;
        va_start(ap, values);
        timesteps = va_arg(ap, int);
        t = va_arg(ap, int);
        va_end(ap);
        assert(timesteps > 0);
        assert(t >= 0);
    }
    for (i = 0; i < input_size; i++) {
        PSFloat val = values[i];
        if (!PSSetActivation(first, val, i, t)) {
            network->status = STATUS_ERROR;
            return 0;
        }
    }
    return 1;
}

int feedforwardThroughTime(PSNeuralNetwork *network, PSFloat *values,
                           int timesteps)
{
    if (network == NULL) return 0;
    PSRecurrentNetworkOptions *rnn_options = network->rnn_options;
    PSLayer *first = PSGetFirstRecurrentLayer(network);
    if (first == NULL) first = network->layers[0];
    int input_size = first->size, first_layer_idx = first->index,
        output_idx = network->size - 1, last_layer_idx = output_idx,
        start_idx = first_layer_idx, i, t, ok = 1;
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network);
    PSLayer *last_recurrent = PSGetLastRecurrentLayer(network);
    if (last_recurrent != NULL)
        last_layer_idx = last_recurrent->index;
    while (!PSIsRecurrent(first)) {
        if (first->index >= output_idx) return 0;
        first = network->layers[first->index + 1];
        if (PSIsRecurrent(first) && first_recurrent == NULL)
            setNetworkContext(network, first_recurrent_layer, first);
    }
    int variable_timesteps = 0, eos = -1;
    /* Values can only be NULL if first recurrent layer is not the
     * input layer. */
    if (values == NULL) {
        if (first->index == 0) {
            PSErr(NULL, "Recurrent network with sequence input cannot have"
                  " receive NULL values");
            return 0;
        }
        if ((variable_timesteps = (timesteps == 0))) {
            if (rnn_options != NULL) {
                timesteps = rnn_options->sequence_stop_criterion.max_steps;
                eos = rnn_options->sequence_stop_criterion.eos;
            }
            if (timesteps <= 0) timesteps = MAX_RECURRENT_OUTPUT_STEPS;
        }
    }
    PSLayer *last_layer = NULL;
    int recurrent_input = (first_layer_idx == 0);
    if (recurrent_input) start_idx++;
    for (t = 0; t < timesteps; t++) {
        if (recurrent_input) {
            ok = inputLayerFeedforward(network, values, timesteps, t);
            if (!ok) return 0;
        }
        for (i = start_idx; i <= last_layer_idx; i++) {
            PSLayer *layer = network->layers[i];
            if (layer == NULL) {
                PSErr(__func__, "Layer %d is NULL", i);
                return 0;
            }
            if (layer->feedforward == NULL) {
                PSErr(__func__, "Layer %d feedforward function is NULL", i);
                return 0;
            }
            if (!PSIsRecurrent(layer)) break;
            else last_layer = layer;
            ok = layer->feedforward(network, layer, timesteps, t);
            if (!ok) return 0;
        }
        if (last_recurrent == NULL && last_layer != NULL)
            setNetworkContext(network, last_recurrent_layer, last_layer);
        if (values != NULL) values += input_size;
        if (variable_timesteps && eos >= 0 && last_layer != NULL) {
            int max_idx = -1;
            if (!PSFindLayerMaxActivation(last_layer, NULL, &max_idx, t))
                return 0;
            if (max_idx == eos) break;
        }
    }
    return 1;
}

int feedforward(PSNeuralNetwork *network, PSFloat *values, int backprop, ...) {
    if (network == NULL) return 0;
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    if (!PSIsNetworkBuilt(network)) {
        PSErr(__func__, "Network is not built!");
        return 0;
    }
    if (values == NULL) {
        PSErr(__func__, "Null values");
        return 0;
    }
    int is_recurrent = PSIsRecurrent(network), recurrent_input = 0, i,
        ok = 1, timesteps = 0, first_idx = 0, output_idx = network->size - 1;
    PSLayer *input_layer = network->layers[0],
            *output_layer = network->layers[output_idx],
            *first_recurrent = NULL,
            *last_recurrent = NULL;
    if (is_recurrent) {
        PSTrainingOptions *opts = NULL;
        int retain_previous = 1;
        if (backprop) {
            va_list args;
            va_start(args, backprop);
            opts = va_arg(args, PSTrainingOptions*);
            va_end(args);
            retain_previous = (
                opts != NULL && (opts->flags & TRAINING_EPOCH_AS_SEQUENCE)
            );
            if (retain_previous && network->training != NULL)
                retain_previous = (network->training->current_batch > 0);
        }
        /* TODO: WARN: In non-backprop mode retain_previous will be always 0! */
        first_recurrent = PSGetFirstRecurrentLayer(network);
        last_recurrent = PSGetLastRecurrentLayer(network);
        if ((recurrent_input = PSIsRecurrent(input_layer))) {
            /* Read timesteps from first element in `values`. */
            timesteps = (int) values[0];
        } else if (backprop && PSIsRecurrent(output_layer)) {
            /* If feedforward is called from backprop and network has
             * recurrent output but non-recurrent input (OneToMany),
             * we should read the timesteps that are the first element
             * of labels (y) that follows values. */
            PSFloat *y = values + network->input_size;
            timesteps = y[0];
        }
        int steps = timesteps;
        if (steps < 0) steps = 0;
        if (!PSResetNetworkRecurrentStates(network, steps, retain_previous)) {
            network->status = STATUS_ERROR;
            PSErr(NULL, "Failed to reset network recurrent states");
            return 0;
        }
    }
    if (recurrent_input) {
        if (timesteps <= 0) {
            PSErr(
                __func__, "Recurrent timesteps must be > 0 (found %d)",
                timesteps
            );
            return 0;
        }
        ok = feedforwardThroughTime(network, values + 1, timesteps);
        if (!ok) return 0;
        if (last_recurrent == NULL && PSIsRecurrent(output_layer)){
            setNetworkContext(network, last_recurrent_layer, output_layer);
            return ok;
        } else if (last_recurrent != NULL) {
            int last_recurrent_idx = last_recurrent->index;
            if (last_recurrent_idx >= output_idx) return ok;
            else first_idx = last_recurrent_idx;
        }
    } else {
        ok = inputLayerFeedforward(network, values);
        if (!ok) return 0;
    }
    for (i = (first_idx + 1); i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) {
            PSErr(__func__, "Layer %d is NULL!", i);
            return 0;
        }
        if (is_recurrent && layer == first_recurrent)
            return feedforwardThroughTime(network, NULL, timesteps);
        if (layer->feedforward == NULL) {
            PSErr(__func__, "Layer %d feedforward function is NULL", i);
            return 0;
        }
        ok = layer->feedforward(network, layer);
        if (!ok) return 0;
    }
    return 1;
}

/* Feedforward data to neural network. Data is an array of PSFloat (`values`)
 * that must have the same length of units (neurons) in the input (first)
 * layer (in non-recurrent networks).
 * In recurrent networks, `values` length should be input layer size + 1, and
 * the first value indicates the number of iterations (timesteps). */
int PSFeedforward(PSNeuralNetwork *network, PSFloat *values) {
    return feedforward(network, values, 0);
}

PSGradient *createLayerGradients(PSLayer *layer) {
    if (layer == NULL) return NULL;
    PSGradient *gradients;
    PSLayerType ltype = layer->type;
    if (ltype == Pooling) return NULL;
    int size = layer->size;
    PSHyperParameters *parameters = NULL;
    if (ltype == Convolutional) {
        parameters = layer->hyper_parameters;
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
                    PSHyperParameters *prev_params =
                        prev_layer->hyper_parameters;
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
        PSErr(__func__, "Feedforward failed");
        return -1;
    };
    int netsize = network->size;
    PSLayer *out = network->layers[netsize - 1];
    int max_idx = 0, t = (int) out->recurrent_states_count - 1;
    if (t < 0) t = 0;
    if (!PSFindLayerMaxActivation(out, NULL, &max_idx, t)) {
        PSErr(__func__, "Failed to find neuron with max value");
        return -1;
    }
    return max_idx;
}

PSGradient **createGradients(PSNeuralNetwork *network) {
    if (network == NULL) return NULL;
    if (network->size < 2) return NULL;
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

PSGradient **cloneGradients(PSGradient **gradients, PSNeuralNetwork *network) {
    if (gradients == NULL) return NULL;
    PSGradient **clone = createGradients(network);
    if (clone == NULL) return NULL;
    int i, j;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int idx = i - 1;
        PSGradient *lgradients = gradients[idx];
        PSGradient *clone_lgradients = clone[idx];
        if (lgradients == NULL) continue;
        int size, wsize;
        if (Convolutional == layer->type) {
            PSHyperParameters *parameters = layer->hyper_parameters;
            if (parameters == NULL) {
                PSErr(NULL, "Layer %d parameters are NULL!", layer->index);
                goto err;
            }
            size = (int) (parameters->parameters[PARAM_FEATURE_COUNT]);
            PSSharedParams *shared = PSGetConvSharedParams(layer);
            if (!shared) goto err;
            wsize = shared->weights_size;
        } else {
            size = layer->size;
            if (layer->neurons == NULL || layer->neurons[0] == NULL) {
                PSErr(NULL, "Invalid layer[%d]\n", i); goto err;
            }
            wsize = layer->neurons[0]->weights_size;
            if (layer->type == LSTM)
                wsize += 4; /*  Make room for LSTM biases */
        }
        for (j = 0; j < size; j++) {
            PSGradient *og = &(lgradients[j]);
            if (og == NULL) goto err;
            PSGradient *dg = &(clone_lgradients[j]);
            if (dg == NULL) goto err;
            dg->bias = og->bias;
            if (dg->weights == NULL || og->weights == NULL) {
                PSErr(NULL, "Invalid gradients\n"); goto err;
            }
            memcpy(dg->weights, og->weights, wsize * sizeof(PSFloat));
        }
    }
    return clone;
err:
    if (clone != NULL) PSDeleteGradients(clone, network);
    return NULL;
}

void PSDeleteLayerGradients(PSGradient *gradient, int size) {
    int i;
    for (i = 0; i < size; i++) {
        PSGradient *g = &(gradient[i]);
        if (g == NULL || g->weights == NULL) continue;
        free(g->weights);
    }
    free(gradient);
}

void PSDeleteGradients(PSGradient **gradients, PSNeuralNetwork *network) {
    if (gradients == NULL) return;
    int i;
    for (i = 1; i < network->size; i++) {
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        PSLayer *layer = network->layers[i];
        int lsize;
        if (layer->type == Convolutional) {
            PSHyperParameters *params = layer->hyper_parameters;
            lsize = (int) (params->parameters[PARAM_FEATURE_COUNT]);
        } else lsize = layer->size;
        PSDeleteLayerGradients(lgradients, lsize);
    }
    free(gradients);
}

static void resetLayerDeltas(PSLayer *layer, int full_reset) {
    if (layer->delta != NULL) {
        int dsize = layer->size;
        if (layer->type == LSTM && full_reset) {
            dsize *= 2;
            for (int i = 0; i < layer->size; i++) {
                PSNeuron *n = layer->neurons[i];
                PSLSTMCell *cell = PSGetLSTMCell(n);
                if (cell != NULL) cell->last_step_delta = 0.0;
            }
        }
        memset(layer->delta, 0, (size_t) dsize * sizeof(PSFloat));
    }
}

static void resetDeltas(PSNeuralNetwork *network) {
    if (network->layers == NULL) return;
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        resetLayerDeltas(layer, 1);
    }
}

static PSTrainingContext *getTrainingContext(PSNeuralNetwork *network) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) return NULL;
    return ctx->training_context;
}

PSTrainingOptions *PSGetNetworkTrainingOptions(PSNeuralNetwork *network) {
    PSTrainingContext *tctx = getTrainingContext(network);
    if (tctx == NULL) return NULL;
    return &(tctx->options);
}

int PSGetTrainingMemoryGradients(PSNeuralNetwork *network,
                                 PSGradient ***mg1, PSGradient ***mg2)
{
    PSTrainingContext *tctx = getTrainingContext(network);
    int count = 0;
    if (mg1 != NULL) *mg1 = NULL;
    if (mg2 != NULL) *mg2 = NULL;
    if (tctx == NULL) return 0;
    if (tctx->momentum_gradients != NULL) {
        count++;
        if (mg1 != NULL) *mg1 = tctx->momentum_gradients;
    }
    if (tctx->aux_gradients != NULL) {
        count++;
        if (mg2 != NULL) *mg2 = tctx->aux_gradients;
    }
    return count;
}

int initTrainingContext(PSNeuralNetwork *network, int mem_gradients_count) {
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) {
        ctx = network->context = calloc(1, sizeof(PSNetworkContext));
        if (ctx == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
    if (ctx->training_context == NULL) {
        ctx->training_context = calloc(1, sizeof(PSTrainingContext));
        if (ctx->training_context == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
    PSSetDefaultTrainingOptions(&(ctx->training_context->options));
    if (mem_gradients_count > 0) {
        ctx->training_context->momentum_gradients = createGradients(network);
        if (ctx->training_context->momentum_gradients == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        if (mem_gradients_count < 2) goto final;
        ctx->training_context->aux_gradients = createGradients(network);
        if (ctx->training_context->aux_gradients == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
final:
    return 1;
}

int outputLayerBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSFloat *y, PSGradient *layer_gradients, ...)
{
    PSNeuralNetwork *network = layer->network;
    int avx_disabled = 1;
#ifdef USE_AVX
    avx_disabled = PSIsAVXDisabled(network);
#endif
    int apply_derivative = outputDerivativeNeeded(network);
    int is_recurrent = PSIsRecurrent(layer);
    int prev_is_recurrent = PSIsRecurrent(previous_layer);
    int onehot = (layer->flags & FLAG_ONEHOT);
    int is_softmax = layer->type == SoftMax;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSFloat *delta = layer->delta;
    PSFloat softmax_sum = 0.0;
    int o, w, t = 0;
    if (is_recurrent) {
        /* Get timestep `t` */
        va_list args;
        va_start(args, layer_gradients);
        t = va_arg(args, int);
        va_end(args);
    } else if (prev_is_recurrent) {
        /* If output layer is not recurrent, but previous layer is recurrent,
         * t is the last hidden state (state count - 1). */
        t = previous_layer->recurrent_states_count - 1;
        if (t < 0) {
            PSErr(
                NULL, "Recurrent layer %d has no hidden states",
                previous_layer->index
            );
            return 0;
        }
    }
    /* Compute delta */
    for (o = 0; o < layer->size; o++) {
        PSNeuron *neuron = layer->neurons[o];
        PSFloat o_val, y_val, d = 0.0;
        o_val = PSGetActivation(layer, o, t);
        if (onehot) y_val = ((int) *y == o);
        else y_val = y[o];
        if (!is_softmax) {
            d = o_val - y_val;
            if (apply_derivative && layer->derivative != NULL)
                d *= layer->derivative(o_val);
        } else {
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
        }
        delta[o] = d;
        if (!is_softmax) {
            /* Update gradient (non Softmax layer) */
            PSGradient *gradient = &(layer_gradients[o]);
            if (use_bias) gradient->bias += d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                int store_mode = (is_recurrent ? AVX_STORE_MODE_ADD : 0);
                AVXIterativeMultiplyValue(
                    wsize, previous_layer->activations, d,
                    gradient->weights, w, is_recurrent, t, store_mode
                );
            }
#endif
            for (; w < wsize; w++) {
                PSFloat prev_a = PSGetActivation(previous_layer, w, t);
                gradient->weights[w] += (d * prev_a);
                if (avx_disabled && previous_layer->delta != NULL)
                    previous_layer->delta[w] += (d * neuron->weights[w]);
            }
            if (!avx_disabled && previous_layer->delta != NULL) {
                for (w = 0; w < wsize; w++) {
                    previous_layer->delta[w] += (d * neuron->weights[w]);
                }
            }
        }
    }
    if (is_softmax) {
        /* Update gradient (Softmax layer) */
        for (o = 0; o < layer->size; o++) {
            PSNeuron *neuron = layer->neurons[o];
            if (apply_derivative) {
                PSFloat o_val = PSGetActivation(layer, o, t);
                delta[o] -= (o_val * softmax_sum);
            }
            PSFloat d = delta[o];
            PSGradient *gradient = &(layer_gradients[o]);
            if (use_bias) gradient->bias += d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                int store_mode = (is_recurrent ? AVX_STORE_MODE_ADD : 0);
                AVXIterativeMultiplyValue(
                    wsize, previous_layer->activations,
                    d, gradient->weights, w, is_recurrent, t, store_mode
                );
            }
#endif
            for (; w < wsize; w++) {
                PSFloat prev_a = PSGetActivation(previous_layer, w, t);
                gradient->weights[w] += (d * prev_a);
                if (avx_disabled && previous_layer->delta != NULL)
                    previous_layer->delta[w] += (d * neuron->weights[w]);
            }
            if (!avx_disabled && previous_layer->delta != NULL) {
                for (w = 0; w < wsize; w++) {
                    previous_layer->delta[w] += (d * neuron->weights[w]);
                }
            }
        }
    }
    return 1;
}

int fullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *layer_gradients, ...)
{
    PSNeuralNetwork *network = layer->network;
    PSFloat *delta = layer->delta;
    int avx_disabled = 1, i;
#ifdef USE_AVX
    avx_disabled = PSIsAVXDisabled(network);
#else
    UNUSED(network);
#endif
    int is_recurrent = PSIsRecurrent(layer),
        use_bias = !(layer->flags & FLAG_NO_BIAS),
        t = 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer_gradients);
        t = va_arg(args, int);
        va_end(args);
    }
    for (i = 0; i < layer->size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSFloat d = delta[i];
        if (layer->derivative != NULL) {
            PSFloat a = PSGetActivation(layer, i, t);
            d *= layer->derivative(a);
            delta[i] = d;
        }
        PSGradient *gradient = &(layer_gradients[i]);
        if (use_bias) gradient->bias += d;
        int wsize = neuron->weights_size, w = 0;
#ifdef USE_AVX
        if (!avx_disabled) {
            int store_mode = (is_recurrent ? AVX_STORE_MODE_ADD : 0);
            AVXIterativeMultiplyValue(
                wsize, previous_layer->activations,
                d, gradient->weights, w, is_recurrent, t, store_mode
            );
        }
#endif
        for (; w < wsize; w++) {
            PSNeuron *prev_neuron = previous_layer->neurons[w];
            PSFloat prev_a = PSGetActivation(previous_layer, w, t);
            gradient->weights[w] += (d * prev_a);
            if (avx_disabled && previous_layer->delta != NULL) {
                if (!isDroppedOut(prev_neuron))
                    previous_layer->delta[w] += (d * neuron->weights[w]);
            }
        }
        if (!avx_disabled && previous_layer->delta != NULL) {
            for (w = 0; w < wsize; w++) {
                if (isDroppedOut(previous_layer->neurons[w])) continue;
                previous_layer->delta[w] += (d * neuron->weights[w]);
            }
        }
    }
    return 1;
}

int backpropThroughTime(PSNeuralNetwork *network, PSFloat *y,
                        PSGradient **gradients, PSTrainingOptions *opts,
                        int timesteps)
{
    if (network == NULL) return 0;
    if (gradients == NULL) return 0;
    int netsize = network->size;
    PSLayer *output_layer = network->layers[netsize - 1],
            *last_recurrent = PSGetLastRecurrentLayer(network),
            *network_last_recurrent = last_recurrent;
    if (last_recurrent == NULL) last_recurrent = output_layer;
    while (!PSIsRecurrent(last_recurrent)) {
        if (last_recurrent->index <= 1) {
            PSErr(NULL, "Failed to find last recurrent layer");
            return 0;
        }
        last_recurrent = network->layers[last_recurrent->index - 1];
    }
    if (network_last_recurrent == NULL && PSIsRecurrent(last_recurrent))
        setNetworkContext(network, last_recurrent_layer, last_recurrent);
    int last_t = timesteps - 1;
    int recurrent_output = (last_recurrent == output_layer);
    int bptt_truncate = (opts != NULL ? opts->bptt_truncate : BPTT_TRUNCATE);
    if (bptt_truncate < 0) bptt_truncate = 0;
    int onehot, osize, ysize, i, j, t, ok = 1;
    PSFloat *eos_labels = NULL;
    if (recurrent_output) {
        PSRecurrentNetworkOptions *rnn_options = network->rnn_options;
        ok = (y != NULL);
        if (!ok) {
            PSErr(NULL, "Label values for recurrent output layer are NULL");
            goto final;
        }
        onehot = (output_layer->flags & FLAG_ONEHOT);
        osize = output_layer->size;
        ysize = (onehot ? 1 : osize);
        int hidden_states_count = output_layer->recurrent_states_count;
        ok = (hidden_states_count > 0);
        if (!ok) {
            PSErr(NULL, "Cannot backpropagate on recurrent network with "
                  "zero hidden states");
            return 0;
        }
        int max_timesteps = (
            hidden_states_count > timesteps ? hidden_states_count : timesteps
        );
        if (timesteps < max_timesteps) {
            /* Feedforward produced more hidden states than sequences provided
             * by `y` (timesteps). In this case, results produced by the
             * recurrent feedforward that exceed training timesteps must be
             * compared to a virtual EOS (end-of-sequence) output.
             * EOS output must be generated depending on the
             * network->rnn_options->sequence_stop_criterion.eos` value, that
             * in a OneHot index of the output vector (ie. 0).
             * In this case, `eos` is mandatory. */
            int eos = -1;
            if (rnn_options != NULL)
                eos = rnn_options->sequence_stop_criterion.eos;
            if (eos < 0 || eos >= osize) {
                PSErr(NULL, "Invalid eos_recurrent_output_index: %d", eos);
                ok = 0;
                goto final;
            }
            eos_labels = calloc(ysize, sizeof(PSFloat));
            ok = (eos_labels != NULL);
            if (!ok) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            if (!onehot) eos_labels[eos] = 1.0;
            else eos_labels[0] = (PSFloat) eos;
        }
        last_t = max_timesteps - 1;
    }
    int do_truncate = bptt_truncate > 0;
    if (recurrent_output) resetDeltas(network);
    PSFloat *delta;
    for (t = last_t; t >= 0; t--) {
        int lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        PSLayer *previous_layer = NULL;
        if (recurrent_output) {
            /* Backpropagate starting from output layer. */
            PSFloat *timestep_y = NULL;
            if (t < timesteps) {
                int timestep_offset = t * ysize;
                timestep_y = y + timestep_offset;
            } else timestep_y = eos_labels;
            PSGradient *lgradients =
                gradients[netsize - 2];/* No gradients for inputs*/
            previous_layer = network->layers[output_layer->index - 1];
            /* If BPTT is truncated, delta value from previous iteration
             * is not cumulated since it has been already backpropagated
             * to previous timesteps during previous iteration.
             * So, reset previous layer deltas. */
            if (do_truncate)
                resetLayerDeltas(previous_layer, 0);
            ok = outputLayerBackprop(
                output_layer, previous_layer, timestep_y, lgradients, t
            );
            if (!ok) goto final;
        } else previous_layer = last_recurrent;

        /*  Cycle through other layers */
        for (i = previous_layer->index; i > 0; i--) {
            PSLayer *layer = network->layers[i];
            previous_layer = network->layers[i - 1];
            PSGradient *lgradients = gradients[i - 1];
            if (!PSIsRecurrent(layer)) break;
            int lsize = layer->size;
            PSLayerType ltype = layer->type;
            int is_recurrent = (Recurrent == ltype);
            int is_lstm = (LSTM == ltype);
            if (!is_recurrent && !is_lstm) continue;
            /* PSLayerType prev_ltype = previous_layer->type; */

            delta = layer->delta;
            /*  Calculate layer deltas */
            for (j = 0; j < lsize; j++) {
                PSNeuron *neuron = layer->neurons[j];
                PSFloat dv = delta[j];
                if (layer->derivative != NULL) {
                    PSFloat s = PSGetActivation(layer, j, t);
                    dv *= layer->derivative(s);
                }
                if (is_lstm) {
                    PSLSTMCell *lstmcell = PSGetLSTMCell(neuron);
                    PSFloat prev_dv = lstmcell->last_step_delta;
                    delta[j] = prev_dv + dv;
                    lstmcell->last_step_delta = delta[j];
                } else delta[j] = dv;
            }
            int ok = 1;
            if (do_truncate) resetLayerDeltas(previous_layer, 0);
            ok = layer->backprop(
                layer, previous_layer, lgradients, t, lowest_t
            );
            if (!ok) goto final;
        }
    }
final:
    if (eos_labels != NULL) free(eos_labels);
    return ok;
}

PSGradient **backprop(PSNeuralNetwork *network, PSFloat *x, PSFloat *y,
                      PSTrainingOptions *opts, PSGradient **gradients)
{
    if (network == NULL) return NULL;
    if (x == NULL) {
        PSErr(NULL, "Backpropagating NULL `x`");
        return NULL;
    }
    if (y == NULL) {
        PSErr(NULL, "Backpropagating NULL `y`");
        return NULL;
    }
    PSGradient **new_gradients = NULL;
    if (gradients == NULL) {
        gradients = createGradients(network);
        new_gradients = gradients;
    }
    if (gradients == NULL) return NULL;
    int netsize = network->size, is_recurrent = PSIsRecurrent(network);
    PSLayer *output_layer = network->layers[netsize - 1];
    PSGradient *lgradients = gradients[netsize - 2]; /* No gradient for
                                                        inputs */
    PSLayer *previous_layer = NULL;
    resetDeltas(network);

    int i, ok = 1;
    ok = feedforward(network, x, 1, opts);
    if (!ok) goto final;
    if (is_recurrent && PSIsRecurrent(output_layer)) {
        PSLayer *input_layer = network->layers[0];
        PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network);
        int timesteps = 0;
        if (PSIsRecurrent(input_layer)) timesteps = (int) *x;
        else timesteps = (int) *(y++);
        if (timesteps == 0) {
            PSErr(
                NULL, "Recurrent timesteps must be > 0 (found %d)",
                timesteps
            );
            ok = 0;
            goto final;
        }
        ok = backpropThroughTime(network, y, gradients, opts, timesteps);
        if (!ok) goto final;
        /* If first recurrent layer is the input layer, backpropThroughTime
         * has already backpropagated the error to the whole network,
         * so finish here. */
        if (first_recurrent == NULL && PSIsRecurrent(input_layer)) {
            /* First recurrent layer was not set but first layer is recurrent.
             * Update context and finish backpropagation. */
            setNetworkContext(network, first_recurrent_layer, input_layer);
            goto final;
        } else if (first_recurrent != NULL) {
            int first_recurrent_idx = first_recurrent->index;
            if (first_recurrent_idx == 0) goto final;
            else previous_layer = network->layers[first_recurrent_idx - 1];
        } else {
            PSErr(
                NULL, "First recurrent layer was not found, cannot "
                "continue backpropagation"
            );
            ok = 0;
            goto final;
        }
    } else {
        previous_layer = network->layers[output_layer->index - 1];
        ok = outputLayerBackprop(output_layer, previous_layer, y, lgradients);
        if (!ok) goto final;
    }
    for (i = previous_layer->index; i > 0; i--) {
        PSLayer *layer = network->layers[i];
        previous_layer = network->layers[i - 1];
        lgradients = gradients[i - 1];
        if (PSIsRecurrent(layer)) {
            int timesteps = layer->recurrent_states_count;
            ok = (timesteps > 0);
            if (!ok) {
                PSErr(
                    __func__, "Could not get hidden state count for "
                    "recurrent layer %d is NULL", i
                );
                goto final;
            }
            ok = backpropThroughTime(network, NULL, gradients, opts, timesteps);
            if (!ok) goto final;
            break;
        }
        PSLayerType ltype = layer->type;
        PSLayerType prev_ltype = previous_layer->type;
        ok = (
            FullyConnected == ltype ||
            (Pooling == ltype && Convolutional == prev_ltype) ||
            Convolutional == ltype
        );
        if (!ok) {
            PSErr(NULL, "Backprop from %s to %s not suported!\n",
                  PSGetLayerTypeLabel(layer),
                  PSGetLayerTypeLabel(previous_layer));
            goto final;
        }
        ok = layer->backprop != NULL;
        if (!ok) {
            PSErr(
                NULL, "Missing backprop function in %d layer %d",
                PSGetLayerTypeLabel(layer), i
            );
            goto final;
        }
        ok = layer->backprop(layer, previous_layer, lgradients);
        if (!ok) goto final;
    }
final:
    if (!ok) {
        if (new_gradients != NULL) PSDeleteGradients(new_gradients, network);
        return NULL;
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
    } else {
        fprintf(
            stderr,
            "FATAL: invalid param type %d in %s\n", param_type, __func__
        );
        abort();
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
        dx =  - rate *correct1 / (PSSqrt(correct2) + eps);
        return param + dx;
    } else if (optimization == AdaGrad) {
        assert(mg != NULL);
        *mptr = *mptr + grad * grad;
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
    int i, j, k, netsize = network->size, gsize = netsize - 1,
        timesteps = 0, iteration = 0, avx_disabled = 1, apply_clip = 0;
    assert(batch_size > 0);
    PSFloat *x = NULL; /* Training element */
    PSFloat *y = NULL; /* Labels */
    PSFloat l1 = 0.0, l2 = 0.0, l1_loss = 0.0, l2_loss = 0.0, momentum = 0.0,
            clip_max = 0.0, clip_min = 0.0;
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
    int is_recurrent = PSIsRecurrent(network),
        recurrent_input = 0, recurrent_output = 0;
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
        recurrent_input = PSIsRecurrent(network->layers[0]);
        recurrent_output = PSIsRecurrent(network->layers[network->size - 1]);
    }
    int do_dump = (
        network->training != NULL &&
        network->training->debug_dump_to != NULL &&
        network->training->current_batch == 0 &&
        network->training->current_epoch == 0
    );
    UNUSED(elements_count); /* TODO: remove elements_count arg if not needed */
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
            /* For the moment, disable AVX if L1 is used since it would add
             * more complexity in AVX computations.
             * TODO: allow L1 and AVX in the futuer. */
            avx_disabled = 1;
        }
        momentum = opts->momentum;
        optimization = opts->optimization;
        if ((apply_clip = (opts->clip != 0.0))) {
            clip_max = PSAbs(opts->clip);
            clip_min = clip_max * -1;
            avx_disabled = 1;
        }
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

    /* Iterate elements of the batch and, for each element, get gradients
     * from the backpropagation of the error. Then, sum the backpropagation
     * gradients to the batch's gradients. */
    for (i = 0; i < batch_size; i++) {
        if (network->training != NULL) {
            network->training->current_element =
                (network->training->current_batch * batch_size) + i;
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
        } else {
            /* Recurrent network */
            x = series[i];
            if (recurrent_input) {
                timesteps = (int) *x;
                if (timesteps == 0) {
                    PSErr(__func__, "Series len must b > 0. (batch = %d)", i);
                    network->status = STATUS_ERROR;
                    goto final;
                }
                y = x + 1 + (timesteps * training_data_size);
            } else y = x + training_data_size;
        }
        bp_gradients = backprop(network, x, y, opts, gradients);
        if (bp_gradients == NULL) {
            network->status = STATUS_ERROR;
            goto final;
        }
        if (PSDumpGradientsPath != NULL)
            PSDumpGradients(network, gradients, NULL, opts);
        if (network->status == STATUS_PAUSED) break;
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
            PSHyperParameters *params = layer->hyper_parameters;
            l_size = (int) (params->parameters[PARAM_FEATURE_COUNT]);
            shared = PSGetConvSharedParams(layer);
        } else l_size = layer->size;
        int is_lstm = ltype == LSTM;
        int use_bias = !(layer->flags & FLAG_NO_BIAS);
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
            if (use_bias) {
                PSFloat gbias = g->bias / (PSFloat) batch_size;
                if (apply_clip) gbias = PSClipValue(gbias, clip_max, clip_min);
                *bias_ptr = applyGradientOnBias(
                    opts, gbias, bias,
                    mg, xg, rate, iteration
                );
                if (is_lstm) PSUpdateLSTMBiases(
                    neuron, g, mg, xg, rate, opts, iteration, batch_size,
                    clip_max
                );
            }

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
                /*if (do_dump && w < wsize) PSTrainingDebugDumpGradient(
                    network, DEBUG_PHASE_UPDATE_GRADS, __func__,
                    layer, k, wsize, w, 0, 0
                );*/ /* TODO: restore gradient step dumping? */
                grad_w = (l1_grad + l2_grad + g->weights[k]) /
                         (PSFloat) batch_size;
                if (apply_clip)
                    grad_w = PSClipValue(grad_w, clip_max, clip_min);
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
    if (recurrent_output) label_data_size *= timesteps;
    PSFloat outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (onehot) {
            int idx = (int) *(y + i);
            outputs[i] = PSGetActivation(out, idx, i);
        } else {
            if (!recurrent_output) outputs[i] = PSGetActivation(out, i);
            else i = fetchRecurrentOutputState(out, outputs, i, 0);
        }
    }
    if (opts == NULL) l1 = l2 = 0.0;
    if (l1 != 0.0) l1_loss *= (opts->l1_decay / batch_size);
    if (l2 != 0.0) l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    int onehot_size = (onehot ? out->size : 0);
    PSFloat loss =  network->loss(outputs, y, label_data_size, onehot_size);
    if (recurrent_output && timesteps > 0) loss /= timesteps;
    return loss + l1_loss + l2_loss;
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
                       int test_size)
{
    PSTrainingContext *training_ctx = getTrainingContext(network);
    if (training_ctx == NULL) {
        network->status = STATUS_ERROR;
        return STATUS_ERROR_LOSS;
    }
    int batches_count = elements_count / batch_size;
    PSFloat **series = NULL, **series_head = NULL;
    int flags = 0, do_validate = 0;
    if (options != NULL) flags = options->flags;
    if (PSIsRecurrent(network)) {
        PSLayer *out = network->layers[network->size - 1];
        int o_size = (out->flags & FLAG_ONEHOT ? 1 : network->output_size);
        series = getRecurrentSeries(
            network, training_data, elements_count, network->input_size, o_size
        );
        if (series == NULL) {
            network->status = STATUS_ERROR;
            return STATUS_ERROR_LOSS;
        }
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffleSeries(series, elements_count);
    } else {
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffle(training_data, elements_count, element_size);
    }
    PSFloat err = 0.0, avg_err = 0.0, acc = 0.0, tot_acc = 0.0, avg_acc = 0.0;
    long tot_t = 0, avg_t, elapsed_t, test_data_size, validations = 0;
    int offset = (element_size * batch_size), validate_every = 0, i;
    PSGradient **momentum_gradients = training_ctx->momentum_gradients,
               **aux_gradients = training_ctx->aux_gradients;
    if (options != NULL) {
        PSTrainingOptimization optimization = options->optimization;
        if (options->momentum != 0 || optimization != NoTrainingOptimization) {
            if (momentum_gradients == NULL) {
                momentum_gradients = createGradients(network);
                if (momentum_gradients == NULL) {
                    network->status = STATUS_ERROR;
                    goto final;
                }
                training_ctx->momentum_gradients = momentum_gradients;
            }
        }
        if (optimization == AdaDelta || optimization == Adam) {
            if (aux_gradients == NULL) {
                aux_gradients = createGradients(network);
                if (aux_gradients == NULL) {
                    network->status = STATUS_ERROR;
                    goto final;
                }
                training_ctx->aux_gradients = aux_gradients;
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
    series_head = series;
    for (i = 0; i < batches_count; i++) {
        network->training->current_batch = i;
        int batch_num = i + 1;
        struct timeval st, et;
        gettimeofday(&st, NULL);
        PSFloat batch_err = updateNetworkParameters(
            network, training_data, batch_size, elements_count, options,
            learning_rate, momentum_gradients, aux_gradients, series_head
        );
        err += batch_err;
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
            if (series != NULL) free(series);
            return STATUS_ERROR_LOSS;
        }
        if (network->onBatchTrained != NULL) {
            network->onBatchTrained(
                network, network->training->current_epoch,
                epochs, avg_err, batch_err, avg_acc, &learning_rate,
                (series != NULL ? *series_head : training_data)
            );
        }
        if (series == NULL) training_data += offset;
        else series_head += batch_size;
        int action = network->training->requested_action;
        if (action == ACTION_ABORT) {
            network->status = action;
            break;
        }
    }
final:
    if (series != NULL) free(series);
    return err / (PSFloat) batches_count;
}

float validate(PSNeuralNetwork *network, PSFloat *test_data, int data_size,
               int log)
{
    int i, j;
    unsigned char previous_status = network->status;
    char *errmsg = NULL;
    float accuracy = 0.0f;
    int correct_results = 0;
    float correct_amount = 0.0f;
    PSLayer *output_layer = network->layers[network->size - 1];
    int input_size = network->input_size;
    int output_size = network->output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int y_size = (onehot ? 1 : output_size);
    int element_size = input_size + output_size;
    int elements_count;
    int recurrent_input = 0, recurrent_output = 0;
    PSFloat **series = NULL;
    if (PSIsRecurrent(network)) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(test_data++);
        data_size--;
        recurrent_input = PSIsRecurrent(network->layers[0]);
        recurrent_output = PSIsRecurrent(output_layer);
        series = getRecurrentSeries(
            network, test_data, elements_count, input_size, y_size
        );
        if (series == NULL) goto err;
    } else elements_count = data_size / element_size;
    /* PSFloat outputs[output_size]; */
    if (log) printf("Test data elements: %d\n", elements_count);
    network->status = STATUS_VALIDATING;
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
        int timesteps = 0;
        if (series == NULL) {
            /*  Not Recurrent */
            inputs = test_data;
            test_data += input_size;
            expected = test_data;

            int ok = PSFeedforward(network, inputs);
            if (!ok) goto err;

            int omax = 0; /* Output index with max value */
            int emax = 0; /* Expected index with max value */
            if (!PSFindLayerMaxActivation(output_layer, NULL, &omax)) {
                PSErr(NULL, "Could not find output layer max activation");
                goto err;
            }
            if (!onehot) emax = arrayMaxIndex(expected, output_size);
            else emax = (int) *(expected);
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            /*  Recurrent */
            inputs = series[i];
            if (recurrent_input) {
                timesteps = (int) *inputs;
                expected = inputs + 1 + (timesteps * input_size);
            } else {
                timesteps = (int) *(inputs + network->input_size);
                expected = inputs + network->input_size + 1;
            }
            if (timesteps == 0) {
                errmsg = "recurrent data with zero timesteps";
                goto err;
            }
            int ok = PSResetNetworkRecurrentStates(network, timesteps, 0);
            if (!ok) goto err;
            ok = PSFeedforward(network, inputs);
            if (!ok) goto err;

            int correct_states = 0;
            if (recurrent_output) {
                int hidden_state_count = output_layer->recurrent_states_count;
                int steps_to_check = timesteps, max_timesteps = timesteps;
                if (hidden_state_count < timesteps) {
                    steps_to_check = hidden_state_count;
                } else if (hidden_state_count > timesteps)
                    max_timesteps = hidden_state_count;
                int label_data_size = y_size * steps_to_check;
                int last_label_idx = (label_data_size - 1);
                PSFloat outputs[label_data_size];
                for (j = 0; j < label_data_size; j++) {
                    int is_last_label = (j == last_label_idx);
                    j = fetchRecurrentOutputState(
                        output_layer, outputs, j, onehot
                    );
                    if (onehot && (outputs[j] == expected[j]))
                        correct_states++;
                    else if (
                        !onehot && j > 0 &&
                        ((j % y_size) == 0 || is_last_label)
                    ) {
                        int t = ((j + 1) / y_size) - 1;
                        int omax = arrayMaxIndex(
                            outputs + (t * y_size), y_size
                        );
                        int emax = arrayMaxIndex(
                            expected + (t * y_size), y_size
                        );
                        if (emax == omax) correct_states++;
                    }
                }
                correct_amount += (
                    (float) correct_states / (float) max_timesteps
                );
            } else {
                int omax = 0; /* Output index with max value */
                int emax = 0; /* Expected index with max value */
                if (!PSFindLayerMaxActivation(output_layer, NULL, &omax)) {
                    PSErr(NULL, "Could not find output layer max activation");
                    goto err;
                }
                if (!onehot) emax = arrayMaxIndex(expected, output_size);
                else emax = (int) *expected;
                if (omax == emax) correct_results++;
            }
        }
    }
    if (log) {
        printf("\n");
        fflush(stdout);
    }
    time(&end_t);
    if (log) printf("\nCompleted in %ld sec.\n", end_t - start_t);
    if (!recurrent_output) {
        accuracy = (float) correct_results / (float) elements_count;
        if (log) printf("Accuracy (%d/%d): %.2f\n",
                        correct_results, elements_count,accuracy);
    } else {
        accuracy = correct_amount / (float) elements_count;
        free(series);
        if (log) printf("Accuracy: %.2f\n", accuracy);
    }
    network->status = previous_status;
    return accuracy;
err:
    if (errmsg == NULL) {
        if (network->status == STATUS_VALIDATING)
            errmsg = "An error occurred while validating, aborting!";
        else
            errmsg = "Failed to validate network!";
    }
    network->status = STATUS_ERROR;
    fprintf(stderr, "\n");
    PSErr(NULL, "%s", errmsg);
    return STATUS_ERROR_LOSS;
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
    options->bptt_truncate = BPTT_TRUNCATE;
}

void PSTrain(PSNeuralNetwork *network,
             PSFloat *training_data,
             int data_size,
             int epochs,
             PSFloat learning_rate,
             int batch_size,
             PSTrainingOptions *options,
             PSFloat *test_data,
             int test_size)
{
    int i, elements_count;
    if (batch_size <= 0) {
        PSErr(__func__, "Invalid batch_size\n");
        return;
    }
    int element_size = network->input_size + network->output_size;
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            network->status = STATUS_ERROR;
            return;
        }
    }
    int valid = PSCheckNetwork(network);
    if (!valid) {
        network->status = STATUS_ERROR;
        return;
    }
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) {
        PSErr(__func__, "Missing network context\n");
        network->status = STATUS_ERROR;
        return;
    }
    PSTrainingContext *training_ctx = ctx->training_context;
    if (training_ctx != NULL && network->status == STATUS_UNTRAINED) {
        deleteTrainingContext(training_ctx, network);
        ctx->training_context = training_ctx = NULL;
    }
    if (training_ctx == NULL) {
        ctx->training_context = calloc(1, sizeof(PSTrainingContext));
        if (ctx->training_context == NULL) {
            network->status = STATUS_ERROR;
            PSPrintMemoryErrorMsg();
            return;
        }
    }
    training_ctx = ctx->training_context;
    int is_recurrent = PSIsRecurrent(network);
    if (is_recurrent) {
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
    int bptt_truncate = BPTT_TRUNCATE;
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
        bptt_truncate = options->bptt_truncate;
        printf("L1 Decay: %g\n", options->l1_decay);
        printf("L2 Decay: %g\n", options->l2_decay);
        printf("Weight Decay: %s\n", (use_weight_decay ? "yes" : "no"));
        printf("Clip: %g\n", PSAbs(options->clip));
        printf("Momentum: %g\n", options->momentum);
        printf("Optimization: %s\n",
            getOptimizationName(options->optimization));
        int single_seq = (options->flags & TRAINING_EPOCH_AS_SEQUENCE),
            no_shuffle = (options->flags & TRAINING_NO_SHUFFLE);
        if (single_seq && !no_shuffle) {
            fprintf(
                stderr, "WARN: flag TRAINING_EPOCH_AS_SEQUENCE requires "
                "TRAINING_NO_SHUFFLE. "
                "Automatically enabling TRAINING_NO_SHUFFLE.\n"
            );
            options->flags |= TRAINING_NO_SHUFFLE;
        }
        if (single_seq) printf("Single sequence: yes\n");
        printf("Data shuffle (SGD): %s\n", (!no_shuffle ? "yes" : "no"));
        training_ctx->options = *options;
    } else PSSetDefaultTrainingOptions(&training_ctx->options);
    if (is_recurrent) printf("BPTT Truncate: %d\n", bptt_truncate);
    if (network->layers[network->size - 1]->flags & FLAG_ONEHOT)
        printf("Onehot Labels: yes\n");
    char *loss_func_name = NULL;
    if (network->loss != NULL) {
        loss_func_name = getLossFunctionName(network->loss);
        printf("Loss Function: %s\n", loss_func_name);
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
        if (is_recurrent) {
            if (!PSResetNetworkRecurrentStates(network, 0, 0)) {
                network->status = STATUS_ERROR;
                PSErr(NULL, "Failed to reset network recurrent states");
                return;
            }
        }
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
            network->onEpochTrained(network, i, epochs, err, err,
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

int PSCheckNetwork(PSNeuralNetwork *network) {
    if (network == NULL) {
        PSErr(__func__, "Network is NULL");
        return 0;
    }
    int size = network->size, i;
    if (size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    int is_recurrent = PSIsRecurrent(network);
    PSLayer *previous = NULL;
    int onehot_input = 0;
    int recurrent_type_layers = 0, recurrent_layers = 0,
        recurrent_input = 0, recurrent_output = 0;
    PSLayer *actual_first_recurrent_layer = NULL,
            *actual_last_recurrent_layer = NULL,
            *first_recurrent_layer = PSGetFirstRecurrentLayer(network),
            *last_recurrent_layer = PSGetLastRecurrentLayer(network);
    PSLayer *output_layer = network->layers[size - 1];
    recurrent_output = PSIsRecurrent(output_layer);
    for (i = 0; i < size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) {
            PSErr(__func__, "Layer[%d] is NULL", i);
            return 0;
        }
        int ltype = layer->type;
        if (Recurrent == ltype || LSTM == ltype) recurrent_type_layers++;
        int is_recurrent_layer = PSIsRecurrent(layer);
        if (is_recurrent_layer) {
            recurrent_layers++;
            if (actual_first_recurrent_layer == NULL)
                actual_first_recurrent_layer = layer;
            actual_last_recurrent_layer = layer;
        }
        if (i == 0) {
            if (ltype != FullyConnected) {
                PSErr(__func__, "Layer[%d] type must be '%s'",
                      i, PSGetLabelForType(FullyConnected));
                return 0;
            }
            if (layer->flags & FLAG_ONEHOT) {
                PSHyperParameters *params = layer->hyper_parameters;
                onehot_input = 1;
                if (params == NULL) {
                    PSErr(
                        __func__,
                        "Layer[%d] uses a onehot vector index as input, "
                        "but it has no hyper parameters", i
                    );
                    return 0;
                }
                if (params->count < 1) {
                    PSErr(
                        __func__,
                        "Layer[%d] uses a onehot vector index as input, "
                        "but hyper parameters count is < 1", i
                    );
                    return 0;
                }
            }
            recurrent_input = is_recurrent_layer;
        }
        if (ltype == SoftMax && layer != output_layer) {
            PSErr(__func__, "SoftMax layer can only be used as output layer");
            return 0;
        }
        if (ltype == Convolutional) {
            if (onehot_input) {
                PSErr(__func__, "ONEHOT input Layer is not supported on "
                      "Convolutional netowrks");
                return 0;
            }
            /* TODO: remove this contraint */
            if (network->flags & FLAG_RECURRENT) {
                PSErr(
                    __func__,
                    "Sorry, Convolutional layers aren't yet supported "
                    "on recurrent networks :("
                );
                return 0;
            }
        }
        /* TODO: remove this contraint */
        if (ltype == Pooling && previous && previous->type != Convolutional) {
            PSErr(__func__, "Layer[%d] type is Pooling, "
                  "but previous type is not Convolutional", i);
            return 0;
        }
        /* TODO: remove this contraint */
        if (ltype != Pooling && previous && previous->type == Convolutional) {
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
        if (layer == output_layer && (layer->flags & FLAG_ONEHOT)) {
            if (SoftMax != ltype) {
                PSErr(
                    __func__, "output layer with flag FLAG_ONEHOT must "
                    "be a SoftMax layer"
                );
                return 0;
            }
        }
        previous = layer;
    }
    if (is_recurrent) {
        PSRecurrentNetworkMode rnn_mode = PSGetRecurrentNetworkMode(network);
        if (rnn_mode == NonRecurrent) {
            PSErr(__func__, "Recurrent network mode is NonRecurrent");
            return 0;
        }
        if (recurrent_layers == 0) {
            PSErr(__func__,
                "Network is recurrent but has no recurrent layers"
            );
            return 0;
        }
        if (recurrent_type_layers == 0) {
            PSErr(__func__,
                "Network is recurrent but has no Recurrent or LSTM layers"
            );
            return 0;
        }
        if (first_recurrent_layer == NULL) {
            PSErr(__func__,
                "Recurrent network is missing first recurrent layer"
            );
            return 0;
        }
        if (last_recurrent_layer == NULL) {
            PSErr(__func__,
                "Recurrent network is missing last recurrent layer"
            );
            return 0;
        }
        if (first_recurrent_layer != actual_first_recurrent_layer) {
            PSErr(__func__,
                "Recurrent network first recurrent layer should be layer %d, "
                "but network is not updated", actual_first_recurrent_layer
            );
            return 0;
        }
        if (last_recurrent_layer != actual_last_recurrent_layer) {
            PSErr(__func__,
                "Recurrent network last recurrent layer should be layer %d, "
                "but network is not updated", actual_last_recurrent_layer
            );
            return 0;
        }
        if (ManyToMany == rnn_mode) {
            if (!recurrent_input && !recurrent_output) {
                PSErr(__func__,
                    "Recurrent network with mode \"%s\" has no recurrent "
                    "input nor recurrent output",
                    PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
        } else if (ManyToOne == rnn_mode) {
            if (!recurrent_input) {
                PSErr(__func__,
                    "Recurrent network with mode \"%s\" has no recurrent "
                    "input", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
            if (recurrent_output) {
                PSErr(__func__,
                    "Recurrent network with mode \"%s\" has recurrent "
                    "output", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
        } else if (OneToMany == rnn_mode) {
            if (recurrent_input) {
                PSErr(__func__,
                    "Recurrent network with mode \"%s\" has recurrent "
                    "input", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
            if (!recurrent_output) {
                PSErr(__func__,
                    "Recurrent network with mode \"%s\" has no recurrent "
                    "output", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
        }
    } else {
        if (recurrent_type_layers > 0) {
            PSErr(__func__,
                "Network is not recurrent but has Recurrent or LSTM layers"
            );
            return 0;
        }
    }
    int softmax_output = (output_layer->type == SoftMax);
    if (network->loss == NULL) {
        PSErr(__func__, "Missing loss function");
        return 0;
    } else {
        if (softmax_output && network->loss != PSCrossEntropyLoss) {
            PSErr(__func__, "SoftMax output requires PSCrossEntropyLoss");
            return 0;
        } else if (!softmax_output && network->loss == PSCrossEntropyLoss) {
            PSErr(__func__, "PSCrossEntropyLoss require PSCrossEntropyLoss");
            return 0;
        }
    }
    return 1;
}

size_t PSIterateLossFunctions(
    void ( *callback) (const char *name, PSLossFunction func)
)
{
    size_t i;
    for (i = 1; i < loss_functions_count; i++) {
        if (callback != NULL) {
            PSLossFunction func = loss_functions[i];
            char *name = getLossFunctionName(func);
            callback((const char*) name, func);
        }
    }
    return loss_functions_count - 1;
}

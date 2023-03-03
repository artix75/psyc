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
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <signal.h>
#include <sys/time.h>
#include <inttypes.h>

#include "platform.h"
#include "buildinfo.h"
#include "psyc.h"
#include "maths.h"
#include "activation.h"
#include "utils.h"
#include "log.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "gru.h"
#include "embedding.h"
#include "dropout.h"
#include "debug.h"

#define STATUS_ERROR_LOSS ((PSFloat) FLT_MIN)
#define BPTT_TRUNCATE   4

#define applyGradientsOnBiases(opts, grads, params, mg, xg, len, r, i, accel) \
    applyGradientsOnParameters(PARAM_TYPE_BIAS, opts, grads, params, mg, xg,\
    0, len, r, i, accel)
#define applyGradientsOnWeights(opts,grad,val,mg,xg,offs,len,r,i,accel) \
    applyGradientsOnParameters(PARAM_TYPE_WEIGHT, opts, grad, val, mg, xg,\
    offs, len, r, i, accel)
#define outputDerivativeNeeded(network) (network->loss != PSCrossEntropyLoss)
#define getNetworkContext(network) ((PSNetworkContext *) network->context)
#define setNetworkContext(network, member, val) (\
    ((PSNetworkContext *) network->context)->member = val)
#define isPretrainableLayer(layer) (layer->pretrain != NULL)
#define layerNeedsPretraining(layer) \
    (isPretrainableLayer(layer) && !layer->pretrained)

#define PSReadSequenceArgs(lastarg, is_recurrent, seqlen, step) do {\
    va_list _args;\
    va_start(_args, lastarg);\
    seqlen = va_arg(_args, int);\
    if (is_recurrent) step = va_arg(_args, int);\
    va_end(_args);\
} while(0);

#ifdef BACKTRACE_AVAILABLE
void segvHandler(int sig, siginfo_t *info, void *secret);
#endif

#define UNUSED(V) ((void) V)

typedef struct {
    PSTrainingOptions options;
    PSGradient **memory_gradients1;
    PSGradient **memory_gradients2;
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

char *getLossFunctionName(PSLossFunction function);
char *getNetworkStatusLabel(PSNeuralNetwork *network);
float validate(PSNeuralNetwork *network, PSFloat *test_data, int data_size,
               int log);
int PSInitConvolutionalLayer(PSNeuralNetwork *network, PSLayer *layer,
                             PSLayerDef *ldef);
int PSInitPoolingLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSLayerDef *ldef);
int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *layer_gradients, ...);
int PSInitRecurrentLayer(PSNeuralNetwork *network, PSLayer *layer,
                         int size,int ws, PSLayerDef *ldef);
int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws, PSLayerDef *ldef);
int PSInitGRULayer(PSNeuralNetwork *network, PSLayer *layer,
                   int size, int ws, PSLayerDef *ldef);
int PSInitDropoutLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSLayerDef *layer_def);
int PSInitEmbeddingLayer(PSLayer *layer, int size, int previous_size,
                         PSLayerDef *ldef);
int PSInitNormalizationLayer(PSLayer *layer, PSLayerDef *ldef);
int PSDumpGradients(PSNeuralNetwork *network, PSGradient **gradients,
                    const char* filename, PSTrainingOptions *opts);
PSGradient **cloneNetworkGradients(PSGradient **gradients,
                                   PSNeuralNetwork *network);
static void deleteTrainingContext(PSTrainingContext *training_ctx,
                                  PSNeuralNetwork *network);
static void deleteNetworkContext(PSNetworkContext *ctx,
                                 PSNeuralNetwork *network);
int writeSerializedFloat(FILE *out, PSFloat fnum, int opts);
int PSBeforeSequenceFeedforward(PSLayer *layer, int seqlen, int t);

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
    if (PSLogLevel > PSLOGLEVEL_INFO) return 0;
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

void dumpFeedforwardStep(int i, PSFloat a, PSFloat b, PSFloat sum,
                         int using_acceleration, PSMathOpts *opts)
{
    UNUSED(a);
    UNUSED(b);
    UNUSED(sum);
    UNUSED(using_acceleration);
    if (opts == NULL) return;
    PSDebugStepInfo *info = (PSDebugStepInfo *) opts->data;
    if (info == NULL || info->layer == NULL || info->network == NULL) return;
    info->training_phase = TRAINING_PHASE_FEEDFORWARD;
    int prev_idx = info->layer->index - 1;
    PSTrainingDebugDumpStep(
        info, "previous_neuron=%d-%d,weight_index=%d\n", prev_idx, i, i
    );
}

/* Feedforward Functions */

int checkLayerForFeedforward(PSLayer *layer) {
    if (layer == NULL) return 0;
    int trainable = !(layer->flags & FLAG_NON_TRAINABLE);
    int needs_neurons = (
        Dropout != layer->type && Normalization != layer->type
    );
    if (layer->neurons == NULL && needs_neurons) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot perform feedforward on layer 0");
        return 0;
    }
    if (layer->network == NULL) {
        PSErr(NULL, "Layer[%d]: layer has no network");
        return 0;
    }
    PSLayer *previous = layer->network->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL", layer->index);
        return 0;
    }
    if (layer->weight_types_count > 0 && trainable) {
        if (layer->weights == NULL) {
            PSErr(NULL, "Layer[%d]: layer has no weights", layer->index);
            return 0;
        }
        for (int i = 0; i < layer->weight_types_count; i++) {
            if (layer->weights[i] == NULL) {
                PSErr(NULL, "Layer[%d]: weights[%d] are NULL", layer->index, i);
                return 0;
            }
        }
    } else {
        if (layer->type != Pooling && layer->type != Dropout) {
            PSErr(NULL, "Layer[%d]: weights required for layer type %s",
                  PSGetLabelForType(layer->type));
            return 0;
        }
    }
    return 1;
}

void handleLayerFeedforwardDebug(PSLayer *layer, const char *func,
                                 PSMathOpts *opts)
{
#ifdef PS_DEBUG_MODE
    PSAddContextualDebug(layer->network, layer, NULL, NULL, "feedforward", 0);
#endif
    PSDebugStepInfo dbginfo =
        {.network = layer->network, .layer = layer, .func = func};
    if (opts != NULL && PSShouldDebugDump(layer->network)) {
        opts->data = &dbginfo;
        opts->debug_step = dumpFeedforwardStep;
    }
}

/* Onehot input layers only have one input corresponding to the index
 * of the activated unit. In this case, just take the value of the
 * corresponding weight, since the input should always be considered
 * as it would be 1 */
int PSOnehotInputsFeedforward(PSLayer *layer, int weights_index,
                              PSFloat *outputs, int t, int apply_biases,
                              int do_activate)
{
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d] has no previous layer", layer->index);
        return 0;
    }
    int onehot_vector_size = PSGetOneHotLayerVectorSize(previous);
    if (weights_index < 0) weights_index = 0;
    if (weights_index >= layer->weight_types_count) {
        PSErr(NULL, "Layer[%d] invalid weights index %d (max: %d)",
              weights_index,  layer->weight_types_count - 1);
        return 0;
    }
    PSMatrix weights = (
        layer->weights != NULL ? layer->weights[weights_index] : NULL
    );
    if (weights == NULL) {
        PSErr(NULL, "Layer[%d] has no weights[%d]",layer->index,weights_index);
        return 0;
    }
    int use_bias = apply_biases && !(layer->flags & FLAG_NO_BIAS);
    do_activate = do_activate && layer->activate != NULL;
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    int seqlen = 1;
    if (PSHandleSequenceAtOnce(layer)) {
        seqlen = PSStateSequenceLength(layer);
        t = 0;
        if (seqlen < 1) {
            PSErr(NULL, "Layer[%d]: sequence length is %d", seqlen);
            return 0;
        }
    }
    PSMatrix tweights = PSMatrixTranspose(weights, 0, &opts);
    if (tweights == NULL) {
        PSWarn("Layer[%d]: failed to transpose weights[%d]",
              layer->index, weights_index);
        goto no_transposition;
    }
    PSFloat *out;
    for (int s = 0; s < seqlen; s++) {
        int tidx = s + t;
        int onehot_idx = (int) PSGetState(previous, 0, tidx);
        if (onehot_idx >= onehot_vector_size) {
            PSErr(
                NULL, "Layer[%d]: onehot index %d is out of range "
                "(vector size = %d)", previous->index, onehot_idx,
                onehot_vector_size
            );
            return 0;
        }
        if (outputs == NULL) out = PSGetStates(layer, tidx);
        else out = outputs + (t * layer->size);
        if (out == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs at %d", layer->index, tidx);
            return 0;
        }
        PSFloat *states = tweights + (onehot_idx * layer->size);
        PSVectorCopy(out, states, layer->size);
        if (use_bias)
            PSSumVectors(out, layer->biases, out, layer->size, &opts);
        if (do_activate) layer->activate(out, NULL, layer->size, &opts);
    }
    return 1;

no_transposition:
    for (int s = 0; s < seqlen; s++) {
        int tidx = s + t;
        int onehot_idx = (int) PSGetState(previous, 0, tidx);
        if (onehot_idx >= onehot_vector_size) {
            PSErr(
                NULL, "Layer[%d]: onehot index %d is out of range "
                "(vector size = %d)", previous->index, onehot_idx,
                onehot_vector_size
            );
            return 0;
        }
        PSFloat *outputs = PSGetStates(layer, tidx);
        if (outputs == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs", layer->index);
            return 0;
        }
        for (int i = 0; i < layer->size; i++) {
            PSNeuron *n = layer->neurons[i];
            PSFloat w;
            if (n != NULL && n->weights != NULL) w = n->weights[onehot_idx];
            else w = weights[(i * layer->size) + onehot_idx];
            outputs[i] = w;
            if (use_bias) outputs[i] += layer->biases[i];
        }
        if (do_activate)
            layer->activate(outputs, NULL, layer->size, &opts);
    }
    return 1;
}

int PSFullFeedforward(PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    PSNeuralNetwork *network = layer->network;
    PSLayer *previous = PSGetPreviousLayer(layer);
    PSMathOpts opts = {.acceleration = network->acceleration};
    handleLayerFeedforwardDebug(layer, __func__, &opts);
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 0, t = 0;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceFeedforward(layer, seqlen, t)) return 0;
    }
    if (previous->flags & FLAG_ONEHOT) {
        /* Onehot inputs */
        if (!PSOnehotInputsFeedforward(layer, 0, NULL, t, 1, 1)) return 0;
        goto final;
    }
    PSMatrix weights = layer->weights[0];
    opts.acceleration = network->acceleration;
    if (!handles_seq) {
        PSFloat *inputs = PSGetStates(previous, t),
                *outputs = PSGetStates(layer, t);
        if (inputs == NULL) {
            PSErr(
                NULL, "Layer[%d]: previous layer[%d] has NULL outputs",
                layer->index, previous->index
            );
            return 0;
        }
        opts.argtype[1] = 'V';
        int ok = PSDot(weights, inputs, outputs, &opts);
        if (!ok) return 0;
        if (use_bias)
            PSSumVectors(outputs, layer->biases, outputs, layer->size, &opts);
        if (layer->activate) layer->activate(outputs, NULL, layer->size, &opts);
    } else {
        opts.transpose = 2;
        if (use_bias) {
            int seqlen = PSMatrixDim(previous->states, 0), i;
            for (i = 0; i < seqlen; i++) {
                PSFloat *outputs = PSGetStates(layer, i);
                PSVectorCopy(outputs, layer->biases, layer->size);
            }
            opts.store_mode = PS_STORE_MODE_ADD;
        }
        int ok = PSDot(previous->states, weights, layer->states, &opts);
        if (!ok) return 0;
        opts.store_mode = PS_STORE_MODE_SET;
        if (layer->activate)
            layer->activate(layer->states, NULL, seqlen * layer->size, &opts);
    }
final:
    return 1;
}

static int softmaxFeedforward(PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    PSMathOpts opts = {0};
    handleLayerFeedforwardDebug(layer, __func__, &opts);
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 0, t = 0;
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceFeedforward(layer, seqlen, t)) return 0;
    }
    if (!PSFullFeedforward(layer, seqlen, t)) return 0;
    if (seqlen < 1 || !handles_seq) seqlen = 1;
    for (int i = 0; i < seqlen; i++) {
        PSFloat *outputs = PSGetStates(layer, i + t);
        if (outputs == NULL) {
            PSErr(NULL, "Layer[%d]: missing outputs[%d]", layer->index, i + t);
            return 0;
        }
        PSSoftmax(outputs, outputs, layer->size, &opts);
    }
    return 1;
}

/* Utils */

static PSFloat norm(PSFloat* vector, int size) {
    PSFloat r = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        PSFloat v = vector[i];
        r += (v * v);
    }
    /*assert(!isnan(PSSqrt(r)));*/
    PSFloat norm = PSSqrt(r);
    if (isnan(norm)) {
        fprintf(stderr, "\n\nPSSqrt(%g) is nan!\n", r);
        for (i = 0; i < size; i++) {
            PSFloat v = vector[i];
            fprintf(stderr, " -> vector[%d] = %g\n", i, v);
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

static void shuffleSequences(PSFloat **sequences, int size) {
    srand ( time(NULL) );
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        /* printf("Shuffle cycle %d: random is %d\n", i, j); */
        PSFloat *tmp_a = sequences[i];
        PSFloat *tmp_b = sequences[j];
        sequences[i] = tmp_b;
        sequences[j] = tmp_a;
    }
}

static PSFloat **getRecurrentSequences(PSNeuralNetwork *network, PSFloat *data,
                                       int sequence_count, int xsize,
                                       int ysize)
{
    int recurrent_input = PSIsRecurrent(network->layers[0]),
        recurrent_output = PSIsRecurrent(network->layers[network->size-1]), i;
    if (!recurrent_input && !recurrent_output) {
        PSErr(NULL, "Network has no recurrent input nor recurrent output");
        return NULL;
    }
    PSFloat **sequences = malloc(sizeof(PSFloat *) * sequence_count);
    if (sequences == NULL) {
        PSErr(NULL, "Could not allocate memory for recurrent sequences!");
        return NULL;
    }
    PSFloat *p = data;
    for (i = 0; i < sequence_count; i++) {
        PSFloat *size_p = (recurrent_input ? p : p + xsize);
        int sequence_size = (int) *size_p;
        if (sequence_size <= 0) {
            PSErr(
                NULL, "Sequence[%d] Invalid length %d at data offset %d",
                i, sequence_size, (int) (size_p - data)
            );
            free(sequences);
            return NULL;
        }
        int xmul = (recurrent_input ? sequence_size : 1),
            ymul = (recurrent_output ? sequence_size : 1);
        sequences[i] = p++;
        p += ((xmul * xsize) + (ymul * ysize));
    }
    return sequences;
}

PSFloat *PSGetInputsFromTrainingData(PSFloat *training_data, int data_size,
                                     int num_elements, int input_size,
                                     int label_size, int recurrent_input,
                                     int recurrent_output,
                                     int *count, size_t *result_size)
{
    if (training_data == NULL || data_size <= 0) return NULL;
    int element_size = input_size + label_size;
    int recurrent = (recurrent_input || recurrent_output);
    if (num_elements <= 0) {
        /* Auto-detect number of elements */
        if (recurrent) {
            /*  First training data number for Recurrent networks must
             *indicate the data elements count */
            num_elements = (int) *(training_data++);
            data_size--;
        } else num_elements = data_size / element_size;
    }
    if (num_elements <= 0) return NULL;
    if (count != NULL) *count = num_elements;
    size_t size = num_elements * input_size * sizeof(PSFloat);
    if (result_size != NULL) *result_size = size;
    PSFloat *inputs = malloc(size);
    if (inputs == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSFloat *data_p = training_data, *inputs_p = inputs;
    int nwritten = 0, tot_recurrent_elements = 0;
    while (num_elements > 0) {
        if (!recurrent) {
            memcpy(inputs_p, data_p, input_size * sizeof(PSFloat));
            data_p += input_size;
            inputs_p += input_size;
        } else {
            int seqlen = *(data_p++);
            int input_len =
                (recurrent_input ? seqlen * input_size : input_size);
            int label_len =
                (recurrent_output ? seqlen * label_size : label_size);
            size_t seq_input_size = input_len * sizeof(PSFloat);
            size_t new_size = nwritten + seq_input_size;
            if (new_size > size) {
                PSFloat *resized = realloc(inputs, new_size);
                if (resized == NULL) {
                    PSPrintMemoryErrorMsg();
                    goto fail;
                }
                inputs_p = resized + (inputs_p - inputs);
                inputs = resized;
                if (result_size != NULL) *result_size = new_size;
            }
            memcpy(inputs_p, data_p, seq_input_size);
            nwritten += seq_input_size;
            inputs_p += input_len;
            data_p += input_len + label_len;
            tot_recurrent_elements += input_len;
        }
        num_elements--;
    }
    if (recurrent && count != NULL) *count = tot_recurrent_elements;
    return inputs;
fail:
    free(inputs);
    return NULL;
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

/* Find max state value and the relative neuron index for layer `layer`,
 * and store them into `max_p` pointer (max state) and `index_p` pointer
 * (index of neuron having maximum state value).
 * At least `max_p` or `index_p` must be provided.
 * If layer is recurrent, an extra argument for timestep must be provided
 * as a variadic argument (as int).
 * If timestep is negative, it will be used to read states in a reverse
 * order (ie. -1 is last timestep, -2 is last timestep - 1, etc.).
 * Timestep must be always in range of processed timesteps (hidden states),
 * otherwise the function will fail.
 * Return value: 1 in case of success, 0 in case of error. */
int PSFindLayerMaxState(PSLayer *layer, PSFloat *max_p, int *index_p, ...)
{
    if (max_p == NULL && index_p == NULL) return 0;
    int seqidx = 0, i;
    int uses_sequences = PSUseSequences(layer);
    if (uses_sequences) {
        va_list args;
        va_start(args, index_p);
        seqidx = va_arg(args, int);
        va_end(args);
    }
    PSFloat max = 0.0;
    int max_idx = -1;
    PSFloat *states = PSGetStates(layer, seqidx);
    if (states == NULL) return 0;
    for (i = 0; i < layer->size; i++) {
        PSFloat state = states[i];
        if (state > max) {
            max = state;
            max_idx = i;
        }
    }
    if (max_idx < 0) return 0;
    if (max_p != NULL) *max_p = max;
    if (index_p != NULL) *index_p = max_idx;
    return 1;
}

static int fetchSequenceOutputState(PSLayer *out, PSFloat *outputs,
                                    int i, int onehot)
{
    int t = (onehot ? i : i / out->size), j;
    int max_idx = 0, oidx = 0;
    PSFloat max = 0.0;
    for (j = 0; j < out->size; j++) {
        PSFloat s = PSGetState(out, j, t);
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
        case GRU:
            return "GRU";
        case Dropout:
            return "Dropout";
        case Embedding:
            return "Embedding";
        case Normalization:
            return "Normalization";
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

char *getOptimizationName(PSOptimization optimization) {
    if (optimization == PSDefaultOptimization) return "Default";
    else if (optimization == PSAdamOptimization) return "Adam";
    else if (optimization == PSAdaGradOptimization) return "AdaGrad";
    else if (optimization == PSAdaDeltaOptimization) return "AdaDelta";
    else if (optimization == PSNesterovOptimization) return "Nesterov";
    else if (optimization == PSWindowGradOptimization) return "WindowGrad";
    return "UNKOWN";
}

PSScalarActivationFunction PSGetScalarActivationFunc(PSActivationFunction func){
    if (func == PSSigmoid) return PSSigmoidS;
    else if (func == PSTanhActivation) return PSTanhS;
    else if (func == PSRelu) return PSReluS;
    else if (func == PSSigmoidDerivative) return PSSigmoidDerivativeS;
    else if (func == PSTanhDerivative) return PSTanhDerivativeS;
    else if (func == PSReluDerivative) return PSReluDerivativeS;
    return NULL;
}

const char *PSGetActivationName(PSActivationFunction func) {
    if (func == PSSigmoid) return "sigmoid";
    else if (func == PSTanhActivation) return "tanh";
    else if (func == PSRelu) return "relu";
    return NULL;
}

PSActivationFunction PSGetActivationDerivative(PSActivationFunction func) {
    if (func == PSSigmoid) return PSSigmoidDerivative;
    else if (func == PSTanhActivation) return PSTanhDerivative;
    else if (func == PSRelu) return PSReluDerivative;
    return NULL;
}

uint64_t PSGetLayerParametersCount(PSLayer *layer, int param_type) {
    if (layer == NULL) return 0;
    if (Pooling == layer->type || layer->index == 0 || layer->size == 0 ||
        Dropout == layer->type) return 0;
    if (param_type == 0) param_type = (PARAM_TYPE_WEIGHT | PARAM_TYPE_BIAS);
    if (layer->get_param_count != NULL)
        return layer->get_param_count(layer, param_type);
    uint64_t count = 0;
    if (param_type & PARAM_TYPE_WEIGHT && layer->weights != NULL) {
        for (int i = 0; i < layer->weight_types_count; i++) {
            PSMatrix weights = layer->weights[i];
            if (weights != NULL) count += PSMatrixLength(weights);
        }
    }
    if (param_type & PARAM_TYPE_BIAS) {
        if (!(layer->flags & FLAG_NO_BIAS) && layer->biases != NULL) {
            int bias_count;
            if (Convolutional == layer->type) bias_count = layer->output_depth;
            else if (LSTM == layer->type) bias_count = layer->size * 4;
            else if (GRU == layer->type) bias_count = layer->size * 3;
            else bias_count = layer->size;
            count += bias_count;
        }
    }
    return count;
}

uint64_t PSGetNetworkParametersCount(PSNeuralNetwork *network) {
    int tot = 0, param_type = (PARAM_TYPE_BIAS | PARAM_TYPE_WEIGHT), i;
    if (network->layers == NULL || network->size == 0) return 0;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        tot += PSGetLayerParametersCount(layer, param_type);
    }
    return tot;
}

int PSGetOneHotLayerVectorSize(PSLayer *layer) {
    if (!(layer->flags & FLAG_ONEHOT)) return layer->size;
    return layer->onehot_vector_size;
}

PSLayer *PSGetPreviousLayer(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (layer->network == NULL) return NULL;
    if (layer->network->layers == NULL) return NULL;
    int previous_layer_idx = layer->index - 1;
    if (previous_layer_idx < 0) return NULL;
    if (previous_layer_idx >= layer->network->size) return NULL;
    return layer->network->layers[previous_layer_idx];
}

PSLayer *PSGetNextLayer(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (layer->network == NULL) return NULL;
    if (layer->network->layers == NULL) return NULL;
    int next_layer_idx = layer->index + 1;
    if (next_layer_idx < 0) return NULL;
    if (next_layer_idx >= layer->network->size) return NULL;
    return layer->network->layers[next_layer_idx];
}

PSLayer *PSGetOutputLayer(PSNeuralNetwork *network) {
    if (network == NULL || network->layers == NULL || network->size == 0)
        return NULL;
    return network->layers[network->size - 1];
}

int PSGetLayerInputSize(PSLayer *layer) {
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) return 0;
    if (previous->flags & FLAG_ONEHOT)
        return PSGetOneHotLayerVectorSize(previous);
    return previous->size;
}

uint64_t PSGetLayerInputWeightsCount(PSLayer *layer, int per_neuron) {
    if (layer == NULL) return 0;
    if (layer->weights == NULL) return 0;
    if (layer->weights[0] == NULL) return 0;
    if (layer->type == Pooling || layer->type == Dropout) return 0;
    uint64_t wcount = PSMatrixLength(layer->weights[0]);
    if (per_neuron && layer->type != Convolutional) wcount /= layer->size;
    return wcount;
}

PSFloat *PSGetNeuronInputWeights(PSNeuron *neuron) {
    if (neuron->layer == NULL) return NULL;
    if (neuron->layer->type == Pooling || neuron->layer->type == Dropout)
        return NULL;
    if (neuron->layer->weights == NULL || neuron->layer->weights[0] == NULL)
        return NULL;
    uint64_t wcount = PSGetLayerInputWeightsCount(neuron->layer, 1);
    uint64_t widx = neuron->index * wcount;
    if (widx >= (wcount * neuron->layer->size)) {
        PSErr(__func__, "Layer[%d] Neuron[%d] index is out of bounds",
              neuron->layer->index, neuron->index);
        return NULL;
    }
    return neuron->layer->weights[0] + widx;
}

void PSPrintLayerInfo(PSLayer *layer) {
    if (layer == NULL) return;
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    char onehot_info[50];
    onehot_info[0] = 0;
    int onehot_input = (layer->index == 0 && layer->flags & FLAG_ONEHOT);
    if (onehot_input)
        sprintf(onehot_info, " (vector size: %d)", layer->onehot_vector_size);
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if (Dropout == layer->type) printf(", dropout = %g", PSGetDropout(layer));
    if (onehot_info[0]) printf(" %s", onehot_info);
    if (ltype == Convolutional || ltype == Pooling) {
        PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
        int output_w = layer->output_columns, output_h = layer->output_rows,
            depth = layer->output_depth, filter_w = 0, filter_h = 0,
            filter_d = 0, input_w = 0, input_h = 0, stride = 0, padding = 0;
        if (settings != NULL) {
            input_w = settings->input_width;
            input_h = settings->input_height;
            filter_w = settings->filter_width;
            filter_h = settings->filter_height;
            filter_d = settings->filter_depth;
            stride = settings->stride;
            padding = settings->padding;
        }
        if (stride <= 0 && ltype == Pooling) stride = filter_w;
        printf(", input size = %dx%d, output_size = %dx%d, depth = %d",
            input_w, input_h, output_w, output_h, depth);
        printf(", filter = %dx%dx%d, stride = %d",
               filter_w, filter_h, filter_d, stride);
        if (ltype == Convolutional) {
            if (padding < 0) padding = 0;
            printf(", padding = %d", padding);
        }
    } else if (ltype == FullyConnected && layer->output_depth > 1) {
       printf(", depth = %d", layer->output_depth);
    } else if (ltype == Embedding) {
        int vocab_size = PSGetEmbeddingVocabularySize(layer);
        if (vocab_size > 0)
            printf(", vocabulary_size = %d", vocab_size);
    }
    const char *activation = PSGetActivationName(layer->activate);
    if (layer->index > 0 && activation != NULL)
        printf(", activation = %s", activation);
    printf("\n");
}

static void printInfoRow(char *label, char *fmt, ...) {
    if (label == NULL) return;
    if (fmt == NULL) fmt = "";
    int use_color = PSLogColorEnabled();
    if (use_color) printf(PSCOLOR_CYAN);
    int padding = 40;
    int len = printf("%s:", label);
    if (use_color) printf(PSCOLOR_RESET);
    printf("%-*s", (padding - len), " ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    printf("\n");
}

void PSPrintNetworkInfo(PSNeuralNetwork *network) {
    if (network == NULL) return;
    const char *name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    printInfoRow("Name", "\"%s\"", name);
    printInfoRow("Size", "%d", network->size);
    int is_recurrent = PSIsRecurrent(network);
    PSRecurrentNetworkMode mode = PSGetRecurrentNetworkMode(network);
    PSNetworkContext *ctx = network->context;
    assert(ctx != NULL);
    PSLayer *first_recurrent_layer = NULL, *last_recurrent_layer = NULL;
    if (is_recurrent || mode != NonRecurrent) {
        first_recurrent_layer = PSGetFirstRecurrentLayer(network);
        last_recurrent_layer = PSGetLastRecurrentLayer(network);
        printInfoRow("Recurrent Network Mode", "%s",
                     PSGetRecurrentModeLabel(mode));
        if (ManyToOne == mode) {
            if (last_recurrent_layer != NULL) {
                printInfoRow("Last Recurrent Layer", "%d",
                             last_recurrent_layer->index);
            }
        } else if (OneToMany == mode) {
            if (first_recurrent_layer != NULL) {
                printInfoRow("First Recurrent Layer", "%d",
                             first_recurrent_layer->index);
            }
        }
    }
    printInfoRow("Total (trainable) parameters", "%" PRIu64,
                 PSGetNetworkParametersCount(network));
    char *loss_name = getLossFunctionName(network->loss);
    if (loss_name != NULL) printInfoRow("Loss Function", "%s", loss_name);
    printInfoRow("Status", "%s", getNetworkStatusLabel(network));
    printInfoRow("AVX", "%s",
                 (PSAVXEnabled(network->acceleration) ? "yes" : "no"));
    printInfoRow("Apple Accelerate Framework", "%s",
                 (PSACFEnabled(network->acceleration) ? "yes" : "no"));
    printInfoRow("BLAS", "%s",
                 (PSBLASEnabled(network->acceleration) ? "yes" : "no"));
    if (PSLogColorEnabled()) printf(PSCOLOR_CYAN);
    printf("Layers:\n");
    if (PSLogColorEnabled()) printf(PSCOLOR_RESET);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        printf("  ");
        PSPrintLayerInfo(layer);
    }
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

int PSStateSequenceLength(PSLayer *layer) {
    if (layer == NULL || layer->states == NULL) return 0;
    int len = PSMatrixDim(layer->states, 0);
    if (len <= 0) return 0;
    if (layer->initial_states != NULL) len--;
    return len;
}

/* Initialize states for layer `layer` (WARN: it will just return states,
 * without actually setting them into layer->states, so, it can also be
 * used for states other than layer->states).
 * New states will be set to zero.
 * `seqlen`: new states sequence length.
 * `retain_previous`: if set to 1, last state from `current` will be
 *                    retained as layer's `initial_state`. This is can be
 *                    used by recurrent network is some cases.
 * `current`: current layer's states
 * `previous`: used to keep pointer to retained previous last state. */
PSMatrix initLayerStates(PSLayer *layer, uint32_t seqlen,
                         int retain_previous,
                         PSMatrix current, PSFloat **previous)
{
    assert(previous != NULL);
    /* If `retain_previous` is true, it means that the last vector in
     * current sequence states must be retained as 'initial state'.
     * This is done by adding one further row to states matrix that will
     * hold the 'initial' vector from previous last vector. */
    int cur_seqlen = PSStateSequenceLength(layer);
    if (cur_seqlen <= 0) retain_previous = 0;
    int nrows = seqlen + (retain_previous ? 1 : 0);
    PSMatrix states = PSMatrixZeros(2, nrows, layer->size);
    if (states == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    if (retain_previous && current != NULL) {
        int last_idx = cur_seqlen - 1;
        assert(last_idx >= 0);
        /* Last step from current states */
        PSFloat *last = PSMatrixGet(current, 1, NULL, last_idx);
        /* Initial step in new states (last row) */
        PSFloat *initial = PSMatrixGet(states, 1, NULL, nrows - 1);
        if (last == NULL || initial == NULL) {
            PSErr(
                NULL, "Could not initialize layer[%d] states (seqlen = %d)",
                layer->index, seqlen
            );
            return NULL;
        }
        PSVectorCopy(initial, last, layer->size);
        *previous = initial;
    } else *previous = NULL;
    return states;
}

/* Resize states sequence for layer `layer` (WARN: it will just return states,
 * without actually setting them into layer->states, so, it can also be
 * used for states other than layer->states).
 * `seqlen`: new states sequence length.
 * `current`: current layer's states
 * `previous`: used to keep pointer to retained previous last state. */
PSMatrix resizeLayerStates(PSLayer *layer, uint32_t seqlen,
                           PSMatrix current, PSFloat **previous)
{
    assert(previous != NULL);
    int nrows = seqlen;
    int cur_seqlen = PSStateSequenceLength(layer), cur_nrows = cur_seqlen;
    if (layer->initial_states != NULL) {
        /* If layer has initial states, they're located in the last sequence's
         * element: states[seqlen] (in this case, states matrix number of rows
         * is seqlen + 1. */
        nrows += 1;
        cur_nrows += 1;
    }
    int steps2add = nrows - cur_nrows;
    if (steps2add <= 0) {
        /* No action needed, just return current. */
        *previous = layer->initial_states;
        return current;
    }
    PSMatrix states = PSMatrixExpand(current, steps2add);
    if (states == NULL) {
        PSErr(NULL, "Layer[%d]: failed to resize states");
        return NULL;
    }
    /* If layer has `initial_states`, `last` will point to row in new states
     * containing values belonging to `initial_states` (its pointer could
     * differ from `initial_states` after resizing, since the matrix could have
     * been reallocated).
     * If layer has no `initial_states`, `last` will simply point to added
     * rows. */
    PSFloat *last = states + (cur_seqlen * layer->size);
    if (*previous != NULL) {
        /* Move initial state from index [cur_seqlen] to index[seqlen]:
         * `prev` points to last rows in new states (states[seqlen]) that
         * will contain values belonging to `initial_states` (pointed by
         * `last`). */
        PSFloat *prev = states + (seqlen * layer->size);
        PSVectorCopy(prev, last, layer->size);
        *previous = prev;
    }
    /* Set added rows to zero. */
    memset(last, 0, (size_t) steps2add * layer->size * sizeof(PSFloat));
    return states;
}

/* Initialize `layer` states with a sequence of `seqlen` rows set to zero.
 * `seqlen`: sequence length
 * `retain_previous`: if set to 1, last state from `current` will be
 *                    retained as layer's `initial_state`. This is can be
 *                    used by recurrent network is some cases.
 * Return value: 1 in case of success, 0 in case of failure. */
int PSInitLayerStates(PSLayer *layer, uint32_t seqlen, int retain_previous) {
    PSMatrix states = layer->states, hstates = NULL;
    int cur_seqlen = PSStateSequenceLength(layer);
    if (cur_seqlen <= 0 || states == NULL) retain_previous = 0;
    if (seqlen == 0 && !retain_previous) {
        layer->states = NULL;
        PSMatrixDelete(states);
        layer->initial_states = NULL;
        if (layer->on_states_init != NULL)
            if (!layer->on_states_init(layer, 0, 0)) goto err;
        return 1;
    }
    hstates = initLayerStates(
        layer, seqlen, retain_previous, states,
        &layer->initial_states
    );
    if (hstates == NULL) return 0;
    layer->states = hstates;
    PSMatrixDelete(states);
    hstates = NULL;
    if (layer->on_states_init != NULL) {
        if (!layer->on_states_init(layer, seqlen, retain_previous))
            goto err;
    }
    return 1;
err:
    if (hstates != NULL) PSMatrixDelete(hstates);
    if (layer->states != NULL) PSMatrixDelete(layer->states);
    layer->states = NULL;
    layer->network->status = STATUS_ERROR;
    return 0;
}

/* Resize states sequence for layer `layer`. The function also calls
 * `on_states_resize` callback if any, allowing different types of layer to
 * resize their own private data.
 * Arguments:
 * `seqlen`: new states sequence length.
 * Return value: 1 in case of success, 0 in case of failure. */
int PSResizeLayerStates(PSLayer *layer, uint32_t seqlen) {
    PSMatrix states = layer->states;
    int cur_seqlen = PSStateSequenceLength(layer);
    if (states == NULL || seqlen == 0 || cur_seqlen == 0)
        return PSInitLayerStates(layer, seqlen, 1);
    else if ((int) seqlen == cur_seqlen) return 1;
    else if ((int) seqlen < cur_seqlen) {
        PSErr(
            NULL, "Layer[%d]: Cannot resize hidden state to a smaller size: %d"
            " (current sequence length = %d)",
            layer->index, seqlen, cur_seqlen
        );
        return 0;
    }
    PSMatrix hstates = resizeLayerStates(
        layer, seqlen, states, &layer->initial_states
    );
    if (hstates == NULL) {
        PSMatrixDelete(states);
        layer->states = NULL;
        layer->initial_states = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    layer->states = hstates;
    if (layer->on_states_resize != NULL)
        if (!layer->on_states_resize(layer, seqlen)) return 0;
    return 1;
}

int PSResetLayerStateSequence(PSLayer *layer, uint32_t seqlen,
                              int retain_previous)
{
    if (layer == NULL) return 0;
    if (!PSUseSequences(layer)) return 0;
    return PSInitLayerStates(layer, seqlen, retain_previous);
}

int PSResetNetworkStateSequences(PSNeuralNetwork *network, uint32_t seqlen,
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
        if (!PSResetLayerStateSequence(layer, seqlen, retain_previous)) {
            PSErr(__func__, "Failed to reset states sequence on layer %d", i);
            return 0;
        }
    }
    return 1;
}

int PSBeforeSequenceFeedforward(PSLayer *layer, int seqlen, int t) {
    if (!PSUseSequences(layer)) return 1;
    if (seqlen < 1) {
        PSErr(__func__, "Layer[%d]: sequence length must be >= 1 (found %d)",
              layer->index, seqlen);
        return 0;
    }
    int cur_seqlen = PSStateSequenceLength(layer);
    if (PSIsRecurrent(layer) && t >= (int) cur_seqlen) {
        /* Recurrent layers may need to resize their sequence steps before
         * feedforward phase if step `t` is beyond current sequence length. */
        if (!PSResizeLayerStates(layer, t + 1)) {
            if (layer->network != NULL) layer->network->status = STATUS_ERROR;
            PSErr(
                NULL, "Could not resize recurrent hidden states for "
                "layer %d", layer->index
            );
            return 0;
        }
    } else if (PSHandleSequenceAtOnce(layer)) {
        /* Non-recurrent layers who handle sequences need to init their
         * states before feddforward step. */
        if (!PSInitLayerStates(layer, seqlen, 0)) {
            PSErr(
                NULL, "Could not init sequence states for "
                "layer %d", layer->index
            );
            return 0;
        }
    }
    return 1;
}

int PSSetState(PSLayer *layer, PSFloat state, int index, ...) {
    if (layer->states == NULL) {
        PSErr(__func__, "Layer %d: null states!", layer->index);
        return 0;
    }
    if (index >= layer->size) {
        PSErr(
            __func__, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0;
    }
    int t = 0;
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, index);
        t = va_arg(args, int);
        va_end(args);
        int seqlen = PSStateSequenceLength(layer);
        if (t >= (int) seqlen) {
            if (!PSResizeLayerStates(layer, t + 1)) {
                if (layer->network) layer->network->status = STATUS_ERROR;
                PSErr(
                    __func__, "Could not resize states sequence for "
                    "layer %d", layer->index
                );
                return 0;
            } else if (layer->states == NULL) return 0;
        } else if (t < 0) {
            PSErr(
                __func__,
                "Invalid sequence index %d for layer %d, neuron %d",
                layer->index, index
            );
            return 0;
        }
        index = (t * layer->size) + index;
    } else if (Dropout == layer->type) {
        PSErr(__func__, "PSSetState cannot be called on Dropout layer");
        return 0;
    }
    layer->states[index] = state;
    return 1;
}

int PSSetNeuronState(PSNeuron *neuron, double state, ...) {
    if (neuron->layer == NULL) return 0.0;
    PSFloat a = (PSFloat) state;
    if (PSUseSequences(neuron->layer)) {
        va_list args;
        va_start(args, state);
        int t = va_arg(args, int);
        va_end(args);
        return PSSetState(neuron->layer, a, neuron->index, t);
    }
    return PSSetState(neuron->layer, a, neuron->index);
}

PSFloat PSGetState(PSLayer *layer, int index, ...) {
    if (layer->states == NULL) return 0.0;
    if (index >= layer->size) {
        PSErr(
            __func__, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0.0;
    }
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, index);
        int t = va_arg(args, int);
        va_end(args);
        /* If t < 0, retrieve previous state, if any. */
        if (t < 0) {
            if (layer->initial_states == NULL) return 0.0;
            else return layer->initial_states[index];
        } else {
            int seqlen = PSStateSequenceLength(layer);
            if (t >= seqlen) {
                PSErr(
                    __func__, "Index %d is out-of-range: layer %d states "
                    "sequence has size: %d", t, layer->index, seqlen
                );
                return 0.0;
            }
            index = (t * layer->size) + index;
        }
    }
    return layer->states[index];
}

PSFloat *PSGetStates(PSLayer *layer, ...) {
    if (layer->states == NULL) return NULL;
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, layer);
        int t = va_arg(args, int);
        va_end(args);
        /* If t < 0, retrieve previous state, if any. */
        if (t < 0) return layer->initial_states;
        else {
            int seqlen = PSStateSequenceLength(layer);
            if (t >= seqlen) {
                PSErr(
                    __func__, "Index %d is out-of-range: layer %d states "
                    "sequence has size: %d", t, layer->index, seqlen
                );
                return NULL;
            }
            return layer->states + (t * layer->size);
        }
    }
    return layer->states;
}

PSFloat PSGetNeuronState(PSNeuron *neuron, ...) {
    if (neuron->layer == NULL) return 0.0;
    if (PSUseSequences(neuron->layer)) {
        va_list args;
        va_start(args, neuron);
        int t = va_arg(args, int);
        va_end(args);
        return PSGetState(neuron->layer, neuron->index, t);
    }
    return PSGetState(neuron->layer, neuron->index);
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
            } else if (Recurrent != type && LSTM != type && GRU != type) {
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
    if (network->layers == NULL) {
        PSErr(__func__, "Network has no layers");
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
    int first_whole_seq_layer = -1, i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) {
            PSErr(__func__, "Layer[%d] is null", i);
            return 0;
        }
        if (PSHandleSequenceAtOnce(layer)) {
            if (first_whole_seq_layer < 0) first_whole_seq_layer = i;
            network->flags |= FLAG_USE_SEQUENCES;
        }
    }
    int is_recurrent = PSIsRecurrent(network);
    PSRecurrentNetworkMode mode = PSGetRecurrentNetworkMode(network);
    PSLayer *input_layer = network->layers[0],
            *output_layer = network->layers[network->size - 1];
    if (is_recurrent) {
        if (first_whole_seq_layer >= 0) {
            PSErr(__func__, "Network has both recurrent layers and whole "
                  "sequence layers");
            return 0;
        }
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
    network->acceleration = PSGlobalAcceleration;
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
    if (clone == NULL) goto memerr;
    if (!layout_only) {
        clone->status = network->status;
        if (network->training != NULL) {
            clone->training = malloc(sizeof(PSTrainingInfo));
            if (clone->training == NULL) goto memerr;
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
    clone->acceleration = network->acceleration;
    clone->loss = network->loss;
    if (network->rnn_options != NULL) {
        clone->rnn_options = malloc(sizeof(PSRecurrentNetworkOptions));
        if (clone->rnn_options == NULL) goto memerr;
        memcpy(
            clone->rnn_options, network->rnn_options,
            sizeof(PSRecurrentNetworkOptions)
        );
    }

    int i, j;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType type = layer->type;
        PSLayerDef ldef = {
            .activation = layer->activate,
            .flags = layer->flags,
            .output_depth = layer->output_depth,
            .output_columns = layer->output_columns,
            .output_rows = layer->output_rows
        };
        if (Dropout == type) ldef.dropout = PSGetDropout(layer);
        if (Convolutional == type || Pooling == type) {
            PSConvolutionalSettings *csettings =
                PSGetConvolutionalSettings(layer);
            ldef.stride = csettings->stride;
            ldef.padding = csettings->padding;
            ldef.filter_width = csettings->filter_width;
            ldef.filter_height = csettings->filter_height;
        }
        ldef.pretrained = layer->pretrained;
        PSLayer *cloned_layer = PSAddLayer(clone, type, layer->size, &ldef);
        if (cloned_layer == NULL) {
            PSDeleteNetwork(clone);
            return NULL;
        }
        cloned_layer->flags = layer->flags;
        if (!layout_only) {
            if (cloned_layer->states != NULL) {
                PSMatrixDelete(cloned_layer->states);
                cloned_layer->states = NULL;
            }
            if (layer->states != NULL) {
                cloned_layer->states = PSMatrixDup(layer->states);
                if (cloned_layer->states == NULL) goto memerr;
                if (layer->initial_states != NULL) {
                    int diff = layer->initial_states -
                               layer->states;
                    cloned_layer->initial_states =
                        cloned_layer->states + diff;
                }
            } else {
                cloned_layer->states = NULL;
                cloned_layer->initial_states = NULL;
            }
            if (layer->weights != NULL) {
                if (layer->weight_types_count == 0) {
                    PSErr(__func__, "Layer[%d]: weights not NULL but "
                         "weight_types_count is 0", layer->index);
                    goto err;
                }
                if (cloned_layer->weights == NULL) {
                    cloned_layer->weights = calloc(
                        layer->weight_types_count, sizeof(PSMatrix)
                    );
                    if (cloned_layer->weights == NULL) goto memerr;
                }
                for (j = 0; j < layer->weight_types_count; j++) {
                    if (layer->weights[j] == NULL) {
                        PSErr(__func__, "Layer[%d]: weights[%d] are NULL",
                             layer->index, j);
                        goto err;
                    }
                    if (cloned_layer->weights[j] != NULL)
                        PSMatrixDelete(cloned_layer->weights[j]);
                    cloned_layer->weights[j] = PSMatrixDup(layer->weights[j]);
                    if (cloned_layer->weights[j] == NULL) goto memerr;
                }
            } else if (cloned_layer->weights != NULL) {
                for (j = 0; j < layer->weight_types_count; j++)
                    PSMatrixDelete(cloned_layer->weights[j]);
                free(cloned_layer->weights);
                cloned_layer->weights = NULL;
            }
            if (layer->biases != NULL && !(layer->flags & FLAG_NO_BIAS)) {
                uint64_t bias_count = PSGetLayerParametersCount(
                    layer, PARAM_TYPE_BIAS
                );
                if (bias_count == 0) {
                    PSErr(__func__, "Layer[%d] bias count is zero",
                          layer->index);
                    goto err;
                }
                size_t bias_size = (bias_count * sizeof(PSFloat));
                if (cloned_layer->biases == NULL) {
                    cloned_layer->biases = malloc(bias_size);
                    if (cloned_layer->biases == NULL) goto memerr;
                }
                memcpy(cloned_layer->biases, layer->biases, bias_size);
            }
            if (layer->on_copy != NULL) {
                if (!layer->on_copy(cloned_layer, layer)) goto err;
            } else {
                for (j = 0; j < layer->size; j++) {
                    PSNeuron *clone_n = cloned_layer->neurons[j];
                    /* if (Pooling == type) continue; */
                    if (cloned_layer->biases != NULL) {
                        clone_n->bias = cloned_layer->biases + j;
                        cloned_layer->biases[j] = layer->biases[j];
                    } else clone_n->bias = NULL;
                    PSMatrix clone_weights = NULL;
                    if (cloned_layer->weights != NULL)
                        clone_weights = cloned_layer->weights[0];
                    if (clone_weights != NULL) {
                        clone_n->weights = clone_weights +
                                           (j * cloned_layer->size);
                    } else clone_n->weights = NULL;
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
            PSGradient **mgradients = training_ctx->memory_gradients1;
            PSGradient **xgradients = training_ctx->memory_gradients2;
            if (mgradients != NULL) {
                clone_training_ctx->memory_gradients1 = cloneNetworkGradients(
                    mgradients, network
                );
                if (clone_training_ctx->memory_gradients1 == NULL) goto err;
            } else {
                if (clone_training_ctx->memory_gradients1) {
                    PSDeleteNetworkGradients(
                        clone_training_ctx->memory_gradients1, network
                    );
                }
                clone_training_ctx->memory_gradients1 = NULL;
            }
            if (xgradients != NULL) {
                clone_training_ctx->memory_gradients2 = cloneNetworkGradients(
                    xgradients, network
                );
                if (clone_training_ctx->memory_gradients2 == NULL) goto err;
            } else {
                if (clone_training_ctx->memory_gradients2) {
                    PSDeleteNetworkGradients(
                        clone_training_ctx->memory_gradients2, network
                    );
                }
                clone_training_ctx->memory_gradients2 = NULL;
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
memerr:
    PSPrintMemoryErrorMsg();
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
    fprintf(dump_file, "layer:index=%d,type=%s,size=%d", layer->index,
        type_name, layer->size);
    int onehot_input = (layer->index == 0 && layer->flags & FLAG_ONEHOT);
    if (onehot_input)
        fprintf(dump_file, ",vector_size=%d", layer->onehot_vector_size);
    if (PSIsRecurrent(layer) && PSIsRecurrent(layer->network))
        fprintf(dump_file, ",recurrent=1");
    if (ltype == Convolutional || ltype == Pooling) {
        PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
        int output_w = layer->output_columns, output_h = layer->output_rows,
            depth = layer->output_depth, filter_w = 0, filter_h = 0,
            input_w = 0, input_h = 0, stride = 0, padding = 0;
        if (settings != NULL) {
            input_w = settings->input_width;
            input_h = settings->input_height;
            filter_w = settings->filter_width;
            filter_h = settings->filter_height;
            stride = settings->stride;
            padding = settings->padding;
        }
        if (stride <= 0 && ltype == Pooling) stride = filter_w;
        fprintf(
            dump_file, ",input_size=%dx%d,output_size=%dx%d,features=%d"
            ",region=%dx%d,stride=%d",
            input_w, input_h, output_w, output_h, depth,
            filter_w, filter_h, stride
        );
        if (ltype == Convolutional) {
            if (padding < 0) padding = 0;
            fprintf(dump_file, ",padding=%d", padding);
        }
    } else if (ltype == FullyConnected && layer->output_depth > 1) {
        fprintf(dump_file, ",depth=%d", layer->output_depth);
    }
    const char *activation = PSGetActivationName(layer->activate);
    if (activation != NULL) fprintf(dump_file, ",activation=%s", activation);
    if (add_new_line) fprintf(dump_file, "\n");
}

int PSDumpNetworkStates(PSNeuralNetwork *network, const char* filename) {
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
        int is_recurrent = PSIsRecurrent(layer),
            has_seq = PSUseSequences(layer), seqlen = 0;
        int nidx = 0, t = 0;
        DumpLayerInfo(layer, f, 0);
        if (has_seq) {
            seqlen = PSStateSequenceLength(layer);
            fprintf(
                f, ",%s=%d", (is_recurrent ? "timesteps" : "sequence_length"),
                seqlen
            );
        }
        fprintf(f, ",activations=(");
        for(; nidx < layer->size; nidx++) {
            PSNeuron *n = layer->neurons[nidx];
            if (n == NULL) {
                PSErr(__func__, "Layer[%d] Neuron[%s] is null", i, nidx);
                return 0;
            }
            if (!has_seq) {
                if (nidx > 0) fprintf(f, ",");
                writeSerializedFloat(f, PSGetState(layer, nidx), opts);
            } else {
                for (t = 0; t < seqlen; t++) {
                    if (nidx > 0 || t > 0) fprintf(f, ",");
                    writeSerializedFloat(
                        f, PSGetState(layer, nidx, t), opts
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
        PSMatrix delta = layer->delta;
        if (delta == NULL) {
            fprintf(f, ",deltas=()\n");
            continue;
        }
        fprintf(f, ",deltas=(");
        int len = PSMatrixLength(delta);
        for(; nidx < len; nidx++) {
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
    if (training_ctx->memory_gradients1 != NULL)
        PSDeleteNetworkGradients(training_ctx->memory_gradients1, network);
    if (training_ctx->memory_gradients2 != NULL)
        PSDeleteNetworkGradients(training_ctx->memory_gradients2, network);
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
    if (network == NULL) return;
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

void PSDeleteNeuron(PSNeuron *neuron) {
    if (neuron == NULL) return;
    if (neuron->extra != NULL) {
        /* TODO: extra currently unused for neurons */
        free(neuron->extra);
    }
    free(neuron);
}

void PSSetDefaultLayerDef(PSLayerDef *ldef, PSLayerType type) {
    memset(ldef, 0, sizeof(*ldef));
    UNUSED(type);
    /* TODO: use specific settings for type? */
}

PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale)
{
    static PSLayerDef default_def = {0};
    if (ldef == NULL) ldef = &default_def;
    PSMatrix weights = NULL;
    if (ldef->weight_init_mode == INIT_MODE_ZERO)
        weights = PSMatrixZeros(2, rows, columns);
    else if (ldef->weight_init_mode == INIT_MODE_VALUE) {
        PSFloat init_value = ldef->init_value;
        weights = PSMatrixCreate(init_value, NULL, 2, rows, columns);
    } else {
        if (ldef->weight_init_mode == INIT_MODE_RAND) {
            range = ldef->init_range;
            scale = ldef->init_scale;
        }
        if (range == 0) range = 1;
        weights = PSMatrixWithGaussianRandom(range, 2, rows, columns);
        if (scale > 0 && weights != NULL) {
            int acceleration = PSGlobalAcceleration;
            if (layer != NULL && layer->network != NULL)
                acceleration = layer->network->acceleration;
            PSMathOpts opts = {.acceleration = acceleration};
            PSMultiplyVectorScalar(
                weights, scale, weights, (rows * columns), &opts
            );
        }
    }
    return weights;
}

PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale)
{
    static PSLayerDef default_def = {0};
    if (ldef == NULL) ldef = &default_def;
    int mode = INIT_MODE_AUTO;
    if (param_type == PARAM_TYPE_BIAS) mode = ldef->bias_init_mode;
    else mode = ldef->weight_init_mode;
    PSFloat param;
    if (mode == INIT_MODE_ZERO) param = 0.0;
    else {
        if (mode == INIT_MODE_RAND) {
            range = ldef->init_range;
            scale = ldef->init_scale;
        }
        if (range == 0) range = 1;
        param = PSGaussianRandom(0, range);
        if (scale) param *= scale;
    }
    return param;
}

int initGenericLayer(PSLayer *layer, int size, int previous_size,
                     PSLayerDef *ldef)
{
    if (layer == NULL) return 0;
    if (layer->network == NULL) {
        PSErr(NULL, "Layer[%d]: missing network");
        goto fail;
    }
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->states = PSMatrixZeros(2, 1, size);
    if (layer->states == NULL) goto memerr;
    PSMatrix weights = NULL;
    if (layer->index > 0 && previous_size > 0) {
        layer->weights = calloc(1, sizeof(PSMatrix));
        if (layer->weights == NULL) goto memerr;
        layer->weights[0] = PSInitWeights(
            layer, size, previous_size, ldef, 1.0, 0
        );
        if (layer->weights[0] == NULL) goto memerr;
        layer->weight_types_count = 1;
        layer->biases = malloc(size * sizeof(PSFloat));
        if (layer->biases == NULL) goto memerr;
        weights = layer->weights[0];
    }
    int i;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) goto memerr;
        neuron->index = i;
        neuron->extra = NULL;
        if (layer->index > 0 && previous_size > 0) {
            neuron->bias = layer->biases + i;
            *(neuron->bias) = PSInitParam(PARAM_TYPE_BIAS, ldef, 1.0, 0.0);
            neuron->weights = weights + (i * previous_size);
        } else {
            neuron->bias = NULL;
            neuron->weights = NULL;
        }
        neuron->layer = layer;
        layer->neurons[i] = neuron;
    }
    if (layer->type != SoftMax) {
        if (layer->activate == NULL) {
            layer->activate = PSSigmoid;
            layer->derivative = PSSigmoidDerivative;
        }
        layer->feedforward = PSFullFeedforward;
        layer->backprop = PSFullBackprop;
    } else {
        layer->activate = NULL;
        layer->derivative = NULL;
        layer->feedforward = softmaxFeedforward;
        layer->backprop = NULL; /* Softmax layer should always be output
                                 * layer. */
        layer->network->loss = PSCrossEntropyLoss;
    }
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
fail:
    return 0;
}

PSLayer *PSAddLayer(PSNeuralNetwork *network, PSLayerType type, int size,
                     PSLayerDef *layer_def)
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
    int verbose = (PSLogLevel == PSLOGLEVEL_DEBUG);
    PSLayerDef default_def = {0};
    if (layer_def == NULL) {
        PSSetDefaultLayerDef(&default_def, type);
        layer_def = &default_def;
    }
    layer->network = network;
    layer->index = network->size++;
    layer->type = type;
    layer->size = size;
    layer->extra = NULL;
    layer->flags = layer_def->flags;
    layer->neurons = NULL;
    layer->delta = NULL;
    layer->states = NULL;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->initial_states = NULL;
    layer->activate = layer_def->activation;
    layer->derivative = PSGetActivationDerivative(layer->activate);
    layer->on_delete = NULL;
    layer->on_copy = NULL;
    layer->get_param_count = NULL;
    layer->weight_types_count = 0;
    layer->onehot_vector_size = size;
    layer->output_depth = layer_def->output_depth;
    layer->pretrained = layer_def->pretrained;
    layer->pretrainer = NULL;
    layer->pretrain = NULL;
    layer->private = NULL;
    layer->before_batch_training = NULL;
    layer->on_states_init = NULL;
    layer->on_states_resize = NULL;
    if (layer->output_depth <= 0) layer->output_depth = 1;
    layer->output_columns = layer_def->output_columns;
    layer->output_rows = layer_def->output_rows;
    if (layer->output_columns < 0) layer->output_columns = 0;
    if (layer->output_rows < 0) layer->output_rows = 0;
    PSLayer *previous = NULL;
    int previous_size = 0;
    int initialized = 0;
    if (verbose) printf("Adding layer %d\n", layer->index);
    if (network->layers == NULL) {
        network->layers = malloc(sizeof(PSLayer*));
        if (network->layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Could not allocate network layers!");
            return NULL;
        }
        if (network->flags & FLAG_ONEHOT) {
            layer->flags |= FLAG_ONEHOT;
            layer->onehot_vector_size = size;
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
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT)
            previous_size = previous->onehot_vector_size;
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
        initialized = initGenericLayer(layer, size, previous_size, layer_def);
    } else if (type == Convolutional) {
        initialized = PSInitConvolutionalLayer(network, layer, layer_def);
        /* TODO: Make PSCrossEntropyLoss default also for convolutional? */
    } else if (type == Pooling) {
        initialized = PSInitPoolingLayer(network, layer, layer_def);
    } else if (type == Recurrent) {
        initialized = PSInitRecurrentLayer(network, layer, size, previous_size,
                                           layer_def);
    } else if (type == LSTM) {
        initialized = PSInitLSTMLayer(network, layer, size, previous_size,
                                      layer_def);
    } else if (type == GRU) {
        initialized = PSInitGRULayer(network, layer, size, previous_size,
                                     layer_def);
    } else if (type == Dropout) {
        initialized = PSInitDropoutLayer(network, layer, layer_def);
    } else if (type == Embedding) {
        initialized = PSInitEmbeddingLayer(layer, size, previous_size,
                                           layer_def);
    } else if (type == Normalization) {
        initialized = PSInitNormalizationLayer(layer, layer_def);
    } else PSErr(__func__, "Invalid layer type %d", type);
    if (!initialized) goto fail;
    if (layer->index > 0 && layer->delta == NULL) {
        layer->delta = PSMatrixZeros(2, 1, layer->size);
        if (layer->delta == NULL) goto fail;
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
                __func__, "Could not set default recurrent mode for layer %d",
                layer->index
            );
            return NULL;
        }
    }
    if (layer_def->load_from != NULL) {
        int loaded = PSLoadLayer(layer, layer_def->load_from);
        if (!loaded) {
            PSAbortLayer(network, layer);
            PSErr(
                __func__, "Could load parameters for layer %d from '%s'",
                layer->index, layer_def->load_from
            );
            return NULL;
        }
    }
    if (layer->flags & FLAG_USE_SEQUENCES)
        network->flags |= FLAG_USE_SEQUENCES;
    if (verbose) PSPrintLayerInfo(layer);
    return layer;
fail:
    if (layer != NULL) {
        PSErr(
            __func__, "Could not initialize layer %d on network '%s'",
            layer->index, network->name
        );
        PSAbortLayer(network, layer);
    }
    return NULL;
}

PSLayer *PSAddConvolutionalLayer(PSNeuralNetwork *network, PSLayerDef *ldef) {
    return PSAddLayer(network, Convolutional, 0, ldef);
}

PSLayer *PSAddPoolingLayer(PSNeuralNetwork *network, PSLayerDef *ldef) {
    return PSAddLayer(network, Pooling, 0, ldef);
}

void PSDeleteLayer(PSLayer* layer) {
    if (layer == NULL) return;
    int size = layer->size, i;
    if (layer->neurons == NULL) size = 0;
    for (i = 0; i < size; i++) {
        PSNeuron* neuron = layer->neurons[i];
        if (neuron == NULL) continue;
        if (layer->type != Convolutional) PSDeleteNeuron(neuron);
        else free(neuron); /* TODO: Why? */
        layer->neurons[i] = NULL;
    }
    if (layer->neurons != NULL) free(layer->neurons);
    if (layer->weights != NULL) {
        for (i = 0; i < layer->weight_types_count; i++) {
            PSMatrix weights = layer->weights[i];
            if (weights != NULL) PSMatrixDelete(weights);
        }
        free(layer->weights);
    }
    if (layer->biases != NULL) free(layer->biases);
    if (layer->on_delete != NULL) layer->on_delete(layer);
    void *extra = layer->extra;
    if (extra != NULL) free(layer->extra);
    if (layer->delta != NULL) PSMatrixDelete(layer->delta);
    if (layer->states != NULL) PSMatrixDelete(layer->states);
    if (layer->pretrainer != NULL) PSDeleteNetwork(layer->pretrainer);
    free(layer);
}

int inputLayerFeedforward(PSNeuralNetwork *network, PSFloat *inputs, ...) {
    PSLayer *first = network->layers[0];
    int input_size = first->size, seqlen = 1, t = 0;
    int is_recurrent = PSIsRecurrent(first),
        seq_at_once = PSHandleSequenceAtOnce(first);
    size_t len = input_size;
    if (is_recurrent || seq_at_once) {
        va_list ap;
        va_start(ap, inputs);
        seqlen = va_arg(ap, int);
        if (is_recurrent) t = va_arg(ap, int);
        else if (seq_at_once) {
            input_size *= seqlen;
            t = 0;
        }
        va_end(ap);
        assert(seqlen > 0);
        assert(t >= 0);
        if (!PSBeforeSequenceFeedforward(first, seqlen, t)) return 0;
    }
    PSFloat *states = PSGetStates(first, t);
    if (states == NULL) {
        PSErr(NULL, "Layer[%d] missing states");
        return 0;
    }
    PSVectorCopy(states, inputs, len);
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
            ok = layer->feedforward(layer, timesteps, t);
            if (!ok) return 0;
        }
        if (last_recurrent == NULL && last_layer != NULL)
            setNetworkContext(network, last_recurrent_layer, last_layer);
        if (values != NULL) values += input_size;
        if (variable_timesteps && eos >= 0 && last_layer != NULL) {
            int max_idx = -1;
            if (!PSFindLayerMaxState(last_layer, NULL, &max_idx, t))
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
        ok = 1, seqlen = 0, first_idx = 0, output_idx = network->size - 1,
        input_is_seq = 0;
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
            /* Read seqlen from first element in `values`. */
            seqlen = (int) values[0];
        } else if (backprop && PSIsRecurrent(output_layer)) {
            /* If feedforward is called from backprop and network has
             * recurrent output but non-recurrent input (OneToMany),
             * we should read the seqlen that are the first element
             * of labels (y) that follows values. */
            PSFloat *y = values + network->input_size;
            seqlen = y[0];
        }
        int steps = seqlen;
        if (steps < 0) steps = 0;
        if (!PSResetNetworkStateSequences(network, steps, retain_previous)) {
            network->status = STATUS_ERROR;
            PSErr(NULL, "Failed to reset network recurrent states");
            return 0;
        }
    } else if (input_layer->flags & FLAG_USE_SEQUENCES) {
        input_is_seq = 1;
        seqlen = (int) values[0];
    }
    if (recurrent_input) {
        if (seqlen <= 0) {
            PSErr(
                __func__, "Recurrent sequence length must be > 0 (found %d)",
                seqlen
            );
            return 0;
        }
        ok = feedforwardThroughTime(network, values + 1, seqlen);
        if (!ok) return 0;
        if (last_recurrent == NULL && PSIsRecurrent(output_layer)){
            setNetworkContext(network, last_recurrent_layer, output_layer);
            return ok;
        } else if (last_recurrent != NULL) {
            int last_recurrent_idx = last_recurrent->index;
            if (last_recurrent_idx >= output_idx) return ok;
            else first_idx = last_recurrent_idx;
        }
    } else if (input_is_seq) {
        if (seqlen <= 0) {
            PSErr(
                __func__, "Sequence length must be > 0 (found %d)",
                seqlen
            );
            return 0;
        }
        ok = inputLayerFeedforward(network, values, seqlen);
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
            return feedforwardThroughTime(network, NULL, seqlen);
        if (layer->feedforward == NULL) {
            PSErr(__func__, "Layer %d feedforward function is NULL", i);
            return 0;
        }
        ok = layer->feedforward(layer);
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
    if (layer->type == Pooling || layer->type == Dropout) return NULL;
    PSGradient *gradients = malloc(sizeof(*gradients));
    if (gradients == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    gradients->bias_count = 0;
    gradients->weight_count = 0;
    gradients->biases = NULL;
    gradients->weights = NULL;
    gradients->tmp = NULL;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    uint64_t bias_count = 0, weights_count = 0, max_count = 0;
    if (use_bias)
        bias_count = PSGetLayerParametersCount(layer, PARAM_TYPE_BIAS);
    weights_count = PSGetLayerParametersCount(layer, PARAM_TYPE_WEIGHT);
    if (bias_count > 0) {
        gradients->biases = calloc(bias_count, sizeof(PSFloat));
        if (gradients->biases == NULL) {
            PSPrintMemoryErrorMsg();
            PSDeleteGradients(gradients);
            return NULL;
        }
        gradients->bias_count = bias_count;
        max_count = bias_count;
    }
    if (weights_count > 0) {
        gradients->weights = calloc(weights_count, sizeof(PSFloat));
        if (gradients->weights == NULL) {
            PSPrintMemoryErrorMsg();
            PSDeleteGradients(gradients);
            return NULL;
        }
        gradients->weight_count = weights_count;
        if (weights_count > max_count) max_count = weights_count;
    }
    if (max_count > 0) {
        gradients->tmp = malloc(max_count * sizeof(PSFloat));
    }
    return gradients;
}

int PSClassify(PSNeuralNetwork *network, PSFloat *inputs) {
    int ok = PSFeedforward(network, inputs);
    if (!ok) {
        PSErr(__func__, "Feedforward failed");
        return -1;
    };
    PSLayer *out = PSGetOutputLayer(network);
    int max_idx = 0, seqlen = PSStateSequenceLength(out),
        t = (int) seqlen - 1;
    if (t < 0) t = 0;
    if (!PSFindLayerMaxState(out, NULL, &max_idx, t)) {
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
        if (gradients[idx] == NULL && layer->type != Pooling &&
            layer->type != Dropout)
        {
            PSPrintMemoryErrorMsg();
            PSDeleteNetworkGradients(gradients, network);
            return NULL;
        }
    }
    return gradients;
}

PSGradient *cloneGradient(PSGradient *gradient) {
    if (gradient == NULL) return NULL;
    PSGradient *clone = calloc(1, sizeof(*clone));
    if (clone == NULL) goto memerr;
    clone->bias_count = gradient->bias_count;
    clone->weight_count = gradient->weight_count;
    clone->tmp = NULL;
    if (gradient->biases != NULL) {
        size_t bias_size = gradient->bias_count * sizeof(PSFloat);
        clone->biases = malloc(bias_size);
        if (clone->biases == NULL) goto memerr;
        memcpy(clone->biases, gradient->biases, bias_size);
    }
    if (gradient->weights != NULL) {
        size_t wsize = gradient->weight_count * sizeof(PSFloat);
        clone->weights = malloc(wsize);
        if (clone->weights == NULL) goto memerr;
        memcpy(clone->weights, gradient->weights, wsize);
    }
    return clone;
memerr:
    PSPrintMemoryErrorMsg();
    if (clone != NULL) PSDeleteGradients(clone);
    return NULL;
}

int copyGradient(PSGradient *dst, PSGradient *src) {
    if (src == NULL && dst != NULL) return 0;
    if (src != NULL && dst == NULL) return 0;
    if (src == NULL && dst == NULL) return 1;
    dst->bias_count = src->bias_count;
    dst->weight_count = src->weight_count;
    if (src->biases != NULL) {
        size_t bias_size = src->bias_count * sizeof(PSFloat);
        if (dst->biases == NULL) {
            dst->biases = malloc(bias_size);
            if (dst->biases == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        memcpy(dst->biases, src->biases, bias_size);
    } else if (dst->biases != NULL) {
        free(dst->biases);
        dst->biases = NULL;
    }
    if (src->weights != NULL) {
        size_t weight_size = src->weight_count * sizeof(PSFloat);
        if (dst->weights == NULL) {
            dst->weights = malloc(weight_size);
            if (dst->weights == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        memcpy(dst->weights, src->weights, weight_size);
    } else if (dst->weights != NULL) {
        free(dst->weights);
        dst->weights = NULL;
    }
    return 1;
}

PSGradient **cloneNetworkGradients(PSGradient **gradients,
                                   PSNeuralNetwork *network)
{
    if (gradients == NULL) return NULL;
    PSGradient **clone = createGradients(network);
    if (clone == NULL) return NULL;
    for (int i = 1; i < network->size; i++) {
        int idx = i - 1;
        PSGradient *lgradients = gradients[idx];
        PSGradient *clone_lgradients = clone[idx];
        if (lgradients == NULL) continue;
        if (!copyGradient(clone_lgradients, lgradients)) {
            PSErr(NULL, "Failed to clone gradients for layer %d", i);
            goto err;
        }
    }
    return clone;
err:
    if (clone != NULL) PSDeleteNetworkGradients(clone, network);
    return NULL;
}

void PSDeleteGradients(PSGradient *gradients) {
    if (gradients == NULL) return;
    free(gradients->biases);
    free(gradients->weights);
    free(gradients->tmp);
    free(gradients);
}

void PSDeleteNetworkGradients(PSGradient **gradients, PSNeuralNetwork *network)
{
    if (gradients == NULL) return;
    int i;
    for (i = 1; i < network->size; i++) {
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        PSDeleteGradients(lgradients);
    }
    free(gradients);
}

static int resetLayerDeltas(PSLayer *layer, int full_reset) {
    if (layer->delta == NULL) return 1;
    int handle_seq = PSHandleSequenceAtOnce(layer);
    if (!full_reset && PSMatrixDim(layer->delta, 1) > layer->size) {
        /* Delta contains data for differente stuff (ie. LSTM layer
         * allocate layer->size * 2 delta in order to store delta
         * for their raw states.
         * In this case, reset is performed only on first N
         * (where N=layer->size) values. */
         assert(!handle_seq);
         int rows = PSMatrixDim(layer->delta, 0);
         for (int i = 0; i < rows; i++) {
            PSFloat *row = PSMatrixGet(layer->delta, 1, NULL, i);
            PSVectorClear(row, layer->size);
         }
         return 1;
    }
    if (handle_seq) {
        int seqlen = PSStateSequenceLength(layer);
        int delta_seqlen = PSMatrixDim(layer->delta, 0);
        if (seqlen < 1) seqlen = 1;
        if (seqlen != delta_seqlen) {
            PSMatrixDelete(layer->delta);
            layer->delta = PSMatrixZeros(2, seqlen, layer->size);
            return layer->delta != NULL;
        }
    }
    PSMatrixClear(layer->delta);
    return 1;
}

static int resetDeltas(PSNeuralNetwork *network) {
    if (network->layers == NULL) return 0;
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) continue;
        if (!resetLayerDeltas(layer, 1)) return 0;
    }
    return 1;
}

void beforeBatchTraining(PSNeuralNetwork *network) {
    if (network == NULL || network->layers == NULL) return;
    int i, j;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) continue;
        if (layer->weights != NULL) {
            for (j = 0; j < layer->weight_types_count; j++) {
                PSMatrix weights = layer->weights[j];
                PSMatrixResetTransposed(weights);
            }
        }
        if (layer->before_batch_training != NULL)
            layer->before_batch_training(layer);
    }
}

void PSResetTransposedWeights(PSNeuralNetwork *network) {
    if (network == NULL || network->layers == NULL) return;
    int i, j;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) continue;
        if (layer->weights == NULL) continue;
        for (j = 0; j < layer->weight_types_count; j++) {
            PSMatrix weights = layer->weights[j];
            PSMatrixResetTransposed(weights);
        }
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
    if (tctx->memory_gradients1 != NULL) {
        count++;
        if (mg1 != NULL) *mg1 = tctx->memory_gradients1;
    }
    if (tctx->memory_gradients2 != NULL) {
        count++;
        if (mg2 != NULL) *mg2 = tctx->memory_gradients2;
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
        ctx->training_context->memory_gradients1 = createGradients(network);
        if (ctx->training_context->memory_gradients1 == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        if (mem_gradients_count < 2) goto final;
        ctx->training_context->memory_gradients2 = createGradients(network);
        if (ctx->training_context->memory_gradients2 == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
final:
    return 1;
}

int PSApplyDerivative(PSActivationFunction derivative, PSFloat *delta,
                      PSFloat *outputs, int size, PSMathOpts *opts)
{
    PSFloat *deriv = calloc(size, sizeof(PSFloat));
    if (deriv == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSMathOpts mopts = {.acceleration = PSGlobalAcceleration};
    if (opts != NULL) mopts.acceleration = opts->acceleration;
    derivative(outputs, deriv, size, &mopts);
    PSMultiplyVectors(delta, deriv, delta, size, &mopts);
    free(deriv);
    return 1;
}

int PSBeforeLayerBackprop(PSLayer *layer, PSLayer *previous, int *step,
                          int *seqlen, PSFloat **outputs, PSFloat **inputs,
                          va_list args)
{
    assert(inputs != NULL);
    assert(outputs != NULL);
    int is_recurrent = PSIsRecurrent(layer),
        prev_is_recurrent = PSIsRecurrent(previous),
        handle_seq = PSHandleSequenceAtOnce(layer),
        t = 0, prev_t = 0, slen = 1;
    *outputs = NULL;
    *inputs = NULL;
    if (is_recurrent) {
        t = va_arg(args, int);
        if (prev_is_recurrent) prev_t = t;
    } else if (prev_is_recurrent) {
        prev_t = (int) PSStateSequenceLength(previous) - 1;
        if (prev_t < 0) {
            PSErr(
                NULL, "Recurrent layer %d has no hidden states",
                previous->index
            );
            return 0;
        }
    } else if (handle_seq) {
        slen = PSStateSequenceLength(layer);
        assert(slen > 0);
        *outputs = layer->states;
        *inputs = previous->states;
        if (PSMatrixDim(layer->delta, 0) != slen)
            if (!resetLayerDeltas(layer, 1)) return 0;
    }
    if (*outputs == NULL && !handle_seq) *outputs = PSGetStates(layer, t);
    if (*inputs == NULL && !handle_seq)
        *inputs = PSGetStates(previous, prev_t);
    if (*outputs == NULL) {
        PSErr(NULL, "Layer[%d]: NULL outputs", layer->index);
        return 0;
    }
    if (*inputs == NULL) {
        PSErr(NULL, "Layer[%d]: NULL inputs", previous->index);
        return 0;
    }
    if (seqlen != NULL) *seqlen = slen;
    if (step != NULL) *step = t;
    return 1;
}

void PSUpdateGradient(PSGradient *gradient, PSMatrix inputs, PSLayer *layer,
                      PSLayer *previous, int use_bias, int seqlen)
{
    if (seqlen < 1) seqlen = 1;
    else if (!PSHandleSequenceAtOnce(layer)) seqlen = 1;
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    PSFloat *delta_p = layer->delta, *input_p = inputs;
    for (int i = 0; i < seqlen; i++) {
        opts.store_mode = PS_STORE_MODE_ADD;
        PSOuterProduct(
            delta_p, input_p, gradient->weights,
            layer->size, previous->size, &opts
        );
        if (use_bias) {
            opts.store_mode = PS_STORE_MODE_SET;
            PSSumVectors(
                delta_p, gradient->biases, gradient->biases, layer->size, &opts
            );
        };
        delta_p += layer->size;
        input_p += layer->size;
    }
}

int PSUpdatePreviousLayerDelta(PSLayer *layer, PSLayer *previous,
                              int weights_index, int seqlen)
{
    if (layer->weights == NULL ||
        weights_index >= layer->weight_types_count ||
        layer->weights[weights_index] == NULL)
    {
        PSErr(NULL, "Layer[%d] NULL weights", layer->index);
        return 0;
    }
    int ok;
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    PSMatrix weights = layer->weights[weights_index];
    PSMatrix delta = layer->delta;
    opts.store_mode = PS_STORE_MODE_ADD;
    if (seqlen < 1 || !PSHandleSequenceAtOnce(layer)) seqlen = 1;
    if (seqlen == 1) {
        opts.transpose = 1; /* Transpose weights */
        ok = PSDotMV(weights, delta, previous->delta, &opts);
    } else {
        ok = PSDot(delta, weights, previous->delta, &opts);
    }
    if (!ok)
        PSErr(NULL, "Layer[%d]: failed backprop (PSDot)", layer->index);
    return ok;
}

int softmaxLayerBackprop(PSLayer *layer, PSLayer *previous_layer, PSFloat *y,
                         PSGradient *gradient, ...)
{
    PSNeuralNetwork *network = layer->network;
    assert(layer->type == SoftMax);
    int t = 0, ok = 1, handle_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 1;
    int apply_derivative = outputDerivativeNeeded(network);
    int onehot = (layer->flags & FLAG_ONEHOT);
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSFloat *outputs = NULL, *inputs = NULL;
    va_list args;
    va_start(args, gradient);
    ok = PSBeforeLayerBackprop(layer, previous_layer, &t, &seqlen, &outputs,
                               &inputs, args);
    va_end(args);
    if (!ok) return 0;
    PSMatrix delta = layer->delta;
    PSMathOpts mopts = {.acceleration = network->acceleration};
    /* Compute delta */
    PSFloat softmax_sum = 0.0;
    uint64_t delta_len = PSMatrixLength(delta);
    if (seqlen < 1 || !handle_seq) {
        seqlen = 1;
        delta_len = layer->size;
    }
    PSVectorCopy(delta, outputs, delta_len);
    PSFloat *delta_p = delta;
    for (int i = 0; i < seqlen; i++) {
        uint64_t oidx;
        if (onehot) oidx = (uint64_t) *(y++);
        else {
            PSVectorMax(y, &oidx, layer->size, &mopts);
            y += layer->size;
        }
        delta_p[oidx] += -1;
        delta_p += layer->size;
    }
    if (apply_derivative) {
        delta_p = delta;
        PSFloat *out_p = outputs;
        for (int i = 0; i < seqlen; i++) {
            PSMultiplyVectors(delta_p, out_p, delta_p, layer->size, &mopts);
            softmax_sum = PSSumVectorElements(delta_p, layer->size, &mopts);
            mopts.store_mode = PS_STORE_MODE_SUB;
            PSMultiplyVectorScalar(
                out_p, softmax_sum, delta_p, layer->size, &mopts
            );
            mopts.store_mode = PS_STORE_MODE_SET;
            delta_p += layer->size;
            out_p += layer->size;
        }
    }
    /* Update gradients */
    PSUpdateGradient(gradient, inputs, layer, previous_layer, use_bias, seqlen);
    /* Update previous layer delta */
    if (previous_layer->delta != NULL) {
        if (!PSUpdatePreviousLayerDelta(layer, previous_layer, 0, seqlen))
            return 0;
    }
    return 1;
}

int outputLayerBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSFloat *y, PSGradient *gradient, ...)
{
    PSNeuralNetwork *network = layer->network;
    int handle_seq = PSHandleSequenceAtOnce(layer);
    int is_softmax = layer->type == SoftMax;
    int t = 0, seqlen = 1, ok = 1;
    PSFloat *outputs = NULL, *inputs = NULL;
    /* Checks */
    va_list args;
    va_start(args, gradient);
    ok = PSBeforeLayerBackprop(layer, previous_layer, &t, &seqlen, &outputs,
                               &inputs, args);
    va_end(args);
    if (!ok) return 0;
    if (is_softmax)
        return softmaxLayerBackprop(layer, previous_layer, y, gradient, t);
    int apply_derivative = outputDerivativeNeeded(network);
    int onehot = (layer->flags & FLAG_ONEHOT);
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSMatrix delta = layer->delta;
    uint64_t delta_len = PSMatrixLength(delta);
    PSMathOpts mopts = {.acceleration = network->acceleration};
    /* Compute delta */
    if (!onehot) PSSubtractVectors(outputs, y, delta, delta_len, &mopts);
    else {
        PSVectorCopy(delta, outputs, delta_len);
        if (seqlen < 1 || !handle_seq) seqlen = 1;
        PSFloat *delta_p = delta;
        for (int i = 0; i < seqlen; i++) {
            int oidx = (int) *(y++);
            delta_p[oidx] -= 1;
            delta_p += layer->size;
        }
    }
    if (apply_derivative && layer->derivative != NULL) {
        ok = PSApplyDerivative(
            layer->derivative, delta, outputs, delta_len, &mopts
        );
        if (!ok) return 0;
    }
    /* Update gradients */
    PSUpdateGradient(gradient, inputs, layer, previous_layer, use_bias, seqlen);
    /* Update previous layer delta */
    if (previous_layer->delta != NULL) {
        if (!PSUpdatePreviousLayerDelta(layer, previous_layer, 0, seqlen))
            return 0;
    }
    return 1;
}

int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *gradient, ...)
{
    PSMatrix delta = layer->delta;
    if (delta == NULL) return 0;
    PSNeuralNetwork *network = layer->network;
    PSMathOpts mopts = {.acceleration = network->acceleration};
    /* Checks */
    int handle_seq = PSHandleSequenceAtOnce(layer),
        use_bias = !(layer->flags & FLAG_NO_BIAS),
        t = 0, seqlen = 1, ok = 1;
    PSFloat *outputs = NULL, *inputs = NULL;
    va_list args;
    va_start(args, gradient);
    ok = PSBeforeLayerBackprop(layer, previous_layer, &t, &seqlen, &outputs,
                               &inputs, args);
    va_end(args);
    if (!ok) return 0;
    /* Activation Derivative */
    int has_derivative = (layer->derivative != NULL);
    if (has_derivative) {
        int size = layer->size;
        if (handle_seq) size *= seqlen;
        if (!PSApplyDerivative(layer->derivative, delta, outputs, size, &mopts))
            return 0;
    }
    /* Update gradient */
    PSUpdateGradient(gradient, inputs, layer, previous_layer, use_bias, seqlen);
    /* Update previous layer delta */
    if (previous_layer->delta != NULL) {
        if (!PSUpdatePreviousLayerDelta(layer, previous_layer, 0, seqlen))
            return 0;
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
    int onehot, osize, ysize, i, t, ok = 1;
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
        int hidden_states_count = PSStateSequenceLength(output_layer);
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
            if (do_truncate && Recurrent == previous_layer->type)
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
            if (layer->pretrained) break;
            PSGradient *lgradients = gradients[i - 1];
            if (!PSIsRecurrent(layer)) break;
            PSLayerType ltype = layer->type;
            int is_recurrent = (Recurrent == ltype);
            int is_lstm = (LSTM == ltype);
            int is_gru = (GRU == ltype);
            if (!is_recurrent && !is_lstm && !is_gru) continue;

            /*  Apply derivative on layer deltas */
            if (!is_lstm && !is_gru) {
                delta = layer->delta;
                PSMathOpts mopts = {
                    .acceleration = layer->network->acceleration
                };
                int ok = PSApplyDerivative(
                    layer->derivative, delta,PSGetStates(layer, t),
                    layer->size, &mopts
                );
                if (!ok) return 0;
            }
            if (do_truncate && Recurrent == previous_layer->type)
                resetLayerDeltas(previous_layer, 0);
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
    int i, ok = 1;
    PSGradient **new_gradients = NULL;
    if (gradients == NULL) {
        gradients = createGradients(network);
        new_gradients = gradients;
    }
    if (gradients == NULL) return NULL;
    int netsize = network->size, is_recurrent = PSIsRecurrent(network);
    PSLayer *output_layer = network->layers[netsize - 1];
    if (output_layer->type != FullyConnected && output_layer->type != SoftMax){
        PSErr(NULL, "Output layer must be FullyConnected or SoftMax");
        ok = 0;
        goto final;
    }
    PSGradient *lgradients = gradients[netsize - 2]; /* No gradient for
                                                        inputs */
    PSLayer *previous_layer = NULL;
    ok = resetDeltas(network);
    if (!ok) {
        PSErr(NULL, "Failed to reset network deltas");
        goto final;
    }

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
        if (layer->pretrained) break;
        lgradients = gradients[i - 1];
        if (PSIsRecurrent(layer)) {
            int timesteps = PSStateSequenceLength(layer);
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
            FullyConnected == ltype || Embedding == ltype ||
            Dropout == ltype ||
            (Pooling == ltype && Convolutional == prev_ltype) ||
            Convolutional == ltype
        );
        if (!ok) {
            PSErr(NULL, "Backprop from %s to %s not supported!\n",
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
        if (new_gradients != NULL) PSDeleteNetworkGradients(
            new_gradients, network
        );
        return NULL;
    }
    return gradients;
}

int applyGradientsOnParameters(
    int param_type, PSTrainingOptions *options, PSGradient *grads,
    PSFloat *params, PSGradient *mg, PSGradient *xg,
    uint64_t offset, uint64_t len, PSFloat rate,
    int iteration, int acceleration
)
{
    if (params == NULL || grads == NULL) return 0;
    PSOptimization optimization = PSDefaultOptimization;
    PSFloat momentum = 0;
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    optimization = options->optimization;
    if (optimization == NULL) optimization = PSDefaultOptimization;
    PSFloat *gptr = NULL, *mptr = NULL, *xptr= NULL;
    if (param_type == PARAM_TYPE_BIAS) {
        gptr = grads->biases;
        if (mg != NULL) mptr = mg->biases;
        if (xg != NULL) xptr = xg->biases;
    } else if (param_type == PARAM_TYPE_WEIGHT) {
        gptr = grads->weights;
        if (mg != NULL) mptr = mg->weights;
        if (xg != NULL) xptr = xg->weights;
    } else {
        fprintf(
            stderr,
            "FATAL: invalid param type %d in %s\n", param_type, __func__
        );
        abort();
    }
    if (offset > 0) {
        gptr += offset;
        if (mptr != NULL) mptr += offset;
        if (xptr != NULL) xptr += offset;
    }
    PSFloat *mtmp = ((mg != NULL) ? mg->tmp : NULL),
            *xtmp = ((xg != NULL) ? xg->tmp : NULL);
    return optimization(
        params, gptr, mptr, xptr, grads->tmp, mtmp, xtmp, rate,  momentum,
        len, acceleration, iteration, options
    );
}

int sumGradients(PSGradient **dstgrads, PSGradient **srcgrads, int count,
                 PSMathOpts *mopts)
{
    for (int i = 0; i < count; i++) {
        PSGradient *src = srcgrads[i];
        PSGradient *dst = dstgrads[i];
        if (src == NULL) continue;
        if (src->bias_count > 0 && src->biases != NULL) {
            if (dst->biases == NULL || dst->bias_count != src->bias_count) {
                PSErr(NULL, "Cannot sum gradients: destination and source "
                      "biases mismatch");
                return 0;
            }
            PSSumVectors(src->biases, dst->biases, dst->biases, src->bias_count,
                         mopts);
        }
        if (src->weight_count > 0 && src->weights != NULL) {
            if (dst->weights == NULL || dst->weight_count != src->weight_count)
            {
                PSErr(NULL, "Cannot sum gradients: destination and source "
                      "weights mismatch");
                return 0;
            }
            PSSumVectors(src->weights, dst->weights, dst->weights,
                         src->weight_count, mopts);
        }
    }
    return 1;
}

void clipGradients(PSGradient **grads, PSFloat min, PSFloat max, int count,
                   PSMathOpts *mopts)
{
    for (int i = 0; i < count; i++) {
        PSGradient *g = grads[i];
        if (g == NULL) continue;
        if (g->bias_count > 0 && g->biases != NULL)
            PSVectorClip(g->biases, min, max, g->biases, g->bias_count, mopts);
        if (g->weight_count > 0 && g->weights != NULL)
            PSVectorClip(g->weights,min,max,g->weights,g->weight_count,mopts);
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
                                PSGradient **memory_gradients1,
                                PSGradient **memory_gradients2, ...)
{
    int i, j, netsize = network->size, gsize = netsize - 1,
        seqlen = 0, iteration = 0, apply_clip = 0;
    assert(batch_size > 0);
    PSFloat *x = NULL; /* Training element */
    PSFloat *y = NULL; /* Labels */
    PSFloat l1 = 0.0, l2 = 0.0, l1_loss = 0.0, l2_loss = 0.0, momentum = 0.0,
            clip_max = 0.0, clip_min = 0.0;
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    /* Create gradients for the current batch. */
    PSGradient **gradients = createGradients(network);
    if (gradients == NULL) {
        network->status = STATUS_ERROR;
        return STATUS_ERROR_LOSS;
    }
    PSGradient **bp_gradients = NULL;
    PSFloat **sequences = NULL;
    int is_recurrent = PSIsRecurrent(network),
        input_is_seq = 0, output_is_seq = 0;
    if (is_recurrent || PSUseSequences(network)) {
        va_list args;
        va_start(args, memory_gradients2);
        sequences = va_arg(args, PSFloat**);
        va_end(args);
        if (sequences == NULL) {
            PSErr(__func__, "Sequences argument is NULL");
            network->status = STATUS_ERROR;
            goto final;
        }
        if (is_recurrent) {
            input_is_seq = PSIsRecurrent(network->layers[0]);
            output_is_seq =
                PSIsRecurrent(network->layers[network->size - 1]);
        } else {
            input_is_seq = PSHandleSequenceAtOnce(network->layers[0]);
            output_is_seq =
                PSHandleSequenceAtOnce(network->layers[network->size - 1]);
        }
    }
    UNUSED(elements_count); /* TODO: remove elements_count arg if not needed */
    PSOptimization optimization = PSDefaultOptimization;
    int use_weight_decay = 0, divide_grads_by_batches = 0;
    if (opts != NULL) {
        optimization = opts->optimization;
        if (optimization == NULL) optimization = PSDefaultOptimization;
        divide_grads_by_batches =  (
            optimization == PSAdaDeltaOptimization ||
            optimization == PSWindowGradOptimization ||
            optimization == PSAdaGradOptimization ||
            optimization == PSAdamOptimization
        );
        use_weight_decay = (opts->flags & TRAINING_WEIGHT_DECAY);
        l1 = opts->l1_decay;
        l2 = opts->l2_decay;
        momentum = opts->momentum;
        if ((apply_clip = (opts->clip != 0.0))) {
            clip_max = PSAbs(opts->clip);
            clip_min = clip_max * -1;
        }
    }
    int apply_momentum = (momentum > 0.0);
    int use_optimization = (optimization != PSDefaultOptimization);
    if (apply_momentum || use_optimization) {
        if (memory_gradients1 == NULL) {
            network->status = STATUS_ERROR;
            goto final;
        }
        if (optimization == PSAdaDeltaOptimization ||
            optimization == PSAdamOptimization)
        {
            if (memory_gradients2 == NULL) {
                network->status = STATUS_ERROR;
                goto final;
            }
        }
    }
    if (!divide_grads_by_batches) rate /= batch_size;
    else divide_grads_by_batches = (batch_size > 1);
    /* If weight decay is enabled (TRAINING_WEIGHT_DECAY flag), l2_decay
     * will be used to directly update weights and not gradients.
     * Furthermore, l1_loss and l2_loss won't be computed nor used in
     * loss calculation.
     * If disabled (default), L1/L2 regularization will be used so l2_decay
     * and l1_decaywill be applied on gradients and L1/L2 loss will be
     * computed and taken into account by final loss. */
    if (l2 != 0.0 && use_weight_decay) {
        if (divide_grads_by_batches) l2 = opts->l2_decay / batch_size;
        l2 = (1 - (rate * l2));
    }
    if (l1 != 0.0 && use_weight_decay) {
        if (divide_grads_by_batches) l1 = opts->l1_decay / batch_size;
        l1 = (1 - (rate * l1));
    }

    PSMathOpts mopts = {.acceleration = network->acceleration};
    PSGradient **bp_dest_gradients = gradients;
    if (apply_clip) bp_dest_gradients = NULL;
    beforeBatchTraining(network);
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
        if (sequences == NULL) {
            /* Non-recurrent and non-sequence network */
            int element_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += element_size;
        } else {
            /* Recurrent network or network handling sequences*/
            x = sequences[i];
            if (input_is_seq) {
                seqlen = (int) *x;
                if (seqlen == 0) {
                    PSErr(__func__, "Sequence len must b > 0. (batch = %d)", i);
                    network->status = STATUS_ERROR;
                    goto final;
                }
                y = x + 1 + (seqlen * training_data_size);
            } else y = x + training_data_size;
        }
        bp_gradients = backprop(network, x, y, opts, bp_dest_gradients);
        if (bp_gradients == NULL) {
            PSErr(NULL, "Backpropagation failed for network '%s'",
                 (network->name != NULL ? network->name : "UNNAMED")
            );
            network->status = STATUS_ERROR;
            goto final;
        }
        if (apply_clip) {
            clipGradients(bp_gradients, clip_min, clip_max, gsize, &mopts);
            int ok = sumGradients(gradients, bp_gradients, gsize, &mopts);
            PSDeleteNetworkGradients(bp_gradients, network);
            if (!ok) {
                network->status = STATUS_ERROR;
                goto final;
            }
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
        if (memory_gradients1 != NULL) mgradients = memory_gradients1[i];
        if (memory_gradients2 != NULL) xgradients = memory_gradients2[i];
        PSLayer *layer = network->layers[i + 1];
        if (layer->pretrained) continue;
        /* Update Biases */
        if (!(layer->flags & FLAG_NO_BIAS)) {
            if (divide_grads_by_batches) PSDivideVectorScalar(
                lgradients->biases, (PSFloat) batch_size, lgradients->biases,
                lgradients->bias_count, &mopts
            );
            int ok = applyGradientsOnBiases(
                opts, lgradients, layer->biases, mgradients, xgradients,
                lgradients->bias_count, rate, iteration, mopts.acceleration
            );
            if (!ok) {
                PSErr(
                    __func__, "Layer[%d]: failed to update biases",layer->index
                );
                network->status = STATUS_ERROR;
                goto final;
            }
        }
        /* Update Weights */
        uint64_t wgrad_offset = 0;
        for (j = 0; j < layer->weight_types_count; j++) {
            PSMatrix weights = layer->weights[j];
            if (weights == NULL) {
                PSErr(
                    __func__, "Layer[%d]: weights[%d] is NULL",
                    layer->index, j
                );
                network->status = STATUS_ERROR;
                goto final;
            }
            uint64_t wlen = PSMatrixLength(weights);
            PSFloat *wgradients = lgradients->weights + wgrad_offset;
            if (l1 != 0.0 || l2 != 0.0) {
                int ok = PSLRegularization(
                    l1, l2, weights, wgradients, lgradients->tmp,
                    wlen, &l1_loss, &l2_loss, 0, use_weight_decay,
                    mopts.acceleration
                );
                if (!ok) {
                    PSErr(
                        __func__, "Layer[%d]: failed to apply L1/L2 "
                        "regularization", layer->index
                    );
                    network->status = STATUS_ERROR;
                    goto final;
                }
            }
            if (divide_grads_by_batches) PSDivideVectorScalar(
                wgradients, (PSFloat) batch_size, wgradients,
                wlen, &mopts
            );
            int ok = applyGradientsOnWeights(
                opts, lgradients, weights, mgradients, xgradients,
                wgrad_offset, wlen, rate, iteration, mopts.acceleration
            );
            if (!ok) {
                PSErr(
                    __func__, "Layer[%d]: failed to update weights[%d]",
                    layer->index, j
                );
                network->status = STATUS_ERROR;
                goto final;
            }
            wgrad_offset += wlen;
        }
    }
final:
    PSDeleteNetworkGradients(gradients, network);
    if (network->status == STATUS_ERROR) return STATUS_ERROR_LOSS;
    PSLayer *out = network->layers[netsize - 1];
    int onehot = out->flags & FLAG_ONEHOT;
    if (onehot) label_data_size = 1;
    if (output_is_seq) label_data_size *= seqlen;
    PSFloat outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (onehot) {
            int idx = (int) *(y + i);
            outputs[i] = PSGetState(out, idx, i);
        } else {
            if (!output_is_seq) outputs[i] = PSGetState(out, i);
            else i = fetchSequenceOutputState(out, outputs, i, 0);
        }
    }
    if (opts == NULL) l1 = l2 = 0.0;
    if (l1 != 0.0) l1_loss *= (opts->l1_decay / batch_size);
    if (l2 != 0.0) l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    int onehot_size = (onehot ? out->size : 0);
    PSFloat loss =  network->loss(outputs, y, label_data_size, onehot_size);
    if (output_is_seq && seqlen > 0) loss /= seqlen;
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
    PSFloat **sequences = NULL, **sequence_head = NULL;
    int flags = 0, do_validate = 0;
    if (options != NULL) flags = options->flags;
    if (PSIsRecurrent(network)) {
        PSLayer *out = network->layers[network->size - 1];
        int o_size = (out->flags & FLAG_ONEHOT ? 1 : network->output_size);
        sequences = getRecurrentSequences(
            network, training_data, elements_count, network->input_size, o_size
        );
        if (sequences == NULL) {
            network->status = STATUS_ERROR;
            return STATUS_ERROR_LOSS;
        }
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffleSequences(sequences, elements_count);
    } else {
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffle(training_data, elements_count, element_size);
    }
    PSFloat err = 0.0, avg_err = 0.0, acc = 0.0, tot_acc = 0.0, avg_acc = 0.0;
    long tot_t = 0, avg_t, elapsed_t, test_data_size, validations = 0;
    int offset = (element_size * batch_size), validate_every = 0, i;
    PSGradient **memory_gradients1 = training_ctx->memory_gradients1,
               **memory_gradients2 = training_ctx->memory_gradients2;
    if (options != NULL) {
        PSOptimization optimization = options->optimization;
        if (options->momentum != 0 || optimization != PSDefaultOptimization) {
            if (memory_gradients1 == NULL) {
                memory_gradients1 = createGradients(network);
                if (memory_gradients1 == NULL) {
                    network->status = STATUS_ERROR;
                    goto final;
                }
                training_ctx->memory_gradients1 = memory_gradients1;
            }
        }
        if (optimization == PSAdaDeltaOptimization ||
            optimization == PSAdamOptimization)
        {
            if (memory_gradients2 == NULL) {
                memory_gradients2 = createGradients(network);
                if (memory_gradients2 == NULL) {
                    network->status = STATUS_ERROR;
                    goto final;
                }
                training_ctx->memory_gradients2 = memory_gradients2;
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
    sequence_head = sequences;
    for (i = 0; i < batches_count; i++) {
        network->training->current_batch = i;
        int batch_num = i + 1;
        struct timeval st, et;
        gettimeofday(&st, NULL);
        PSFloat batch_err = updateNetworkParameters(
            network, training_data, batch_size, elements_count, options,
            learning_rate, memory_gradients1, memory_gradients2, sequence_head
        );
        err += batch_err;
        if (network->status == STATUS_ERROR) {
            PSErr(NULL, "SGD failed at batch %d for network '%s'", i,
                  (network->name != NULL ? network->name : "UNNAMED")
            );
            goto final;
        }
        gettimeofday(&et, NULL);
        elapsed_t = PSGetElapsedTimeUS(st, et);
        tot_t += elapsed_t;
        avg_t = (tot_t / batch_num);
        if (batch_num < batches_count) {
            avg_err = err / (PSFloat) batch_num;
            char *elapsed_str = PSGetElapsedTimeString(avg_t, 0);
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
                    "loss = %.2lf, acc. = %.2lf, avg. time = %s",
                    avg_err, avg_acc, elapsed_str);
            } else {
                PSLogTrainingProgress(network, epochs, batches_count, 1,
                    "loss = %.2lf, avg. time = %s",
                    avg_err, elapsed_str);
            }
        } else PSLogTrainingProgress(network, epochs, batches_count, 1, NULL);
        if (network->status == STATUS_ERROR) {
            if (sequences != NULL) free(sequences);
            return STATUS_ERROR_LOSS;
        }
        if (network->onBatchTrained != NULL) {
            network->onBatchTrained(
                network, network->training->current_epoch,
                epochs, avg_err, batch_err, avg_acc, &learning_rate,
                (sequences != NULL ? *sequence_head : training_data)
            );
        }
        if (sequences == NULL) training_data += offset;
        else sequence_head += batch_size;
        int action = network->training->requested_action;
        if (action == ACTION_ABORT) {
            network->status = action;
            break;
        }
    }
final:
    if (sequences != NULL) free(sequences);
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
    PSLayer *output_layer = PSGetOutputLayer(network);
    int input_size = network->input_size;
    int output_size = network->output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int y_size = (onehot ? 1 : output_size);
    int element_size = input_size + output_size;
    int elements_count;
    int reads_input_sequence = 0, emits_output_sequence = 0;
    PSFloat **sequences = NULL;
    if (PSUseSequences(network)) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(test_data++);
        data_size--;
        reads_input_sequence = PSUseSequences(network->layers[0]);
        emits_output_sequence = PSUseSequences(output_layer);
        sequences = getRecurrentSequences(
            network, test_data, elements_count, input_size, y_size
        );
        if (sequences == NULL) goto err;
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
    if (log) PSInfo("Testing started at %s", timestr);
    for (i = 0; i < elements_count; i++) {
        if (log) printf("\rTesting %d/%d", i + 1, elements_count);
        fflush(stdout);
        PSFloat *inputs = NULL;
        PSFloat *expected = NULL;
        int seqlen = 0;
        if (sequences == NULL) {
            /*  Non Recurrent and no sequences*/
            inputs = test_data;
            test_data += input_size;
            expected = test_data;

            int ok = PSFeedforward(network, inputs);
            if (!ok) goto err;

            int omax = 0; /* Output index with max value */
            int emax = 0; /* Expected index with max value */
            if (!PSFindLayerMaxState(output_layer, NULL, &omax)) {
                PSErr(NULL, "Could not find output layer max state");
                goto err;
            }
            if (!onehot) emax = arrayMaxIndex(expected, output_size);
            else emax = (int) *(expected);
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            /*  Recurrent or sequences*/
            inputs = sequences[i];
            if (reads_input_sequence) {
                seqlen = (int) *inputs;
                expected = inputs + 1 + (seqlen * input_size);
            } else {
                seqlen = (int) *(inputs + network->input_size);
                expected = inputs + network->input_size + 1;
            }
            if (seqlen == 0) {
                errmsg = "recurrent data with zero seqlen";
                goto err;
            }
            int ok = PSResetNetworkStateSequences(network, seqlen, 0);
            if (!ok) goto err;
            ok = PSFeedforward(network, inputs);
            if (!ok) goto err;

            int correct_states = 0;
            if (emits_output_sequence) {
                int output_seqlen = PSStateSequenceLength(output_layer);
                int steps_to_check = seqlen, max_seqlen = seqlen;
                if (output_seqlen < seqlen) {
                    steps_to_check = output_seqlen;
                } else if (output_seqlen > seqlen)
                    max_seqlen = output_seqlen;
                int label_data_size = y_size * steps_to_check;
                int last_label_idx = (label_data_size - 1);
                if (label_data_size <= 0) goto err;
                PSFloat outputs[label_data_size];
                for (j = 0; j < label_data_size; j++) {
                    int is_last_label = (j == last_label_idx);
                    j = fetchSequenceOutputState(
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
                    (float) correct_states / (float) max_seqlen
                );
            } else {
                int omax = 0; /* Output index with max value */
                int emax = 0; /* Expected index with max value */
                if (!PSFindLayerMaxState(output_layer, NULL, &omax)) {
                    PSErr(NULL, "Could not find output layer max state");
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
    if (!emits_output_sequence) {
        accuracy = (float) correct_results / (float) elements_count;
        if (log) printf("Accuracy (%d/%d): %.2f\n",
                        correct_results, elements_count,accuracy);
    } else {
        accuracy = correct_amount / (float) elements_count;
        free(sequences);
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
    if (options->optimization == NULL)
        options->optimization = PSDefaultOptimization;
    if (options->optimization != PSDefaultOptimization) {
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
    options->optimization = PSDefaultOptimization;
    if (options->epochs <= 0) options->epochs = 1;
    if (options->batch_size <= 0) options->batch_size = 1;
}

int PSPretrainLayers(PSNeuralNetwork *network, PSFloat *training_data,
                     int data_size)
{
    if (network->flags & FLAG_PRETRAINER) return 1;
    int original_status = network->status;
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer == NULL) continue;
        if (!isPretrainableLayer(layer)) continue;
        if (layer->pretrained) continue;
        PSInfo("Pretraining layer[%d] (%s)", i, PSGetLayerTypeLabel(layer));
        network->status = STATUS_PRETRAINING;
        int trained = layer->pretrain(layer, training_data, data_size);
        if (!trained) {
            network->status = STATUS_ERROR;
            return 0;
        }
        PSInfo("Successfully pretrained layer[%d] (%s)",
               i, PSGetLayerTypeLabel(layer));
        network->status = original_status;
    }
    return 1;
}

void PSTrain(PSNeuralNetwork *network,
             PSFloat *training_data,
             int data_size,
             PSFloat *test_data,
             int test_size,
             PSTrainingOptions *options)
{
    int epochs = 0, batch_size = 0, i, elements_count;
    PSFloat learning_rate = 0.0;
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
    PSTrainingOptions default_options = {0};
    if (options == NULL) {
        options = &default_options;
        PSSetDefaultTrainingOptions(options);
    }
    checkTrainingOptions(options);
    epochs = options->epochs;
    batch_size = options->batch_size;
    learning_rate = options->learning_rate;
    if (learning_rate <= 0.0) {
        PSErr(__func__, "learning_rate must be > 0.0");
        return;
    }
    if (batch_size <= 0) batch_size = options->batch_size = 1;
    if (epochs <= 0) epochs = options->epochs = 1;
    int element_size = network->input_size + network->output_size;
    PSNetworkContext *ctx = getNetworkContext(network);
    if (ctx == NULL) {
        PSErr(__func__, "Missing network context");
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
    training_ctx->options = *options;
    /* Eventually pretrain layers (if pretrainable layers are found) */
    if (!PSPretrainLayers(network, training_data, data_size)) {
        network->status = STATUS_ERROR;
        PSErr(__func__, "Failed to pretrain network '%s'",
              network->name != NULL ? network->name : "UNNAMED");
        return;
    }
    int is_recurrent = PSIsRecurrent(network);
    if (is_recurrent) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(training_data++);
        data_size--;
    } else elements_count = data_size / element_size;
    const char *name = network->name != NULL ? network->name : "UNNAMED";
    PSLog(PSLOGLEVEL_NOTICE, "Training network \"%s\"\n", name);
    PSInfo("Training data elements:     %d", elements_count);
    PSInfo("Batch Size:                 %d", batch_size);
    PSInfo("Learning Rate:              %g", learning_rate);
    if (options->validate_every_batches > 0) {
        printf(
            "Validate Every: %d batch(es)\n",  options->validate_every_batches
        );
    }
    int use_weight_decay = (
        (options->l1_decay != 0 || options->l2_decay != 0) &&
        (options->flags & TRAINING_WEIGHT_DECAY)
    );
    PSInfo("L1 Decay:                   %g", options->l1_decay);
    PSInfo("L2 Decay:                   %g", options->l2_decay);
    PSInfo("Weight Decay:               %s",
            (use_weight_decay ? "yes" : "no"));
    PSInfo("Clip:                       %g", PSAbs(options->clip));
    PSInfo("Momentum:                   %g", options->momentum);
    PSInfo("Optimization:               %s",
        getOptimizationName(options->optimization));
    int single_seq = (options->flags & TRAINING_EPOCH_AS_SEQUENCE),
        no_shuffle = (options->flags & TRAINING_NO_SHUFFLE);
    if (single_seq && !no_shuffle) {
        PSWarn(
            "flag TRAINING_EPOCH_AS_SEQUENCE requires "
            "TRAINING_NO_SHUFFLE. "
            "Automatically enabling TRAINING_NO_SHUFFLE."
        );
        options->flags |= TRAINING_NO_SHUFFLE;
    }
    if (single_seq) PSInfo("Single sequence:            yes");
    PSInfo("Data shuffle (SGD):         %s", (!no_shuffle ? "yes" : "no"));
    if (is_recurrent)
        PSInfo("BPTT Truncate:              %d", options->bptt_truncate);
    if (network->layers[network->size - 1]->flags & FLAG_ONEHOT)
        PSInfo("Onehot Labels:              yes");
    char *loss_func_name = NULL;
    if (network->loss != NULL) {
        loss_func_name = getLossFunctionName(network->loss);
        PSInfo("Loss Function:              %s", loss_func_name);
    }
    /* Start training */
    int was_paused = (network->status == STATUS_PAUSED);
    network->status = STATUS_TRAINING;
    time_t start_t, end_t;
    char timestr[80];
    struct tm *tminfo;
    struct timeval epoch_st, epoch_et;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    PSLog(PSLOGLEVEL_NOTICE, "Training started at %s\n", timestr);
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
            if (!PSResetNetworkStateSequences(network, 0, 0)) {
                network->status = STATUS_ERROR;
                PSErr(__func__, "Failed to reset network recurrent states");
                return;
            }
        }
        gettimeofday(&epoch_st, NULL);
        PSFloat err = gradientDescent(network, training_data, element_size,
                                      elements_count, learning_rate,
                                      batch_size, options, epochs,
                                      test_data, test_size);
        if (network->status == STATUS_ERROR) {
            PSLog(
                PSLOGLEVEL_ERROR, "\nAn error occurred while training, "
                "aborting!\n"
            );
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
        gettimeofday(&epoch_et, NULL);
        time_t elapsed_t = PSGetElapsedTimeUS(epoch_st, epoch_et);
        char * elapsed_str = PSGetElapsedTimeString(elapsed_t, 0);
        if (i > 0 && err > prev_err && adjust_rate)
            learning_rate *= 0.5;
        if (network->onEpochTrained != NULL)
            network->onEpochTrained(network, i, epochs, err, err,
                                    acc, &learning_rate, NULL);
        prev_err = err;
        PSLogTrainingProgress(network, epochs, batches_count, 1,
            "loss = %.2lf%s (%s)\n", err, accuracy_msg, elapsed_str
        );
        fflush(stdout);
        int action = network->training->requested_action;
        if (action == ACTION_ABORT || action == ACTION_PAUSE) {
            network->status = action;
            break;
        }
    }
    time(&end_t);
    fflush(stdout);
    PSLog(PSLOGLEVEL_SUCCESS, "\nCompleted in %ld sec.\n", end_t - start_t);
    network->training->ended_at = end_t;
    if (network->status == STATUS_TRAINING) network->status = STATUS_TRAINED;
}

float PSTest(PSNeuralNetwork *network, PSFloat *test_data, int data_size) {
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO);
    return validate(network, test_data, data_size, do_log);
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
        if (Recurrent == ltype || LSTM == ltype || GRU == ltype)
            recurrent_type_layers++;
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
            if (layer->flags & FLAG_ONEHOT) onehot_input = 1;
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
                "Network is recurrent but has no Recurrent, LSTM or GRU layers"
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
                "Network is not recurrent but has Recurrent, LSTM or GRU layers"
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

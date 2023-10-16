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
#include "attention.h"
#include "operator_layer.h"
#include "debug.h"

#define LAYER_PLACEHOLDER_TYPE -1
#define STATUS_ERROR_LOSS ((PSFloat) FLT_MIN)
#define BPTT_TRUNCATE   0

#define applyGradientsOnBiases(opts, grads, params, mg, xg, len, r, i, accel) \
    applyGradientsOnParameters(PARAM_TYPE_BIAS, opts, grads, params, mg, xg,\
    0, len, r, i, accel)
#define applyGradientsOnWeights(opts,grad,val,mg,xg,offs,len,r,i,accel) \
    applyGradientsOnParameters(PARAM_TYPE_WEIGHT, opts, grad, val, mg, xg,\
    offs, len, r, i, accel)
#define outputDerivativeNeeded(model) (model->loss != PSCrossEntropyLoss)
#define getModelContext(model) ((PSModelContext *) model->context)
#define setModelContext(model, member, val) (\
    ((PSModelContext *) model->context)->member = val)
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
    PSGradient **memory_gradients[PS_MAX_MEMORY_GRADIENTS];
} PSTrainingContext;

typedef struct {
    int                 built;
    PSLayer             *first_recurrent_layer;
    PSLayer             *last_recurrent_layer;
    PSModel             *head_model;
    PSModel             *last_model;
    int                 model_chain_length;
    PSFloat             *sequence_start;
    PSTrainingContext   *training_context;
} PSModelContext;

static PSLossFunction loss_functions[] = {
    NULL,
    PSQuadraticLoss,
    PSCrossEntropyLoss
};

static size_t loss_functions_count = sizeof(loss_functions) /
                                     sizeof(PSLossFunction);

/* Function Prototypes */

char *getLossFunctionName(PSLossFunction function);
char *getModelStatusLabel(PSModel *model);
float validate(PSModel *model, PSFloat *test_data, int data_size,
               PSTrainingOptions *opts, int log);
int PSInitConvolutionalLayer(PSModel *model, PSLayer *layer,
                             PSLayerDef *ldef);
int PSInitPoolingLayer(PSModel *model, PSLayer *layer,
                       PSLayerDef *ldef);
int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *layer_gradients, ...);
int PSInitRecurrentLayer(PSModel *model, PSLayer *layer,
                         int size,int ws, PSLayerDef *ldef);
int PSInitLSTMLayer(PSModel *model, PSLayer *layer,
                    int size, int ws, PSLayerDef *ldef);
int PSInitGRULayer(PSModel *model, PSLayer *layer,
                   int size, int ws, PSLayerDef *ldef);
int PSInitDropoutLayer(PSModel *model, PSLayer *layer,
                       PSLayerDef *layer_def);
int PSInitEmbeddingLayer(PSLayer *layer, int size, PSLayerDef *ldef);
int PSInitNormalizationLayer(PSLayer *layer, PSLayerDef *ldef);
int PSInitAttentiontionLayer(PSLayer *layer, PSLayerDef *ldef);
int PSInitOperatorLayer(PSLayer *layer, PSLayerDef *ldef);
int PSInitPositionalLayer(PSLayer *layer, PSLayerDef *layer_def);
int PSDumpGradients(PSModel *model, PSGradient ***gradients,
                    const char* filename, PSTrainingOptions *opts);
PSGradient **cloneModelGradients(PSGradient **gradients,
                                   PSModel *model);
static void deleteTrainingContext(PSTrainingContext *training_ctx,
                                  PSModel *model);
static void deleteModelContext(PSModelContext *ctx,
                                 PSModel *model);
int writeSerializedFloat(FILE *out, PSFloat fnum, int opts);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
int updateModelChain(PSModel *head);
int useAutoRegression(PSModel *model,
                      PSForwardOptions *forward_opts,
                      PSTrainingOptions *training_opts);
PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSModel *model);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);
static PSModel *cloneModel(PSModel *model, int layout_only,
                                     PSModel *parent);
int PSIsLayerPlaceholder(PSLayer *layer);

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

void PSLogTrainingProgress(PSModel *model, int status, int epochs,
                           int batches, PSFloat *loss, PSFloat *accuracy,
                           time_t *elapsed, int validating_current,
                           int validating_tot)
{
    UNUSED(validating_current);
    static int epoch_printed = -1;
    if (PSLogLevel > PSLOGLEVEL_INFO) return;
    if (model->training == NULL) return;
    if (status == STATUS_TRAINED || status == STATUS_ERROR) {
        PSLineEnd();
        epoch_printed = -1;
        return;
    }
    if (model->training->current_batch == 0) {
        if (epoch_printed != model->training->current_epoch) {
            PSLineEnd();
            printf("Epoch %d/%d:\n", model->training->current_epoch + 1,
                    epochs);
            fflush(stdout);
            epoch_printed = model->training->current_epoch;
        }
    }
    int batch_num = model->training->current_batch + 1;
    int percent =
        (int) roundf(((float) batch_num / (float) batches) * 100.0f);
    int pad = PSCalcIntStringLength(batches);
    int lnflags = PS_LINE_PLAIN_ASCII | PS_LINE_FILL;
    PSLineStart(
        PS_LINE_OVERWRITE, " - Batch %*d/%d %3d%%", pad, batch_num,
        batches, percent
    );
    char *elapsed_str = "";
    if (elapsed != NULL) elapsed_str = PSGetElapsedTimeString(*elapsed, 0);
    if (status == STATUS_VALIDATING) {
        if (validating_tot > 0)
            PSLineAppend(lnflags, ", validating %d element(s)", validating_tot);
        else PSLineAppend(lnflags, ", validating...");
    } else if (status == STATUS_TRAINING) {
        if (batch_num < batches) {
            assert(loss != NULL);
            if (accuracy != NULL) {
                PSLineAppend(
                    lnflags, ", loss = %.2lf, acc. = %.2lf, avg. time = %s",
                    *loss, *accuracy, elapsed_str
                );
            } else {
                PSLineAppend(lnflags, ", loss = %.2lf, avg. time = %s",
                             *loss, elapsed_str);
            }
        } else {
            if (loss != NULL) {
                if (accuracy != NULL) {
                    PSLineAppend(lnflags, ", loss = %.2lf, acc. = %.2f (%s)",
                                 *loss, *accuracy, elapsed_str);
                } else {
                    PSLineAppend(lnflags, ", loss = %.2lf (%s)", *loss,
                                 elapsed_str);
                }
                PSLineEnd();
            } else PSLineFill();
        }
    } else {
        PSLineFill();
    }
}

void PSLogTrainingProgressBar(PSModel *model, int status, int epochs,
                              int batches, PSFloat *loss, PSFloat *accuracy,
                              time_t *elapsed, int validating_current,
                              int validating_tot)
{
    UNUSED(validating_current);
    UNUSED(validating_tot);
    static int epoch_printed = -1;
    static int min_sfx_len = -1;
    if (PSLogLevel > PSLOGLEVEL_INFO) return;
    if (model->training == NULL) return;
    if (status == STATUS_TRAINED || status == STATUS_ERROR) {
        PSLineEnd();
        epoch_printed = -1;
        min_sfx_len = -1;
        return;
    }
    if (model->training->current_batch == 0) {
        if (epoch_printed != model->training->current_epoch) {
            PSLineEnd();
            printf("Epoch %d/%d:\n", model->training->current_epoch + 1,
                    epochs);
            fflush(stdout);
            epoch_printed = model->training->current_epoch;
        }
    }
    int tw = PSGetTerminalColumns();
    int batch_num = model->training->current_batch + 1;
    int pad = PSCalcIntStringLength(batches);
    PSLineStart(
        PS_LINE_OVERWRITE, "Batch %*d/%d ", pad, batch_num, batches
    );
    int maxlen = tw - 1;
    char *elapsed_str = "";
    char sfx[35] = {0};
    int epoch_ended = 0;
    if (status == STATUS_VALIDATING) {
        maxlen -= snprintf(sfx, 35, " | validating...");
    } else if (status == STATUS_TRAINING) {
        if (elapsed != NULL) elapsed_str = PSGetElapsedTimeString(*elapsed, 0);
        if (batch_num < batches) {
            if (accuracy != NULL) {
                maxlen -= snprintf(
                    sfx, 35, " | loss: %.2lf | acc: %.2lf | %s",
                    *loss, *accuracy, elapsed_str
                );
            } else {
                maxlen -= snprintf(
                    sfx, 35, " | loss: %.2lf | %s", *loss, elapsed_str
                );
            }
        } else {
            if (loss != NULL) {
                if (accuracy != NULL) {
                    maxlen -= snprintf(
                        sfx, 35, " | loss: %.2lf |acc: %.2f | %s",
                        *loss, *accuracy, elapsed_str
                    );
                } else {
                    maxlen -= snprintf(
                        sfx, 35, " | loss: %.2lf | %s", *loss, elapsed_str
                    );
                }
                epoch_ended = 1;
            }
        }
    }
    if (min_sfx_len < 0) min_sfx_len = (tw - 1 - maxlen);
    else {
        int minlen = (tw - 1 - min_sfx_len);
        if (maxlen > minlen) maxlen = minlen;
    }
    int style = PS_PROGRESS_STYLE_DOUBLE_DASH, color = 0,
        flags = PS_PROGRESS_FLAG_JUST_BAR;
    if (PSLogColorEnabled()) {
        color = 1;
        style = PS_PROGRESS_STYLE_LINE;
    }
    PSProgressBar(batch_num, batches, style, color, flags, maxlen, NULL);
    if (sfx[0]) {
        int lnflags = PS_LINE_PLAIN_ASCII | PS_LINE_FILL;
        PSLineAppend(lnflags, "%s", sfx);
    }
    if (epoch_ended) PSLineEnd();
}

void dumpForwardStep(int i, PSFloat a, PSFloat b, PSFloat sum,
                     int using_acceleration, PSMathOpts *opts)
{
    UNUSED(a);
    UNUSED(b);
    UNUSED(sum);
    UNUSED(using_acceleration);
    if (opts == NULL) return;
    PSDebugStepInfo *info = (PSDebugStepInfo *) opts->data;
    if (info == NULL || info->layer == NULL || info->model == NULL) return;
    info->training_phase = TRAINING_PHASE_FORWARD;
    int prev_idx = info->layer->index - 1;
    PSTrainingDebugDumpStep(
        info, "previous_neuron=%d-%d,weight_index=%d\n", prev_idx, i, i
    );
}

/* Forward Functions */

int checkLayerForForward(PSLayer *layer) {
    if (layer == NULL) return 0;
    int trainable = !(layer->flags & FLAG_NON_TRAINABLE);
    if (layer->index == 0) {
        PSErr(NULL, "Cannot perform forward on layer 0");
        return 0;
    }
    if (layer->model == NULL) {
        PSErr(NULL, "Layer[%d]: layer has no model");
        return 0;
    }
    PSLayer *previous = layer->model->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL", layer->index);
        return 0;
    }
    if (layer->weight_types_count > 0 && trainable) {
        if (layer->weights == NULL) {
            PSErr(NULL, "Layer[%d]: layer has no weights", layer->index);
            return 0;
        }
        int trainable_params = 0xFFFF;
        if (layer->type == Attention)
            trainable_params = PSGetAttentionTrainableParameters(layer);
        for (int i = 0; i < layer->weight_types_count; i++) {
            int ok = layer->weights[i] != NULL;
            if (!ok) ok = !(trainable_params & (1 << i));
            if (!ok) {
                PSErr(NULL, "Layer[%d]: weights[%d] are NULL", layer->index, i);
                return 0;
            }
        }
    } else if (trainable) {
        if (layer->type != Pooling && layer->type != Dropout &&
            layer->type != OperatorLayer)
        {
            PSErr(NULL, "Layer[%d]: weights required for layer type %s",
                  PSGetLabelForType(layer->type));
            return 0;
        }
    }
    return 1;
}

void handleLayerForwardDebug(PSLayer *layer, const char *func,
                             PSMathOpts *opts)
{
#ifdef PS_DEBUG_MODE
    PSAddContextualDebug(layer->model, layer, NULL, NULL, "forward", 0);
#endif
    PSDebugStepInfo dbginfo =
        {.model = layer->model, .layer = layer, .func = func};
    if (opts != NULL && PSShouldDebugDump(layer->model)) {
        opts->data = &dbginfo;
        opts->debug_step = dumpForwardStep;
    }
}

/* Onehot input layers only have one input corresponding to the index
 * of the activated unit. In this case, just take the value of the
 * corresponding weight, since the input should always be considered
 * as it would be 1 */
int PSOnehotInputsForward(PSLayer *layer, int weights_index,
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
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
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
            PSAddVectors(out, layer->biases, out, layer->size, &opts);
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
            PSFloat w = weights[(i * layer->size) + onehot_idx];
            outputs[i] = w;
            if (use_bias) outputs[i] += layer->biases[i];
        }
        if (do_activate)
            layer->activate(outputs, NULL, layer->size, &opts);
    }
    return 1;
}

int PSFullForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSModel *model = layer->model;
    PSLayer *previous = PSGetPreviousLayer(layer);
    PSMathOpts opts = {.acceleration = model->acceleration};
    handleLayerForwardDebug(layer, __func__, &opts);
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 0, t = 0;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    if (previous->flags & FLAG_ONEHOT) {
        /* Onehot inputs */
        if (!PSOnehotInputsForward(layer, 0, NULL, t, 1, 1)) return 0;
        goto final;
    }
    PSMatrix weights = layer->weights[0];
    opts.acceleration = model->acceleration;
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
            PSAddVectors(outputs, layer->biases, outputs, layer->size, &opts);
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
        opts.transpose = 0;
        opts.store_mode = PS_STORE_MODE_SET;
        if (layer->activate)
            layer->activate(layer->states, NULL, seqlen * layer->size, &opts);
    }
final:
    return 1;
}

static int softmaxForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSMathOpts opts = {0};
    handleLayerForwardDebug(layer, __func__, &opts);
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 0, t = 0;
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    if (!PSFullForward(layer, seqlen, t)) return 0;
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

static int parseSequenceData(PSModel *model, PSFloat *data,
                             int sequence_index, int flags, int backprop,
                             int *x_seqlen, PSFloat **x,
                             int *y_seqlen, PSFloat **y)
{
    int datalen = 0, xlen = 0, ylen = 0;
    if (x_seqlen != NULL) *x_seqlen = 0;
    if (y_seqlen != NULL) *y_seqlen = 0;
    if (x != NULL) *x = NULL;
    if (y != NULL) *y = NULL;
    PSModel *input_model = model, *output_model = model;
    int num_models = PSModelChainLength(model);
    if (num_models > 1) {
        input_model = PSModelChainHead(model);
        output_model = PSModelChainTail(model);
        assert(input_model != NULL && output_model != NULL);
    }
    PSLayer *input_layer = input_model->layers[0],
            *output_layer = output_model->layers[output_model->size - 1];
    assert(input_layer != NULL && output_layer != NULL);
    int input_seq = PSUseSequences(input_layer),
        output_seq = PSUseSequences(output_layer);
    if (!input_seq && !output_seq) return 0;
    int input_size = input_layer->size, output_size = output_layer->size;
    if (output_layer->flags & FLAG_ONEHOT) output_size = 1;
    int autoregression = output_model->flags & FLAG_AUTOREGRESSION;
    int seq2seq = backprop && (flags & TRAINING_FLAG_SEQ2SEQ);
    if (seq2seq && !autoregression) return 0;
    else if (!seq2seq && autoregression) seq2seq = 1;
    if (!input_seq && seq2seq) return 0;
    int seqlen_nelems = (seq2seq ? 2 : 1),
        selfsupervised = flags & TRAINING_FLAG_SELFSUPERVISED;
    PSFloat *xsize_p = NULL, *ysize_p = NULL, *xp = NULL, *yp = NULL;
    if (input_seq) {
        xsize_p = data;
        xlen = (int) *xsize_p;
        xp = xsize_p + 1;
    } else {
        xlen = 1;
        xp = data;
    }
    if (xsize_p != NULL && xlen <= 0) {
        PSErr(
            NULL, "Sequence[%d] Invalid length %d at data offset %d",
            sequence_index, xlen, (int) (xsize_p - data)
        );
        return 0;
    }
    if (x_seqlen != NULL) *x_seqlen = xlen;
    if (x != NULL) *x = xp;
    int xdatalen = (xlen * input_size), ydatalen = 0;
    if (backprop) {
        if (seq2seq && !selfsupervised) {
            /* x_seqlen[1] | X[xdatalen] | y_seqlen[1] | Y[ydatalen] */
            ysize_p = data + 1 + xdatalen;
            yp = ysize_p + 1;
        } else if (!input_seq) {
            /* X[xdatalen] | y_seqlen[1] | Y[ydatalen] */
            ysize_p = data + xdatalen;
            yp = ysize_p + 1;
        }
        if (ysize_p != NULL) {
            ylen = (int) *ysize_p;
            if (ylen <= 0) {
                PSErr(
                    NULL, "Sequence[%d] Invalid length %d at data offset %d",
                    sequence_index, ylen, (int) (ysize_p - data)
                );
                return 0;
            }
        } else {
            ylen = xlen;
            yp = data + 1 + xdatalen;
        }
        ydatalen = ylen * output_size;
        if (y_seqlen != NULL) *y_seqlen = ylen;
        if (y != NULL) *y = yp;
    }
    datalen = xdatalen + ydatalen + seqlen_nelems;
    return datalen;
}

static PSFloat **getDatasetSequences(PSModel *model, PSFloat *data,
                                     int sequence_count, int flags)
{
    PSModel *input_model = model, *output_model = model;
    int num_models = PSModelChainLength(model);
    if (num_models > 1) {
        input_model = PSModelChainHead(model);
        output_model = PSModelChainTail(model);
        assert(input_model != NULL && output_model != NULL);
    }
    PSLayer *input_layer = input_model->layers[0],
            *output_layer = output_model->layers[output_model->size - 1];
    assert(input_layer != NULL && output_layer != NULL);
    int input_is_seq = PSUseSequences(input_layer),
        output_is_seq = PSUseSequences(output_layer);
    if (!input_is_seq && !output_is_seq) {
        PSErr(NULL, "model does not use sequences");
        return NULL;
    }
    PSFloat **sequences = malloc(sizeof(PSFloat *) * sequence_count);
    if (sequences == NULL) {
        PSErr(NULL, "could not allocate memory for recurrent sequences!");
        return NULL;
    }
    PSFloat *seqdata = data;
    for (int i = 0; i < sequence_count; i++) {
        int datalen = parseSequenceData(
            model, seqdata, i, flags, 1, NULL, NULL, NULL, NULL
        );
        if (datalen <= 0) goto fail;
        sequences[i] = seqdata;
        seqdata += datalen;
    }
    return sequences;
fail:
    free(sequences);
    return NULL;
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
    PSFloat max = PSFLOAT_MIN;
    int max_idx = -1;
    PSFloat *states = PSGetStates(layer, seqidx);
    if (states == NULL) return 0;
    for (i = 0; i < layer->size; i++) {
        PSFloat state = states[i];
        if (max_idx < 0 || state > max) {
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
        case Attention:
            return "Attention";
        case OperatorLayer:
            return "Operator Layer";
        case Linear:
            return "Linear";
        case PositionalEncoding:
            return "Positional Encoding";
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

char *getModelStatusLabel(PSModel *model) {
    if (model == NULL) return "";
    int status = PSModelGetStatus(model);
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
    else if (optimization == PSRMSPropOptimization) return "RMSProp";
    return "UNKOWN";
}

PSScalarActivationFunction PSGetScalarActivationFunc(PSActivationFunction func){
    if (func == PSSigmoid) return PSSigmoidS;
    else if (func == PSTanhActivation) return PSTanhS;
    else if (func == PSRelu) return PSReluS;
    else if (func == PSGelu) return PSGeluS;
    else if (func == PSSigmoidDerivative) return PSSigmoidDerivativeS;
    else if (func == PSTanhDerivative) return PSTanhDerivativeS;
    else if (func == PSReluDerivative) return PSReluDerivativeS;
    else if (func == PSGeluDerivative) return PSGeluDerivativeS;
    return NULL;
}

const char *PSGetActivationName(PSActivationFunction func) {
    if (func == PSSigmoid) return "sigmoid";
    else if (func == PSTanhActivation) return "tanh";
    else if (func == PSRelu) return "relu";
    else if (func == PSGelu) return "gelu";
    return NULL;
}

PSActivationFunction PSGetActivationDerivative(PSActivationFunction func) {
    if (func == PSSigmoid) return PSSigmoidDerivative;
    else if (func == PSTanhActivation) return PSTanhDerivative;
    else if (func == PSRelu) return PSReluDerivative;
    else if (func == PSGelu) return PSGeluDerivative;
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
            else if (Attention == layer->type)
                bias_count = (PS_SCORES_IDX * layer->size) + 1;
            else bias_count = layer->size;
            count += bias_count;
        }
    }
    return count;
}

uint64_t PSGeModelParametersCount(PSModel *model) {
    int tot = 0, param_type = (PARAM_TYPE_BIAS | PARAM_TYPE_WEIGHT), i;
    if (model->layers == NULL || model->size == 0) return 0;
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
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
    if (layer->model == NULL) return NULL;
    if (layer->model->layers == NULL) return NULL;
    int previous_layer_idx = layer->index - 1;
    if (previous_layer_idx < 0) return NULL;
    if (previous_layer_idx >= layer->model->size) return NULL;
    return layer->model->layers[previous_layer_idx];
}

PSLayer *PSGetNextLayer(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (layer->model == NULL) return NULL;
    if (layer->model->layers == NULL) return NULL;
    int next_layer_idx = layer->index + 1;
    if (next_layer_idx < 0) return NULL;
    if (next_layer_idx >= layer->model->size) return NULL;
    return layer->model->layers[next_layer_idx];
}

PSLayer *PSGetOutputLayer(PSModel *model) {
    if (model == NULL || model->layers == NULL || model->size == 0)
        return NULL;
    if (PSIsModelChain(model)) {
        model = PSModelChainTail(model);
        if (model == NULL) {
            PSErrNN(__func__, model, NULL, "broken model chain");
            return NULL;
        }
    }
    return model->layers[model->size - 1];
}

PSLayer *PSGetLayerByIndex(PSModel *model, int layer_index, int model_index) {
    if (model == NULL) return NULL;
    if (PSIsModelChain(model)) {
        int num_models = PSModelChainLength(model);
        model = PSModelChainHead(model);
        if (model == NULL || num_models <= 0) {
            PSErrNN(__func__, model, NULL, "broken model chain");
            return NULL;
        }
        if (model_index < 0) {
            model_index = num_models + model_index;
        }
        if (model_index < 0 || model_index >= num_models) return NULL;
        PSModel *current = model;
        while (current != NULL && current->index != model_index)
            current = current->next;
        if (current == NULL) return NULL;
        model = current;
    } else if (model_index > 0) {
        PSErr(__func__, "invalid model index %d for non chained model %d",
              model_index, model->index);
        return NULL;
    }
    if (layer_index < 0) layer_index = model->size + layer_index;
    if (layer_index < 0 || layer_index >= model->size) return NULL;
    return model->layers[layer_index];
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
    static int min_indent = 0;
    if (min_indent == 0) min_indent = strlen("  Layer[]: ");
    int indent = min_indent + PSCalcIntStringLength(layer->index);
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
        if (ltype == Convolutional) {
            printf(", filter = %dx%dx%d, stride = %d",
                   filter_w, filter_h, filter_d, stride);
        } else {
            printf(", filter = %dx%d, stride = %d", filter_w, filter_h, stride);
        }
        if (ltype == Convolutional) {
            if (padding < 0) padding = 0;
            printf(", padding = %d", padding);
        }
    } else if (ltype == FullyConnected && layer->output_depth > 1) {
       printf(", depth = %d", layer->output_depth);
    } else if (ltype == Attention) {
        PSAttentionType type = PSGetAttentionType(layer);
        if (type == PSAdditiveAttention)
            printf(", attention type = additive");
        else if (type == PSDotAttention)
            printf(", attention type = dot");
        int nheads = PSGetAttentionHeadCount(layer);
        if (nheads > 1) printf(", heads = %d", nheads);
        PSLayer *qprov = NULL, *kprov = NULL, *vprov = NULL;
        int ok = PSGetAttentionProviders(layer, &qprov, &kprov, &vprov), i;
        if (ok) {
            PSLayer *providers[] = {qprov, kprov, vprov};
            char *names[] = {"query", "keys", "values"};
            printf("\n");
            for (i = 0; i < 3; i++) {
                PSLayer *provider = providers[i];
                int nc = printf("%*c%s provider: ", indent, ' ', names[i]);
                int vindent = 30 - nc;
                if (provider != NULL) {
                    printf("%*c%d:%d (%s)%s", vindent, ' ',
                           provider->model->index, provider->index,
                           PSGetLayerTypeLabel(provider), (i < 2 ? "\n" : ""));
                } else printf("%*cnull%s", vindent,' ',(i < 2 ? "\n" : ""));
            }
        } else printf("\n%*c%s", indent, ' ', "no providers");
    } else if (ltype == Embedding) {
        int vocab_size = PSGetEmbeddingVocabularySize(layer);
        if (vocab_size > 0)
            printf(", vocabulary_size = %d", vocab_size);
    } else if (ltype == OperatorLayer) {
        PSOperatorType op = PSGetOperatorLayerType(layer);
        printf(", operator = %s", PSGetOperatorLayerTypeLabel(op));
        int prv_count = 0, i;
        PSLayer **providers = PSGetOperatorLayerProviders(layer, &prv_count);
        printf(", providers = %d", prv_count);
        if (prv_count > 0) {
            printf("\n");
            int last_idx = prv_count - 1;
            for (i = 0; i < prv_count; i++) {
                PSLayer *provider = providers[i];
                printf("%*c[%d]", indent, ' ', i);
                if (provider != NULL) {
                    printf(" %d:%d (%s)%s", provider->model->index,
                           provider->index,
                           PSGetLayerTypeLabel(provider),
                           (i < last_idx ? "\n" : ""));
                } else printf(" null%s", (i < last_idx ? "\n" : ""));
            }
        }
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

void PSModelPrintInfo(PSModel *model) {
    if (model == NULL) return;
    const char *name = model->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED MODEL";
    int count = PSModelChainLength(model),
        print_chain = (count > 1);
    if (print_chain)
        printf(PSCOLOR_BOLD "Model[%d]\n" PSCOLOR_RESET, model->index);
    printInfoRow("Name", "\"%s\"", name);
    printInfoRow("Size", "%d", model->size);
    int is_recurrent = PSIsRecurrent(model);
    PSRecurrentNetworkMode mode = model->rnn_mode;
    PSModelContext *ctx = model->context;
    assert(ctx != NULL);
    PSLayer *first_recurrent_layer = NULL, *last_recurrent_layer = NULL;
    if (is_recurrent || mode != NonRecurrent) {
        first_recurrent_layer = PSGetFirstRecurrentLayer(model);
        last_recurrent_layer = PSGetLastRecurrentLayer(model);
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
                 PSGeModelParametersCount(model));
    char *loss_name = getLossFunctionName(model->loss);
    if (loss_name != NULL) printInfoRow("Loss Function", "%s", loss_name);
    printInfoRow("Status", "%s", getModelStatusLabel(model));
    printInfoRow("AVX", "%s",
                 (PSAVXEnabled(model->acceleration) ? "yes" : "no"));
    printInfoRow("Apple Accelerate Framework", "%s",
                 (PSACFEnabled(model->acceleration) ? "yes" : "no"));
    printInfoRow("BLAS", "%s",
                 (PSBLASEnabled(model->acceleration) ? "yes" : "no"));
    if (PSLogColorEnabled()) printf(PSCOLOR_CYAN);
    printf("Layers:\n");
    if (PSLogColorEnabled()) printf(PSCOLOR_RESET);
    int i;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        printf("  ");
        PSPrintLayerInfo(layer);
    }
    if (model->previous == NULL) {
        PSModel *next = model->next;
        while (next != NULL) {
            PSModelPrintInfo(next);
            next = next->next;
        }
    }
}

/* Loss Functions */

PSFloat PSQuadraticLoss(PSFloat *outputs, PSFloat *expected, int size,
                        int onehot_size)
{
    PSFloat *_diffs;
    PSFloat diffs[size];
    if (!onehot_size) {
        int i;
        for (i = 0; i < size; i++) {
            PSFloat d = outputs[i] - expected[i];
            diffs[i] = d;
            if (isnan(d)) {
                fprintf(stderr,
                    "\n\nPSQuadraticLoss: diffs[%d] is nan!\n"
                    " -> output=%g, expected=%g\n",
                    i, outputs[i], expected[i]
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

PSFloat PSCrossEntropyLoss(PSFloat *outputs, PSFloat *expected, int size,
                        int onehot_size)
{
    PSFloat loss = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        PSFloat o = outputs[i];
        if (onehot_size) loss += (PSMathLog(o + PSFLOAT_EPS));
        else {
            PSFloat y = expected[i];
            loss += (y * PSMathLog(o + PSFLOAT_EPS) +
                    (1 - y) * PSMathLog((1 - o) + PSFLOAT_EPS));
        }
    }
    loss *= -1;
    return loss;
}

/* Neural Network Functions */

void PSModelSetStatus(PSModel *model, int status, int *old) {
    if (model == NULL) return;
    if (old != NULL) *old = model->status;
    model->status = status;
    if (PSIsModelChain(model)) {
        PSModel *head = PSModelChainHead(model);
        if (head != NULL && head != model) {
            head->status = status;
        }
    }
}

int PSModelGetStatus(PSModel *model) {
    if (model == NULL) return 0;
    if (PSIsModelChain(model)) model = PSModelChainHead(model);
    if (model == NULL) return 0;
    return model->status;
}

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
    /* If `retain_previous` is true, it means that the last vector in
     * current sequence states must be retained as 'initial state'.
     * This is done by adding one further row to states matrix that will
     * hold the 'initial' vector from previous last vector. */
    int cur_seqlen =  0;
    if (current == layer->states)
        cur_seqlen = PSStateSequenceLength(layer);
    else if (current != NULL) {
        cur_seqlen = PSMatrixDim(current, 0);
        if (previous && *previous != NULL) cur_seqlen--;
    }
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
            PSErrNN(
                NULL, NULL, layer,
                "could not initialize layer states (seqlen = %d)", seqlen
            );
            return NULL;
        }
        PSVectorCopy(initial, last, layer->size);
        if (previous != NULL) *previous = initial;
    } else if (previous != NULL) *previous = NULL;
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
    int nrows = seqlen, cur_seqlen, cur_nrows;
    if (current == layer->states)
        cur_seqlen = PSStateSequenceLength(layer);
    else {
        if (current == NULL) {
            if (previous != NULL) *previous = NULL;
            return NULL;
        }
        cur_seqlen = PSMatrixDim(current, 0);
        if (previous && *previous != NULL) cur_seqlen--;
    }
    cur_nrows = cur_seqlen;
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
        if (previous != NULL) *previous = layer->initial_states;
        return current;
    }
    PSMatrix states = PSMatrixExpand(current, steps2add, 0);
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
    if (previous && *previous != NULL) {
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
    PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
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
        PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
        return 0;
    }
    layer->states = hstates;
    if (layer->on_states_resize != NULL)
        if (!layer->on_states_resize(layer, seqlen, cur_seqlen)) return 0;
    return 1;
}

int PSResetLayerStateSequence(PSLayer *layer, uint32_t seqlen,
                              int retain_previous)
{
    if (layer == NULL) return 0;
    if (!PSUseSequences(layer)) return 0;
    return PSInitLayerStates(layer, seqlen, retain_previous);
}

int PSResetModelStateSequences(PSModel *model, uint32_t seqlen,
                               int retain_previous)
{
    if (model == NULL) return 0;
    if (!PSModelIsBuilt(model)) {
        PSErr(__func__, "model is not build");
        return 0;
    }
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) continue;
        if (!PSResetLayerStateSequence(layer, seqlen, retain_previous)) {
            PSErr(__func__, "Failed to reset states sequence on layer %d", i);
            return 0;
        }
    }
    return 1;
}

int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t) {
    if (!PSUseSequences(layer)) return 1;
    if (seqlen < 1) {
        PSErr(__func__, "Layer[%d]: sequence length must be >= 1 (found %d)",
              layer->index, seqlen);
        return 0;
    }
    int cur_seqlen = PSStateSequenceLength(layer);
    if (PSIsRecurrent(layer) && t >= (int) cur_seqlen) {
        /* Recurrent layers may need to resize their sequence steps before
         * forward phase if step `t` is beyond current sequence length. */
        if (!PSResizeLayerStates(layer, t + 1)) {
            if (layer->model != NULL)
                PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
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
                if (layer->model)
                    PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
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

PSFloat *PSGetOutputs(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (PSUseSequences(layer)) {
        int seqlen = PSStateSequenceLength(layer);
        return PSGetStates(layer, seqlen - 1);
    }
    return PSGetStates(layer, 0);
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

PSLayer *PSGetFirstRecurrentLayer(PSModel *model) {
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) return NULL;
    return ctx->first_recurrent_layer;
}

PSLayer *PSGetLastRecurrentLayer(PSModel *model) {
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) return NULL;
    return ctx->last_recurrent_layer;
}

int PSModelIsBuilt(PSModel *model) {
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) return 0;
    return ctx->built;
}

static int discoverFirstLastRecurrentLayers(PSModel *model,
                                            PSLayer **first_recurrent,
                                            PSLayer **last_recurrent)
{
    PSLayer *first = PSGetFirstRecurrentLayer(model),
            *last = PSGetLastRecurrentLayer(model);
    if (first != NULL && last != NULL) {
        if (first_recurrent != NULL) *first_recurrent = first;
        if (last_recurrent != NULL) *last_recurrent = last;
    }
    if (first_recurrent != NULL) *first_recurrent = NULL;
    if (last_recurrent != NULL) *last_recurrent = NULL;
    PSLayer *last_rec = NULL;
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        assert(layer != NULL);
        if (PSIsRecurrent(layer)) {
            last_rec = layer;
            if (first == NULL) {
                first = layer;
                setModelContext(model, first_recurrent_layer, first);
                continue;
            }
        }
    }
    if (last == NULL) {
        if (last_rec != NULL) {
            last = last_rec;
            setModelContext(model, last_recurrent_layer, last);
        } else {
            if (first_recurrent != NULL) *first_recurrent = first;
            PSErr(NULL, "Unable to determine last recurrent layer");
            return 0;
        }
    }
    if (first_recurrent != NULL) *first_recurrent = first;
    if (last_recurrent != NULL) *last_recurrent = last;
    return 1;
}

static void updateModelForRecurrentMode(PSModel *model,
                                        PSRecurrentNetworkMode mode)
{
    PSModelContext *ctx = getModelContext(model);
    assert(ctx != NULL);
    if (mode == ManyToOne || mode == OneToMany) {
        ctx->first_recurrent_layer = NULL;
        ctx->last_recurrent_layer = NULL;
    }
    int recurrent_input = (mode == ManyToMany || mode == ManyToOne),
        recurrent_output = (mode == ManyToMany || mode == OneToMany),
        last_layer_idx = (model->size - 1), i, j;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
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
                        model->layers[j]->flags |= FLAG_RECURRENT;
                }
            } else if (Recurrent != type && LSTM != type && GRU != type) {
                model->layers[i]->flags &= (unsigned) (~FLAG_RECURRENT);
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

int PSModelBuild(PSModel *model) {
    if (model == NULL) {
        PSErr(__func__, "model is null");
        return 0;
    }
    if (model->size == 0) {
        PSErr(__func__, "model is empty");
        return 0;
    }
    if (model->layers == NULL) {
        PSErr(__func__, "model has no layers");
        return 0;
    }
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) {
        ctx = model->context = calloc(1, sizeof(PSModelContext));
        if (ctx == NULL) {
            PSPrintMemoryErrorMsg();
            PSErr(__func__, "cannot build model");
            return 0;
        }
    } else if (ctx->built) return 1;
    int first_whole_seq_layer = -1, i;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) {
            PSErrNN(__func__, model, layer, "null layer");
            return 0;
        }
        if (PSIsLayerPlaceholder(layer)) {
            layer->index = i;
            PSLayer *resolved = PSResolveLayerPlaceholder(layer, model);
            if (resolved == NULL) {
                PSErrNN(__func__, model, layer,
                        "could not resolve layer placeholder");
                return 0;
            }
            PSDeleteLayer(layer);
            model->layers[i] = resolved;
        } else if (layer->build != NULL) {
            if (!layer->build(layer)) return 0;
        }
        if (PSHandleSequenceAtOnce(layer)) {
            if (first_whole_seq_layer < 0) first_whole_seq_layer = i;
            model->flags |= FLAG_USE_SEQUENCES;
        } else if (PSHandleSequenceAtOnce(model)) {
            if (first_whole_seq_layer < 0) first_whole_seq_layer = i;
            layer->flags |= FLAG_USE_SEQUENCES;
            if (PSIsRecurrent(layer)) {
                PSErrNN(__func__, model, layer, "model uses whole "
                        "sequences but layer is recurrent");
                return 0;
            }
        }
    }
    int is_recurrent = PSIsRecurrent(model);
    PSRecurrentNetworkMode mode = model->rnn_mode;
    PSLayer *input_layer = model->layers[0],
            *output_layer = model->layers[model->size - 1];
    if (is_recurrent) {
        if (first_whole_seq_layer >= 0) {
            PSErr(__func__, "model has both recurrent layers and whole "
                  "sequence layers");
            return 0;
        }
        if (mode != NonRecurrent) updateModelForRecurrentMode(model, mode);
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
            updateModelForRecurrentMode(model, mode);
        }
    }
    if (model->loss == NULL) {
        if (output_layer->type == SoftMax) model->loss = PSCrossEntropyLoss;
        else model->loss = PSQuadraticLoss;
    }
    if (model->previous != NULL || model->next != NULL) {
        PSModel *head = model;
        while (head->previous != NULL) head = head->previous;
        updateModelChain(head);
    }
    ctx->built = 1;
    if (model->previous == NULL) {
        PSModel *cur = model->next;
        while (cur != NULL) {
            int built = PSModelBuild(cur);
            if (!built) {
                PSErr(__func__, "failed to build model[%d]",
                      model->index);
                return 0;
            }
            cur = cur->next;
        }
    }
    return 1;
}

int PSModelRebuild(PSModel *model) {
    if (model == NULL) {
        PSErr(__func__, "model is null!");
        return 0;
    }
    if (PSModelIsBuilt(model)) {
        PSModelContext *ctx = getModelContext(model);
        ctx->built = 0;
    }
    return PSModelBuild(model);
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

int PSSetRecurrentNetworkMode(PSModel *model,
                              PSRecurrentNetworkMode mode)
{
    if (model == NULL) {
        PSErr(__func__, "model is null");
        return 0;
    }
    if (model->size == 0) goto final;
    if (mode == NonRecurrent && PSIsRecurrent(model)) {
        PSErr(
            __func__, "invalid mode NonRecurrent: model is already "
            "recurrent"
        );
        return 0;
    }
    updateModelForRecurrentMode(model, mode);
final:
    model->rnn_mode = mode;
    if (mode != NonRecurrent) model->flags |= FLAG_RECURRENT;
    return 1;
}

PSModel *PSModelCreate(const char* name) {
    PSModel *model = (malloc(sizeof(PSModel)));
    if (model == NULL) {
        return NULL;
    }
    model->context = calloc(1, sizeof(PSModelContext));
    if (model->context == NULL) goto memory_err;
    model->name = name;
    model->size = 0;
    model->index = 0;
    model->layers = NULL;
    model->input_size = 0;
    model->output_size = 0;
    model->status = STATUS_UNTRAINED;
    model->flags = FLAG_NONE;
    model->acceleration = PSGlobalAcceleration;
    model->loss = PSQuadraticLoss;
    model->training = NULL;
    model->beforeForward = NULL;
    model->beforeBackprop = NULL;
    model->onEpochTrained = NULL;
    model->onBatchTrained = NULL;
    model->previous = NULL;
    model->next = NULL;
    model->previous_model_link = NULL;
    model->rnn_mode = NonRecurrent;
    PSSequenceSettings *sequence_settings = &(model->sequence_settings);
    memset(sequence_settings, 0, sizeof(PSSequenceSettings));
    sequence_settings->end = -1;
    return model;
memory_err:
    if (model != NULL) PSModelDelete(model);
    PSErr(__func__, "could not allocate memory for model");
    return NULL;
}

PSModelLink *findLinkToPreviousModel(PSModel *model, PSModel *previous) {
    if (model == NULL || previous == NULL) return NULL;
    PSModelLink *link = NULL;
    int i, j, prev_output_idx = previous->size - 1;
    if (prev_output_idx < 0) {
        PSErr(NULL, "model %s is empty", previous->name);
        return NULL;
    }
    PSLayer *input_layer = NULL, *output_layer = NULL;
    for (i = 0; i < model->size; i++) {
        int is_input = (i == 0);
        PSLayer *layer = model->layers[i];
        int input_size = layer->size, onehot_size = 0;
        if (is_input && layer->flags & FLAG_ONEHOT)
            onehot_size = PSGetOneHotLayerVectorSize(layer);
        for (j = prev_output_idx; j >= 0; j--) {
            PSLayer *prev_layer = previous->layers[j];
            int output_size = prev_layer->size;
            if (input_size == output_size || onehot_size == output_size) {
                input_layer = layer;
                output_layer = prev_layer;
                break;
            }
        }
    }
    if (input_layer != NULL && output_layer != NULL) {
        link = malloc(sizeof(*link));
        if (link == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
        link->layer = input_layer;
        link->previous_layer = output_layer;
    }
    return link;
}

int checkModelLink(PSModelLink *link) {
    if (link == NULL) return 0;
    PSLayer *input_layer = link->layer;
    PSLayer *output_layer = link->previous_layer;
    if (input_layer == NULL) {
        PSErr(NULL, "missing layer in PSModelLink");
        return 0;
    }
    if (output_layer == NULL) {
        PSErr(NULL, "missing previous_layer in PSModelLink");
        return 0;
    }
    PSModel *input_model = input_layer->model;
    PSModel *output_model = output_layer->model;
    if (input_model == NULL) {
        PSErr(NULL, "PSModelLink layer has no model");
        return 0;
    }
    if (output_model == NULL) {
        PSErr(NULL, "PSModelLink previous_layer has no model");
        return 0;
    }
    if (output_model->index >= input_model->index) {
        PSErr(NULL, "PSModelLink previous_layer's model must "
              "precede layer's model");
        return 0;
    }
    int ok = (input_layer->size == output_layer->size);
    if (!ok && input_layer->index == 0 && input_layer->flags & FLAG_ONEHOT) {
        int onehot_vector_size = PSGetOneHotLayerVectorSize(input_layer);
        ok = (onehot_vector_size == output_layer->size);
    }
    return ok;
}

int updateModelChain(PSModel *head) {
    if (head == NULL) return 0;
    while (head->previous != NULL) head = head->previous;
    PSModel *cur = head, *last = NULL;
    int count = 0;
    while (cur != NULL) {
        count++;
        PSModelContext *ctx = getModelContext(cur);
        if (ctx == NULL) {
            ctx = cur->context = calloc(1, sizeof(*ctx));
            if (ctx == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        setModelContext(cur, head_model, head);
        last = cur;
        cur = cur->next;
    }
    cur = last;
    while (cur != NULL) {
        PSModelContext *ctx = getModelContext(cur);
        if (ctx == NULL) {
            ctx = cur->context = calloc(1, sizeof(*ctx));
            if (ctx == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        setModelContext(cur, last_model, last);
        setModelContext(cur, model_chain_length, count);
        cur = cur->previous;
    }
    return 1;
}

int PSAddModel(PSModel *parent, PSModel *model, PSModelLink *link) {
    if (parent == NULL || model == NULL) {
        PSErr(__func__, "`parent` and `model` cannot be null");
        return 0;
    }
    if (model->previous != NULL) {
        PSErr(__func__, "`model` already has previous model");
        return 0;
    }
    PSModel *prev = parent;
    while (prev->next != NULL) prev = prev->next;
    while (parent->previous != NULL) parent = parent->previous;
    if (prev->next != NULL) {
        PSErr(__func__, "previous model already has next model");
        return 0;
    }
    model->index = prev->index + 1;
    model->previous = prev;
    prev->next = model;
    if (link == NULL) {
        link = findLinkToPreviousModel(model, prev);
        if (link == NULL) {
            PSErr(__func__, "could not automatically find link from "
                  "model %d to modelk %d", model->index, model->index-1);
            goto fail;
        }
        model->previous_model_link = link;
    } else {
        if (!checkModelLink(link)) {
            PSErr(__func__, "provided model link is not valid");
            goto fail;
        }
        model->previous_model_link = malloc(sizeof(*link));
        if (model->previous_model_link == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        memcpy(model->previous_model_link, link, sizeof(*link));
    }
    if (!updateModelChain(parent)) {
        PSErr(__func__, "broken model chain");
        goto fail;
    }
    return 1;
fail:
    model->previous = NULL;
    prev->next = NULL;
    free(model->previous_model_link);
    model->previous_model_link = NULL;
    return 0;
}

int cloneModelChain(PSModel *model, PSModel *clone, int layout_only) {
    assert(model != NULL && model->previous == NULL);
    PSModel *next = model->next, *clone_next = NULL;
    while (next != NULL) {
        if (next->previous == NULL) {
            PSErr("PSModelClone", "broken model chain");
            return 0;
        }
        clone_next = cloneModel(next, layout_only, clone);
        if (clone_next == NULL) return 0;
        if (clone_next->previous != NULL) {
            PSModel *prev = clone_next->previous;
            if (prev != NULL && prev->next == clone_next) prev->next = NULL;
            clone_next->previous = NULL;
        }
        PSModelLink *link = next->previous_model_link,
                            *clone_link = NULL;
        if (link != NULL) {
            if (link->layer == NULL || link->previous_layer == NULL ||
                link->previous_layer->model == NULL)
            {
                PSErr("PSModelClone", "invalid link in model %d",
                      next->index);
                PSModelDelete(clone_next);
                return 0;
            }
            if (link->layer->index >= clone_next->size ||
                clone_next->layers[link->layer->index] == NULL)
            {
                PSErr("PSModelClone", "missing layer %d in cloned model %d",
                      link->layer->index, next->index);
                PSModelDelete(clone_next);
                return 0;
            }
            PSModel *prev_model = NULL, *cur = clone;
            int prev_idx = link->previous_layer->model->index;
            while (prev_idx >= 0) {
                if (cur->index == prev_idx--) {
                    prev_model = cur;
                    break;
                }
                cur = cur->previous;
                if (cur == NULL) break;
            }
            if (prev_model == NULL) {
                PSErr("PSModelClone", "could not find linked model %d",
                      prev_idx);
                PSModelDelete(clone_next);
                return 0;
            }
            if (link->previous_layer->index >= prev_model->size ||
                prev_model->layers[link->previous_layer->index] == NULL)
            {
                PSErr("PSModelClone", "missing layer %d in cloned model %d",
                      link->previous_layer->index, prev_model->index);
                PSModelDelete(clone_next);
                return 0;
            }
            clone_link = malloc(sizeof(*clone_link));
            if (clone_link == NULL) {
                PSPrintMemoryErrorMsg();
                PSModelDelete(clone_next);
                return 0;
            }
            clone_link->layer = clone_next->layers[link->layer->index];
            clone_link->previous_layer =
                prev_model->layers[link->previous_layer->index];
        }
        if (clone_link == NULL) {
            PSErr("PSModelClone", "could not make link for cloned model %d",
                  next->index);
            PSModelDelete(clone_next);
            return 0;
        }
        int added = PSAddModel(clone, clone_next, clone_link);
        if (!added) {
            PSErr("PSModelClone", "unable to add cloned model %d",
                  next->index);
            PSModelDelete(clone_next);
            free(clone_link);
            return 0;
        }
        free(clone_link);
        next = next->next;
    }
    return 1;
}

static PSModel *cloneModel(PSModel *model, int layout_only, PSModel *parent) {
    if (model == NULL) return NULL;
    int clone_next = 0;
    if (model->previous != NULL || model->next != NULL)
        clone_next = model->previous == NULL;
    PSModel *clone = PSModelCreate(NULL);
    if (clone == NULL) goto memerr;
    int is_chain = (parent != NULL),
        is_child = (is_chain && model->index > 0);
    if (is_child) {
        clone->index = model->index;
        if (clone->previous == NULL) {
            PSModel *prev = NULL, *cur = parent;
            while (cur != NULL) {
                if (cur->index == (model->index - 1)) {
                    prev = cur;
                    break;
                }
                cur = cur->next;
            }
            if (prev == NULL)
                PSWarn("could not find previous model in model chain");
            else {
                clone->previous = prev;
                prev->next = clone;
            }
        }
    }
    if (!layout_only) {
        clone->status = model->status;
        if (model->training != NULL) {
            clone->training = malloc(sizeof(PSTrainingInfo));
            if (clone->training == NULL) goto memerr;
            clone->training->current_epoch = model->training->current_epoch;
            clone->training->current_batch = model->training->current_batch;
            clone->training->current_element =
                model->training->current_element;
            clone->training->batch_size = model->training->batch_size;
            clone->training->started_at = model->training->started_at;
            clone->training->ended_at = model->training->ended_at;
            clone->training->debug_dump_to = NULL;
        }
    }
    clone->flags = model->flags;
    clone->acceleration = model->acceleration;
    clone->loss = model->loss;
    clone->rnn_mode = model->rnn_mode;

    int i, j;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        PSLayerType type = layer->type;
        PSLayerDef ldef = {
            .activation = layer->activate,
            .flags = layer->flags,
            .output_depth = layer->output_depth,
            .output_columns = layer->output_columns,
            .output_rows = layer->output_rows
        };
        if (Dropout == type) ldef.dropout = PSGetDropout(layer);
        else if (Convolutional == type || Pooling == type) {
            PSConvolutionalSettings *csettings =
                PSGetConvolutionalSettings(layer);
            ldef.stride = csettings->stride;
            ldef.padding = csettings->padding;
            ldef.filter_width = csettings->filter_width;
            ldef.filter_height = csettings->filter_height;
        } else if (Attention == type) {
            ldef.attention_type = PSGetAttentionType(layer);
            ldef.attention_heads = PSGetAttentionHeadCount(layer);
            ldef.causal_attention = PSIsCausalAttention(layer);
            ldef.attention_scale = PSGetAttentionScale(layer);
            ldef.trainable_parameters =
                PSGetAttentionTrainableParameters(layer);
            PSLayer *qprovider = NULL, *kprovider = NULL, *vprovider = NULL;
            int ok = PSGetAttentionProviders(layer, &qprovider, &kprovider,
                                             &vprovider), pidx;
            if (!ok) {
                PSErrNN(__func__, NULL, layer, "could not retrieve attention "
                        "provders");
                goto err;
            }
            PSLayer *srcproviders[] = {qprovider, kprovider, vprovider};
            PSLayer **dstproviders[] = {
                &ldef.query_provider, &ldef.keys_provider,&ldef.values_provider
            };
            for (pidx = 0; pidx < 3; pidx++) {
                PSLayer *srcprovider = srcproviders[pidx];
                if (srcprovider == NULL || srcprovider->model == NULL)
                    continue;
                int nidx = srcprovider->model->index,
                    lidx = srcprovider->index;
                PSLayer **dstprovider_p = dstproviders[pidx];
                PSLayer *provider = NULL;
                if (!is_chain)
                    provider = PSGetLayerByIndex(clone, lidx, nidx);
                else if (is_child && nidx < clone->index)
                    provider = PSGetLayerByIndex(parent, lidx, nidx);
                else if (is_child && nidx == clone->index && lidx<layer->index)
                    provider = PSGetLayerByIndex(clone, lidx, 0);
                if (provider == NULL) {
                    provider = PSMakeLayerPlaceholder(lidx, nidx);
                    if (provider == NULL) goto err;
                    provider->size = srcprovider->size;
                    provider->flags = srcprovider->flags;
                }
                *dstprovider_p = provider;
            }
        } else if (OperatorLayer == type) {
            int count = 0, p;
            ldef.operator = PSGetOperatorLayerType(layer);
            PSLayer **lproviders = PSGetOperatorLayerProviders(layer, &count);
            PSLayer *providers[PS_MAX_PROVIDERS] = {0};
            ldef.providers_count = count;
            if (lproviders == NULL && count > 0) {
                PSErrNN(__func__, NULL, layer, "layer should have %d "
                        "provider(s) but is missing providers at all", count);
                goto err;
            }
            for (p = 0; p < count; p++) {
                PSLayer *provider = lproviders[p];
                if (provider == NULL) {
                    providers[i] = NULL;
                    continue;
                }
                if (provider->model == NULL) {
                    PSErrNN(__func__, NULL, layer,
                            "provider[%d] has no model", p);
                    goto err;
                }
                int nidx = provider->model->index, lidx = provider->index;
                PSLayer *clone_provider = NULL;
                if (!is_chain)
                    clone_provider = PSGetLayerByIndex(clone, lidx, nidx);
                else if (is_child && nidx < clone->index)
                    clone_provider = PSGetLayerByIndex(parent, lidx, nidx);
                else if (is_child && nidx == clone->index && lidx<layer->index)
                    clone_provider = PSGetLayerByIndex(clone, lidx, 0);
                if (clone_provider == NULL) {
                    clone_provider = PSMakeLayerPlaceholder(
                        provider->index, provider->model->index
                    );
                    if (clone_provider == NULL) goto err;
                    clone_provider->size = provider->size;
                    clone_provider->flags = provider->flags;
                }
                providers[p] = clone_provider;
            }
            ldef.providers = providers;
        }
        ldef.pretrained = layer->pretrained;
        PSLayer *cloned_layer = PSAddLayer(clone, type, layer->size, &ldef);
        if (cloned_layer == NULL) {
            PSModelDelete(clone);
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
                        if (type == Attention) continue;
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
            if (layer->on_copy != NULL)
                if (!layer->on_copy(cloned_layer, layer)) goto err;
        }
    }
    if (clone->layers == NULL) {
        if (model->layers == NULL) return clone;
        else goto err;
    }
    clone->sequence_settings.max_length = model->sequence_settings.max_length;
    clone->sequence_settings.end = model->sequence_settings.end;
    if (model->context != NULL) {
        memcpy(clone->context, model->context, sizeof(PSModelContext));
        setModelContext(clone, built, 0);
        setModelContext(clone, head_model, NULL);
        setModelContext(clone, last_model, NULL);
        setModelContext(clone, model_chain_length, 1);
        PSLayer *first_recurrent = PSGetFirstRecurrentLayer(model);
        PSLayer *last_recurrent = PSGetLastRecurrentLayer(model);
        if (first_recurrent != NULL) {
            setModelContext(
                clone, first_recurrent_layer,
                clone->layers[first_recurrent->index]
            );
        } else setModelContext(clone, first_recurrent_layer, NULL);
        if (last_recurrent != NULL) {
            setModelContext(
                clone, last_recurrent_layer,
                clone->layers[last_recurrent->index]
            );
        } else setModelContext(clone, last_recurrent_layer, NULL);
        PSModelContext *ctx = model->context, *clone_ctx = clone->context;
        clone_ctx->sequence_start = NULL;
        if (model->sequence_settings.start != NULL) {
            clone_ctx->sequence_start = malloc(
                model->input_size * sizeof(PSFloat)
            );
            if (clone_ctx->sequence_start == NULL) goto memerr;
            PSVectorCopy(
                clone_ctx->sequence_start, model->sequence_settings.start,
                model->input_size
            );
            model->sequence_settings.start = clone_ctx->sequence_start;
        }
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
            for (i = 0; i < PS_MAX_MEMORY_GRADIENTS; i++) {
                PSGradient **memgrads = training_ctx->memory_gradients[i];
                if (memgrads != NULL) {
                    PSGradient **clone_memgrads = cloneModelGradients(
                        memgrads, model
                    );
                    if (clone_memgrads == NULL) goto err;
                    clone_training_ctx->memory_gradients[i] = clone_memgrads;
                } else {
                    if (clone_training_ctx->memory_gradients[i] != NULL) {
                        PSDeleteModelGradients(
                            clone_training_ctx->memory_gradients[i], model
                        );
                    }
                    clone_training_ctx->memory_gradients[i] = NULL;
                }
            }
        } else {
            if (clone_ctx->training_context != NULL)
                deleteTrainingContext(clone_ctx->training_context, clone);
            clone_ctx->training_context = NULL;
        }
    } else {
        if (clone->context != NULL)
            deleteModelContext(clone->context, clone);
        clone->context = NULL;
    }
    if (clone_next) {
        if (!cloneModelChain(model, clone, layout_only)) {
            PSErr(__func__, "unable to clone model chain");
            goto err;
        }
    }
    return clone;
memerr:
    PSPrintMemoryErrorMsg();
err:
    if (clone != NULL) PSModelDelete(clone);
    return NULL;
}

PSModel *PSModelClone(PSModel *model, int layout_only) {
    PSModel *clone =  cloneModel(model, layout_only, NULL);
    if (clone == NULL) return NULL;
    if (PSModelIsBuilt(model) && !PSModelBuild(clone)) {
        PSModelDelete(clone);
        return NULL;
    }
    return clone;
}

int PSModelChainLength(PSModel *model) {
    if (model == NULL) return 0;
    if (!PSIsModelChain(model)) return 1;
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL || ctx->model_chain_length <= 1) {
        if (!updateModelChain(model)) goto broken_chain;
        ctx = getModelContext(model);
        if (ctx == NULL || ctx->model_chain_length <= 1) goto broken_chain;
    }
    return ctx->model_chain_length;
broken_chain:
    PSErr(__func__, "broken model chain");
    return 0;
}

PSModel *PSGetModelAtIndex(PSModel *entrypoint, int index) {
    if (entrypoint == NULL) return NULL;
    if (index < 0) {
        int len = PSModelChainLength(entrypoint);
        if (len < 0) {
            PSErr(__func__, "broken model chain");
            return NULL;
        }
        index = len - index;
        if (index < 0) return entrypoint;
    }
    if (entrypoint->index == index) return entrypoint;
    else if (index > entrypoint->index) {
        PSModel *next = entrypoint->next;
        while (next != NULL) {
            if (next->index == index) return next;
            next = next->next;
        }
        return NULL;
    } else if (index < entrypoint->index) {
        PSModel *previous = entrypoint->previous;
        while (previous != NULL) {
            if (previous->index == index) return previous;
            previous = previous->previous;
        }
        return NULL;
    }
    return NULL;
}

PSModel *PSModelChainHead(PSModel *model) {
    if (model == NULL) return NULL;
    int is_model_chain = PSIsModelChain(model);
    if (!is_model_chain) return model;
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL || ctx->head_model == NULL) {
        if (!updateModelChain(model)) goto broken_chain;
        ctx = getModelContext(model);
         if (ctx == NULL || ctx->head_model == NULL) goto broken_chain;
        return NULL;
    }
    return ctx->head_model;
broken_chain:
    PSErr(__func__, "broken model chain");
    return NULL;
}

PSModel *PSModelChainTail(PSModel *model) {
    if (model == NULL) return NULL;
    int is_model_chain = PSIsModelChain(model);
    if (!is_model_chain) return model;
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL || ctx->last_model == NULL) {
        if (!updateModelChain(model)) goto broken_chain;
        ctx = getModelContext(model);
        if (ctx == NULL || ctx->last_model == NULL) goto broken_chain;
        return NULL;
    }
    return ctx->last_model;
broken_chain:
    PSErr(__func__, "broken model chain");
    return NULL;
}

int PSModelChainContains(PSModel *chain, PSModel *model) {
    if (!PSIsModelChain(chain)) return chain == model;
    PSModel *current = PSModelChainHead(chain);
    if (current == NULL) {
        PSErrNN(__func__, chain, NULL, "broken model chain");
        return 0;
    }
    while (current != NULL) {
        if (current == model) return 1;
        current = current->next;
    }
    return 0;
}

static void DumpModelHeader(PSModel *model, FILE *dump_file) {
    fprintf(dump_file, "psyc:version=%s\n", PSYC_VERSION);
    const char *name = model->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED MODEL";
    fprintf(
        dump_file, "model:name=%s,size=%d,status=%s\n", name, model->size,
        getModelStatusLabel(model)
    );
    if (model->training != NULL) {
        fprintf(dump_file,
            "training:started_at=%ld,current_epoch=%d,current_batch=%d,"
            "current_element=%d,batch_size=%d\n",
            model->training->started_at, model->training->current_epoch,
            model->training->current_batch,
            model->training->current_element, model->training->batch_size
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
    if (PSIsRecurrent(layer) && PSIsRecurrent(layer->model))
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

int PSModelDumpStates(PSModel *model, const char* filename) {
    if (model->size == 0) {
        PSErr(__func__, "empty model");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        PSErr(__func__, "Cannot open %s for writing!", filename);
        return 0;
    }
    int opts = 0;
    DumpModelHeader(model, f);
    int i;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
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

int PSModelDumpDeltas(PSModel *model, const char* filename) {
    if (model->size == 0) {
        PSErr(__func__, "empty model");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s for writing!\n", filename);
        return 0;
    }
    int opts = 0;
    DumpModelHeader(model, f);
    int i;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
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
                                  PSModel *model)
{
    int i;
    for (i = 0; i < PS_MAX_MEMORY_GRADIENTS; i++) {
        if (training_ctx->memory_gradients[i] != NULL) {
            PSDeleteModelGradients(training_ctx->memory_gradients[i], model);
            training_ctx->memory_gradients[i] = NULL;
        }
    }
    free(training_ctx);
}

static void deleteModelContext(PSModelContext *ctx, PSModel *model) {
    PSTrainingContext *training_ctx = ctx->training_context;
    if (training_ctx != NULL) deleteTrainingContext(training_ctx, model);
    free(ctx->sequence_start);
    free(ctx);
}

void PSModelDelete(PSModel *model) {
    if (model == NULL) return;
    PSModelContext *ctx = getModelContext(model);
    if (ctx != NULL) deleteModelContext(model->context, model);
    int size = model->size;
    int i, is_recurrent = (model->flags & FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        PSLayer *layer = NULL;
        if (model->layers != NULL) layer = model->layers[i];
        if (layer == NULL) continue;
        if (is_recurrent) layer->flags |= FLAG_RECURRENT;
        PSDeleteLayer(layer);
    }
    free(model->layers);
    if (model->training != NULL) free(model->training);
    PSModel *next = model->next;
    PSModel *prev = model->previous;
    PSModelLink *link = model->previous_model_link;
    free(model);
    if (prev != NULL && prev->next == model) prev->next = NULL;
    if (next != NULL && next->previous == model) {
        next->previous = NULL;
        PSModelDelete(next);
    }
    free(link);
}

void PSDeleteNeuron(PSNeuron *neuron) {
    if (neuron == NULL) return;
    if (neuron->extra != NULL) {
        /* TODO: extra currently unused for neurons */
        free(neuron->extra);
    }
    free(neuron);
}

PSNeuron *PSGetNeuron(PSLayer *layer, int index, PSNeuron *neuron) {
    if (layer == NULL) return NULL;
    if (index >= layer->size) {
        PSWarn("%s: neuron index %d for layer %d (size = %d) is "
               "out-of-bounds", __func__, index, layer->index, layer->size);
        return NULL;
    }
    int allocated = 0;
    if (neuron == NULL) {
        neuron = calloc(1, sizeof(*neuron));
        if (neuron == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
        allocated = 1;
    } else memset(neuron, 0, sizeof(*neuron));
    neuron->layer = layer;
    neuron->index = index;
    if (Convolutional == layer->type) {
        if (layer->output_depth == 0) {
            PSErrNN(__func__, NULL, layer, "convolutional layer has no "
                    "output_depth");
            goto err;
        }
        int feature_size = layer->size / layer->output_depth;
        int feature_idx = index / feature_size;
        neuron->bias = layer->biases + feature_idx;
        neuron->weights = layer->weights[feature_idx];
    } else {
        neuron->bias = NULL;
        neuron->weights = NULL;
        switch (layer->type) {
            case Attention:
            case Pooling:
            case Dropout:
            case Normalization:
            case PositionalEncoding:
            case OperatorLayer: goto final; break;
            default: break;
        }
        if (layer->biases != NULL) neuron->bias = layer->biases + index;
        PSMatrix weights = NULL;
        PSLayer *previous = PSGetPreviousLayer(layer);
        if (layer->weights != NULL && layer->weights[0] != NULL) {
            if (previous == NULL) {
                PSErrNN(__func__, NULL, layer,"could not find previous layer");
                goto err;
            }
            weights = layer->weights[0];
            int prevsize = previous->size;
            if (previous->flags & FLAG_ONEHOT)
                prevsize = PSGetOneHotLayerVectorSize(previous);
            neuron->weights = weights + (index * prevsize);
        }
    }
final:
    return neuron;
err:
    if (allocated) PSDeleteNeuron(neuron);
    return NULL;
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
            if (layer != NULL && layer->model != NULL)
                acceleration = layer->model->acceleration;
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
    if (layer->model == NULL) {
        PSErr(NULL, "Layer[%d]: missing model");
        goto fail;
    }
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
        if (layer->index > 0 && previous_size > 0)
            layer->biases[i] = PSInitParam(PARAM_TYPE_BIAS, ldef, 1.0, 0.0);
    }
    if (layer->type != SoftMax) {
        int is_linear = layer->type == Linear;
        if (layer->activate == NULL && !is_linear) {
            layer->activate = PSSigmoid;
            layer->derivative = PSSigmoidDerivative;
        } else if (is_linear) {
            layer->activate = NULL;
            layer->derivative = NULL;
        }
        layer->forward = PSFullForward;
    } else {
        layer->activate = NULL;
        layer->derivative = NULL;
        layer->forward = softmaxForward;
        layer->model->loss = PSCrossEntropyLoss;
    }
    layer->backprop = PSFullBackprop;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
fail:
    return 0;
}

PSLayer *PSAddLayer(PSModel *model, PSLayerType type, int size,
                    PSLayerDef *layer_def)
{
    if (model == NULL) return NULL;
    if (model->size == 0 && type != FullyConnected) {
        PSErr(__func__, "First layer type must be FullyConnected");
        return NULL;
    }
    PSLayer *layer = malloc(sizeof(PSLayer));
    if (layer == NULL) {
        PSErr(__func__, "Could not allocate layer %d!", model->size);
        return NULL;
    }
    int verbose = (PSLogLevel == PSLOGLEVEL_DEBUG);
    PSLayerDef default_def = {0};
    if (layer_def == NULL) {
        PSSetDefaultLayerDef(&default_def, type);
        layer_def = &default_def;
    }
    layer->model = model;
    layer->index = model->size++;
    layer->type = type;
    layer->size = size;
    layer->extra = NULL;
    layer->flags = layer_def->flags;
    layer->delta = NULL;
    layer->states = NULL;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->initial_states = NULL;
    layer->activate = layer_def->activation;
    layer->derivative = PSGetActivationDerivative(layer->activate);
    layer->on_delete = NULL;
    layer->on_copy = NULL;
    layer->build = NULL;
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
    layer->get_input_from_link = NULL;
    if (layer->output_depth <= 0) layer->output_depth = 1;
    layer->output_columns = layer_def->output_columns;
    layer->output_rows = layer_def->output_rows;
    if (layer->output_columns < 0) layer->output_columns = 0;
    if (layer->output_rows < 0) layer->output_rows = 0;
    PSLayer *previous = NULL;
    int previous_size = 0;
    int initialized = 0;
    if (verbose) printf("Adding layer %d\n", layer->index);
    if (model->layers == NULL) {
        model->layers = malloc(sizeof(PSLayer*));
        if (model->layers == NULL) {
            PSAbortLayer(model, layer);
            PSErr(__func__, "could not allocate model layers");
            return NULL;
        }
        if (layer->flags & FLAG_ONEHOT) model->flags |= FLAG_ONEHOT;
        if (model->flags & FLAG_ONEHOT) {
            layer->flags |= FLAG_ONEHOT;
            layer->onehot_vector_size = size;
            size = 1;
            layer->size = 1;
        }
        model->input_size = size;
    } else {
        PSLayer **layers = realloc(model->layers,
                                   sizeof(PSLayer*) * model->size);
        if (layers == NULL) {
            PSAbortLayer(model, layer);
            PSErr(__func__, "could not reallocate model layers");
            return NULL;
        }
        model->layers = layers;
        previous = model->layers[layer->index - 1];
        if (previous == NULL) {
            PSAbortLayer(model, layer);
            PSErr(__func__, "Previous layer is NULL!");
            return NULL;
        }
        previous_size = previous->size;
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT)
            previous_size = previous->onehot_vector_size;
        model->output_size = size;
    }
    if (type == FullyConnected || type == SoftMax || type == Linear) {
        initialized = initGenericLayer(layer, size, previous_size, layer_def);
    } else if (type == Convolutional) {
        initialized = PSInitConvolutionalLayer(model, layer, layer_def);
        /* TODO: Make PSCrossEntropyLoss default also for convolutional? */
    } else if (type == Pooling) {
        initialized = PSInitPoolingLayer(model, layer, layer_def);
    } else if (type == Recurrent) {
        initialized = PSInitRecurrentLayer(model, layer, size, previous_size,
                                           layer_def);
    } else if (type == LSTM) {
        initialized = PSInitLSTMLayer(model, layer, size, previous_size,
                                      layer_def);
    } else if (type == GRU) {
        initialized = PSInitGRULayer(model, layer, size, previous_size,
                                     layer_def);
    } else if (type == Dropout) {
        initialized = PSInitDropoutLayer(model, layer, layer_def);
    } else if (type == Embedding) {
        initialized = PSInitEmbeddingLayer(layer, size, layer_def);
    } else if (type == Normalization) {
        initialized = PSInitNormalizationLayer(layer, layer_def);
    } else if (type == Attention) {
        initialized = PSInitAttentiontionLayer(layer, layer_def);
    } else if (type == OperatorLayer) {
        initialized = PSInitOperatorLayer(layer, layer_def);
    } else if (type == PositionalEncoding) {
        initialized = PSInitPositionalLayer(layer, layer_def);
    } else PSErr(__func__, "Invalid layer type %d", type);
    if (!initialized) goto fail;
    if (layer->index > 0 && layer->delta == NULL) {
        layer->delta = PSMatrixZeros(2, 1, layer->size);
        if (layer->delta == NULL) goto fail;
    }
    model->layers[layer->index] = layer;
    if (PSIsRecurrent(model) || PSIsRecurrent(layer)) {
        int ok = 1;
        PSRecurrentNetworkMode rnn_mode = model->rnn_mode;
        if (rnn_mode == NonRecurrent)
            ok = PSSetRecurrentNetworkMode(model, DEFAULT_RECURRENT_MODE);
        else updateModelForRecurrentMode(model, rnn_mode);
        if (!ok) {
            PSAbortLayer(model, layer);
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
            PSAbortLayer(model, layer);
            PSErr(
                __func__, "Could load parameters for layer %d from '%s'",
                layer->index, layer_def->load_from
            );
            return NULL;
        }
    }
    if (layer->flags & FLAG_USE_SEQUENCES)
        model->flags |= FLAG_USE_SEQUENCES;
    if (verbose) PSPrintLayerInfo(layer);
    return layer;
fail:
    if (layer != NULL) {
        PSErrNN(
            __func__, NULL, layer,
            "could not initialize layer on model '%s'", model->name
        );
        PSAbortLayer(model, layer);
    }
    return NULL;
}

PSLayer *PSAddConvolutionalLayer(PSModel *model, PSLayerDef *ldef) {
    return PSAddLayer(model, Convolutional, 0, ldef);
}

PSLayer *PSAddPoolingLayer(PSModel *model, PSLayerDef *ldef) {
    return PSAddLayer(model, Pooling, 0, ldef);
}

void PSDeleteLayer(PSLayer* layer) {
    if (layer == NULL) return;
    int i;
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
    if (layer->pretrainer != NULL) PSModelDelete(layer->pretrainer);
    free(layer);
}

int PSIsLayerPlaceholder(PSLayer *layer) {
    if (layer == NULL) return 0;
    return (signed) (layer->type) == LAYER_PLACEHOLDER_TYPE;
}

PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSModel *model) {
    if (placeholder == NULL) return NULL;
    if ((signed)placeholder->type != LAYER_PLACEHOLDER_TYPE)
        return placeholder;
    if (model == NULL) {
        PSErr(__func__, "`model` argument is NULL");
        return NULL;
    }
    int *indices = (int *) placeholder->extra;
    if (indices == NULL) {
        PSErr(__func__, "invalid layer placeholder");
        return NULL;
    }
    return PSGetLayerByIndex(model, indices[0], indices[1]);
}

PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index) {
    int *indices = malloc(2 * sizeof(int));
    if (indices == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSLayer *placeholder = calloc(1, sizeof(*placeholder));
    if (placeholder == NULL) {
        free(indices);
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    indices[0] = layer_index;
    indices[1] = model_index;
    placeholder->type = LAYER_PLACEHOLDER_TYPE;
    placeholder->extra = indices;
    return placeholder;
}

int inputLayerForward(PSModel *model, PSFloat *inputs, ...) {
    PSLayer *first = model->layers[0];
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
            len *= seqlen;
            t = 0;
        }
        va_end(ap);
        assert(seqlen > 0);
        assert(t >= 0);
        if (!PSBeforeSequenceForward(first, seqlen, t)) return 0;
    }
    PSFloat *states = PSGetStates(first, t);
    if (states == NULL) {
        PSErr(NULL, "Layer[%d] missing states");
        return 0;
    }
    PSVectorCopy(states, inputs, len);
    return 1;
}

int forwardThroughTime(PSModel *model, PSFloat *inputs,
                       int timesteps, int backprop, void *opts)
{
    if (model == NULL) return 0;
    PSMathOpts mopts = {.acceleration = model->acceleration};
    PSFloat *tmpinputs = NULL;
    int output_idx = model->size - 1, i, t, ok = 1;
    PSLayer *first = PSGetFirstRecurrentLayer(model),
            *last = PSGetLastRecurrentLayer(model),
            *input_layer = model->layers[0],
            *output_layer = model->layers[output_idx];
    if (first == NULL || last == NULL) {
        /* First recurrent layer or last recurrent layer may be not set,
         * so let's discover them now. */
        if (!discoverFirstLastRecurrentLayers(model, &first, &last))
            return 0;
    }
    int input_size = input_layer->size, last_layer_idx = last->index,
        start_idx = first->index;
    PSTrainingOptions *training_opts = NULL;
    PSForwardOptions *forward_opts = NULL;
    if (backprop) training_opts = (PSTrainingOptions *) opts;
    else forward_opts = (PSForwardOptions *) opts;
    int end = -1, seqlen = timesteps, autogression_since = 0;
    int recurrent_input = (first->index == 0),
        recurrent_output = PSIsRecurrent(output_layer);
    PSSequenceSettings *sequence_settings = NULL;
    if (forward_opts != NULL)
        sequence_settings = forward_opts->sequence_settings;
    if (sequence_settings == NULL)
        sequence_settings = &(model->sequence_settings);
    int autoregression = (
        timesteps == 0 ||
        inputs == NULL ||
        !recurrent_input ||
        useAutoRegression(model, forward_opts, training_opts)
    );
    int randomized_autoregression = 0, autoregression_feed_output = 0;
    if (autoregression) {
        if (backprop) {
            int teacher_forcing = (
                training_opts != NULL &&
                training_opts->flags & TRAINING_FLAG_TEACHER_FORCING
            );
            if (teacher_forcing) autoregression = 0;
        } else if (sequence_settings != NULL) end = sequence_settings->end;
    }
    if (autoregression) {
        PSFloat *start = NULL;
        start = sequence_settings->start;
        int maxlen = sequence_settings->max_length;
        if (maxlen <= 0) maxlen = MAX_SEQUENCE_LENGTH;
        if (timesteps <= 0 || inputs == NULL) {
            if (timesteps <= 0) timesteps = 1;
            seqlen = 1;
        }
        if (seqlen >= maxlen) {
            autoregression = 0;
            goto forward_steps;
        }
        autogression_since = seqlen - 1;
        if (!backprop) timesteps = maxlen;
        if (recurrent_input) {
            tmpinputs = PSVectorZero(timesteps * input_size);
            if (tmpinputs == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            if (inputs == NULL) {
                if (start != NULL) PSVectorCopy(tmpinputs, start, first->size);
                inputs = tmpinputs;
            } else {
                PSVectorCopy(tmpinputs, inputs, first->size * seqlen);
            }
            randomized_autoregression = (
                model->flags & FLAG_RANDREGRESSION ||
                (forward_opts && forward_opts->flags & FLAG_RANDREGRESSION)
            );
            if (recurrent_output) {
                int osize = output_layer->size, isize = input_size;
                if (input_layer->flags & FLAG_ONEHOT)
                    isize = PSGetOneHotLayerVectorSize(input_layer);
                autoregression_feed_output = (osize == isize);
            }
        }
    } else {
        ok = !recurrent_input || inputs != NULL;
        if (!ok) {
            PSErr(NULL, "Recurrent network with sequence input cannot "
                  " receive NULL inputs");
            goto final;
        }
    }
forward_steps:
    if (recurrent_input) start_idx++;
    for (t = 0; t < timesteps; t++) {
        if (recurrent_input) {
            ok = inputLayerForward(model, inputs, timesteps, t);
            if (!ok) goto final;
        }
        for (i = start_idx; i <= last_layer_idx; i++) {
            PSLayer *layer = model->layers[i];
            ok = layer != NULL;
            if (!ok) {
                PSErr(__func__, "Layer %d is NULL", i);
                goto final;
            }
            ok = (layer->forward != NULL);
            if (!ok) {
                PSErr(__func__, "Layer %d forward function is NULL", i);
                goto final;
            }
            if (!PSIsRecurrent(layer)) break;
            ok = layer->forward(layer, timesteps, t);
            if (!ok) goto final;
        }
        if (autoregression && t >= autogression_since) {
            if (t == timesteps - 1) break;
            int max_idx = -1;
            PSFloat *output_states = NULL;
            if (!randomized_autoregression) {
                ok = PSFindLayerMaxState(last, NULL, &max_idx, t);
                if (!ok) goto final;
            } else {
                output_states = PSGetStates(last, t);
                ok = output_states != NULL;
                if (!ok) {
                    PSErr(NULL, "Layer[%d] has no states at step %d",
                          last->index, t);
                    goto final;
                }
                int err = 0;
                max_idx = PSRandomInt(last->size, output_states, &err, &mopts);
                ok = !err;
                if (!ok) {
                    PSErr(NULL, "Failed to get random weighted index from "
                          "layer %d", last->index);
                    goto final;
                }
            }
            if (end >= 0 && max_idx == end) break;
            if (autoregression_feed_output && tmpinputs != NULL) {
                int next_t = t + 1;
                inputs = tmpinputs + next_t;
                if (first->flags & FLAG_ONEHOT) *inputs = (PSFloat) max_idx;
                else {
                    if (output_states == NULL)
                        output_states = PSGetStates(last, t);
                    if (output_states == NULL) {
                        PSErr(NULL, "Layer[%d] has no states at step %d",
                              last->index, t);
                        ok = 0;
                        goto final;
                    }
                    PSVectorCopy(inputs, output_states, last->size);
                }
            }
        } else if (inputs != NULL) inputs += input_size;
    }
final:
    free(tmpinputs);
    return ok;
}

int useAutoRegression(PSModel *model,
                      PSForwardOptions *forward_opts,
                      PSTrainingOptions *training_opts)
{
    if (!PSUseSequences(model)) return 0;
    if (!PSUseSequences(model->layers[model->size - 1])) return 0;
    if (model->flags & FLAG_AUTOREGRESSION) return 1;
    else if (forward_opts != NULL)
        return forward_opts->flags & FLAG_AUTOREGRESSION;
    else if (training_opts != NULL && model->next == NULL)
        return training_opts->flags & FLAG_AUTOREGRESSION;
    return 0;
}

int beforeModelForward(PSModel *model, PSFloat **inputs_p,
                       int backprop, void *opts)
{
    PSFloat *inputs = *inputs_p, *model_inputs = *inputs_p;
    PSTrainingOptions *train_opts = NULL;
    PSForwardOptions *forward_opts = NULL;
    if (backprop) train_opts = opts;
    else forward_opts = opts;
    int is_input_model = model->previous == NULL,
        autoregression = useAutoRegression(model, forward_opts, train_opts);
    if (inputs == NULL) {
        if (!autoregression || (backprop && is_input_model)) {
            PSErr(__func__, "No inputs");
            return 0;
        }
    }
    if (!is_input_model) {
        PSModel *prev = model->previous;
        model_inputs = NULL;
        PSModelLink *link = model->previous_model_link;
        if (link == NULL) {
            link = findLinkToPreviousModel(model, prev);
            model->previous_model_link = link;
        }
        if (link == NULL || !link->layer || !link->previous_layer) {
            PSErr(__func__, "broken model chain: missing or invalid link in "
                  "model %d", model->index);
            if (link != NULL) free(link);
            model->previous_model_link = NULL;
            return 0;
        }
        int do_feed_input = (
            link->layer->index == 0 &&
            link->layer->size == link->previous_layer->size
        );
        PSFloat *outputs = NULL;
        int input_whole_seq = PSHandleSequenceAtOnce(link->layer);
        int seqlen = 1;
        if (!input_whole_seq) {
            outputs = PSGetOutputs(link->previous_layer);
        } else {
            seqlen = PSStateSequenceLength(link->previous_layer);
            outputs = PSGetStates(link->previous_layer);
        }
        if (do_feed_input) *inputs_p = outputs;
        else {
            /* TODO (S2S): return 0 if model doesn't support autogression ? */
            if (outputs == NULL) {
                PSErr(__func__, "no outputs from model %d, layer %d",
                      link->previous_layer->model->index,
                      link->previous_layer->index);
                return 0;
            }
            if (link->layer->get_input_from_link != NULL) {
                if (!link->layer->get_input_from_link(link->previous_layer))
                    return 0;
            } else {
                if (PSUseSequences(link->layer)) {
                    if (!PSResetLayerStateSequence(link->layer, seqlen, 0))
                        return 0;
                }
                if (link->layer->states == NULL) {
                    PSErr(__func__, "model %d layer %d has no states",
                          model->index, link->layer->index);
                    return 0;
                }
                PSVectorCopy(link->layer->states, outputs,
                             link->layer->size * seqlen);
            }
        }
        *inputs_p = model_inputs;
    }
    return 1;
}

int modelForward(PSModel *model, PSFloat *inputs, PSFloat *global_inputs,
                 int backprop, void *opts)
{
    if (model == NULL) return 0;
    if (model->size == 0) {
        PSErr(__func__, "empty model");
        return 0;
    }
    if (!PSModelIsBuilt(model)) {
        PSErr(__func__, "model is not built", model->index);
        return 0;
    }
    PSForwardOptions *forward_opts = NULL;
    PSTrainingOptions *train_opts = NULL;
    if (backprop) train_opts = opts;
    else forward_opts = opts;
    int is_recurrent = PSIsRecurrent(model), recurrent_input = 0, i,
        ok = 1, seqlen = 0, first_idx = 0, output_idx = model->size - 1,
        input_is_seq = 0, is_input_model = model->previous == NULL,
        use_seq = PSUseSequences(model);
    PSLayer *input_layer = model->layers[0],
            *output_layer = model->layers[output_idx],
            *first_recurrent = NULL,
            *last_recurrent = NULL;
    int autoregression = useAutoRegression(model, forward_opts, train_opts);
    if (inputs == NULL && !autoregression) {
        PSErr(__func__, "Null inputs");
        return 0;
    }
    PSFloat *tmpinputs = NULL;
    if (use_seq) {
        int retain_previous = 0;
        if (is_recurrent) {
            retain_previous = 1;
            if (backprop && is_input_model) {
                retain_previous = (
                    train_opts != NULL &&
                    (train_opts->flags & TRAINING_EPOCH_AS_SEQUENCE)
                );
                if (retain_previous && model->training != NULL)
                    retain_previous = (model->training->current_batch > 0);
            }
            first_recurrent = PSGetFirstRecurrentLayer(model);
            last_recurrent = PSGetLastRecurrentLayer(model);
            recurrent_input = PSIsRecurrent(input_layer);
        } else if (input_layer->flags & FLAG_USE_SEQUENCES) {
            input_is_seq = 1;
        }
        if (is_input_model && inputs != NULL) {
            if (recurrent_input || input_is_seq) {
                /* Read seqlen from first element in `inputs`. */
                seqlen = (int) inputs[0];
            } else if (backprop && PSIsRecurrent(output_layer)) {
                /* If forward is called from backprop and model has
                 * recurrent output but non-recurrent input (OneToMany),
                 * we should read the seqlen that are the first element
                 * of labels (y) that follows inputs. */
                PSFloat *y = inputs + model->input_size;
                seqlen = y[0];
            } else {
                PSErr(__func__, "cannot determine input sequence length");
                ok = 0;
                goto final;
            }
        } else if (!is_input_model && backprop && global_inputs != NULL) {
            int seq2seq = (
                train_opts && (train_opts->flags & TRAINING_FLAG_SEQ2SEQ)
            );
            /* TODO (S2S): only check seq2seq or also autoregression ?? */
            if (autoregression || seq2seq) {
                int training_flags = (train_opts ? train_opts->flags : 0);
                PSFloat *y = NULL;
                int datalen = parseSequenceData(
                    model, global_inputs, 0, training_flags, 1,
                    NULL, NULL, &seqlen, &y
                );
                ok = (datalen > 0);
                if (!ok) {
                    PSErr(__func__, "could not retrieve training prediction "
                          "sequence count");
                    goto final;
                }
                int teacher_forcing = (
                    inputs == NULL && model->next == NULL &&
                    train_opts != NULL &&
                    train_opts->flags & TRAINING_FLAG_TEACHER_FORCING
                );
                if (teacher_forcing && seqlen > 0) {
                    int input_size = input_layer->size,
                        ysize = input_size * seqlen,
                        /* Make room for <start> */
                        full_ysize = input_size * ++seqlen,
                        /* Add one float for sequence length */
                        datasize = (full_ysize + 1) * sizeof(PSFloat);
                    PSFloat *start = (&(model->sequence_settings))->start;
                    tmpinputs = malloc(datasize);
                    ok = (tmpinputs != NULL);
                    if (!ok) {
                        PSPrintMemoryErrorMsg();
                        goto final;
                    }
                    PSFloat *yseq = tmpinputs;
                    *(yseq++) = (PSFloat) seqlen;
                    if (start != NULL)
                        PSVectorCopy(yseq, start, input_size);
                    else PSVectorClear(yseq, input_size);
                    PSVectorCopy(yseq + input_size, y, ysize);
                    inputs = tmpinputs;
                }
            }
        }
        if (is_recurrent) { /* TODO (S2S): only for recurrent input? */
            int steps = seqlen;
            if (steps <= 0) steps = 1;
            ok = PSResetModelStateSequences(model, steps, retain_previous);
            if (!ok) {
                PSErr(NULL, "Failed to reset model recurrent states");
                goto final;
            }
        }
    }
    if (model->beforeForward != NULL) {
        ok = model->beforeForward(model, inputs, seqlen, backprop, opts);
        if (!ok) goto final;
    }
    if (recurrent_input) {
        if (seqlen <= 0 && !autoregression) {
            PSErr(
                __func__, "Recurrent sequence length must be > 0 (found %d)",
                seqlen
            );
            ok = 0;
            goto final;
        }
        PSFloat *rnn_inputs = (inputs != NULL ? inputs + 1 : NULL);
        ok = forwardThroughTime(model, rnn_inputs, seqlen, backprop, opts);
        if (!ok) goto final;
        if (last_recurrent == NULL && PSIsRecurrent(output_layer)){
            setModelContext(model, last_recurrent_layer, output_layer);
            goto final;
        } else if (last_recurrent != NULL) {
            int last_recurrent_idx = last_recurrent->index;
            if (last_recurrent_idx >= output_idx) goto final;
            else first_idx = last_recurrent_idx;
        }
    } else if (input_is_seq) {
        ok = seqlen > 0;
        if (!ok) {
            PSErr(
                __func__, "Sequence length must be > 0 (found %d)",
                seqlen
            );
            goto final;
        }
        PSFloat *seq_inputs = (inputs != NULL ? inputs + 1 : NULL);
        ok = inputLayerForward(model, seq_inputs, seqlen);
    } else {
        ok = inputLayerForward(model, inputs);
        if (!ok) goto final;
    }
    for (i = (first_idx + 1); i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        ok = layer != NULL;
        if (!ok) {
            PSErr(__func__, "Layer %d is NULL!", i);
            goto final;
        }
        if (is_recurrent && layer == first_recurrent) {
            ok = forwardThroughTime(model, NULL, seqlen, backprop, opts);
            goto final;
        }
        ok = layer->forward != NULL;
        if (!ok) {
            PSErr(__func__, "Layer %d forward function is NULL", i);
            goto final;
        }
        ok = layer->forward(layer, seqlen);
        if (!ok) goto final;
    }
final:
    if (tmpinputs != NULL) free(tmpinputs);
    if (!ok) PSModelSetStatus(model, STATUS_ERROR, NULL);
    return ok;
}

int forward(PSModel *model, PSFloat *inputs, int backprop, void *opts)
{
    int ok = 1;
    if (model == NULL) return 0;
    PSFloat *global_inputs = inputs;
    while (model != NULL) {
        ok = beforeModelForward(model, &inputs, backprop, opts);
        if (!ok) return 0;
        ok = modelForward(model, inputs, global_inputs, backprop, opts);
        if (!ok) return 0;
        model = model->next;
    }
    return ok;
}

/* Forward data to neural network. Data is an array of PSFloat (`inputs`)
 * that must have the same length of units (neurons) in the input (first)
 * layer (in non-recurrent networks).
 * In recurrent networks, `inputs` length should be input layer size + 1, and
 * the first value indicates the number of iterations (timesteps). */
int PSForward(PSModel *model, PSFloat *inputs) {
    return forward(model, inputs, 0, NULL);
}

int PSAutoregression(PSModel *model, PSFloat *inputs,
                     int randomized, PSSequenceSettings *sequence_settings)
{
    if (!PSIsRecurrent(model) && !PSHandleSequenceAtOnce(model)) {
        PSErr(__func__, "autoregression is only available in model "
              "that use sequences");
        return 0;
    }
    PSForwardOptions opts = {.flags = FLAG_AUTOREGRESSION};
    if (randomized) opts.flags |= FLAG_RANDREGRESSION;
    if (sequence_settings) opts.sequence_settings = sequence_settings;
    return forward(model, inputs, 0, &opts);
}

/* Forward `inputs` to `model` and get the index of the maximum state
 * from the output layer. */
int PSClassify(PSModel *model, PSFloat *inputs) {
    int ok = PSForward(model, inputs);
    if (!ok) {
        PSErr(__func__, "forward failed");
        return -1;
    };
    PSLayer *out = PSGetOutputLayer(model);
    int max_idx = 0, seqlen = PSStateSequenceLength(out),
        t = (int) seqlen - 1;
    if (t < 0) t = 0;
    if (!PSFindLayerMaxState(out, NULL, &max_idx, t)) {
        PSErr(__func__, "Failed to find neuron with max value");
        return -1;
    }
    return max_idx;
}

PSGradient *createLayerGradient(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (layer->type == Pooling || layer->type == Dropout ||
        layer->type == OperatorLayer) return NULL;
    PSGradient *gradient = malloc(sizeof(*gradient));
    if (gradient == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    gradient->bias_count = 0;
    gradient->weight_count = 0;
    gradient->biases = NULL;
    gradient->weights = NULL;
    gradient->tmp = NULL;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    uint64_t bias_count = 0, weights_count = 0, max_count = 0;
    if (use_bias)
        bias_count = PSGetLayerParametersCount(layer, PARAM_TYPE_BIAS);
    weights_count = PSGetLayerParametersCount(layer, PARAM_TYPE_WEIGHT);
    if (bias_count > 0) {
        gradient->biases = calloc(bias_count, sizeof(PSFloat));
        if (gradient->biases == NULL) {
            PSPrintMemoryErrorMsg();
            PSDeleteGradient(gradient);
            return NULL;
        }
        gradient->bias_count = bias_count;
        max_count = bias_count;
    }
    if (weights_count > 0) {
        gradient->weights = calloc(weights_count, sizeof(PSFloat));
        if (gradient->weights == NULL) {
            PSPrintMemoryErrorMsg();
            PSDeleteGradient(gradient);
            return NULL;
        }
        gradient->weight_count = weights_count;
        if (weights_count > max_count) max_count = weights_count;
    }
    if (max_count > 0) {
        gradient->tmp = malloc(max_count * sizeof(PSFloat));
    }
    return gradient;
}

PSGradient **createModelGradients(PSModel *model) {
    if (model == NULL) return NULL;
    if (model->size < 2) return NULL;
    PSGradient **gradients = malloc(sizeof(PSGradient*) * model->size - 1);
    if (gradients == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int i;
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        int idx = i - 1;
        gradients[idx] = createLayerGradient(layer);
        if (gradients[idx] == NULL && layer->type != Pooling &&
            layer->type != Dropout && layer->type != OperatorLayer)
        {
            PSErrNN(NULL, NULL, layer, "could not create gradients");
            PSDeleteModelGradients(gradients, model);
            return NULL;
        }
    }
    return gradients;
}

PSGradient ***createGradients(PSModel *model) {
    if (model ==  NULL) return NULL;
    assert(model->previous == NULL);
    int num_models = PSModelChainLength(model), len = 0;
    if (num_models <= 0) return NULL;
    PSGradient ***gradients = calloc(num_models, sizeof(PSGradient **));
    if (gradients == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    for (int i = 0; i < num_models; i++) {
        if (model == NULL) {
            PSErr(NULL, "broken model chain");
            goto err;
        }
        gradients[i] = createModelGradients(model);
        if (gradients[i] == NULL) goto err;
        len++;
        model = model->next;
    }
    return gradients;
err:
    if (gradients != NULL) {
        for (int i = 0; i < len; i++)
            PSDeleteModelGradients(gradients[i], model);
        free(gradients);
    }
    return NULL;
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
    if (clone != NULL) PSDeleteGradient(clone);
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

PSGradient **cloneModelGradients(PSGradient **gradients, PSModel *model) {
    if (gradients == NULL) return NULL;
    PSGradient **clone = createModelGradients(model);
    if (clone == NULL) return NULL;
    for (int i = 1; i < model->size; i++) {
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
    if (clone != NULL) PSDeleteModelGradients(clone, model);
    return NULL;
}

void PSDeleteGradient(PSGradient *gradient) {
    if (gradient == NULL) return;
    free(gradient->biases);
    free(gradient->weights);
    free(gradient->tmp);
    free(gradient);
}

void PSDeleteModelGradients(PSGradient **gradients, PSModel *model)
{
    if (gradients == NULL) return;
    int i;
    for (i = 1; i < model->size; i++) {
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        PSDeleteGradient(lgradients);
    }
    free(gradients);
}

void PSDeleteGradientsChain(PSGradient ***gradients, PSModel *model){
    if (gradients == NULL) return;
    if (model == NULL) {
        PSWarn(__func__, "missing mandatory argument `model`");
        return;
    }
    PSModel *current = PSModelChainHead(model);
    int idx = 0;
    while (current != NULL) {
        PSDeleteModelGradients(gradients[idx++], current);
        current = current->next;
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

static int resetDeltas(PSModel *model) {
    if (model->layers == NULL) return 0;
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) continue;
        if (!resetLayerDeltas(layer, 1)) return 0;
    }
    return 1;
}

static int propagateDeltaFromNextModel(PSModel *model) {
    if (model->next == NULL) return 0;
    PSModelLink *link = model->next->previous_model_link;
    int ok = link != NULL;
    if (!ok) {
        PSErr(NULL, "model %d has not previous_model_link",
              model->next->index);
        return 0;
    }
    ok = (link->layer != NULL && link->previous_layer != NULL);
    if (!ok) {
        PSErr(NULL, "model %d has invalid previous_model_link",
              model->next->index);
        return 0;
    }
    if (link->previous_layer->model != model) {
        PSErr(NULL, "model %d linked to another model (%d)",
              model->next->index, link->previous_layer->model->index);
        return 0;
    }
    if (link->previous_layer->delta != NULL) {
        PSMatrixDelete(link->previous_layer->delta);
        link->previous_layer->delta = NULL;
    }
    if (LSTM == link->previous_layer->type && link->layer->type != LSTM) {
        link->previous_layer->delta =
            PSMatrixZeros(2, 1, link->previous_layer->size * 2);
        if (link->previous_layer->delta == NULL) return 0;
        PSVectorCopy(link->previous_layer->delta, link->layer->delta,
                     link->previous_layer->size);
    } else link->previous_layer->delta = PSMatrixDup(link->layer->delta);
    return link->previous_layer->delta != NULL;
}

void beforeBatchTraining(PSModel *model) {
    if (model == NULL || model->layers == NULL) return;
    int i, j;
    while (model) {
        for (i = 0; i < model->size; i++) {
            PSLayer *layer = model->layers[i];
            if (layer == NULL) continue;
            if (layer->weights != NULL) {
                for (j = 0; j < layer->weight_types_count; j++) {
                    PSMatrix weights = layer->weights[j];
                    if (weights == NULL) continue;
                    PSMatrixResetTransposed(weights);
                }
            }
            if (layer->before_batch_training != NULL)
                layer->before_batch_training(layer);
        }
        model = model->next;
    }
}

void PSResetTransposedWeights(PSModel *model) {
    if (model == NULL || model->layers == NULL) return;
    int i, j;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) continue;
        if (layer->weights == NULL) continue;
        for (j = 0; j < layer->weight_types_count; j++) {
            PSMatrix weights = layer->weights[j];
            if (weights == NULL) continue;
            PSMatrixResetTransposed(weights);
        }
    }
}

static PSTrainingContext *getTrainingContext(PSModel *model) {
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) return NULL;
    return ctx->training_context;
}

PSTrainingOptions *PSGetModelTrainingOptions(PSModel *model) {
    PSTrainingContext *tctx = getTrainingContext(model);
    if (tctx == NULL) return NULL;
    return &(tctx->options);
}

int PSGetTrainingMemoryGradients(PSModel *model,
                                 PSGradient ***grads_p)
{
    PSTrainingContext *tctx = getTrainingContext(model);
    if (tctx == NULL) return 0;
    int count = 0, i;
    for (i = 0; i < PS_MAX_MEMORY_GRADIENTS; i++) {
        PSGradient **mgrads = tctx->memory_gradients[i];
        if (mgrads != NULL) {
            if (grads_p != NULL) grads_p[count++] = mgrads;
            else count++;
        }
    }
    return count;
}

int getRequiredMemoryGradientsCount(PSTrainingOptions *opts) {
    int required_memory_gradients = 0;
    PSFloat momentum = 0.0;
    PSOptimization optimization = PSDefaultOptimization;
    if (opts != NULL) {
        momentum = opts->momentum;
        optimization = opts->optimization;
    }
    if (momentum > 0.0 || optimization != PSDefaultOptimization) {
        required_memory_gradients++;
        if (optimization == PSAdaDeltaOptimization ||
            optimization == PSAdamOptimization) required_memory_gradients++;
    }
    return required_memory_gradients;
}

int initMemoryGradients(PSModel *nn, PSTrainingContext *ctx,
                        int mem_gradients_count)
{
    if (nn == NULL) return 0;
    if (mem_gradients_count > PS_MAX_MEMORY_GRADIENTS)
        mem_gradients_count = PS_MAX_MEMORY_GRADIENTS;
    int i;
    for (i = 0; i < mem_gradients_count; i++) {
        ctx->memory_gradients[i] = createModelGradients(nn);
        if (ctx->memory_gradients[i] == NULL) {
            PSErr(NULL, "could not create memory gradients[%d]", i);
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
    return 1;
}

int initTrainingContext(PSModel *model,
                        PSTrainingOptions *training_options,
                        int mem_gradients_count)
{
    if (model == NULL) return 0;
    int success = 1;
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) {
        ctx = model->context = calloc(1, sizeof(PSModelContext));
        success = (ctx != NULL);
        if (!success) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
    }
    if (ctx->training_context == NULL) {
        ctx->training_context = calloc(1, sizeof(PSTrainingContext));
        success = (ctx->training_context != NULL);
        if (!success) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        memset(
            ctx->training_context->memory_gradients, 0,
            sizeof(ctx->training_context->memory_gradients)
        );
    }
    if (training_options != NULL) {
        memcpy(
            &(ctx->training_context->options), training_options,
            sizeof(PSTrainingOptions)
        );
    } else PSSetDefaultTrainingOptions(&(ctx->training_context->options));
    if (mem_gradients_count > 0) {
        success = initMemoryGradients(
            model, ctx->training_context, mem_gradients_count
        );
        if (!success) goto final;
    }
final:
    return success;
}

PSFloat *PSSetSequenceStart(PSModel *model, PSFloat *start, int len){
    if (len <= 0 || start == NULL) return 0;
    PSFloat *dup = malloc(len * sizeof(PSFloat));
    if (dup == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSVectorCopy(dup, start, len);
    setModelContext(model, sequence_start, dup);
    model->sequence_settings.start = dup;
    return dup;
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
    if (previous == NULL) previous = PSGetPreviousLayer(layer);
    int is_recurrent = PSIsRecurrent(layer),
        prev_is_recurrent = (previous != NULL && PSIsRecurrent(previous)),
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
        if (previous != NULL) *inputs = previous->states;
        else *inputs = NULL;
        if (PSMatrixDim(layer->delta, 0) != slen)
            if (!resetLayerDeltas(layer, 1)) return 0;
    }
    if (*outputs == NULL && !handle_seq) *outputs = PSGetStates(layer, t);
    if (*inputs == NULL && previous != NULL && !handle_seq)
        *inputs = PSGetStates(previous, prev_t);
    if (*outputs == NULL) {
        PSErr(NULL, "Layer[%d]: NULL outputs", layer->index);
        return 0;
    }
    if (*inputs == NULL && previous != NULL) {
        PSErrNN(NULL, NULL, layer, "NULL inputs");
        return 0;
    }
    if (seqlen != NULL) *seqlen = slen;
    if (step != NULL) *step = t;
    return 1;
}

void PSUpdateGradientData(PSFloat *gradient_weights, PSFloat *gradient_biases,
                          PSFloat *inputs, PSFloat *delta,
                          int size, int input_size,
                          int seqlen, int acceleration)
{
    if (seqlen < 1) seqlen = 1;
    PSMathOpts opts = {.acceleration = acceleration};
    PSFloat *delta_p = delta, *input_p = inputs;
    for (int i = 0; i < seqlen; i++) {
        opts.store_mode = PS_STORE_MODE_ADD;
        PSOuterProduct(
            delta_p, input_p, gradient_weights,
            size, input_size, &opts
        );
        if (gradient_biases != NULL) {
            opts.store_mode = PS_STORE_MODE_SET;
            PSAddVectors(
                delta_p, gradient_biases, gradient_biases, size, &opts
            );
        };
        delta_p += size;
        input_p += input_size;
    }
}

int PSUpdateDelta(PSMatrix destdelta, PSMatrix srcdelta, PSMatrix weights,
                  int seqlen, int acceleration)
{
    int ok;
    PSMathOpts opts = {.acceleration = acceleration};
    opts.store_mode = PS_STORE_MODE_ADD;
    if (seqlen < 1) seqlen = 1;
    if (seqlen == 1) {
        opts.transpose = 1; /* Transpose weights */
        ok = PSDotMV(weights, srcdelta, destdelta, &opts);
    } else {
        ok = PSDot(srcdelta, weights, destdelta, &opts);
    }
    return ok;
}

void PSUpdateGradient(PSGradient *gradient, PSMatrix inputs, PSLayer *layer,
                      PSLayer *previous, int use_bias, int seqlen)
{
    if (seqlen < 1 || !PSHandleSequenceAtOnce(layer)) seqlen = 1;
    PSFloat *gweights = gradient->weights, *gbias = NULL;
    if (use_bias) gbias = gradient->biases;
    PSUpdateGradientData(gweights, gbias, inputs, layer->delta,
                         layer->size, previous->size, seqlen,
                         layer->model->acceleration);
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
    if (seqlen < 1 || !PSHandleSequenceAtOnce(layer)) seqlen = 1;
    PSMatrix weights = layer->weights[weights_index];
    int ok = PSUpdateDelta(previous->delta, layer->delta, weights,
                  seqlen, layer->model->acceleration);

    if (!ok)
        PSErr(NULL, "Layer[%d]: failed backprop (PSDot) (seqlen = %d)",
              layer->index);
    return ok;
}

int PSSoftmaxBackward(PSFloat *softmax_out, PSFloat *delta, PSFloat *dest,
                      uint64_t len, int acceleration)
{
    PSMatrix diagonal = PSDiagonalFlattenVector(softmax_out, len);
    if (diagonal == NULL) return 0;
    int success = 1;
    PSMatrix sout = PSMatrixFromArray(softmax_out, 2, len, 1), tmpdest = NULL;
    success = (sout != NULL);
    if (!success) goto final;
    PSMathOpts opts = {.acceleration = acceleration};
    opts.transpose = 2;
    success = PSMatrixProduct(sout, sout, &tmpdest, &opts);
    if (!success) goto final;
    PSSubtractVectors(diagonal, tmpdest, diagonal, len * len, &opts);
    opts.argtype[1] = 'V';
    success = PSMatrixProductVM(delta, diagonal, len, &dest, &opts);
final:
    PSMatrixDelete(diagonal);
    PSMatrixDelete(tmpdest);
    PSMatrixDelete(sout);
    return success;
}

int computeSoftmaxOutputDelta(PSLayer *layer, PSFloat *y, ...) {
    PSModel *model = layer->model;
    assert(layer->type == SoftMax);
    int t = 0, ok = 1, handle_seq = PSHandleSequenceAtOnce(layer),
        seqlen = 1;
    int apply_derivative = outputDerivativeNeeded(model);
    int onehot = (layer->flags & FLAG_ONEHOT);
    PSFloat *outputs = NULL, *inputs = NULL;
    va_list args;
    va_start(args, y);
    ok = PSBeforeLayerBackprop(layer, NULL, &t, &seqlen, &outputs,
                               &inputs, args);
    va_end(args);
    if (!ok) return 0;
    PSMatrix delta = layer->delta;
    PSMathOpts mopts = {.acceleration = model->acceleration};
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
            softmax_sum = PSVectorReduceSum(delta_p, layer->size, &mopts);
            mopts.store_mode = PS_STORE_MODE_SUB;
            PSMultiplyVectorScalar(
                out_p, softmax_sum, delta_p, layer->size, &mopts
            );
            mopts.store_mode = PS_STORE_MODE_SET;
            delta_p += layer->size;
            out_p += layer->size;
        }
    }
    return 1;
}

int computeOutputDelta(PSLayer *layer, PSFloat *y, ...) {
    PSModel *model = layer->model;
    int handle_seq = PSHandleSequenceAtOnce(layer);
    int is_softmax = layer->type == SoftMax;
    int t = 0, seqlen = 1, ok = 1;
    PSFloat *outputs = NULL, *inputs = NULL;
    /* Checks */
    va_list args;
    va_start(args, y);
    ok = PSBeforeLayerBackprop(layer, NULL, &t, &seqlen,&outputs,&inputs,args);
    va_end(args);
    if (!ok) return 0;
    if (is_softmax) return computeSoftmaxOutputDelta(layer, y, t);
    int onehot = (layer->flags & FLAG_ONEHOT);
    PSMatrix delta = layer->delta;
    if (delta == NULL) {
        PSErr(NULL, "Output layer[%d] has no delta");
        return 0;
    }
    uint64_t delta_len = PSMatrixLength(delta);
    PSMathOpts mopts = {.acceleration = model->acceleration};
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
    return 1;
}

int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *gradient, ...)
{
    PSMatrix delta = layer->delta;
    if (delta == NULL) return 0;
    PSModel *model = layer->model;
    PSMathOpts mopts = {.acceleration = model->acceleration};
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

int backpropThroughTime(PSModel *model, PSFloat *y,
                        PSGradient **gradients, PSTrainingOptions *opts,
                        int timesteps)
{
    if (model == NULL) return 0;
    if (gradients == NULL) return 0;
    int nlayers = model->size;
    PSLayer *output_layer = model->layers[nlayers - 1],
            *last_recurrent = PSGetLastRecurrentLayer(model),
            *model_last_recurrent = last_recurrent;
    if (last_recurrent == NULL) last_recurrent = output_layer;
    while (!PSIsRecurrent(last_recurrent)) {
        if (last_recurrent->index <= 1) {
            PSErr(NULL, "Failed to find last recurrent layer");
            return 0;
        }
        last_recurrent = model->layers[last_recurrent->index - 1];
    }
    if (model_last_recurrent == NULL && PSIsRecurrent(last_recurrent))
        setModelContext(model, last_recurrent_layer, last_recurrent);
    int last_t = timesteps - 1;
    int recurrent_output = (last_recurrent == output_layer);
    int is_output_model = (model->next == NULL);
    int bptt_truncate = (opts != NULL ? opts->bptt_truncate : BPTT_TRUNCATE);
    if (bptt_truncate < 0) bptt_truncate = 0;
    int onehot, osize, ysize, i, t, ok = 1;
    PSFloat *eos_labels = NULL;
    if (recurrent_output && is_output_model) {
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
            PSErr(NULL, "Cannot backpropagate on recurrent model with "
                  "zero hidden states");
            return 0;
        }
        ok = (hidden_states_count == timesteps);
        if (!ok) {
            PSErr(NULL, "Hidden states count %d differs from timesteps %d",
                  hidden_states_count, timesteps);
            return 0;
        }
    }
    int do_truncate = bptt_truncate > 0;
    PSFloat *delta;
    for (t = last_t; t >= 0; t--) {
        int lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        if (recurrent_output && is_output_model) {
            /* Backpropagate starting from output layer. */
            int timestep_offset = t * ysize;
            PSFloat *timestep_y = y + timestep_offset;
            ok = computeOutputDelta(output_layer, timestep_y, t);
            if (!ok) goto final;
        }
        /*  Cycle through layers */
        for (i = output_layer->index; i > 0; i--) {
            PSLayer *layer = model->layers[i];
            PSLayer *previous_layer = model->layers[i - 1];
            if (layer->pretrained) break;
            if (!PSIsRecurrent(layer)) break;
            PSGradient *lgradients = gradients[i - 1];
            PSLayerType ltype = layer->type;
            int is_lstm = (LSTM == ltype);
            int is_gru = (GRU == ltype);

            /*  Apply derivative on layer deltas */
            if (layer->derivative != NULL && !is_lstm && !is_gru) {
                delta = layer->delta;
                PSMathOpts mopts = {
                    .acceleration = layer->model->acceleration
                };
                int ok = PSApplyDerivative(
                    layer->derivative, delta,PSGetStates(layer, t),
                    layer->size, &mopts
                );
                if (!ok) return 0;
            }
            /* If BPTT is truncated, delta value from previous iteration
             * is not cumulated since it has been already backpropagated
             * to previous timesteps during previous iteration.
             * So, reset previous layer deltas. */
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

PSGradient **modelBackprop(PSModel *model,
                           PSFloat *y,
                           PSTrainingOptions *opts,
                           PSGradient **gradients)
{
    if (model == NULL) return NULL;
    int i, ok = 1;
    PSFloat *tmpy = NULL;
    PSGradient **new_gradients = NULL;
    if (gradients == NULL) {
        gradients = createModelGradients(model);
        new_gradients = gradients;
    }
    if (gradients == NULL) return NULL;
    int nlayers = model->size, is_recurrent = PSIsRecurrent(model),
        seqlen = 0;
    PSLayer *output_layer = model->layers[nlayers - 1];
    PSGradient *lgradients = gradients[nlayers - 2]; /* No gradient for
                                                        inputs */
    int teacher_forcing = (
        y != NULL && model->next == NULL &&
        PSUseSequences(model) && PSUseSequences(model->layers[0]) &&
        opts != NULL && opts->flags & TRAINING_FLAG_TEACHER_FORCING
    );
    if (teacher_forcing && PSUseSequences(output_layer)) {
        seqlen = PSStateSequenceLength(model->layers[0]);
        if (seqlen > 0) {
            int onehot_labels = output_layer->flags & FLAG_ONEHOT;
            int osize = (onehot_labels ? 1 : output_layer->size);
            tmpy = malloc(osize * seqlen * sizeof(PSFloat));
            ok = (tmpy != NULL);
            if (!ok) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            PSVectorCopy(tmpy, y, osize * (seqlen - 1));
            int end = model->sequence_settings.end;
            if (end < 0) end = 0;
            if (onehot_labels) tmpy[seqlen - 1] = (PSFloat) end;
            else {
                if (end >= osize) {
                    ok = 0;
                    PSErr(NULL, "model sequence_settings.end (%d) exceeds "
                          "output layer size (%d)", end, osize);
                    goto final;
                }
                PSFloat *last_item = tmpy + ((seqlen - 1) * osize);
                PSVectorClear(last_item, osize);
                last_item[end] = 1;
            }
            y = tmpy;
        }
    }
    PSLayer *backprop_from = output_layer, *previous_layer = NULL;
    ok = resetDeltas(model);
    if (!ok) {
        PSErr(NULL, "failed to reset model deltas");
        goto final;
    }
    if (model->next != NULL) {
        ok = propagateDeltaFromNextModel(model);
        if (!ok) goto final;
    }
    if (model->beforeBackprop != NULL) {
        ok = model->beforeBackprop(model, y, opts, gradients);
        if (!ok) goto final;
    }
    if (is_recurrent && PSIsRecurrent(output_layer)) {
        PSLayer *input_layer = model->layers[0];
        PSLayer *first_recurrent = PSGetFirstRecurrentLayer(model);
        int timesteps = seqlen ? seqlen : PSStateSequenceLength(output_layer);
        if (timesteps == 0) {
            PSErr(
                NULL, "Recurrent timesteps must be > 0 (found %d)",
                timesteps
            );
            ok = 0;
            goto final;
        }
        ok = backpropThroughTime(model, y, gradients, opts, timesteps);
        if (!ok) goto final;
        /* If first recurrent layer is the input layer, backpropThroughTime
         * has already backpropagated the error to the whole model,
         * so finish here. */
        if (first_recurrent == NULL && PSIsRecurrent(input_layer)) {
            /* First recurrent layer was not set but first layer is recurrent.
             * Update context and finish backpropagation. */
            setModelContext(model, first_recurrent_layer, input_layer);
            goto final;
        } else if (first_recurrent != NULL) {
            int first_recurrent_idx = first_recurrent->index;
            if (first_recurrent_idx == 0) goto final;
            else backprop_from = model->layers[first_recurrent_idx - 1];
        } else {
            PSErr(
                NULL, "First recurrent layer was not found, cannot "
                "continue backpropagation"
            );
            ok = 0;
            goto final;
        }
    } else {
        ok = computeOutputDelta(output_layer, y);
        if (!ok) goto final;
    }
    for (i = backprop_from->index; i > 0; i--) {
        PSLayer *layer = model->layers[i];
        previous_layer = model->layers[i - 1];
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
            ok = backpropThroughTime(model, NULL, gradients, opts, timesteps);
            if (!ok) goto final;
            break;
        }
        PSLayerType ltype = layer->type;
        ok = (
            FullyConnected == ltype || Embedding == ltype ||
            Dropout == ltype || Normalization == ltype ||
            Pooling == ltype || Convolutional == ltype || SoftMax == ltype ||
            Attention == ltype || OperatorLayer == ltype ||
            Linear == ltype
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
    if (tmpy != NULL) free(tmpy);
    if (!ok) {
        if (new_gradients != NULL)
            PSDeleteModelGradients(new_gradients, model);
        return NULL;
    }
    return gradients;
}

PSGradient ***backprop(PSModel *model, PSFloat *x, PSFloat *y,
                       PSTrainingOptions *opts, PSGradient ***gradients)
{
    if (model == NULL) return NULL;
    assert(model->previous == NULL);
    if (x == NULL) {
        PSErr(NULL, "Backpropagating NULL `x`");
        return NULL;
    }
    if (y == NULL) {
        PSErr(NULL, "Backpropagating NULL `y`");
        return NULL;
    }
    int ok = 1;
    PSGradient ***new_gradients = NULL;
    if (gradients == NULL) {
        gradients = createGradients(model);
        new_gradients = gradients;
    }
    if (gradients == NULL) return NULL;
    /* Forward pass */
    ok = forward(model, x, 1, opts);
    if (!ok) goto final;
    /* Backward pass */
    int num_models = 1;
    if (PSIsModelChain(model)) {
        num_models = PSModelChainLength(model);
        ok = num_models > 1;
        if (!ok) goto final;
        PSModel *current = PSModelChainTail(model);
        ok = current != NULL;
        if (!ok) goto final;
        int idx = num_models - 1;
        while (current != NULL) {
            PSGradient **grads = gradients[idx--], **res = NULL;
            res = modelBackprop(current, y, opts, grads);
            ok = (res != NULL && res == grads);
            if (!ok) goto final;
            current = current->previous;
        }
    } else {
        PSGradient **grads = modelBackprop(model, y, opts, gradients[0]);
        ok = (grads != NULL && grads == gradients[0]);
    }
final:
    if (!ok) {
        if (new_gradients != NULL) PSDeleteGradientsChain(new_gradients, model);
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
            PSAddVectors(src->biases, dst->biases, dst->biases, src->bias_count,
                         mopts);
        }
        if (src->weight_count > 0 && src->weights != NULL) {
            if (dst->weights == NULL || dst->weight_count != src->weight_count)
            {
                PSErr(NULL, "Cannot sum gradients: destination and source "
                      "weights mismatch");
                return 0;
            }
            PSAddVectors(src->weights, dst->weights, dst->weights,
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
 * Then, gradients are applied on model's parameters (weights and biases)
 * with the specified learing rate and eventually optimized by weight decay
 * (L1, L2, Weight decay) and the specified optimization method (ie. AdaGrad,
 * AdaDelta, etc.).
 * The function will return the calculated error (loss). */
PSFloat updateModelParameters(PSModel *model,
                              PSFloat *training_data,
                              int batch_size, int elements_count,
                              PSFloat rate, PSTrainingOptions* opts, ...)
{
    assert(model->previous == NULL);
    int i, j, gradsize = 0, midx = 0, x_seqlen = 0, y_seqlen = 0,
        iteration = 0, apply_clip = 0;
    assert(batch_size > 0);
    PSFloat *x = NULL; /* Training element */
    PSFloat *y = NULL; /* Labels */
    PSFloat l1 = 0.0, l2 = 0.0, l1_loss = 0.0, l2_loss = 0.0, momentum = 0.0,
            clip_max = 0.0, clip_min = 0.0;
    PSModel *output_model = model;
    int num_models = PSModelChainLength(model);
    if (num_models < 1) {
        PSErr(NULL, "broken model chain");
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return STATUS_ERROR_LOSS;
    } else if (num_models > 1) {
        output_model = PSModelChainTail(model);
        if (output_model == NULL) {
            PSErr(NULL, "broken model chain");
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            return STATUS_ERROR_LOSS;
        }
    }
    PSLayer *output_layer = PSGetOutputLayer(model);
    if (output_layer == NULL) {
        PSErr(NULL, "no output layer");
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return STATUS_ERROR_LOSS;
    }
    int training_data_size = model->input_size;
    int label_data_size = output_layer->size;
    /* Create gradients for the current batch. */
    PSGradient ***gradients = createGradients(model);
    if (gradients == NULL) {
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return STATUS_ERROR_LOSS;
    }
    PSGradient ***bp_gradients = NULL;
    PSFloat **sequences = NULL;
    int is_recurrent = PSIsRecurrent(model), output_is_seq = 0;
    if (is_recurrent || PSUseSequences(model)) {
        va_list args;
        va_start(args, opts);
        sequences = va_arg(args, PSFloat**);
        va_end(args);
        if (sequences == NULL) {
            PSErr(__func__, "Sequences argument is NULL");
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            goto final;
        }
        if (is_recurrent)
            output_is_seq = PSIsRecurrent(output_layer);
        else
            output_is_seq = PSHandleSequenceAtOnce(output_layer);
    }
    UNUSED(elements_count); /* TODO: remove elements_count arg if not needed */
    PSOptimization optimization = PSDefaultOptimization;
    int use_weight_decay = 0, divide_grads_by_batches = 0, training_flags = 0;
    if (opts != NULL) {
        training_flags = opts->flags;
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
    int required_memory_gradients = getRequiredMemoryGradientsCount(opts);
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

    PSMathOpts mopts = {.acceleration = model->acceleration};
    PSGradient ***bp_dest_gradients = gradients;
    if (apply_clip) bp_dest_gradients = NULL;
    beforeBatchTraining(model);
    /* Iterate elements of the batch and, for each element, get gradients
     * from the backpropagation of the error. Then, sum the backpropagation
     * gradients to the batch's gradients. */
    for (i = 0; i < batch_size; i++) {
        int curelem = 0;
        if (model->training != NULL) {
            model->training->current_element =
                (model->training->current_batch * batch_size) + i;
            iteration = model->training->current_element + 1;
            curelem = model->training->current_element;
        }
        /* Backpropagate the error through the model layers and get
         * gradients for the current element. */
        if (sequences == NULL) {
            /* Non-recurrent and non-sequence model */
            int element_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += element_size;
        } else {
            /* Recurrent model or model handling sequences*/
            x = sequences[i];
            int datalen = parseSequenceData(
                model, x, curelem, training_flags, 1,
                &x_seqlen, NULL, &y_seqlen, &y
            );
            if (datalen <= 0 || x_seqlen <= 0 || y_seqlen <= 0) {
                PSModelSetStatus(model, STATUS_ERROR, NULL);
                goto final;
            }
        }
        bp_gradients = backprop(model, x, y, opts, bp_dest_gradients);
        if (bp_gradients == NULL) {
            PSErr(NULL, "Backpropagation failed for model '%s'",
                 (model->name != NULL ? model->name : "UNNAMED")
            );
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            goto final;
        }
        if (apply_clip) {
            PSModel *cur = model;
            while (cur != NULL) {
                PSGradient **srcgrads = bp_gradients[cur->index],
                           **dstgrads = gradients[cur->index];
                gradsize = cur->size - 1;
                clipGradients(srcgrads, clip_min, clip_max, gradsize, &mopts);
                int ok = sumGradients(dstgrads, srcgrads, gradsize, &mopts);
                PSDeleteModelGradients(srcgrads, cur);
                bp_gradients[cur->index] = NULL;
                if (!ok) {
                    PSModelSetStatus(model, STATUS_ERROR, NULL);
                    goto final;
                }
                cur = cur->next;
            }
        }
        if (PSDumpGradientsPath != NULL)
            PSDumpGradients(model, gradients, NULL, opts);
        if (PSModelGetStatus(model) == STATUS_PAUSED) break;
    }

    PSModel *cur = model;
    midx = 0;
    while (cur != NULL) {
        /* Update model paramenters (biases, weights, etc.) by apply
         * batch gradients. */
        gradsize = cur->size - 1;
        PSGradient **grads = gradients[midx];
        if (grads == NULL) {
            PSErr(NULL, "no gradients found for model[%d] (%s)", midx,
                  (cur->name != NULL ? cur->name : "UNNAMED"));
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            goto final;
        }
        PSTrainingContext *training_ctx = getTrainingContext(cur);
        if (training_ctx == NULL) {
            if (!initTrainingContext(cur, opts, required_memory_gradients)) {
                PSErr(NULL, "could not initialize training context on model "
                      "%d", cur->index);
                PSModelSetStatus(model, STATUS_ERROR, NULL);
                goto final;
            }
            training_ctx = getTrainingContext(cur);
            assert(training_ctx != NULL);
        }
        PSGradient **memory_gradients[PS_MAX_MEMORY_GRADIENTS] = {0};
        int memgrad_count = PSGetTrainingMemoryGradients(cur,memory_gradients);
        if (required_memory_gradients > memgrad_count) {
            int ok = initMemoryGradients(
                cur, training_ctx, required_memory_gradients
            );
            if (ok) {
                memgrad_count = PSGetTrainingMemoryGradients(
                    cur, memory_gradients
                );
                ok = (required_memory_gradients == memgrad_count);
            }
            if (!ok) {
                PSErr(NULL, "could not initialize memory gradients on model "
                      "%d", cur->index);
                PSModelSetStatus(model, STATUS_ERROR, NULL);
                return STATUS_ERROR_LOSS;
            }
        }
        for (i = 0; i < gradsize; i++) {
            /* Get layer gradients */
            PSGradient *lgradients = grads[i], *mgradients = NULL,
                       *xgradients = NULL;
            if (lgradients == NULL) continue;
            if (memory_gradients[0] != NULL)
                mgradients = memory_gradients[0][i];
            if (memory_gradients[1] != NULL)
                xgradients = memory_gradients[1][i];
            PSLayer *layer = cur->layers[i + 1];
            if (layer->pretrained) continue;
            if (layer->flags & FLAG_NON_TRAINABLE) continue;
            /* Update Biases */
            if (!(layer->flags & FLAG_NO_BIAS)) {
                if (divide_grads_by_batches) PSDivideVectorScalar(
                    lgradients->biases, (PSFloat) batch_size,
                    lgradients->biases, lgradients->bias_count, &mopts
                );
                int ok = applyGradientsOnBiases(
                    opts, lgradients, layer->biases, mgradients, xgradients,
                    lgradients->bias_count, rate, iteration, mopts.acceleration
                );
                if (!ok) {
                    PSErrNN(__func__, NULL, layer, "failed to update biases");
                    PSModelSetStatus(model, STATUS_ERROR, NULL);
                    goto final;
                }
            }
            /* Update Weights */
            uint64_t wgrad_offset = 0;
            int partially_trainable = (layer->type == Attention);
            for (j = 0; j < layer->weight_types_count; j++) {
                PSMatrix weights = layer->weights[j];
                if (weights == NULL) {
                    if (partially_trainable) continue;
                    PSErr(
                        __func__, "Layer[%d]: weights[%d] is NULL",
                        layer->index, j
                    );
                    PSModelSetStatus(model, STATUS_ERROR, NULL);
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
                        PSModelSetStatus(model, STATUS_ERROR, NULL);
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
                    PSModelSetStatus(model, STATUS_ERROR, NULL);
                    goto final;
                }
                wgrad_offset += wlen;
            }
        }
        midx++;
        cur = cur->next;
    }
final:
    PSDeleteGradientsChain(gradients, model);
    if (PSModelGetStatus(model) == STATUS_ERROR) return STATUS_ERROR_LOSS;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    if (onehot) label_data_size = 1;
    if (output_is_seq) label_data_size *= y_seqlen;
    PSFloat outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (onehot) {
            int idx = (int) *(y + i);
            outputs[i] = PSGetState(output_layer, idx, i);
        } else {
            if (!output_is_seq) outputs[i] = PSGetState(output_layer, i);
            else i = fetchSequenceOutputState(output_layer, outputs, i, 0);
        }
    }
    if (opts == NULL) l1 = l2 = 0.0;
    if (l1 != 0.0) l1_loss *= (opts->l1_decay / batch_size);
    if (l2 != 0.0) l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    int onehot_size = (onehot ? output_layer->size : 0);
    PSFloat loss =
        output_model->loss(outputs, y, label_data_size, onehot_size);
    if (output_is_seq && y_seqlen > 0) loss /= y_seqlen;
    return loss + l1_loss + l2_loss;
}

/* Iterate training data for the entire epoch. Unless the training flag
 * TRAINING_NO_SHUFFLE is set, training data is randomly shuffled
 * (Stochastic Gradient Descent). Training data is divided into batches
 * depending on `batch_size` and, for each batch, gradients are generated
 * and applied on model's by the `updateModelParameters` */
PSFloat gradientDescent(PSModel *model,
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
    PSTrainingContext *training_ctx = getTrainingContext(model);
    if (training_ctx == NULL) {
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return STATUS_ERROR_LOSS;
    }
    PSLogTrainingProgressFunc log_progress = NULL;
    int batches_count = elements_count / batch_size;
    PSFloat **sequences = NULL, **sequence_head = NULL;
    int flags = 0, do_validate = 0;
    int is_model_chain = PSIsModelChain(model);
    if (options != NULL) flags = options->flags;
    if (PSIsRecurrent(model) || PSHandleSequenceAtOnce(model)) {
        PSLayer *out = PSGetOutputLayer(model);
        if (out == NULL) {
            PSErr(NULL, "no output layer");
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            return STATUS_ERROR_LOSS;
        }
        PSModel *output_model = model;
        if (is_model_chain) {
            output_model = PSModelChainTail(model);
            if (output_model == NULL) {
                PSModelSetStatus(model, STATUS_ERROR, NULL);
                return STATUS_ERROR_LOSS;
            }
        }
        sequences = getDatasetSequences(
            model, training_data, elements_count, flags
        );
        if (sequences == NULL) {
            PSModelSetStatus(model, STATUS_ERROR, NULL);
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
    if (options != NULL) {
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
        log_progress = options->log_progress;
    }
    if (log_progress == NULL) log_progress = PSLogTrainingProgress;
    sequence_head = sequences;
    for (i = 0; i < batches_count; i++) {
        model->training->current_batch = i;
        int batch_num = i + 1;
        struct timeval st, et;
        gettimeofday(&st, NULL);
        PSFloat batch_err = updateModelParameters(
            model, training_data, batch_size, elements_count, learning_rate,
            options, sequence_head
        );
        gettimeofday(&et, NULL);
        elapsed_t = PSGetElapsedTimeUS(st, et);
        err += batch_err;
        if (PSModelGetStatus(model) == STATUS_ERROR) {
            PSErr(NULL, "Gradient descent failed at batch %d for model '%s'",
                  i, (model->name != NULL ? model->name : "UNNAMED")
            );
            goto final;
        }
        tot_t += elapsed_t;
        avg_t = (tot_t / batch_num);
        if (batch_num < batches_count) {
            avg_err = err / (PSFloat) batch_num;
            if (do_validate) {
                if (i > 0 && (batch_num % validate_every) == 0) {
                    log_progress(
                        model, STATUS_VALIDATING, epochs, batches_count,
                        NULL, NULL, NULL, 0, test_data_size / element_size
                    );
                    acc = validate(
                        model, test_data, test_data_size, options, 0
                    );
                    tot_acc += acc;
                    avg_acc = tot_acc / (PSFloat) ++validations;
                }
                log_progress(
                    model, STATUS_TRAINING, epochs, batches_count,
                    &avg_err, &avg_acc, &avg_t, 0, 0
                );
            } else {
                log_progress(
                    model, STATUS_TRAINING, epochs, batches_count,
                    &avg_err, NULL, &avg_t, 0, 0
                );
            }
        } else {
            log_progress(
                model, STATUS_TRAINING, epochs, batches_count, NULL, NULL,
                NULL, 0, 0
            );
        }
        if (PSModelGetStatus(model) == STATUS_ERROR) {
            if (sequences != NULL) free(sequences);
            return STATUS_ERROR_LOSS;
        }
        if (model->onBatchTrained != NULL) {
            model->onBatchTrained(
                model, model->training->current_epoch,
                epochs, avg_err, batch_err, avg_acc, &learning_rate,
                (sequences != NULL ? *sequence_head : training_data)
            );
        }
        if (sequences == NULL) training_data += offset;
        else sequence_head += batch_size;
        int action = model->training->requested_action;
        if (action == ACTION_ABORT) {
            PSModelSetStatus(model, action, NULL);
            break;
        }
    }
final:
    if (sequences != NULL) free(sequences);
    return err / (PSFloat) batches_count;
}

float validate(PSModel *model, PSFloat *test_data, int data_size,
               PSTrainingOptions *opts, int log)
{
    int i, j;
    unsigned char previous_status = PSModelGetStatus(model);
    char *errmsg = NULL;
    float accuracy = 0.0f;
    int correct_results = 0;
    float correct_amount = 0.0f;
    PSLayer *output_layer = PSGetOutputLayer(model);
    int input_size = model->input_size;
    int output_size = model->output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int y_size = (onehot ? 1 : output_size);
    int element_size = input_size + output_size;
    int elements_count;
    int reads_input_sequence = 0, emits_output_sequence = 0;
    int flags = (opts != NULL ? opts->flags : 0);
    PSFloat **sequences = NULL;
    if (PSUseSequences(model)) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(test_data++);
        data_size--;
        reads_input_sequence = PSUseSequences(model->layers[0]);
        emits_output_sequence = PSUseSequences(output_layer);
        sequences = getDatasetSequences(
            model, test_data, elements_count, flags
        );
        if (sequences == NULL) goto err;
    } else elements_count = data_size / element_size;
    /* PSFloat outputs[output_size]; */
    if (log) printf("Test data elements: %d\n", elements_count);
    PSModelSetStatus(model, STATUS_VALIDATING, NULL);
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

            int ok = PSForward(model, inputs);
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
                seqlen = (int) *(inputs + model->input_size);
                expected = inputs + model->input_size + 1;
            }
            if (seqlen == 0) {
                errmsg = "recurrent data with zero seqlen";
                goto err;
            }
            int ok = PSResetModelStateSequences(model, seqlen, 0);
            if (!ok) goto err;
            ok = PSForward(model, inputs);
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
    PSModelSetStatus(model, previous_status, NULL);
    return accuracy;
err:
    if (errmsg == NULL) {
        if (PSModelGetStatus(model) == STATUS_VALIDATING)
            errmsg = "an error occurred while validating, aborting!";
        else
            errmsg = "failed to validate model";
    }
    PSModelSetStatus(model, STATUS_ERROR, NULL);
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
    if (options->log_progress == NULL)
        options->log_progress = PSLogTrainingProgress;
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
    if (options->log_progress == NULL)
        options->log_progress = PSLogTrainingProgress;
}

int isSequence2SequenceAvailable(PSModel *model, char **err) {
    if (model == NULL) return 0;
    PSModel *input_model = model, *output_model = model;
    int count = PSModelChainLength(model);
    if (err != NULL) *err = NULL;
    if (count > 1) {
        input_model = PSModelChainHead(model);
        output_model = PSModelChainTail(model);
        if (input_model == NULL || output_model == NULL) {
            if (err != NULL) *err = "broken model chain";
            return 0;
        }
    }
    PSLayer *input_layer = input_model->layers[0];
    PSLayer *output_layer = output_model->layers[output_model->size - 1];
    if (input_layer == NULL || output_layer == NULL) {
        if (err != NULL) *err = "could not determine input and output layers";
        return 0;
    }
    if (!PSUseSequences(input_layer)) {
        if (err != NULL) *err = "input layer does not accept sequences";
        return 0;
    }
    if (!PSUseSequences(output_layer)) {
        if (err != NULL) *err = "output layer does not produce sequences";
        return 0;
    }
    if (!(output_model->flags & FLAG_AUTOREGRESSION)) {
        if (err != NULL)
            *err = "no FLAG_AUTOREGRESSION in output model's flags";
        return 0;
    }
    return 1;
}

int PSPretrainLayers(PSModel *model, PSFloat *training_data,
                     int data_size)
{
    if (model->flags & FLAG_PRETRAINER) return 1;
    int original_status = PSModelGetStatus(model);
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) continue;
        if (!isPretrainableLayer(layer)) continue;
        if (layer->pretrained) continue;
        PSInfo("Pretraining layer[%d] (%s)", i, PSGetLayerTypeLabel(layer));
        PSModelSetStatus(model, STATUS_PRETRAINING, NULL);
        int trained = layer->pretrain(layer, training_data, data_size);
        if (!trained) {
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            return 0;
        }
        PSInfo("Successfully pretrained layer[%d] (%s)",
               i, PSGetLayerTypeLabel(layer));
        PSModelSetStatus(model, original_status, NULL);
    }
    return 1;
}

/* Train `model` over `training_data`. Training epochs, batch size,
 * optimization, and other optimizer settings are defined into optional
 * `options`.
 * Arguments:
 *  - `model`: The neural model to be trained (mandatory)
 *  - `training_data`: an array of `PSFloat` containing the tarining dataset
 *     (ie. inputs, expected predictions)
 *  - `data_size`: length of `training_data` array.
 *  - `test_data`: optional dataset that can be used for testing purpose
 *  - `test_size`: length of `test_data` array.
 *  - `options`: optional training options (see `PSTrainingOptions`).
 *               If NULL, the training process will use default options.
 * Training/test data layout:
 *  - For normal feedforward models, the array must contain alternating
 *    inputs/predictions pairs, one pair for each element to be trained.
 *    So, each training/test element pair must contain:
 *      - Input values, having the same length of the model's input layer
 *      - Prediction values, having the same length of the model's output
 *        layer. If output layer has the `FLAG_ONEHOT` flag, predictions length
 *        muse be 1, and it must contain the index of the expected maximum
 *        state.
 *    Total number of traing elements is given by:
 *      array size / (input_size + output_size)
 *  - For recurrent model or models using sequences, layout can have
 *    different forms.
 *    Regardless of that, first element of the array must contain the total
 *    number of training/test elements.
 *    For each training/test sequence, the sequence length must be specified.
 *    Different forms can be:
 *    - Many-to-many: the default mode for recurrent models that produce
 *      sequences having the same length of the input sequence.
 *      In this case, the first element of the sequence segment is the
 *      sequence length, followed by inputs/predictions pair.
 */
void PSTrain(PSModel *model,
             PSFloat *training_data,
             int data_size,
             PSFloat *test_data,
             int test_size,
             PSTrainingOptions *options)
{
    int epochs = 0, batch_size = 0, i, elements_count;
    PSFloat learning_rate = 0.0;
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            return;
        }
    }
    int valid = PSModelCheck(model);
    if (!valid) {
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return;
    }
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) {
        PSErr(__func__, "missing model context");
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        return;
    }
    PSTrainingContext *training_ctx = ctx->training_context;
    if (training_ctx && PSModelGetStatus(model) == STATUS_UNTRAINED) {
        deleteTrainingContext(training_ctx, model);
        ctx->training_context = training_ctx = NULL;
    }
    if (training_ctx == NULL) {
        ctx->training_context = calloc(1, sizeof(PSTrainingContext));
        if (ctx->training_context == NULL) {
            PSModelSetStatus(model, STATUS_ERROR, NULL);
            PSPrintMemoryErrorMsg();
            return;
        }
    }
    PSTrainingOptions default_options = {0};
    if (options == NULL) {
        options = &default_options;
        PSSetDefaultTrainingOptions(options);
    }
    checkTrainingOptions(options);
    PSLogTrainingProgressFunc log_progress = options->log_progress;
    epochs = options->epochs;
    batch_size = options->batch_size;
    learning_rate = options->learning_rate;
    if (learning_rate <= 0.0) {
        PSErr(__func__, "learning_rate must be > 0.0");
        return;
    }
    if (batch_size <= 0) batch_size = options->batch_size = 1;
    if (epochs <= 0) epochs = options->epochs = 1;
    training_ctx = ctx->training_context;
    training_ctx->options = *options;
    int num_models = PSModelChainLength(model);
    PSModel *input_model = model, *output_model = model;
    if (num_models > 1) {
        input_model = PSModelChainHead(model);
        output_model = PSModelChainTail(model);
        if (input_model == NULL || output_model == NULL) {
            PSErr(__func__, "broken model chain");
        }
    }
    int input_size = input_model->input_size,
        output_size = output_model->output_size;
    int element_size = input_size + output_size;
    /* Eventually pretrain layers (if pretrainable layers are found) */
    if (!PSPretrainLayers(model, training_data, data_size)) {
        PSModelSetStatus(model, STATUS_ERROR, NULL);
        PSErr(__func__, "failed to pretrain model '%s'",
              model->name != NULL ? model->name : "UNNAMED");
        return;
    }
    int is_recurrent = PSIsRecurrent(model);
    int handle_seq = PSHandleSequenceAtOnce(model);
    int use_sequences = is_recurrent || handle_seq;
    if (use_sequences) {
        /*  First training data number for Recurrent networks must indicate */
        /*  the data elements count */
        elements_count = (int) *(training_data++);
        data_size--;
    } else elements_count = data_size / element_size;
    if (options->flags & TRAINING_FLAG_SEQ2SEQ) {
        char *err = NULL;
        if (!isSequence2SequenceAvailable(model, &err)) {
            PSErr(__func__, "Sequence-to-sequence (TRAINING_FLAG_SEQ2SEQ) is "
                  "not available for this model: %s", err);
            return;
        }
    }
    const char *name = model->name != NULL ? model->name : "UNNAMED";
    if (num_models == 1)
        PSLog(PSLOGLEVEL_NOTICE, "Training model \"%s\"\n", name);
    else {
        PSLog(PSLOGLEVEL_NOTICE, "Training multi-model model\n");
        PSInfo("Number of models:         %d", num_models);
        const char *input_name = (
            input_model->name != NULL ? input_model->name : "UNNAMED"
        );
        const char *output_name = (
            output_model->name != NULL ? output_model->name : "UNNAMED"
        );
        PSInfo("Input model:               \"%s\"", input_name);
        PSInfo("Output model:              \"%s\"", output_name);
    }
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
    PSInfo("Data shuffle:               %s", (!no_shuffle ? "yes" : "no"));
    if (is_recurrent)
        PSInfo("BPTT Truncate:              %d", options->bptt_truncate);
    if (model->layers[model->size - 1]->flags & FLAG_ONEHOT)
        PSInfo("Onehot Labels:              yes");
    if (options->flags & TRAINING_FLAG_TEACHER_FORCING)
        PSInfo("Teacher forcing:            yes");
    char *loss_func_name = NULL;
    if (output_model->loss != NULL) {
        loss_func_name = getLossFunctionName(output_model->loss);
        PSInfo("Loss Function:              %s", loss_func_name);
    }
    /* Start training */
    int was_paused = (PSModelGetStatus(model) == STATUS_PAUSED);
    PSModelSetStatus(model, STATUS_TRAINING, NULL);
    time_t start_t, end_t;
    char timestr[80];
    struct tm *tminfo;
    struct timeval epoch_st, epoch_et;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    PSLog(PSLOGLEVEL_NOTICE, "Training started at %s\n", timestr);
    PSFloat prev_loss = 0.0;
    float acc = -999.99f;
    int adjust_rate = 0;
    if (options != NULL) adjust_rate = (options->flags & TRAINING_ADJUST_RATE);
    int first_epoch = 0;
    if (model->training != NULL) {
        if (was_paused) first_epoch = model->training->current_epoch;
    } else {
        model->training = malloc(sizeof(PSTrainingInfo));
        model->training->started_at = start_t;
        model->training->ended_at = (time_t) 0;
        if (options != NULL)
            model->training->debug_dump_to = options->debug_dump_to;
        else model->training->debug_dump_to = NULL;
        if (model->training->debug_dump_to) PSTrainingDebugDumpHeader(
            model, data_size, test_size, epochs, learning_rate, batch_size
        );
    }
    model->training->batch_size = batch_size;
    model->training->requested_action = ACTION_NONE;
    for (i = first_epoch; i < epochs; i++) {
        model->training->current_epoch = i;
        if (is_recurrent) {
            if (!PSResetModelStateSequences(model, 0, 0)) {
                PSModelSetStatus(model, STATUS_ERROR, NULL);
                PSErr(__func__, "Failed to reset model recurrent states");
                return;
            }
        }
        gettimeofday(&epoch_st, NULL);
        PSFloat loss = gradientDescent(model, training_data, element_size,
                                       elements_count, learning_rate,
                                       batch_size, options, epochs,
                                       test_data, test_size);
        gettimeofday(&epoch_et, NULL);
        time_t elapsed_t = PSGetElapsedTimeUS(epoch_st, epoch_et);
        if (PSModelGetStatus(model) == STATUS_ERROR) {
            PSLog(
                PSLOGLEVEL_ERROR, "\nAn error occurred while training, "
                "aborting!\n"
            );
            return;
        }
        int batches_count = elements_count / batch_size;
        PSFloat *acc_p = NULL;
        if (test_data  && PSModelGetStatus(model) == STATUS_TRAINING) {
            log_progress(
                model, STATUS_VALIDATING, epochs, batches_count, NULL,
                NULL, NULL, 0, 0
            );
            acc = validate(model, test_data, test_size, options, 0);
            acc_p = (PSFloat *) &acc;
        }
        if (i > 0 && loss > prev_loss && adjust_rate)
            learning_rate *= 0.5;
        if (model->onEpochTrained != NULL) {
            model->onEpochTrained(
                model, i, epochs, loss, loss, acc, &learning_rate, NULL
            );
        }
        prev_loss = loss;
        log_progress(
            model, STATUS_TRAINING, epochs, batches_count, &loss,
            acc_p, &elapsed_t, 0, 0
        );
        fflush(stdout);
        int action = model->training->requested_action;
        if (action == ACTION_ABORT || action == ACTION_PAUSE) {
            PSModelSetStatus(model, action, NULL);
            break;
        }
    }
    time(&end_t);
    log_progress(model,STATUS_TRAINED,epochs,0,NULL,NULL,NULL,0,0);
    PSLineEnd();
    fflush(stdout);
    PSLog(PSLOGLEVEL_SUCCESS, "\nCompleted in %ld sec.\n", end_t - start_t);
    model->training->ended_at = end_t;
    if (PSModelGetStatus(model) == STATUS_TRAINING)
        PSModelSetStatus(model, STATUS_TRAINED, NULL);
    if (is_recurrent) {
        PSModel *current = input_model;
        while (current != NULL) {
            PSResetModelStateSequences(current, 0, 0);
            current = current->next;
        }
    }
}

float PSTest(PSModel *model, PSFloat *test_data, int data_size,
             PSTrainingOptions *options)
{
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO);
    return validate(model, test_data, data_size, options, do_log);
}

void PSPauseTraining(PSModel *model) {
    if (model->training != NULL) {
        printf("\nPause requested, "
               "training will stop after current epoch will be completed.\n");
        model->training->requested_action = ACTION_PAUSE;
    }
}

void PSAbortTraining(PSModel *model) {
    if (model->training != NULL) {
        printf("\nAborting...\n");
        model->training->requested_action = ACTION_ABORT;
    }
}

int PSModelCheck(PSModel *model) {
    if (model == NULL) {
        PSErr(__func__, "model is null");
        return 0;
    }
    int size = model->size, i;
    if (size == 0) {
        PSErr(__func__, "empty model");
        return 0;
    }
    int is_recurrent = PSIsRecurrent(model);
    PSLayer *previous = NULL;
    int onehot_input = 0;
    int recurrent_type_layers = 0, recurrent_layers = 0,
        recurrent_input = 0, recurrent_output = 0;
    PSLayer *actual_first_recurrent_layer = NULL,
            *actual_last_recurrent_layer = NULL,
            *first_recurrent_layer = PSGetFirstRecurrentLayer(model),
            *last_recurrent_layer = PSGetLastRecurrentLayer(model);
    PSLayer *output_layer = model->layers[size - 1];
    recurrent_output = PSIsRecurrent(output_layer);
    for (i = 0; i < size; i++) {
        PSLayer *layer = model->layers[i];
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
                      "Convolutional Neural Networks");
                return 0;
            }
            /* TODO: remove this contraint */
            if (model->flags & FLAG_RECURRENT) {
                PSErr(
                    __func__,
                    "Sorry, Convolutional layers aren't yet supported "
                    "on Recurrent Neural Networks :("
                );
                return 0;
            }
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
        if (layer->activate == PSGelu &&
            layer->derivative != PSGeluDerivative)
        {
            PSErr(__func__,
                  "Layer[%d] activate function is PSGelu, "
                  "but derivative function is not PSGeluDerivative", i);
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
        PSRecurrentNetworkMode rnn_mode = model->rnn_mode;
        if (rnn_mode == NonRecurrent) {
            PSErr(__func__, "Recurrent network mode is NonRecurrent");
            return 0;
        }
        if (recurrent_layers == 0) {
            PSErr(__func__,
                "model is recurrent but has no recurrent layers"
            );
            return 0;
        }
        if (recurrent_type_layers == 0) {
            PSErr(__func__,
                "model is recurrent but has no Recurrent, LSTM or GRU layers"
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
                "but model is not updated", actual_first_recurrent_layer
            );
            return 0;
        }
        if (last_recurrent_layer != actual_last_recurrent_layer) {
            PSErr(__func__,
                "Recurrent network last recurrent layer should be layer %d, "
                "but model is not updated", actual_last_recurrent_layer
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
            PSErr(
                __func__, "model is not recurrent but has Recurrent, "
                "LSTM or GRU layers"
            );
            return 0;
        }
    }
    int softmax_output = (output_layer->type == SoftMax);
    if (model->loss == NULL) {
        PSErr(__func__, "Missing loss function");
        return 0;
    } else {
        if (softmax_output && model->loss != PSCrossEntropyLoss) {
            PSErr(__func__, "SoftMax output requires PSCrossEntropyLoss");
            return 0;
        } else if (!softmax_output && model->loss == PSCrossEntropyLoss) {
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

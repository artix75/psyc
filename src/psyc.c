/*
 * Copyright (C) 2016-2024 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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
#include "operator-layer.h"
#include "debug.h"

#define LAYER_PLACEHOLDER_TYPE -1
#define PS_STATUS_ERROR_LOSS ((PSFloat) FLT_MIN)
#define BPTT_TRUNCATE   0
#define SEQ_MODE_PREPEND_START      (1 << 0)
#define SEQ_MODE_APPEND_END         (1 << 1)
#define SEQ_MODE_PREPEND_SEQLEN     (1 << 2)
#define VALIDATE_LOG_ALL        1
#define VALIDATE_LOG_PROGRESS   2

#define applyGradientsOnBiases(opts, grads, params, mg, xg, len, r, i, accel) \
    applyGradientsOnParameters(PS_PARAM_BIAS, opts, grads, params, mg, xg, 0, \
    len, r, i, accel)
#define applyGradientsOnWeights(opts,grad,val,mg,xg,offs,len,r,i,accel) \
    applyGradientsOnParameters(PS_PARAM_WEIGHT, opts, grad, val, mg, xg,\
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
    seqlen = va_arg(_args, long);\
    if (is_recurrent) step = va_arg(_args, long);\
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
    const char          *allocated_name;
} PSModelContext;

static PSLossFunction loss_functions[] = {
    NULL,
    PSQuadraticLoss,
    PSCrossEntropyLoss
};

static size_t loss_functions_count = sizeof(loss_functions) /
                                     sizeof(PSLossFunction);

/* Forward Declarations */

char *getLossFunctionName(PSLossFunction function);
char *getModelStatusLabel(PSModel *model);
float validate(PSModel *model, PSFloat *test_data, long data_size,
               PSTrainingOptions *opts, PSFloat *loss, int log);
int PSInitConvolutionalLayer(PSModel *model, PSLayer *layer,
                             PSLayerDef *ldef);
int PSInitPoolingLayer(PSModel *model, PSLayer *layer,
                       PSLayerDef *ldef);
int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *layer_gradients, ...);
int PSInitRecurrentLayer(PSModel *model, PSLayer *layer,
                         long size, long ws, PSLayerDef *ldef);
int PSInitLSTMLayer(PSModel *model, PSLayer *layer,
                    long size, long ws, PSLayerDef *ldef);
int PSInitGRULayer(PSModel *model, PSLayer *layer,
                   long size, long ws, PSLayerDef *ldef);
int PSInitDropoutLayer(PSModel *model, PSLayer *layer,
                       PSLayerDef *layer_def);
int PSInitEmbeddingLayer(PSLayer *layer, long size, PSLayerDef *ldef);
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
size_t writeSerializedFloat(FILE *out, PSFloat fnum, int opts);
int PSBeforeSequenceForward(PSLayer *layer, long seqlen, long t);
char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
int updateModelChain(PSModel *head);
static int useAutoRegression(PSModel *model, PSForwardOptions *forward_opts,
                             PSTrainingOptions *training_opts);
PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSModel *model);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);
static PSModel *cloneModel(PSModel *model, int layout_only,
                                     PSModel *parent);
int PSIsLayerPlaceholder(PSLayer *layer);
void PSAbortLayer(PSModel *model, PSLayer *layer);
PSTrainingOptions *PSGetModelTrainingOptions(PSModel *model);
void updateTrainingAccuracy(PSModel *model, PSFloat *outputs, PSFloat *targets,
                            long seqlen);

/**** Miscellaneous functions ****/

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
                           long batches, PSFloat *loss, float *accuracy,
                           PSFloat *test_loss,  float *test_accuracy,
                           time_t *elapsed)
{
    static int printed_epoch = -1;
    static size_t min_fixed_len = 0;
    static size_t fixed_prefix_len = 0;
    static char *default_prefix = "Batch ";
    unsigned int tcols = PSGetTerminalColumns();
    if (PSLogLevel > PSLOGLEVEL_INFO) return;
    if (model->training == NULL) return;
    if (status == PS_STATUS_TRAINED || status == PS_STATUS_ERROR) {
        PSLineEnd();
        printed_epoch = -1;
        return;
    }
    int use_color = PSLogColorEnabled();
    if (min_fixed_len == 0) {
        /* Predict possible minimum length for fixed elements. */
        min_fixed_len = strlen(
            "999%, loss=99.99, val.loss=99.99, val.acc=9.99 (999.9ms)"
        );
        fixed_prefix_len = strlen(default_prefix);
    }
    char *line_prefix = "";
    if (!use_color) line_prefix = default_prefix;
    if (model->training->current_batch == 0) {
        if (printed_epoch != model->training->current_epoch) {
            PSLineEnd();
            if (use_color) printf(PSCOLOR_BOLD);
            printf("Epoch %d/%d:\n", model->training->current_epoch + 1,
                    epochs);
            if (use_color) printf(PSCOLOR_RESET);
            fflush(stdout);
            printed_epoch = model->training->current_epoch;
        }
    }
    long batch_num = model->training->current_batch + 1;
    int percent =
        (int) roundf(((float) batch_num / (float) batches) * 100.0f);
    size_t pad = PSCalcIntStringLength(batches),
           min_len = min_fixed_len + 1 + pad * 2;
    if (!use_color && (min_len + fixed_prefix_len) > tcols) line_prefix = "";
    int lnflags = PS_LINE_PLAIN_ASCII | PS_LINE_FILL;
    PSLineStart(
        PS_LINE_OVERWRITE, "%s%*d/%d %3d%%", line_prefix, pad, batch_num,
        batches, percent
    );
    char *elapsed_str = NULL, *eta_str = NULL;
    if (elapsed != NULL) {
        elapsed_str = PSGetElapsedTimeString(*elapsed, 0);
        if (status == PS_STATUS_TRAINING && batch_num < batches) {
            time_t eta = *elapsed * (batches - batch_num);
            eta_str = PSGetElapsedTimeString(eta, PS_OPT_TIME_ROUND_SEC);
        }
    }
    if (status == PS_STATUS_VALIDATING) {
        if (model->training != NULL && model->training->num_tests > 0) {
            int val_perc = (int) roundf(
                (
                    (float) (model->training->current_test + 1) /
                    (float) model->training->num_tests
                ) * 100.0f
            );
            PSLineAppend(lnflags, ", validating %d%%", val_perc);
        } else PSLineAppend(lnflags, ", validating...");
    } else if (status == PS_STATUS_TRAINING) {
        if (batch_num < batches) {
            assert(loss != NULL);
            if (accuracy != NULL) {
                PSLineAppend(
                    PS_LINE_PLAIN_ASCII, ", loss=%.2lf, acc=%.2lf, %s/batch",
                    *loss, *accuracy, elapsed_str
                );
            } else {
                PSLineAppend(PS_LINE_PLAIN_ASCII, ", loss=%.2lf, %s/batch",
                             *loss, elapsed_str);
            }
            if (eta_str != NULL) PSLineAppend(lnflags, " (ETA: %s)", eta_str);
            else PSLineFill();
        } else {
            int n_info = 0;
            if (loss != NULL) {
                PSLineAppend(PS_LINE_PLAIN_ASCII, ", loss=%.2lf", *loss);
                n_info++;
            }
            if (accuracy != NULL) {
                PSLineAppend(PS_LINE_PLAIN_ASCII, ", acc=%.2f", *accuracy);
                n_info++;
            }
            if (test_loss != NULL) {
                PSLineAppend(
                    PS_LINE_PLAIN_ASCII, ", val.loss=%.2lf", *test_loss
                );
                n_info++;
            }
            if (test_accuracy != NULL) {
                PSLineAppend(
                    PS_LINE_PLAIN_ASCII, ", val.acc=%.2f", *test_accuracy
                );
                n_info++;
            }
            if (elapsed != NULL) {
                PSLineAppend(PS_LINE_PLAIN_ASCII, " (%s)", elapsed_str);
                n_info++;
            }
            PSLineFill();
            if (n_info > 0) PSLineEnd();
        }
    } else {
        PSLineFill();
    }
    free(elapsed_str);
    free(eta_str);
}

void PSTrainingProgressBar(PSModel *model, int status, int epochs,
                           long batches, PSFloat *loss, float *accuracy,
                           PSFloat *test_loss,  float *test_accuracy,
                           time_t *elapsed)
{
    UNUSED(test_accuracy);
    UNUSED(test_loss);
    static int epoch_printed = -1;
    static int bar_width = -1;
    static int print_batches = 1;
    if (PSLogLevel > PSLOGLEVEL_INFO) return;
    if (model->training == NULL) return;
    int training_ended = (
        status == PS_STATUS_TRAINED || status == PS_STATUS_ERROR ||
        status == PS_STATUS_ABORTED
    );
    if (training_ended) {
        PSLineEnd();
        epoch_printed = -1;
        bar_width = -1;
        return;
    }
    unsigned int tw = PSGetTerminalColumns();
    unsigned int pad = (unsigned int) PSCalcIntStringLength(batches);
    long batch_num = model->training->current_batch + 1;
    if (bar_width < 0) {
        /* Determine progress bar width depending on the length of the other
         * elements. */
        PSTrainingOptions *topts = PSGetModelTrainingOptions(model);
        int metrics = (topts != NULL ? topts->metrics : 0),
            has_tests = (model->training->test_size > 0);
        char sfx_sample[50] = {0};
        int sfxlen;
        if (!has_tests) sfxlen = snprintf(sfx_sample, 50, " | loss: 99.99");
        else sfxlen = snprintf(sfx_sample, 50, " | loss=99.99");
        if (sfxlen < 0) return;
        char *sfx_p = sfx_sample + sfxlen;
        if (metrics & PS_TRAINING_METRICS_ACCURACY) {
            if (!has_tests)
                sfxlen += snprintf(sfx_p, 50 - sfxlen, "| acc: 9.99");
            else
                sfxlen += snprintf(sfx_p, 50 - sfxlen, "| acc=9.99");
            sfx_p = sfx_sample + sfxlen;
        }
        if (has_tests) {
            sfxlen += snprintf(
                sfx_p, 50 - sfxlen, " | v.loss: 99.99 | v.acc: 9.99"
            );
            sfx_p = sfx_sample + sfxlen;
        } else sfxlen += snprintf(sfx_p, 50 - sfxlen, " | 999.9ms");
        bar_width = tw - (unsigned) sfxlen - ((pad * 2) + 2);
        if (bar_width < 26) {
            print_batches = 0;
            bar_width += ((pad * 2) + 2);
        }
    }
    int use_color = PSLogColorEnabled();
    if (model->training->current_batch == 0) {
        if (epoch_printed != model->training->current_epoch) {
            PSLineEnd();
            /*if (use_color) printf(PSCOLOR_BOLD);*/
            printf("Epoch %d/%d:\n", model->training->current_epoch + 1,
                    epochs);
            /*if (use_color) printf(PSCOLOR_RESET);*/
            fflush(stdout);
            epoch_printed = model->training->current_epoch;
        }
    }
    if (print_batches)
        PSLineStart(PS_LINE_OVERWRITE, "%*d/%d ", pad, batch_num, batches);
    else {
        int percent = (int)roundf(((batch_num + 1) / (float)batches) * 100.0f);
        PSLineStart(PS_LINE_OVERWRITE, "%*-3d%% ", percent);
    }
    char *elapsed_str = NULL;
    char sfx[35] = {0};
    int epoch_ended = 0;
    if (status == PS_STATUS_VALIDATING) snprintf(sfx, 35, " | validating...");
    else if (status == PS_STATUS_TRAINING) {
        if (elapsed != NULL) elapsed_str = PSGetElapsedTimeString(*elapsed, 0);
        if (batch_num < batches) {
            if (accuracy != NULL) {
                snprintf(
                    sfx, 35, " | loss: %5.2lf | acc: %.2lf | %s",
                    *loss, *accuracy, elapsed_str
                );
            } else {
                snprintf(
                    sfx, 35, " | loss: %5.2lf | %s", *loss, elapsed_str
                );
            }
        } else {
            if (loss != NULL) {
                if (accuracy != NULL) {
                    snprintf(
                        sfx, 35, " | loss: %5.2lf | acc: %.2f | %s",
                        *loss, *accuracy, elapsed_str
                    );
                } else {
                    snprintf(
                        sfx, 35, " | loss: %5.2lf | %s", *loss, elapsed_str
                    );
                }
                epoch_ended = 1;
            }
        }
    }
    int style = PS_PROGRESS_STYLE_DOUBLE_DASH, color = 0,
        flags = PS_PROGRESS_FLAG_JUST_BAR;
    if (use_color) {
        color = 1;
        style = PS_PROGRESS_STYLE_LINE;
    }
    PSProgressBar(batch_num, batches, style, color, flags, bar_width, NULL);
    if (sfx[0]) {
        int lnflags = PS_LINE_PLAIN_ASCII | PS_LINE_FILL;
        PSLineAppend(lnflags, "%s", sfx);
    }
    if (epoch_ended) PSLineEnd();
    free(elapsed_str);
}

void dumpForwardStep(long i, PSFloat a, PSFloat b, PSFloat sum,
                     int using_acceleration, PSMathOpts *opts)
{
    UNUSED(a);
    UNUSED(b);
    UNUSED(sum);
    UNUSED(using_acceleration);
    if (opts == NULL) return;
    PSDebugStepInfo *info = (PSDebugStepInfo *) opts->data;
    if (info == NULL || info->layer == NULL || info->model == NULL) return;
    info->training_phase = PS_TRAINING_PHASE_FORWARD;
    long prev_idx = info->layer->index - 1;
    PSTrainingDebugDumpStep(
        info, "previous_neuron=%ld-%ld,weight_index=%ld\n", prev_idx, i, i
    );
}

static int sequenceHasStartItem(PSFloat *seq, long itemsize,
                                PSSequenceSettings *sequence_settings)
{
    if (sequence_settings->start == NULL) return 0;
    if (itemsize <= 0) return 0;
    if (itemsize == 1) return *seq == *(sequence_settings->start);
    else {
        return memcmp(
            seq, sequence_settings->start, (size_t) itemsize * sizeof(PSFloat)
        ) == 0;
    }
}

static long sequenceFindLastNonPadItem(PSFloat *seq, long len, long itemsize,
                                       long pad, PSFloat **pad_vector, int *err)
{
    if (err != NULL) *err = 0;
    if (itemsize <= 0 || len <= 0) return -1;
    long idx = len;
    PSFloat *padvec = NULL;
    if (itemsize > 1) {
        PSFloat *padvec = PSOneHotVector(pad, itemsize);
        if (padvec == NULL) {
            if (err != NULL) *err = 1;
            return -2;
        }
        if (pad_vector != NULL) *pad_vector = padvec;
    }
    while (--idx > -1) {
        if (itemsize == 1) {
            long item = (long) seq[idx];
            if (item != pad) break;
        } else {
            PSFloat *last = seq + ((size_t) itemsize * ((size_t) idx));
            int is_pad = memcmp(
                last, padvec, (size_t) itemsize * sizeof(PSFloat)
            ) == 0;
            if (!is_pad) break;
        }
    }
    if (pad_vector == NULL) free(padvec);
    return idx;
}

static int sequenceHasEndItem(PSFloat *seq, long len, long itemsize,
                              PSSequenceSettings *sequence_settings,
                              PSFloat **end_vector, long *pad_idx,
                              PSFloat **pad_vector, int *err)
{
    if (err != NULL) *err = 0;
    if (pad_idx != NULL) *pad_idx = -1;
    if (itemsize <= 0 || len <= 0) return 0;
    long end = sequence_settings->end;
    if (end < 0) return 0;
    size_t idx = (size_t) len - 1;
    if (sequence_settings->pad >= 0) {
        long non_pad_idx = sequenceFindLastNonPadItem(
            seq, len, itemsize, sequence_settings->pad, pad_vector, err
        );
        if (non_pad_idx < 0) return 0;
        idx = (size_t) non_pad_idx;
        if (pad_idx != NULL) {
            long next_idx = non_pad_idx + 1;
            if (next_idx<len && (long)seq[next_idx] == sequence_settings->pad)
                *pad_idx = next_idx;
        }
    }
    if (itemsize == 1) return (long) seq[idx] == end;
    else {
        PSFloat *onehot_vec = PSOneHotVector(end, itemsize);
        if (onehot_vec == NULL) {
            if (err != NULL) *err = 1;
            return 0;
        }
        if (end_vector != NULL) *end_vector = onehot_vec;
        PSFloat *last = seq + ((size_t) itemsize * idx);
        int is_end = memcmp(
            last, onehot_vec, (size_t) itemsize * sizeof(PSFloat)
        ) == 0;
        if (end_vector == NULL) free(onehot_vec);
        return is_end;
    }
}

/* Prepare sequence depending on the value of `mode`. If flag
 * `SEQ_MODE_PREPEND_START` is set in `mode`, prepend the starting element
 * defined by `sequence_settings` if needed (sequence is missing it),
 * otherwise, remove it from the sequence if found and the flag is not set.
 * Do the same for `SEQ_MODE_APPEND_END` and the ending element defined by
 * `sequence_settings`.
 * If `SEQ_MODE_PREPEND_SEQLEN` flag is set in `mode`, also prepend prepend
 * the sequence length to the sequence itself.
 * If the pointer `new_len` is noy NULL, it will be used to store the length
 * of the resulting sequence, excluding the first element of the resulting
 * vector if it's used to store the sequence length itself (ie. if
 * `SEQ_MODE_PREPEND_SEQLEN` has been set).
 * The function allocates a new sequence on success. If something goes wrong,
 * it returns NULL. */
static PSFloat *prepareSequence(PSFloat *seq, long len, long min_len,
                                long itemsize, long *new_len,
                                PSSequenceSettings *sequence_settings,
                                int mode)
{
    PSFloat *new_seq = NULL, *end_vec = NULL, *pad_vec = NULL;
    if (new_len != NULL) *new_len = 0;
    long final_len = len, datalen = len, pad_len = 0, pad_idx = -1;
    int err = 0;
    if (len <= 0 || itemsize <= 0) goto final;
    int has_start = sequenceHasStartItem(seq, itemsize, sequence_settings);
    int has_end = sequenceHasEndItem(
        seq, len, itemsize, sequence_settings, &end_vec, &pad_idx,
        &pad_vec, &err
    );
    if (datalen <= 0) {
        PSErr(NULL, "invalid sequence");
        goto final;
    }
    if (err) goto final;
    long offset = 0;
    int prepend_start = mode & SEQ_MODE_PREPEND_START,
        append_end = mode & SEQ_MODE_APPEND_END,
        prepend_seqlen = mode & SEQ_MODE_PREPEND_SEQLEN;
    int has_pad = (pad_idx >= 0 && pad_idx < len);
    if (prepend_start && !has_start) final_len++;
    else if (!prepend_start && has_start) {
        final_len--;
        datalen--;
        offset = 1;
    }
    if (append_end && !has_end) {
        if (!has_pad) final_len++;
    } else if (!append_end && has_end) {
        final_len--;
        datalen--;
    }
    if (final_len < min_len && append_end && has_end) {
        pad_len = min_len - final_len;
        final_len = min_len;
    }
    if ((err = final_len <= 0)) {
        PSErr(NULL, "invalid sequence");
        goto final;
    }
    size_t seqsize = (size_t) itemsize * (size_t) final_len * sizeof(PSFloat);
    if (prepend_seqlen) seqsize += sizeof(PSFloat);
    new_seq = malloc(seqsize);
    if (new_seq == NULL) {
        PSPrintMemoryErrorMsg();
        goto final;
    }
    PSFloat *dest = new_seq;
    if (prepend_seqlen) *(dest++) = (PSFloat) final_len;
    if (prepend_start && !has_start) {
        PSFloat *start = sequence_settings->start, *tmpstart = NULL;
        if (start == NULL) {
            tmpstart = PSVectorZero((size_t) itemsize);
            if (tmpstart == NULL) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            start = tmpstart;
        }
        if (itemsize == 1) *(dest++) = *start;
        else {
            PSVectorCopy(dest, start, itemsize);
            dest += itemsize;
        }
        free(tmpstart);
    }
    PSFloat *first = dest;
    if (datalen == 1) *(dest++) = *(seq + (offset * itemsize));
    else {
        PSVectorCopy(dest, seq + (offset * itemsize), itemsize * datalen);
        dest += (itemsize * datalen);
    }
    if (append_end && (!has_end || pad_len > 0)) {
        long end = sequence_settings->end, pad = sequence_settings->pad;
        if (end < 0) end = 0;
        long item = end;
        PSFloat *item_vec = end_vec;
        if (has_pad) {
            if (itemsize == 1) first[pad_idx] = (PSFloat) end;
            else {
                err = (pad_vec == NULL);
                if (err) goto final;
                PSVectorCopy(first + (pad_idx * itemsize), end_vec, itemsize);
            }
            has_end = 1;
            dest += itemsize;
            pad_len--;
        }
        while (!has_end || pad_len > 0) {
            if (has_end && has_pad) {
                item = pad;
                item_vec = pad_vec;
            }
            if (itemsize == 1) *(dest++) = (PSFloat) item;
            else {
                err = (item_vec == NULL);
                if (err) goto final;
                PSVectorCopy(dest, item_vec, itemsize);
                dest += itemsize;
            }
            has_end = 1;
            pad_len--;
        }
    }
final:
    free(end_vec);
    if (err) {
        free(new_seq);
        new_seq = NULL;
    } else if (new_len != NULL) *new_len = final_len;
    return new_seq;
}

/**** Forward Functions ****/

int checkLayerForForward(PSLayer *layer) {
    if (layer == NULL) return 0;
    int trainable = !(layer->flags & PS_FLAG_NON_TRAINABLE);
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
    if (layer->weight_types > 0 && trainable) {
        if (layer->weights == NULL) {
            PSErr(NULL, "Layer[%d]: layer has no weights", layer->index);
            return 0;
        }
        int trainable_params = 0xFFFF;
        if (layer->type == Attention)
            trainable_params = PSGetAttentionEnabledProjections(layer);
        for (int i = 0; i < layer->weight_types; i++) {
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
        opts->debugStep = dumpForwardStep;
    }
}

/* Onehot input layers only have one input corresponding to the index
 * of the activated unit. In this case, just take the value of the
 * corresponding weight, since the input should always be considered
 * as it would be 1 */
int PSOnehotInputsForward(PSLayer *layer, int weights_index,
                          PSFloat *outputs, long t, int apply_biases,
                          int do_activate)
{
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d] has no previous layer", layer->index);
        return 0;
    }
    long onehot_vector_size = PSGetOneHotLayerVectorSize(previous);
    if (weights_index < 0) weights_index = 0;
    if (weights_index >= layer->weight_types) {
        PSErr(NULL, "Layer[%d] invalid weights index %d (max: %d)",
              weights_index,  layer->weight_types - 1);
        return 0;
    }
    PSMatrix weights = (
        layer->weights != NULL ? layer->weights[weights_index] : NULL
    );
    if (weights == NULL) {
        PSErr(NULL, "Layer[%d] has no weights[%d]",layer->index,weights_index);
        return 0;
    }
    int use_bias = apply_biases && !(layer->flags & PS_FLAG_NO_BIAS);
    do_activate = do_activate && layer->activate != NULL;
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    long seqlen = 1;
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
    for (long s = 0; s < seqlen; s++) {
        long tidx = s + t;
        long onehot_idx = (long) PSGetState(previous, 0, tidx);
        if (onehot_idx >= onehot_vector_size) {
            PSErr(
                NULL, "Layer[%d]: onehot index %d is out of range "
                "(vector size = %d)", previous->index, onehot_idx,
                onehot_vector_size
            );
            return 0;
        }
        if (outputs == NULL) out = PSLayerStates(layer, tidx);
        else out = outputs + (t * layer->size);
        if (out == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs at %d", layer->index, tidx);
            return 0;
        }
        PSFloat *states = tweights + (onehot_idx * layer->size);
        PSVectorCopy(out, states, layer->size);
        if (use_bias)
            PSAddVectors(out, layer->biases, out, layer->size, &opts);
        if (do_activate)
            layer->activate(out, NULL, layer->size, opts.acceleration);
    }
    return 1;
no_transposition:
    for (long s = 0; s < seqlen; s++) {
        long tidx = s + t;
        long onehot_idx = (long) PSGetState(previous, 0, tidx);
        if (onehot_idx >= onehot_vector_size) {
            PSErr(
                NULL, "Layer[%d]: onehot index %ld is out of range "
                "(vector size = %ld)", previous->index, onehot_idx,
                onehot_vector_size
            );
            return 0;
        }
        PSFloat *outputs = PSLayerStates(layer, tidx);
        if (outputs == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs", layer->index);
            return 0;
        }
        for (long i = 0; i < layer->size; i++) {
            PSFloat w = weights[(i * layer->size) + onehot_idx];
            outputs[i] = w;
            if (use_bias) outputs[i] += layer->biases[i];
        }
        if (do_activate)
            layer->activate(outputs, NULL, layer->size, opts.acceleration);
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
        use_bias = !(layer->flags & PS_FLAG_NO_BIAS);
    long seqlen = 0, t = 0;
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    if (previous->flags & PS_FLAG_ONEHOT) {
        /* Onehot inputs */
        if (!PSOnehotInputsForward(layer, 0, NULL, t, 1, 1)) return 0;
        goto final;
    }
    PSMatrix weights = layer->weights[0];
    opts.acceleration = model->acceleration;
    if (!handles_seq) {
        PSFloat *inputs = PSLayerStates(previous, t),
                *outputs = PSLayerStates(layer, t);
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
        if (layer->activate)
            layer->activate(outputs, NULL, layer->size, opts.acceleration);
    } else {
        opts.transpose = 2;
        if (use_bias) {
            long seqlen = PSMatrixDim(previous->states, 0), i;
            for (i = 0; i < seqlen; i++) {
                PSFloat *outputs = PSLayerStates(layer, i);
                PSVectorCopy(outputs, layer->biases, layer->size);
            }
            opts.store_mode = PS_STORE_MODE_ADD;
        }
        int ok = PSDot(previous->states, weights, layer->states, &opts);
        if (!ok) return 0;
        opts.transpose = 0;
        opts.store_mode = PS_STORE_MODE_SET;
        if (layer->activate) {
            layer->activate(
                layer->states, NULL, seqlen * layer->size, opts.acceleration
            );
        }
    }
final:
    return 1;
}

static int softmaxForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSMathOpts opts = {0};
    handleLayerForwardDebug(layer, __func__, &opts);
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer);
    long seqlen = 0, t = 0;
    if (is_recurrent || handles_seq) {
        PSReadSequenceArgs(layer, is_recurrent, seqlen, t);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    if (!PSFullForward(layer, seqlen, t)) return 0;
    if (seqlen < 1 || !handles_seq) seqlen = 1;
    for (long i = 0; i < seqlen; i++) {
        PSFloat *outputs = PSLayerStates(layer, i + t);
        if (outputs == NULL) {
            PSErr(NULL, "Layer[%d]: missing outputs[%d]", layer->index, i + t);
            return 0;
        }
        PSSoftmax(outputs, outputs, layer->size, opts.acceleration);
    }
    return 1;
}

/* Utils */

static PSFloat norm(PSFloat* vector, long size) {
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

static void shuffle(PSFloat *array, long size, long example_size) {
    srand(time(NULL));
    size_t byte_size = (size_t) example_size * sizeof(PSFloat);
    for (long i = size - 1; i > 0; i--) {
        long j = rand() % (i + 1);
        /* printf("Shuffle cycle %d: random is %d\n", i, j); */
        PSFloat tmp_a[example_size];
        PSFloat tmp_b[example_size];
        long idx_a = i * example_size;
        long idx_b = j * example_size;
        /* printf("-> idx_a: %d\n", idx_a); */
        /* printf("-> idx_b: %d\n", idx_b); */
        memcpy(tmp_a, array + idx_a, byte_size);
        memcpy(tmp_b, array + idx_b, byte_size);
        memcpy(array + idx_a, tmp_b, byte_size);
        memcpy(array + idx_b, tmp_a, byte_size);
    }
}

static void shuffleSequences(PSFloat **sequences, long size) {
    srand(time(NULL));
    for (long i = size - 1; i > 0; i--) {
        long j = rand() % (i+1);
        /* printf("Shuffle cycle %d: random is %d\n", i, j); */
        PSFloat *tmp_a = sequences[i];
        PSFloat *tmp_b = sequences[j];
        sequences[i] = tmp_b;
        sequences[j] = tmp_a;
    }
}

static long parseSequenceData(PSModel *model, PSFloat *data,
                              long sequence_index, int flags, int has_targets,
                              long *x_seqlen, PSFloat **x,
                              long *y_seqlen, PSFloat **y)
{
    long datalen = 0, xlen = 0, ylen = 0;
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
    long input_size = input_layer->size, output_size = output_layer->size;
    if (output_layer->flags & PS_FLAG_ONEHOT) output_size = 1;
    int autoregression = output_model->flags & PS_FLAG_AUTOREGRESSION;
    int seq2seq = has_targets && (flags & PS_TRAINING_FLAG_SEQ2SEQ);
    if (seq2seq && !autoregression) return 0;
    else if (!seq2seq && autoregression) seq2seq = 1;
    if (!input_seq && seq2seq) return 0;
    int seqlen_nelems = (seq2seq ? 2 : 1),
        selfsupervised = flags & PS_TRAINING_FLAG_SELFSUPERVISED;
    PSFloat *xsize_p = NULL, *ysize_p = NULL, *xp = NULL, *yp = NULL;
    if (input_seq) {
        xsize_p = data;
        xlen = (long) *xsize_p;
        xp = xsize_p + 1;
    } else {
        xlen = 1;
        xp = data;
    }
    if (xsize_p != NULL && xlen <= 0) {
        PSErr(
            NULL, "Sequence[%d] Invalid length %d at data offset %d",
            sequence_index, xlen, (long) (xsize_p - data)
        );
        return 0;
    }
    if (x_seqlen != NULL) *x_seqlen = xlen;
    if (x != NULL) *x = xp;
    long xdatalen = (xlen * input_size), ydatalen = 0;
    if (has_targets) {
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
            ylen = (long) *ysize_p;
            if (ylen <= 0) {
                PSErr(
                    NULL, "Sequence[%d] Invalid length %d at data offset %ld",
                    sequence_index, ylen, (long) (ysize_p - data)
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
                                     long sequence_count, int flags)
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
        long datalen = parseSequenceData(
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

PSFloat *PSGetInputsFromTrainingData(PSFloat *training_data, long data_size,
                                     long num_examples, long input_size,
                                     long label_size, int recurrent_input,
                                     int recurrent_output,
                                     long *count, long *result_size)
{
    if (training_data == NULL || data_size <= 0) return NULL;
    long example_size = input_size + label_size;
    int recurrent = (recurrent_input || recurrent_output);
    if (num_examples <= 0) {
        /* Auto-detect number of examples. */
        if (recurrent) {
            /* First training data element for sequence datasets must
             * indicate the number fo sequences in the dataset itself. */
            num_examples = (long) *(training_data++);
            data_size--;
        } else num_examples = data_size / example_size;
    }
    if (num_examples <= 0) return NULL;
    if (count != NULL) *count = num_examples;
    size_t size = num_examples * input_size * sizeof(PSFloat);
    if (result_size != NULL) *result_size = size;
    PSFloat *inputs = malloc(size);
    if (inputs == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSFloat *data_p = training_data, *inputs_p = inputs;
    int nwritten = 0, tot_recurrent_examples = 0;
    while (num_examples > 0) {
        if (!recurrent) {
            memcpy(inputs_p, data_p, input_size * sizeof(PSFloat));
            data_p += input_size;
            inputs_p += input_size;
        } else {
            long seqlen = *(data_p++);
            long input_len =
                (recurrent_input ? seqlen * input_size : input_size);
            long label_len =
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
            tot_recurrent_examples += input_len;
        }
        num_examples--;
    }
    if (recurrent && count != NULL) *count = tot_recurrent_examples;
    return inputs;
fail:
    free(inputs);
    return NULL;
}

static long arrayMaxIndex(PSFloat *array, long len) {
    long max_idx = 0, i;
    PSFloat max = 0;
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
int PSFindLayerMaxState(PSLayer *layer, PSFloat *max_p, long *index_p, ...)
{
    if (max_p == NULL && index_p == NULL) return 0;
    long seqidx = 0, i;
    int uses_sequences = PSUseSequences(layer);
    if (uses_sequences) {
        va_list args;
        va_start(args, index_p);
        seqidx = va_arg(args, long);
        va_end(args);
    }
    PSFloat max = PSFLOAT_MIN;
    long max_idx = -1;
    PSFloat *states = PSLayerStates(layer, seqidx);
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

static long fetchSequenceOutputState(PSLayer *out, PSFloat *outputs,
                                     long i, int onehot)
{
    long t = (onehot ? i : i / out->size), j;
    long max_idx = 0, oidx = 0;
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
        case RNNLayer:
            return "RNN Layer";
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
    case PS_STATUS_UNTRAINED:
        return "untrained";
    case PS_STATUS_TRAINING:
        return "training";
    case PS_STATUS_TRAINED:
        return "trained";
    case PS_STATUS_ERROR:
        return "error";
    case PS_STATUS_PAUSED:
        return "paused";
    case PS_STATUS_ABORTED:
        return "aborted";
    case PS_STATUS_VALIDATING:
        return "validating";
    }
    return "UNKOWN";
}

char *getOptimizationName(PSOptimization optimization) {
    if (optimization == PSSGDOptimization) return "SGD";
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

long PSGetLayerParametersCount(PSLayer *layer, int param_type) {
    if (layer == NULL) return 0;
    if (Pooling == layer->type || layer->index == 0 || layer->size == 0 ||
        Dropout == layer->type) return 0;
    if (param_type == 0) param_type = (PS_PARAM_WEIGHT | PS_PARAM_BIAS);
    if (layer->getParamCount != NULL)
        return layer->getParamCount(layer, param_type);
    long count = 0;
    if (param_type & PS_PARAM_WEIGHT && layer->weights != NULL) {
        for (int i = 0; i < layer->weight_types; i++) {
            PSMatrix weights = layer->weights[i];
            if (weights != NULL) count += PSMatrixLength(weights);
        }
    }
    if (param_type & PS_PARAM_BIAS) {
        if (!(layer->flags & PS_FLAG_NO_BIAS) && layer->biases != NULL) {
            long bias_count;
            if (Convolutional == layer->type) bias_count = layer->output_depth;
            else if (LSTM == layer->type) bias_count = layer->size * 4;
            else if (GRU == layer->type) bias_count = layer->size * 3;
            else if (Attention == layer->type)
                bias_count = (PS_SCORES_PROJ_IDX * layer->size) + 1;
            else bias_count = layer->size;
            count += bias_count;
        }
    }
    return count;
}

long PSGeModelParametersCount(PSModel *model) {
    long tot = 0;
    int param_type = (PS_PARAM_BIAS | PS_PARAM_WEIGHT), i;
    if (model->layers == NULL || model->size == 0) return 0;
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        tot += PSGetLayerParametersCount(layer, param_type);
    }
    return tot;
}

/* Get the onehot vector size of `layer`. If the flag `PS_FLAG_ONEHOT` is not
 * set into layer's flags, just return the layer size. */
long PSGetOneHotLayerVectorSize(PSLayer *layer) {
    if (!(layer->flags & PS_FLAG_ONEHOT)) return layer->size;
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

/* Return the output layer (basically, the last layer) of `model`. If `model`
 * is part of a multi-model chain, the function will return the output layer of
 * the output model (the last model) of the chain.
 * Return value: the output layer or NULL if:
 *  - `model` is NULL or `model` has no layers.
 *  - `model` is part of a multi-model chain, but the chain is broken. */
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

/* Get the layer at index `layer_index` in `model`. If `model` is part of a
 * multi-model chain, the argument `model_index` can be used to specify the
 * index of the sub-model of the model chain.
 * Both `layer_index` and `model_index` accept negative values: in this case,
 * the index is calculated from the last layer/model, for example:
 * if `layer_index` is -1 and `model` has 5 layers, the actual index will be
 * the last layer's index (4).
 * Return value: the layer at specified index/indices or NULL if:
 *  - `model` is NULL.
 *  - `model` is a multi-model chain but it's broken/invalid.
 *  - `layer_index` is out-of-bounds.
 *  - `model_index` is out-of-bounds. */
PSLayer *PSGetLayerByIndex(PSModel *model, int layer_index, int model_index) {
    if (model == NULL) return NULL;
    if (PSIsModelChain(model)) {
        int num_models = PSModelChainLength(model);
        model = PSModelChainHead(model);
        if (model == NULL || num_models <= 0) {
            PSErrNN(__func__, model, NULL, "broken model chain");
            return NULL;
        }
        if (model_index < 0) model_index = num_models + model_index;
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

/* Determine the input size of `layer`, depending on the size of its
 * previous layer, if any.
 * If previous layer has set the flag `PS_FLAG_ONEHOT`, the function will
 * determine the input size by previous layer's onehot vector size
 * (`PSGetOneHotLayerVectorSize`).
 * Return value: the input size of `layer` or zero if:
 *  - `layer` is NULL.
 *  - `layer` has no previous layer. */
long PSGetLayerInputSize(PSLayer *layer) {
    if (layer == NULL) return 0;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) return 0;
    if (previous->flags & PS_FLAG_ONEHOT)
        return PSGetOneHotLayerVectorSize(previous);
    return previous->size;
}

long PSGetLayerInputWeightsCount(PSLayer *layer, int per_neuron) {
    if (layer == NULL) return 0;
    if (layer->weights == NULL) return 0;
    if (layer->weights[0] == NULL) return 0;
    if (layer->type == Pooling || layer->type == Dropout) return 0;
    long wcount = PSMatrixLength(layer->weights[0]);
    if (per_neuron && layer->type != Convolutional) wcount /= layer->size;
    return wcount;
}

PSFloat *PSGetNeuronInputWeights(PSNeuron *neuron) {
    if (neuron->layer == NULL) return NULL;
    if (neuron->layer->type == Pooling || neuron->layer->type == Dropout)
        return NULL;
    if (neuron->layer->weights == NULL || neuron->layer->weights[0] == NULL)
        return NULL;
    long wcount = PSGetLayerInputWeightsCount(neuron->layer, 1);
    long widx = neuron->index * wcount;
    if (widx >= (wcount * neuron->layer->size)) {
        PSErr(__func__, "Layer[%d] Neuron[%d] index is out of bounds",
              neuron->layer->index, neuron->index);
        return NULL;
    }
    return neuron->layer->weights[0] + widx;
}

void PSPrintLayerInfo(PSLayer *layer) {
    static int min_indent = 0;
    if (layer == NULL) return;
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    PSLayer *linked_to = NULL;
    PSModelLink *link = NULL;
    if (layer->model != NULL) {
        link = layer->model->previous_model_link;
        if (link != NULL && link->layer == layer)
            linked_to = link->previous_layer;
        else if (layer->model->next != NULL) {
            link = layer->model->next->previous_model_link;
            if (link != NULL && link->previous_layer == layer)
                linked_to = link->layer;
        }
    }
    int color_enabled = PSLogColorEnabled();
    char onehot_info[50];
    onehot_info[0] = 0;
    int onehot_input = (layer->index == 0 && layer->flags & PS_FLAG_ONEHOT);
    if (onehot_input)
        sprintf(onehot_info, "(vector size: %ld)", layer->onehot_vector_size);
    if (min_indent == 0) min_indent = strlen("  Layer[]: ");
    int indent = min_indent + (int) PSCalcIntStringLength(layer->index);
    printf("Layer[%d]: %s, size = %ld", layer->index, type_name, layer->size);
    if (Dropout == layer->type) printf(", dropout = %g", PSGetDropout(layer));
    if (onehot_info[0]) printf(" %s", onehot_info);
    if (ltype == Convolutional || ltype == Pooling) {
        PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
        long output_w = layer->output_columns, output_h = layer->output_rows,
             depth = layer->output_depth, filter_w = 0, filter_h = 0,
             filter_d = 0, input_w = 0, input_h = 0;
        int stride = 0, padding = 0;
        if (settings != NULL) {
            input_w = settings->input_width;
            input_h = settings->input_height;
            filter_w = settings->filter_width;
            filter_h = settings->filter_height;
            filter_d = settings->filter_depth;
            stride = settings->stride;
            padding = settings->padding;
        }
        if (stride <= 0 && ltype == Pooling) stride = (int) filter_w;
        printf(
            "\n%*cinput size = %ldx%ld, output_size = %ldx%ld, depth = %ld",
            indent, ' ', input_w, input_h, output_w, output_h, depth
        );
        if (ltype == Convolutional) {
            printf(
                "\n%*cfilter = %ldx%ldx%ld, stride = %d",
                indent, ' ', filter_w, filter_h, filter_d, stride
            );
        } else {
            printf(
                "\n%*cfilter = %ldx%ld, stride = %d",
                indent, ' ', filter_w, filter_h, stride
            );
        }
        if (ltype == Convolutional) {
            if (padding < 0) padding = 0;
            printf(", padding = %d", padding);
        }
    } else if (ltype == FullyConnected && layer->output_depth > 1) {
       printf(", depth = %ld", layer->output_depth);
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
        long vocab_size = PSGetEmbeddingVocabularySize(layer);
        if (vocab_size > 0) printf(", vocabulary_size = %ld", vocab_size);
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
    if (linked_to != NULL && linked_to->model != NULL) {
        char *color = "", *end_color = "";
        if (color_enabled) {
            color = PSCOLOR_LIGHT_GREEN;
            end_color = PSCOLOR_RESET;
        }
        int is_prev = (linked_to->model->index < layer->model->index);
        printf(
            "\n%*cLinked to %slayer[%d]%s of %s model", indent, ' ',
            color, linked_to->index, end_color,
            (is_prev ? "previous" : "next")
        );
    }
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
    char *loss_name = NULL;
    if (model->next == NULL)
        loss_name = getLossFunctionName(model->loss);
    if (loss_name != NULL) printInfoRow("Loss Function", "%s", loss_name);
    printInfoRow("Status", "%s", getModelStatusLabel(model));
    printInfoRow("AVX", "%s",
                 (PSAVXEnabled(model->acceleration) ? "yes" : "no"));
    printInfoRow("Apple(R) Accelerate Framework", "%s",
                 (PSAccelerateEnabled(model->acceleration) ? "yes" : "no"));
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

/**** Loss Functions ****/

PSFloat PSQuadraticLoss(PSFloat *outputs, PSFloat *expected, long size,
                        long onehot_size)
{
    PSFloat *_diffs;
    PSFloat diffs[size];
    if (!onehot_size) {
        long i;
        for (i = 0; i < size; i++) {
            PSFloat d = outputs[i] - expected[i];
            diffs[i] = d;
            if (isnan(d)) {
                fprintf(stderr,
                    "\n\nPSQuadraticLoss: diffs[%ld] is nan!\n"
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

PSFloat PSCrossEntropyLoss(PSFloat *outputs, PSFloat *expected, long size,
                           long onehot_size)
{
    PSFloat loss = 0.0;
    long i;
    for (i = 0; i < size; i++) {
        PSFloat o = outputs[i];
        if (onehot_size) loss += (PSLog(o + PSFLOAT_EPS));
        else {
            PSFloat y = expected[i];
            loss += (
                y * PSLog(o + PSFLOAT_EPS) +
                (1 - y) * PSLog((1 - o) + PSFLOAT_EPS)
            );
        }
    }
    loss *= -1;
    return loss;
}

/**** Neural Network Functions ****/

/* Set `status` as the status of `model`. The optional argument `old` can be
 * used to retrieve the old status of `model` before updating it with the
 * value of `status`.
 * If `model` is part of a multi-model chain, hte new status will be set on
 * all the models that are part of the model chain.
 * Common used status values are:
 *  - `PS_STATUS_UNTRAINED`
 *  - `PS_STATUS_TRAINED`
 *  - `PS_STATUS_TRAINING`
 *  - `PS_STATUS_VALIDATING`
 *  - `PS_STATUS_PAUSED`
 *  - `PS_STATUS_ABORTED`
 *  - `PS_STATUS_ERROR`
 */
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

/* Get the value of `status` of `model`. If `model` is part of a multi-model
 * chain, the function will retrieve the status of the first model of the
 * chain.
 * Common status values are:
 *  - `PS_STATUS_UNTRAINED`
 *  - `PS_STATUS_TRAINED`
 *  - `PS_STATUS_TRAINING`
 *  - `PS_STATUS_VALIDATING`
 *  - `PS_STATUS_PAUSED`
 *  - `PS_STATUS_ABORTED`
 *  - `PS_STATUS_ERROR`
 * Return value: the status of `model` or 0 if `model` is NULL.
 */
int PSModelGetStatus(PSModel *model) {
    if (model == NULL) return 0;
    if (PSIsModelChain(model)) model = PSModelChainHead(model);
    if (model == NULL) return 0;
    return model->status;
}

/* Set the name of `model` with the string provided with the argument `name`.
 * If `name` is NULL and `model` already has a name, model's `name` will be
 * cleared.
 * NOTE: the model will duplicate the provided `name` and it will keep it
 * inside its internal data. The duplicated string will be automatically freed
 * by freeing the whole model (`PSModelFree`). If `model` already has a name,
 * the original name will be automatically freed.
 * Return value: 1 is name is successfully set or 0 if:
 *  - `model` is NULL.
 *  - memory cannot be allocated. */
int PSModelSetName(PSModel *model, char *name) {
    if (model == NULL) return 0;
    if (model->name != NULL) {
        PSModelContext *ctx = getModelContext(model);
        if (ctx == NULL) return 0;
        if (model->name == ctx->allocated_name)
            free((void *) ctx->allocated_name);
    }
    if (name == NULL) {
        model->name = NULL;
        return 1;
    }
    const char *new_name = strdup(name);
    if (new_name == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    model->name = new_name;
    setModelContext(model, allocated_name, new_name);
    return 1;
}

long PSStateSequenceLength(PSLayer *layer) {
    if (layer == NULL || layer->states == NULL) return 0;
    long len = PSMatrixDim(layer->states, 0);
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
PSMatrix initLayerStates(PSLayer *layer, long seqlen,
                         int retain_previous,
                         PSMatrix current, PSFloat **previous)
{
    assert(seqlen >= 0);
    /* If `retain_previous` is true, it means that the last vector in
     * current sequence states must be retained as 'initial state'.
     * This is done by adding one further row to states matrix that will
     * hold the 'initial' vector from previous last vector. */
    long cur_seqlen =  0;
    if (current == layer->states)
        cur_seqlen = PSStateSequenceLength(layer);
    else if (current != NULL) {
        cur_seqlen = PSMatrixDim(current, 0);
        if (previous && *previous != NULL) cur_seqlen--;
    }
    if (cur_seqlen <= 0) retain_previous = 0;
    long nrows = seqlen + (retain_previous ? 1 : 0);
    PSMatrix states = PSMatrixZeros(2, nrows, layer->size);
    if (states == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    if (retain_previous && current != NULL) {
        long last_idx = cur_seqlen - 1;
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
PSMatrix resizeLayerStates(PSLayer *layer, long seqlen,
                           PSMatrix current, PSFloat **previous)
{
    long nrows = seqlen, cur_seqlen, cur_nrows;
    if (current == layer->states) cur_seqlen = PSStateSequenceLength(layer);
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
    long steps2add = nrows - cur_nrows;
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
int PSInitLayerStates(PSLayer *layer, long seqlen, int retain_previous) {
    assert(seqlen >= 0);
    PSMatrix states = layer->states, hstates = NULL;
    long cur_seqlen = PSStateSequenceLength(layer);
    if (cur_seqlen <= 0 || states == NULL) retain_previous = 0;
    if (seqlen == 0 && !retain_previous) {
        layer->states = NULL;
        PSMatrixFree(states);
        layer->initial_states = NULL;
        if (layer->onStatesInit != NULL)
            if (!layer->onStatesInit(layer, 0, 0)) goto err;
        return 1;
    }
    hstates = initLayerStates(
        layer, seqlen, retain_previous, states,
        &layer->initial_states
    );
    if (hstates == NULL) return 0;
    layer->states = hstates;
    PSMatrixFree(states);
    hstates = NULL;
    if (layer->onStatesInit != NULL) {
        if (!layer->onStatesInit(layer, seqlen, retain_previous))
            goto err;
    }
    return 1;
err:
    if (hstates != NULL) PSMatrixFree(hstates);
    if (layer->states != NULL) PSMatrixFree(layer->states);
    layer->states = NULL;
    PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
    return 0;
}

/* Resize states sequence for layer `layer`. The function also calls
 * `onStatesResize` callback if any, allowing different types of layer to
 * resize their own private data.
 * Arguments:
 * `seqlen`: new states sequence length.
 * Return value: 1 in case of success, 0 in case of failure. */
int PSResizeLayerStates(PSLayer *layer, long seqlen) {
    assert(seqlen >= 0);
    PSMatrix states = layer->states;
    long cur_seqlen = PSStateSequenceLength(layer);
    if (states == NULL || seqlen == 0 || cur_seqlen == 0)
        return PSInitLayerStates(layer, seqlen, 1);
    else if (seqlen == cur_seqlen) return 1;
    else if (seqlen < cur_seqlen) {
        PSErr(
            NULL, "Layer[%d]: Cannot resize hidden state to a smaller size: "
            "%ld (current sequence length = %ld)",
            layer->index, seqlen, cur_seqlen
        );
        return 0;
    }
    PSMatrix hstates = resizeLayerStates(
        layer, seqlen, states, &layer->initial_states
    );
    if (hstates == NULL) {
        PSMatrixFree(states);
        layer->states = NULL;
        layer->initial_states = NULL;
        PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
        return 0;
    }
    layer->states = hstates;
    if (layer->onStatesResize != NULL)
        if (!layer->onStatesResize(layer, seqlen, cur_seqlen)) return 0;
    return 1;
}

int PSResetLayerStateSequence(PSLayer *layer, long seqlen,
                              int retain_previous)
{
    if (layer == NULL) return 0;
    if (!PSUseSequences(layer)) return 0;
    return PSInitLayerStates(layer, seqlen, retain_previous);
}

int PSResetModelStateSequences(PSModel *model, long seqlen,
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

int PSBeforeSequenceForward(PSLayer *layer, long seqlen, long t) {
    if (!PSUseSequences(layer)) return 1;
    if (seqlen < 1) {
        PSErr(__func__, "Layer[%d]: sequence length must be >= 1 (found %ld)",
              layer->index, seqlen);
        return 0;
    }
    long cur_seqlen = PSStateSequenceLength(layer);
    if (PSIsRecurrent(layer) && t >= cur_seqlen) {
        /* Recurrent layers may need to resize their sequence steps before
         * forward phase if step `t` is beyond current sequence length. */
        if (!PSResizeLayerStates(layer, t + 1)) {
            if (layer->model != NULL)
                PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
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

int PSSetState(PSLayer *layer, PSFloat state, long index, ...) {
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
    long t = 0;
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, index);
        t = va_arg(args, long);
        va_end(args);
        long seqlen = PSStateSequenceLength(layer);
        if (t >= seqlen) {
            if (!PSResizeLayerStates(layer, t + 1)) {
                if (layer->model)
                    PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
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
        long t = va_arg(args, long);
        va_end(args);
        return PSSetState(neuron->layer, a, neuron->index, t);
    }
    return PSSetState(neuron->layer, a, neuron->index);
}

PSFloat PSGetState(PSLayer *layer, long index, ...) {
    if (layer->states == NULL) return 0.0;
    if (index >= layer->size) {
        PSErr(
            __func__, "Neuron index %ld is out-of-range for layer %ld "
            "of size %ld", index, layer->index, layer->size
        );
        return 0.0;
    }
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, index);
        long t = va_arg(args, long);
        va_end(args);
        /* If t < 0, retrieve previous state, if any. */
        if (t < 0) {
            if (layer->initial_states == NULL) return 0.0;
            else return layer->initial_states[index];
        } else {
            long seqlen = PSStateSequenceLength(layer);
            if (t >= seqlen) {
                PSErr(
                    __func__, "Index %ld is out-of-range: layer %ld states "
                    "sequence has size: %ld", t, layer->index, seqlen
                );
                return 0.0;
            }
            index = (t * layer->size) + index;
        }
    }
    return layer->states[index];
}

/* Grt the states (unit activation values) of `layer`. If `layer` uses
 * sequences (ie. it's recurrent or if has flag `PS_FLAG_USE_SEQUENCES`),
 * the function will also read the first variadic argument after `layer`
 * that indicates the index of the states to retrieve inside the sequence.
 * If the sequence index is negative, the function will retrieve the
 * `initial_states` of the layer, that are the initial values of the
 * layer states before it has received the input sequence (the can be NULL).
 * Return value: the states of `layer` or NULL if:
 *  - `layer` is NULL.
 *  - `layer->states` is NULL.
 *  - the index provided with the variadic argument is out-of-bounds (ie.
 *    it's equal or greater than the current sequence length).
 *  - the index provided with the variadic argument is negative but the
 *    layer has no `initial_states`. */
PSFloat *PSLayerStates(PSLayer *layer, ...) {
    if (layer == NULL) return NULL;
    if (layer->states == NULL) return NULL;
    if (PSUseSequences(layer)) {
        va_list args;
        va_start(args, layer);
        long t = va_arg(args, long);
        va_end(args);
        /* If t < 0, retrieve previous state, if any. */
        if (t < 0) return layer->initial_states;
        else {
            long seqlen = PSStateSequenceLength(layer);
            if (t >= seqlen) {
                PSErr(
                    __func__, "Index %ld is out-of-range: layer %d states "
                    "sequence has size: %ld", t, layer->index, seqlen
                );
                return NULL;
            }
            return layer->states + (t * layer->size);
        }
    }
    return layer->states;
}

/* Get the output values of `layer`. If layer uses sequences (ie. it's
 * recurrent or if has flag `PS_FLAG_USE_SEQUENCES`), the function will pick
 * the last states of the sequence, otherwise it will just return the
 * layer `states`.
 * Return value: the output values of `layer` or NULL if:
 *  - `layer` is NULL.
 *  - `layer->states` is NULL. */
PSFloat *PSLayerOutputs(PSLayer *layer) {
    if (layer == NULL) return NULL;
    if (PSUseSequences(layer)) {
        long seqlen = PSStateSequenceLength(layer);
        return PSLayerStates(layer, seqlen - 1);
    }
    return PSLayerStates(layer, 0);
}

/* Get the output values of the output (last) layer of `model`. The function
 * basically calls `PSLayerOutputs` on the layer returned by `PSGetOutputLayer`.
 */
PSFloat *PSModelOutputs(PSModel *model) {
    if (model == NULL) return NULL;
    return PSLayerOutputs(PSGetOutputLayer(model));
}

PSFloat PSGetNeuronState(PSNeuron *neuron, ...) {
    if (neuron->layer == NULL) return 0.0;
    if (PSUseSequences(neuron->layer)) {
        va_list args;
        va_start(args, neuron);
        long t = va_arg(args, long);
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

/* Check whether `model` is built (see: `PSModelBuild`).
 * Return value: 1 if the model is built, 0 if it's not built. */
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
            layer->flags |= PS_FLAG_RECURRENT;
            ctx->first_recurrent_layer = layer;
        } else if (i == last_layer_idx && recurrent_output) {
            layer->flags |= PS_FLAG_RECURRENT;
            ctx->last_recurrent_layer = layer;
        } else if (ManyToMany == mode) {
            layer->flags |= PS_FLAG_RECURRENT;
        } else if (ManyToOne == mode) {
            if (i < last_layer_idx) {
                if (PSIsRecurrent(layer)) {
                    ctx->last_recurrent_layer = layer;
                    for (j = 1; j < layer->index; j++)
                        model->layers[j]->flags |= PS_FLAG_RECURRENT;
                }
            } else if (RNNLayer != type && LSTM != type && GRU != type) {
                model->layers[i]->flags &= (unsigned) (~PS_FLAG_RECURRENT);
            }
        } else if (OneToMany == mode) {
            if (PSIsRecurrent(layer)) ctx->first_recurrent_layer = layer;
            else if (ctx->first_recurrent_layer != NULL) {
                if (layer->index > ctx->first_recurrent_layer->index)
                    layer->flags |= PS_FLAG_RECURRENT;
            }
        }
    }
}

/* Build `model` so that it can be used for training of for predictions.
 * If the model is already built, the function will just return 1. In order
 * to force rebuilding an already built model, `PSModelRebuild` should be used.
 * The function will check the model's architecture and it will perfrom various
 * actions on it:
 *  - It will allocate and initialize all needed internal data.
 *  - It will set proper flags both on the model and the layers.
 *  - It will determine and set the eventual recurrent mode
 *    (`PSRecurrentNetworkMode`) depening on model's architecture.
 *  - It will resolve eventual layer placeholders making them real layers.
 *  - If the loss function (member `loss` of `model`) is NULL, it will
 *    automatically determine it:
 *    - `PSCrossEntropyLoss` will be used if the output layer is a SoftMax
 *      layer.
 *    - PSQuadraticLoss in all the other cases.
 *  - If `model` is part of a multi-model chain, it will check and update all
 *    the chain properties.
 * Return value: 1 is `model` is successfully built, 0 if:
 *  - `model` is NULL.
 *  - `model` is empty (it contains no layers).
 *  - There was some memory allocation issue.
 *  - The structure of the `model` is not valid (ie. some layer is NULL)
 *  - `model` contains one or more layer placeholders and the function failed
 *    to resolve one of them.
 *  - `model` (or one of its layers) handles sequences-at-once but some of
 *    its layers is recurrent.
 *  - `model` (or one of its layers) is recurrent but some of its layers uses
 *    sequences-at-once.
 *  - `model` is recurrent but both the input layer and the output layer are
 *    not.
 *  - `model` is part of a multi-model chain but the chain is broken or not
 *    valid. */
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
            PSLayerFree(layer);
            model->layers[i] = resolved;
        } else if (layer->build != NULL) {
            if (!layer->build(layer)) return 0;
        }
        if (PSHandleSequenceAtOnce(layer)) {
            if (first_whole_seq_layer < 0) first_whole_seq_layer = i;
            model->flags |= PS_FLAG_USE_SEQUENCES;
        } else if (PSHandleSequenceAtOnce(model)) {
            if (first_whole_seq_layer < 0) first_whole_seq_layer = i;
            layer->flags |= PS_FLAG_USE_SEQUENCES;
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

/* Rebuild an already built `model` by resetting its `built` state and calling
 * `PSModelBuild`. If `model` is not built, calling this function is the same
 * as directly calling `PSModelBuild`.
 * Return value: see `PSModelBuild`. */
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
    if (mode != NonRecurrent) model->flags |= PS_FLAG_RECURRENT;
    return 1;
}

/* Create a new, empty model. The optional argument `name` can be used to
 * give a name to the model.
 * NOTE: the model will duplicate the eventually provided `name` and it will
 * keep it inside its internal data. The duplicated string will be
 * automatically freed by freeing the whole model (`PSModelFree`).
 * Return value: pointer to the created model or NULL if memory could not be
 * allocated for it. */
PSModel *PSModelCreate(const char* name) {
    PSModel *model = (malloc(sizeof(PSModel)));
    if (model == NULL) return NULL;
    model->context = calloc(1, sizeof(PSModelContext));
    if (model->context == NULL) goto memory_err;
    if (name != NULL) {
        model->name = strdup(name);
        if (model->name == NULL) goto memory_err;
        setModelContext(model, allocated_name, model->name);
    } else model->name = NULL;
    model->size = 0;
    model->index = 0;
    model->layers = NULL;
    model->input_size = 0;
    model->output_size = 0;
    model->status = PS_STATUS_UNTRAINED;
    model->flags = PS_FLAG_NONE;
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
    sequence_settings->pad = -1;
    return model;
memory_err:
    if (model != NULL) PSModelFree(model);
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
        long input_size = layer->size, onehot_size = 0;
        if (is_input && layer->flags & PS_FLAG_ONEHOT)
            onehot_size = PSGetOneHotLayerVectorSize(layer);
        for (j = prev_output_idx; j >= 0; j--) {
            PSLayer *prev_layer = previous->layers[j];
            long output_size = prev_layer->size;
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
    if (!ok && input_layer->index == 0 && input_layer->flags & PS_FLAG_ONEHOT) {
        long onehot_vector_size = PSGetOneHotLayerVectorSize(input_layer);
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

/* Add `model` to another model (`parent`), creating a chained, multi-model
 * model.
 * If `parent` is already member of a multi-model chain but it's not the
 * chain head, the function will automatically find the actual chain head
 * and it will append `model` to the chain tail.
 * The argument `link` allows setting the rules for data propagation (both
 * forward propagation and bacpropagation) between `model` and the model
 * preceding it in the model chain:
 *  - The `layer` member of `link` can be used to set the layer in `model`
 *    that will receive inputs from previous model (in forward propagation)
 *    or that will back-propagate the error (delta) to previous model.
 *  - The `previous_layer` member of `link` can be used to set the layer in
 *    the previous model (the model in the chain that precedes `model`)
 *    that will forward its outputs to `model` (in forward propagation) or
 *    that will receive the error (deltas) from `model` in backpropagation.
 * If `link` is NULL, the function will try to automatically determine it by
 * searching for the first layer in `model` whose size matches a layer in
 * the previous model.
 * Return value: 1 if `model` is successfully added, 0 if something goes wrong.
 * Possible failure reasons:
 *  - `model` is NULL or `parent` is NULL or both are NULL.
 *  - `model` is already part of a multi-model chain.
 *  - `link` is NULL and it's not possible to automatically determine it.
 *  - `link` is not NULL but it's not valid, because:
 *     - `link->layer` is NULL or `link->previous_layer` is NULL.
 *     - size of `link->layer` differs from size of `link->previous_layer`.
 *  - Memory issues. */
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
                PSModelFree(clone_next);
                return 0;
            }
            if (link->layer->index >= clone_next->size ||
                clone_next->layers[link->layer->index] == NULL)
            {
                PSErr("PSModelClone", "missing layer %d in cloned model %d",
                      link->layer->index, next->index);
                PSModelFree(clone_next);
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
                PSModelFree(clone_next);
                return 0;
            }
            if (link->previous_layer->index >= prev_model->size ||
                prev_model->layers[link->previous_layer->index] == NULL)
            {
                PSErr("PSModelClone", "missing layer %d in cloned model %d",
                      link->previous_layer->index, prev_model->index);
                PSModelFree(clone_next);
                return 0;
            }
            clone_link = malloc(sizeof(*clone_link));
            if (clone_link == NULL) {
                PSPrintMemoryErrorMsg();
                PSModelFree(clone_next);
                return 0;
            }
            clone_link->layer = clone_next->layers[link->layer->index];
            clone_link->previous_layer =
                prev_model->layers[link->previous_layer->index];
        }
        if (clone_link == NULL) {
            PSErr("PSModelClone", "could not make link for cloned model %d",
                  next->index);
            PSModelFree(clone_next);
            return 0;
        }
        int added = PSAddModel(clone, clone_next, clone_link);
        if (!added) {
            PSErr("PSModelClone", "unable to add cloned model %d",
                  next->index);
            PSModelFree(clone_next);
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
            clone->training->current_example =
                model->training->current_example;
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
            ldef.enabled_projections =
                PSGetAttentionEnabledProjections(layer);
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
            PSModelFree(clone);
            return NULL;
        }
        cloned_layer->flags = layer->flags;
        if (!layout_only) {
            if (cloned_layer->states != NULL) {
                PSMatrixFree(cloned_layer->states);
                cloned_layer->states = NULL;
            }
            if (layer->states != NULL) {
                cloned_layer->states = PSMatrixDup(layer->states);
                if (cloned_layer->states == NULL) goto memerr;
                if (layer->initial_states != NULL) {
                    long diff = (
                        layer->initial_states - layer->states
                    );
                    cloned_layer->initial_states = cloned_layer->states + diff;
                }
            } else {
                cloned_layer->states = NULL;
                cloned_layer->initial_states = NULL;
            }
            if (layer->weights != NULL) {
                if (layer->weight_types == 0) {
                    PSErr(__func__, "Layer[%d]: weights not NULL but "
                          "weight_types is 0", layer->index);
                    goto err;
                }
                if (cloned_layer->weights == NULL) {
                    cloned_layer->weights = calloc(
                        layer->weight_types, sizeof(PSMatrix)
                    );
                    if (cloned_layer->weights == NULL) goto memerr;
                }
                for (j = 0; j < layer->weight_types; j++) {
                    if (layer->weights[j] == NULL) {
                        if (type == Attention) continue;
                        PSErr(__func__, "Layer[%d]: weights[%d] are NULL",
                              layer->index, j);
                        goto err;
                    }
                    if (cloned_layer->weights[j] != NULL)
                        PSMatrixFree(cloned_layer->weights[j]);
                    cloned_layer->weights[j] = PSMatrixDup(layer->weights[j]);
                    if (cloned_layer->weights[j] == NULL) goto memerr;
                }
            } else if (cloned_layer->weights != NULL) {
                for (j = 0; j < layer->weight_types; j++)
                    PSMatrixFree(cloned_layer->weights[j]);
                free(cloned_layer->weights);
                cloned_layer->weights = NULL;
            }
            if (layer->biases != NULL && !(layer->flags & PS_FLAG_NO_BIAS)) {
                uint64_t bias_count = PSGetLayerParametersCount(
                    layer, PS_PARAM_BIAS
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
            if (layer->onCopy != NULL)
                if (!layer->onCopy(cloned_layer, layer)) goto err;
        }
    }
    if (clone->layers == NULL) {
        if (model->layers == NULL) return clone;
        else goto err;
    }
    clone->sequence_settings.max_length = model->sequence_settings.max_length;
    clone->sequence_settings.end = model->sequence_settings.end;
    clone->sequence_settings.pad = model->sequence_settings.pad;
    if (model->context != NULL) {
        memcpy(clone->context, model->context, sizeof(PSModelContext));
        setModelContext(clone, built, 0);
        setModelContext(clone, head_model, NULL);
        setModelContext(clone, last_model, NULL);
        setModelContext(clone, model_chain_length, 1);
        setModelContext(clone, allocated_name, NULL);
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
    if (clone != NULL) PSModelFree(clone);
    return NULL;
}

PSModel *PSModelClone(PSModel *model, int layout_only) {
    PSModel *clone =  cloneModel(model, layout_only, NULL);
    if (clone == NULL) return NULL;
    if (PSModelIsBuilt(model) && !PSModelBuild(clone)) {
        PSModelFree(clone);
        return NULL;
    }
    return clone;
}

/* Get the number of models in multi-model `model`.
 * Return value: the number of models or:
    - 0 if `model` is NULL or if the model chain is broken
    - 1 if `model` is not a multi-model chain. */
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

/* Get the model at `index` in the multi-model chain that contains the model
 * `entrypoint`. If `index` is negative, it will be counted from the end of
 * the model chain (ie. -1 is the last model, or tail,  of the chain).
 * Return value: the model or NULL if:
 *  - `entrypoint` is NULL
 *  - the model chain is broken
 *  - `index` is out of bounds. */
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

/* Get the first model (head) of the multi-model chain that contains `model`.
 * If `model` is not a multi-model chain, the function will return the `model`
 * itself.
 * Return value: the first model of the chain or NULL if:
 *  - `model` is NULL
 *  - the chain is broken. */
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

/* Get the last model (tail) of the multi-model chain that contains `model`.
 * If `model` is not a multi-model chain, the function will return the `model`
 * itself.
 * Return value: the last model of the chain or NULL if:
 *  - `model` is NULL
 *  - the chain is broken. */
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

/* Check whether `model` is contained by the multi-model chain `chain`.
 * Return value:
 *  - 1 if `model` is contained by `chain` or `model` == `chain`
 *  - 0 if `model` is not contained by `chain` or the chain is broken. */
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
            "training:started_at=%ld,current_epoch=%d,current_batch=%ld,"
            "current_example=%ld,batch_size=%ld\n",
            model->training->started_at, model->training->current_epoch,
            model->training->current_batch,
            model->training->current_example, model->training->batch_size
        );
    }
}

void DumpLayerInfo(PSLayer *layer, FILE *dump_file, int add_new_line) {
    PSLayerType ltype = layer->type;
    char *type_name = PSGetLayerTypeLabel(layer);
    fprintf(dump_file, "layer:index=%d,type=%s,size=%ld", layer->index,
        type_name, layer->size);
    int onehot_input = (layer->index == 0 && layer->flags & PS_FLAG_ONEHOT);
    if (onehot_input)
        fprintf(dump_file, ",vector_size=%ld", layer->onehot_vector_size);
    if (PSIsRecurrent(layer) && PSIsRecurrent(layer->model))
        fprintf(dump_file, ",recurrent=1");
    if (ltype == Convolutional || ltype == Pooling) {
        PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
        long output_w = layer->output_columns, output_h = layer->output_rows,
             depth = layer->output_depth, filter_w = 0, filter_h = 0,
             input_w = 0, input_h = 0;
        int stride = 0, padding = 0;
        if (settings != NULL) {
            input_w = settings->input_width;
            input_h = settings->input_height;
            filter_w = settings->filter_width;
            filter_h = settings->filter_height;
            stride = settings->stride;
            padding = settings->padding;
        }
        if (stride <= 0 && ltype == Pooling) stride = (int) filter_w;
        fprintf(
            dump_file, ",input_size=%ldx%ld,output_size=%ldx%ld,features=%ld"
            ",region=%ldx%ld,stride=%d",
            input_w, input_h, output_w, output_h, depth,
            filter_w, filter_h, stride
        );
        if (ltype == Convolutional) {
            if (padding < 0) padding = 0;
            fprintf(dump_file, ",padding=%d", padding);
        }
    } else if (ltype == FullyConnected && layer->output_depth > 1) {
        fprintf(dump_file, ",depth=%ld", layer->output_depth);
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
            has_seq = PSUseSequences(layer);
        long nidx = 0, t = 0, seqlen = 0;
        DumpLayerInfo(layer, f, 0);
        if (has_seq) {
            seqlen = PSStateSequenceLength(layer);
            fprintf(
                f, ",%s=%ld", (is_recurrent ? "timesteps" : "sequence_length"),
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
        long nidx = 0;
        DumpLayerInfo(layer, f, 0);
        PSMatrix delta = layer->delta;
        if (delta == NULL) {
            fprintf(f, ",deltas=()\n");
            continue;
        }
        fprintf(f, ",deltas=(");
        long len = PSMatrixLength(delta);
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
    const char *name = ctx->allocated_name;
    if (model->name == name) model->name = NULL;
    free((void*) ctx->allocated_name);
    free(ctx->sequence_start);
    free(ctx);
}

/* Free `model` and all its related objects (layers, data, ...). If the model
 * is part of a multi-model chain, all models following `model` will also be
 * freed.
 * The functions safely checks whether `model` is NULL and it does nothing
 * in this case. */
void PSModelFree(PSModel *model) {
    if (model == NULL) return;
    PSModelContext *ctx = getModelContext(model);
    if (ctx != NULL) deleteModelContext(model->context, model);
    int size = model->size;
    int i, is_recurrent = (model->flags & PS_FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        PSLayer *layer = NULL;
        if (model->layers != NULL) layer = model->layers[i];
        if (layer == NULL) continue;
        if (is_recurrent) layer->flags |= PS_FLAG_RECURRENT;
        PSLayerFree(layer);
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
        PSModelFree(next);
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

PSNeuron *PSGetNeuron(PSLayer *layer, long index, PSNeuron *neuron) {
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
        long feature_size = layer->size / layer->output_depth;
        long feature_idx = index / feature_size;
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
            long prevsize = previous->size;
            if (previous->flags & PS_FLAG_ONEHOT)
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

/* TODO: Remove? */
void PSSetDefaultLayerDef(PSLayerDef *ldef, PSLayerType type) {
    memset(ldef, 0, sizeof(*ldef));
    UNUSED(type);
    /* TODO: use specific settings for type? */
}

PSMatrix PSInitWeights(PSLayer *layer, long rows, long columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale)
{
    static PSLayerDef default_def = {0};
    if (ldef == NULL) ldef = &default_def;
    PSMatrix weights = NULL;
    if (ldef->weight_init_mode == PS_INIT_MODE_ZERO)
        weights = PSMatrixZeros(2, rows, columns);
    else if (ldef->weight_init_mode == PS_INIT_MODE_VALUE) {
        PSFloat init_value = ldef->init_value;
        weights = PSMatrixCreate(init_value, NULL, 2, rows, columns);
    } else {
        if (ldef->weight_init_mode == PS_INIT_MODE_RAND) {
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
    int mode = PS_INIT_MODE_AUTO;
    if (param_type == PS_PARAM_BIAS) mode = ldef->bias_init_mode;
    else mode = ldef->weight_init_mode;
    PSFloat param;
    if (mode == PS_INIT_MODE_ZERO) param = 0.0;
    else {
        if (mode == PS_INIT_MODE_RAND) {
            range = ldef->init_range;
            scale = ldef->init_scale;
        }
        if (range == 0) range = 1;
        param = PSGaussianRandom(0, range);
        if (scale) param *= scale;
    }
    return param;
}

int initGenericLayer(PSLayer *layer, long size, long previous_size,
                     PSLayerDef *ldef)
{
    if (layer == NULL) return 0;
    if (layer->model == NULL) {
        PSErr(NULL, "Layer[%d]: missing model");
        goto fail;
    }
    layer->states = PSMatrixZeros(2, 1, size);
    if (layer->states == NULL) goto memerr;
    if (layer->index > 0 && previous_size > 0) {
        layer->weights = calloc(1, sizeof(PSMatrix));
        if (layer->weights == NULL) goto memerr;
        layer->weights[0] = PSInitWeights(
            layer, size, previous_size, ldef, 1.0, 0
        );
        if (layer->weights[0] == NULL) goto memerr;
        layer->weight_types = 1;
        layer->biases = malloc(size * sizeof(PSFloat));
        if (layer->biases == NULL) goto memerr;
    }
    int i;
    for (i = 0; i < size; i++) {
        if (layer->index > 0 && previous_size > 0)
            layer->biases[i] = PSInitParam(PS_PARAM_BIAS, ldef, 1.0, 0.0);
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

/* Add a new layer (instance of `PSLayer`) of type `type` and size `size` to
 * `model`. Special layer properties can be defined by the optional argument
 * `layer_def`.
 * If `layer_def` is NULL, the function will use the default layer
 * configuration.
 * The member `load_from` of `layer_def` can be used to load the new layer's
 * parameters from a file.
 * The new model will be automatically allocated and added to model layers.
 * NOTE: the new layer should never be freed directly. By freeing `model`
 * (`PSModelFree`), all model's layers will be automatically freed.
 * Return value: pointer to the added layer or NULL if something goes wrong.
 * Possible failure reasons:
 *  - `model` is NULL
 *  - `model` is empty and `type` is not `FullyConnected` (the first layer
 *    must be always of type FullyConnected).
 *  - The new layer cannot be allocated into memory or the model's `layers`
 *    array cannot be resized.
 *  - The model's last layer is NULL.
 *  - The new layer cannot be initialized. The reason for the initialization
 *    failure can vary depending on the layer type.
 *  - The new layer is recurrent or the model is recurrent but the recurrent
 *    mode of all layers is not consistent. In order to build consistent
 *    recurrent models, one of the following feature must be satisfied:
 *    - All layers must be recurrent, or
 *    - first N layers are recurrent and the remaining layers are not
 *       recurrent, or
 *    - first N layers are not recurrent the remaining layers are recurrent.
 */
PSLayer *PSAddLayer(PSModel *model, PSLayerType type, long size,
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
    layer->onDelete = NULL;
    layer->onCopy = NULL;
    layer->build = NULL;
    layer->getParamCount = NULL;
    layer->weight_types = 0;
    layer->onehot_vector_size = size;
    layer->output_depth = layer_def->output_depth;
    layer->pretrained = layer_def->pretrained;
    layer->pretrainer = NULL;
    layer->pretrain = NULL;
    layer->private = NULL;
    layer->beforeBatchTraining = NULL;
    layer->onStatesInit = NULL;
    layer->onStatesResize = NULL;
    layer->getInputFromLink = NULL;
    if (layer->output_depth <= 0) layer->output_depth = 1;
    layer->output_columns = layer_def->output_columns;
    layer->output_rows = layer_def->output_rows;
    if (layer->output_columns < 0) layer->output_columns = 0;
    if (layer->output_rows < 0) layer->output_rows = 0;
    PSLayer *previous = NULL;
    long previous_size = 0;
    int initialized = 0;
    if (verbose) printf("Adding layer %d\n", layer->index);
    if (model->layers == NULL) {
        model->layers = malloc(sizeof(PSLayer*));
        if (model->layers == NULL) {
            PSAbortLayer(model, layer);
            PSErr(__func__, "could not allocate model layers");
            return NULL;
        }
        if (layer->flags & PS_FLAG_ONEHOT) model->flags |= PS_FLAG_ONEHOT;
        if (model->flags & PS_FLAG_ONEHOT) {
            layer->flags |= PS_FLAG_ONEHOT;
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
        if (layer->index == 1 && previous->flags & PS_FLAG_ONEHOT)
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
    } else if (type == RNNLayer) {
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
            ok = PSSetRecurrentNetworkMode(model, PS_DEFAULT_RECURRENT_MODE);
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
        int loaded = PSLayerLoad(layer, layer_def->load_from);
        if (!loaded) {
            PSAbortLayer(model, layer);
            PSErr(
                __func__, "Could load parameters for layer %d from '%s'",
                layer->index, layer_def->load_from
            );
            return NULL;
        }
    }
    if (layer->flags & PS_FLAG_USE_SEQUENCES)
        model->flags |= PS_FLAG_USE_SEQUENCES;
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

/* Add input layer of size `size` to `model`. The layer type will be set to
 * the default `FullyConnected` type.
 * Special layer properties can be defined by the optional argument
 * `layer_def`.
 * If `layer_def` is NULL, the function will use the default layer
 * configuration.
 * See `PSAddLayer` for further details about adding layers.
 * Return value: the added layer or NULL if:
 *  - `model` is NULL.
 *  - `model` is not empty.
 *  - See `PSAddLayer` for more failure reasons. */
PSLayer *PSAddInputLayer(PSModel *model, long size, PSLayerDef *layer_def) {
    if (model == NULL) return NULL;
    if (model->size > 0) {
        PSErrNN(__func__, model, NULL, "model is not empty");
        return NULL;
    }
    return PSAddLayer(model, FullyConnected, size, layer_def);
}

/* Add a convolutional layer to `model`. This function basically calls
 * `PSAddLayer`:
 * ```
 * PSAddLayer(model, Convolutional, 0, ldef);
 * ```
 * Return value: see `PSAddLayer`. */
PSLayer *PSAddConvolutionalLayer(PSModel *model, PSLayerDef *ldef) {
    return PSAddLayer(model, Convolutional, 0, ldef);
}

/* Add a pooling layer to `model`. This function basically calls `PSAddLayer`:
 * ```
 * PSAddLayer(model, Pooling, 0, ldef);
 * ```
 * Return value: see `PSAddLayer`. */
PSLayer *PSAddPoolingLayer(PSModel *model, PSLayerDef *ldef) {
    return PSAddLayer(model, Pooling, 0, ldef);
}

/* Free memory allocated for `layer` and all of its objects (ie. weights,
 * states).
 * WARN: this function should be called only for layers not being part of
 * any model, since by freeing models (`PSModelFree`), all their layers
 * will be automatically freed. */
void PSLayerFree(PSLayer* layer) {
    if (layer == NULL) return;
    int i;
    if (layer->weights != NULL) {
        for (i = 0; i < layer->weight_types; i++) {
            PSMatrix weights = layer->weights[i];
            if (weights != NULL) PSMatrixFree(weights);
        }
        free(layer->weights);
    }
    if (layer->biases != NULL) free(layer->biases);
    if (layer->onDelete != NULL) layer->onDelete(layer);
    void *extra = layer->extra;
    if (extra != NULL) free(layer->extra);
    if (layer->delta != NULL) PSMatrixFree(layer->delta);
    if (layer->states != NULL) PSMatrixFree(layer->states);
    if (layer->pretrainer != NULL) PSModelFree(layer->pretrainer);
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
    long input_size = first->size, seqlen = 1, t = 0;
    int is_recurrent = PSIsRecurrent(first),
        seq_at_once = PSHandleSequenceAtOnce(first);
    long len = input_size;
    if (is_recurrent || seq_at_once) {
        va_list ap;
        va_start(ap, inputs);
        seqlen = va_arg(ap, long);
        if (is_recurrent) t = va_arg(ap, long);
        else if (seq_at_once) {
            len *= seqlen;
            t = 0;
        }
        va_end(ap);
        assert(seqlen > 0);
        assert(t >= 0);
        if (!PSBeforeSequenceForward(first, seqlen, t)) return 0;
    }
    PSFloat *states = PSLayerStates(first, t);
    if (states == NULL) {
        PSErr(NULL, "Layer[%d] missing states");
        return 0;
    }
    PSVectorCopy(states, inputs, len);
    return 1;
}

/* Forward recurrent inputs to model. The inputs vector must represent the
 * actual input data to be forwarded (without the sequence length that must
 * be passed via the `timesteps` argument. */
int forwardThroughTime(PSModel *model, PSFloat *inputs,
                       long timesteps, int backprop, void *opts)
{
    if (model == NULL) return 0;
    PSMathOpts mopts = {.acceleration = model->acceleration};
    PSFloat *tmpinputs = NULL;
    long output_idx = model->size - 1, i, t;
    int ok = 1;
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
    long input_size = input_layer->size, last_layer_idx = last->index,
         start_idx = first->index;
    PSTrainingOptions *training_opts = NULL;
    PSForwardOptions *forward_opts = NULL;
    if (backprop) training_opts = (PSTrainingOptions *) opts;
    else forward_opts = (PSForwardOptions *) opts;
    long end = -1, seqlen = timesteps, autoregression_since = 0;
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
                training_opts->flags & PS_TRAINING_FLAG_TEACHER_FORCING
            );
            /* Teacher forcing actually disables autoregression, since it
             * uses the target sequence (the expected values) as input
             * sequence. */
            if (teacher_forcing) autoregression = 0;
        } else if (sequence_settings != NULL) end = sequence_settings->end;
    }
    if (autoregression) {
        PSFloat *start = NULL;
        start = sequence_settings->start;
        long maxlen = sequence_settings->max_length;
        if (maxlen <= 0) maxlen = PS_MAX_SEQUENCE_LENGTH;
        if (timesteps <= 0 || inputs == NULL) {
            if (timesteps <= 0) timesteps = 1;
            seqlen = 1;
        }
        if (seqlen >= maxlen) {
            autoregression = 0;
            goto forward_steps;
        }
        autoregression_since = seqlen - 1;
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
                model->flags & PS_FLAG_RANDREGRESSION ||
                (forward_opts && forward_opts->flags & PS_FLAG_RANDREGRESSION)
            );
            if (recurrent_output) {
                long osize = output_layer->size, isize = input_size;
                if (input_layer->flags & PS_FLAG_ONEHOT)
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
        if (autoregression && t >= autoregression_since) {
            if (t == timesteps - 1) break;
            long max_idx = -1;
            PSFloat *output_states = NULL;
            if (!randomized_autoregression) {
                ok = PSFindLayerMaxState(last, NULL, &max_idx, t);
                if (!ok) goto final;
            } else {
                output_states = PSLayerStates(last, t);
                ok = output_states != NULL;
                if (!ok) {
                    PSErr(NULL, "Layer[%d] has no states at step %d",
                          last->index, t);
                    goto final;
                }
                max_idx = PSRandomInt(last->size, output_states, &mopts);
                ok = (max_idx >= 0);
                if (!ok) {
                    PSErr(NULL, "Failed to get random weighted index from "
                          "layer %d", last->index);
                    goto final;
                }
            }
            if (end >= 0 && max_idx == end) break;
            if (autoregression_feed_output && tmpinputs != NULL) {
                long next_t = t + 1;
                inputs = tmpinputs + next_t;
                if (first->flags & PS_FLAG_ONEHOT) *inputs = (PSFloat) max_idx;
                else {
                    if (output_states == NULL)
                        output_states = PSLayerStates(last, t);
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

static int useAutoRegression(PSModel *model, PSForwardOptions *forward_opts,
                             PSTrainingOptions *training_opts)
{
    if (!PSUseSequences(model)) return 0;
    if (!PSUseSequences(model->layers[model->size - 1])) return 0;
    if (PSIsModelChain(model) && model != PSModelChainTail(model))
        return 0;
    if (model->flags & PS_FLAG_AUTOREGRESSION) return 1;
    else if (forward_opts != NULL)
        return forward_opts->flags & PS_TRAINING_FLAG_AUTOREGRESSION;
    else if (training_opts != NULL && model->next == NULL)
        return training_opts->flags & PS_TRAINING_FLAG_AUTOREGRESSION;
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
        long seqlen = 1;
        if (!input_whole_seq) {
            outputs = PSLayerOutputs(link->previous_layer);
        } else {
            seqlen = PSStateSequenceLength(link->previous_layer);
            outputs = PSLayerStates(link->previous_layer);
        }
        if (do_feed_input) *inputs_p = outputs;
        else {
            /* Use link to forward data from previous model. */
            /* TODO (S2S): return 0 if model doesn't support autogression ? */
            if (outputs == NULL) {
                PSErr(__func__, "no outputs from model %d, layer %d",
                      link->previous_layer->model->index,
                      link->previous_layer->index);
                return 0;
            }
            if (link->layer->getInputFromLink != NULL) {
                if (!link->layer->getInputFromLink(link->previous_layer))
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
    int is_recurrent = PSIsRecurrent(model), recurrent_input = 0,
        input_is_seq = 0, is_input_model = model->previous == NULL,
        use_seq = PSUseSequences(model), ok = 1;
    long seqlen = 0, first_idx = 0, output_idx = model->size - 1, i;
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
                    (train_opts->flags & PS_TRAINING_EPOCH_AS_SEQUENCE)
                );
                if (retain_previous && model->training != NULL)
                    retain_previous = (model->training->current_batch > 0);
            }
            first_recurrent = PSGetFirstRecurrentLayer(model);
            last_recurrent = PSGetLastRecurrentLayer(model);
            recurrent_input = PSIsRecurrent(input_layer);
        } else if (input_layer->flags & PS_FLAG_USE_SEQUENCES) {
            input_is_seq = 1;
        }
        if (is_input_model && inputs != NULL) {
            if (recurrent_input || input_is_seq) {
                /* Read seqlen from first element in `inputs`. */
                seqlen = (long) inputs[0];
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
                train_opts && (train_opts->flags & PS_TRAINING_FLAG_SEQ2SEQ)
            );
            /* TODO (S2S): only check seq2seq or also autoregression ?? */
            if (autoregression || seq2seq) {
                int training_flags = (train_opts ? train_opts->flags : 0);
                PSFloat *y = NULL;
                long datalen = parseSequenceData(
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
                    train_opts->flags & PS_TRAINING_FLAG_TEACHER_FORCING
                );
                if (teacher_forcing && seqlen > 0) {
                    long final_len = 0;
                    tmpinputs = prepareSequence(
                        y, seqlen, 0, input_layer->size, &final_len,
                        &(model->sequence_settings),
                        SEQ_MODE_PREPEND_START | SEQ_MODE_PREPEND_SEQLEN
                    );
                    ok = (tmpinputs != NULL && final_len > 1);
                    if (!ok) {
                        PSErrNN(NULL, model, NULL, "forward: cannot prepare "
                                "input sequence for teacher forcing");
                        goto final;
                    }
                    inputs = tmpinputs;
                    seqlen = final_len;
                    *inputs = (PSFloat) seqlen;
                }
            }
        }
        if (is_recurrent) { /* TODO (S2S): only for recurrent input? */
            long steps = seqlen;
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
        /* First inputs element contains the sequence length, so actual
         * input data start at inputs + 1. */
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
    if (!ok) PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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

/* Forward `inputs` to `model`. If `model` is part of a multi-model chain,
 * `inputs` are forwarded to the first layer of the first model of the chain.
 * If the input layer doen't accept sequences as inputs, the `inputs` array's
 * length must match the `size` of the first layer.
 * When the first layer takes sequences (if it has the flags `PS_FLAG_RECURRENT`
 * or PS_FLAG_USE_SEQUENCES` set), the length of `inputs` should be the
 * (input layer size * sequence length) + 1, and the first element of `inputs`
 * should contain the length of the sequence.
 * Return value: 1 if the process succeeds or 0 if:
 *  - `model` is NULL
 *  - `model` is not built.
 *  - The input layer doesn't take sequences as inputs and the output layer
 *    doesn't produce sequences as outputs.
 *  - Something else in the forward process fails. */
int PSForward(PSModel *model, PSFloat *inputs) {
    return forward(model, inputs, 0, NULL);
}

/* Forward `inputs` to `model` with autoregression mode. The model must take
 * sequences as inputs and produce sequences as outputs.
 * If `model` is part of a multi-model chain, `inputs` are forwarded to the
 * first layer of the first model of the chain and outputs are produced by
 * the last layer of the last model of the chain.
 * When the full input sequence as been forwarded, all subsequent outputs
 * produced by the model (including the last outputs produced by the input
 * sequence) are forwarded as the next inputs to the model itself.
 * The size of the output layer must match the size of the input layer or,
 * if the input layer has the `PS_FLAG_ONEHOT` set, its `onehot_vector_size`.
 * If the input layer has the `PS_FLAG_ONEHOT` set, the index of the highest
 * element of the produced outputs is forwarded to the input layer. In this
 * case, if the `randomized` argument is true, a random index is generated
 * by using the values of the outputs as a probability distribution.
 * The iteration keeps forwarding outputs as the next inputs until one of the
 * following events happens:
 *  - The length of the whole output sequence produced (including the outputs
 *    produced by the original input sequence) reaches the maximum length
 *    defined into the (optional) argument `sequence_settings` or by
 *    the default value of `PS_MAX_SEQUENCE_LENGTH`.
 *  - The index of the highest value of the produced outputs matches the
 *    value of `end` in the optional argument `sequence_settings`. If
 *    the `randomized` argument is true, the random index generated by
 *    using outputs as a probability distribution is compared with `end`.
 *    If `sequence_settings` is NULL or the value of the `end` member is
 *    negative, the end-matching event is ignored.
 * Return value: 1 if the process succeeds or 0 if:
 *  - `model` is NULL
 *  - `model` is not built.
 *  - The input layer doesn't take sequences as inputs and the output layer
 *    doesn't produce sequences as outputs.
 *  - The size of the output layer doesn't match the size of the input layer
 *    (or input layer's `onehot_vector_size` if the input layer has the
 *    flag `PS_FLAG_ONEHOT` set).
 *  - Something else in the forward process fails. */
int PSAutoregression(PSModel *model, PSFloat *inputs,
                     int randomized, PSSequenceSettings *sequence_settings)
{
    if (!PSIsRecurrent(model) && !PSHandleSequenceAtOnce(model)) {
        PSErr(__func__, "autoregression is only available in model "
              "that use sequences");
        return 0;
    }
    PSForwardOptions opts = {.flags = PS_FLAG_AUTOREGRESSION};
    if (randomized) opts.flags |= PS_FLAG_RANDREGRESSION;
    if (sequence_settings) opts.sequence_settings = sequence_settings;
    return forward(model, inputs, 0, &opts);
}

/* Forward `inputs` to `model` and get the index of the maximum state
 * from the output layer.
 * Return value: 1 if the process succeeds or 0 if:
 *  - `model` is NULL
 *  - `model` is not built.
 *  - The input layer doesn't take sequences as inputs and the output layer
 *    doesn't produce sequences as outputs.
 *  - The index of the maximum state could not be determined.
 *  - Something else in the forward process fails. */
long PSClassify(PSModel *model, PSFloat *inputs) {
    int ok = PSForward(model, inputs);
    if (!ok) {
        PSErr(__func__, "forward failed");
        return -1;
    };
    PSLayer *out = PSGetOutputLayer(model);
    long max_idx = 0, seqlen = PSStateSequenceLength(out),
         t = (long) seqlen - 1;
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
    int use_bias = !(layer->flags & PS_FLAG_NO_BIAS);
    uint64_t bias_count = 0, weights_count = 0, max_count = 0;
    if (use_bias)
        bias_count = PSGetLayerParametersCount(layer, PS_PARAM_BIAS);
    weights_count = PSGetLayerParametersCount(layer, PS_PARAM_WEIGHT);
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

void PSDeleteGradientsChain(PSGradient ***gradients, PSModel *model) {
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
    int whole_seq = PSHandleSequenceAtOnce(layer);
    if (!full_reset && PSMatrixDim(layer->delta, 1) > layer->size) {
        /* Delta contains data for differente stuff (ie. LSTM layer
         * allocate layer->size * 2 delta in order to store delta
         * for their raw states.
         * In this case, reset is performed only on first N
         * (where N=layer->size) values. */
         assert(!whole_seq);
         long rows = PSMatrixDim(layer->delta, 0), i;
         for (i = 0; i < rows; i++) {
            PSFloat *row = PSMatrixGet(layer->delta, 1, NULL, i);
            PSVectorClear(row, layer->size);
         }
         return 1;
    }
    if (whole_seq) {
        long seqlen = PSStateSequenceLength(layer),
             delta_seqlen = PSMatrixDim(layer->delta, 0);
        if (seqlen < 1) seqlen = 1;
        if (seqlen != delta_seqlen) {
            PSMatrixFree(layer->delta);
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
        PSMatrixFree(link->previous_layer->delta);
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
                for (j = 0; j < layer->weight_types; j++) {
                    PSMatrix weights = layer->weights[j];
                    if (weights == NULL) continue;
                    PSMatrixResetTransposed(weights);
                }
            }
            if (layer->beforeBatchTraining != NULL)
                layer->beforeBatchTraining(layer);
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
        for (j = 0; j < layer->weight_types; j++) {
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
    PSOptimization optimization = PSSGDOptimization;
    if (opts != NULL) {
        momentum = opts->momentum;
        optimization = opts->optimization;
    }
    if (momentum > 0.0 || optimization != PSSGDOptimization) {
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

PSFloat *PSSetSequenceStart(PSModel *model, PSFloat *start, long len) {
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
                      PSFloat *outputs, long size, PSMathOpts *opts)
{
    PSFloat *deriv = calloc(size, sizeof(PSFloat));
    if (deriv == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSMathOpts mopts = {.acceleration = PSGlobalAcceleration};
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
    derivative(outputs, deriv, size, acceleration);
    PSMultiplyVectors(delta, deriv, delta, size, &mopts);
    free(deriv);
    return 1;
}

int PSBeforeLayerBackprop(PSLayer *layer, PSLayer *previous, long *step,
                          long *seqlen, PSFloat **outputs, PSFloat **inputs,
                          va_list args)
{
    assert(inputs != NULL);
    assert(outputs != NULL);
    if (previous == NULL) previous = PSGetPreviousLayer(layer);
    int is_recurrent = PSIsRecurrent(layer),
        prev_is_recurrent = (previous != NULL && PSIsRecurrent(previous)),
        handle_seq = PSHandleSequenceAtOnce(layer);
    long t = 0, prev_t = 0, slen = 1;
    *outputs = NULL;
    *inputs = NULL;
    if (is_recurrent) {
        t = va_arg(args, long);
        if (prev_is_recurrent) prev_t = t;
    } else if (prev_is_recurrent) {
        prev_t = PSStateSequenceLength(previous) - 1;
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
    if (*outputs == NULL && !handle_seq) *outputs = PSLayerStates(layer, t);
    if (*inputs == NULL && previous != NULL && !handle_seq)
        *inputs = PSLayerStates(previous, prev_t);
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
                          long size, long input_size,
                          long seqlen, int acceleration)
{
    if (seqlen < 1) seqlen = 1;
    PSMathOpts opts = {.acceleration = acceleration};
    PSFloat *delta_p = delta, *input_p = inputs;
    for (long i = 0; i < seqlen; i++) {
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
                  long seqlen, int acceleration)
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
                      PSLayer *previous, int use_bias, long seqlen)
{
    if (seqlen < 1 || !PSHandleSequenceAtOnce(layer)) seqlen = 1;
    PSFloat *gweights = gradient->weights, *gbias = NULL;
    if (use_bias) gbias = gradient->biases;
    PSUpdateGradientData(gweights, gbias, inputs, layer->delta,
                         layer->size, previous->size, seqlen,
                         layer->model->acceleration);
}

int PSUpdatePreviousLayerDelta(PSLayer *layer, PSLayer *previous,
                               int weights_index, long seqlen)
{
    if (layer->weights == NULL ||
        weights_index >= layer->weight_types ||
        layer->weights[weights_index] == NULL)
    {
        PSErr(NULL, "Layer[%d] NULL weights", layer->index);
        return 0;
    }
    if (seqlen < 1 || !PSHandleSequenceAtOnce(layer)) seqlen = 1;
    PSMatrix weights = layer->weights[weights_index];
    int ok = PSUpdateDelta(
        previous->delta, layer->delta, weights, seqlen,
        layer->model->acceleration
    );
    if (!ok) {
        PSErr(
            NULL, "Layer[%d]: failed backprop (PSDot) (seqlen = %ld)",
            layer->index, seqlen
        );
    }
    return ok;
}

int PSSoftmaxBackward(PSFloat *softmax_out, PSFloat *delta, PSFloat *dest,
                      long len, int acceleration)
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
    PSMatrixFree(diagonal);
    PSMatrixFree(tmpdest);
    PSMatrixFree(sout);
    return success;
}

int computeSoftmaxOutputDelta(PSLayer *layer, PSFloat *y, ...) {
    PSModel *model = layer->model;
    assert(layer->type == SoftMax);
    int ok = 1, handle_seq = PSHandleSequenceAtOnce(layer);
    long t = 0, seqlen = 1;
    int apply_derivative = outputDerivativeNeeded(model);
    int onehot = (layer->flags & PS_FLAG_ONEHOT);
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
    long delta_len = PSMatrixLength(delta);
    if (seqlen < 1 || !handle_seq) {
        seqlen = 1;
        delta_len = layer->size;
    }
    PSVectorCopy(delta, outputs, delta_len);
    PSFloat *delta_p = delta;
    PSFloat *targets = y;
    for (long i = 0; i < seqlen; i++) {
        long oidx;
        if (onehot) oidx = (long) *(y++);
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
        for (long i = 0; i < seqlen; i++) {
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
    PSTrainingOptions *opts = PSGetModelTrainingOptions(model);
    if (opts != NULL && opts->metrics & PS_TRAINING_METRICS_ACCURACY)
        updateTrainingAccuracy(model, outputs, targets, seqlen);
    return 1;
}

int computeOutputDelta(PSLayer *layer, PSFloat *y, ...) {
    PSModel *model = layer->model;
    int handle_seq = PSHandleSequenceAtOnce(layer), ok = 1;
    int is_softmax = layer->type == SoftMax;
    long t = 0, seqlen = 1;
    PSFloat *outputs = NULL, *inputs = NULL;
    /* Checks */
    va_list args;
    va_start(args, y);
    ok = PSBeforeLayerBackprop(layer, NULL, &t, &seqlen,&outputs,&inputs,args);
    va_end(args);
    if (!ok) return 0;
    if (is_softmax) return computeSoftmaxOutputDelta(layer, y, t);
    int onehot = (layer->flags & PS_FLAG_ONEHOT);
    PSMatrix delta = layer->delta;
    if (delta == NULL) {
        PSErr(NULL, "Output layer[%d] has no delta");
        return 0;
    }
    long delta_len = PSMatrixLength(delta);
    PSMathOpts mopts = {.acceleration = model->acceleration};
    /* Compute delta */
    PSFloat *targets = y;
    if (!onehot) PSSubtractVectors(outputs, y, delta, delta_len, &mopts);
    else {
        PSVectorCopy(delta, outputs, delta_len);
        if (seqlen < 1 || !handle_seq) seqlen = 1;
        PSFloat *delta_p = delta;
        for (long i = 0; i < seqlen; i++) {
            long oidx = (long) *(y++);
            delta_p[oidx] -= 1;
            delta_p += layer->size;
        }
    }
    PSTrainingOptions *opts = PSGetModelTrainingOptions(model);
    if (opts != NULL && opts->metrics & PS_TRAINING_METRICS_ACCURACY)
        updateTrainingAccuracy(model, outputs, targets, seqlen);
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
        use_bias = !(layer->flags & PS_FLAG_NO_BIAS),
        ok = 1;
    long t = 0, seqlen = 1;
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
        long size = layer->size;
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

/* Backward propagate of the error (loss) for expected target (`y`) and
 * output predictions on a recurrent model and update `gradients`.
 * The `y` vector (expected targets) only contains actual data (the sequence
 * length must be passed to `timesteps` argument). */
int backpropThroughTime(PSModel *model, PSFloat *y,
                        PSGradient **gradients, PSTrainingOptions *opts,
                        long timesteps)
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
    long last_t = timesteps - 1;
    int recurrent_output = (last_recurrent == output_layer);
    int is_output_model = (model->next == NULL);
    int bptt_truncate = (opts != NULL ? opts->bptt_truncate : BPTT_TRUNCATE);
    if (bptt_truncate < 0) bptt_truncate = 0;
    int onehot = 0, ok = 1;
    long osize, ysize, i, t;
    PSFloat *eos_labels = NULL;
    if (recurrent_output && is_output_model) {
        ok = (y != NULL);
        if (!ok) {
            PSErr(NULL, "Label values for recurrent output layer are NULL");
            goto final;
        }
        onehot = (output_layer->flags & PS_FLAG_ONEHOT);
        osize = output_layer->size;
        ysize = (onehot ? 1 : osize);
        long hidden_states_count = PSStateSequenceLength(output_layer);
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
        long lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        if (recurrent_output && is_output_model) {
            /* Backpropagate starting from output layer. */
            long timestep_offset = t * ysize;
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
                    layer->derivative, delta, PSLayerStates(layer, t),
                    layer->size, &mopts
                );
                if (!ok) return 0;
            }
            /* If BPTT is truncated, delta value from previous iteration
             * is not cumulated since it has been already backpropagated
             * to previous timesteps during previous iteration.
             * So, reset previous layer deltas. */
            if (do_truncate && RNNLayer == previous_layer->type)
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
    int nlayers = model->size, is_recurrent = PSIsRecurrent(model);
    long seqlen = 0;
    PSLayer *output_layer = model->layers[nlayers - 1];
    PSGradient *lgradients = gradients[nlayers - 2]; /* No gradient for
                                                        inputs */
    int teacher_forcing = (
        y != NULL && model->next == NULL &&
        PSUseSequences(model) && PSUseSequences(model->layers[0]) &&
        opts != NULL && opts->flags & PS_TRAINING_FLAG_TEACHER_FORCING
    );
    if (teacher_forcing && PSUseSequences(output_layer)) {
        /* Since teacher forcing used expected targets (y) as inputs during
         * forward step after prepending the sequence start
         * (sequence_settings.start) to them ([<start>,y0,y1,...]), the
         * sequence ending item will be appended to the actual expected
         * targets: [y0,y1,...,<end>]. */
        seqlen = PSStateSequenceLength(model->layers[0]);
        ok = (seqlen > 1);
        if (!ok) {
            PSErrNN(NULL, model, NULL, "using teacher forcing but input "
                    "sequence length has less then two items.");
        }
        int onehot_labels = output_layer->flags & PS_FLAG_ONEHOT;
        long osize = (onehot_labels ? 1 : output_layer->size);
        long ylen = seqlen - 1, tmpy_len = 0;
        int seq2seq = (
            (opts != NULL && opts->flags & PS_TRAINING_FLAG_SEQ2SEQ) ||
            model->flags & PS_FLAG_AUTOREGRESSION
        );
        if (seq2seq) ylen = (long) *(y - 1);
        /* Append the sequence 'end' item if not already found at the end
         * of the `y` sequence and eventually trim the sequence 'start' item
         * if found at the beginning of the sequence. */
        tmpy = prepareSequence(
            y, ylen, seqlen, osize, &tmpy_len,
            &(model->sequence_settings), SEQ_MODE_APPEND_END
        );
        ok = (tmpy != NULL && seqlen == tmpy_len);
        if (!ok) {
            PSErrNN(NULL, model, NULL, "backprop: cannot prepare target "
                    "sequence for teacher forcing");
            free(tmpy);
            tmpy = NULL;
            goto final;
        }
        y = tmpy;
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
        long timesteps = seqlen ? seqlen : PSStateSequenceLength(output_layer);
        if (timesteps == 0) {
            PSErr(
                NULL, "Recurrent timesteps must be > 0 (found %ld)",
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
            long timesteps = PSStateSequenceLength(layer);
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
    long offset, long len, PSFloat rate, long iteration, int acceleration
)
{
    if (params == NULL || grads == NULL) return 0;
    PSOptimization optimization = PSSGDOptimization;
    PSFloat momentum = 0;
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    optimization = options->optimization;
    if (optimization == NULL) optimization = PSSGDOptimization;
    PSFloat *gptr = NULL, *mptr = NULL, *xptr= NULL;
    if (param_type == PS_PARAM_BIAS) {
        gptr = grads->biases;
        if (mg != NULL) mptr = mg->biases;
        if (xg != NULL) xptr = xg->biases;
    } else if (param_type == PS_PARAM_WEIGHT) {
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

int sumGradients(PSGradient **dstgrads, PSGradient **srcgrads, long count,
                 PSMathOpts *mopts)
{
    for (long i = 0; i < count; i++) {
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

void clipGradients(PSGradient **grads, PSFloat min, PSFloat max, long count,
                   PSMathOpts *mopts)
{
    for (long i = 0; i < count; i++) {
        PSGradient *g = grads[i];
        if (g == NULL) continue;
        if (g->bias_count > 0 && g->biases != NULL)
            PSVectorClip(g->biases, min, max, g->biases, g->bias_count, mopts);
        if (g->weight_count > 0 && g->weights != NULL)
            PSVectorClip(g->weights,min,max,g->weights,g->weight_count,mopts);
    }
}

static PSFloat getLoss(PSModel *model, PSFloat *y, long y_seqlen,
                       int backprop, PSFloat *outputs,
                       PSFloat l1_loss, PSFloat l2_loss,
                       PSFloat *net_loss, PSTrainingOptions *opts)
{
    static PSTrainingOptions dfopts = {0};
    if (opts == NULL) opts = &dfopts;
    if (net_loss != NULL) *net_loss = 0;
    long batch_size = opts->batch_size;
    if (batch_size <= 0) batch_size = 1;
    PSLayer *output_layer = PSGetOutputLayer(model);
    assert(output_layer != NULL);
    PSModel *output_model = output_layer->model;
    assert(output_model != NULL);
    long target_elem_size = output_layer->size, ok = 1, i;
    int onehot = output_layer->flags & PS_FLAG_ONEHOT;
    if (onehot) target_elem_size = 1;
    long outputs_size = target_elem_size;
    int is_recurrent = PSIsRecurrent(model), output_is_seq = 0;
    if (is_recurrent || PSUseSequences(model)) {
        if (is_recurrent) output_is_seq = PSIsRecurrent(output_layer);
        else output_is_seq = PSHandleSequenceAtOnce(output_layer);
    }
    long num_states = 1, resized_outputs = 0;
    PSFloat *tmpy = NULL;
    PSFloat *tmpoutputs = NULL;
    if (output_is_seq) {
        num_states = PSStateSequenceLength(output_layer);
        assert(num_states > 0);
        if (num_states != y_seqlen) {
            PSModel *outmodel = output_layer->model;
            int teacher_forcing = (
                opts->flags & PS_TRAINING_FLAG_TEACHER_FORCING &&
                outmodel->previous != NULL
            );
            if (!teacher_forcing) {
                PSErrNN(NULL, model, NULL, "sequence length differs from "
                        "hidden states count but teacher forcing is not "
                        "enabled");
                ok = 0;
                goto final;
            }
            if (backprop) {
                long tmpy_len = 0;
                tmpy = prepareSequence(
                    y, y_seqlen, num_states, target_elem_size, &tmpy_len,
                    &(outmodel->sequence_settings), SEQ_MODE_APPEND_END
                );
                ok = (tmpy != NULL && num_states == tmpy_len);
                if (!ok) {
                    PSErrNN(
                        __func__, model, NULL, "cannot prepare target sequence "
                        "for teacher forcing (hidden states = %d, target seq. "
                        "length = %d, original seq. length = %d)", num_states,
                        tmpy_len, y_seqlen
                    );
                    goto final;
                }
                y = tmpy;
                y_seqlen = num_states;
            }
            if (outputs != NULL) {
                if (y_seqlen > num_states) {
                    PSFloat *resized = calloc(
                        y_seqlen, target_elem_size * sizeof(PSFloat)
                    );
                    ok = (resized != NULL);
                    if (!ok) {
                        PSPrintMemoryErrorMsg();
                        goto final;
                    }
                    PSVectorCopy(
                        resized, outputs, target_elem_size * num_states
                    );
                    resized_outputs = 1;
                    outputs = resized;
                    tmpoutputs = outputs;
                } else if (num_states > y_seqlen) {
                    PSFloat *resized = calloc(
                        num_states, target_elem_size * sizeof(PSFloat)
                    );
                    ok = (resized != NULL);
                    if (!ok) {
                        PSPrintMemoryErrorMsg();
                        goto final;
                    }
                    PSVectorCopy(
                        resized, y, target_elem_size * y_seqlen
                    );
                    if (y == tmpy) free(tmpy);
                    y = resized;
                    tmpy = y;
                }
            }
        }
        outputs_size *= y_seqlen;
    }
    if (outputs == NULL) {
        tmpoutputs = malloc(outputs_size * sizeof(PSFloat));
        outputs = tmpoutputs;
    }
    for (i = 0; i < outputs_size; i++) {
        if (onehot) {
            long idx = (long) *(y + i);
            PSFloat state;
            if (i >= num_states) {
                ok = resized_outputs && i < y_seqlen;
                if (!ok) {
                    PSErrNN(__func__, model, NULL, "index %d is out-of-bounds");
                    goto final;
                }
                state = 0;
            } else state = PSGetState(output_layer, idx, i);
            outputs[i] = state;
        } else {
            if (!output_is_seq) outputs[i] = PSGetState(output_layer, i);
            else i = fetchSequenceOutputState(output_layer, outputs, i, 0);
        }
    }
    if (opts->l1_decay != 0)
        l1_loss *= (opts->l1_decay / batch_size);
    if (opts->l2_decay != 0)
        l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    long onehot_size = (onehot ? output_layer->size : 0);
    PSFloat loss = output_model->loss(
        outputs, y, outputs_size, onehot_size
    );
    if (output_is_seq && y_seqlen > 0) loss /= y_seqlen;
final:
    free(tmpy);
    free(tmpoutputs);
    if (!ok) {
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return PS_STATUS_ERROR_LOSS;
    }
    if (net_loss != NULL) *net_loss = loss;
    return loss + l1_loss + l2_loss;
}

void updateTrainingAccuracy(PSModel *model, PSFloat *outputs, PSFloat *targets,
                            long seqlen)
{
    if (model == NULL) return;
    if (PSIsModelChain(model)) model = PSModelChainHead(model);
    if (model == NULL) return;
    if (model->training == NULL) return;
    PSLayer *outlayer = PSGetOutputLayer(model);
    assert(outlayer != NULL);
    PSMathOpts mopts = {.acceleration = model->acceleration};
    int onehot = (outlayer->flags & PS_FLAG_ONEHOT), i;
    if (seqlen < 1) seqlen = 1;
    for (i = 0; i < seqlen; i++) {
        long predict_idx = 0, correct_idx = 0;
        PSVectorMax(outputs, &predict_idx, outlayer->size, &mopts);
        if (onehot) correct_idx = (long) *(targets++);
        else {
            PSVectorMax(targets, &correct_idx, outlayer->size, &mopts);
            targets += outlayer->size;
        }
        model->training->tot_results++;
        if (correct_idx == predict_idx) model->training->correct_results++;
        outputs += outlayer->size;
    }
}

/* Iterate over a single batch of training examples (`training_data`) and
 * obtain  batch's gradients by back-propagation on each element of the batch
 * itself (by calling the `backprop` function).
 * Then, gradients are applied on model's parameters (weights and biases)
 * with the specified learing rate and eventually optimized by weight decay
 * (L1, L2, Weight decay) and the specified optimization method (ie. AdaGrad,
 * AdaDelta, etc.).
 * The function will return the calculated error (loss). */
PSFloat updateModelParameters(PSModel *model,
                              PSFloat *training_data,
                              long num_examples,
                              PSFloat rate, PSTrainingOptions* opts, ...)
{
    static PSTrainingOptions dfopts = {0};
    assert(model->previous == NULL);
    if (opts == NULL) opts = &dfopts;
    int i, j, apply_clip = 0;
    long gradsize = 0, midx = 0, x_seqlen = 0, y_seqlen = 0,
         batch_size = opts->batch_size, iteration = 0;
    if (batch_size <= 0) batch_size = 1;
    PSFloat *x = NULL; /* Inputs */
    PSFloat *y = NULL; /* Tragets */
    PSFloat l1 = 0.0, l2 = 0.0, l1_loss = 0.0, l2_loss = 0.0,
            clip_max = 0.0, clip_min = 0.0;
    PSModel *output_model = model;
    int num_models = PSModelChainLength(model);
    if (num_models < 1) {
        PSErr(NULL, "broken model chain");
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return PS_STATUS_ERROR_LOSS;
    } else if (num_models > 1) {
        output_model = PSModelChainTail(model);
        if (output_model == NULL) {
            PSErr(NULL, "broken model chain");
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            return PS_STATUS_ERROR_LOSS;
        }
    }
    PSLayer *output_layer = PSGetOutputLayer(model);
    if (output_layer == NULL) {
        PSErr(NULL, "no output layer");
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return PS_STATUS_ERROR_LOSS;
    }
    long training_data_size = model->input_size;
    long label_data_size = output_layer->size;
    /* Create gradients for the current batch. */
    PSGradient ***gradients = createGradients(model);
    if (gradients == NULL) {
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return PS_STATUS_ERROR_LOSS;
    }
    PSGradient ***bp_gradients = NULL;
    PSFloat **sequences = NULL;
    int is_recurrent = PSIsRecurrent(model);
    if (is_recurrent || PSUseSequences(model)) {
        va_list args;
        va_start(args, opts);
        sequences = va_arg(args, PSFloat**);
        va_end(args);
        if (sequences == NULL) {
            PSErr(__func__, "Sequences argument is NULL");
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            goto final;
        }
    }
    UNUSED(num_examples); /* TODO: remove num_examples arg if not needed */
    PSOptimization optimization = PSSGDOptimization;
    int use_weight_decay = 0, divide_grads_by_batches = 0, training_flags = 0;
    if (opts != NULL) {
        training_flags = opts->flags;
        optimization = opts->optimization;
        if (optimization == NULL) optimization = PSSGDOptimization;
        divide_grads_by_batches =  (
            optimization == PSAdaDeltaOptimization ||
            optimization == PSWindowGradOptimization ||
            optimization == PSAdaGradOptimization ||
            optimization == PSAdamOptimization
        );
        use_weight_decay = (opts->flags & PS_TRAINING_WEIGHT_DECAY);
        l1 = opts->l1_decay;
        l2 = opts->l2_decay;
        if ((apply_clip = (opts->clip != 0.0))) {
            clip_max = PSAbs(opts->clip);
            clip_min = clip_max * -1;
        }
    }
    int required_memory_gradients = getRequiredMemoryGradientsCount(opts);
    if (!divide_grads_by_batches) rate /= batch_size;
    else divide_grads_by_batches = (batch_size > 1);
    /* If weight decay is enabled (PS_TRAINING_WEIGHT_DECAY flag), l2_decay
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
    /* Iterate the examples of the batch and, for each example, get gradients
     * from the backpropagation of the error. Then, sum the backpropagation
     * gradients to the batch's gradients. */
    for (i = 0; i < batch_size; i++) {
        long cur_example = 0;
        if (model->training != NULL) {
            model->training->current_example =
                (model->training->current_batch * batch_size) + i;
            iteration = model->training->current_example + 1;
            cur_example = model->training->current_example;
        }
        /* Backpropagate the error through the model layers and get
         * gradients for the current element. */
        if (sequences == NULL) {
            /* Non-recurrent and non-sequence model */
            long example_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += example_size;
        } else {
            /* Recurrent model or model handling sequences */
            x = sequences[i];
            long datalen = parseSequenceData(
                model, x, cur_example, training_flags, 1,
                &x_seqlen, NULL, &y_seqlen, &y
            );
            if (datalen <= 0 || x_seqlen <= 0 || y_seqlen <= 0) {
                PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                goto final;
            }
        }
        bp_gradients = backprop(model, x, y, opts, bp_dest_gradients);
        if (bp_gradients == NULL) {
            PSErr(
                NULL, "Backpropagation failed for model '%s'",
                (model->name != NULL ? model->name : "UNNAMED")
            );
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
                    PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                    goto final;
                }
                cur = cur->next;
            }
        }
        if (PSDumpGradientsPath != NULL)
            PSDumpGradients(model, gradients, NULL, opts);
        if (PSModelGetStatus(model) == PS_STATUS_PAUSED) break;
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
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            goto final;
        }
        PSTrainingContext *training_ctx = getTrainingContext(cur);
        if (training_ctx == NULL) {
            if (!initTrainingContext(cur, opts, required_memory_gradients)) {
                PSErr(NULL, "could not initialize training context on model "
                      "%d", cur->index);
                PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
                PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                return PS_STATUS_ERROR_LOSS;
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
            if (layer->flags & PS_FLAG_NON_TRAINABLE) continue;
            /* Update Biases */
            if (!(layer->flags & PS_FLAG_NO_BIAS)) {
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
                    PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                    goto final;
                }
            }
            /* Update Weights */
            long wgrad_offset = 0;
            int partially_trainable = (layer->type == Attention);
            for (j = 0; j < layer->weight_types; j++) {
                PSMatrix weights = layer->weights[j];
                if (weights == NULL) {
                    if (partially_trainable) continue;
                    PSErr(
                        __func__, "Layer[%d]: weights[%d] is NULL",
                        layer->index, j
                    );
                    PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                    goto final;
                }
                long wlen = PSMatrixLength(weights);
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
                        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
                    PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
    if (PSModelGetStatus(model) == PS_STATUS_ERROR) return PS_STATUS_ERROR_LOSS;
    return getLoss(model, y, y_seqlen, 1, NULL, l1_loss,l2_loss, NULL, opts);
}

/* Iterate training data for the entire epoch. Unless the training flag
 * PS_TRAINING_NO_SHUFFLE is set, training data is randomly shuffled
 * (Stochastic Gradient Descent). Training data is divided into batches
 * depending on `batch_size` and, for each batch, gradients are generated
 * and used to update model's parameters using `updateModelParameters`. */
PSFloat trainEpoch(PSModel *model,
                   PSFloat *training_data,
                   long example_size,
                   long num_examples,
                   PSFloat learning_rate,
                   PSTrainingOptions *options,
                   int epochs, float *training_accuracy)
{
    static PSTrainingOptions dfopts = {0};
    PSTrainingContext *training_ctx = getTrainingContext(model);
    if (training_ctx == NULL) {
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return PS_STATUS_ERROR_LOSS;
    }
    if (options == NULL) options = &dfopts;
    if (training_accuracy != NULL) *training_accuracy = 0;
    PSTrainingProgressFunc printProgress = NULL;
    int flags = options->flags, is_model_chain = PSIsModelChain(model);
    long batch_size = options->batch_size;
    if (batch_size <= 0) batch_size = 1;
    long batch_count = num_examples / batch_size;
    PSFloat **sequences = NULL, **sequence_head = NULL;
    if (PSIsRecurrent(model) || PSHandleSequenceAtOnce(model)) {
        PSLayer *out = PSGetOutputLayer(model);
        if (out == NULL) {
            PSErr(NULL, "no output layer");
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            return PS_STATUS_ERROR_LOSS;
        }
        PSModel *output_model = model;
        if (is_model_chain) {
            output_model = PSModelChainTail(model);
            if (output_model == NULL) {
                PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                return PS_STATUS_ERROR_LOSS;
            }
        }
        sequences = getDatasetSequences(
            model, training_data, num_examples, flags
        );
        if (sequences == NULL) {
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            return PS_STATUS_ERROR_LOSS;
        }
        if (!(flags & PS_TRAINING_NO_SHUFFLE))
            shuffleSequences(sequences, num_examples);
    } else {
        if (!(flags & PS_TRAINING_NO_SHUFFLE))
            shuffle(training_data, num_examples, example_size);
    }
    PSFloat loss = 0.0, avg_loss = 0.0;
    float accuracy = 0.0;
    float *accuracy_p = NULL;
    long tot_t = 0, avg_t, elapsed_t;
    long step_size = (example_size * batch_size), i;
    int measure_accuracy = options->metrics & PS_TRAINING_METRICS_ACCURACY;
    int min_acc_results = 10;
    if (measure_accuracy) {
        model->training->tot_results = 0;
        model->training->correct_results = 0;
        accuracy_p = &accuracy;
    }
    printProgress = options->printProgress;
    if (printProgress == NULL) printProgress = PSLogTrainingProgress;
    sequence_head = sequences;
    /* Iterate all epoch's batches. */
    for (i = 0; i < batch_count; i++) {
        model->training->current_batch = i;
        long batch_num = i + 1;
        /* Update model's parameters and get loss for current batch. */
        struct timeval st, et;
        gettimeofday(&st, NULL);
        PSFloat batch_loss = updateModelParameters(
            model, training_data, num_examples, learning_rate,
            options, sequence_head
        );
        if (PSModelGetStatus(model) == PS_STATUS_ERROR) {
            PSErr(NULL, "Gradient descent failed at batch %d for model '%s'",
                  i, (model->name != NULL ? model->name : "UNNAMED")
            );
            goto final;
        }
        loss += batch_loss;
        if (measure_accuracy && model->training->tot_results>=min_acc_results) {
            /* Measure training accuracy. */
            accuracy = (float) model->training->correct_results /
                       (float) model->training->tot_results;
        }
        if (batch_num < batch_count) {
            avg_loss = loss / (PSFloat) batch_num;
            long *elapsed_p = (i > 0 ? &avg_t : NULL);
            printProgress(
                model, PS_STATUS_TRAINING, epochs, batch_count,
                &avg_loss, accuracy_p, NULL, NULL, elapsed_p
            );
        } else {
            printProgress(
                model, PS_STATUS_TRAINING, epochs, batch_count, NULL, NULL,
                NULL, NULL, NULL
            );
        }
        if (model->onBatchTrained != NULL) {
            model->onBatchTrained(
                model, model->training->current_epoch, epochs, avg_loss,
                batch_loss, 0, accuracy, 0, &learning_rate,
                (sequences != NULL ? *sequence_head : training_data)
            );
        }
        if (sequences == NULL) training_data += step_size;
        else sequence_head += batch_size;
        int action = model->training->requested_action;
        if (action == PS_ACTION_ABORT) {
            PSModelSetStatus(model, action, NULL);
            break;
        }
        gettimeofday(&et, NULL);
        elapsed_t = PSGetElapsedTimeUS(st, et);
        tot_t += elapsed_t;
        avg_t = (tot_t / batch_num);
    }
final:
    if (sequences != NULL) free(sequences);
    if (measure_accuracy && training_accuracy != NULL)
        *training_accuracy = accuracy;
    return loss / (PSFloat) batch_count;
}

float validate(PSModel *model, PSFloat *test_data, long data_size,
               PSTrainingOptions *opts, PSFloat *loss, int log)
{
    long i, j;
    int previous_status = PSModelGetStatus(model);
    char *errmsg = NULL;
    float accuracy = 0.0f;
    long correct_results = 0, tot_results = 0;
    PSLayer *output_layer = PSGetOutputLayer(model);
    long input_size = model->input_size;
    long output_size = model->output_size;
    int onehot = output_layer->flags & PS_FLAG_ONEHOT;
    long y_size = (onehot ? 1 : output_size);
    long example_size = input_size + output_size;
    long num_examples;
    int emits_output_sequence = 0;
    int flags = (opts != NULL ? opts->flags : 0);
    int teacher_forcing = 0;
    long steps_to_check = 0, max_seqlen = 0;
    int print_progress = (
        log == VALIDATE_LOG_PROGRESS && opts != NULL &&
        opts->printProgress != NULL && model->training != NULL
    );
    log = (log == VALIDATE_LOG_ALL);
    PSFloat *outputs = NULL;
    PSFloat *tmpy = NULL;
    PSModel *output_model = NULL;
    PSFloat **sequences = NULL;
    if (PSUseSequences(model)) {
        /*  First training data element for sequence datasets must indicate */
        /*  the number fo sequences in the dataset itself. */
        num_examples = (long) *(test_data++);
        data_size--;
        emits_output_sequence = PSUseSequences(output_layer);
        sequences = getDatasetSequences(
            model, test_data, num_examples, flags
        );
        if (sequences == NULL) goto err;
        if (PSIsModelChain(model) && PSModelChainLength(model) > 1) {
            output_model = PSModelChainTail(model);
            if (output_model == NULL) {
                PSErrNN(NULL, model, NULL, "broken model chain");
                goto err;
            }
            teacher_forcing = (
                flags & PS_TRAINING_FLAG_TEACHER_FORCING &&
                PSUseSequences(output_model)
            );
        }
    } else {
        num_examples = data_size / example_size;
        tot_results = num_examples;
    }
    PSForwardOptions fwopts = {0};
    if (flags & PS_TRAINING_FLAG_AUTOREGRESSION)
        fwopts.flags |= PS_TRAINING_FLAG_AUTOREGRESSION;
    if (log) printf("Test data examples: %ld\n", num_examples);
    if (model->training != NULL) {
        model->training->test_size = data_size;
        model->training->num_tests = num_examples;
    }
    PSModelSetStatus(model, PS_STATUS_VALIDATING, NULL);
    time_t start_t, end_t;
    char timestr[80];
    struct tm *tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    PSFloat tot_loss = 0.0;
    if (log) PSInfo("Testing started at %s", timestr);
    for (i = 0; i < num_examples; i++) {
        if (model->training != NULL) model->training->current_test = i;
        if (log) printf("\rTesting %ld/%ld", i + 1, num_examples);
        else if (print_progress) opts->printProgress(
            model, PS_STATUS_VALIDATING, opts->epochs,
            model->training->current_batch + 1, NULL, NULL, NULL, NULL, NULL
        );
        fflush(stdout);
        PSFloat *inputs = NULL;
        PSFloat *targets = NULL;
        long seqlen = 0;
        if (sequences == NULL) {
            /*  Non Recurrent and no sequences*/
            inputs = test_data;
            test_data += input_size;
            targets = test_data;

            int ok = forward(model, inputs, 0, &fwopts);
            if (!ok) goto err;

            long omax = 0; /* Output index with max value */
            long emax = 0; /* Target index with max value */
            if (!PSFindLayerMaxState(output_layer, NULL, &omax)) {
                PSErr(NULL, "Could not find output layer max state");
                goto err;
            }
            if (!onehot) emax = arrayMaxIndex(targets, output_size);
            else emax = (long) *(targets);
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            /*  Recurrent or sequences*/
            inputs = sequences[i];
            long seq_datalen = parseSequenceData(
                model, inputs, i, flags, 1, NULL, NULL, &seqlen, &targets
            );
            if (seqlen == 0 || seq_datalen <= 0) {
                errmsg = "recurrent data with zero seqlen";
                goto err;
            }
            int ok = PSResetModelStateSequences(model, seqlen, 0);
            if (!ok) goto err;
            ok = forward(model, inputs, 0, &fwopts);
            if (!ok) goto err;

            long correct_states = 0;
            if (emits_output_sequence) {
                long output_seqlen = PSStateSequenceLength(output_layer);
                max_seqlen = steps_to_check = seqlen;
                long tmpy_len = 0;
                if (teacher_forcing) {
                    tmpy = prepareSequence(
                        targets, seqlen, seqlen, y_size, &tmpy_len,
                        &(output_model->sequence_settings), SEQ_MODE_APPEND_END
                    );
                    if (tmpy == NULL) {
                        PSErrNN(__func__, model, NULL, "cannot prepare target "
                                "sequence for teacher forcing validation");
                        goto err;
                    }
                    targets = tmpy;
                    seqlen = tmpy_len;
                    steps_to_check = max_seqlen = seqlen;
                }
                if (output_seqlen < seqlen)
                    steps_to_check = output_seqlen;
                else if (output_seqlen > seqlen)
                    max_seqlen = output_seqlen;
                long label_data_size = y_size * steps_to_check;
                long last_label_idx = (label_data_size - 1);
                if (label_data_size <= 0) goto err;
                outputs = PSVectorCreate(label_data_size);
                if (outputs == NULL) {
                    PSPrintMemoryErrorMsg();
                    goto err;
                }
                for (j = 0; j < label_data_size; j++) {
                    int is_last_label = (j == last_label_idx);
                    j = fetchSequenceOutputState(
                        output_layer, outputs, j, onehot
                    );
                    if (onehot && (outputs[j] == targets[j]))
                        correct_states++;
                    else if (
                        !onehot && j > 0 &&
                        ((j % y_size) == 0 || is_last_label)
                    ) {
                        long t = ((j + 1) / y_size) - 1;
                        long omax = arrayMaxIndex(
                            outputs + (t * y_size), y_size
                        );
                        long emax = arrayMaxIndex(
                            targets + (t * y_size), y_size
                        );
                        if (emax == omax) correct_states++;
                    }
                }
                correct_results += correct_states;
                tot_results += max_seqlen;
            } else {
                long omax = 0; /* Output index with max value */
                long emax = 0; /* targets index with max value */
                if (!PSFindLayerMaxState(output_layer, NULL, &omax)) {
                    PSErr(NULL, "Could not find output layer max state");
                    goto err;
                }
                if (!onehot) emax = arrayMaxIndex(targets, output_size);
                else emax = (long) *targets;
                if (omax == emax) correct_results++;
            }
        }
        if (loss != NULL) {
            tot_loss += getLoss(
                model, targets, seqlen, 0, outputs, 0, 0, NULL, opts
            );
            if (PSModelGetStatus(model) == PS_STATUS_ERROR) goto err;
        }
    }
    if (log) {
        printf("\n");
        fflush(stdout);
    }
    time(&end_t);
    if (log) printf("\nCompleted in %ld sec.\n", end_t - start_t);
    accuracy = (float) correct_results / (float) tot_results;
    if (log) {
        printf(
            "Accuracy (%ld/%ld): %.2f\n", correct_results,
            tot_results, accuracy
        );
    }
    if (emits_output_sequence) free(sequences);
    PSModelSetStatus(model, previous_status, NULL);
    if (loss != NULL) {
        *loss = tot_loss / num_examples;
        if (log) printf("Loss: %.2f\n", *loss);
    }
    free(outputs);
    free(tmpy);
    return accuracy;
err:
    free(outputs);
    free(tmpy);
    if (errmsg == NULL) {
        if (PSModelGetStatus(model) == PS_STATUS_VALIDATING)
            errmsg = "an error occurred while validating, aborting!";
        else
            errmsg = "failed to validate model";
    }
    PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
    fprintf(stderr, "\n");
    PSErr(NULL, "%s", errmsg);
    return PS_STATUS_ERROR_LOSS;
}

static void checkTrainingOptions(PSTrainingOptions *options) {
    if (options->optimization == NULL)
        options->optimization = PSSGDOptimization;
    if (options->optimization != PSSGDOptimization) {
        if (options->eps == 0) options->eps = PS_DEFAULT_EPS;
        if (options->rho == 0) options->rho = PS_DEFAULT_RHO;
        if (options->beta1 == 0) options->beta1 = PS_DEFAULT_BETA1;
        if (options->beta2 == 0) options->beta2 = PS_DEFAULT_BETA2;
    }
    if (options->printProgress == NULL)
        options->printProgress = PSLogTrainingProgress;
}

void PSSetDefaultTrainingOptions(PSTrainingOptions *options) {
    options->rho = PS_DEFAULT_RHO;
    options->eps = PS_DEFAULT_EPS;
    options->beta1 = PS_DEFAULT_BETA1;
    options->beta2 = PS_DEFAULT_BETA2;
    options->bptt_truncate = BPTT_TRUNCATE;
    options->optimization = PSSGDOptimization;
    if (options->epochs <= 0) options->epochs = 1;
    if (options->batch_size <= 0) options->batch_size = 1;
    if (options->printProgress == NULL)
        options->printProgress = PSLogTrainingProgress;
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
    if (!(output_model->flags & PS_FLAG_AUTOREGRESSION)) {
        if (err != NULL)
            *err = "no PS_FLAG_AUTOREGRESSION in output model's flags";
        return 0;
    }
    return 1;
}

int PSPretrainLayers(PSModel *model, PSFloat *training_data,
                     long data_size)
{
    if (model->flags & PS_FLAG_PRETRAINER) return 1;
    int original_status = PSModelGetStatus(model);
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer == NULL) continue;
        if (!isPretrainableLayer(layer)) continue;
        if (layer->pretrained) continue;
        PSInfo("Pretraining layer[%d] (%s)", i, PSGetLayerTypeLabel(layer));
        PSModelSetStatus(model, PS_STATUS_PRETRAINING, NULL);
        int trained = layer->pretrain(layer, training_data, data_size);
        if (!trained) {
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
 * `options` argument.
 * Arguments:
 *  - `model`: The neural model to be trained (mandatory)
 *  - `training_data`: an array of `PSFloat` containing the tarining dataset
 *     (ie. inputs, expected predictions)
 *  - `data_size`: length of `training_data` array.
 *  - `test_data`: optional dataset that can be used for testing (validation).
 *  - `test_size`: length of `test_data` array.
 *  - `options`: optional training options (see `PSTrainingOptions`).
 *               If NULL, the training process will use default options.
 * Training/test data layout:
 *  - For normal feedforward models, the array must contain alternating
 *    inputs/predictions pairs, one pair for each example in the dataset.
 *    So, each training/test example pair must contain:
 *      - Input values, having the same length of the model's input layer
 *      - Target values, having the same length of the model's output
 *        layer. If output layer has the `PS_FLAG_ONEHOT` flag, predictions
 *        length muse be 1, and it must contain the index of the expected
 *        maximum state.
 *    The total number of training examples is given by:
 *      array size / (input_size + output_size)
 *  - For recurrent model or models using sequences, the layout of the dataset
 *    can have different forms.
 *    Regardless of that, the first element of the array must contain the total
 *    number of training/test sequences.
 *    For each training/test sequence, the sequence length must be specified.
 *    Different forms can be:
 *    - Many-to-many: the default mode for recurrent models that produce
 *      sequences having the same length of the input sequence.
 *      In this case, the first element of the sequence segment is the
 *      sequence length, followed by inputs/predictions pair.
 * If some error occurs, `PS_STATUS_ERROR` will be set on `model` and the
 * function will immediately exit.
 * If `model` is not built, the function will automatically try to build it by
 * calling `PSModelBuild`.
 * Possible failure reasons:
 *  - `model` is NULL
 *  - `model` is not built and it cannot be build.
 *  - The learning rate is negative.
 *  - `model` is part of a multi-model chain but the chain is broken or invalid.
 *  - `PS_TRAINING_FLAG_SEQ2SEQ` is set into flags of `options` but the model's
 *    architecture is not valid for sequence-to-sequence mode (ie. the model
 *    does not use sequences at all).
 */
void PSTrain(PSModel *model,
             PSFloat *training_data,
             long data_size,
             PSFloat *test_data,
             long test_size,
             PSTrainingOptions *options)
{
    int epochs = 0, i;
    long batch_size = 0, num_examples = 0, num_test_examples = 0;
    PSFloat learning_rate = 0.0;
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
            return;
        }
    }
    int valid = PSModelCheck(model);
    if (!valid) {
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return;
    }
    PSModelContext *ctx = getModelContext(model);
    if (ctx == NULL) {
        PSErr(__func__, "missing model context");
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        return;
    }
    PSTrainingContext *training_ctx = ctx->training_context;
    if (training_ctx && PSModelGetStatus(model) == PS_STATUS_UNTRAINED) {
        deleteTrainingContext(training_ctx, model);
        ctx->training_context = training_ctx = NULL;
    }
    if (training_ctx == NULL) {
        ctx->training_context = calloc(1, sizeof(PSTrainingContext));
        if (ctx->training_context == NULL) {
            PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
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
    PSTrainingProgressFunc printProgress = options->printProgress;
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
    long input_size = input_model->input_size,
         output_size = output_model->output_size;
    long example_size = input_size + output_size;
    /* Eventually pretrain layers (if pretrainable layers are found) */
    if (!PSPretrainLayers(model, training_data, data_size)) {
        PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
        PSErr(__func__, "failed to pretrain model '%s'",
              model->name != NULL ? model->name : "UNNAMED");
        return;
    }
    float training_accuracy = 0;
    float *training_accuracy_p = NULL;
    int is_recurrent = PSIsRecurrent(model);
    int handle_seq = PSHandleSequenceAtOnce(model);
    int use_sequences = is_recurrent || handle_seq;
    if (use_sequences) {
        /* First training data element for sequence datasets must
         * indicate the number fo sequences in the dataset itself. */
        num_examples = (long) *(training_data++);
        data_size--;
        if (test_data != NULL) num_test_examples = (long) *(test_data);
    } else {
        num_examples = data_size / example_size;
        if (test_data != NULL) num_test_examples = test_size / example_size;
    }
    if (options->flags & PS_TRAINING_FLAG_SEQ2SEQ) {
        char *err = NULL;
        if (!isSequence2SequenceAvailable(model, &err)) {
            PSErr(__func__, "Sequence-to-sequence (PS_TRAINING_FLAG_SEQ2SEQ) "
                  "is not available for this model: %s", err);
            return;
        }
    }
    int measure_accuracy = (options->metrics & PS_TRAINING_METRICS_ACCURACY);
    if (measure_accuracy) training_accuracy_p = &training_accuracy;
    const char *name = model->name != NULL ? model->name : "UNNAMED";
    if (num_models == 1)
        PSNotice("Training model \"%s\"\n", name);
    else {
        PSNotice("Training multi-model model\n");
        PSInfo("Number of models:           %d", num_models);
        const char *input_name = (
            input_model->name != NULL ? input_model->name : "UNNAMED"
        );
        const char *output_name = (
            output_model->name != NULL ? output_model->name : "UNNAMED"
        );
        PSInfo("Input model:                \"%s\"", input_name);
        PSInfo("Output model:               \"%s\"", output_name);
    }
    PSInfo("Training data examples:     %d", num_examples);
    if (test_data != NULL && num_test_examples > 0)
        PSInfo("Test data examples:         %d", num_test_examples);
    PSInfo("Batch Size:                 %d", batch_size);
    PSInfo("Learning Rate:              %g", learning_rate);
    int use_weight_decay = (
        (options->l1_decay != 0 || options->l2_decay != 0) &&
        (options->flags & PS_TRAINING_WEIGHT_DECAY)
    );
    PSInfo("L1 Decay:                   %g", options->l1_decay);
    PSInfo("L2 Decay:                   %g", options->l2_decay);
    PSInfo("Weight Decay:               %s",
           (use_weight_decay ? "yes" : "no"));
    PSInfo("Clip:                       %g", PSAbs(options->clip));
    PSInfo("Momentum:                   %g", options->momentum);
    PSInfo("Optimization:               %s",
           getOptimizationName(options->optimization));
    int single_seq = (options->flags & PS_TRAINING_EPOCH_AS_SEQUENCE),
        no_shuffle = (options->flags & PS_TRAINING_NO_SHUFFLE);
    if (single_seq && !no_shuffle) {
        PSWarn(
            "flag PS_TRAINING_EPOCH_AS_SEQUENCE requires "
            "PS_TRAINING_NO_SHUFFLE. "
            "Automatically enabling PS_TRAINING_NO_SHUFFLE."
        );
        options->flags |= PS_TRAINING_NO_SHUFFLE;
    }
    if (single_seq) PSInfo("Single sequence:            yes");
    PSInfo("Data shuffle:               %s", (!no_shuffle ? "yes" : "no"));
    if (is_recurrent)
        PSInfo("BPTT Truncate:              %d", options->bptt_truncate);
    if (model->layers[model->size - 1]->flags & PS_FLAG_ONEHOT)
        PSInfo("Onehot Labels:              yes");
    if (options->flags & PS_TRAINING_FLAG_TEACHER_FORCING)
        PSInfo("Teacher forcing:            yes");
    char *loss_func_name = NULL;
    if (output_model->loss != NULL) {
        loss_func_name = getLossFunctionName(output_model->loss);
        PSInfo("Loss Function:              %s", loss_func_name);
    }
    if (measure_accuracy)
        PSPrint(PSLOGLEVEL_INFO, "Metrics:                    accuracy\n");
    int was_paused = (PSModelGetStatus(model) == PS_STATUS_PAUSED);
    /* Start training */
    PSModelSetStatus(model, PS_STATUS_TRAINING, NULL);
    time_t start_t, end_t;
    char timestr[80];
    struct tm *tminfo;
    struct timeval epoch_st, epoch_et;
    PSFloat prev_loss = 0.0;
    float acc = -999.99f;
    int adjust_rate = adjust_rate = (options->flags & PS_TRAINING_ADJUST_RATE);
    int first_epoch = 0;
    if (model->training != NULL) {
        if (was_paused) first_epoch = model->training->current_epoch;
    } else {
        model->training = malloc(sizeof(PSTrainingInfo));
        if (options != NULL)
            model->training->debug_dump_to = options->debug_dump_to;
        else model->training->debug_dump_to = NULL;
        if (model->training->debug_dump_to) PSTrainingDebugDumpHeader(
            model, data_size, test_size, epochs, learning_rate, batch_size
        );
    }
    model->training->num_examples = num_examples;
    model->training->batch_size = batch_size;
    model->training->requested_action = PS_ACTION_NONE;
    model->training->data_size = data_size;
    model->training->test_size = test_size;
    model->training->current_test = 0;
    model->training->num_tests = 0;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    PSPrint(PSLOGLEVEL_NOTICE, "Training started at %s\n", timestr);
    model->training->started_at = start_t;
    model->training->ended_at = (time_t) 0;
    for (i = first_epoch; i < epochs; i++) {
        PSFloat test_loss = 0.0;
        training_accuracy = 0;
        model->training->current_epoch = i;
        if (is_recurrent) {
            if (!PSResetModelStateSequences(model, 0, 0)) {
                PSModelSetStatus(model, PS_STATUS_ERROR, NULL);
                PSErr(__func__, "Failed to reset model recurrent states");
                return;
            }
        }
        gettimeofday(&epoch_st, NULL);
        PSFloat loss = trainEpoch(
            model, training_data, example_size, num_examples, learning_rate,
            options, epochs, &training_accuracy
        );
        gettimeofday(&epoch_et, NULL);
        time_t elapsed_t = PSGetElapsedTimeUS(epoch_st, epoch_et);
        if (PSModelGetStatus(model) == PS_STATUS_ERROR) {
            PSPrint(
                PSLOGLEVEL_ERROR, "\nAn error occurred while training, "
                "aborting!\n"
            );
            return;
        }
        long batches_count = num_examples / batch_size;
        float *acc_p = NULL;
        if (test_data  && PSModelGetStatus(model) == PS_STATUS_TRAINING) {
            model->training->current_test = 0;
            printProgress(
                model, PS_STATUS_VALIDATING, epochs, batches_count, NULL,
                NULL, NULL, NULL, NULL
            );
            acc = validate(
                model, test_data, test_size, options, &test_loss,
                VALIDATE_LOG_PROGRESS
            );
            acc_p = &acc;
        }
        if (i > 0 && loss > prev_loss && adjust_rate)
            learning_rate *= 0.5;
        if (model->onEpochTrained != NULL) {
            model->onEpochTrained(
                model, i, epochs, loss, loss, test_loss, training_accuracy,
                acc, &learning_rate, NULL
            );
        }
        prev_loss = loss;
        printProgress(
            model, PS_STATUS_TRAINING, epochs, batches_count, &loss,
            training_accuracy_p, &test_loss, acc_p, &elapsed_t
        );
        fflush(stdout);
        int action = model->training->requested_action;
        if (action == PS_ACTION_ABORT || action == PS_ACTION_PAUSE) {
            PSModelSetStatus(model, action, NULL);
            break;
        }
    }
    time(&end_t);
    printProgress(
        model, PS_STATUS_TRAINED, epochs, 0, NULL, NULL, NULL, NULL, NULL
    );
    PSLineEnd();
    fflush(stdout);
    PSPrint(PSLOGLEVEL_SUCCESS, "\nCompleted in %ld sec.\n", end_t - start_t);
    model->training->ended_at = end_t;
    if (PSModelGetStatus(model) == PS_STATUS_TRAINING)
        PSModelSetStatus(model, PS_STATUS_TRAINED, NULL);
    if (is_recurrent) {
        PSModel *current = input_model;
        while (current != NULL) {
            PSResetModelStateSequences(current, 0, 0);
            current = current->next;
        }
    }
}

/* Test `model` the against `test_data` dataset having length defined by the
 * `data_size` argument.
 * Tests are usualy performed on a different dataset than the one used for
 * training in order to measure how the model performs on different data.
 * This can be useful to determine undefitting (the model is not sufficiently
 * trained) or overfitting (the model has been trained to much on the training
 * dataset and it cannot generalize its predictions to different examples).
 * Underfitting generally leads to lower performances in the training data,
 * while overfitting generally leads to better performances on the training
 * dataset than on the one used for testing.
 * The function computes the accuracy of the predictions (the number of correct
 * prediction with respect to the expected targets given by the dataset itself).
 * In addition, the function can also compute the overall loss of the
 * predictions made by using the pointer `loss`.
 * The argument `opts` can be used to set the same training options used for
 * training (ie. the `flags`).
 * Return value: the accuracy of the predictions, where 1.0 means that all
 * predictions were correct while 0.0 means that no prediction was correct. */
float PSTest(PSModel *model, PSFloat *test_data, long data_size,
             PSFloat *loss, PSTrainingOptions *options)
{
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO ? 1 : 0);
    return validate(model, test_data, data_size, options, loss, do_log);
}

void PSPauseTraining(PSModel *model) {
    if (model->training != NULL) {
        printf("\nPause requested, "
               "training will stop after current epoch will be completed.\n");
        model->training->requested_action = PS_ACTION_PAUSE;
    }
}

void PSAbortTraining(PSModel *model) {
    if (model->training != NULL) {
        PSInfo("\nAborting...");
        model->training->requested_action = PS_ACTION_ABORT;
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
        if (RNNLayer == ltype || LSTM == ltype || GRU == ltype)
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
            if (layer->flags & PS_FLAG_ONEHOT) onehot_input = 1;
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
            if (model->flags & PS_FLAG_RECURRENT) {
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
            PSErr(
                __func__,
                "Layer[%d] activate function is PSSigmoid, "
                "but derivative function is not PSSigmoidDerivative", i
            );
            return 0;
        }
        if (layer->activate == PSRelu &&
            layer->derivative != PSReluDerivative)
        {
            PSErr(
                __func__,
                "Layer[%d] activate function is PSRelu, "
                "but derivative function is not PSReluDerivative", i
            );
            return 0;
        }
        if (layer->activate == PSGelu &&
            layer->derivative != PSGeluDerivative)
        {
            PSErr(
                __func__,
                "Layer[%d] activate function is PSGelu, "
                "but derivative function is not PSGeluDerivative", i
            );
            return 0;
        }
        if (layer->activate == PSTanhActivation &&
            layer->derivative != PSTanhDerivative)
        {
            PSErr(
                __func__,
                "Layer[%d] activate function is PSTanhActivation, "
                "but derivative function is not PSTanhDerivative", i
            );
            return 0;
        }
        if (layer == output_layer && (layer->flags & PS_FLAG_ONEHOT)) {
            if (SoftMax != ltype) {
                PSErr(
                    __func__, "output layer with flag PS_FLAG_ONEHOT must "
                    "be a SoftMax layer"
                );
                return 0;
            }
        }
    }
    if (is_recurrent) {
        PSRecurrentNetworkMode rnn_mode = model->rnn_mode;
        if (rnn_mode == NonRecurrent) {
            PSErr(__func__, "Recurrent network mode is NonRecurrent");
            return 0;
        }
        if (recurrent_layers == 0) {
            PSErr(
                __func__, "model is recurrent but has no recurrent layers"
            );
            return 0;
        }
        if (recurrent_type_layers == 0) {
            PSErr(
                __func__, "model is recurrent but has no Recurrent, "
                "LSTM or GRU layers"
            );
            return 0;
        }
        if (first_recurrent_layer == NULL) {
            PSErr(
                __func__, "Recurrent network is missing first recurrent layer"
            );
            return 0;
        }
        if (last_recurrent_layer == NULL) {
            PSErr(
                __func__, "Recurrent network is missing last recurrent layer"
            );
            return 0;
        }
        if (first_recurrent_layer != actual_first_recurrent_layer) {
            PSErr(
                __func__,
                "Recurrent network first recurrent layer should be layer %d, "
                "but model is not updated", actual_first_recurrent_layer
            );
            return 0;
        }
        if (last_recurrent_layer != actual_last_recurrent_layer) {
            PSErr
                (__func__,
                "Recurrent network last recurrent layer should be layer %d, "
                "but model is not updated", actual_last_recurrent_layer
            );
            return 0;
        }
        if (ManyToMany == rnn_mode) {
            if (!recurrent_input && !recurrent_output) {
                PSErr(
                    __func__,
                    "Recurrent network with mode \"%s\" has no recurrent "
                    "input nor recurrent output",
                    PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
        } else if (ManyToOne == rnn_mode) {
            if (!recurrent_input) {
                PSErr(
                    __func__,
                    "Recurrent network with mode \"%s\" has no recurrent "
                    "input", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
            if (recurrent_output) {
                PSErr(
                    __func__,
                    "Recurrent network with mode \"%s\" has recurrent "
                    "output", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
        } else if (OneToMany == rnn_mode) {
            if (recurrent_input) {
                PSErr(
                    __func__,
                    "Recurrent network with mode \"%s\" has recurrent "
                    "input", PSGetRecurrentModeLabel(rnn_mode)
                );
                return 0;
            }
            if (!recurrent_output) {
                PSErr(
                    __func__,
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

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

#include "psyc.h"
#include "activation.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "debug.h"
#include "log.h"

#define PSCalculateConvolutionalSide(s,rs,st,pad) \
    PSFloor(((PSFloat)(s - rs + 2 * pad) / (PSFloat) st) + 1)
#define PSCalculatePoolingSide(s, rs) PSFloor((s - rs) / rs + 1)

#define DumpConvolveStep(x,y,rx,ry,rx2,ry2,prev,fidx,nidx,nx,ny,widx) \
 PSTrainingDebugDumpStep(&dbginfo,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),weight_idx=%d,"\
 "srcline=%d\n",\
 x, y, rx, ry, rx2, ry2,prev->index, prev->index, fidx, nidx, nx, ny, widx,\
 __LINE__)

#define DumpConvolveAVXStep(x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny,\
w,steplen,step,rowlen) \
 PSTrainingDebugDumpStep(&dbginfo,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),weight_idx=%d,"\
 "avx=1,avx_step_len=%d,avx_step=%d,rowlen=%d,srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index,prv->index, fidx, nidx, nx, ny, widx,\
 steplen, step, rowlen, __LINE__)

#define DumpPoolStep(x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny) \
 PSTrainingDebugDumpStep(&dbginfo,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index,prv->index, fidx, nidx, nx, ny, __LINE__)

#define DumpPoolBackpropStep(x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny) \
 PSTrainingDebugDumpStep(&dbginfo, \
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_neuron=%d-%d-%d,"\
 "previous_neuron_pos=(%d,%d),srcline=%d\n", x, y, rx, ry, rx2, ry2,\
 prv->index, fidx, nidx, nx, ny, __LINE__)

#define DumpConvBackpropStep(x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny,w) \
 PSTrainingDebugDumpStep(&dbginfo,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_neuron=%d-%d-%d,"\
 "previous_neuron_pos=(%d,%d),weight_idx=%d,srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index, fidx, nidx, nx, ny, widx, __LINE__)

#define UNUSED(V) ((void) V)

/* Forward declarations */

PSFloat applyDropout(PSNeuron *neuron, PSFloat value);
int PSConvolve(PSNeuralNetwork *net, PSLayer *layer, ...);
int PSPool(PSNeuralNetwork *net, PSLayer *layer, ...);
int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer *prev_layer,
                            PSGradient *lgradients, ...);
int PSPoolingBackprop(PSLayer *pooling_layer, PSLayer *convolutional_layer,
                      PSGradient *layer_gradients, ...);

uint8_t isDroppedOut(PSNeuron *neuron, ...);
int checkLayerForFeedforward(PSLayer *layer);

/* Generic Functions */

int PSConvolutionalLayerCopy(PSLayer *layer, PSLayer *src) {
    UNUSED(src);
    PSHyperParameters *parameters = layer->hyper_parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSFloat *params = parameters->parameters;
    if (params == NULL || layer->weights == NULL || layer->biases == NULL)
        return 0;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]), i, j;
    int feature_size = layer->size / feature_count;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
            if (neuron == NULL) return 0;
            neuron->bias = layer->biases + i;
            neuron->weights = layer->weights[i];
        }
    }
    return 1;
}

/* Init Functions */

int PSInitConvolutionalLayer(PSNeuralNetwork *network, PSLayer *layer,
                             PSHyperParameters *parameters)
{
    int index = layer->index;
    if (index == 0) {
        PSErr(__func__, "First (input) layer cannot be a convolutional layer!");
        goto err;
    }
    if (parameters == NULL) {
        PSErr(__func__, "Layer parameters is NULL!");
        goto err;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(__func__, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        goto err;
    }
    layer->on_copy = PSConvolutionalLayerCopy;
    PSLayer *previous = network->layers[index - 1];
    PSFloat *params = parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    if (feature_count <= 0) {
        PSErr(__func__, "FEATURE_COUNT must be > 0 (given: %d)", feature_count);
        goto err;
    }
    PSFloat region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(__func__, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        goto err;
    }
    int previous_size = previous->size, prev_features;
    PSHyperParameters *previous_params = previous->hyper_parameters;
    PSFloat input_w, input_h, output_w, output_h;
    int use_relu = (int) (params[PARAM_USE_RELU]);
    if (previous_params == NULL) {
        PSFloat w = (PSFloat) PSRound(PSSqrt(previous_size));
        input_w = w; input_h = w;
        previous_params = PSCreateConvolutionalParameters(1, 0, 0, 0, 0);
        previous_params->parameters[PARAM_OUTPUT_WIDTH] = input_w;
        previous_params->parameters[PARAM_OUTPUT_HEIGHT] = input_h;
        previous->hyper_parameters = previous_params;
        prev_features = 1;
    } else {
        input_w = previous_params->parameters[PARAM_OUTPUT_WIDTH];
        input_h = previous_params->parameters[PARAM_OUTPUT_HEIGHT];
        prev_features = (int) previous_params->parameters[PARAM_FEATURE_COUNT];
        if (input_h == 0) input_h = input_w;
        if (input_w == 0) {
            if (prev_features < 1) prev_features = 1;
            PSFloat featsize = previous_size / prev_features;
            input_w = PSSqrt(featsize);
            input_h = input_w;
        }
        PSFloat prev_area = input_w * input_h * (PSFloat) prev_features;
        if ((int) prev_area != previous_size) {
            PSErr(
                __func__, "Previous size %d != %d (%gx%gx%g)",
                 previous_size, (int) prev_area, input_w, input_h,
                 (PSFloat) prev_features
            );
            goto err;
        }
    }
    params[PARAM_INPUT_WIDTH] = input_w;
    params[PARAM_INPUT_HEIGHT] = input_h;
    int stride = (int) params[PARAM_STRIDE];
    int padding = (int) params[PARAM_PADDING];
    if (stride <= 0) stride = 1;
    output_w = PSCalculateConvolutionalSide(input_w, region_size,
                                            stride, (PSFloat) padding);
    output_h = PSCalculateConvolutionalSide(input_h, region_size,
                                            stride, (PSFloat) padding);
    params[PARAM_OUTPUT_WIDTH] = output_w;
    params[PARAM_OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    int weights_size = (int)(region_size * region_size) * prev_features;
    layer->size = size;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) goto memerr;
    layer->biases = malloc(feature_count * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->weights = malloc(feature_count * sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    PSFloat wscale = PSSqrt(1.0 / weights_size);
    int i, j;
    layer->weight_types_count = 0;
    for (i = 0; i < feature_count; i++) {
        layer->biases[i] = (use_relu ? 0.1 : 0.0);
        layer->weights[i] = PSMatrixWithGaussianRandom(
            wscale, 3, prev_features, (int) region_size, (int) region_size
        );
        if (layer->weights[i] == NULL) goto memerr;
        layer->weight_types_count++;
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(__func__, "Layer[%d]: Couldn't allocate neuron!",index);
                goto err;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->bias = layer->biases + i;
            neuron->weights = layer->weights[i];
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    if (!use_relu) {
        layer->activate = PSSigmoid;
        layer->derivative = PSSigmoidDerivative;
    } else {
        layer->activate = PSRelu;
        layer->derivative = PSReluDerivative;
    }
    layer->feedforward = PSConvolve;
    layer->backprop = PSConvolutionalBackprop;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
err:
    return 0;
}

int PSInitPoolingLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSHyperParameters *parameters)
{
    int index = layer->index;
    layer->weights = NULL;
    layer->biases = NULL;
    PSLayer *previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        PSErr(
            __func__, "Pooling's previous layer must be a Convolutional layer!"
        );
        return 0;
    }
    if (parameters == NULL) {
        PSErr(__func__, "Layer parameters is NULL!");
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(__func__, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        return 0;
    }
    PSFloat *params = parameters->parameters;
    PSHyperParameters *previous_parameters = previous->hyper_parameters;
    if (previous_parameters == NULL) {
        PSErr(__func__, "Previous layer parameters is NULL!");
        return 0;
    }
    if (previous_parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(__func__, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSFloat *previous_params = previous_parameters->parameters;
    int feature_count = (int) (previous_params[PARAM_FEATURE_COUNT]);
    params[PARAM_FEATURE_COUNT] = (PSFloat) feature_count;
    PSFloat region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(__func__, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSFloat input_w, input_h, output_w, output_h;
    input_w = previous_params[PARAM_OUTPUT_WIDTH];
    input_h = previous_params[PARAM_OUTPUT_HEIGHT];
    params[PARAM_INPUT_WIDTH] = input_w;
    params[PARAM_INPUT_HEIGHT] = input_h;
    output_w = PSCalculatePoolingSide(input_w, region_size);
    output_h = PSCalculatePoolingSide(input_h, region_size);
    params[PARAM_OUTPUT_WIDTH] = output_w;
    params[PARAM_OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(__func__, "Layer[%d]: Could not allocate neurons!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) {
        PSPrintMemoryErrorMsg();
        PSAbortLayer(network, layer);
        return 0;
    }
    int i, j;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(__func__, "Layer[%d]: Couldn't allocate neuron!", index);
                PSAbortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->bias = NULL;
            neuron->weights = NULL;
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->feedforward = PSPool;
    layer->backprop = PSPoolingBackprop;
    return 1;
}

/* Feedforward Functions */

int PSConvolve(PSNeuralNetwork *net, PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    int size = layer->size;
    int do_dump =
        (net->training != NULL && net->training->debug_dump_to != NULL);
    PSDebugStepInfo dbginfo = {
        .network = net,
        .layer = layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_FEEDFORWARD
    };
    PSLayer *previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    if (previous->flags & FLAG_ONEHOT) {
        PSErr(NULL, "Layer[%d]: convolutional layer cannot be fed with"
              "onehot input", layer->index);
        return 0;
    }
    int i, j, k, x, y, row, col;
    PSHyperParameters *parameters = layer->hyper_parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSHyperParameters *previous_parameters = previous->hyper_parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int is_recurrent = PSIsRecurrent(layer), times = 0, t = 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
        UNUSED(times);
    }
    PSFloat *params = parameters->parameters;
    PSFloat *previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    int stride = (int) (params[PARAM_STRIDE]);
    int padding = (int) (params[PARAM_PADDING]);
    PSFloat region_size = params[PARAM_REGION_SIZE];
    PSFloat region_area = region_size *region_size;
    PSFloat input_w = previous_params[PARAM_OUTPUT_WIDTH];
    PSFloat input_h = previous_params[PARAM_OUTPUT_HEIGHT];
    PSFloat output_w = params[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int apply_dropout = PSShouldApplyDropout(layer);
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
#ifdef USE_AVX
    int avx_disabled = !PSAVXEnabled(net->acceleration);
    /* AVX doesn't offer performance increase if not applied on big vectors */
    int avx_min_size = AVX_MIN_VECTOR_SIZE * 2;
#endif
    int previous_feature_size = 0, prev_features = 1;
    prev_features = (int) (previous_params[PARAM_FEATURE_COUNT]);
    if (prev_features == 0) prev_features = 1;
    previous_feature_size = previous->size / prev_features;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        PSFloat bias = (use_bias ? layer->biases[i] : 0.0);
        PSFloat *weights = layer->weights[i];
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
            dbginfo.neuron = neuron;
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = (row *stride) - padding;
            int r_col = (col *stride) - padding;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            PSFloat sum = 0;
            for (k = 0; k < prev_features; k++) {
                int widx = k * (int) region_area;
                int feature_offset = k * previous_feature_size;
                if (do_dump) PSTrainingDebugDump(
                    net, "#### Previous Feature[%d], offset=%d, weight_offset="
                    "%d\n", k, feature_offset, widx
                );
                for (y = r_row; y < max_y; y++) {
                    /* If y is outside layer's area (ie. padding area) and
                     * after all feature's rows, stop cycling y for this
                     * feature and go to next one.*/
                    if (y >= input_h) break;
                    if (y < 0) {
                        /* If y is outside layer's area (ie. padding area) and
                         * before it (y < 0), skip to next row after adding
                         * skipped weights to weight index (widx). */
                        widx += region_size;
                        continue;
                    }
                    x = r_col;
                    if (x < 0) {
                        /* If x is outside layer's area (ie. padding area)
                         * and before it (x < 0), jump to column 0
                         * after adding skipped weights to weight index
                         * (widx). */
                        widx += (0 - x);
                        x = 0;
                    }
                    int x2 = max_x;
                    if (x2 >= input_w) x2 = input_w - 1;
#ifdef USE_AVX
                    /*int rowlen = x2 - x;
                    if (!avx_disabled && rowlen >= avx_min_size) {
                        int avx_step_len = AVXGetDotStepLen(rowlen);
                        int avx_steps = 0, avx_step;
                        if (avx_step_len > 0)
                            avx_steps = rowlen / avx_step_len;
                        PSFloat *prev_activations = previous->activations;
                        assert(prev_activations != NULL);
                        if (PSIsRecurrent(previous))
                            prev_activations += (t * previous->size);
                        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                            int nidx = feature_offset + (y * input_w) + x;
                            PSFloat *x_vector =
                                prev_activations + nidx;
                            PSFloat *y_vector = weights + widx;
                            if (do_dump) DumpConvolveAVXStep(
                                col, row, r_col, r_row,
                                max_x, max_y, previous, k, nidx, x, y, widx,
                                avx_step_len, avx_step, rowlen
                            );
                            sum += AVXDotProduct(x_vector, y_vector,
                                                 avx_step_len, NULL);
                            x += avx_step_len;
                            widx += avx_step_len;
                        }
                    }*/ //DELME
#endif
                    for (; x < max_x; x++) {
                        int nidx = feature_offset + (y * input_w) + x;
                        /* printf("  -> %d,%d [%d]\n", x, y, nidx); */
                        assert(nidx >= 0);
                        if (nidx >= previous->size) break;
                        if (x >= input_w) {
                            /* If x is outside layer's area (ie. padding area)
                             * and after it (x >= input_w), stop cycling row
                             * and jump to next one, after adding skipped
                             * weights to weight index (widx). */
                            int skip_w = max_x - input_w;
                            if (skip_w > 0) widx += skip_w;
                            break;
                        }
                        PSFloat s = PSGetState(previous, nidx, t);
                        if (do_dump) DumpConvolveStep(
                            col, row, r_col, r_row, max_x, max_y, previous,
                            k, nidx, x, y, widx
                        );
                        sum += (s * weights[widx++]);
                    }
                }
                /* weights += (int) region_area; */
            }
            neuron->z_value = sum + bias;
            PSFloat s = layer->activate(neuron->z_value);
            if (apply_dropout) s = applyDropout(neuron, s);
            int ok = PSSetState(layer, s, idx, t);
            if (!ok) {
                PSErr(
                    NULL, "Failed to set state on layer %d, neuron %d",
                    layer->index, idx
                );
                if (layer->network) layer->network->status = STATUS_ERROR;
                return 0;
            }
        }
    }
    return 1;
}

int PSPool(PSNeuralNetwork *net, PSLayer *layer, ...) {
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
    int i, j, x, y, row, col;
    PSHyperParameters *parameters = layer->hyper_parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSHyperParameters *previous_parameters = previous->hyper_parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int do_dump =
        (net->training != NULL && net->training->debug_dump_to != NULL);
    PSDebugStepInfo dbginfo = {
        .network = net,
        .layer = layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_FEEDFORWARD
    };
    int is_recurrent = PSIsRecurrent(layer), times = 0, t = 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
        UNUSED(times);
    }
    PSFloat *params = parameters->parameters;
    PSFloat *previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    PSFloat region_size = params[PARAM_REGION_SIZE];
    PSFloat input_w = previous_params[PARAM_OUTPUT_WIDTH];
    PSFloat output_w = params[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int prev_size = previous->size / feature_count;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
            dbginfo.neuron = neuron;
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row *region_size;
            int r_col = col *region_size;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            PSFloat max = 0.0, max_z = 0.0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size *i);
                    PSNeuron *prev_neuron = previous->neurons[nidx];
                    PSFloat a = PSGetState(previous, nidx, t);
                    PSFloat z = prev_neuron->z_value;
                    if (a > max) {
                        max = a;
                        max_z = z;
                    }
                    if (do_dump) DumpPoolStep(
                        col, row, r_col, r_row, max_x, max_y, previous,
                        i, nidx, x, y
                    );
                }
            }
            neuron->z_value = max_z;
            int ok = PSSetState(layer, max, idx, t);
            if (!ok) {
                PSErr(
                    NULL, "Failed to set state on layer %d, neuron %d",
                    layer->index, idx
                );
                if (layer->network) layer->network->status = STATUS_ERROR;
                return 0;
            }
        }
    }
    return 1;
}

/* Backpropagation Functions */

int PSPoolingBackprop(PSLayer *pooling_layer, PSLayer *convolutional_layer,
                      PSGradient *layer_gradients, ...)
{
    UNUSED(layer_gradients);
    PSFloat *delta = pooling_layer->delta;
    PSFloat *conv_delta = convolutional_layer->delta;
    PSHyperParameters *pool_params = pooling_layer->hyper_parameters;
    PSHyperParameters *conv_params = convolutional_layer->hyper_parameters;
    int feature_count = (int) (conv_params->parameters[PARAM_FEATURE_COUNT]);
    int pool_size = (int) (pool_params->parameters[PARAM_REGION_SIZE]);
    int feature_size = pooling_layer->size / feature_count;
    PSFloat input_w = pool_params->parameters[PARAM_INPUT_WIDTH];
    PSFloat output_w = pool_params->parameters[PARAM_OUTPUT_WIDTH];
    int prev_feat_size = convolutional_layer->size / feature_count;
    PSNeuralNetwork *net = (PSNeuralNetwork *) pooling_layer->network;
    int do_dump = 0;
    if (net != NULL) {
        do_dump = (
            net->training != NULL && net->training->debug_dump_to != NULL
        );
    }
    PSDebugStepInfo dbginfo = {
        .network = net,
        .layer = pooling_layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_BACKPROP
    };
    int is_recurrent = PSIsRecurrent(pooling_layer), t = 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer_gradients);
        t = va_arg(args, int);
        va_end(args);
    }
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            dbginfo.neuron = pooling_layer->neurons[idx];
            PSFloat d = delta[idx];
            PSFloat pool_state = PSGetState(pooling_layer, idx, t);
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row *pool_size;
            int r_col = col *pool_size;
            int max_x = pool_size + r_col;
            int max_y = pool_size + r_row;
            /* PSFloat max = 0; */
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_feat_size *i);
                    PSNeuron *prev_neuron = convolutional_layer->neurons[nidx];
                    if (do_dump) DumpPoolBackpropStep(
                        col, row, r_col, r_row, max_x, max_y,
                        convolutional_layer, i, nidx, x, y
                    );
                    if (isDroppedOut(prev_neuron, t)) continue;
                    PSFloat s = PSGetState(convolutional_layer, nidx, t);
                    PSFloat dv = (s < pool_state ? 0 : d);
                    if (dv != 0 && convolutional_layer->derivative != NULL)
                        dv *= convolutional_layer->derivative(pool_state);
                    conv_delta[nidx] = dv;
                }
            }
        }
    }
    return 1;
}

int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer *prev_layer,
                            PSGradient *gradient, ...)
{
    PSFloat *delta = convolutional_layer->delta;
    PSFloat *prev_delta = prev_layer->delta;
    int size = convolutional_layer->size;
    PSHyperParameters *params = convolutional_layer->hyper_parameters;
    assert(params != NULL);
    assert(convolutional_layer->weights != NULL);
    assert(convolutional_layer->weights[0] != NULL);
    int feature_count = (int) (params->parameters[PARAM_FEATURE_COUNT]);
    int region_size = (int) (params->parameters[PARAM_REGION_SIZE]);
    int region_area = region_size *region_size;
    int stride = (int) (params->parameters[PARAM_STRIDE]);
    int padding = (int) (params->parameters[PARAM_PADDING]);
    PSFloat input_w = params->parameters[PARAM_INPUT_WIDTH];
    PSFloat input_h = params->parameters[PARAM_INPUT_HEIGHT];
    PSFloat output_w = params->parameters[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int previous_feature_size = 0, prev_features = 1;
    PSHyperParameters *prev_params = prev_layer->hyper_parameters;
    if (prev_params != NULL) {
        prev_features = (int) (prev_params->parameters[PARAM_FEATURE_COUNT]);
        previous_feature_size = prev_layer->size / prev_features;
    }
    int weight_size = PSMatrixLength(convolutional_layer->weights[0]);
    PSNeuralNetwork *net = (PSNeuralNetwork *) convolutional_layer->network;
    int do_dump = 0;
    if (net != NULL) {
        do_dump = (
            net->training != NULL && net->training->debug_dump_to != NULL
        );
    }
    PSDebugStepInfo dbginfo = {
        .network = net,
        .layer = convolutional_layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_BACKPROP
    };
    int is_recurrent = PSIsRecurrent(convolutional_layer), t = 0;
    int use_bias = !(convolutional_layer->flags & FLAG_NO_BIAS);
    if (is_recurrent) {
        va_list args;
        va_start(args, gradient);
        t = va_arg(args, int);
        va_end(args);
    }
    int i, j, k, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        int weight_offset = (weight_size * i);
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            dbginfo.neuron = convolutional_layer->neurons[idx];
            PSFloat d = delta[idx];
            if (use_bias) gradient->biases[i] += d;
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = (row *stride) - padding;
            int r_col = (col *stride) - padding;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            for (k = 0; k < prev_features; k++) {
                int feature_offset = k * previous_feature_size;
                int widx = k * region_area;
                if (do_dump && k <= 1) PSTrainingDebugDump(
                    net, "#### Previous Feature[%d], offset=%d, weight_offset="
                    "%d\n", k, feature_offset, widx
                );
                for (y = r_row; y < max_y; y++) {
                    if (y >= input_h) break; /* If y is beyond area, go to
                                              * next feature. */
                    for (x = r_col; x < max_x; x++) {
                        if (x < 0 || y < 0) {
                            /* If x or y are before area (ie. padding), skip
                             * unit and add skipped weight to weight index. */
                            widx++;
                            continue;
                        }
                        int nidx = feature_offset + (y * input_w) + x;
                        /* printf("  -> %d,%d [%d]\n", x, y, nidx); */
                        assert(nidx >= 0);
                        if (nidx >= prev_layer->size) break;
                        /* If x is outside layer (ie. inside padding area),
                         * exit x cycle and go to next row after adding
                         * skipped padding units to weight index (widx). */
                        if (x >= input_w && max_x > input_w) {
                            int skip_w = max_x - input_w;
                            if (skip_w > 0) widx += skip_w;
                            break;
                        }
                        PSNeuron *prev_neuron = prev_layer->neurons[nidx];
                        PSFloat a = PSGetState(prev_layer, nidx, t);
                        assert(widx >= 0);
                        if (widx >= weight_size) {
                            /* Ensure that weight index (widx) never exceeds
                             * shared weights. */
                            fprintf(stderr,
                                    "\nwidx=%d, weights_size=%d,"
                                    "x=%d,y=%d,k=%d,r_col=%d,r_row=%d,"
                                    "max_x=%d,max_y=%d, layer=%d\n",
                                    widx, weight_size, x, y, k,
                                    r_col, r_row, max_x, max_y,
                                    convolutional_layer->index);
                            assert(widx < weight_size);
                        }
                        if (do_dump) DumpConvBackpropStep(
                            col, row, r_col, r_row, max_x, max_y,
                            prev_layer, k, nidx, x, y, widx
                        );
                        gradient->weights[weight_offset + widx] += (a * d);
                        if (prev_delta != NULL && !isDroppedOut(prev_neuron,t)){
                            PSNeuron *neuron =
                                convolutional_layer->neurons[idx];
                            prev_delta[nidx] += (d * neuron->weights[widx]);
                        }
                        widx++;
                    }
                }
            }
        }
    }
    return 1;
}

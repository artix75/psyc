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

PSActivationFunction PSGetActivationDerivative(PSActivationFunction func);
int PSConvolve(PSNeuralNetwork *net, PSLayer *layer, ...);
int PSPool(PSNeuralNetwork *net, PSLayer *layer, ...);
int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer *prev_layer,
                            PSGradient *lgradients, ...);
int PSPoolingBackprop(PSLayer *pooling_layer, PSLayer *convolutional_layer,
                      PSGradient *layer_gradients, ...);

int checkLayerForFeedforward(PSLayer *layer);

/* Generic Functions */

int PSConvolutionalLayerCopy(PSLayer *layer, PSLayer *src) {
    PSConvolutionalSettings *srcsettings = PSGetConvolutionalSettings(src);
    PSConvolutionalSettings *dstsettings = PSGetConvolutionalSettings(layer);
    if (srcsettings == NULL) {
        PSErr(NULL, "Layer[%d]: missing convolutional settings", layer->index);
        return 0;
    }
    memcpy(dstsettings, srcsettings, sizeof(PSConvolutionalSettings));
    layer->output_columns = src->output_columns;
    layer->output_rows = src->output_rows;
    layer->output_depth = src->output_depth;
    if (Pooling == layer->type) return 1;
    if (layer->weights == NULL || layer->biases == NULL)
        return 0;
    int i, j;
    int feature_size = layer->size / layer->output_depth;
    for (i = 0; i < layer->output_depth; i++) {
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

void PSDeleteConvolutionalLayer(PSLayer *layer) {
    if (layer->extra != NULL) free(layer->extra);
    layer->extra = NULL;
}

/* Init Functions */

static PSMatrix initConvWeights(PSLayer *layer, int depth, int rows, int cols,
                                PSLayerDef *ldef, PSFloat range, PSFloat scale)
{
    static PSLayerDef default_def = {0};
    if (ldef == NULL) ldef = &default_def;
    PSMatrix weights = NULL;
    if (ldef->weight_init_mode == INIT_MODE_ZERO)
        weights = PSMatrixZeros(3, depth, rows, cols);
    else {
        if (ldef->weight_init_mode == INIT_MODE_RAND) {
            range = ldef->init_range;
            scale = ldef->init_scale;
        }
        if (range == 0) range = 1;
        weights = PSMatrixWithGaussianRandom(range, 3, depth, rows, cols);
        if (scale > 0 && weights != NULL) {
            int acceleration = PSGlobalAcceleration;
            if (layer != NULL && layer->network != NULL)
                acceleration = layer->network->acceleration;
            PSMathOpts opts = {.acceleration = acceleration};
            PSMultiplyVectorScalar(weights, scale, weights, (rows*cols), &opts);
        }
    }
    return weights;
}

static PSFloat convRandomBias(PSLayerDef *ldef) {
    static PSLayerDef default_def = {0};
    if (ldef == NULL) ldef = &default_def;
    PSFloat bias, range = ldef->init_range, scale = ldef->init_scale;
    if (range <= 0) range = 1.0;
    bias = PSGaussianRandom(0.0, range);
    if (scale > 0) bias *= scale;
    return bias;
}

int PSInitConvolutionalLayer(PSNeuralNetwork *network, PSLayer *layer,
                             PSLayerDef *layer_def)
{
    int index = layer->index;
    if (index == 0) {
        PSErr(__func__, "First (input) layer cannot be a convolutional layer!");
        goto err;
    }
    if (layer_def == NULL) {
        PSErr(__func__, "Layer def. is mandatory for convolutonal layers");
        goto err;
    }
    layer->on_copy = PSConvolutionalLayerCopy;
    layer->on_delete = PSDeleteConvolutionalLayer;
    layer->extra = calloc(1, sizeof(PSConvolutionalSettings));
    if (layer->extra == NULL) goto memerr;
    PSConvolutionalSettings *settings = (PSConvolutionalSettings *)layer->extra;
    PSLayer *previous = network->layers[index - 1];
    layer->output_depth = layer_def->output_depth;
    if (layer->output_depth <= 0) {
        PSErr(__func__, "output_depth must be > 0 (given: %d)",
              layer->output_depth);
        goto err;
    }
    settings->filter_width = layer_def->filter_width;
    settings->filter_height = layer_def->filter_height;
    settings->filter_depth = previous->output_depth;
    if (settings->filter_width == 0) {
        PSErr(__func__, "filter_width must be > 0 (given: %d)",
              settings->filter_width);
        goto err;
    }
    if (settings->filter_depth <= 0)
        settings->filter_depth = previous->output_depth = 1;
    if (settings->filter_height == 0)
        settings->filter_height = settings->filter_width;
    if (layer_def->activation != NULL) layer->activate = layer_def->activation;
    else layer->activate = PSSigmoid;
    layer->derivative = PSGetActivationDerivative(layer->activate);
    int input_w, input_h, output_w, output_h;
    input_w = previous->output_columns;
    input_h = previous->output_rows;
    if (input_w <= 0) {
        int w = (int) PSRound(PSSqrt(previous->size));
        input_w = w; input_h = w;
        previous->output_columns = input_w;
        previous->output_rows = input_h;
    } else {
        if (input_h == 0) input_h = input_w;
        if (input_w == 0) {
            PSFloat featsize = (PSFloat) previous->size /
                                         previous->output_depth;
            input_w = (int) PSSqrt(featsize);
            input_h = input_w;
        }
        int prev_area = input_w * input_h * previous->output_depth;
        if (prev_area != previous->size) {
            PSErr(
                __func__, "Previous size %d != %d (%dx%dx%d)",
                 previous->size, prev_area, input_w, input_h,
                 previous->output_depth
            );
            goto err;
        }
    }
    settings->input_width = input_w;
    settings->input_height = input_h;
    settings->input_depth = previous->output_depth;
    int stride = layer_def->stride;
    int padding = layer_def->padding;
    if (stride <= 0) stride = 1;
    if (padding < 0) padding = 0;
    settings->stride = stride;
    settings->padding = padding;
    output_w = PSCalculateConvolutionalSide(input_w, settings->filter_width,
                                            stride, (PSFloat) padding);
    output_h = PSCalculateConvolutionalSide(input_h, settings->filter_width,
                                            stride, (PSFloat) padding);
    layer->output_columns = output_w;
    layer->output_rows = output_h;
    int area = (output_w * output_h);
    int size = area * layer->output_depth;
    int weights_size = settings->filter_width * settings->filter_height *
                       settings->filter_depth;;
    layer->size = size;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) goto memerr;
    layer->biases = malloc(layer->output_depth * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->weights = malloc(layer->output_depth * sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    PSFloat wrange = PSSqrt(1.0 / weights_size);
    int i, j, use_relu = (layer->activate == PSRelu), rand_bias = 0;
    layer->weight_types_count = 0;
    PSFloat default_bias = (use_relu ? 0.1 : 0.0);
    if (layer_def->bias_init_mode == INIT_MODE_ZERO) default_bias = 0.0;
    else rand_bias = (layer_def->bias_init_mode == INIT_MODE_RAND);
    for (i = 0; i < layer->output_depth; i++) {
        layer->biases[i] = (
            rand_bias ? convRandomBias(layer_def) : default_bias
        );
        layer->weights[i] = initConvWeights(
            layer, previous->output_depth, settings->filter_width,
            settings->filter_height, layer_def, wrange, 1.0
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
    layer->feedforward = PSConvolve;
    layer->backprop = PSConvolutionalBackprop;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
err:
    return 0;
}

int PSInitPoolingLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSLayerDef *layer_def)
{
    int index = layer->index;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->on_delete = PSDeleteConvolutionalLayer;
    layer->on_copy = PSConvolutionalLayerCopy;
    PSLayer *previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        PSErr(
            __func__, "Pooling's previous layer must be a Convolutional layer!"
        );
        return 0;
    }
    PSLayerDef default_def = {.stride = 1, .filter_width = 2, .filter_height=2};
    if (layer_def == NULL) layer_def = &default_def;
    layer->extra = calloc(1, sizeof(PSConvolutionalSettings));
    if (layer->extra == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSConvolutionalSettings *settings = (PSConvolutionalSettings*)layer->extra;
    layer->output_depth = previous->output_depth;
    settings->filter_width = layer_def->filter_width;
    settings->filter_height = layer_def->filter_height;
    if (settings->filter_width <= 0) {
        PSErr(__func__, "filter_width must be > 0 (given: %d)",
              settings->filter_width);
        return 0;
    }
    PSFloat input_w, input_h, output_w, output_h;
    input_w = previous->output_columns;
    input_h = previous->output_rows;
    settings->input_width = input_w;
    settings->input_height = input_h;
    output_w = PSCalculatePoolingSide(input_w, settings->filter_width);
    output_h = PSCalculatePoolingSide(input_h, settings->filter_width);
    layer->output_columns = output_w;
    layer->output_rows = output_h;
    int area = (output_w * output_h);
    int size = area * layer->output_depth;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) goto memerr;
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) goto memerr;
    int i, j;
    for (i = 0; i < layer->output_depth; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) goto memerr;
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
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Feedforward Functions */

int PSConvolve(PSNeuralNetwork *net, PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
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
    PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
    if (settings == NULL) {
        PSErr(NULL, "Layer[%d]: convolutional layer has no settings",
              layer->index);
        return 0;
    }
    if (layer->output_depth == 0) layer->output_depth = 1;
    int i, j, k, x, y, row, col;
    int is_recurrent = PSIsRecurrent(layer), times = 0, t = 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
        UNUSED(times);
    }
    int stride = settings->stride;
    int padding = settings->padding;
    int filter_area = settings->filter_width * settings->filter_height;
    int feature_size = layer->size / layer->output_depth;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int input_w = settings->input_width, input_h = settings->input_height;
#ifdef USE_AVX
    int avx_disabled = !PSAVXEnabled(net->acceleration);
    /* AVX doesn't offer performance increase if not applied on big vectors */
    int avx_min_size = AVX_MIN_VECTOR_SIZE * 2;
#endif
    int previous_feature_size = 0;
    if (previous->output_depth == 0) previous->output_depth = 1;
    previous_feature_size = previous->size / previous->output_depth;
    for (i = 0; i < layer->output_depth; i++) {
        if (do_dump && i > 1) do_dump = 0;
        PSFloat bias = (use_bias ? layer->biases[i] : 0.0);
        PSFloat *weights = layer->weights[i];
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
            dbginfo.neuron = neuron;
            col = idx % layer->output_columns;
            if (col == 0 && j > 0) row++;
            int r_row = (row * stride) - padding;
            int r_col = (col * stride) - padding;
            int max_x = settings->filter_width + r_col;
            int max_y = settings->filter_height + r_row;
            PSFloat sum = 0;
            for (k = 0; k < previous->output_depth; k++) {
                int widx = k * filter_area;
                int foffset = k * previous_feature_size;
                if (do_dump) PSTrainingDebugDump(
                    net, "#### Previous Feature[%d], offset=%d, weight_offset="
                    "%d\n", k, foffset, widx
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
                        widx += settings->filter_width;
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
                    if (x2 >= settings->input_width) x2 = input_w - 1;
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
                            int nidx = foffset + (y * input_w) + x;
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
                        int nidx = foffset + (y * settings->input_width) + x;
                        /* printf("  -> %d,%d [%d]\n", x, y, nidx); */
                        assert(nidx >= 0);
                        if (nidx >= previous->size) break;
                        if (x >= settings->input_width) {
                            /* If x is outside layer's area (ie. padding area)
                             * and after it (x >= input_w), stop cycling row
                             * and jump to next one, after adding skipped
                             * weights to weight index (widx). */
                            int skip_w = max_x - settings->input_width;
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
            PSFloat state = sum + bias;
            state = layer->activate(state);
            int ok = PSSetState(layer, state, idx, t);
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
    PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
    if (settings == NULL) {
        PSErr(NULL, "Layer[%d]: convolutional layer has no settings",
              layer->index);
        return 0;
    }
    int i, j, x, y, row, col;
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
    int input_w = previous->output_columns;
    int output_w = layer->output_columns;
    int feature_size = layer->size / layer->output_depth;
    int prev_size = previous->size / previous->output_depth;
    if (settings->filter_height == 0)
        settings->filter_height = settings->filter_width;
    for (i = 0; i < layer->output_depth; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
            dbginfo.neuron = neuron;
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * settings->filter_height;
            int r_col = col * settings->filter_width;
            int max_x = settings->filter_width + r_col;
            int max_y = settings->filter_height + r_row;
            PSFloat max = 0.0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size *i);
                    PSFloat a = PSGetState(previous, nidx, t);
                    if (a > max) max = a;
                    if (do_dump) DumpPoolStep(
                        col, row, r_col, r_row, max_x, max_y, previous,
                        i, nidx, x, y
                    );
                }
            }
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
    PSConvolutionalSettings *settings =
        PSGetConvolutionalSettings(pooling_layer);
    if (settings == NULL) {
        PSErr(NULL, "Layer[%d]: pooling layer has no settings",
              pooling_layer->index);
        return 0;
    }
    PSFloat *delta = pooling_layer->delta;
    PSFloat *conv_delta = convolutional_layer->delta;
    int feature_size = pooling_layer->size / pooling_layer->output_depth;
    int input_w = settings->input_width;
    int output_w = pooling_layer->output_columns;
    int prev_feat_size = convolutional_layer->size /
                         convolutional_layer->output_depth;
    if (settings->filter_height <= 0)
        settings->filter_height = settings->filter_width;
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
    for (i = 0; i < pooling_layer->output_depth; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            dbginfo.neuron = pooling_layer->neurons[idx];
            PSFloat d = delta[idx];
            PSFloat pool_state = PSGetState(pooling_layer, idx, t);
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * settings->filter_height;
            int r_col = col * settings->filter_width;
            int max_x = settings->filter_width + r_col;
            int max_y = settings->filter_height + r_row;
            /* PSFloat max = 0; */
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_feat_size *i);
                    if (do_dump) DumpPoolBackpropStep(
                        col, row, r_col, r_row, max_x, max_y,
                        convolutional_layer, i, nidx, x, y
                    );
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
    if (convolutional_layer->weights == NULL ||
        convolutional_layer->weights[0] == NULL) {
        PSErr(NULL, "Layer[%d]: missing weights", convolutional_layer->index);
        return 0;
    }
    PSConvolutionalSettings *settings =
        PSGetConvolutionalSettings(convolutional_layer);
    if (settings == NULL) {
        PSErr(NULL, "Layer[%d]: convolutional layer has no settings",
              convolutional_layer->index);
        return 0;
    }
    int filter_area = settings->filter_width * settings->filter_height;
    int stride = settings->stride;
    int padding = settings->padding;
    int input_w = settings->input_width;
    int input_h = settings->input_height;
    int output_w = convolutional_layer->output_columns;
    int feature_size = convolutional_layer->size /
                       convolutional_layer->output_depth;
    int previous_feature_size = 0;
    int prev_depth = prev_layer->output_depth;
    if (prev_depth <= 0) prev_depth = 1;
    previous_feature_size = prev_layer->size / prev_depth;
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
    for (i = 0; i < convolutional_layer->output_depth; i++) {
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
            int r_row = (row * stride) - padding;
            int r_col = (col * stride) - padding;
            int max_x = settings->filter_width + r_col;
            int max_y = settings->filter_height + r_row;
            for (k = 0; k < prev_depth; k++) {
                int feature_offset = k * previous_feature_size;
                int widx = k * filter_area;
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
                        if (prev_delta != NULL) {
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

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

#ifdef USE_AVX
#include "avx.h"
#endif

#include "psyc.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "debug.h"

#define DumpConvolveStep(net,l,n,x,y,rx,ry,rx2,ry2,prev,fidx,nidx,nx,ny,widx) \
 PSTrainingDebugDumpStep(net,TRAINING_PHASE_FEEDFORWARD, "PSConvolve",l,n,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),weight_idx=%d,"\
 "srcline=%d\n",\
 x, y, rx, ry, rx2, ry2,prev->index, prev->index, fidx, nidx, nx, ny, widx,\
 __LINE__)

#define DumpConvolveAVXStep(net,l,n,x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny,\
w,steplen,step,rowlen) \
 PSTrainingDebugDumpStep(net,TRAINING_PHASE_FEEDFORWARD, "PSConvolve",l,n,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),weight_idx=%d,"\
 "avx=1,avx_step_len=%d,avx_step=%d,rowlen=%d,srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index,prv->index, fidx, nidx, nx, ny, widx,\
 steplen, step, rowlen, __LINE__)

#define DumpPoolStep(net,l,n,x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny) \
 PSTrainingDebugDumpStep(net, TRAINING_PHASE_FEEDFORWARD, "PSPool", l, n,\
 "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_layer=%d,"\
 "previous_neuron=%d-%d-%d,previous_neuron_pos=(%d,%d),srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index,prv->index, fidx, nidx, nx, ny, __LINE__)

#define DumpPoolBackpropStep(net,l,n,x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny) \
 PSTrainingDebugDumpStep(net, TRAINING_PHASE_BACKPROP, "PSPoolingBackprop",\
 l, n, "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_neuron=%d-%d-%d,"\
 "previous_neuron_pos=(%d,%d),srcline=%d\n", x, y, rx, ry, rx2, ry2,\
 prv->index, fidx, nidx, nx, ny, __LINE__)

#define DumpConvBackpropStep(net,l,n,x,y,rx,ry,rx2,ry2,prv,fidx,nidx,nx,ny,w) \
 PSTrainingDebugDumpStep(net,TRAINING_PHASE_BACKPROP,"PSConvolutionalBackprop",\
 l, n, "neuron_pos=(%d,%d),region=(%d,%d,%d,%d),previous_neuron=%d-%d-%d,"\
 "previous_neuron_pos=(%d,%d),weight_idx=%d,srcline=%d\n",\
 x, y, rx, ry, rx2, ry2, prv->index, fidx, nidx, nx, ny, widx, __LINE__)


/* Init Functions */

int PSInitConvolutionalLayer(PSNeuralNetwork *network, PSLayer *layer,
                             PSLayerParameters *parameters) {
    int index = layer->index;
    char *func = "initConvolutionalLayer";
    if (index == 0) {
        PSErr(func, "First (input) layer cannot be a convolutional layer!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        PSErr(func, "Layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSLayer *previous = network->layers[index - 1];
    PSFloat *params = parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    if (feature_count <= 0) {
        PSErr(func, "FEATURE_COUNT must be > 0 (given: %d)", feature_count);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSFloat region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        PSAbortLayer(network, layer);
        return 0;
    }
    int previous_size = previous->size, prev_features;
    PSLayerParameters *previous_params = previous->parameters;
    PSFloat input_w, input_h, output_w, output_h;
    int use_relu = (int) (params[PARAM_USE_RELU]);
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    if (previous_params == NULL) {
        PSFloat w = PSSqrt(previous_size);
        input_w = w; input_h = w;
        previous_params = PSCreateConvolutionalParameters(1, 0, 0, 0, 0);
        previous_params->parameters[PARAM_OUTPUT_WIDTH] = input_w;
        previous_params->parameters[PARAM_OUTPUT_HEIGHT] = input_h;
        previous->parameters = previous_params;
        prev_features = 1;
    } else {
        input_w = previous_params->parameters[PARAM_OUTPUT_WIDTH];
        input_h = previous_params->parameters[PARAM_OUTPUT_HEIGHT];
        prev_features = (int) previous_params->parameters[PARAM_FEATURE_COUNT];
        PSFloat prev_area = input_w *input_h * (PSFloat) prev_features;
        if ((int) prev_area != previous_size) {
            PSErr(func, "Previous size %d != %lfx%lf",
                  previous_size, input_w, input_h);
            PSAbortLayer(network, layer);
            return 0;
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
    int area = (int)(output_w *output_h);
    int size = area *feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate neurons!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    if (!avx_disabled) {
        layer->avx_activation_cache = calloc(size, sizeof(PSFloat));
        if (layer->avx_activation_cache == NULL) {
            PSPrintMemoryErrorMsg();
            PSAbortLayer(network, layer);
            return 0;
        }
    }
#endif
    PSSharedParams *shared = malloc(sizeof(PSSharedParams));
    if (shared == NULL) {
        PSErr(func, "Layer[%d]: Couldn't allocate shared params!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
    shared->feature_count = feature_count;
    shared->weights_size = (int)(region_size *region_size) * prev_features;
    shared->biases = malloc(feature_count * sizeof(PSFloat));
    shared->weights = malloc(feature_count * sizeof(PSFloat*));
    if (shared->biases == NULL || shared->weights == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate memory!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
    layer->extra = shared;
    int i, j, w;
    PSFloat wscale = PSSqrt(1.0 / shared->weights_size);
    for (i = 0; i < feature_count; i++) {
        shared->biases[i] = (use_relu ? 0.1 : 0.0);
        /* shared->biases[i] = PSGaussianRandom(0, 1);*/
        shared->weights[i] = malloc(shared->weights_size * sizeof(PSFloat));
        if (shared->weights[i] == NULL) {
            PSErr(func, "Layer[%d]: Could not allocate weights!", index);
            PSAbortLayer(network, layer);
            return 0;
        }
        for (w = 0; w < shared->weights_size; w++) {
            shared->weights[i][w] = PSGaussianRandom(0, wscale);
        }
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(func, "Layer[%d]: Couldn't allocate neuron!",index);
                PSAbortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = shared->weights_size;
            neuron->bias = shared->biases[i];
            neuron->weights = shared->weights[i];
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
    return 1;
}

int PSInitPoolingLayer(PSNeuralNetwork *network, PSLayer *layer,
                       PSLayerParameters *parameters) {
    int index = layer->index;
    char *func = "initPoolingLayer";
    PSLayer *previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        PSErr(func, "Pooling's previous layer must be a Convolutional layer!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        PSErr(func, "Layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSFloat *params = parameters->parameters;
    PSLayerParameters *previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(func, "Previous layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (previous_parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    PSFloat *previous_params = previous_parameters->parameters;
    int feature_count = (int) (previous_params[PARAM_FEATURE_COUNT]);
    params[PARAM_FEATURE_COUNT] = (PSFloat) feature_count;
    PSFloat region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
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
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(network);
#endif
    int area = (int)(output_w *output_h);
    int size = area *feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate neurons!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    if (!avx_disabled) {
        layer->avx_activation_cache = calloc(size, sizeof(PSFloat));
        if (layer->avx_activation_cache == NULL) {
            PSPrintMemoryErrorMsg();
            PSAbortLayer(network, layer);
            return 0;
        }
    }
#endif
    int i, j;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron *neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(func, "Layer[%d]: Couldn't allocate neuron!", index);
                PSAbortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = 0;
            neuron->bias = PS_NULL_VALUE;
            neuron->weights = NULL;
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->feedforward = PSPool;
    return 1;
}

/* Feedforward Functions */

int PSConvolve(PSNeuralNetwork *net, PSLayer *layer, ...) {
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
    int do_dump =
        (net->training != NULL && net->training->debug_dump_to != NULL);
    int i, j, k, x, y, row, col;
    PSLayerParameters *parameters = layer->parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSLayerParameters *previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
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
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
    /* AVX doesn't offer performance increase if not applied on big vectors */
    int avx_min_size = AVX_MIN_VECTOR_SIZE * 2;
#endif
    PSSharedParams *shared = PSGetConvSharedParams(layer);
    if (shared == NULL) {
        PSErr(NULL, "Layer[%d]: shared params are NULL!", layer->index);
        return 0;
    }
    int previous_feature_size = 0, prev_features = 1;
    prev_features = (int) (previous_params[PARAM_FEATURE_COUNT]);
    if (prev_features == 0) prev_features = 1;
    previous_feature_size = previous->size / prev_features;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        PSFloat bias = shared->biases[i];
        PSFloat *weights = shared->weights[i];
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
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
                    int rowlen = x2 - x;
                    if (!avx_disabled && rowlen >= avx_min_size) {
                        int avx_step_len = AVXGetDotStepLen(rowlen);
                        int avx_steps = 0, avx_step;
                        if (avx_step_len > 0)
                            avx_steps = rowlen / avx_step_len;
                        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                            int nidx = feature_offset + (y * input_w) + x;
                            PSFloat *x_vector =
                                previous->avx_activation_cache + nidx;
                            if (is_recurrent) x_vector += (t * previous->size);
                            PSFloat *y_vector = weights + widx;
                            if (do_dump) DumpConvolveAVXStep(
                                net, layer, neuron, col, row, r_col, r_row,
                                max_x, max_y, previous, k, nidx, x, y, widx,
                                avx_step_len, avx_step, rowlen
                            );
                            sum += AVXDotProduct(x_vector, y_vector,
                                                 avx_step_len, NULL);
                            x += avx_step_len;
                            widx += avx_step_len;
                        }
                    }
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
                        PSNeuron *prev_neuron = previous->neurons[nidx];
                        PSFloat a = prev_neuron->activation;
                        if (do_dump) DumpConvolveStep(net, layer, neuron,
                            col, row, r_col, r_row, max_x, max_y,previous,
                            k, nidx, x, y, widx
                        );
                        sum += (a * weights[widx++]);
                    }
                }
                /* weights += (int) region_area; */
            }
            neuron->z_value = sum + bias;
            neuron->activation = layer->activate(neuron->z_value);
#ifdef USE_AVX
            if (!is_recurrent && !avx_disabled)
                layer->avx_activation_cache[idx] = neuron->activation;
#endif
            if (is_recurrent) {
                PSAddRecurrentState(net, neuron, neuron->activation, times, t);
                if (neuron->extra == NULL) {
                    PSErr("convolve", "Failed to allocate Recurrent Cell!");
                    return 0;
                }
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
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#endif
    PSLayerParameters *parameters = layer->parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSLayerParameters *previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int do_dump =
        (net->training != NULL && net->training->debug_dump_to != NULL);
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
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
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron *neuron = layer->neurons[idx];
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
                    PSFloat a = prev_neuron->activation;
                    PSFloat z = prev_neuron->z_value;
                    if (a > max) {
                        max = a;
                        max_z = z;
                    }
                    if (do_dump) DumpPoolStep(net, layer, neuron, col, row,
                        r_col, r_row, max_x, max_y, previous, i, nidx, x, y
                    );
                }
            }
            neuron->z_value = max_z;
            neuron->activation = max;
#ifdef USE_AVX
            if (!is_recurrent && !avx_disabled)
                layer->avx_activation_cache[idx] = neuron->activation;
#endif
            if (is_recurrent) {
                PSAddRecurrentState(net, neuron, neuron->activation, times, t);
                if (neuron->extra == NULL) {
                    PSErr("pool", "Failed to allocate Recurrent Cell!");
                    return 0;
                }
            }
        }
    }
    return 1;
}

/* Backpropagation Functions */

int PSPoolingBackprop(PSLayer *pooling_layer, PSLayer *convolutional_layer,
                      PSFloat *delta)
{
    PSFloat *conv_delta = convolutional_layer->delta;
    PSLayerParameters *pool_params = pooling_layer->parameters;
    PSLayerParameters *conv_params = convolutional_layer->parameters;
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
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            PSFloat d = delta[idx];
            PSNeuron *neuron = pooling_layer->neurons[idx];
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
                        net, pooling_layer, pooling_layer->neurons[idx],
                        col, row, r_col, r_row, max_x, max_y,
                        convolutional_layer, i, nidx, x, y
                    );
                    PSFloat a = prev_neuron->activation;
                    PSFloat dv = (a < neuron->activation ? 0 : d);
                    if (dv != 0 && convolutional_layer->derivative != NULL) {
                        dv *= convolutional_layer->derivative(
                            neuron->activation
                        );
                    }
                    conv_delta[nidx] = dv;
                }
            }
        }
    }
    return 1;
}

int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer *prev_layer,
                            PSGradient *lgradients)
{
    PSFloat *delta = convolutional_layer->delta;
    PSFloat *prev_delta = prev_layer->delta;
    int size = convolutional_layer->size;
    PSLayerParameters *params = convolutional_layer->parameters;
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
    PSLayerParameters *prev_params = prev_layer->parameters;
    if (prev_params != NULL) {
        prev_features = (int) (prev_params->parameters[PARAM_FEATURE_COUNT]);
        previous_feature_size = prev_layer->size / prev_features;
    }
    PSNeuralNetwork *net = (PSNeuralNetwork *) convolutional_layer->network;
    PSSharedParams *shared = PSGetConvSharedParams(convolutional_layer);
    int do_dump = 0;
    if (net != NULL) {
        do_dump = (
            net->training != NULL && net->training->debug_dump_to != NULL
        );
    }
    int i, j, k, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        if (do_dump && i > 1) do_dump = 0;
        PSGradient *feature_gradient = &(lgradients[i]);
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            PSFloat d = delta[idx];
            feature_gradient->bias += d;
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
                        PSFloat a = prev_neuron->activation;
                        assert(widx >= 0);
                        if (widx >= shared->weights_size) {
                            /* Ensure that weight index (widx) never exceeds
                             * shared weights. */
                            fprintf(stderr,
                                    "\nwidx=%d, shared->weights_size=%d,"
                                    "x=%d,y=%d,k=%d,r_col=%d,r_row=%d,"
                                    "max_x=%d,max_y=%d, layer=%d\n",
                                    widx, shared->weights_size, x, y, k,
                                    r_col, r_row, max_x, max_y,
                                    convolutional_layer->index);
                            assert(widx < shared->weights_size);
                        }
                        if (do_dump) DumpConvBackpropStep(
                            net, convolutional_layer,
                            convolutional_layer->neurons[idx],
                            col, row, r_col, r_row, max_x, max_y,
                            prev_layer, k, nidx, x, y, widx
                        );
                        feature_gradient->weights[widx] += (a * d);
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

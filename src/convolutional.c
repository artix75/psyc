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
#include "maths.h"
#include "blas.h"
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

#define GetConvPrivateData(layer) ((PSPrivateConvData *) layer->private)

typedef struct {
    PSFloat *im2col;
    PSFloat *w2col;
    PSFloat *bias_mul; /* Used by BLAS backpropagation, always set to 1 */
    int im2col_size;
} PSPrivateConvData;

/* Forward declarations */

PSActivationFunction PSGetActivationDerivative(PSActivationFunction func);
int PSConvolutionalForward(PSLayer *layer, ...);
int PSPool(PSLayer *layer, ...);
int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer *prev_layer,
                            PSGradient *lgradients, ...);
int PSPoolingBackprop(PSLayer *pooling_layer, PSLayer *convolutional_layer,
                      PSGradient *layer_gradients, ...);
PSScalarActivationFunction PSGetScalarActivationFunc(PSActivationFunction func);
int checkLayerForForward(PSLayer *layer);
PSGradient *createLayerGradients(PSLayer *layer);

/* Helper functions */

static PSFloat *im2col(PSFloat *inputs, int input_size, int channels,
                       int width, int height, int kernel_width,
                       int kernel_height, int padding, int stride, int dilation,
                       int *output_size, int *output_columns, int *output_rows)
{
    if (output_size != NULL) *output_size = 0;
    if (inputs == NULL || input_size <= 0) return NULL;
    if (dilation <= 0) dilation = 1;
    if (stride <= 0) stride = 1;
    if (padding < 0) padding = 0;
    if (height <= 0) height = width;
    if (kernel_height <= 0) kernel_height = kernel_width;
    int output_w = (width + 2 * padding - (dilation * (kernel_width - 1) + 1)) /
                   stride + 1;
    int output_h = (height + 2 * padding -  (dilation * (kernel_height - 1)+1))/
                   stride + 1;
    int channel_size = height * width;
    if (output_w <= 0 || output_h <= 0 || channel_size <= 0) goto invalid_size;
    int out_cols = kernel_width * kernel_height * channels;
    if (out_cols <= 0) goto invalid_size;
    int outsize = channels * output_w * output_h * kernel_width * kernel_height;
    if (outsize <= 0) goto invalid_size;
    int out_rows = outsize / out_cols;
    if (out_rows <= 0) goto invalid_size;
    PSFloat *output = malloc(outsize * sizeof(PSFloat));
    if (output == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    if (output_size != NULL) *output_size = outsize;
    if (output_columns != NULL) *output_columns = out_cols;
    if (output_rows != NULL) *output_rows = out_rows;
    PSFloat *output_p = output;
    PSFloat *inputs_p = inputs;
    for (int ch = 0; ch < channels; ch++) {
        for (int krow = 0; krow < kernel_height; krow++) {
            for (int kcol = 0; kcol < kernel_width; kcol++) {
                int input_row = -padding + krow * dilation;
                int output_rows = output_h;
                while (output_rows-- > 0) {
                    if (input_row < 0 || input_row >= height) {
                        int output_cols = output_w;
                        while (output_cols-- > 0) *(output_p++) = 0.0;
                    } else {
                        int input_col = -padding + kcol * dilation;
                        int output_col = output_w;
                        while (output_col-- > 0) {
                            PSFloat val = 0.0;
                            if (input_col >= 0 && input_col < width)
                                val = inputs_p[input_row * width + input_col];
                            *(output_p++) = val;
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
        inputs_p += channel_size;
    }
    return output;
invalid_size:
    PSErr(__func__, "invalid size");
    return NULL;
}

static PSFloat *col2im(PSFloat *input, int channels, int width, int height,
                       int kernel_width, int kernel_height, int padding,
                       int stride, int dilation, PSFloat *dest, int dest_size)
{
    if (input == NULL) return NULL;
    int outsize = height * width * channels;
    if (outsize <= 0) goto invalid_size;
    int output_h = (height + 2 * padding - (dilation * (kernel_height-1)+1)) /
                   stride + 1;
    int output_w = (width + 2 * padding - (dilation * (kernel_width-1)+1)) /
                   stride + 1;
    if (output_h <= 0 || output_w <= 0) goto invalid_size;
    PSFloat *output = dest;
    if (output == NULL) {
        output = calloc(outsize, sizeof(PSFloat));
        if (output == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
    } else {
        if (dest_size != outsize) goto invalid_size;
        memset(output, 0, outsize * sizeof(PSFloat));
    }
    int channel_size = height * width;
    PSFloat *output_p = output;
    PSFloat *input_p = input;
    for (int ch = 0; ch < channels; ch++) {
        for (int krow = 0; krow < kernel_height; krow ++) {
            for (int kcol = 0; kcol < kernel_width; kcol++) {
                int input_row = -padding + krow * dilation;
                int output_rows = output_h;
                while (output_rows-- > 0) {
                    if (input_row < 0 || input_row >= height)
                        input_p += output_w;
                    else {
                        int input_col = -padding + kcol * dilation;
                        int output_col = output_w;
                        while (output_col-- > 0) {
                            if (input_col >= 0 && input_col < width) {
                                int oidx = (input_row * width + input_col);
                                output_p[oidx] += *input_p;
                            }
                            input_p++;
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
        output_p += channel_size;
    }
    return output;
invalid_size:
    PSErr(__func__, "invalid size");
    return NULL;
}

static PSFloat *weights2col(PSFloat **lweights, int filter_width,
                            int filter_height, int filter_depth,
                            int out_depth, int *output_size,
                            int *output_columns, int *output_rows,
                            int transpose)
{
    if (output_size != NULL) *output_size = 0;
    if (lweights == NULL) return NULL;
    if (filter_height <= 0) filter_height = filter_width;
    int fsize = filter_width * filter_height * filter_depth;
    if (fsize <= 0) goto invalid_size;
    int rows = fsize;
    int cols = out_depth;
    int size = rows * cols;
    if (size <= 0) goto invalid_size;
    PSFloat *outputs = malloc(size * sizeof(PSFloat));
    if (outputs == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    if (output_size != NULL) *output_size = size;
    if (output_columns != NULL) *output_columns = cols;
    if (output_rows != NULL) *output_rows = rows;
    for (int col = 0; col < out_depth; col++) {
        PSFloat *weights = lweights[col];
        if (weights == NULL) {
            PSErr(__func__, "missing weights[%d]", col);
            free(outputs);
            if (output_size != NULL) *output_size = 0;
            return NULL;
        }
        if (!transpose) {
            memcpy(outputs + (col * fsize), weights, fsize * sizeof(PSFloat));
            continue;
        }
        for (int row = 0; row < fsize; row++)
            outputs[(row * cols) + col] = weights[row];
    }
    return outputs;
invalid_size:
    PSErr(__func__, "invalid size");
    return NULL;
}

/* Accelerated convolution with BLAS */

static int AcceleratedConvolve(PSLayer *layer, PSFloat *inputs, int input_size,
                               PSFloat *outputs)
{
    if (inputs == NULL) {
        PSErr(__func__, "NULL inputs");
        return 0;
    }
    if (outputs == NULL) {
        PSErr(__func__, "NULL outputs");
        return 0;
    }
    PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
    if (settings == NULL) {
        PSErr(__func__, "Layer[%d] has no settings");
        return 0;
    }
    PSPrivateConvData *privdata = GetConvPrivateData(layer);
    if (privdata == NULL) {
        privdata = calloc(1, sizeof(PSPrivateConvData));
        if (privdata == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        layer->private = privdata;
    }
    int i2c_size = 0, i2c_rows = 0, i2c_cols = 0, w2c_size = 0, w2c_cols = 0,
        w2c_rows = 0, success = 1;
    PSFloat *i2c = im2col(inputs, input_size, settings->input_depth,
                          settings->input_width, settings->input_height,
                          settings->filter_width,
                          settings->filter_height, settings->padding,
                          settings->stride, 1,
                          &i2c_size, &i2c_cols, &i2c_rows);
    if (i2c == NULL) return 0;
    if (privdata->im2col != NULL) free(privdata->im2col);
    privdata->im2col = i2c;
    privdata->im2col_size = i2c_size;
    PSFloat *w2c = privdata->w2col;
    if (w2c == NULL) {
        w2c = weights2col(layer->weights, settings->filter_width,
                          settings->filter_height, settings->input_depth,
                          layer->output_depth, &w2c_size, &w2c_cols,
                          &w2c_rows, 0);
        privdata->w2col = w2c;
    }
    if (w2c == NULL) {
        success = 0;
        goto final;
    }
    int m = layer->output_depth;
    int n = layer->size / layer->output_depth;
    int k = settings->filter_width * settings->filter_height *
            settings->input_depth;
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    success = PSMatMul(w2c, i2c, outputs, m, n, k, &opts);
final:
    return success;
}

static int AcceleratyedConvBackprop(PSLayer *layer, PSGradient *gradient) {
    int success = 1;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) return 0;
    PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
    if (settings == NULL) {
        PSErr(__func__, "Layer[%d] has no settings");
        return 0;
    }
    PSPrivateConvData *privdata = GetConvPrivateData(layer);
    if (privdata == NULL) {
        PSErr(__func__, "Layer[%d] has no private data");
        return 0;
    }
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    PSFloat *delta = layer->delta;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int feature_size = layer->size / layer->output_depth;
    int ksize = settings->filter_width * settings->filter_height *
                settings->input_depth;
    if (use_bias) {
        /* Update gradient biases */
        if (privdata->bias_mul == NULL) {
            privdata->bias_mul = malloc(feature_size * sizeof(PSFloat));
            success = (privdata->bias_mul != NULL);
            if (!success) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            PSVectorFill(privdata->bias_mul, 1.0, feature_size, &opts);
        }
        PSMathOpts mmopts = {
            .acceleration = layer->model->acceleration,
            .store_mode = PS_STORE_MODE_ADD
        };
        success = PSMatMul(
            delta, privdata->bias_mul, gradient->biases, layer->output_depth, 1,
            feature_size, &mmopts
        );
        if (!success) goto final;
    }
    int i2c_size = privdata->im2col_size;
    PSFloat *i2c = privdata->im2col;
    if (i2c == NULL) {
        PSFloat *inputs = PSGetStates(previous, 0);
        success = inputs != NULL;
        if (!success) goto final;
        i2c = im2col(inputs, previous->size, settings->input_depth,
                     settings->input_width, settings->input_height,
                     settings->filter_width,
                     settings->filter_height, settings->padding,
                     settings->stride, 1, &i2c_size, NULL, NULL);
        success = (i2c != NULL);
        if (!success) goto final;
    }
    /* Update gradient weights */
    int m = layer->output_depth, n = ksize, k = feature_size;
    PSMathOpts mmopts = {
        .acceleration = layer->model->acceleration,
        .store_mode = PS_STORE_MODE_ADD,
        .transpose = 2
    };
    success = PSMatMul(delta, i2c, gradient->weights, m, n, k, &mmopts);
    if (!success) goto final;
    if (previous->delta == NULL) goto final;
    /* Update previous layer delta */
    PSFloat *w2c = privdata->w2col;
    if (w2c == NULL) {
        w2c = weights2col(layer->weights, settings->filter_width,
                          settings->filter_height, settings->input_depth,
                          layer->output_depth, NULL, NULL, NULL, 0);
        privdata->w2col = w2c;
    }
    success = (w2c != NULL);
    if (!success) goto final;
    m = ksize;
    n = feature_size;
    k = layer->output_depth;
    mmopts.transpose = 1;
    mmopts.store_mode = PS_STORE_MODE_SET;
    success = PSMatMul(w2c, delta, i2c, m, n, k, &mmopts);
    if (!success) goto final;
    PSFloat *prev_delta = col2im(
        i2c, settings->input_depth, settings->input_width,
        settings->input_height, settings->filter_width,
        settings->filter_height, settings->padding, settings->stride,
        1, previous->delta, previous->size
    );
    success = (prev_delta != NULL);
    if (!success) goto final;
final:
    free(privdata->im2col);
    privdata->im2col = NULL;
    return success;
}

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
    return 1;
}

void PSDeleteConvolutionalLayer(PSLayer *layer) {
    if (layer->extra != NULL) free(layer->extra);
    layer->extra = NULL;
    if (layer->private != NULL) {
        PSPrivateConvData *privdata = (PSPrivateConvData *) layer->private;
        free(privdata->im2col);
        free(privdata->w2col);
        free(privdata->bias_mul);
        free(layer->private);
        layer->private = NULL;
    }
}

void PSBeforeConvolutionalBatchTraining(PSLayer *layer) {
    if (layer == NULL) return;
    PSPrivateConvData *privdata = GetConvPrivateData(layer);
    if (privdata != NULL) {
        if (privdata->im2col != NULL) free(privdata->im2col);
        if (privdata->w2col != NULL) free(privdata->w2col);
        privdata->im2col = privdata->w2col = NULL;
    }
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
            if (layer != NULL && layer->model != NULL)
                acceleration = layer->model->acceleration;
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

int PSInitConvolutionalLayer(PSModel *model, PSLayer *layer,
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
    layer->before_batch_training = PSBeforeConvolutionalBatchTraining;
    layer->extra = calloc(1, sizeof(PSConvolutionalSettings));
    if (layer->extra == NULL) goto memerr;
    layer->private = calloc(1, sizeof(PSPrivateConvData));
    if (layer->private == NULL) goto memerr;
    PSConvolutionalSettings *settings = (PSConvolutionalSettings *)layer->extra;
    PSLayer *previous = model->layers[index - 1];
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
    layer->states = PSMatrixZeros(2, 1, size);
    if (layer->states == NULL) goto memerr;
    layer->biases = malloc(layer->output_depth * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->weights = malloc(layer->output_depth * sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    PSFloat wrange = PSSqrt(1.0 / weights_size);
    int i, use_relu = (layer->activate == PSRelu), rand_bias = 0;
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
    }
    layer->forward = PSConvolutionalForward;
    layer->backprop = PSConvolutionalBackprop;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
err:
    return 0;
}

int PSInitPoolingLayer(PSModel *model, PSLayer *layer, PSLayerDef *layer_def) {
    int index = layer->index;
    layer->weights = NULL;
    layer->biases = NULL;
    layer->on_delete = PSDeleteConvolutionalLayer;
    layer->on_copy = PSConvolutionalLayerCopy;
    layer->flags |= FLAG_NON_TRAINABLE;
    PSLayer *previous = model->layers[index - 1];
    PSLayerDef default_def = {
        .stride = 1, .filter_width = 2, .filter_height = 2
    };
    if (layer_def == NULL) layer_def = &default_def;
    layer->extra = calloc(1, sizeof(PSConvolutionalSettings));
    if (layer->extra == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSConvolutionalSettings *settings = (PSConvolutionalSettings*)layer->extra;
    layer->output_depth = previous->output_depth;
    if (layer->output_depth < 1) layer->output_depth = 1;
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
    if (input_w == 0 || input_h == 0) {
        if (input_w == 0 && input_h == 0) {
            input_w = previous->size;
            input_h = 1;
        } else if (input_w == 0) {
            if ((previous->size % (int) input_h) != 0) {
                PSErr(__func__, "invalid input height %d", input_h);
                return 0;
            }
            input_w = previous->size / input_h;
        } else if (input_h == 0) {
            if ((previous->size % (int) input_w) != 0) {
                PSErr(__func__, "invalid input width %d", input_w);
                return 0;
            }
            input_h = previous->size / input_w;
        }
    }
    settings->input_width = input_w;
    settings->input_height = input_h;
    output_w = PSCalculatePoolingSide(input_w, settings->filter_width);
    output_h = PSCalculatePoolingSide(input_h, settings->filter_width);
    layer->output_columns = output_w;
    layer->output_rows = output_h;
    int area = (output_w * output_h);
    int size = area * layer->output_depth;
    layer->size = size;
    layer->states = PSMatrixZeros(2, 1, size);
    if (layer->states == NULL) goto memerr;
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->forward = PSPool;
    layer->backprop = PSPoolingBackprop;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Forward Functions */

int PSConvolutionalForward(PSLayer *layer, ...) {
    PSModel *model = NULL;
    if (!checkLayerForForward(layer)) goto failed;
    if (PSHandleSequenceAtOnce(layer)) {
        PSErr(
            NULL, "Layer[%d]: sequence input not currently supported in "
            "convolutonal layers: remove FLAG_USE_SEQUENCES.", layer->index
        );
        return 0;
    }
    model = layer->model;
    int do_dump =
        (model->training != NULL && model->training->debug_dump_to != NULL);
    PSDebugStepInfo dbginfo = {
        .model = model,
        .layer = layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_FORWARD
    };
    PSLayer *previous = model->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        goto failed;
    }
    if (previous->flags & FLAG_ONEHOT) {
        PSErr(NULL, "Layer[%d]: convolutional layer cannot be fed with"
              "onehot input", layer->index);
        goto failed;
    }
    PSConvolutionalSettings *settings = PSGetConvolutionalSettings(layer);
    if (settings == NULL) {
        PSErr(NULL, "Layer[%d]: convolutional layer has no settings",
              layer->index);
        goto failed;
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
    int use_acceleration = (
        model->acceleration != PSAcceleration_None && !is_recurrent
    );
    int previous_feature_size = 0;
    if (previous->output_depth == 0) previous->output_depth = 1;
    previous_feature_size = previous->size / previous->output_depth;
    if (use_acceleration) {
        PSMathOpts mopts = {.acceleration = model->acceleration};
        PSFloat *inputs = PSGetStates(previous, t);
        PSFloat *outputs = PSGetStates(layer, t);
        if (inputs == NULL || outputs == NULL) {
            PSErr(NULL, "Layer[%d]: missing inputs and/or outputs");
            goto failed;
        }
        int ok = AcceleratedConvolve(layer, inputs, previous->size, outputs);
        if (!ok) {
            PSErr(NULL, "Layer[%d]: convolutional layer failed forward",
                  layer->index);
            goto failed;
        }
        if (use_bias) {
            for (i = 0; i < layer->output_depth; i++) {
                PSFloat bias = layer->biases[i];
                PSFloat *feature_map = outputs + (i * feature_size);
                PSAddVectorScalar(feature_map, bias, feature_map,
                                  feature_size, &mopts);
            }
        }
        if (layer->activate != NULL)
            layer->activate(outputs, outputs, layer->size, &mopts);
        return 1;
    }
    PSScalarActivationFunction activate = NULL;
    if (layer->activate != NULL)
        activate = PSGetScalarActivationFunc(layer->activate);
    for (i = 0; i < layer->output_depth; i++) {
        if (do_dump && i > 1) do_dump = 0;
        PSFloat bias = (use_bias ? layer->biases[i] : 0.0);
        PSFloat *weights = layer->weights[i];
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
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
                    model,
                    "#### Previous Feature[%d], offset=%d, weight_offset="
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
            if (activate != NULL) state = activate(state);
            int ok = PSSetState(layer, state, idx, t);
            if (!ok) {
                PSErr(
                    NULL, "Failed to set state on layer %d, neuron %d",
                    layer->index, idx
                );
                PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
                return 0;
            }
        }
    }
    return 1;
failed:
    PSModelSetStatus(model, STATUS_ERROR, NULL);
    return 0;
}

int PSPool(PSLayer *layer, ...) {
    PSModel *model = layer->model;
    if (PSHandleSequenceAtOnce(layer)) {
        PSErr(
            NULL, "Layer[%d]: sequence input not currently supported in "
            "pooling layers: remove FLAG_USE_SEQUENCES.", layer->index
        );
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot forward on layer 0!");
        return 0;
    }
    PSLayer *previous = model->layers[layer->index - 1];
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
        (model->training != NULL && model->training->debug_dump_to != NULL);
    PSDebugStepInfo dbginfo = {
        .model = model,
        .layer = layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_FORWARD
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
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * settings->filter_height;
            int r_col = col * settings->filter_width;
            int max_x = settings->filter_width + r_col;
            int max_y = settings->filter_height + r_row;
            PSFloat max = 0.0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size * i);
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
                PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
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
    PSModel *model = (PSModel *) pooling_layer->model;
    int do_dump = 0;
    if (model != NULL) {
        do_dump = (
            model->training != NULL && model->training->debug_dump_to != NULL
        );
    }
    PSDebugStepInfo dbginfo = {
        .model = model,
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
    PSScalarActivationFunction derivative = NULL;
    if (convolutional_layer->derivative != NULL)
        derivative = PSGetScalarActivationFunc(convolutional_layer->derivative);
    for (i = 0; i < pooling_layer->output_depth; i++) {
        if (do_dump && i > 1) do_dump = 0;
        row = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
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
                    int nidx = ((y * input_w) + x) + (prev_feat_size * i);
                    if (do_dump) DumpPoolBackpropStep(
                        col, row, r_col, r_row, max_x, max_y,
                        convolutional_layer, i, nidx, x, y
                    );
                    PSFloat s = PSGetState(convolutional_layer, nidx, t);
                    PSFloat dv = (s < pool_state ? 0 : d);
                    if (dv != 0 && derivative != NULL)
                        dv *= derivative(pool_state);
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
    PSModel *model = (PSModel *) convolutional_layer->model;
    if (model == NULL) return 0;
    if (gradient == NULL) return 0;
    int is_recurrent = PSIsRecurrent(convolutional_layer), t = 0;
    int use_acceleration = (
        model->acceleration != PSAcceleration_None && !is_recurrent
    );
    if (use_acceleration) {
        if (!AcceleratyedConvBackprop(convolutional_layer, gradient)) {
            PSErr(NULL, "Layer[%d]: failed backprop using BLAS acceleration",
                  convolutional_layer->index);
            return 0;
        }
        return 1;
    }
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
    int do_dump = 0;
    if (model != NULL) {
        do_dump = (
            model->training != NULL && model->training->debug_dump_to != NULL
        );
    }
    PSDebugStepInfo dbginfo = {
        .model = model,
        .layer = convolutional_layer,
        .func = __func__,
        .training_phase = TRAINING_PHASE_BACKPROP
    };
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
                    model,
                    "#### Previous Feature[%d], offset=%d, weight_offset="
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
                            PSNeuron neuron;
                            if (!PSGetNeuron(convolutional_layer, idx, &neuron))
                            {
                                PSErrNN(NULL, NULL, convolutional_layer,
                                        "failed to read neuron %d", idx);
                                return 0;
                            }
                            prev_delta[nidx] += (d * neuron.weights[widx]);
                        }
                        widx++;
                    }
                }
            }
        }
    }
    return 1;
}

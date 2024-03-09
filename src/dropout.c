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

#include <stdarg.h>
#include <string.h>
#include "dropout.h"
#include "maths.h"
#include "log.h"

#define PSGetDropoutData(layer) ((PSDropoutData *) layer->extra);
#define UNUSED(V) ((void) V)

typedef struct PSDropoutData {
    PSFloat dropout;
    PSFloat *dropout_mask;
} PSDropoutData;

/* Forward declarations */

int PSDropoutForward(PSLayer *layer, ...);
int PSDropoutBackprop(PSLayer *layer, PSLayer *previous_layer,
                      PSGradient *gradients, ...);
int checkLayerForForward(PSLayer *layer);
int PSBeforeSequenceForward(PSLayer *layer, long seqlen, long t);

/* Dropout Layer functions */

PSLayer *PSGetDropoutLayer(PSLayer *parent_layer) {
    PSLayer *next = PSGetNextLayer(parent_layer);
    if (next == NULL) return NULL;
    if (Dropout != next->type) return NULL;
    return next;
}

PSFloat PSGetDropout(PSLayer *dropout_layer) {
    if (dropout_layer == NULL) return 0.0;
    if (Dropout != dropout_layer->type) return 0.0;
    PSDropoutData *data = PSGetDropoutData(dropout_layer);
    if (data == NULL) return 0.0;
    PSFloat dropout = data->dropout;
    if (dropout < 0.0) dropout = data->dropout = 0.0;
    if (dropout > 1.0) dropout = data->dropout = 1.0;
    return dropout;
}

void PSSetDropout(PSLayer *dropout_layer, PSFloat dropout) {
    if (dropout_layer == NULL) return;
    if (Dropout != dropout_layer->type) return;
    PSDropoutData *data = PSGetDropoutData(dropout_layer);
    if (data == NULL) return;
    data->dropout = dropout;
}

int PSInitDropoutMask(PSLayer *layer, long seqlen, int retain_previous) {
    UNUSED(retain_previous);
    if (layer == NULL) return 0;
    PSDropoutData *data = PSGetDropoutData(layer);
    if (data == NULL) {
        PSErr(__func__, "Layer[%d]: missing Dropout data", layer->index);
        return 0;
    }
    if (seqlen > 0) {
        if (data->dropout_mask != NULL) free(data->dropout_mask);
        data->dropout_mask = malloc(layer->size * seqlen * sizeof(PSFloat));
        if (data->dropout_mask == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    } else if (data->dropout_mask != NULL) {
        free(data->dropout_mask);
        data->dropout_mask = NULL;
    }
    return 1;
}

int PSResizeDropoutMask(PSLayer *layer, long seqlen, long prevlen) {
    if (layer == NULL) return 0;
    PSDropoutData *data = PSGetDropoutData(layer);
    if (data == NULL) {
        PSErr(__func__, "Layer[%d]: missing Dropout data", layer->index);
        return 0;
    }
    size_t size = (size_t) layer->size * (size_t) seqlen * sizeof(PSFloat);
    PSFloat *dropout_mask = realloc(data->dropout_mask, size);
    if (dropout_mask == NULL) {
        free(data->dropout_mask);
        data->dropout_mask = NULL;
        PSPrintMemoryErrorMsg();
        PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
        return 0;
    }
    long diff = seqlen - prevlen;
    if (diff > 0) {
        PSFloat *new_segment = dropout_mask + (prevlen * layer->size);
        memset(new_segment, 0, (size_t) diff * layer->size * sizeof(PSFloat));
    }
    data->dropout_mask = dropout_mask;
    return 1;
}

PSFloat *PSGetDropoutMask(PSLayer *layer, long t) {
    if (layer == NULL) return NULL;
    PSDropoutData *data = PSGetDropoutData(layer);
    if (data == NULL) return NULL;
    PSFloat *mask = data->dropout_mask;
    if (PSUseSequences(layer)) {
        if (t < 0) return NULL;
        long cur_seqlen = PSStateSequenceLength(layer);
        if (t >= cur_seqlen) {
            if (!PSResizeDropoutMask(layer, t + 1, cur_seqlen)) return NULL;
            mask = data->dropout_mask;
            if (mask == NULL) return NULL;
        } else if (mask == NULL) {
            if (!PSInitDropoutMask(layer, cur_seqlen, 0)) return NULL;
            mask = data->dropout_mask;
            if (mask == NULL) return NULL;
        }
        mask += (t * layer->size);
    } else if (mask == NULL) {
        mask = malloc(layer->size * sizeof(PSFloat));
        if (mask == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
        data->dropout_mask = mask;
    }
    return mask;
}


static void deleteDropoutData(PSDropoutData *data) {
    if (data == NULL) return;
    free(data->dropout_mask);
    free(data);
}

int PSDropoutLayerCopy(PSLayer *layer, PSLayer *src) {
    PSDropoutData *srcdata = PSGetDropoutData(src);
    PSDropoutData *dstdata = PSGetDropoutData(layer);
    if (dstdata == NULL && srcdata != NULL) {
        dstdata = calloc(1, sizeof(*dstdata));
        if (dstdata == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        layer->extra = dstdata;
    } else if (srcdata == NULL) {
        if (dstdata != NULL) deleteDropoutData(dstdata);
        layer->extra = NULL;
        return 1;
    }
    dstdata->dropout = srcdata->dropout;
    long seqlen = 1;
    if (PSUseSequences(src)) seqlen = PSStateSequenceLength(layer);
    if (seqlen > 0 && srcdata->dropout_mask != NULL) {
        size_t size = src->size * seqlen * sizeof(PSFloat);
        dstdata->dropout_mask = malloc(size);
        if (dstdata == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(dstdata->dropout_mask, srcdata->dropout_mask, size);
    }
    return 1;
}

void PSDeleteDropoutLayer(PSLayer *layer) {
    if (layer == NULL) return;
    PSDropoutData *data = PSGetDropoutData(layer);
    deleteDropoutData(data);
    layer->extra = NULL;
}

int PSInitDropoutLayer(PSModel *model, PSLayer *layer,
                       PSLayerDef *layer_def)
{
    UNUSED(model);
    if (layer->index == 0) {
        PSErr(NULL, "Dropout layer cannot be the first layer");
        return 0;
    }
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "Dropout has no previous layer");
        return 0;
    }
    if (previous->size == 0) {
        PSErr(NULL, "Dropout layer has previous layer with size zero");
        return 0;
    }
    PSFloat dropout = 0.0;
    if (layer_def != NULL) dropout = layer_def->dropout;
    if (dropout <= 0.0) {
        PSErr(NULL, "Dropout property is mandatory for Dropout layer");
        return 0;
    }
    if (layer->extra != NULL) {
        PSErr(NULL, "Dropout layer has already initialized data");
        return 0;
    }
    if (dropout > 1.0) dropout = 1.0;
    layer->flags |= PS_FLAG_NON_TRAINABLE;
    if (PSIsRecurrent(previous)) layer->flags |= PS_FLAG_RECURRENT;
    if (PSHandleSequenceAtOnce(previous))
        layer->flags |= PS_FLAG_USE_SEQUENCES;
    layer->onCopy = PSDropoutLayerCopy;
    layer->onDelete = PSDeleteDropoutLayer;
    layer->onStatesInit = PSInitDropoutMask;
    layer->onStatesResize = PSResizeDropoutMask;
    if (PSIsRecurrent(previous)) layer->flags |= PS_FLAG_RECURRENT;
    layer->size = previous->size;
    if (previous->output_columns > 0)
        layer->output_columns = previous->output_columns;
    if (previous->output_rows > 0)
        layer->output_rows = previous->output_rows;
    if (previous->output_depth > 0)
        layer->output_depth = previous->output_depth;
    if (layer->biases != NULL) free(layer->biases);
    layer->biases = NULL;
    if (layer->weights != NULL) {
        for (int i = 0; i < layer->weight_types; i++) {
            PSMatrix weights = layer->weights[i];
            if (weights != NULL) PSMatrixFree(weights);
        }
        free(layer->weights);
    }
    layer->weights = NULL;
    layer->delta = PSMatrixZeros(2, 1, layer->size);
    if (layer->delta == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSDropoutData *data = calloc(1, sizeof(*data));
    if (data == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    data->dropout = dropout;
    layer->extra = data;
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->states = PSMatrixZeros(2, 1, layer->size);
    layer->forward = PSDropoutForward;
    layer->backprop = PSDropoutBackprop;
    return 1;
}

/* Forward */

int PSDropoutForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSModel *model = layer->model;
    int success = 1, is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer);
    long t = 0, seqlen = 1;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (is_recurrent || handles_seq) {
        va_list args;
        va_start(args, layer);
        seqlen = va_arg(args, long);
        if (is_recurrent) t = va_arg(args, long);
        va_end(args);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    PSFloat *inputs = PSLayerStates(previous, t),
            *outputs = PSLayerStates(layer, t);
    if (inputs == NULL) {
        PSErr(NULL, "Layer[%d]: Dropout layer has no inputs");
        return 0;
    }
    if (outputs == NULL) {
        PSErr(NULL, "Layer[%d]: Dropout layer has no outputs");
        return 0;
    }
    PSFloat dropout = PSGetDropout(layer);
    PSMathOpts mopts = {.acceleration = layer->model->acceleration};
    long len = layer->size;
    if (handles_seq && seqlen > 1) len *= seqlen;
    if (PSModelGetStatus(model) != PS_STATUS_TRAINING) {
        PSVectorCopy(outputs, inputs, len);
        PSMultiplyVectorScalar(outputs, dropout, outputs, len, &mopts);
        return 1;
    }
    PSFloat *dropout_mask = PSGetDropoutMask(layer, t);
    if (dropout_mask == NULL) {
        PSErr(NULL, "Layer[%d]: Dropout layer has no dropout mask");
        return 0;
    }
    for (long i = 0; i < len; i++) {
        PSFloat r = PSNormalizedRandom();
        PSFloat input = inputs[i];
        int dropped = (r < dropout);
        if (dropped) outputs[i] = 0.0;
        else outputs[i] = input;
        dropout_mask[i] = (PSFloat) !dropped;
    }
    return success;
}

/* Backpropagation */

int PSDropoutBackprop(PSLayer *layer, PSLayer *previous_layer,
                      PSGradient *gradients, ...)
{
    if (layer == NULL) return 0;
    if (previous_layer == NULL) return 0;
    if (previous_layer->delta == NULL) return 1;
    if (layer->delta == NULL) return 0;
    int is_recurrent = PSIsRecurrent(layer),
        handles_seq = PSHandleSequenceAtOnce(layer),
        success = 1;
    long t = 0, seqlen = 1;
    if (is_recurrent) {
        va_list args;
        va_start(args, gradients);
        t = va_arg(args, long);
        va_end(args);
    } else if (handles_seq) seqlen = PSStateSequenceLength(layer);
    PSFloat *dropout_mask = PSGetDropoutMask(layer, t);
    if (dropout_mask == NULL) {
        PSErr(NULL, "Layer[%d]: Dropout layer has no dropout mask");
        return 0;
    }
    if (seqlen < 1) seqlen = 1;
    long len = layer->size;
    if (handles_seq && seqlen > 1) len *= seqlen;
    PSMathOpts mopts = {
        .acceleration = layer->model->acceleration,
        .store_mode = PS_STORE_MODE_ADD
    };
    PSMultiplyVectors(
        layer->delta, dropout_mask, previous_layer->delta, len, &mopts
    );
    return success;
}

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

#include <stdarg.h>
#include <string.h>
#include "positional_encoding.h"
#include "maths.h"
#include "log.h"

#define UNUSED(V) ((void) V)
#define DEFAULT_POSITIONAL_SEQLEN 1024
#define DEFAULT_POSITIONAL_BASE   10000

#define PSGetPositionalSettings(layer) ((PSPositionalSettings *) layer->extra)

typedef struct PSPositionalSettings {
    int base;
} PSPositionalSettings;

/* Forward declarations. */
int checkLayerForForward(PSLayer *layer);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
int PSPositionalForward(PSLayer *layer, ...);
int PSPositionalBackprop(PSLayer *layer, PSLayer *previous_layer,
                      PSGradient *gradients, ...);

int PSGetPositionalEncodingLength(PSLayer *layer) {
    if (layer == NULL || layer->weights == NULL || layer->weights[0] == NULL)
        return 0;
    return PSMatrixDim(layer->weights[0], 0);
}

int PSGetPositionalEncodingBase(PSLayer *layer) {
    if (layer == NULL) return 0;
    PSPositionalSettings *settings = PSGetPositionalSettings(layer);
    if (settings == NULL) return 0;
    return settings->base;
}

static void deletePositionalLayer(PSLayer *layer) {
    if (layer == NULL) return;
    free(layer->extra);
    layer->extra = NULL;
}

static int copyPositionalLayer(PSLayer *layer, PSLayer *src) {
    if (layer == NULL || src == NULL) return 0;
    PSPositionalSettings *srcsettings = PSGetPositionalSettings(src);
    if (srcsettings != NULL) {
        free(layer->extra);
        layer->extra = malloc(sizeof(PSPositionalSettings));
        if (layer->extra == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(layer->extra, srcsettings, sizeof(PSPositionalSettings));
    } else {
        free(layer->extra);
        layer->extra = NULL;
    }
    return 1;
}

static int initOrResizePositionalEncodings(PSLayer *layer, uint32_t seqlen) {
    uint32_t capacity = (uint32_t) PSGetPositionalEncodingLength(layer);
    if (seqlen > capacity) {
        PSPositionalSettings *settings = PSGetPositionalSettings(layer);
        if (settings == NULL || layer->weights == NULL) return 0;
        PSMatrix encodings = PSGetPositionalEncoding(
            seqlen, layer->size, settings->base
        );
        if (encodings == NULL) return 0;
        PSMatrixFree(layer->weights[0]);
        layer->weights[0] = encodings;
    }
    return 1;
}

int PSInitPositionalStates(PSLayer *layer, uint32_t seqlen, int retain_prev) {
    UNUSED(retain_prev);
    return initOrResizePositionalEncodings(layer, seqlen);
}

int PSResizePositionalStates(PSLayer *layer, uint32_t seqlen, uint32_t prevl) {
    UNUSED(prevl);
    return initOrResizePositionalEncodings(layer, seqlen);
}

PSMatrix PSGetPositionalEncoding(int seqlen, int size, int base) {
    if (base <= 0) base = DEFAULT_POSITIONAL_BASE;
    if (seqlen <= 0) seqlen = DEFAULT_POSITIONAL_SEQLEN;
    if (size <= 0) {
        PSErr(__func__, "size must be > 0");
        return NULL;
    }
    PSMatrix encodings = PSMatrixZeros(2, seqlen, size);
    if (encodings == NULL) {
        PSErr(__func__, "could not create %s positional encodings of size %d",
              seqlen, size);
        return NULL;
    }
    int steps = size / 2, i, k;
    PSFloat *enc_p = encodings;
    for (k = 0; k < seqlen; k++) {
        for (i = 0; i < steps; i++) {
            PSFloat denominator = PSPow(base, 2 * (i / (PSFloat) size));
            PSFloat n = (PSFloat) k / denominator;
            int idx = i * 2;
            enc_p[idx] = PSSin(n);
            enc_p[idx + 1] = PSCos(n);
        }
        enc_p += size;
    }
    return encodings;
}

int PSInitPositionalLayer(PSLayer *layer, PSLayerDef *layer_def) {
    layer->onDelete = deletePositionalLayer;
    layer->onCopy = copyPositionalLayer;
    layer->onStatesInit = PSInitPositionalStates;
    layer->onStatesResize = PSResizePositionalStates;
    if (layer->index == 0) {
        PSErr(NULL, "PositionalEncoding layer cannot be the first layer");
        return 0;
    }
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "PositionalEncoding has no previous layer");
        return 0;
    }
    if (previous->size == 0) {
        PSErr(NULL, "PositionalEncoding layer has previous layer with "
              "size zero");
        return 0;
    }
    PSModel *model = layer->model;
    if (!PSHandleSequenceAtOnce(model)) {
        PSErr(NULL, "PositionalEncoding layer can only be used in models "
              "handling whole sequences at onece");
        return 0;
    }
    layer->flags |= FLAG_USE_SEQUENCES;
    layer->flags &= ~((unsigned) FLAG_RECURRENT);
    layer->flags |= (FLAG_NO_BIAS | FLAG_NON_TRAINABLE);
    int onehot_input = (
        previous->flags & FLAG_ONEHOT ||
        ((model->flags & FLAG_ONEHOT) && previous->size == 1)
    );
    int min_capacity = 0;
    if (onehot_input) {
        if (layer->size <= 0) {
            PSErr(NULL, "layer size must be defined in PositionalEncoding "
                  "layers when previous layer in onehot");
            return 0;
        }
        min_capacity = PSGetOneHotLayerVectorSize(previous);
    } else layer->size = previous->size;
    layer->extra = calloc(1, sizeof(PSPositionalSettings));
    if (layer->extra == NULL) goto memerr;
    layer->states = PSMatrixZeros(2, 1, layer->size);
    if (layer->states == NULL) goto memerr;
    int success = 1;
    int capacity = min_capacity, base = 0;
    if (layer_def != NULL) {
        capacity = layer_def->positional_initial_capacity;
        if (capacity < min_capacity) capacity = min_capacity;
        base = layer_def->positional_base;
    }
    if (capacity <= 0) capacity = DEFAULT_POSITIONAL_SEQLEN;
    if (base <= 0) base = DEFAULT_POSITIONAL_BASE;
    PSPositionalSettings *settings = (PSPositionalSettings *) layer->extra;
    settings->base = base;
    layer->weight_types = 1;
    layer->weights = calloc(1, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weights[0] = PSGetPositionalEncoding(capacity, layer->size, base);
    success = (layer->weights[0] != NULL);
    if (!success) {
        PSErr(NULL, "could not initialize positional encodings");
        goto final;
    }
    layer->forward = PSPositionalForward;
    layer->backprop = PSPositionalBackprop;
final:
    return success;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Forward */

int PSPositionalForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    int success = 1, t = 0, seqlen = 1, i;
    PSLayer *previous = PSGetPreviousLayer(layer);
    va_list args;
    va_start(args, layer);
    seqlen = va_arg(args, int);
    va_end(args);
    if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    int capacity = PSGetPositionalEncodingLength(layer);
    if (seqlen > capacity) {
        if (!initOrResizePositionalEncodings(layer,  seqlen)) {
            PSErrNN(__func__, NULL, layer,
                    "failed to resize positional encodings to %d,%d",
                    seqlen, layer->size);
            return 0;
        }
    }
    PSMatrix inputs = NULL, tmpinputs = NULL;
    int onehot_input = (
        previous->flags & FLAG_ONEHOT ||
        ((layer->model->flags & FLAG_ONEHOT) && previous->size == 1)
    );
    PSMatrix weights = layer->weights[0];
    if (!onehot_input) inputs = previous->states;
    else {
        if (seqlen != PSStateSequenceLength(previous)) {
            PSErrNN(__func__, NULL, layer,
                    "previous layer sequence length mismatch");
            success = 0;
            goto final;
        }
        if (previous->states == NULL) {
            PSErrNN(__func__, NULL, layer, "previous layer has no states");
            success = 0;
            goto final;
        }
        tmpinputs = PSMatrixZeros(2, seqlen, layer->size);
        success =  (tmpinputs != NULL);
        if (!success) {
            PSErrNN(__func__, NULL, layer,
                    "could not create embeddings for onehot inputs");
            goto final;
        }
        PSFloat *dest_p = tmpinputs, *inputs_p = previous->states;
        for (i = 0; i < seqlen; i++) {
            long index = (long) *inputs_p;
            if (index >= capacity) {
                success = initOrResizePositionalEncodings(layer,  index + 1);
                if (!success) {
                    PSErrNN(__func__, NULL, layer,
                            "failed to resize positional encodings to %d,%d",
                            index + 1, layer->size);
                    goto final;
                }
                weights = layer->weights[0];
            }
            PSFloat *encodings = weights + (index * layer->size);
            PSVectorCopy(dest_p, encodings, layer->size);
            dest_p += layer->size;
            inputs_p += previous->size;
        }
        inputs = tmpinputs;
    }
    success = (inputs != NULL);
    if (!success) {
        PSErrNN(__func__, NULL, layer, "no inputs");
        goto final;
    }
    PSFloat *dest_p = layer->states;
    PSFloat *inputs_p = inputs;
    PSFloat *encodings = weights;
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    for (i = 0; i < seqlen; i++) {
        PSAddVectors(inputs_p, encodings, dest_p, layer->size, &opts);
        dest_p += layer->size;
        inputs_p += layer->size;
        encodings += layer->size;
    }
final:
    PSMatrixFree(tmpinputs);
    return success;
}

/* Backpropagation */

int PSPositionalBackprop(PSLayer *layer, PSLayer *previous_layer,
                         PSGradient *gradients, ...)
{
    UNUSED(gradients);
    PSMatrix delta = layer->delta;
    if (previous_layer == NULL) previous_layer = PSGetPreviousLayer(layer);
    if (previous_layer == NULL) return 0;
    PSMatrix prev_delta = previous_layer->delta;
    if (delta == NULL || prev_delta == NULL) return 0;
    uint64_t len = PSMatrixLength(delta);
    if (PSMatrixLength(prev_delta) != len) {
        PSErrNN(__func__, NULL, layer, "previous delta size mismatch");
        return 0;
    }
    PSVectorCopy(prev_delta, delta, len);
    return 1;
}

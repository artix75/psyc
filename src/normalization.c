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

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "normalization.h"
#include "psyc.h"
#include "log.h"
#include "maths.h"
#include "config.h"

#define GetNormalizationCache(layer) \
    ((PSNormalizationLayerCache*) layer->private)
#define UNUSED(V) ((void) V)

typedef struct {
    PSFloat *normalized_values;
    PSFloat mean;
    PSFloat variance;
    PSFloat stddev;
} PSNormalizationLayerCache;

/* Forward declarations */
int PSNormalizationFeedforward(PSLayer *layer, ...);
int PSNormalizationBackprop(PSLayer *layer, PSLayer *previous,
                            PSGradient *gradient, ...);
int PSInitNormalizationCache(PSLayer *layer, uint32_t steps,
                             int retain_previous);
int checkLayerForFeedforward(PSLayer *layer);
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
int PSBeforeSequenceFeedforward(PSLayer *layer, int seqlen, int t);

/* Layer functions */

static void deleteNormalizationCache(PSNormalizationLayerCache *cache,
                                     uint32_t seqlen)
{
    if (cache == NULL) return;
    if (seqlen <= 0) seqlen = 1;
    for (uint32_t i = 0; i < seqlen; i++) {
        PSNormalizationLayerCache *cache_t = cache + i;
        if (cache_t == NULL) continue;
        free(cache_t->normalized_values);
    }
    free(cache);
}

static void deleteNormalizationLayerCache(PSLayer *layer) {
    if (layer == NULL) return;
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (cache == NULL) return;
    int seqlen = 1;
    if (PSUseSequences(layer)) seqlen = PSStateSequenceLength(layer);
    if (seqlen <= 0) seqlen = 1;
    deleteNormalizationCache(cache, seqlen);
    layer->private = NULL;
}

static void deleteNormalizationLayer(PSLayer *layer) {
    deleteNormalizationLayerCache(layer);
    free(layer->extra);
    layer->extra = NULL;
}

static int copyNormalizationLayer(PSLayer *layer, PSLayer *src) {
    PSNormalizationLayerCache *dstcache = GetNormalizationCache(layer);
    PSNormalizationLayerCache *srccache = GetNormalizationCache(src);
    if (srccache != NULL) {
        if (dstcache != NULL) deleteNormalizationLayerCache(layer);
        uint32_t seqlen = 1;
        if (PSUseSequences(layer)) {
            seqlen = PSStateSequenceLength(src);
            if (seqlen == 0) return 0;
            if (!PSInitNormalizationCache(layer, seqlen, 0)) return 0;
            dstcache = GetNormalizationCache(layer);
            memcpy(dstcache, srccache, seqlen * sizeof(*dstcache));
            for (uint32_t i = 0; i < seqlen; i++) {
                PSNormalizationLayerCache *cache_t = dstcache + i;
                PSNormalizationLayerCache *srccache_t = srccache + i;
                free(cache_t->normalized_values);
                cache_t->normalized_values = NULL;
                if (srccache_t->normalized_values != NULL) {
                    cache_t->normalized_values =
                        malloc(layer->size * sizeof(PSFloat));
                    if (!cache_t->normalized_values) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                    memcpy(
                        cache_t->normalized_values,
                        srccache_t->normalized_values,
                        layer->size * sizeof(PSFloat)
                    );
                }
            }
        }
    } else deleteNormalizationLayerCache(layer);
    PSNormalizationLayerSettings *dstsettings =
        PSGetNormalizationSettings(layer);
    PSNormalizationLayerSettings *srcsettings = PSGetNormalizationSettings(src);
    if (srcsettings != NULL) {
        if (dstsettings == NULL) dstsettings = malloc(sizeof(*dstsettings));
        if (dstsettings == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(dstsettings, srcsettings, sizeof(*dstsettings));
    } else {
        free(dstsettings);
        layer->extra = NULL;
    }
    return 1;
}

PSNormalizationLayerCache *createNormalizationCache(PSLayer *layer,
                                                   uint32_t seqlen)
{
    PSNormalizationLayerCache *cache = calloc(seqlen, sizeof(*cache));
    if (cache == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    for (uint32_t i = 0; i < seqlen; i++) {
        PSNormalizationLayerCache *cache_t = cache + i;
        cache_t->normalized_values = calloc(layer->size, sizeof(PSFloat));
        if (cache_t == NULL) {
            PSPrintMemoryErrorMsg();
            deleteNormalizationCache(cache, i);
            return NULL;
        }
    }
    return cache;
}

int PSInitNormalizationCache(PSLayer *layer, uint32_t seqlen,
                             int retain_previous)
{
    UNUSED(retain_previous);
    if (layer == NULL) return 0;
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (seqlen > 0) {
        if (cache != NULL) deleteNormalizationLayerCache(layer);
        cache = createNormalizationCache(layer, seqlen);
        if (cache == NULL) return 0;
        layer->private = cache;
    } else if (layer->private != NULL) {
        deleteNormalizationLayerCache(layer);
        layer->private = NULL;
    }
    return 1;
}

int PSResizeNormalizationCache(PSLayer *layer, uint32_t seqlen) {
    if (layer == NULL) return 0;
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (cache == NULL) return PSInitNormalizationCache(layer, seqlen, 0);
    size_t size = (size_t) seqlen * sizeof(PSNormalizationLayerCache);
    PSNormalizationLayerCache *new_cache = realloc(cache, size);
    if (new_cache == NULL) goto memerr;
    int cur_seqlen = PSStateSequenceLength(layer);
    int diff = seqlen - cur_seqlen;
    if (diff > 0) {
        PSNormalizationLayerCache *added_caches = new_cache + cur_seqlen;
        memset(added_caches, 0, (size_t) diff * sizeof(*added_caches));
        for (int i = 0; i < diff; i++) {
            PSNormalizationLayerCache *added = added_caches + i;
            added->normalized_values = calloc(layer->size, sizeof(PSFloat));
            if (added->normalized_values == NULL) goto memerr;
        }
    }
    layer->private = new_cache;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    if (new_cache != NULL) deleteNormalizationCache(new_cache, seqlen);
    deleteNormalizationLayerCache(layer);
    layer->private = NULL;
    layer->network->status = STATUS_ERROR;
    return 0;
}

/* Init */
int PSInitNormalizationLayer(PSLayer *layer, PSLayerDef *ldef) {
    if (layer->index == 0) {
        PSErr(NULL, "Normalization layer cannot be the first layer");
        return 0;
    }
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "Normalization layer has no previous layer");
        return 0;
    }
    layer->on_delete = deleteNormalizationLayer;
    layer->on_copy = copyNormalizationLayer;
    layer->size = previous->size;
    if (!(layer->flags & FLAG_NON_TRAINABLE)) {
        layer->weights = malloc(sizeof(PSMatrix));
        if (layer->weights == NULL) goto memerr;
        if (ldef->weight_init_mode != INIT_MODE_AUTO) {
            layer->weights[0] =
                PSInitWeights(layer, 1, layer->size, ldef, 1, 1);
        } else layer->weights[0] = PSMatrixCreate(1, NULL, 2, 1, layer->size);
        if (layer->weights[0] == NULL) goto memerr;
        layer->weight_types_count = 1;
        if (!(layer->flags & FLAG_NO_BIAS)) {
            layer->biases = calloc(layer->size, sizeof(PSFloat));
            if (layer->biases == NULL) goto memerr;
            if (ldef->weight_init_mode != INIT_MODE_AUTO) {
                for (int i = 0; i < layer->size; i++) {
                    layer->biases[i] = PSInitParam(PARAM_TYPE_BIAS, ldef, 1,1);
                }
            }
        }
    }
    PSMatrix states = PSMatrixZeros(2, 1, layer->size);
    if (states == NULL) goto memerr;
    layer->states = states;
    if (!PSInitNormalizationCache(layer, 1, 0)) return 0;
    PSNormalizationLayerSettings *settings = calloc(1, sizeof(*settings));
    if (settings == NULL) goto memerr;
    settings->epsilon = ldef->epsilon;
    if (settings->epsilon == 0) settings->epsilon = PSDEFAULT_NORM_EPSILON;
    layer->extra = settings;
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->feedforward = PSNormalizationFeedforward;
    layer->backprop = PSNormalizationBackprop;
    layer->on_recurrent_states_init = PSInitNormalizationCache;
    layer->on_recurrent_states_resize = PSResizeNormalizationCache;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Feddforward */
int PSNormalizationFeedforward(PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    int t = 0, seqlen = 0, is_recurrent = PSIsRecurrent(layer),
        sequence_at_once = PSHandleSequenceAtOnce(layer);
    PSFloat *meandiff = NULL, *tmp = NULL;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (is_recurrent || sequence_at_once) {
        va_list args;
        va_start(args, layer);
        seqlen = va_arg(args, int);
        if (is_recurrent) t = va_arg(args, int);
        va_end(args);
        if (!PSBeforeSequenceFeedforward(layer, seqlen, t)) return 0;
    }
    PSFloat *inputs = PSGetStates(previous, t),
            *outputs = PSGetStates(layer, t);
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (inputs == NULL) {
        PSErr(__func__, "Layer[%d]: no inputs", layer->index);
        return 0;
    }
    if (outputs == NULL) {
        PSErr(__func__, "Layer[%d]: no outputs", layer->index);
        return 0;
    }
    if (cache == NULL) {
        uint32_t slen = seqlen;
        if (slen == 0) slen = 1;
        if (PSInitNormalizationCache(layer, slen, 0)) return 0;
    }
    PSNormalizationLayerSettings *settings = PSGetNormalizationSettings(layer);
    if (is_recurrent) cache += t;
    size_t alloc_size = layer->size * sizeof(PSFloat);
    if (cache->normalized_values == NULL) {
        cache->normalized_values = malloc(alloc_size);
        if (cache->normalized_values == NULL) goto memerr;
    }
    /* Normalization */
    PSFloat *normalized = cache->normalized_values;
    meandiff = malloc(alloc_size);
    tmp = malloc(alloc_size);
    if (meandiff == NULL || tmp == NULL) goto memerr;
    PSMathOpts mopts = {.acceleration = layer->network->acceleration};
    /* Compute, variance and stddev */
    PSFloat mean = PSMean(inputs, layer->size, &mopts);
    PSSubtractVectorScalar(inputs, mean, meandiff, layer->size, &mopts);
    PSMultiplyVectors(meandiff, meandiff, tmp, layer->size, &mopts);
    PSFloat variance = PSMean(tmp, layer->size, &mopts);
    PSFloat epsilon = PSDEFAULT_NORM_EPSILON;
    if (settings != NULL) epsilon = settings->epsilon;
    if (epsilon == 0) epsilon = PSDEFAULT_NORM_EPSILON;
    PSFloat stddev = PSSqrt(variance + epsilon);
    /* Normalize values: (x - mean) / stddtev */
    PSDivideVectorScalar(meandiff, stddev, normalized, layer->size, &mopts);
    cache->mean = mean;
    cache->variance = variance;
    cache->stddev = stddev;
    if (!(layer->flags & FLAG_NON_TRAINABLE)) {
        /* Apply weights and biases */
        PSMultiplyVectors(normalized, layer->weights[0], outputs, layer->size,
                          &mopts);
        PSSumVectors(outputs, layer->biases, outputs, layer->size, &mopts);
    } else memcpy(outputs, normalized, alloc_size);
    free(meandiff);
    free(tmp);
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
fail:
    free(meandiff);
    free(tmp);
    return 0;
}

int PSNormalizationBackprop(PSLayer *layer, PSLayer *previous,
                            PSGradient *gradient, ...)
{
    if (layer == NULL) return 0;
    if (previous == NULL) return 0;
    if (layer->delta == NULL) return 0;
    PSFloat *delta_norm = NULL, *tmp1 = NULL, *tmp2 = NULL;
    PSFloat *delta = layer->delta;
    int is_recurrent = PSIsRecurrent(layer), t = 0, success = 1;
    int use_bias = !(layer->flags & FLAG_NO_BIAS),
        trainable = !(layer->flags & FLAG_NON_TRAINABLE);
    if (is_recurrent) {
        va_list args;
        va_start(args, gradient);
        t = va_arg(args, int);
        va_end(args);
    }
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (cache == NULL) {
        PSErr(NULL, "Layer[%d]: missing normalization cache", layer->index);
        return 0;
    }
    if (is_recurrent) cache += t;
    PSFloat *normalized = cache->normalized_values;
    if (normalized == NULL) {
        PSErr(NULL, "Layer[%d]: missing normalized values in "
                    "normalization cache", layer->index);
        return 0;
    }
    PSNormalizationLayerSettings *settings = PSGetNormalizationSettings(layer);
    PSFloat epsilon = PSDEFAULT_NORM_EPSILON;
    if (settings != NULL) epsilon = settings->epsilon;
    if (epsilon == 0) epsilon = PSDEFAULT_NORM_EPSILON;
    PSMathOpts mopts = {.acceleration = layer->network->acceleration};
    size_t alloc_size = layer->size * sizeof(PSFloat);
    if (trainable) {
        delta_norm = malloc(alloc_size);
        if (delta_norm == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        if (gradient == NULL) return 0;
        if (use_bias) {
            PSSumVectors(gradient->biases, delta, gradient->biases,
                         layer->size, &mopts);
        }
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectors(delta, normalized, gradient->weights,
                          layer->size, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectors(layer->weights[0], delta, delta_norm,
                          layer->size, &mopts);
    } else delta_norm = delta;
    if (previous->delta == NULL) goto final;
    tmp1 = malloc(alloc_size);
    tmp2 = malloc(alloc_size);
    success = (tmp1 != NULL && tmp2 != NULL);
    if (!success) {
        PSPrintMemoryErrorMsg();
        goto final;
    }
    PSFloat stddev = cache->stddev;
    if (stddev == 0.0) {
        PSFloat epsilon = 1e-5;
        /* TODO: epsilon from settings */
        stddev = PSSqrt(cache->variance + epsilon);
    }
    success = (stddev != 0);
    if (!success) {
        PSErr(NULL, "Layer[%d]: stddev is zero", layer->index);
        goto final;
    }
    mopts.store_mode = MATHS_STORE_MODE_NORM;
    PSMultiplyVectors(delta_norm, normalized, tmp2, layer->size, &mopts);
    PSMultiplyVectorScalar(delta_norm, layer->size, tmp1, layer->size, &mopts);
    PSFloat dnorm_sum = PSSumVectorElements(delta_norm, layer->size, &mopts);
    PSFloat dnorm_norm_sum = PSSumVectorElements(tmp2, layer->size, &mopts);
    PSMultiplyVectorScalar(normalized, dnorm_norm_sum, tmp2, layer->size,
                           &mopts);
    PSSubtractVectorScalar(tmp1, dnorm_sum, tmp1, layer->size, &mopts);
    PSSubtractVectors(tmp1, tmp2, tmp1, layer->size, &mopts);
    mopts.store_mode = MATHS_STORE_MODE_ADD;
    PSDivideVectorScalar(tmp1, (layer->size * stddev), previous->delta,
                         layer->size, &mopts);
final:
    if (delta_norm != delta && delta_norm != NULL) free(delta_norm);
    free(tmp1);
    free(tmp2);
    return success;
}

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
    PSFloat *mean;
    PSFloat *variance;
    PSFloat *stddev;
    uint32_t seqlen;
} PSNormalizationLayerCache;

/* Forward declarations */
int PSNormalizationForward(PSLayer *layer, ...);
int PSNormalizationBackprop(PSLayer *layer, PSLayer *previous,
                            PSGradient *gradient, ...);
int PSInitNormalizationCache(PSLayer *layer, uint32_t steps,
                             int retain_previous);
int checkLayerForForward(PSLayer *layer);
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);

/* Layer functions */

static void deleteNormalizationCache(PSNormalizationLayerCache *cache,
                                     uint32_t seqlen)
{
    if (cache == NULL) return;
    if (seqlen <= 0) seqlen = 1;
    int checked_seqlen = 0;
    for (uint32_t i = 0; i < seqlen; i++) {
        PSNormalizationLayerCache *cache_t = cache + i;
        if (!checked_seqlen && cache_t->seqlen) {
            seqlen = cache_t->seqlen;
            checked_seqlen = 1;
        }
        free(cache_t->normalized_values);
        cache_t->normalized_values = NULL;
        free(cache_t->mean);
        cache_t->mean = NULL;
        free(cache_t->variance);
        cache_t->variance = NULL;
        free(cache_t->stddev);
        cache_t->stddev = NULL;
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
    int outdepth = layer->output_depth;
    if (outdepth < 1) outdepth = 1;
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
                free(cache_t->mean);
                cache_t->mean = NULL;
                free(cache_t->variance);
                cache_t->variance = NULL;
                free(cache_t->stddev);
                cache_t->stddev = NULL;
                if (srccache_t->normalized_values != NULL) {
                    cache_t->normalized_values =
                        malloc(layer->size * sizeof(PSFloat));
                    if (cache_t->normalized_values == NULL) goto memerr;
                    PSVectorCopy(
                        cache_t->normalized_values,
                        srccache_t->normalized_values,
                        layer->size
                    );
                }
                if (srccache_t->mean != NULL) {
                    cache_t->mean = malloc(outdepth * sizeof(PSFloat));
                    if (cache_t->mean == NULL) goto memerr;
                    PSVectorCopy(cache_t->mean, srccache_t->mean, outdepth);
                }
                if (srccache_t->variance != NULL) {
                    cache_t->variance = malloc(outdepth * sizeof(PSFloat));
                    if (cache_t->variance == NULL) goto memerr;
                    PSVectorCopy(
                        cache_t->variance, srccache_t->variance, outdepth
                    );
                }
                if (srccache_t->stddev != NULL) {
                    cache_t->stddev = malloc(outdepth * sizeof(PSFloat));
                    if (cache_t->stddev == NULL) goto memerr;
                    PSVectorCopy(cache_t->stddev, srccache_t->stddev, outdepth);
                }
            }
        }
    } else deleteNormalizationLayerCache(layer);
    PSNormalizationLayerSettings *dstsettings =
        PSGetNormalizationSettings(layer);
    PSNormalizationLayerSettings *srcsettings = PSGetNormalizationSettings(src);
    if (srcsettings != NULL) {
        if (dstsettings == NULL) {
            dstsettings = malloc(sizeof(*dstsettings));
            layer->extra = dstsettings;
        }
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
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

static int checkNormalizationCache(PSNormalizationLayerCache *cache,
                                   int size, int n_features)
{
    if (cache->normalized_values == NULL) {
        cache->normalized_values = malloc(size * sizeof(PSFloat));
        if (cache->normalized_values == NULL) goto memerr;
    }
    if (cache->mean == NULL) {
        cache->mean = calloc(n_features, sizeof(PSFloat));
        if (cache->mean == NULL) goto memerr;
    }
    if (cache->variance == NULL) {
        cache->variance = calloc(n_features, sizeof(PSFloat));
        if (cache->variance == NULL) goto memerr;
    }
    if (cache->stddev == NULL) {
        cache->stddev = calloc(n_features, sizeof(PSFloat));
        if (cache->stddev == NULL) goto memerr;
    }
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

PSNormalizationLayerCache *createNormalizationCache(PSLayer *layer,
                                                    uint32_t seqlen)
{
    PSNormalizationLayerCache *cache = calloc(seqlen, sizeof(*cache));
    if (cache == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int outdepth = layer->output_depth;
    uint32_t i;
    if (outdepth < 1) outdepth = 1;
    for (i = 0; i < seqlen; i++) {
        PSNormalizationLayerCache *cache_t = cache + i;
        cache_t->seqlen = seqlen;
        cache_t->normalized_values = calloc(layer->size, sizeof(PSFloat));
        if (cache_t->normalized_values == NULL) goto memerr;
        cache_t->mean = calloc(outdepth, sizeof(PSFloat));
        if (cache_t->mean == NULL) goto memerr;
        cache_t->variance = calloc(outdepth, sizeof(PSFloat));
        if (cache_t->variance == NULL) goto memerr;
        cache_t->stddev = calloc(outdepth, sizeof(PSFloat));
        if (cache_t->stddev == NULL) goto memerr;
    }
    return cache;
memerr:
    PSPrintMemoryErrorMsg();
    deleteNormalizationCache(cache, i);
    return NULL;
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
        layer->private = cache;
        if (cache == NULL) return 0;
    } else if (layer->private != NULL) {
        deleteNormalizationLayerCache(layer);
        layer->private = NULL;
    }
    return 1;
}

int PSResizeNormalizationCache(PSLayer *layer, uint32_t seqlen, uint32_t prev) {
    if (layer == NULL) return 0;
    PSNormalizationLayerCache *cache = GetNormalizationCache(layer);
    if (cache == NULL) return PSInitNormalizationCache(layer, seqlen, 0);
    PSNormalizationLayerCache *new_cache = calloc(seqlen, sizeof(*new_cache));
    if (new_cache == NULL) goto memerr;
    size_t size = (size_t) seqlen * sizeof(PSNormalizationLayerCache);
    memcpy(new_cache, cache, size);
    int outdepth = layer->output_depth;
    if (outdepth < 1) outdepth = 1;
    int cur_seqlen = prev;
    int diff = seqlen - cur_seqlen;
    if (diff > 0) {
        PSNormalizationLayerCache *added_caches = new_cache + cur_seqlen;
        memset(added_caches, 0, (size_t) diff * sizeof(*added_caches));
        for (int i = 0; i < diff; i++) {
            PSNormalizationLayerCache *added = added_caches + i;
            added->normalized_values = calloc(layer->size, sizeof(PSFloat));
            if (added->normalized_values == NULL) goto memerr;
            added->mean = calloc(outdepth, sizeof(PSFloat));
            if (added->mean == NULL) goto memerr;
            added->variance = calloc(outdepth, sizeof(PSFloat));
            if (added->variance == NULL) goto memerr;
            added->stddev = calloc(outdepth, sizeof(PSFloat));
            if (added->stddev == NULL) goto memerr;
        }
    }
    free(cache);
    layer->private = new_cache;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    if (new_cache != NULL) deleteNormalizationCache(new_cache, seqlen);
    deleteNormalizationLayerCache(layer);
    layer->private = NULL;
    PSModelSetStatus(layer->model, PS_STATUS_ERROR, NULL);
    return 0;
}

int PSNormalize(PSFloat *inputs, PSFloat *outputs, PSFloat *normalized,
                int size, PSFloat *scale, PSFloat *intercept,
                PSFloat *mean_p, PSFloat *variance_p, PSFloat *stddev_p,
                PSFloat epsilon, PSMathOpts *opts)
{
    PSMathOpts dfopts = {.acceleration = PSGlobalAcceleration};
    if (opts == NULL) opts = &dfopts;
    size_t alloc_size = size * sizeof(PSFloat);
    PSFloat *meandiff = malloc(alloc_size);
    PSFloat *tmp = malloc(alloc_size);
    if (meandiff == NULL || tmp == NULL) goto memerr;
    /* Compute, variance and stddev */
    PSFloat mean = PSMean(inputs, size, opts);
    PSSubtractVectorScalar(inputs, mean, meandiff, size, opts);
    PSMultiplyVectors(meandiff, meandiff, tmp, size, opts);
    PSFloat variance = PSMean(tmp, size, opts);
    PSFloat stddev = PSSqrt(variance + epsilon);
    /* Normalize values: (x - mean) / stddtev */
    PSDivideVectorScalar(meandiff, stddev, normalized, size, opts);
    if (mean_p != NULL) *mean_p = mean;
    if (variance_p != NULL) *variance_p = variance;
    if (stddev_p != NULL) *stddev_p = stddev;
    if (scale != NULL || intercept != NULL) {
        if (scale != NULL)
            PSMultiplyVectors(normalized, scale, outputs, size, opts);
        if (intercept != NULL)
            PSAddVectors(outputs, intercept, outputs, size, opts);
    } else {
        if (outputs != normalized) memcpy(outputs, normalized, alloc_size);
    }
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
    layer->onDelete = deleteNormalizationLayer;
    layer->onCopy = copyNormalizationLayer;
    layer->size = previous->size;
    if (previous->output_columns > 0)
        layer->output_columns = previous->output_columns;
    if (previous->output_rows > 0)
        layer->output_rows = previous->output_rows;
    if (previous->output_depth > 0 && ldef->output_depth != 1) {
        /* By default, previous layer's output_depth is duplicated onto
         * normalization layer if it's greater than zero.
         * This behavior allows normalization to be applied on single
         * layer's feature maps (ie. on Convolutional or Pooling layers).
         * However, output_depth can be forced to 1 when it's required
         * to apply it on flatten inputs, by setting `output_depth` = 1 on
         * `ldef`*/
        layer->output_depth = previous->output_depth;
    } else layer->output_depth = 1;
    if (!(layer->flags & PS_FLAG_NON_TRAINABLE)) {
        layer->weights = malloc(sizeof(PSMatrix));
        if (layer->weights == NULL) goto memerr;
        if (ldef->weight_init_mode != PS_INIT_MODE_AUTO) {
            layer->weights[0] =
                PSInitWeights(layer, 1, layer->size, ldef, 1, 1);
        } else layer->weights[0] = PSMatrixCreate(1, NULL, 2, 1, layer->size);
        if (layer->weights[0] == NULL) goto memerr;
        layer->weight_types = 1;
        if (!(layer->flags & PS_FLAG_NO_BIAS)) {
            layer->biases = calloc(layer->size, sizeof(PSFloat));
            if (layer->biases == NULL) goto memerr;
            if (ldef->weight_init_mode != PS_INIT_MODE_AUTO) {
                for (int i = 0; i < layer->size; i++) {
                    layer->biases[i] = PSInitParam(PS_PARAM_BIAS, ldef, 1,1);
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
    layer->forward = PSNormalizationForward;
    layer->backprop = PSNormalizationBackprop;
    layer->onStatesInit = PSInitNormalizationCache;
    layer->onStatesResize = PSResizeNormalizationCache;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Feedforward */
int PSNormalizationForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    int t = 0, seqlen = 1, is_recurrent = PSIsRecurrent(layer),
        sequence_at_once = PSHandleSequenceAtOnce(layer);
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (is_recurrent || sequence_at_once) {
        va_list args;
        va_start(args, layer);
        seqlen = va_arg(args, int);
        if (is_recurrent) t = va_arg(args, int);
        va_end(args);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    PSFloat *inputs = PSLayerStates(previous, t),
            *outputs = PSLayerStates(layer, t);
    PSNormalizationLayerCache *lcache = GetNormalizationCache(layer);
    if (inputs == NULL) {
        PSErr(__func__, "Layer[%d]: no inputs", layer->index);
        return 0;
    }
    if (outputs == NULL) {
        PSErr(__func__, "Layer[%d]: no outputs", layer->index);
        return 0;
    }
    if (seqlen < 1 || !sequence_at_once) seqlen = 1;
    if (lcache == NULL) {
        if (PSInitNormalizationCache(layer, seqlen, 0)) return 0;
        lcache = GetNormalizationCache(layer);
        if (lcache == NULL) return 0;
    }
    PSNormalizationLayerSettings *settings = PSGetNormalizationSettings(layer);
    PSFloat epsilon = PSDEFAULT_NORM_EPSILON;
    if (settings != NULL) epsilon = settings->epsilon;
    if (epsilon == 0) epsilon = PSDEFAULT_NORM_EPSILON;
    PSMathOpts mopts = {.acceleration = layer->model->acceleration};
    PSFloat *input_p = inputs, *output_p = outputs;
    int size = layer->size, n_features = layer->output_depth,
        trainable = !(layer->flags & PS_FLAG_NON_TRAINABLE), i, j;
    if (n_features > 1) size /= n_features;
    else n_features = 1;
    for (i = 0; i < seqlen; i++) {
        int tidx = i + t;
        PSNormalizationLayerCache *cache = lcache + tidx;
        if (!checkNormalizationCache(cache, layer->size, n_features))
            return 0;
        /* Normalization */
        for (j = 0; j < n_features; j++) {
            int offset = j * size;
            PSFloat *normalized = cache->normalized_values + offset;
            PSFloat *scale = NULL, *intercept = NULL;
            PSFloat *mean_p = cache->mean + j,
                    *var_p = cache->variance + j,
                    *stddev_p = cache->stddev + j;
            if (trainable) {
                scale = layer->weights[0] + offset;
                intercept = layer->biases + offset;
            }
            int ok = PSNormalize(input_p, output_p, normalized, size, scale,
                                 intercept, mean_p, var_p, stddev_p,
                                 epsilon, &mopts);
            if (!ok) return 0;
            input_p += size;
            output_p += size;
        }
    }
    return 1;
}

int PSNormalizationBackprop(PSLayer *layer, PSLayer *previous,
                            PSGradient *gradient, ...)
{
    if (layer == NULL) return 0;
    if (previous == NULL) return 0;
    if (layer->delta == NULL) return 0;
    PSFloat *delta_norm = NULL, *tmp1 = NULL, *tmp2 = NULL;
    int is_recurrent = PSIsRecurrent(layer), t = 0, seqlen = 1, success = 1,
        sequence_at_once = PSHandleSequenceAtOnce(layer);
    int use_bias = !(layer->flags & PS_FLAG_NO_BIAS),
        trainable = !(layer->flags & PS_FLAG_NON_TRAINABLE);
    if (trainable && gradient == NULL) return 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, gradient);
        t = va_arg(args, int);
        va_end(args);
    } else if (sequence_at_once) seqlen = PSStateSequenceLength(layer);
    PSNormalizationLayerCache *lcache = GetNormalizationCache(layer);
    if (lcache == NULL) {
        PSErr(NULL, "Layer[%d]: missing normalization cache", layer->index);
        return 0;
    }
    if (seqlen < 1) seqlen = 1;
    PSNormalizationLayerSettings *settings =
        PSGetNormalizationSettings(layer);
    PSFloat epsilon = PSDEFAULT_NORM_EPSILON;
    if (settings != NULL) epsilon = settings->epsilon;
    if (epsilon == 0) epsilon = PSDEFAULT_NORM_EPSILON;
    PSMathOpts mopts = {.acceleration = layer->model->acceleration};
    size_t alloc_size = layer->size * sizeof(PSFloat);
    int n_features = layer->output_depth, size = layer->size, i, j;
    if (n_features < 1) n_features = 1;
    else if (n_features > 1) size /= n_features;
    PSFloat *delta_p = layer->delta, *prev_delta_p = previous->delta;
    PSFloat *delta_norm_p = delta_norm = delta_p;
    if (trainable) {
        delta_norm = malloc(seqlen * alloc_size);
        success = (delta_norm != NULL);
        if (!success) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
    }
    if (previous->delta != NULL) {
        tmp1 = malloc(alloc_size);
        tmp2 = malloc(alloc_size);
        success = (tmp1 != NULL && tmp2 != NULL);
        if (!success) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
    }
    PSFloat *tmp1_p = tmp1, *tmp2_p = tmp2;
    for (i = 0; i < seqlen; i++) {
        int tidx = i + t;
        PSNormalizationLayerCache *cache = lcache + tidx;
        success = (cache->normalized_values != NULL);
        if (!success) {
            PSErr(
                NULL, "Layer[%d]: missing normalized values in "
                "normalization cache at step %d", layer->index, tidx
            );
            goto final;
        }
        for (j = 0; j < n_features; j++) {
            int offset = size * j;
            PSFloat *normalized = cache->normalized_values + offset;
            if (trainable) {
                PSFloat *gbiases = gradient->biases + offset,
                        *gweights = gradient->weights + offset,
                        *lweights = layer->weights[0] + offset;
                mopts.store_mode = PS_STORE_MODE_ADD;
                if (use_bias)
                    PSAddVectors(gbiases, delta_p, gbiases, size, &mopts);
                PSMultiplyVectors(delta_p, normalized, gweights, size, &mopts);
                mopts.store_mode = PS_STORE_MODE_SET;
                /* delta_norm = weights * delta[tidx] */
                PSMultiplyVectors(lweights, delta_p, delta_norm_p,size, &mopts);
            } else delta_norm_p = delta_p; /* delta_norm = delta[tidx] */
            if (previous->delta == NULL) continue;
            PSFloat stddev = cache->stddev[j];
            if (stddev == 0.0) stddev = PSSqrt(cache->variance[j] + epsilon);
            success = (stddev != 0);
            if (!success) {
                PSErr(NULL, "Layer[%d]: stddev is zero", layer->index);
                goto final;
            }
            mopts.store_mode = PS_STORE_MODE_SET;
            PSMultiplyVectors(delta_norm_p, normalized, tmp2_p, size, &mopts);
            PSMultiplyVectorScalar(delta_norm_p, size, tmp1_p, size, &mopts);
            /* dnorm_sum = sum(delta_norm)
             * dnorm_norm_sum = sum(delta_norm * normalized) */
            PSFloat dnorm_sum = PSVectorReduceSum(delta_norm_p, size, &mopts);
            PSFloat dnorm_norm_sum = PSVectorReduceSum(tmp2_p, size, &mopts);
            PSMultiplyVectorScalar(
                normalized, dnorm_norm_sum, tmp2_p, size,&mopts
            );
            PSSubtractVectorScalar(tmp1_p, dnorm_sum, tmp1_p, size, &mopts);
            PSSubtractVectors(tmp1_p, tmp2_p, tmp1_p, size, &mopts);
            mopts.store_mode = PS_STORE_MODE_ADD;
            /* tmp1 = ((delta_norm / layer_size) - dnorm_sum) -
                       (normalized * dnorm_norm_sum)) */
            /* prev_delta += (tmp1 * (layer_size * stddev)) */
            PSDivideVectorScalar(
                tmp1_p, (size * stddev), prev_delta_p, size, &mopts
            );
            delta_p += size;
            prev_delta_p += size;
            delta_norm_p += size;
            tmp1_p += size;
            tmp2_p += size;
        }
    }
final:
    if (delta_norm != layer->delta && delta_norm != NULL) free(delta_norm);
    free(tmp1);
    free(tmp2);
    return success;
}

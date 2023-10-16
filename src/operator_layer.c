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
#include <assert.h>
#include "operator_layer.h"
#include "psyc.h"
#include "log.h"
#include "maths.h"
#include "utils.h"
#include "config.h"

#define UNUSED(V) ((void) V)
#define INVALID_OP_LABEL "Invalid Operator"
#define PSGetOperatorLayerSettings(l) ((PSOperatorLayerSettings*) l->extra)

typedef struct {
    PSOperatorType operator;
    int providers_count;
    PSLayer **providers;
    PSBitmap placeholders;
} PSOperatorLayerSettings;

/* Forward declarations */
int checkLayerForForward(PSLayer *layer);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSModel *model);
int PSIsLayerPlaceholder(PSLayer *layer);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);
int PSOperatorForward(PSLayer *layer, ...);
int PSOperatorBackprop(PSLayer *layer, PSLayer *previous,
                            PSGradient *gradient, ...);


/* Layer utils */

static PSLayer *resolveProviderPlaceholder(PSLayer *layer, PSLayer *provider) {
    if (!PSIsLayerPlaceholder(provider)) return provider;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL || settings->providers == NULL) {
        PSErrNN(NULL, NULL, layer, "missing or invalid settings");
        return NULL;
    }
    PSLayer *placeholder = provider;
    provider = PSResolveLayerPlaceholder(placeholder, layer->model);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "invalid provider placeholder");
        return NULL;
    }
    int count = settings->providers_count, i;
    for (i = 0; i < count; i++) {
        if (settings->providers[i] == placeholder)
            settings->providers[i] = provider;
    }
    PSDeleteLayer(placeholder);
    return provider;
}

static void deleteOperatorLayer(PSLayer *layer) {
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings != NULL) {
        if (settings->providers != NULL && settings->placeholders != NULL) {
            int i;
            for (i = 0; i < settings->providers_count; i++) {
                PSLayer *provider = settings->providers[i];
                if (provider == NULL) continue;
                int is_placeholder = PSBitmapGetBit(settings->placeholders, i);
                if (is_placeholder) {
                    PSDeleteLayer(provider);
                    settings->providers[i] = NULL;
                }
            }
        }
        free(settings->providers);
        PSBitmapRelease(settings->placeholders);
    }
    free(settings);
    layer->extra = NULL;
}

static int copyOperatorLayer(PSLayer *layer, PSLayer *src) {
    PSOperatorLayerSettings *srcsettings = PSGetOperatorLayerSettings(src);
    PSOperatorLayerSettings *dstsettings = PSGetOperatorLayerSettings(layer);
    if (dstsettings == NULL || srcsettings == NULL) {
        PSErrNN(__func__, NULL, layer, "missing settings");
        return 0;
    }
    dstsettings->operator = srcsettings->operator;
    if (dstsettings->placeholders != NULL)
        PSBitmapRelease(dstsettings->placeholders);
    dstsettings->placeholders = NULL;
    if (srcsettings->placeholders != NULL) {
        dstsettings->placeholders = PSBitmapDup(srcsettings->placeholders);
        if (dstsettings->placeholders == NULL) return 0;
    } else {
        PSErrNN(
            NULL, NULL, layer, "missing info for placeholders in source layer"
        );
        return 0;
    }
    dstsettings->providers_count = srcsettings->providers_count;
    if (dstsettings->providers != NULL) free(dstsettings->providers);
    dstsettings->providers = NULL;
    if (srcsettings->providers != NULL) {
        dstsettings->providers = calloc(
            srcsettings->providers_count, sizeof(PSLayer*)
        );
        if (dstsettings->providers == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        int i;
        for (i = 0; i < srcsettings->providers_count; i++) {
            PSLayer *srcprovider = srcsettings->providers[i];
            if (srcprovider == NULL) continue;
            PSModel *provider_model = srcprovider->model;
            if (provider_model == NULL) {
                PSErrNN(NULL, NULL, src, "provider[%d] has no model");
                return 0;
            }
            int is_placeholder = PSBitmapGetBit(srcsettings->placeholders, i);
            if (is_placeholder) {
                if (!PSIsLayerPlaceholder(srcprovider)) {
                    PSErrNN(NULL, NULL, layer, "provider[%d] should be "
                            "a player placeholder but it's not", i);
                    return 0;
                }
                int *indices = (int *) srcprovider->extra;
                if (indices == NULL) {
                    PSErrNN(NULL, NULL, layer, "provider[%d] is an invalid "
                            "layer placeholder", i);
                    return 0;
                }
                dstsettings->providers[i] = PSMakeLayerPlaceholder(
                    indices[0], indices[1]
                );
                if (dstsettings->providers[i] == NULL) return 0;
            } else {
                PSLayer *dstprovider = NULL;
                if (layer->model == NULL) {
                    PSErrNN(NULL, NULL, layer, "layer has no model");
                    return 0;
                }
                dstprovider = PSGetLayerByIndex(
                    layer->model, srcprovider->index, provider_model->index
                );
                if (dstprovider == NULL) {
                    dstprovider = PSMakeLayerPlaceholder(
                        srcprovider->index, provider_model->index
                    );
                }
                if (dstprovider == NULL) return 0;
                dstsettings->providers[i] = dstprovider;
            }
        }
    }
    return 1;
}

static int buildOperatorLayer(PSLayer *layer) {
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) {
        PSErrNN(NULL, NULL, layer, "missing operator layer settings");
        return 0;
    }
    int count = settings->providers_count, i;
    if (count < 1 || settings->providers == NULL) {
        PSErrNN(NULL, NULL, layer, "missing operator layer providers");
        return 0;
    }
    for (i = 0; i < count; i++) {
        PSLayer *provider = settings->providers[i];
        if (PSIsLayerPlaceholder(provider)) {
            provider = resolveProviderPlaceholder(layer, provider);
            if (provider == NULL) {
                PSErrNN(
                    NULL, NULL, layer, "could not resolve provider placeholder"
                );
                return 0;
            }
            settings->providers[i] = provider;
        }
    }
    return 1;
}

static PSFloat *getInputsFromProvider(PSLayer *layer, PSLayer *provider, int t)
{
    if (layer == NULL || provider == NULL) return NULL;
    if (PSIsLayerPlaceholder(provider)) {
        provider = resolveProviderPlaceholder(layer, provider);
        if (provider == NULL) {
            PSErrNN(NULL,NULL,layer,"could not resolve provider placeholder");
            return NULL;
        }
    }
    if (!PSUseSequences(provider)) return PSGetStates(provider, 0);
    if (layer->model->index > provider->model->index)
        return PSGetOutputs(provider);
    if (provider->index > layer->index && t >= 0) t--;
    if (!PSUseSequences(layer)) return PSGetOutputs(provider);
    return PSGetStates(provider, t);
}

int PSConcatenateForward(PSLayer *layer, int seqlen, int t) {
    int success = 1;
    if (layer == NULL) return 0;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return 0;
    int providers_count = settings->providers_count, i;
    int whole_seq = PSHandleSequenceAtOnce(layer);
    if (whole_seq) t = 0;
    else {
        if (!PSIsRecurrent(layer)) t = 0;
        /* Always set seqlen to 1 since, even in recurrent forward,
         * inputs are only stored in one sequence's segment. */
        seqlen = 1;
    }
    PSFloat *outputs = PSGetStates(layer, t);
    PSFloat *out_p = outputs;
    for (; t < seqlen; t++) {
        for (i = 0; i < providers_count; i++) {
            PSLayer *provider = settings->providers[i];
            if (provider == NULL) return 0;
            PSFloat *inputs = getInputsFromProvider(layer, provider, t);
            if (inputs != NULL) PSVectorCopy(out_p, inputs, provider->size);
            else PSVectorClear(out_p, provider->size);
            out_p += provider->size;
        }
    }
    return success;
}

int PSOperationForward(PSLayer *layer, int seqlen, int t) {
    int success = 1;
    if (layer == NULL) return 0;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return 0;
    int providers_count = settings->providers_count, i;
    int whole_seq = PSHandleSequenceAtOnce(layer), len = layer->size;
    if (whole_seq) {
        t = 0;
        len *= seqlen;
    } else seqlen = 1;
    PSOperatorType op = settings->operator;
    PSFloat *outputs = PSGetStates(layer, t);
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    int is_add = op == PSAddOperator;
    for (i = 0; i < providers_count; i++) {
        PSLayer *provider = settings->providers[i];
        if (provider == NULL) return 0;
        PSFloat *inputs = getInputsFromProvider(layer, provider, t);
        if (inputs == NULL) continue;
        if (i == 0) {
            PSVectorCopy(outputs, inputs, len);
            continue;
        }
        if (is_add) PSAddVectors(outputs, inputs, outputs, len, &opts);
        else PSMultiplyVectors(outputs, inputs, outputs, len, &opts);
    }
    return success;
}

int PSConcatenateBackward(PSLayer *layer, int seqlen, int t) {
    int success = 1;
    if (layer == NULL) return 0;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return 0;
    PSFloat *delta = layer->delta;
    if (delta == NULL) return 0;
    int providers_count = settings->providers_count, i;
    int whole_seq = PSHandleSequenceAtOnce(layer);
    if (whole_seq) t = 0;
    else {
        if (!PSIsRecurrent(layer)) t = 0;
        /* Always set seqlen to 1 since, even in recurrent forward,
         * inputs are only stored in one sequence's segment. */
        seqlen = 1;
    }
    PSMathOpts opts = {
        .acceleration = layer->model->acceleration,
        .store_mode = PS_STORE_MODE_ADD
    };
    for (; t < seqlen; t++) {
        for (i = 0; i < providers_count; i++) {
            PSLayer *provider = settings->providers[i];
            if (provider == NULL) return 0;
            PSFloat *output_delta = provider->delta;
            if (output_delta == NULL) goto next;
            output_delta += (t * provider->size);
            PSAddVectors(
                output_delta, delta, output_delta, provider->size, &opts
            );
next:
            delta += provider->size;
        }
    }
    return success;
}

int PSOperationBackward(PSLayer *layer, int seqlen, int t) {
    UNUSED(seqlen);
    int success = 1;
    if (layer == NULL) return 0;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return 0;
    if (layer->delta == NULL) return 0;
    int providers_count = settings->providers_count;
    int whole_seq = PSHandleSequenceAtOnce(layer);
    uint64_t len;
    if (whole_seq) {
        t = 0;
        len = PSMatrixLength((PSMatrix) layer->delta);
    } else {
        len = layer->size;
        if (!PSIsRecurrent(layer)) t = 0;
    }
    PSFloat *outputs = PSGetStates(layer, t);
    PSFloat *delta = malloc((size_t) len * sizeof(PSFloat));
    if (delta == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSOperatorType op = settings->operator;
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    PSMultiplyVectors(layer->delta, outputs, delta, len, &opts);
    int is_add = (PSAddOperator == op), is_mul = (PSMultiplyOperator == op);
    for (int i = 0; i < providers_count; i++) {
        PSLayer *provider = settings->providers[i];
        success = (provider != NULL);
        if (!success) goto final;
        PSMatrix output_delta = provider->delta;
        if (output_delta == NULL) continue;
        PSMatrix dptr = (
            is_add ? output_delta : PSMatrixDupShape(output_delta)
        );
        success = (dptr != NULL);
        if (!success) goto final;
        PSAddVectors(output_delta, delta, dptr, len, &opts);
        if (is_add) continue;
        else if (is_mul) {
            for (int j = 0; j < providers_count; j++) {
                if (j == i) continue;
                PSLayer *xprovider = settings->providers[j];
                success = (xprovider != NULL);
                if (!success) {
                    PSMatrixDelete(dptr);
                    goto final;
                }
                PSFloat *inputs = getInputsFromProvider(layer, xprovider, t);
                PSMultiplyVectors(dptr, inputs, dptr, len, &opts);
            }
            PSAddVectors(output_delta, dptr, output_delta, len, &opts);
        }
        if (dptr != output_delta) PSMatrixDelete(dptr);
    }
final:
    free(delta);
    return success;
}

PSOperatorType PSGetOperatorLayerType(PSLayer *layer) {
    if (layer == NULL) return PSInvalidOperator;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return PSInvalidOperator;
    PSOperatorType operator = settings->operator;
    if (operator > PSMultiplyOperator) return PSInvalidOperator;
    return operator;
}

const char *PSGetOperatorLayerTypeLabel(PSOperatorType operator) {
    switch (operator) {
        case PSConcatenateOperator: return "Concatenate";
        case PSAddOperator: return "Add";
        case PSMultiplyOperator: return "Multiply";
        case PSInvalidOperator: return INVALID_OP_LABEL;
    }
    return INVALID_OP_LABEL;
}

PSLayer **PSGetOperatorLayerProviders(PSLayer *layer, int *count) {
    if (count != NULL) *count = 0;
    if (layer == NULL) return NULL;
    PSOperatorLayerSettings *settings = PSGetOperatorLayerSettings(layer);
    if (settings == NULL) return NULL;
    if (count != NULL) *count = settings->providers_count;
    return settings->providers;
}

/* Init */
int PSInitOperatorLayer(PSLayer *layer, PSLayerDef *ldef) {
    int success = 1;
    layer->on_delete = deleteOperatorLayer;
    layer->on_copy = copyOperatorLayer;
    layer->build = buildOperatorLayer;
    layer->weights = NULL;
    layer->flags |= FLAG_NON_TRAINABLE;
    layer->biases = NULL;
    layer->weight_types_count = 0;
    if (ldef == NULL) {
        PSErrNN(NULL, NULL, layer, "missing layer definition");
        return 0;
    }
    if (ldef->providers_count <= 0) {
        PSErrNN(NULL, NULL, layer, "invalid `providers_count` in layer "
                "definition: %d", ldef->providers_count);
        return 0;
    }
    if (ldef->providers == NULL) {
        PSErrNN(NULL, NULL, layer, "missing `providers` in layer definition");
        return 0;
    }
    PSOperatorLayerSettings *settings = calloc(1, sizeof(*settings));
    if (settings == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    layer->extra = settings;
    settings->operator = ldef->operator;
    if (settings->operator > PSMultiplyOperator) {
        PSErrNN(NULL, NULL, layer, "invalid operator");
        return 0;
    }
    settings->providers_count = ldef->providers_count;
    if (settings->providers_count <= 0) {
        PSErrNN(NULL, NULL, layer, "layer definition requires "
                "providers_count that must be > 0");
        return 0;
    }
    if (settings->providers_count > PS_MAX_PROVIDERS) {
        PSErrNN(NULL, NULL, layer, "max. providers_count is %d",
                PS_MAX_PROVIDERS);
        return 0;
    }
    int include_prev_layer = 0;
    PSLayer *prev = NULL;
    if (settings->providers_count == 1) {
        prev = PSGetPreviousLayer(layer);
        if (prev == NULL) {
            PSErrNN(NULL, NULL, layer, "`providers_count` is 1 but there's "
                    "no previous layer");
            return 0;
        }
        include_prev_layer = 1;
        settings->providers_count++;
    }
    size_t provider_sz = settings->providers_count * sizeof(PSLayer*);
    settings->providers = malloc(provider_sz);
    if (settings->providers == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    settings->placeholders = PSBitmapCreate((size_t)settings->providers_count);
    if (settings->placeholders == NULL) return 0;
    PSLayer **providers_p = settings->providers;
    if (include_prev_layer) {
        *(providers_p++) = prev;
        provider_sz = (settings->providers_count - 1) * sizeof(PSLayer*);
    }
    memcpy(providers_p, ldef->providers, provider_sz);
    layer->size = 0;
    int whole_seq = PSHandleSequenceAtOnce(layer),
        is_concat = settings->operator == PSConcatenateOperator;
    for (int i = 0; i < settings->providers_count; i++) {
        PSLayer *provider = settings->providers[i];
        if (provider == NULL) {
            PSErrNN(NULL, NULL, layer, "provider[%d] is null", i);
            return 0;
        }
        if (provider == layer) {
            PSErrNN(NULL, NULL, layer, "provider[%d] is the layer itself", i);
            return 0;
        }
        if (whole_seq && !PSHandleSequenceAtOnce(provider) && !is_concat) {
            PSErrNN(NULL, NULL, layer, "provider[%d] does not take whole "
                    "but layer does. On PSConcatenateOperator operator "
                    "is available", i);
            return 0;
        }
        if (is_concat) layer->size += provider->size;
        else {
            if (layer->size == 0) layer->size = provider->size;
            else if (provider->size != layer->size) {
                /* Providers expected to be of the same size */
                PSErrNN(NULL, NULL, layer,
                        "providers must be of the same size (provider[%d] "
                        "size %d != %d)", i, provider->size, layer->size);
                return 0;
            }
        }
    }
    PSMatrix states = PSMatrixZeros(2, 1, layer->size);
    if (states == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    layer->states = states;
    layer->activate = NULL;
    layer->derivative = NULL;
    layer->forward = PSOperatorForward;
    layer->backprop = PSOperatorBackprop;
    return success;
}

int PSOperatorForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSOperatorType op = PSGetOperatorLayerType(layer);
    if (op == PSInvalidOperator) {
        PSErrNN(NULL, NULL, layer, "invalid operator");
        return 0;
    }
    int t = 0, seqlen = 1, is_recurrent = PSIsRecurrent(layer),
        sequence_at_once = PSHandleSequenceAtOnce(layer);
    if (is_recurrent || sequence_at_once) {
        va_list args;
        va_start(args, layer);
        seqlen = va_arg(args, int);
        if (is_recurrent) t = va_arg(args, int);
        va_end(args);
        if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    }
    int success = 1;
    if (PSConcatenateOperator == op)
        success = PSConcatenateForward(layer, seqlen, t);
    else success = PSOperationForward(layer, seqlen, t);
    return success;
}

int PSOperatorBackprop(PSLayer *layer, PSLayer *previous,
                       PSGradient *gradient, ...)
{
    UNUSED(previous);
    if (layer == NULL) return 0;
    if (layer->delta == NULL) return 0;
    PSOperatorType op = PSGetOperatorLayerType(layer);
    if (op == PSInvalidOperator) {
        PSErrNN(NULL, NULL, layer, "invalid operator");
        return 0;
    }
    int is_recurrent = PSIsRecurrent(layer), t = 0, seqlen = 1, success = 1,
        whole_seq = PSHandleSequenceAtOnce(layer);
    int trainable = !(layer->flags & FLAG_NON_TRAINABLE);
    if (trainable && gradient == NULL) return 0;
    if (is_recurrent) {
        va_list args;
        va_start(args, gradient);
        t = va_arg(args, int);
        va_end(args);
    } else if (whole_seq) seqlen = PSStateSequenceLength(layer);
    if (op == PSConcatenateOperator)
        success = PSConcatenateBackward(layer, seqlen, t);
    else success = PSOperationBackward(layer, seqlen, t);
    return success;
}

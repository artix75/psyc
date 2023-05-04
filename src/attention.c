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
#include "attention.h"
#include "log.h"
#include "maths.h"
#include "config.h"

#define ATTENTION_WEIGHT_TYPES_COUNT 5
#define PSGetAttentionSettings(layer) ((PSAttentionSettings*) layer->extra)
#define PSGetAttentionData(layer) ((PSAttentionData*) layer->private)

typedef struct {
    PSAttentionType type;
    PSLayer *query_provider;
    PSLayer *keys_provider;
    PSLayer *values_provider;
    PSFloat scale;
    int num_heads;
    int causal;
    int trainable_parameters;
} PSAttentionSettings;

typedef struct {
    PSMatrix query;
    PSMatrix keys;
    PSMatrix values;
    PSMatrix query_inputs;
    PSMatrix key_inputs;
    PSMatrix value_inputs;
    PSMatrix attention_weights;
    PSMatrix causal_mask;
} PSAttentionData;

/* Forward delcarations */

PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
PSFloat *initLayerStates(PSLayer *layer, uint32_t steps,
                             int retain_previous, PSFloat *current,
                             PSFloat **previous);
PSFloat *resizeLayerStates(PSLayer *layer, uint32_t steps,
                               PSFloat *current, PSFloat **previous);
int PSResizeLayerStates(PSLayer *layer, uint32_t steps);
int checkLayerForForward(PSLayer *layer);
static PSLayer *getKeysProvider(PSLayer *layer);
static int hasTrainableQuery(PSLayer *layer);
int PSAttentionForward(PSLayer *layer, ...);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
int PSIsLayerPlaceholder(PSLayer *layer);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int network_index);
PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSNeuralNetwork *net);

/* Attention functions */

static void deleteHeads(PSMatrix *heads, int num_heads, int is_vector) {
    if (heads == NULL) return;
    for (int i = 0; i < num_heads; i++) {
        if (!is_vector) PSMatrixDelete(heads[i]);
        else free(heads[i]);
    }
    free(heads);
}

static void deleteAttentionLayer(PSLayer *layer) {
    free(layer->extra);
    layer->extra = NULL;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data != NULL) {
        PSMatrixDelete(data->query);
        PSMatrixDelete(data->keys);
        PSMatrixDelete(data->values);
        PSMatrixDelete(data->query_inputs);
        PSMatrixDelete(data->key_inputs);
        PSMatrixDelete(data->value_inputs);
        PSMatrixDelete(data->causal_mask);
        PSMatrixDelete(data->attention_weights);
        free(data);
    }
}

static int copyAttentionLayer(PSLayer *layer, PSLayer *src) {
    int success = 1;
    PSAttentionSettings *srcsettings = PSGetAttentionSettings(src);
    PSAttentionData *srcdata = PSGetAttentionData(src);
    if (layer->extra != NULL) {
        free(layer->extra);
        layer->extra = NULL;
    }
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data != NULL) {
        PSMatrixDelete(data->query);
        PSMatrixDelete(data->keys);
        PSMatrixDelete(data->values);
        PSMatrixDelete(data->query_inputs);
        PSMatrixDelete(data->key_inputs);
        PSMatrixDelete(data->value_inputs);
        PSMatrixDelete(data->causal_mask);
        PSMatrixDelete(data->attention_weights);
        free(data);
        data = NULL;
        layer->private = NULL;
    }
    if (srcsettings != NULL) {
        PSAttentionSettings *settings = malloc(sizeof(*srcsettings));
        if (settings == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(settings, srcsettings, sizeof(*srcsettings));
        settings->query_provider = NULL;
        settings->keys_provider = NULL;
        settings->values_provider = NULL;
        PSLayer **srcproviders[] = {
            &srcsettings->query_provider, &srcsettings->keys_provider,
            &srcsettings->values_provider
        };
        PSLayer **providers[] = {
            &settings->query_provider, &settings->keys_provider,
            &settings->values_provider
        };
        for (size_t i = 0; i < (sizeof(providers) / sizeof(PSFloat**)); i++) {
            PSLayer *srcprovider = *(srcproviders[i]);
            PSLayer **provider_p = providers[i];
            if (srcprovider != NULL) {
                *provider_p = PSGetLayerByIndex(
                    layer->network, srcprovider->index,
                    srcprovider->network->index
                );
                if (*provider_p == NULL) {
                    int is_after = (
                        srcprovider->network->index > layer->network->index ||
                        (srcprovider->network->index == layer->network->index &&
                         srcprovider->index > layer->index)
                    );
                    if (!is_after) {
                        PSErrNN(NULL, NULL, layer, "could not find layer "
                                "query_provider");
                        return 0;
                    } else {
                        *provider_p = PSMakeLayerPlaceholder(
                            srcprovider->index, srcprovider->network->index
                        );
                    }
                }
            }
        }
    }
    if (srcdata != NULL) {
        data = layer->private = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        if (srcdata->query != NULL) {
            data->query = PSMatrixDup(srcdata->query);
            if (data->query == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->keys != NULL) {
            data->keys = PSMatrixDup(srcdata->keys);
            if (data->keys == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->values != NULL) {
            data->values = PSMatrixDup(srcdata->values);
            if (data->values == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->query_inputs != NULL) {
            data->query_inputs = PSMatrixDup(srcdata->query_inputs);
            if (data->query_inputs == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->key_inputs != NULL) {
            data->key_inputs = PSMatrixDup(srcdata->key_inputs);
            if (data->key_inputs == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->value_inputs != NULL) {
            data->value_inputs = PSMatrixDup(srcdata->value_inputs);
            if (data->value_inputs == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->attention_weights != NULL) {
            data->attention_weights = PSMatrixDup(srcdata->attention_weights);
            if (data->attention_weights == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcdata->causal_mask != NULL) {
            data->causal_mask = PSMatrixDup(srcdata->causal_mask);
            if (data->causal_mask == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
    }
    return success;
}

PSMatrix createCausalMask(int size, PSMathOpts *opts) {
    PSMathOpts dfopts = {.acceleration = PSGlobalAcceleration};
    PSMatrix mask = PSDiagonalMask(size);
    if (mask == NULL) return NULL;
    if (opts == NULL) opts = &dfopts;
    PSFloat eps = (sizeof(PSFloat) == sizeof(float) ? -1e6 : -1e10);
    for (int i = 0; i < size; i++) {
        PSFloat *row = mask + (i * size);
        PSSubtractScalarVector(1, row, row, size, opts);
        PSMultiplyVectorScalar(row, eps, row, size, opts);
    }
    return mask;
}

static PSMatrix initOrResizeCausalMask(PSLayer *layer, PSAttentionData *data,
                                       int seqlen)
{
    if (data == NULL) data = PSGetAttentionData(layer);
    if (data == NULL) return NULL;
    PSMatrix mask = data->causal_mask;
    if (mask != NULL) {
        int mask_seqlen = PSMatrixDim(mask, 0);
        if (seqlen > mask_seqlen) {
            PSMatrixDelete(mask);
            data->causal_mask = NULL;
            mask = NULL;
        }
    }
    if (mask == NULL) {
        PSMathOpts opts = {.acceleration = layer->network->acceleration};
        mask = createCausalMask(seqlen, &opts);
        if (mask == NULL) {
            PSErrNN(NULL, NULL, layer,
                    "failed to create causal mask of size %dx%d", seqlen,
                    seqlen);
            return NULL;
        }
        data->causal_mask = mask;
    }
    return mask;
}

static PSMatrix initOrResizeAttentionWeights(PSLayer *layer,
                                             PSAttentionData *data,
                                             int seqlen)
{
    if (data == NULL) data = PSGetAttentionData(layer);
    if (data == NULL) return NULL;
    PSMatrix attn_weights = data->attention_weights;
    if (attn_weights != NULL) {
        int attn_weights_seqlen = PSMatrixDim(attn_weights, 0);
        if (seqlen > attn_weights_seqlen || seqlen <= 0) {
            PSMatrixDelete(attn_weights);
            data->attention_weights = NULL;
            attn_weights = NULL;
        }
    }
    if (seqlen <= 0) return NULL;
    if (attn_weights == NULL) {
        PSLayer *kprovider = getKeysProvider(layer);
        if (kprovider == NULL) {
            PSErrNN(NULL, NULL, layer, "missing key provider");
            return NULL;
        }
        int kseqlen = PSStateSequenceLength(kprovider);
        if (kseqlen < 1) {
            PSErrNN(NULL, NULL, layer, "key provider is empty");
            return NULL;
        }
        int nheads = 0;
        PSAttentionSettings *settings = PSGetAttentionSettings(layer);
        if (settings != NULL) nheads = settings->num_heads;
        if (nheads > 1)
            attn_weights = PSMatrixZeros(3, seqlen, nheads, kseqlen);
        else attn_weights = PSMatrixZeros(2, seqlen, kseqlen);
        if (attn_weights == NULL) {
            PSErrNN(NULL, NULL, layer,
                    "failed to create causal attention_weights");
            return NULL;
        }
        data->attention_weights = attn_weights;
    }
    return attn_weights;
}

int PSInitAttentionStates(PSLayer *layer, uint32_t steps, int retain_previous) {
    if (layer == NULL) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSErrNN(__func__, NULL, layer, "missing attention data");
            return 0;
        }
        layer->private = data;
    }

    if (data->keys != NULL) PSMatrixDelete(data->keys);
    data->keys = NULL;

    if (data->key_inputs != NULL) PSMatrixDelete(data->key_inputs);
    data->key_inputs = NULL;

    if (data->values != NULL) PSMatrixDelete(data->values);
    data->values = NULL;

    if (data->value_inputs != NULL) PSMatrixDelete(data->value_inputs);
    data->value_inputs = NULL;

    if (steps == 0) {
        if (data->query != NULL) {
            PSMatrixDelete(data->query);
            data->query = NULL;
        }
        if (data->query_inputs != NULL) {
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = NULL;
        }
        if (data->attention_weights != NULL) {
            PSMatrixDelete(data->attention_weights);
            data->attention_weights = NULL;
        }
        return 1;
    }
    PSMatrix query = initLayerStates(
        layer, steps, retain_previous, data->query, &layer->initial_states
    );
    if (query == NULL) return 0;
    if (data->query != NULL) PSMatrixDelete(data->query);
    data->query = query;
    if (hasTrainableQuery(layer)) {
        PSMatrix query_inputs = initLayerStates(
            layer, steps, retain_previous, data->query_inputs,
            &layer->initial_states
        );
        if (query_inputs == NULL) return 0;
        if (data->query_inputs != NULL) PSMatrixDelete(data->query_inputs);
        data->query_inputs = query_inputs;
    }

    if (initOrResizeAttentionWeights(layer, data, steps) == NULL)
        return 0;

    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings != NULL && settings->causal)
        if (initOrResizeCausalMask(layer, data, steps) == NULL) return 0;

    return 1;
}

int PSResizeAttentionStates(PSLayer *layer, uint32_t steps) {
    if (layer == NULL) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSErrNN(__func__, NULL, layer, "missing attention data");
            return 0;
        }
        layer->private = data;
    }

    PSMatrix query = resizeLayerStates(
        layer, steps, data->query, &layer->initial_states
    );
    if (query == NULL) {
        PSMatrixDelete(data->query);
        data->query = NULL;
        layer->initial_states = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
        return 0;
    }
    data->query = query;

    if (hasTrainableQuery(layer)) {
        PSMatrix query_inputs = resizeLayerStates(
            layer, steps, data->query_inputs, &layer->initial_states
        );
        if (query_inputs == NULL) {
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = NULL;
            layer->initial_states = NULL;
            PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
            return 0;
        }
        data->query_inputs = query_inputs;
    }

    if (initOrResizeAttentionWeights(layer, data, steps) == NULL)
        return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings != NULL && settings->causal)
        if (!initOrResizeCausalMask(layer, data, steps)) return 0;

    return 1;
}

static int hasTrainableKeys(PSLayer *layer) {
    return layer->weights != NULL && layer->weights[PS_KEYS_IDX] != NULL;
}

static int hasTrainableQuery(PSLayer *layer) {
    return layer->weights != NULL && layer->weights[PS_QUERY_IDX] != NULL;
}

static int hasTrainableValues(PSLayer *layer) {
    return layer->weights != NULL && layer->weights[PS_VALUES_IDX] != NULL;
}

static int hasTrainableScores(PSLayer *layer) {
    return layer->weights != NULL && layer->weights[PS_SCORES_IDX] != NULL;
}

static int useOutputProjection(PSLayer *layer) {
    return layer->weights != NULL && layer->weights[PS_PROJECTION_IDX] != NULL;
}

static int isValidProvider(PSLayer *provider, PSLayer *keys_provider) {
    if (provider == NULL || keys_provider == NULL) return 0;
    return PSUseSequences(provider) && provider->size == keys_provider->size;
}

static PSLayer *getKeysProvider(PSLayer *layer) {
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return NULL;
    return settings->keys_provider;
}

static PSLayer *getQueryProvider(PSLayer *layer) {
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return NULL;
    if (PSIsLayerPlaceholder(settings->query_provider)) {
        settings->query_provider = PSResolveLayerPlaceholder(
            settings->query_provider, layer->network
        );
        if (settings->query_provider == NULL) {
            PSErrNN(NULL, NULL, layer, "could not resolve query provider "
                    "placeholder");
            return NULL;
        }
    }
    return settings->query_provider;
}

static PSLayer *getValuesProvider(PSLayer *layer) {
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return NULL;
    PSLayer *provider = settings->values_provider;
    if (PSIsLayerPlaceholder(provider)) {
        settings->values_provider = provider = PSResolveLayerPlaceholder(
            provider, layer->network
        );
        if (settings->values_provider == NULL) {
            PSErrNN(NULL, NULL, layer, "could not resolve values provider "
                    "placeholder");
            return NULL;
        }
    }
    if (provider == NULL) provider = settings->keys_provider;
    return provider;
}

static int buildAttentionLayer(PSLayer *layer) {
    if (layer == NULL) return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) {
        PSErrNN(__func__, NULL, layer, "attention layer has no settings");
        return 0;
    }
    PSLayer *provider = getKeysProvider(layer);
    if (provider == NULL) {
        PSErrNN(__func__, NULL, layer, "missing keys provider");
        return 0;
    } else if (PSIsLayerPlaceholder(provider)) {
        PSErrNN(__func__, NULL, layer, "keys provider is a placeholder");
        return 0;
    }
    provider = getQueryProvider(layer);
    if (provider == NULL) {
        PSErrNN(__func__, NULL, layer, "missing query provider");
        return 0;
    }
    provider = getValuesProvider(layer);
    if (provider == NULL) {
        PSErrNN(__func__, NULL, layer, "missing values provider");
        return 0;
    }
    return 1;
}

static int attentionFeedforward(PSMatrix x, PSMatrix weights, PSFloat *biases,
                                PSActivationFunction activate, PSFloat *dest,
                                int x_is_vec, int acceleration)
{
    int input_size = PSMatrixDim(weights, 1), success = 1;
    PSMathOpts opts = {.acceleration = acceleration};
    if (x_is_vec) {
        success = PSDotMV(weights, (PSFloat *)x, dest, &opts);
        if (!success) return 0;
        if (biases != NULL) PSSumVectors(dest, biases, dest, input_size, &opts);
        if (activate) activate(dest, dest, input_size, &opts);
    } else {
        opts.transpose = 2;
        success = PSDot(x, weights, dest, &opts);
        if (!success) return 0;
        opts.transpose = 0;
        if (biases != NULL || activate != NULL) {
            int nrows = PSMatrixDim(x, 0), i;
            PSFloat *row = dest;
            for (i = 0; i < nrows; i++) {
                if (biases != NULL)
                    PSSumVectors(row, biases, row, input_size, &opts);
                if (activate)
                    activate(row, row, input_size, &opts);
            }
        }
    }
    return success;
}

PSFloat *PSGetAttentionQuery(PSLayer *layer, int t) {
    int trainable = hasTrainableQuery(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
            return NULL;
        }
    }
    PSFloat *query = NULL;
    PSLayer *provider = getQueryProvider(layer);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention query provider");
        return NULL;
    }
    int from_prev_network = provider->network->index < layer->network->index;
    int is_after = provider->network->index == layer->network->index &&
                   provider->index > layer->index;
    int seqlen = 1;
    int whole_seq = PSHandleSequenceAtOnce(layer);
    PSFloat *query_inputs = NULL;
    if (whole_seq) {
        if (is_after) {
            PSErrNN(NULL, NULL, layer, "cannot get queries from future layers");
            return NULL;
        }
        seqlen = PSStateSequenceLength(provider);
        if (seqlen <= 0 || provider->states == NULL) goto empty_provider;
        query = PSMatrixZeros(2, seqlen, layer->size);
        if (query == NULL) goto memerr;
        PSVectorCopy(query, provider->states, seqlen * layer->size);
        if (data->query != NULL) PSMatrixDelete(data->query);
        data->query = query;
        if (trainable) {
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = PSMatrixDup(data->query);
        }
    } else {
        PSFloat *inputs = NULL;
        if (is_after) inputs = PSGetStates(provider, t - 1);
        else if (from_prev_network) inputs = PSGetOutputs(provider);
        else inputs = PSGetStates(provider, t);
        if (inputs == NULL) {
            if (t > 0) goto empty_provider;
            inputs = layer->initial_states;
            if (inputs == NULL) goto empty_provider;
        }
        int cur_seqlen = PSStateSequenceLength(layer);
        if (t >= cur_seqlen || data->query == NULL) {
            if (!PSResizeLayerStates(layer, t + 1)) {
                if (layer->network)
                    PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
                PSErrNN(
                    NULL, NULL, layer,
                    "could not resize recurrent hidden states"
                );
                return NULL;
            }
        }
        query = data->query + (t * layer->size);
        PSVectorCopy(query, inputs, layer->size);
        if (trainable) {
            query_inputs = data->query_inputs + (t * layer->size);
            PSVectorCopy(query_inputs, inputs, layer->size);
        }
    }
    if (trainable) {
        int acceleration = layer->network->acceleration, ok;
        PSMatrix weights = layer->weights[PS_QUERY_IDX];
        int use_bias = !(layer->flags & FLAG_NO_BIAS);
        PSFloat *biases = (use_bias ? layer->biases : NULL);
        if (whole_seq) {
            ok = attentionFeedforward(
                data->query_inputs, weights, biases, layer->activate,
                query, 0, acceleration
            );
        } else {
            ok = attentionFeedforward(
                query_inputs, weights, biases, layer->activate,
                query, 1, acceleration
            );
        }
        if (!ok) {
            PSErrNN(NULL,NULL,layer, "attention query feedforward failed");
            return NULL;
        }
    }
    return query;
empty_provider:
    PSErrNN(NULL, NULL, layer, "empty attention query provider");
    return NULL;
memerr:
    PSPrintMemoryErrorMsg();
    return NULL;
}

PSMatrix PSGetAttentionKeys(PSLayer *layer) {
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
            return NULL;
        }
    }
    int trainable = hasTrainableKeys(layer);
    PSMatrix keys = data->keys;
    if (keys != NULL) return keys;
    PSLayer *provider = getKeysProvider(layer);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention keys provider");
        return NULL;
    }
    int seqlen = PSStateSequenceLength(provider);
    if (seqlen <= 0) {
        PSErrNN(NULL, NULL, layer, "empty attention keys provider");
        return NULL;
    }
    keys = PSMatrixZeros(2, seqlen, layer->size);
    if (keys == NULL) {
        PSErrNN(NULL, NULL, layer, "could not allocate keys");
        return NULL;
    }
    PSVectorCopy(keys, provider->states, seqlen * layer->size);
    data->keys = keys;
    if (trainable) {
        PSMatrixDelete(data->key_inputs);
        data->key_inputs = PSMatrixDup(keys);
        int acceleration = layer->network->acceleration;
        PSMatrix weights = layer->weights[PS_KEYS_IDX];
        PSFloat *biases = NULL;
        if (!(layer->flags & FLAG_NO_BIAS))
            biases = layer->biases + (layer->size * PS_KEYS_IDX);
        int ok = attentionFeedforward(
            data->key_inputs, weights, biases, layer->activate, keys,
            0, acceleration
        );
        if (!ok) {
            PSErrNN(NULL, NULL, layer, "attention keys feedforward failed");
            return NULL;
        }
    }
    return keys;
}

PSMatrix PSGetAttentionValues(PSLayer *layer) {
    PSMatrix values = NULL;
    int trainable = hasTrainableValues(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
            return NULL;
        }
    }
    values = data->values;
    if (values != NULL) return values;
    PSLayer *provider = getValuesProvider(layer);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention values provider");
        return NULL;
    }
    if (!trainable && provider == getKeysProvider(layer))
        return PSGetAttentionKeys(layer);
    int seqlen = PSStateSequenceLength(provider);
    if (seqlen <= 0) {
        PSErrNN(NULL, NULL, layer, "empty attention values provider");
        return NULL;
    }
    values = PSMatrixZeros(2, seqlen, layer->size);
    if (values == NULL) {
        PSErrNN(NULL, NULL, layer, "could not allocate values");
        return NULL;
    }
    PSVectorCopy(values, provider->states, seqlen * layer->size);
    data->values = values;
    if (trainable) {
        PSMatrixDelete(data->value_inputs);
        data->value_inputs = PSMatrixDup(values);
        int acceleration = layer->network->acceleration;
        PSMatrix weights = layer->weights[PS_VALUES_IDX];
        PSFloat *biases = NULL;
        if (!(layer->flags & FLAG_NO_BIAS))
            biases = layer->biases + (layer->size * PS_VALUES_IDX);
        int ok = attentionFeedforward(
            data->value_inputs, weights, biases, layer->activate, values,
            0, acceleration
        );
        if (!ok) {
            PSErrNN(NULL, NULL, layer, "attention values feedforward failed");
            return NULL;
        }
    }
    return values;
}

PSMatrix PSGetCausalMask(PSLayer *layer) {
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) return NULL;
    return data->causal_mask;
}

static PSLayer *findKeysProvider(PSLayer *layer) {
    PSLayer *provider = NULL;
    PSNeuralNetwork *net = layer->network;
    while (provider == NULL && net != NULL) {
        PSLayer *prev = NULL;
        if (net == layer->network)
            prev = PSGetPreviousLayer(layer);
        else prev = PSGetOutputLayer(layer->network);
        while (prev != NULL) {
            if (PSUseSequences(prev)) {
                provider = prev;
                break;
            }
            prev = PSGetPreviousLayer(prev);
        }
        net = net->previous;
    }
    return provider;
}

PSMatrix PSGetAdditiveScores(PSLayer *layer, PSFloat *query, PSMatrix keys) {
    if (query == NULL || keys == NULL) return NULL;
    PSMatrix scores = NULL, sum = NULL;
    PSMatrix score_weights = NULL;
    int trainable = hasTrainableScores(layer);
    int whole_seq = PSHandleSequenceAtOnce(layer);
    int num_keys = PSMatrixDim(keys, 0), num_qry = 1;
    int success = (num_keys > 0);
    int size = layer->size;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "empty keys");
        goto final;
    }
    if (whole_seq) {
        num_qry = PSMatrixDim((PSMatrix) query, 0);
        success = (num_qry == num_keys);
        if (!success) {
            PSErrNN(NULL, NULL, layer,
                    "query count mismatches keys count: %d != %d",
                    num_qry, num_keys);
            return NULL;
        }
        sum = PSMatrixZeros(3, num_qry, num_keys, layer->size);
        scores = PSMatrixZeros(2, num_qry, num_keys);
    } else {
        size = PSMatrixDim(keys, 1);
        sum = PSMatrixZeros(2, num_keys, size);
        scores = PSMatrixZeros(1, num_keys);
    }
    success = (scores != NULL && sum != NULL);
    if (!success) goto final;
    if (trainable) score_weights = layer->weights[PS_SCORES_IDX];
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    for (int i = 0; i < num_qry; i++) {
        int qry_offset = (i * size);
        PSFloat *qrysum = sum + (i * num_keys * size);
        for (int k = 0; k < num_keys; k++) {
            int offset = (k * size);
            PSFloat *dest = qrysum + offset;
            PSFloat *key = keys + offset;
            PSFloat *qry = query + qry_offset;
            PSSumVectors(key, qry, dest, size, &opts);
            PSTanhActivation(dest, dest, size, &opts);
            if (!trainable)
                scores[k] = PSSumVectorElements(dest, size, &opts);
        }
    }
    if (trainable) {
        if (!whole_seq) success = PSDotMV(sum, score_weights, scores, &opts);
        else {
            int n_iter = num_qry * num_keys;
            PSFloat *p = sum;
            for (int i = 0; i < n_iter; i++) {
                /* TODO: size here could be less than layer size in mha.
                 * Should we use head offset for weights? */
                scores[i] = PSDotProduct(p, score_weights, size, &opts);
                p += layer->size;
            }
        }
        if (!success) goto final;
        /*if (PSMatrixNumDims(scores) > 2) {
            PSMatrix reshaped = PSMatrixReshape(scores, 2, num_qry, num_keys);
            success = reshaped != NULL;
            if (!success) goto final;
            PSMatrixDelete(scores);
            scores = reshaped;
        }*/
    }
final:
    PSMatrixDelete(sum);
    if (!success) {
        PSMatrixDelete(scores);
        scores = NULL;
    }
    return scores;
}

PSMatrix PSGetDotProductScores(PSLayer *layer, PSFloat *query, PSMatrix keys) {
    if (query == NULL || keys == NULL) return NULL;
    PSMatrix scores = NULL;
    PSFloat scale = 0.0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings != NULL) scale = settings->scale;
    if (scale == 0.0) {
        scale = (1 / PSSqrt((PSFloat) layer->size));
        settings->scale = scale;
    }
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    opts.transpose = 2;
    int success = 0;
    int size = PSMatrixDim(keys, 1);
    if (PSHandleSequenceAtOnce(layer))
        success = PSMatrixProduct((PSMatrix) query, keys, &scores, &opts);
    else success = PSMatrixProductVM(query, keys, size, &scores, &opts);
    if (!success || scores == NULL) goto final;
    if (scale != 1) {
        int len = PSMatrixLength(scores);
        PSMultiplyVectorScalar(scores, scale, scores, len, &opts);
    }
final:
    if (!success) {
        PSMatrixDelete(scores);
        scores = NULL;
    }
    return scores;
}

PSMatrix *PSGetAttentionHeads(PSMatrix src, int num_heads) {
    int shape[3] = {0};
    int ndims = PSMatrixDimensions(src, shape);
    int axis = ndims - 1;
    int axis_size = shape[axis];
    if (num_heads <= 1) {
        PSErr(NULL, "invalid number of heads %d: num_heads must be > 1",
              num_heads);
        return NULL;
    }
    if ((axis_size % num_heads) != 0) {
        PSErr(NULL, "matrix split by %d heads wouldn't result in an equal "
              "division", num_heads);
        return NULL;
    }
    int segment_size = axis_size / num_heads;
    if (axis_size <= 0) {
        PSErr(NULL, "invalid number of heads: %d", num_heads);
        return NULL;
    }
    PSMatrix *result = calloc(num_heads, sizeof(PSMatrix));
    if (result == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int head_shape[3];
    memcpy(head_shape, shape, 3 * sizeof(int));
    head_shape[axis] = segment_size;
    int rows = head_shape[0];
    for (int i = 0; i < num_heads; i++) {
        PSMatrix head = PSMatrixZeros(ndims, head_shape[0], head_shape[1],
                                      head_shape[2]);
        if (head == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        for (int r = 0; r < rows; r++) {
            PSFloat *head_row = head + (r * segment_size);
            PSFloat *src_row = src + (r * axis_size) + (i * segment_size);
            PSVectorCopy(head_row, src_row, segment_size);
        }
        result[i] = head;
    }
    return result;
fail:
    deleteHeads(result, num_heads, 0);
    return NULL;
}

PSMatrix PSAttention(PSLayer *layer, PSFloat *query, PSMatrix keys,
                     PSMatrix values, PSMatrix mask,
                     PSMatrix *attention_weights_ptr)
{
    if (values == NULL) values = keys;
    PSMathOpts opts = {.acceleration = layer->network->acceleration};
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention settings");
        return NULL;
    }
    PSMatrix result = NULL, scores = NULL, attention_weights = NULL;
    /* Compute alignment scores */
    if (settings->type == PSAdditiveAttention)
        scores = PSGetAdditiveScores(layer, query, keys);
    else
        scores = PSGetDotProductScores(layer, query, keys);
    int success = scores != NULL;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "failed to compute scores");
        goto final;
    }
    attention_weights = PSMatrixDupShape(scores);
    success = (attention_weights != NULL);
    if (!success) goto final;
    int score_shape[3] = {0};
    int whole_seq = PSHandleSequenceAtOnce(layer);
    int score_ndims = PSMatrixDimensions(scores, score_shape);
    success = score_ndims > 0;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "invalid scores");
        goto final;
    }
    int score_size = score_shape[score_ndims - 1];
    if (mask != NULL) {
        int mask_size = PSMatrixDim(mask, 0);
        if (whole_seq) {
            for (int i = 0; i < score_shape[0]; i++) {
                PSFloat *score_p = scores + (i * score_size);
                PSFloat *mask_p = mask + (i * mask_size);
                PSSumVectors(score_p, mask_p, score_p, score_size, &opts);
            }
        } else {
            PSSumVectors(scores, mask, scores, score_size, &opts);
        }
    }
    /* Compute attention weights */
    if (!whole_seq) PSSoftmax(scores, attention_weights, score_size, &opts);
    else {
        int nscores = score_shape[0], i;
        for (i = 0; i < nscores; i++) {
            int offset = (i * score_size);
            PSFloat *scores_i = scores + offset;
            PSFloat *weights_i = attention_weights + offset;
            PSSoftmax(scores_i, weights_i, score_size, &opts);
        }
    }
    /* Compute attention results */
    /*success = PSMatrixProduct(attention_weights, values, &result, &opts);*/
    if (!whole_seq) {
        opts.transpose = 1;
        success = PSMatrixProduct(values, attention_weights, &result, &opts);
    } else success = PSMatrixProduct(attention_weights, values, &result, &opts);
final:
    if (!success) {
        PSMatrixDelete(result);
        result = NULL;
        PSMatrixDelete(attention_weights);
        attention_weights = NULL;
    }
    PSMatrixDelete(scores);
    if (attention_weights_ptr != NULL)
        *attention_weights_ptr = attention_weights;
    else PSMatrixDelete(attention_weights);
    return result;
}

PSMatrix PSMultiHeadAttention(PSLayer *layer, PSFloat *query, PSMatrix keys,
                              PSMatrix values, int num_heads, PSMatrix mask,
                              PSMatrix *attention_weights_ptr)
{
    PSMatrix result = NULL;
    if (keys == NULL) {
        PSErrNN(NULL, NULL, layer, "missing keys");
        return NULL;
    }
    if (values == NULL) values = keys;
    if (num_heads <= 1)
        return PSAttention(layer,query,keys,values,mask,attention_weights_ptr);
    int success = 1, qry_seqlen = 1, keys_seqlen = PSMatrixDim(keys, 0);
    PSMatrix *k_heads = NULL, *v_heads = NULL, *q_heads = NULL;
    k_heads = PSGetAttentionHeads(keys, num_heads);
    success = k_heads != NULL && k_heads[0] != NULL;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not create heads for keys");
        return NULL;
    }
    v_heads = PSGetAttentionHeads(values, num_heads);
    success = v_heads != NULL;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not create heads for values");
        return NULL;
    }
    int head_hidden_size = PSMatrixDim(k_heads[0], 1);
    if (head_hidden_size != (layer->size / num_heads)) {
        PSErrNN(NULL, NULL, layer, "heads hidden size should be %d, got %d",
                (layer->size / num_heads), head_hidden_size);
    }
    int whole_seq = PSHandleSequenceAtOnce(layer);
    if (whole_seq) {
        qry_seqlen = PSMatrixDim((PSMatrix) query, 0);
        q_heads = PSGetAttentionHeads((PSMatrix) query, num_heads);
        result = PSMatrixZeros(2, qry_seqlen, layer->size);
    } else {
        q_heads = PSVectorSplit(query, layer->size, num_heads);
        result = PSMatrixZeros(1, layer->size);
    }
    success = result != NULL;
    if (!success) goto final;
    success = q_heads != NULL;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not create heads for query");
        return NULL;
    }
    PSFloat *attn_wp = NULL;
    if (attention_weights_ptr != NULL) attn_wp = *attention_weights_ptr;
    for (int n = 0; n < num_heads; n++) {
        PSFloat  *q = q_heads[n];
        PSMatrix k = k_heads[n];
        PSMatrix v = v_heads[n];
        success = q != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "query head[%d] is null", n);
            goto final;
        }
        success = k != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "keys head[%d] is null", n);
            goto final;
        }
        success = v != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "values head[%d] is null", n);
            goto final;
        }
        PSMatrix hres = PSAttention(layer, q, k, v, mask, &attn_wp);
        for (int i = 0; i < qry_seqlen; i++) {
            int head_offset = (i * head_hidden_size);
            int dst_offset = (i * layer->size) + (n * head_hidden_size);
            PSFloat *src = hres + head_offset;
            PSFloat *dst = result + dst_offset;
            PSVectorCopy(dst, src, head_hidden_size);
        }
        PSMatrixDelete(hres);
        attn_wp += (qry_seqlen * keys_seqlen);
    }
final:
    deleteHeads(k_heads, num_heads, 0);
    deleteHeads(v_heads, num_heads, 0);
    deleteHeads(q_heads, num_heads, !whole_seq);
    if (!success) {
        PSMatrixDelete(result);
        result = NULL;
    }
    return result;
}

const char *PSGetAttentionTypeLabel(PSAttentionType type) {
    if (type == PSAdditiveAttention) return "additive";
    else if (type == PSDotAttention) return "dot-product";
    return "invalid";
}

PSAttentionType PSGetAttentionType(PSLayer *layer) {
   if (layer == NULL || layer->type != Attention) return PSInvalidAttention;
   PSAttentionSettings *settings = PSGetAttentionSettings(layer);
   if (settings == NULL) return PSInvalidAttention;
   return settings->type;
}

PSFloat PSGetAttentionScale(PSLayer *layer) {
   if (layer == NULL || layer->type != Attention) return 0.0;
   PSAttentionSettings *settings = PSGetAttentionSettings(layer);
   if (settings == NULL) return 0.0;
   return settings->scale;
}

int PSGetAttentionHeadCount(PSLayer *layer) {
   if (layer == NULL || layer->type != Attention) return 0;
   PSAttentionSettings *settings = PSGetAttentionSettings(layer);
   if (settings == NULL) return 0;
   return settings->num_heads;
}

int PSIsCausalAttention(PSLayer *layer) {
    if (layer == NULL || layer->type != Attention) return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return 0;
    return settings->causal;
}

int PSGetAttnetionTrainableParameters(PSLayer *layer) {
    if (layer == NULL || layer->type != Attention) return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return 0;
    return settings->trainable_parameters;
}

int PSGetAttentionProviders(PSLayer *layer, PSLayer **query_provider,
                            PSLayer **keys_provider, PSLayer **values_provider)
{
    if (layer == NULL || layer->type != Attention) return 0;
    if (query_provider != NULL) *query_provider = getQueryProvider(layer);
    if (keys_provider != NULL) *keys_provider = getKeysProvider(layer);
    if (values_provider != NULL) *values_provider = getValuesProvider(layer);
    return 1;
}

int PSSetAttentionQueryProvider(PSLayer *layer, PSLayer *provider) {
    if (layer == NULL || provider == NULL) return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return 0;
    if (layer->network == NULL) {
        PSErr(__func__, "layer has no network");
        return 0;
    }
    if (PSIsNetworkBuilt(layer->network)) {
        PSErrNN(__func__, NULL, layer, "layer's network is already built");
        return 0;
    }
    if (layer->network != provider->network) {
        int do_raise_err = !PSIsNetworkChain(layer->network);
        if (!do_raise_err) {
            do_raise_err = !PSNetworkChainContains(
                layer->network, provider->network
            );
        }
        if (do_raise_err) {
            PSErrNN(__func__, NULL, layer, "query provider's networks differs "
                    "from layer's network");
            return 0;
        }
    }
    if (!PSUseSequences(provider)) {
        PSErrNN(__func__, NULL, layer, "provider does not use sequences");
        return 0;
    }
    if (provider->size != layer->size) {
        PSErrNN(__func__, NULL, layer, "different provider size and layer "
                "size: %d != %d", provider->size, layer->size);
        return 0;
    }
    settings->query_provider = provider;
    return 1;
}

int PSInitAttentiontionLayer(PSLayer *layer, PSLayerDef *ldef) {
    int success = 1;
    layer->on_delete = deleteAttentionLayer;
    layer->on_copy = copyAttentionLayer;
    layer->build = buildAttentionLayer;
    layer->on_states_init = PSInitAttentionStates;
    layer->on_states_resize = PSResizeAttentionStates;
    PSAttentionSettings *settings = calloc(1, sizeof(*settings));
    if (settings == NULL) goto memerr;
    settings->type = PSAdditiveAttention;
    settings->scale = 0.0;
    settings->num_heads = 0;
    settings->causal = 0;
    settings->trainable_parameters = (
        PS_TRAINABLE_QUERY | PS_TRAINABLE_KEYS | PS_TRAINABLE_VALUES
    );
    int defined_trainble_params = 0;
    if (ldef != NULL) {
        if (ldef->attention_type <= PSDotAttention)
            settings->type = ldef->attention_type;
        settings->scale = ldef->attention_scale;
        settings->causal = ldef->causal_attention;
        settings->num_heads = ldef->attention_heads;
        settings->query_provider = ldef->query_provider;
        settings->keys_provider = ldef->keys_provider;
        settings->values_provider = ldef->values_provider;
        defined_trainble_params = (ldef->trainable_parameters > 0);
        if (defined_trainble_params)
            settings->trainable_parameters = ldef->trainable_parameters;
    }
    if (!defined_trainble_params) {
        if (settings->type == PSAdditiveAttention)
            settings->trainable_parameters |= PS_TRAINABLE_SCORES;
        else settings->trainable_parameters |= PS_TRAINABLE_PROJECTION;
    }
    if (settings->keys_provider == NULL) {
        settings->keys_provider = findKeysProvider(layer);
        success = settings->keys_provider != NULL;
        if (!success) {
            PSErrNN(NULL, layer->network, layer, "No keys_provider");
            goto final;
        }
    }
    success = PSUseSequences(settings->keys_provider);
    if (!success) {
        PSErrNN(NULL, layer->network, layer,
                "keys_provider must use sequences");
        goto final;
    }
    int self_attention = layer->flags & FLAG_SELF_ATTENTION;
    if (settings->query_provider == NULL && self_attention)
        settings->query_provider = settings->keys_provider;
    if (settings->values_provider == NULL || self_attention)
        settings->values_provider = settings->keys_provider;
    success = settings->query_provider == NULL || isValidProvider(
        settings->query_provider,settings->keys_provider
    );
    if (!success) {
        PSErrNN(NULL, layer->network, layer,
                "invalid query_provider");
        goto final;
    }
    success = settings->values_provider != NULL && isValidProvider(
        settings->values_provider,settings->keys_provider
    );
    if (!success) {
        PSErrNN(NULL, NULL, layer, "invalid values_provider");
        goto final;
    }
    layer->extra = settings;
    layer->size = settings->keys_provider->size;
    if (settings->num_heads > 1 && layer->size % settings->num_heads != 0) {
        PSErrNN(NULL, NULL, layer, "invalid num_heads %d: layer size %d must "
                "be multiple of num_heads");
        success = 0;
        goto final;
    }
    PSAttentionData *data = calloc(1, sizeof(*data));
    if (data == NULL) goto memerr;
    int param_types = ATTENTION_WEIGHT_TYPES_COUNT;
    layer->weight_types_count = param_types;
    layer->weights = calloc(param_types, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->biases = calloc((param_types * layer->size) + 1, sizeof(PSFloat));
    int trainable_count = 0, use_bias = !(layer->flags & FLAG_NO_BIAS);
    for (int i = 0; i < param_types; i++) {
        int param_flag = (1 << i), size = layer->size;
        int psize = (i == PS_SCORES_IDX ? 1 : layer->size);
        if (settings->trainable_parameters & param_flag) {
            layer->weights[i] = PSInitWeights(
                layer, psize, size, ldef, 1, 0
            );
            if (layer->weights[i] == NULL) goto memerr;
            if (use_bias) {
                PSFloat *biases = layer->biases + (layer->size * i);
                for (int b = 0; b < psize; b++)
                    biases[b] = PSInitParam(PARAM_TYPE_BIAS, ldef, 1, 0);
            }
            trainable_count++;
        }
    }
    if (trainable_count == 0) layer->flags |= FLAG_NON_TRAINABLE;
    layer->private = data;
    if (!PSUseSequences(layer)) {
        if (PSHandleSequenceAtOnce(layer->network))
            layer->flags |= FLAG_USE_SEQUENCES;
        else if (PSIsRecurrent(layer->network))
            layer->flags |= FLAG_RECURRENT;
        else {
            PSErrNN(NULL, layer->network, layer,
                    "attention layer must use sequences");
            success = 0;
            goto final;
        }
    }
    layer->forward = PSAttentionForward;
final:
    return success;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

int PSAttentionForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    int success = 1;
    va_list args;
    va_start(args, layer);
    int seqlen = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    PSFloat *query = PSGetAttentionQuery(layer, t);
    if (query == NULL) {
        PSErrNN(NULL, NULL, layer, "Could not retrieve query");
        return 0;
    }
    PSMatrix keys = PSGetAttentionKeys(layer);
    if (keys == NULL) {
        PSErrNN(NULL, NULL, layer, "Could not retrieve keys");
        return 0;
    }
    PSMatrix values = PSGetAttentionValues(layer);
    if (values == NULL) {
        PSErrNN(NULL, NULL, layer, "Could not retrieve values");
        return 0;
    }
    int causal = 0, n_heads = 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (settings != NULL) {
        causal = settings->causal;
        n_heads = settings->num_heads;
    }
    PSMatrix mask = NULL;
    int mask_size = 0;
    if (causal) {
        mask = PSGetCausalMask(layer);
        success = mask != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing causal mask");
            goto final;
        }
        mask_size = PSMatrixDim(mask, 0);
        if (t >= mask_size) {
            mask = initOrResizeCausalMask(layer, NULL, t + 1);
            success = mask != NULL;
            if (!success) goto final;
        }
    }
    PSMatrix attention_result = NULL;
    PSMatrix attn_w = data->attention_weights;
    if (n_heads > 1) {
        attention_result = PSMultiHeadAttention(
            layer, query, keys, values,  n_heads, mask, &attn_w
        );
    } else {
        if (mask != NULL) mask = mask + (t * mask_size);
        attention_result = PSAttention(
            layer, query, keys, values, mask, &attn_w
        );
    }
    success = (attention_result != NULL);
    if (!success) {
        PSErrNN(NULL, NULL, layer, "failed to compute attention");
        goto final;
    }
    if (useOutputProjection(layer)) {
        PSMatrix proj_weights = layer->weights[PS_PROJECTION_IDX];
        PSFloat *biases = NULL;
        if (!(layer->flags & FLAG_NO_BIAS))
            biases = layer->biases + (PS_PROJECTION_IDX * layer->size);
        int success = (proj_weights != NULL);
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing projection weights");
            goto final;
        }
        PSMatrix new_result = PSMatrixDupShape(attention_result);
        success = (new_result != NULL);
        if (!success) goto final;
        success = attentionFeedforward(
            attention_result, proj_weights, biases, NULL, new_result, 0,
            layer->network->acceleration
        );
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed to compute attention "
                    "projection");
            goto final;
        }
        PSMatrixDelete(attention_result);
        attention_result = new_result;
    }
    if (PSHandleSequenceAtOnce(layer)) {
        PSMatrixDelete(layer->states);
        layer->states = attention_result;
    } else {
        PSFloat *states = PSGetStates(layer, t);
        PSVectorCopy(states, attention_result, layer->size);
        PSMatrixDelete(attention_result);
        attention_result = NULL;
    }
final:
    return success;
}

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

#define UNUSED(V) ((void) V)
#define ATTENTION_WEIGHT_TYPES_COUNT 5
#define PSGetAttentionSettings(layer) ((PSAttentionSettings *) layer->extra)
#define PSGetAttentionData(layer) ((PSAttentionData *) layer->private)

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
    PSMatrix score_inputs;
    PSMatrix attention_weights;
    PSMatrix output_projection_inputs;
    PSMatrix causal_mask;
    PSMatrix *q_heads;
    PSMatrix *k_heads;
    PSMatrix *v_heads;
    int provider_placeholders;
} PSAttentionData;

/* Forward declarations */

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
int PSBeforeLayerBackprop(PSLayer *layer, PSLayer *previous, int *step,
                          int *seqlen, PSFloat **outputs, PSFloat **inputs,
                          va_list args);
int checkLayerForForward(PSLayer *layer);
static PSLayer *getKeysProvider(PSLayer *layer);
static int hasTrainableQuery(PSLayer *layer);
int PSAttentionForward(PSLayer *layer, ...);
int PSAttentionBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *gradient, ...);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
int PSIsLayerPlaceholder(PSLayer *layer);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);
PSLayer *PSResolveLayerPlaceholder(PSLayer *placeholder, PSModel *model);
void PSUpdateGradientData(PSFloat *gradient_weights, PSFloat *gradient_biases,
                          PSFloat *inputs, PSFloat *delta,
                          int size, int input_size,
                          int seqlen, int acceleration);
int PSUpdateDelta(PSMatrix destdelta, PSMatrix srcdelta, PSMatrix weights,
                  int seqlen, int acceleration);
int PSSoftmaxBackward(PSFloat *softmax_out, PSFloat *delta, PSFloat *dest,
                      uint64_t len, int acceleration);
static int useOutputProjection(PSLayer *layer);

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
    PSAttentionData *data = PSGetAttentionData(layer);
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    int n_heads = 0;
    int placeholders = (data ? data->provider_placeholders : 0);
    if (settings != NULL) {
        n_heads = settings->num_heads;
        PSLayer *provider = settings->query_provider;
        if (provider && (placeholders & 1) && PSIsLayerPlaceholder(provider))
            PSDeleteLayer(provider);
        provider = settings->keys_provider;
        if (provider && (placeholders & 2) && PSIsLayerPlaceholder(provider))
            PSDeleteLayer(provider);
        provider = settings->values_provider;
        if (provider && (placeholders & 3) && PSIsLayerPlaceholder(provider))
            PSDeleteLayer(provider);
    }
    if (data != NULL) {
        PSMatrixDelete(data->query);
        PSMatrixDelete(data->keys);
        PSMatrixDelete(data->values);
        PSMatrixDelete(data->query_inputs);
        PSMatrixDelete(data->key_inputs);
        PSMatrixDelete(data->value_inputs);
        PSMatrixDelete(data->causal_mask);
        PSMatrixDelete(data->score_inputs);
        PSMatrixDelete(data->attention_weights);
        PSMatrixDelete(data->output_projection_inputs);
        deleteHeads(data->q_heads, n_heads, 0);
        deleteHeads(data->k_heads, n_heads, 0);
        deleteHeads(data->v_heads, n_heads, 0);
        free(data);
    }
    free(layer->extra);
    layer->extra = NULL;
}

static int copyAttentionLayer(PSLayer *layer, PSLayer *src) {
    int success = 1;
    PSAttentionSettings *srcsettings = PSGetAttentionSettings(src);
    PSAttentionData *srcdata = PSGetAttentionData(src);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data != NULL) {
        PSMatrixDelete(data->query);
        PSMatrixDelete(data->keys);
        PSMatrixDelete(data->values);
        PSMatrixDelete(data->query_inputs);
        PSMatrixDelete(data->key_inputs);
        PSMatrixDelete(data->value_inputs);
        PSMatrixDelete(data->causal_mask);
        PSMatrixDelete(data->score_inputs);
        PSMatrixDelete(data->attention_weights);
        PSMatrixDelete(data->output_projection_inputs);
        free(data);
        data = NULL;
        layer->private = NULL;
    }
    if (srcsettings != NULL) {
        PSAttentionSettings *cursettings = PSGetAttentionSettings(layer);
        PSAttentionSettings *settings = malloc(sizeof(*srcsettings));
        if (settings == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(settings, srcsettings, sizeof(*srcsettings));
        if (cursettings != NULL) {
            settings->query_provider = cursettings->query_provider;
            settings->keys_provider = cursettings->keys_provider;
            settings->values_provider = cursettings->values_provider;
        }
        if (layer->extra != NULL) free(layer->extra);
        layer->extra = settings;
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
        if (srcdata->score_inputs != NULL) {
            data->score_inputs = PSMatrixDup(srcdata->score_inputs);
            if (data->score_inputs == NULL) {
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
        if (srcdata->output_projection_inputs != NULL) {
            data->output_projection_inputs =
                PSMatrixDup(srcdata->output_projection_inputs);
            if (data->output_projection_inputs == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        if (srcsettings && srcsettings->num_heads > 0) {
            PSMatrix *head_ptrs[] = {
                srcdata->q_heads, data->q_heads,
                srcdata->k_heads, data->k_heads,
                srcdata->v_heads, data->v_heads,
            };
            for (int i = 0; i < 6; i += 2) {
                PSMatrix *srchead = head_ptrs[i];
                PSMatrix *dsthead = head_ptrs[i + 1];
                if (dsthead != NULL)
                    deleteHeads(dsthead, srcsettings->num_heads, 0);
                if (srchead == NULL) continue;
                dsthead = calloc(srcsettings->num_heads, sizeof(PSMatrix));
                if (dsthead == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
                for (int h = 0; h < srcsettings->num_heads; h++) {
                    dsthead[h] = PSMatrixDup(srchead[h]);
                    if (dsthead[h] == NULL && srchead[h] != NULL) {
                        deleteHeads(dsthead, srcsettings->num_heads, 0);
                        return 0;
                    }
                }
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
        if (seqlen != mask_seqlen) {
            PSMatrixDelete(mask);
            data->causal_mask = NULL;
            mask = NULL;
        }
    }
    if (mask == NULL && seqlen > 0) {
        PSMathOpts opts = {.acceleration = layer->model->acceleration};
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

static PSMatrix initOrResizeAttentionData(PSLayer *layer,
                                          PSMatrix *data_pointer,
                                          int seqlen, int do_init,
                                          char *data_name)
{
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) return NULL;
    if (data_pointer == NULL) return NULL;
    PSMatrix matrix = *data_pointer;
    if (matrix != NULL) {
        int data_seqlen = PSMatrixDim(matrix, 0);
        if (do_init || seqlen <= 0) {
            PSMatrixDelete(matrix);
            *data_pointer = NULL;
            matrix = NULL;
        } else if (seqlen > data_seqlen) {
            int steps2add = seqlen - data_seqlen;
            PSMatrix resized = PSMatrixExpand(matrix, steps2add, 0);
            if (resized == NULL) {
                PSErrNN(NULL, NULL, layer, "failed to resize %s", data_name);
                return NULL;
            }
            *data_pointer = matrix = resized;
            return matrix;
        }
    }
    if (seqlen <= 0) return NULL;
    if (matrix == NULL) {
        int size = layer->size, use_heads = 0, nheads = 0;
        if (data_pointer == &data->attention_weights) {
            PSLayer *kprovider = getKeysProvider(layer);
            if (kprovider == NULL) {
                PSErrNN(NULL, NULL, layer, "missing key provider");
                return NULL;
            }
            size = PSStateSequenceLength(kprovider);
            if (size < 1) {
                PSErrNN(NULL, NULL, layer, "key provider is empty");
                return NULL;
            }
            use_heads = 1;
        } else if (data_pointer == &data->score_inputs) {
            /*use_heads = 1;*/
            PSLayer *kprovider = getKeysProvider(layer);
            if (kprovider == NULL) {
                PSErrNN(NULL, NULL, layer, "missing key provider");
                return NULL;
            }
            int klen = PSStateSequenceLength(kprovider);
            matrix = PSMatrixZeros(3, seqlen, klen, size);
            goto set_data_pointer;
        }
        if (use_heads) {
            PSAttentionSettings *settings = PSGetAttentionSettings(layer);
            if (settings != NULL) nheads = settings->num_heads;
        }
        if (use_heads && nheads > 1) {
            if (!PSHandleSequenceAtOnce(layer))
                matrix = PSMatrixZeros(3, seqlen, nheads, size);
            else matrix = PSMatrixZeros(3, nheads, seqlen, size);
        } else matrix = PSMatrixZeros(2, seqlen, size);
set_data_pointer:
        if (matrix == NULL) {
            PSErrNN(NULL, NULL, layer, "failed to create %s", data_name);
            return NULL;
        }
        *data_pointer = matrix;
    }
    return matrix;
}

static int initOrResizeCachedQueryHeads(PSLayer *layer, int steps, int init) {
    int num_heads = PSGetAttentionHeadCount(layer);
    if (num_heads < 1) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) return 0;
    int success = 1;
    if (data->q_heads == NULL) {
        success = init;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing cache for query heads");
            goto final;
        }
        data->q_heads = calloc(num_heads, sizeof(PSMatrix));
        success = (data->q_heads != NULL);
        if (!success) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
    }
    int latent_dim = layer->size / num_heads;
    for (int i = 0; i < num_heads; i++) {
        PSMatrix qh = data->q_heads[i];
        if (qh == NULL) {
            data->q_heads[i] = PSMatrixZeros(2, steps, latent_dim);
            if (data->q_heads[i] == NULL) return 0;
        } else {
            int cursteps = PSMatrixDim(qh, 0);
            if (steps > cursteps) {
                int add = steps - cursteps;
                data->q_heads[i] = PSMatrixExpand(qh, add, 0);
                if (data->q_heads[i] == NULL) return 0;
            }
        }
    }
final:
    return success;
}

int PSInitAttentionStates(PSLayer *layer, uint32_t steps, int retain_previous) {
    UNUSED(retain_previous);
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

    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings != NULL && settings->causal)
        if (initOrResizeCausalMask(layer, data, steps) == NULL) return 0;
    int n_heads = (settings ? settings->num_heads : 0);
    if (n_heads > 0) {
        deleteHeads(data->k_heads, n_heads, 0);
        data->k_heads = NULL;
        deleteHeads(data->v_heads, n_heads, 0);
        data->v_heads = NULL;
    }

    if (steps == 0) {
        if (data->query != NULL) {
            PSMatrixDelete(data->query);
            data->query = NULL;
        }
        if (data->query_inputs != NULL) {
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = NULL;
        }
        if (data->score_inputs != NULL) {
            PSMatrixDelete(data->score_inputs);
            data->score_inputs = NULL;
        }
        if (data->attention_weights != NULL) {
            PSMatrixDelete(data->attention_weights);
            data->attention_weights = NULL;
        }
        if (data->output_projection_inputs != NULL) {
            PSMatrixDelete(data->output_projection_inputs);
            data->output_projection_inputs = NULL;
        }
        if (n_heads > 0) {
            deleteHeads(data->q_heads, n_heads, 0);
            data->q_heads = NULL;
        }
        return 1;
    }
    PSMatrix query = initLayerStates(layer, steps, 0, data->query, NULL);
    if (query == NULL) return 0;
    if (data->query != NULL) PSMatrixDelete(data->query);
    data->query = query;
    int is_training = PSIsModelTraining(layer->model);
    /* If model is not training, other data used for backpropagation is not
     * needed, so exit now. */
    if (!is_training) return 1;
    if (hasTrainableQuery(layer)) {
        PSMatrix query_inputs = initLayerStates(
            layer, steps, 0, data->query_inputs, NULL
        );
        if (query_inputs == NULL) return 0;
        if (data->query_inputs != NULL) PSMatrixDelete(data->query_inputs);
        data->query_inputs = query_inputs;
    }
    PSMatrix *data_p[] = {
        &data->score_inputs,
        &data->attention_weights,
        (useOutputProjection(layer) ? &data->output_projection_inputs : NULL)
    };
    char *names[] = {
        "score_inputs", "attenton weights", "output projection inputs"
    };
    for (size_t i = 0; i < (sizeof(data_p) / sizeof(PSMatrix *)); i++) {
        PSMatrix *mptr = data_p[i];
        if (mptr == NULL) continue;
        char *name = names[i];
        if (!initOrResizeAttentionData(layer, mptr, steps, 1, name))
            return 0;
    }
    if (n_heads > 0 && !PSHandleSequenceAtOnce(layer)) {
        deleteHeads(data->q_heads, n_heads, 0);
        data->q_heads = NULL;
        if (!initOrResizeCachedQueryHeads(layer, steps, 1)) return 0;
    }
    return 1;
}

int PSResizeAttentionStates(PSLayer *layer, uint32_t steps, uint32_t prevlen) {
    UNUSED(prevlen);
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
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    int is_training = PSIsModelTraining(layer->model);

    PSMatrix query = resizeLayerStates(
        layer, steps, data->query, &layer->initial_states
    );
    if (query == NULL) {
        PSMatrixDelete(data->query);
        data->query = NULL;
        layer->initial_states = NULL;
        PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
        return 0;
    }
    data->query = query;

    if (is_training && hasTrainableQuery(layer)) {
        PSMatrix query_inputs = resizeLayerStates(
            layer, steps, data->query_inputs, &layer->initial_states
        );
        if (query_inputs == NULL) {
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = NULL;
            layer->initial_states = NULL;
            PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
            return 0;
        }
        data->query_inputs = query_inputs;
    }
    if (settings != NULL && settings->causal)
        if (!initOrResizeCausalMask(layer, data, steps)) return 0;

    /* If model is not training, other data used for backpropagation is not
     * needed, so exit now. */
    if (!is_training) return 1;
    PSMatrix *data_p[] = {
        &data->score_inputs,
        &data->attention_weights,
        (useOutputProjection(layer) ? &data->output_projection_inputs : NULL)
    };
    char *names[] = {
        "score_inputs", "attenton weights", "output projection inputs"
    };
    for (size_t i = 0; i < (sizeof(data_p) / sizeof(PSMatrix *)); i++) {
        PSMatrix *mptr = data_p[i];
        if (mptr == NULL) continue;
        char *name = names[i];
        if (!initOrResizeAttentionData(layer, mptr, steps, 0, name))
            return 0;
    }
    if (settings != NULL && settings->num_heads > 0) {
        if (PSHandleSequenceAtOnce(layer)) {
            deleteHeads(data->q_heads, settings->num_heads, 0);
            data->q_heads = NULL;
        } else if (!initOrResizeCachedQueryHeads(layer, steps, 0)) return 0;
    }
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
    if (provider == keys_provider) return 1;
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
    PSAttentionData *data = PSGetAttentionData(layer);
    if (PSIsLayerPlaceholder(settings->query_provider)) {
        PSLayer *resolved = PSResolveLayerPlaceholder(
            settings->query_provider, layer->model
        );
        if (resolved == NULL) {
            if (data != NULL) data->provider_placeholders |= 1;
            PSErrNN(NULL, NULL, layer, "could not resolve query provider "
                    "placeholder");
            return NULL;
        }
        if (data != NULL)
            data->provider_placeholders &= ~((unsigned) 1);
        PSDeleteLayer(settings->query_provider);
        settings->query_provider = resolved;
    }
    return settings->query_provider;
}

static PSLayer *getValuesProvider(PSLayer *layer) {
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) return NULL;
    PSAttentionData *data = PSGetAttentionData(layer);
    PSLayer *provider = settings->values_provider;
    if (PSIsLayerPlaceholder(provider)) {
        PSLayer *resolved = PSResolveLayerPlaceholder(
            provider, layer->model
        );
        if (resolved == NULL) {
            if (data != NULL) data->provider_placeholders |= 3;
            PSErrNN(NULL, NULL, layer, "could not resolve values provider "
                    "placeholder");
            return NULL;
        }
        PSDeleteLayer(provider);
        settings->values_provider = provider = resolved;
    }
    if (provider == NULL) provider = settings->keys_provider;
    return provider;
}

static int buildAttentionLayer(PSLayer *layer) {
    if (layer == NULL) return 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (settings == NULL) {
        PSErrNN(__func__, NULL, layer, "attention layer has no settings");
        return 0;
    }
    PSLayer *provider = getKeysProvider(layer);
    if (provider == NULL) {
        PSErrNN(__func__, NULL, layer, "missing keys provider");
        return 0;
    } else if (PSIsLayerPlaceholder(provider)) {
        if (data != NULL) data->provider_placeholders |= 2;
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
        if (biases != NULL) PSAddVectors(dest, biases, dest, input_size, &opts);
        if (activate) activate(dest, dest, input_size, &opts);
    } else {
        opts.transpose = 2;
        success = PSDot(x, weights, dest, &opts);
        if (!success) return 0;
        opts.transpose = 0;
        if (biases != NULL || activate != NULL) {
            int xshape[3] = {0};
            int nd = PSMatrixDimensions(x, xshape), i;
            int nrows = (nd > 1 ? xshape[0] : 1);
            PSFloat *row = dest;
            for (i = 0; i < nrows; i++) {
                if (biases != NULL)
                    PSAddVectors(row, biases, row, input_size, &opts);
                if (activate)
                    activate(row, row, input_size, &opts);
                row += input_size;
            }
        }
    }
    return success;
}

static int applyCausalMask(PSLayer *layer, PSFloat *mask, PSMatrix scores,
                           int score_size)
{
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    if (PSHandleSequenceAtOnce(layer)) {
        int masklen = PSMatrixLength((PSMatrix) mask),
            scorelen = PSMatrixLength(scores);
        if (masklen != scorelen) {
            PSErrNN(NULL, NULL, layer, "mask and scores with different "
                    "sizes: %d != %d", masklen, scorelen);
            return 0;
        }
        PSAddVectors(scores, mask, scores, masklen, &opts);
    } else {
        if (score_size <= 0) {
            int shape[3] = {0};
            int ndims = PSMatrixDimensions(scores, shape);
            if (ndims <= 0) return 0;
            score_size = shape[ndims - 1];
        }
        PSAddVectors(scores, mask, scores, score_size, &opts);
    }
    return 1;
}

PSFloat *PSGetAttentionQuery(PSLayer *layer, int t) {
    int is_training = PSIsModelTraining(layer->model),
        trainable = hasTrainableQuery(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
            return NULL;
        }
        layer->private = data;
    }
    PSFloat *query = NULL;
    PSLayer *provider = getQueryProvider(layer);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention query provider");
        return NULL;
    }
    int from_prev_model = provider->model->index < layer->model->index;
    int is_after = provider->model->index == layer->model->index &&
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
        if (query == NULL) return NULL;
        PSVectorCopy(query, provider->states, seqlen * layer->size);
        if (data->query != NULL) PSMatrixDelete(data->query);
        data->query = query;
        if (is_training && trainable) {
            if (data->query_inputs == NULL) {
                PSErrNN(NULL, NULL, layer, "missing query_inputs");
                return NULL;
            }
            PSMatrixDelete(data->query_inputs);
            data->query_inputs = PSMatrixDup(data->query);
            if (data->query_inputs == NULL) {
                PSErrNN(NULL, NULL, layer, "could not store query_inputs into "
                        "cache");
                return NULL;
            }
        }
    } else {
        PSFloat *inputs = NULL;
        if (is_after) inputs = PSGetStates(provider, t - 1);
        else if (from_prev_model) inputs = PSGetOutputs(provider);
        else inputs = PSGetStates(provider, t);
        if (inputs == NULL) {
            if (t > 0) goto empty_provider;
            inputs = layer->initial_states;
            if (inputs == NULL) goto empty_provider;
        }
        int cur_seqlen = PSStateSequenceLength(layer);
        if (t >= cur_seqlen || data->query == NULL) {
            int steps = (t >= cur_seqlen ? t + 1 : cur_seqlen);
            if (!PSResizeLayerStates(layer, steps)) {
                if (layer->model)
                    PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
                PSErrNN(
                    NULL, NULL, layer,
                    "could not resize recurrent hidden states"
                );
                return NULL;
            }
        }
        query = data->query + (t * layer->size);
        PSVectorCopy(query, inputs, layer->size);
        if (is_training && trainable) {
            if (data->query_inputs == NULL) {
                PSErrNN(NULL, NULL, layer, "missing query_inputs");
                return NULL;
            }
            query_inputs = data->query_inputs + (t * layer->size);
            PSVectorCopy(query_inputs, inputs, layer->size);
        }
    }
    if (trainable) {
        int acceleration = layer->model->acceleration, ok;
        PSMatrix weights = layer->weights[PS_QUERY_IDX];
        int use_bias = !(layer->flags & FLAG_NO_BIAS);
        PSFloat *biases = (use_bias ? layer->biases : NULL);
        if (whole_seq) {
            PSMatrix updated_query = PSMatrixDupShape((PSMatrix) query);
            if (updated_query == NULL) return NULL;
            ok = attentionFeedforward(
                query, weights, biases, layer->activate,
                updated_query, 0, acceleration
            );
            PSMatrixDelete(data->query);
            query = data->query = updated_query;
        } else {
            int do_free_inputs = 0;
            if (query_inputs == NULL) {
                query_inputs = malloc(layer->size * sizeof(PSFloat));
                ok = (query_inputs != NULL);
                if (!ok) goto memerr;
                PSVectorCopy(query_inputs, query, layer->size);
                do_free_inputs = 1;
            }
            ok = attentionFeedforward(
                query_inputs, weights, biases, layer->activate,
                query, 1, acceleration
            );
            if (do_free_inputs) free(query_inputs);
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
            PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
            return NULL;
        }
        layer->private = data;
    }
    int is_training = PSIsModelTraining(layer->model),
        trainable = hasTrainableKeys(layer);
    PSMatrix keys = data->keys;
    if (keys != NULL) return keys; /*TODO: only if key provider in prev. net?*/
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
        PSMatrix updated_keys = PSMatrixDupShape(keys);
        if (updated_keys == NULL) return NULL;
        if (is_training) {
            PSMatrixDelete(data->key_inputs);
            data->key_inputs = PSMatrixDup(keys);
            if (data->key_inputs == NULL) {
                PSErrNN(NULL, NULL, layer, "could not store key_inputs into "
                        "cache");
                return NULL;
            }
        }
        int acceleration = layer->model->acceleration;
        PSMatrix weights = layer->weights[PS_KEYS_IDX];
        PSFloat *biases = NULL;
        if (!(layer->flags & FLAG_NO_BIAS))
            biases = layer->biases + (layer->size * PS_KEYS_IDX);
        int ok = attentionFeedforward(
            keys, weights, biases, layer->activate,
            updated_keys, 0, acceleration
        );
        if (!ok) {
            PSErrNN(NULL, NULL, layer, "attention keys feedforward failed");
            PSMatrixDelete(updated_keys);
            return NULL;
        }
        PSMatrixDelete(keys);
        keys = data->keys = updated_keys;
    }
    return keys;
}

PSMatrix PSGetAttentionValues(PSLayer *layer) {
    PSMatrix values = NULL;
    int is_training = PSIsModelTraining(layer->model),
        trainable = hasTrainableValues(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) {
        data = calloc(1, sizeof(*data));
        if (data == NULL) {
            PSPrintMemoryErrorMsg();
            PSModelSetStatus(layer->model, STATUS_ERROR, NULL);
            return NULL;
        }
        layer->private = data;
    }
    values = data->values;
    if (values != NULL) return values;
    PSLayer *provider = getValuesProvider(layer);
    if (provider == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention values provider");
        return NULL;
    }
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
        PSMatrix updated_values = PSMatrixDupShape(values);
        if (updated_values == NULL) return NULL;
        if (is_training) {
            PSMatrixDelete(data->value_inputs);
            data->value_inputs = PSMatrixDup(values);
            if (data->value_inputs == NULL) {
                PSErrNN(NULL,NULL,layer,"could not store value inputs cache");
                PSMatrixDelete(updated_values);
                return NULL;
            }
        }
        int acceleration = layer->model->acceleration;
        PSMatrix weights = layer->weights[PS_VALUES_IDX];
        PSFloat *biases = NULL;
        if (!(layer->flags & FLAG_NO_BIAS))
            biases = layer->biases + (layer->size * PS_VALUES_IDX);
        int ok = attentionFeedforward(
            values, weights, biases, layer->activate, updated_values,
            0, acceleration
        );
        if (!ok) {
            PSErrNN(NULL, NULL, layer, "attention values feedforward failed");
            PSMatrixDelete(updated_values);
            return NULL;
        }
        PSMatrixDelete(values);
        values = data->values = updated_values;
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
    PSModel *model = layer->model;
    while (provider == NULL && model != NULL) {
        PSLayer *prev = NULL;
        if (model == layer->model)
            prev = PSGetPreviousLayer(layer);
        else prev = PSGetOutputLayer(layer->model);
        while (prev != NULL) {
            if (PSUseSequences(prev)) {
                provider = prev;
                break;
            }
            prev = PSGetPreviousLayer(prev);
        }
        model = model->previous;
    }
    return provider;
}

static void getGradientWeightsMap(PSLayer *layer, PSGradient *gradient,
                                  PSFloat **weights_map)
{
    PSFloat *grad_p = gradient->weights;
    for (int i = 0; i < ATTENTION_WEIGHT_TYPES_COUNT; i++) {
        if (layer->weights[i] != NULL) {
            weights_map[i] = grad_p;
            grad_p += PSMatrixLength(layer->weights[i]);
        } else weights_map[i] = NULL;
    }
}

static PSFloat *storeAttentionStates(PSLayer *layer, PSMatrix *states,
                                     PSMatrix *dest, int t)
{
    if (dest == NULL) dest = &(layer->states);
    if (*dest == NULL || *states == NULL) return NULL;
    if (PSHandleSequenceAtOnce(layer)) {
        PSMatrixDelete(*dest);
        *dest = *states;
        return *dest;
    } else {
        int len = PSMatrixLength(*states);
        int dest_shape[3] = {0};
        int dest_ndims = PSMatrixDimensions(*dest, dest_shape);
        if (dest_ndims == 1) {
            PSMatrixDelete(*dest);
            *dest = *states;
            return *dest;
        }
        if (t >= dest_shape[0]) {
            PSErrNN(NULL, NULL, layer, "step %d is out of bounds for "
                    "destination (%d)", t, dest_shape[0]);
            return NULL;
        } else if (PSMatrixStride(*dest, 0) != len) {
            PSErrNN(NULL, NULL, layer, "size of data to be stored differs from "
                    "destination stride: %d != %d", len,
                    PSMatrixStride(*dest, 0));
            return NULL;
        }
        PSFloat *dest_p = *dest + (t * len);
        PSVectorCopy(dest_p, *states, len);
        PSMatrixDelete(*states);
        *states = NULL;
        return dest_p;
    }
}

static int storeQueryHeads(PSLayer *layer, PSFloat **q_heads, int t) {
    int n_heads = PSGetAttentionHeadCount(layer);
    if (n_heads < 1) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) return 0;
    int do_resize = 0;
    if (data->q_heads == NULL || data->q_heads[0] == NULL) {
        if (t > 0) {
            PSErrNN(NULL, NULL, layer, "missing cache for query heads");
            return 0;
        }
        do_resize = 1;
    } else {
        int curlen = PSMatrixDim(data->q_heads[0], 0);
        do_resize =  (t >= curlen);
    }
    if (do_resize) {
        if (!initOrResizeCachedQueryHeads(layer, t, t == 0)) {
            PSErrNN(NULL, NULL, layer, "could not resize or init query heads");
            return 0;
        }
        if (data->q_heads == NULL || data->q_heads[0] == NULL) return 0;
    }
    int latent_dim = layer->size / n_heads;
    for (int i = 0; i < n_heads; i++) {
        PSFloat *qh = q_heads[i];
        PSMatrix data_qh = data->q_heads[i];
        if (data_qh == NULL) return 0;
        PSVectorCopy(data_qh + (t * latent_dim), qh, latent_dim);
    }
    return 1;
}

static int updateAttentionGradientsAndDelta(PSLayer *layer, PSFloat **gweights,
                                            PSFloat *gbiases, int param_type,
                                            PSFloat *inputs, PSMatrix delta,
                                            PSMatrix new_delta, int seqlen)
{
    PSMatrix weights = layer->weights[param_type];
    PSFloat *gw = gweights[param_type], *gb = NULL;
    if (weights == NULL || gw == NULL) {
        PSErrNN(__func__, NULL, layer, "paramters of type %d are not trainable",
                param_type);
        return 0;
    }
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (use_bias && gbiases) gb = gbiases + (layer->size * param_type);
    int shape[3] = {0};
    int ndims = PSMatrixDimensions(weights, shape);
    int size;
    if (param_type == PS_SCORES_IDX) size = 1;
    else size = (ndims > 1 ? shape[0] : 1);
    int acceleration = layer->model->acceleration;
    if (seqlen < 1) seqlen = 1;
    PSUpdateGradientData(gw, gb, inputs, delta, size, layer->size,
                         seqlen, acceleration);
    if (new_delta == NULL) return 1;
    return PSUpdateDelta(new_delta, delta, weights, seqlen, acceleration);
}

PSMatrix PSGetAdditiveScores(PSLayer *layer, PSFloat *query, PSMatrix keys,
                             PSMatrix *score_inputs_ptr)
{
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
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    for (int i = 0; i < num_qry; i++) {
        int qry_offset = (i * size);
        PSFloat *qrysum = sum + (i * num_keys * size);
        for (int k = 0; k < num_keys; k++) {
            int offset = (k * size);
            PSFloat *dest = qrysum + offset;
            PSFloat *key = keys + offset;
            PSFloat *qry = query + qry_offset;
            PSAddVectors(key, qry, dest, size, &opts);
            PSTanhActivation(dest, dest, size, &opts);
            if (!trainable)
                scores[k] = PSVectorReduceSum(dest, size, &opts);
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
    }
    if (score_inputs_ptr != NULL)
        *score_inputs_ptr = (success ? sum : NULL);
final:
    if (score_inputs_ptr == NULL || *score_inputs_ptr != sum)
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
    int size = PSMatrixDim(keys, 1);
    if (scale == 0.0) {
        scale = (1 / PSSqrt((PSFloat) size));
        settings->scale = scale;
    }
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    opts.transpose = 2;
    int success = 0;
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
                     PSMatrix values, PSFloat *mask,
                     PSMatrix *attention_weights_ptr,
                     PSMatrix *score_inputs_ptr)
{
    if (values == NULL) values = keys;
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention settings");
        return NULL;
    }
    PSMatrix result = NULL, scores = NULL, attention_weights = NULL;
    /* Compute alignment scores */
    if (settings->type == PSAdditiveAttention)
        scores = PSGetAdditiveScores(layer, query, keys, score_inputs_ptr);
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
        success = applyCausalMask(layer, mask, scores, score_size);
        if (!success) {
            PSErrNN(__func__, NULL, layer, "could not apply causal mask");
            goto final;
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
                              PSMatrix *attention_weights_ptr,
                              PSFloat ***heads_ptr)
{
    PSMatrix result = NULL;
    if (keys == NULL) {
        PSErrNN(NULL, NULL, layer, "missing keys");
        return NULL;
    }
    if (values == NULL) values = keys;
    if (num_heads <= 1) {
        return PSAttention(
            layer, query, keys, values, mask, attention_weights_ptr, NULL
        );
    }
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
    if (attention_weights_ptr != NULL) {
        if (!whole_seq)
            *attention_weights_ptr = PSMatrixZeros(2, num_heads, keys_seqlen);
        else {
            *attention_weights_ptr = PSMatrixZeros(
                3, num_heads, qry_seqlen, keys_seqlen
            );
        }
        success = *attention_weights_ptr != NULL;
        if (!success) goto final;
    }
    for (int n = 0; n < num_heads; n++) {
        PSFloat *q = q_heads[n];
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
        PSMatrix attn_w = NULL;
        PSMatrix *attn_wp = (attention_weights_ptr ? &attn_w : NULL);
        PSMatrix hres = PSAttention(layer, q, k, v, mask, attn_wp, NULL);
        success = hres != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed to get attention results "
                    "for head %d", n);
            goto final;
        }
        if (attention_weights_ptr != NULL) success = attn_w != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed to get attention weights "
                    "for head %d", n);
            PSMatrixDelete(hres);
            goto final;
        }
        for (int i = 0; i < qry_seqlen; i++) {
            int head_offset = (i * head_hidden_size);
            int dst_offset = (i * layer->size) + (n * head_hidden_size);
            PSFloat *src = hres + head_offset;
            PSFloat *dst = result + dst_offset;
            PSVectorCopy(dst, src, head_hidden_size);
        }
        PSMatrixDelete(hres);
        if (attn_w != NULL) {
            int len = PSMatrixLength(attn_w);
            PSFloat *dest = *attention_weights_ptr + (n * len);
            PSVectorCopy(dest, attn_w, len);
            PSMatrixDelete(attn_w);
            attn_w = NULL;
        }
    }
final:
    if (success && heads_ptr != NULL) {
        heads_ptr[0] = q_heads;
        heads_ptr[1] = k_heads;
        heads_ptr[2] = v_heads;
    } else {
        deleteHeads(k_heads, num_heads, 0);
        deleteHeads(v_heads, num_heads, 0);
        deleteHeads(q_heads, num_heads, !whole_seq);
    }
    if (!success) {
        PSMatrixDelete(result);
        result = NULL;
    }
    return result;
}

int PSAdditiveAttentionBackward(PSLayer *layer, PSMatrix *dscores,
                                PSFloat *score_inputs, PSMatrix keys,
                                PSFloat *query, PSFloat **dquery,
                                PSMatrix *dkeys, PSGradient *gradient)
{
    UNUSED(query);
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    int success = 1, klen = PSMatrixLength(keys), size = PSMatrixDim(keys, 1),
        nkeys = PSMatrixDim(keys, 0), i;
    if (size == 0 || klen == 0) return 0;
    PSMatrix delta = PSMatrixDupShape(keys), score_deriv = NULL;
    if (delta == NULL) return 0;
    score_deriv = PSMatrixDupShape(keys);
    success = score_deriv != NULL;
    if (!success) goto final;
    PSFloat *delta_p = delta;
    if (hasTrainableScores(layer)) {
        PSMatrix weights = layer->weights[PS_SCORES_IDX];
        success = weights != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing weights for scores");
            goto final;
        }
        success = (gradient != NULL);
        if (!success) {
            PSErrNN(NULL, NULL, layer, "no gradient for score parameters");
            goto final;
        }
        PSFloat *gradient_weights[ATTENTION_WEIGHT_TYPES_COUNT] = {0};
        getGradientWeightsMap(layer, gradient, gradient_weights);
        success = updateAttentionGradientsAndDelta(
            layer, gradient_weights, gradient->biases, PS_SCORES_IDX,
            score_inputs, *dscores, NULL, nkeys
        );
        if (!success) {
            PSErrNN(NULL, NULL, layer, "could not update score gradients");
            goto final;
        }
        for (i = 0; i < nkeys; i++) {
            PSFloat ds = (*dscores)[i];
            PSMultiplyVectorScalar(weights, ds, delta_p, size, &opts);
            delta_p += size;
        }
    } else {
        for (i = 0; i < nkeys; i++) {
            PSFloat ds = *dscores[i];
            PSVectorFill(delta_p, ds, size, &opts);
            delta_p += size;
        }
    }
    PSTanhDerivative(score_inputs, score_deriv, klen, &opts);
    PSMultiplyVectors(delta, score_deriv, delta, klen, &opts);
    if (*dquery == NULL) *dquery = delta;
    else {
        if (PSHandleSequenceAtOnce(layer))
            PSAddVectors(*dquery, delta, *dquery, klen, &opts);
        else {
            delta_p = delta;
            for (i = 0; i < nkeys; i++) {
                PSAddVectors(*dquery, delta_p, *dquery, size, &opts);
                delta_p += size;
            }
        }
    }
    if (*dkeys == NULL) *dkeys = delta;
    else PSAddVectors(*dkeys, delta, *dkeys, klen, &opts);
final:
    if (delta && delta != *dkeys) PSMatrixDelete(delta);
    PSMatrixDelete(score_deriv);
    return success;
}

int PSDotAttentionBackward(PSLayer *layer, PSMatrix *dscores, PSMatrix keys,
                           PSFloat *query, PSFloat **dquery,
                           PSMatrix *dkeys)
{
    PSFloat scale = 0.0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings != NULL) scale = settings->scale;
    if (scale == 0.0) {
        int size = PSMatrixDim(keys, 1);
        scale = (1 / PSSqrt((PSFloat) size));
        if (settings != NULL) settings->scale = scale;
    }
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    int whole_seq = PSHandleSequenceAtOnce(layer), success = 1,
        klen = PSMatrixLength(keys), qlen;
    opts.argtype[2] = 'V';
    success = PSMatrixProduct(*dscores, keys, (PSMatrix *) dquery, &opts);
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not compute delta for query");
        goto final;
    }
    opts.argtype[2] = '\0';
    opts.transpose = 1;
    if (whole_seq) {
        qlen = PSMatrixLength(*((PSMatrix *) dquery));
        success = PSMatrixProduct(*dscores, query, dkeys, &opts);
    } else {
        /* Read qlen for keys last axis since they're equal even in
         * multihead attention. */
        qlen = PSMatrixDim(keys, 1);
        PSMatrix q = PSMatrixFromArray(query, 2, 1, qlen);
        success = q != NULL;
        if (!success) goto final;
        success = PSMatrixProduct(*dscores, q, dkeys, &opts);
        PSMatrixDelete(q);
    }
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not compute delta for keys");
        goto final;
    }
    if (scale != 1) {
        PSMultiplyVectorScalar(*dquery, scale, *dquery, qlen, &opts);
        PSMultiplyVectorScalar(*dkeys, scale, *dkeys, klen, &opts);
    }
final:
    return success;
}

int PSAttentionBackward(PSLayer *layer, PSMatrix delta, PSFloat *query,
                        PSMatrix keys, PSMatrix values, PSFloat *mask,
                        PSFloat *attention_weights, PSFloat *score_inputs,
                        PSFloat *dquery, PSMatrix *dkeys, PSMatrix *dvalues,
                        PSGradient *gradient)
{
    if (layer == NULL || delta == NULL) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (data == NULL || settings == NULL) return 0;
    if (query == NULL) query = data->query;
    if (keys == NULL) keys = data->keys;
    if (values == NULL) values = data->values;
    if (attention_weights == NULL) attention_weights = data->attention_weights;
    if (score_inputs == NULL) score_inputs = data->score_inputs;
    PSMatrix dweights = NULL, dscores = NULL;
    int keys_seqlen = PSMatrixDim(keys, 0);
    PSMathOpts opts = {.acceleration = layer->model->acceleration};
    /* Compute delta for values */
    int attn_seqlen = 1, whole_seq = PSHandleSequenceAtOnce(layer);
    if (whole_seq) attn_seqlen = PSStateSequenceLength(layer);
    PSMatrix attn_w = PSMatrixFromArray(
        attention_weights, 2, attn_seqlen, keys_seqlen
    );
    int ok = attn_w != NULL;
    if (!ok) goto final;
    opts.transpose = 1;
    ok = PSMatrixProduct(attn_w, delta, dvalues, &opts);
    PSMatrixDelete(attn_w);
    attn_w = NULL;
    if (!ok) {
        PSErrNN(NULL, NULL, layer, "could not compute delta for values");
        goto final;
    }
    /* Compute delta for attention weights */
    opts.transpose = 2;
    ok = PSMatrixProduct(delta, values, &dweights, &opts);
    if (!ok) {
        PSErrNN(NULL, NULL, layer, "could not compute delta for attention "
                "weights");
        goto final;
    }
    opts.transpose = 0;
    /* Compute delta for scores */
    dscores = PSMatrixDupShape(dweights);
    int dwshape[3] = {0};
    int dwdims = PSMatrixDimensions(dweights, dwshape), dwseqlen = 1,
        dwstride, i;
    ok = keys_seqlen > 0;
    if (!ok) {
        PSErrNN(NULL, NULL, layer, "invalid sequence length for keys: %s",
                keys_seqlen);
        goto final;
    }
    if (dwdims == 1) dwstride = dwshape[0];
    else {
        dwseqlen = dwshape[0];
        dwstride = dwshape[dwdims - 1];
    }
    PSFloat *attw_p = attention_weights, *dweights_p = dweights,
            *dscore_p = dscores;
    for (i = 0; i < dwseqlen; i++) {
        ok =  PSSoftmaxBackward(attw_p, dweights_p, dscore_p, dwstride,
                                opts.acceleration);
        attw_p += keys_seqlen;
        dweights_p += dwstride;
        dscore_p += dwstride;
        if (!ok) goto final;
    }
    /* NOTE: Not sure about applying mask during backward step */
    UNUSED(mask);
    /*if (mask != NULL) {
        ok = applyCausalMask(layer, mask, dscores, 0);
        if (!ok) {
            PSErrNN(__func__, NULL, layer, "could not apply causal mask");
            goto final;
        }
    }*/
    /* Compute delta for query and keys */
    if (settings->type == PSAdditiveAttention) {
        ok = PSAdditiveAttentionBackward(
            layer, &dscores, score_inputs, keys, query, &dquery, dkeys,gradient
        );
    } else {
        ok = PSDotAttentionBackward(layer, &dscores, keys, query,
                                    &dquery, dkeys);
    }
final:
    PSMatrixDelete(dweights);
    PSMatrixDelete(dscores);
    return ok;
}

int PSMultiHeadAttentionBackward(PSLayer *layer, PSMatrix delta, PSFloat *mask,
                                 PSFloat *dquery, PSMatrix *dkeys,
                                 PSMatrix *dvalues, int t)
{
    if (layer == NULL || delta == NULL) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (data == NULL || settings == NULL) return 0;
    int n_heads = settings->num_heads;
    if (n_heads < 1) return 0;
    PSMatrix attention_weights = data->attention_weights;
    if (data->q_heads == NULL) {
        PSErrNN(NULL, NULL, layer, "missing cached query heads");
        return 0;
    }
    if (data->k_heads == NULL) {
        PSErrNN(NULL, NULL, layer, "missing cached key heads");
        return 0;
    }
    if (data->v_heads == NULL) {
        PSErrNN(NULL, NULL, layer, "missing cached value heads");
        return 0;
    }
    if (attention_weights == NULL) {
        PSErrNN(NULL, NULL, layer, "missing cached attention weights");
        return 0;
    }
    int success = 1, whole_seq = PSHandleSequenceAtOnce(layer);
    PSMatrix *d_heads = PSGetAttentionHeads(delta, n_heads);
    success = (d_heads != NULL);
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not create heads for delta");
        goto final;
    }
    PSFloat *attn_w = attention_weights;
    int attn_stride;
    if (!whole_seq) {
        if (t > 0) attn_w += (t * PSMatrixStride(attention_weights, 0));
        attn_stride = PSMatrixStride(attention_weights, 1);
    } else {
        attn_stride = PSMatrixStride(attention_weights, 0);
    }
    int latent_dim = layer->size / n_heads, qlen = 1, klen;
    for (int n = 0; n < n_heads; n++) {
        PSMatrix dh = d_heads[n];
        PSMatrix qh = data->q_heads[n];
        PSMatrix kh = data->k_heads[n];
        PSMatrix vh = data->v_heads[n];
        success = (dh && qh && kh && vh);
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing some cached heads");
            goto final;
        }
        PSFloat *qh_p = qh;
        if (!whole_seq) qh_p = qh_p + (t * latent_dim);
        if (n == 0) {
            if (whole_seq) qlen = PSMatrixDim(qh, 0);
            klen = PSMatrixDim(kh, 0);
            success = (qlen > 0 && klen > 0);
            if (!success) goto final;
        }
        PSMatrix dqh = PSMatrixZeros(2, qlen, latent_dim);
        PSMatrix dkh = PSMatrixDupShape(kh);
        PSMatrix dvh = PSMatrixDupShape(vh);
        success = (dqh && dkh && dvh);
        if (!success) {
            PSMatrixDelete(dqh);
            PSMatrixDelete(dkh);
            PSMatrixDelete(dvh);
            goto final;
        }
        PSFloat *attn_wp = attn_w + (n * attn_stride);
        success = PSAttentionBackward(layer, dh, qh_p, kh, vh, mask, attn_wp,
                                      NULL, dqh, &dkh, &dvh, NULL);
        if (!success) {
            PSMatrixDelete(dqh);
            PSMatrixDelete(dkh);
            PSMatrixDelete(dvh);
            goto final;
        }
        for (int i = 0; i < qlen; i++) {
            PSFloat *dh_p = dqh + (i * latent_dim);
            PSFloat *dst_p = dquery + (i * layer->size) + (n * latent_dim);
            PSVectorCopy(dst_p, dh_p, latent_dim);
        }
        for (int i = 0; i < klen; i++) {
            PSFloat *dkh_p = dkh + (i * latent_dim);
            PSFloat *dvh_p = dvh + (i * latent_dim);
            PSFloat *dst_k_p = *dkeys + (i * layer->size) + (n * latent_dim);
            PSFloat *dst_v_p = *dvalues + (i * layer->size) + (n * latent_dim);
            PSVectorCopy(dst_k_p, dkh_p, latent_dim);
            PSVectorCopy(dst_v_p, dvh_p, latent_dim);
        }
        PSMatrixDelete(dqh);
        PSMatrixDelete(dkh);
        PSMatrixDelete(dvh);
    }
final:
    deleteHeads(d_heads, n_heads, 0);
    return success;
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

int PSGetAttentionTrainableParameters(PSLayer *layer) {
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
    if (layer->model == NULL) {
        PSErr(__func__, "layer has no model");
        return 0;
    }
    if (PSModelIsBuilt(layer->model)) {
        PSErrNN(__func__, NULL, layer, "layer's model is already built");
        return 0;
    }
    if (layer->model != provider->model) {
        int do_raise_err = !PSIsModelChain(layer->model);
        if (!do_raise_err)
            do_raise_err = !PSModelChainContains(layer->model,provider->model);
        if (do_raise_err) {
            PSErrNN(__func__, NULL, layer, "query provider's model differs "
                    "from layer's model");
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
    layer->extra = settings;
    PSAttentionData *data = calloc(1, sizeof(*data));
    if (data == NULL) goto memerr;
    layer->private = data;
    settings->type = PSAdditiveAttention;
    settings->scale = 0.0;
    settings->num_heads = 0;
    settings->causal = 0;
    settings->trainable_parameters = (
        PS_TRAINABLE_QUERY | PS_TRAINABLE_KEYS
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
        else {
            settings->trainable_parameters |= PS_TRAINABLE_VALUES;
            settings->trainable_parameters |= PS_TRAINABLE_PROJECTION;
        }
    }
    if (settings->keys_provider == NULL) {
        settings->keys_provider = findKeysProvider(layer);
        success = settings->keys_provider != NULL;
        if (!success) {
            PSErrNN(NULL, layer->model, layer, "No keys_provider");
            goto final;
        }
    }
    success = PSUseSequences(settings->keys_provider);
    if (!success) {
        PSErrNN(NULL, layer->model, layer,
                "keys_provider must use sequences");
        goto final;
    }
    int self_attention = layer->flags & FLAG_SELF_ATTENTION;
    if (settings->query_provider == NULL && self_attention)
        settings->query_provider = settings->keys_provider;
    if (settings->values_provider == NULL || self_attention)
        settings->values_provider = settings->keys_provider;
    int qprovider_is_placeholder =
        PSIsLayerPlaceholder(settings->query_provider);
    if (qprovider_is_placeholder) data->provider_placeholders |= 1;
    success = settings->query_provider == NULL || qprovider_is_placeholder ||
              isValidProvider(settings->query_provider,settings->keys_provider);
    if (!success) {
        PSErrNN(NULL, layer->model, layer,
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
    layer->size = settings->keys_provider->size;
    int is_multihead = settings->num_heads > 1;
    if (is_multihead && PSAdditiveAttention == settings->type) {
        settings->num_heads = 1;
        is_multihead = 0;
        PSWarn("Layer[%d]: multihead attention is not supported with additive "
               "attention type");
    }
    if (is_multihead && layer->size % settings->num_heads != 0) {
        PSErrNN(NULL, NULL, layer, "invalid num_heads %d: layer size %d must "
                "be multiple of num_heads");
        success = 0;
        goto final;
    }
    int param_types = ATTENTION_WEIGHT_TYPES_COUNT;
    layer->weight_types_count = param_types;
    layer->weights = calloc(param_types, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    int bias_size = ((param_types - 1) * layer->size) + 1;
    layer->biases = calloc(bias_size, sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
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
    if (!PSUseSequences(layer)) {
        if (PSHandleSequenceAtOnce(layer->model))
            layer->flags |= FLAG_USE_SEQUENCES;
        else if (PSIsRecurrent(layer->model))
            layer->flags |= FLAG_RECURRENT;
    }
    layer->forward = PSAttentionForward;
    layer->backprop = PSAttentionBackprop;
final:
    return success;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

int PSAttentionForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    int is_training = PSIsModelTraining(layer->model);
    int whole_seq = PSHandleSequenceAtOnce(layer);
    int success = 1, t = 0;
    va_list args;
    va_start(args, layer);
    int seqlen = va_arg(args, int);
    if (!whole_seq) t = va_arg(args, int);
    va_end(args);
    if (!PSBeforeSequenceForward(layer, seqlen, t)) return 0;
    PSFloat *query = PSGetAttentionQuery(layer, t);
    if (query == NULL) {
        PSErrNN(NULL, NULL, layer, "could not retrieve query");
        return 0;
    }
    PSMatrix keys = PSGetAttentionKeys(layer);
    if (keys == NULL) {
        PSErrNN(NULL, NULL, layer, "could not retrieve keys");
        return 0;
    }
    PSMatrix values = PSGetAttentionValues(layer);
    if (values == NULL) {
        PSErrNN(NULL, NULL, layer, "could not retrieve values");
        return 0;
    }
    int causal = 0, n_heads = 0;
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    PSAttentionData *data = PSGetAttentionData(layer);
    if (is_training && data == NULL) {
        PSErrNN(NULL, NULL, layer, "missing attention data cache");
        return 0;
    }
    if (settings != NULL) {
        causal = settings->causal;
        n_heads = settings->num_heads;
    }
    PSMatrix attention_result = NULL;
    PSMatrix attn_w = NULL;
    PSMatrix score_ipt = NULL;
    PSMatrix mask = NULL;
    PSFloat *mask_p = NULL;
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
            mask = initOrResizeCausalMask(layer, NULL, t);
            success = mask != NULL;
            if (!success) goto final;
        }
        if (!whole_seq && t > 0) mask_p = mask + (t * mask_size);
        else mask_p = mask;
    }
    PSMatrix *attn_w_ptr = NULL, *score_ipt_ptr = NULL;
    if (is_training) {
        attn_w_ptr = &attn_w;
        if (PSAdditiveAttention == settings->type) score_ipt_ptr = &score_ipt;
    }
    PSFloat **heads[3] = {0};
    int multihead = n_heads > 1;
    if (multihead) {
        PSFloat ***head_ptr = (is_training ? heads : NULL);
        attention_result = PSMultiHeadAttention(
            layer, query, keys, values, n_heads, mask_p, attn_w_ptr, head_ptr
        );
    } else {
        attention_result = PSAttention(
            layer, query, keys, values, mask_p, attn_w_ptr, score_ipt_ptr
        );
    }
    success = (attention_result != NULL);
    if (!success) {
        PSErrNN(NULL, NULL, layer, "failed to compute attention");
        goto final;
    }
    PSFloat *stored = NULL;
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
            layer->model->acceleration
        );
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed to compute attention "
                    "projection");
            PSMatrixDelete(new_result);
            goto final;
        }
        if (is_training) {
            stored = storeAttentionStates(
                layer, &attention_result, &data->output_projection_inputs, t
            );
            success = stored != NULL;
            if (!success) {
                PSErrNN(NULL, NULL, layer, "could not store "
                        "output_projection_inputs");
                PSMatrixDelete(new_result);
                goto final;
            }
        }
        if (data->output_projection_inputs != attention_result)
            PSMatrixDelete(attention_result);
        attention_result = new_result;
    }
    stored = storeAttentionStates(layer, &attention_result, NULL, t);
    success = stored != NULL;
    if (!success) {
        PSErrNN(NULL, NULL, layer, "could not store attention results");
        goto final;
    }
    /* Eventually store data for backpropagation */
    if (!is_training) goto final;
    if (attn_w != NULL) {
        stored = storeAttentionStates(
            layer, &attn_w, &data->attention_weights, t
        );
        success = stored != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "could not store attention weights");
            goto final;
        }
    }
    if (score_ipt != NULL) {
        stored = storeAttentionStates(
            layer, &score_ipt, &data->score_inputs, t
        );
        success = stored != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "could not store score inputs");
            goto final;
        }
    }
    if (multihead) {
        PSFloat **q_heads = heads[0];
        PSMatrix *k_heads = heads[1];
        PSMatrix *v_heads = heads[2];
        deleteHeads(data->k_heads, n_heads, 0);
        data->k_heads = k_heads;
        deleteHeads(data->v_heads, n_heads, 0);
        data->v_heads = v_heads;
        if (whole_seq) {
            deleteHeads(data->q_heads, n_heads, 0);
            data->q_heads = q_heads;
        } else {
            success = storeQueryHeads(layer, q_heads, t);
            if (!success) goto final;
            deleteHeads(q_heads, n_heads, 1);
            q_heads = heads[0] = NULL;
        }
    }
final:
    if (!success) {
        if (attn_w != NULL && attn_w != data->attention_weights)
            PSMatrixDelete(attn_w);
        if (score_ipt != NULL && score_ipt != data->score_inputs)
            PSMatrixDelete(score_ipt);
        if (attention_result && attention_result != layer->states)
            PSMatrixDelete(attention_result);
        if (heads[0] != NULL && heads[0] != data->q_heads)
            deleteHeads(heads[0], n_heads, !whole_seq);
        if (heads[1] != NULL && heads[1] != data->k_heads)
            deleteHeads(heads[1], n_heads, 0);
        if (heads[2] != NULL && heads[2] != data->v_heads)
            deleteHeads(heads[2], n_heads, 0);
    }
    return success;
}

int PSAttentionBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *gradient, ...)
{
    PSMatrix delta = layer->delta;
    if (delta == NULL) return 0;
    PSAttentionData *data = PSGetAttentionData(layer);
    if (data == NULL) return 0;
    if (data->query == NULL) {
        PSErrNN(__func__, NULL, layer, "missing cached query");
        return 0;
    }
    if (data->keys == NULL) {
        PSErrNN(__func__, NULL, layer, "missing cached keys");
        return 0;
    }
    if (data->values == NULL) {
        PSErrNN(__func__, NULL, layer, "missing cached values");
        return 0;
    }
    PSAttentionSettings *settings = PSGetAttentionSettings(layer);
    if (settings == NULL) {
        PSErrNN(__func__, NULL, layer, "missing stored settings");
        return 0;
    }
    int whole_seq = PSHandleSequenceAtOnce(layer),
        t = 0, seqlen = 1, nheads = 1;
    if (settings != NULL) nheads = settings->num_heads;
    int success = 1;
    PSFloat *outputs = NULL, *inputs = NULL;
    va_list args;
    va_start(args, gradient);
    success = PSBeforeLayerBackprop(layer, previous_layer, &t, &seqlen,
                                    &outputs, &inputs, args);
    va_end(args);
    if (!success) return 0;
    PSMatrix dquery = NULL, dkeys = NULL, dvalues = NULL;
    PSFloat *gradient_weights[ATTENTION_WEIGHT_TYPES_COUNT] = {0};
    getGradientWeightsMap(layer, gradient, gradient_weights);
    if (useOutputProjection(layer)) {
        delta = PSMatrixDupShape(delta);
        if (delta == NULL) {
            PSErrNN(NULL, NULL, layer, "could not create updated delta");
            return 0;
        }
        PSFloat *proj_inputs = data->output_projection_inputs;
        if (!whole_seq) proj_inputs += (t * layer->size);
        success = updateAttentionGradientsAndDelta(layer, gradient_weights,
            gradient->biases, PS_PROJECTION_IDX, proj_inputs,
            layer->delta, delta, (whole_seq ? seqlen : 1)
        );
        if (!success) {
            PSMatrixDelete(delta);
            PSErrNN(NULL, NULL, layer, "failed backward pass for output "
                    "projections");
            goto final;
        }
        PSMatrixDelete(layer->delta);
        layer->delta = delta;
    }
    dkeys = PSMatrixDupShape(data->keys);
    dvalues = PSMatrixDupShape(data->values);
    if (whole_seq) dquery = PSMatrixDupShape(data->query);
    else dquery = PSMatrixZeros(2, 1, layer->size);
    success = (dkeys && dvalues && dquery);
    if (!success) goto final;
    PSMatrix mask = PSGetCausalMask(layer);
    PSFloat *mask_p = NULL;
    if (mask != NULL) {
        if (!whole_seq && t > 0) {
            int mask_size = PSMatrixDim(mask, 0);
            mask_p = mask + (t * mask_size);
        } else mask_p = mask;
    }
    if (nheads > 1) {
        success = PSMultiHeadAttentionBackward(layer, delta, mask_p, dquery,
                                               &dkeys, &dvalues, t);
    } else {
        PSFloat *query = data->query + (t * layer->size);
        PSFloat *attn_w = data->attention_weights;
        success = query != NULL && attn_w != NULL;
        if (!success) {
            PSErrNN(__func__, NULL, layer, "missing cached query and/or "
                    "attention weights");
            goto final;
        }
        PSFloat *score_inputs = data->score_inputs;
        if (PSAdditiveAttention == settings->type) {
            success = score_inputs != NULL;
            if (!success) {
                PSErrNN(__func__, NULL, layer, "missing cached score inputs");
                goto final;
            }
            score_inputs += (t * PSMatrixStride(score_inputs, 0));
        }
        int attn_w_shape[3] = {0};
        int attn_w_ndims = PSMatrixDimensions(data->attention_weights,
                                              attn_w_shape);
        int attn_w_size = attn_w_shape[attn_w_ndims - 1];
        attn_w += (t * attn_w_size);
        success = PSAttentionBackward(layer, delta, query, data->keys,
                                      data->values, mask_p, attn_w,
                                      score_inputs, dquery,
                                      &dkeys, &dvalues, gradient);
    }
    if (!success) goto final;
    if (hasTrainableValues(layer)) {
        success = dvalues != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing delta for values");
            goto final;
        }
        PSFloat *proj_inputs = data->value_inputs;
        success = updateAttentionGradientsAndDelta(layer, gradient_weights,
            gradient->biases, PS_VALUES_IDX, proj_inputs,
            dvalues, NULL, PSMatrixDim(data->value_inputs, 0)
        );
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed update gradients for value "
                    "input projections");
            goto final;
        }
    }
    if (hasTrainableKeys(layer)) {
        success = dkeys != NULL;
        if (!success) {
            PSErrNN(NULL, NULL, layer, "missing delta for keys");
            goto final;
        }
        PSFloat *proj_inputs = data->key_inputs;
        success = updateAttentionGradientsAndDelta(layer, gradient_weights,
            gradient->biases, PS_KEYS_IDX, proj_inputs,
            dkeys, NULL, PSMatrixDim(data->key_inputs, 0)
        );
        if (!success) {
            PSErrNN(NULL, NULL, layer, "failed update gradients for key "
                    "input projections");
            goto final;
        }
    }
    if (hasTrainableQuery(layer)) {
        delta = PSMatrixDupShape(delta);
        if (delta == NULL) {
            PSErrNN(NULL, NULL, layer, "could not create updated delta");
            return 0;
        }
        if (dquery == NULL) {
            PSErrNN(NULL, NULL, layer, "missing delta for query");
            return 0;
        }
        PSFloat *proj_inputs = data->query_inputs;
        if (!whole_seq) proj_inputs += (t * layer->size);
        success = updateAttentionGradientsAndDelta(layer, gradient_weights,
            gradient->biases, PS_QUERY_IDX, proj_inputs,
            dquery, delta, (whole_seq ? seqlen : 1)
        );
        if (!success) {
            PSMatrixDelete(delta);
            PSErrNN(NULL, NULL, layer, "failed backward pass for query input "
                    "projections");
            goto final;
        }
        PSMatrixDelete(layer->delta);
        layer->delta = delta;
    }
final:
    PSMatrixDelete(dkeys);
    PSMatrixDelete(dvalues);
    PSMatrixDelete(dquery);
    return success;
}

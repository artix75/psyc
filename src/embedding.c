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
#include "embedding.h"
#include "log.h"
#include "maths.h"
#include "config.h"

#define Word2VecTraingDataElements(n_tokens, window) \
    (n_tokens * (window * 2) - (window * (window + 1)))
#define GetEmbeddingSettings(layer) ((PSEmbeddingSettings *) layer->extra)

typedef struct {
    PSEmbeddingType embedding_type;
    int vocabulary_size;
    int window_size;
    PSTrainingOptions training_options;
    PSFloat *training_data;
    int training_data_size;
    const char *save_pretrained_to;
} PSEmbeddingSettings;

typedef PSFloat *(*PSEmbeddingDataGenerator) (PSFloat *tokens,
                                              size_t token_count,
                                              int window_size,
                                              int vocabulary_size,
                                              int onehot,
                                              int *n_elements);

/* Forward declarations */
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
PSActivationFunction PSGetActivationDerivative(PSActivationFunction func);
int PSFullBackprop(PSLayer *layer, PSLayer *previous_layer,
                 PSGradient *gradient, ...);
int PSFullFeedforward(PSNeuralNetwork *network, PSLayer *layer, ...);
PSFloat *PSGetInputsFromTrainingData(PSFloat *training_data, int data_size,
                                     int num_elements, int input_size,
                                     int label_size, int recurrent_input,
                                     int recurrent_output,
                                     int *count, size_t *result_size);

/* Helpers */

static int getWord2VecTrainingDataLength(int tokens_count, int window_size,
                                         int onehot, int vocabulary_size)
{
    int n_elements = Word2VecTraingDataElements(tokens_count, window_size);
    if (n_elements <= 0) return 0;
    n_elements *= 2; /* Include both inputs and labels */
    if (onehot && vocabulary_size > 0) n_elements *= vocabulary_size;
    return n_elements;
}

PSFloat *PSCreateWord2VecTrainingData(PSFloat *tokens, size_t token_count,
                                      int window_size, int vocabulary_size,
                                      int onehot, int *num_elements_p)
{
    if (token_count == 0) return NULL;
    PSFloat *training_data = NULL;
    int n_elements = getWord2VecTrainingDataLength(
        token_count, window_size, onehot, vocabulary_size
    );
    if (n_elements <= 0) {
        PSErr(__func__, "traing data would contain no elements");
        return NULL;
    }
    if (num_elements_p != NULL) *num_elements_p = n_elements;
    size_t size = (size_t) n_elements * sizeof(PSFloat);
    training_data = malloc(size * sizeof(PSFloat));
    if (training_data == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int last_idx = (int) token_count - 1, i, j;
    PSFloat *data_p = training_data + 1;
    for (i = 0; i < (int) token_count; i++) {
        int min_idx = (int) i - window_size;
        int max_idx = (int) i + window_size;
        if (min_idx < 0) min_idx = 0;
        if (min_idx >= last_idx) max_idx = last_idx;
        for (j = min_idx; j < max_idx; j++) {
            if (j == i) continue;
            if (onehot) {
                PSFloat input = tokens[i], label = tokens[j];
                if (input >= vocabulary_size) {
                    PSErr(
                        __func__, "Token[%d] is out-of-range (%lld > %d))",
                        i, (long long) input, vocabulary_size - 1
                    );
                    goto fail;
                }
                if (label >= vocabulary_size) {
                    PSErr(
                        __func__, "Token[%d] is out-of-range (%lld > %d))",
                        j, (long long) label, vocabulary_size - 1
                    );
                    goto fail;
                }
                *(data_p++) = input;
                *(data_p++) = label;
            } else {
                PSFloat *input = tokens + i, *label = tokens + j;
                memcpy(data_p, input, vocabulary_size + sizeof(PSFloat));
                data_p += vocabulary_size;
                memcpy(data_p, label, vocabulary_size + sizeof(PSFloat));
                data_p += vocabulary_size;
            }
        }
    }
    return training_data;
fail:
    free(training_data);
    return NULL;
}

/* Pretraining */

PSNeuralNetwork *PSCreateEmbeddingTrainer(PSLayer *layer) {
    PSNeuralNetwork *trainer = NULL;
    if (layer == NULL) return NULL;
    if (layer->type != Embedding) return NULL;
    if (layer->pretrainer != NULL) return layer->pretrainer;
    if (layer->network != NULL && layer->network->flags & FLAG_PRETRAINER)
        return NULL;
    PSEmbeddingSettings *settings = GetEmbeddingSettings(layer);
    int vocabulary_size = 0;
    if (settings != NULL) vocabulary_size = settings->vocabulary_size;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (vocabulary_size <= 0) {
        if (previous != NULL) {
            if (previous->flags & FLAG_ONEHOT)
                vocabulary_size = PSGetOneHotLayerVectorSize(previous);
            else vocabulary_size = previous->size;
        }
        if (vocabulary_size <= 0) {
            PSErr(__func__, "Layer[%d]: could not determine vocabulary_size",
                  layer->index);
            return NULL;
        }
    }
    trainer = PSCreateNetwork("Embedding Layer Trainer");
    trainer->flags |= FLAG_PRETRAINER;
    int onehot_network = (layer->network->flags & FLAG_ONEHOT);
    if (onehot_network) trainer->flags |= FLAG_ONEHOT;
    int input_flags = (previous != NULL ? previous->flags : 0);
    PSRemoveFlag(input_flags, FLAG_RECURRENT);
    if (onehot_network) input_flags |= FLAG_ONEHOT;
    int onehot_input = input_flags & FLAG_ONEHOT;
    PSLayer *input_layer = PSAddLayer(
        trainer, FullyConnected, vocabulary_size, PSLDEF(.flags = input_flags)
    );
    int onehot = onehot_network | onehot_input;
    if (input_layer == NULL) goto fail;
    PSLayer *embedding_layer = PSAddLayer(
        trainer, Embedding, layer->size, NULL
    );
    if (embedding_layer == NULL) goto fail;
    PSLayer *output_layer = PSAddLayer(trainer, SoftMax, vocabulary_size, NULL);
    if (onehot) output_layer->flags |= FLAG_ONEHOT;
    layer->pretrainer = trainer;
    return trainer;
fail:
    if (trainer != NULL) PSDeleteNetwork(trainer);
    return NULL;
}

int PSPretrainEmbeddingLayer(PSLayer *layer, PSFloat *training_data,
                             int data_size)
{
    if (layer == NULL) return 0;
    PSNeuralNetwork *network = layer->network;
    if (network == NULL) {
        PSErr(__func__, "Layer[%d] has no network");
        return 0;
    }
    PSEmbeddingSettings *settings = GetEmbeddingSettings(layer);
    if (settings == NULL) {
        PSErr(__func__, "Layer[%d] has no embedding settings", layer->index);
        return 0;
    }
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(__func__, "Layer[%d] has no previous layer", layer->index);
        return 0;
    }
    int success = 1;
    PSFloat *pretrain_data = NULL, *tokens = NULL;
    if (settings->training_data != NULL && settings->training_data_size > 0) {
        training_data = settings->training_data;
        data_size = settings->training_data_size;
    }
    if (training_data == NULL) {
        success = 0;
        goto final;
    }
    PSEmbeddingDataGenerator make_data = NULL;
    if (settings->embedding_type == PSWord2Vec)
        make_data = PSCreateWord2VecTrainingData;
    else {
        PSErr(__func__, "Layer[%d]: invalid embedding_type %d",
              layer->index, settings->embedding_type);
        success = 0;
        goto final;
    }
    int num_tokens = 0;
    PSLayer *output_layer = network->layers[network->size - 1];
    if (output_layer == NULL) {
        PSErr(__func__, "Network '%d' has no output layer",
              network->name);
        success = 0;
        goto final;
    }
    int ysize = (output_layer->flags & FLAG_ONEHOT ? 1 : output_layer->size);
    int recurrent_input = PSIsRecurrent(network->layers[0]),
        recurrent_output = PSIsRecurrent(output_layer);
    tokens = PSGetInputsFromTrainingData(training_data, data_size,
                                     0, network->input_size,
                                     ysize, recurrent_input,
                                     recurrent_output,
                                     &num_tokens, NULL);
    if (tokens == NULL) {
        PSErr(__func__, "Layer[%d]: failed to extract tokens from "
              "training data", layer->index);
        success = 0;
        goto final;
    }
    int window_size = settings->window_size;
    if (window_size <= 0) window_size = 2;
    int vocabulary_size = settings->vocabulary_size;
    if (vocabulary_size <= 0) {
        PSErr(__func__, "Layer[%d]: invalid vocabulary_size", layer->index);
        success = 0;
        goto final;
    }
    int onehot = previous->flags & FLAG_ONEHOT;
    int pretaing_num_elements = 0;
    pretrain_data = make_data(tokens, (size_t) num_tokens, window_size,
                              vocabulary_size, onehot, &pretaing_num_elements);
    if (pretrain_data == NULL) {
        PSErr(__func__, "Layer[%d]: failed to generate pretraining data",
              layer->index);
        success = 0;
        goto final;
    }
    free(tokens);
    tokens = NULL;
    PSNeuralNetwork *pretrainer = PSCreateEmbeddingTrainer(layer);
    if (pretrainer == NULL) {
        PSErr(__func__, "Layer[%d]: failed to build pretrainer",
              layer->index);
        success = 0;
        goto final;
    }
    PSTrainingOptions *options = &(settings->training_options);
    if (options->epochs <= 0)
        options->epochs = 50;/* TODO: use a constant or automatic calc.*/
    if (options->learning_rate <= 0)
        options->learning_rate = 0.1; /* TODO: use a constant or autocalc.*/
    PSTrain(pretrainer, pretrain_data, pretaing_num_elements, NULL, 0,
            options);
    if (pretrainer->status == STATUS_ERROR) {
        PSErr(__func__, "Layer[%d]: pretraining failed!",
              layer->index);
        success = 0;
        goto final;
    }
    layer->pretrained = 1;
    PSLayer *pretrained_layer = pretrainer->layers[1];
    uint64_t wlen = PSMatrixLength(pretrained_layer->weights[0]);
    memcpy(layer->weights[0], pretrained_layer->weights[0], wlen);
    PSInfo("Successfully pretrained embedding layer %d", layer->index);
    if (settings->save_pretrained_to != NULL)
        success = PSSaveLayer(layer, settings->save_pretrained_to, 0);
final:
    free(tokens);
    free(pretrain_data);
    return success;
}

/* Embedding laer functions */

int PSGetEmbeddingVocabularySize(PSLayer *layer) {
    if (layer == NULL) return 0;
    PSEmbeddingSettings *settings = GetEmbeddingSettings(layer);
    if (settings == NULL) return 0;
    return settings->vocabulary_size;
}

void PSDeleteEmbeddingSettings(PSEmbeddingSettings *settings) {
    if (settings->training_data != NULL) free(settings->training_data);
    free(settings);
}

void PSDeleteEmbeddingLayer(PSLayer *layer) {
    if (layer == NULL) return;
    if (layer->extra != NULL) {
        PSEmbeddingSettings *settings = GetEmbeddingSettings(layer);
        PSDeleteEmbeddingSettings(settings);
        layer->extra = NULL;
    }
}

int PSEmbeddingLayerCopy(PSLayer *layer, PSLayer *src) {
    if (layer == NULL || src == NULL) return 0;
    if (src->extra != NULL) {
        if (layer->extra == NULL) {
            layer->extra = calloc(1, sizeof(PSEmbeddingSettings));
            if (layer->extra == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
        PSEmbeddingSettings *dstsettings = GetEmbeddingSettings(layer);
        PSEmbeddingSettings *srcsettings = GetEmbeddingSettings(src);
        memcpy(dstsettings, srcsettings, sizeof(*srcsettings));
        dstsettings->training_data = NULL;
        if (srcsettings->training_data != NULL &&
            srcsettings->training_data_size > 0)
        {
            size_t datasize = srcsettings->training_data_size * sizeof(PSFloat);
            dstsettings->training_data = malloc(datasize);
            if (dstsettings->training_data == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            memcpy(dstsettings->training_data, srcsettings->training_data,
                   datasize);
        }
    } else if (layer->extra != NULL) {
        PSDeleteEmbeddingSettings((PSEmbeddingSettings *) layer->extra);
        layer->extra = NULL;
    }
    return 1;
}

/* Initialization */
int PSInitEmbeddingLayer(PSLayer *layer, int size, int previous_size,
                         PSLayerDef *ldef)
{
    static PSLayerDef default_def = {.embedding_type = PSWord2Vec};
    if (layer->index == 0) {
        PSErr(NULL, "Embedding layer cannot be the first layer");
        return 0;
    }
    layer->on_delete = PSDeleteEmbeddingLayer;
    layer->on_copy = PSEmbeddingLayerCopy;
    layer->size = size;
    int vocabulary_size, i;
    PSLayer *previous = PSGetPreviousLayer(layer);
    if (previous == NULL) {
        PSErr(NULL, "Embedding layer has no previous layer");
        return 0;
    }
    if (previous->flags & FLAG_ONEHOT)
        vocabulary_size = PSGetOneHotLayerVectorSize(previous);
    else vocabulary_size = previous->size;
    if (vocabulary_size == 0) {
        PSErr(NULL, "Embedding layer's vocabulary size is zero");
        return 0;
    }
    PSEmbeddingSettings *settings = calloc(1, sizeof(*settings));
    if (settings == NULL) goto memerr;
    layer->extra = settings;
    if (ldef == NULL) ldef = &default_def;
    settings->embedding_type = ldef->embedding_type;
    settings->vocabulary_size = vocabulary_size;
    if (ldef->pretraining_options != NULL)
        settings->training_options = *ldef->pretraining_options;
    if (ldef->training_data != NULL && ldef->training_data_size > 0) {
        settings->training_data_size = ldef->training_data_size;
        size_t datasize = ldef->training_data_size * sizeof(PSFloat);
        settings->training_data = malloc(datasize);
        if (settings->training_data == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        memcpy(settings->training_data, ldef->training_data, datasize);
    }
    settings->save_pretrained_to = ldef->save_pretrained_to;
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) goto memerr;
    layer->weights = malloc(sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weights[0] = PSInitWeights(layer, size, vocabulary_size, ldef, 1, 0);
    if (layer->weights == NULL) {
        PSErr(NULL, "Could not create embedding weights");
        return 0;
    }
    layer->weight_types_count = 1;
    layer->biases = calloc(size, sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->activate = ldef->activation;
    if (layer->activate != NULL)
        layer->derivative = PSGetActivationDerivative(layer->activate);
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) goto memerr;
        neuron->index = i;
        neuron->extra = NULL;
        neuron->bias = layer->biases + i;
        *(neuron->bias) = PSInitParam(PARAM_TYPE_BIAS, ldef, 1.0, 0.0);
        if (layer->index > 0 && previous_size > 0) {
            neuron->weights = layer->weights[0] + (i * previous_size);
        } else {
            neuron->bias = NULL;
            neuron->weights = NULL;
        }
        neuron->layer = layer;
        layer->neurons[i] = neuron;
    }
    layer->feedforward = PSFullFeedforward;
    layer->backprop = PSFullBackprop;
    layer->pretrain = PSPretrainEmbeddingLayer;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

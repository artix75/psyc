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

#include "recurrent.h"
#include "maths.h"
#include "activation.h"
#include "utils.h"
#include "log.h"

#define RNN_WEIGHT_TYPES_COUNT 2
#define RNN_INIT_SCALE 0.01
#define UNUSED(V) ((void) V)

/* Forward declaration. */

int PSResizeRecurrentHiddenStates(PSLayer *layer, uint32_t steps);
int PSRecurrentBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *lgradients, ...);
int PSRecurrentFeedforward(PSLayer *layer, ...);
int PSBeforeSequenceFeedforward(PSLayer *layer, int seqlen, int t);

/* External functions */

int checkLayerForFeedforward(PSLayer *layer);
int PSOnehotInputsFeedforward(PSLayer *layer, int weights_index,
                              PSFloat *outputs, int t, int apply_biases,
                              int do_activate);
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
void handleLayerFeedforwardDebug(PSLayer *layer, const char *func,
                                 PSMathOpts *opts);
int PSApplyDerivative(PSActivationFunction derivative, PSFloat *delta,
                      PSFloat *outputs, int size, PSMathOpts *opts);
/* Recurrent network functions */

PSMatrix PSGetRecurrentHiddenWeights(PSLayer *layer) {
    if (layer->type != Recurrent || layer->weights == NULL) return NULL;
    return layer->weights[1];
}

void PSDeleteRNNLayer(PSLayer *layer) {
    if (layer->extra != NULL) free(layer->extra);
    layer->extra = NULL;
}

int PSInitRecurrentLayer(PSNeuralNetwork *network, PSLayer *layer,
                         int size, int ws, PSLayerDef *ldef)
{
    int i;
    layer->on_delete = PSDeleteRNNLayer;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->states = PSMatrixZeros(2, 1, size);
    if (layer->states == NULL) goto memerr;
    layer->weights = calloc(RNN_WEIGHT_TYPES_COUNT, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weights[0] = PSInitWeights(
        layer, size, ws, ldef, 1, RNN_INIT_SCALE
    );
    if (layer->weights[0] == NULL) goto memerr;
    layer->weight_types_count = 1;
    layer->weights[1] = PSInitWeights(
        layer, size, size, ldef, 1, RNN_INIT_SCALE
    );
    if (layer->weights[1] == NULL) goto memerr;
    layer->weight_types_count = 2;
    layer->biases = malloc(size * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->extra = calloc(size, sizeof(PSFloat));
    if (layer->extra == NULL) goto memerr;
    PSMatrix weights = layer->weights[0];
    int bias_init_mode = (ldef != NULL ? ldef->bias_init_mode : INIT_MODE_AUTO);
    int rand_bias = (bias_init_mode == INIT_MODE_RAND);
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) goto memerr;
        neuron->index = i;
        if (rand_bias) layer->biases[i] = PSInitParam(PARAM_TYPE_BIAS,ldef,1,0);
        else layer->biases[i] = 0.0;
        neuron->weights = weights + (i * ws);
        neuron->bias = layer->biases + i;
        layer->neurons[i] = neuron;
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    if (layer->activate == NULL) {
        layer->activate = PSTanhActivation;
        layer->derivative = PSTanhDerivative;
    }
    layer->feedforward = PSRecurrentFeedforward;
    layer->backprop = PSRecurrentBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

PSFloat *PSGetRecurrentNeuronHiddenWeights(PSNeuron *neuron) {
    if (neuron->layer == NULL) return NULL;
    if (neuron->layer->type != Recurrent) return NULL;
    if (neuron->layer->weights == NULL || neuron->layer->weights[1] == NULL)
        return NULL;
    uint64_t widx = neuron->index * neuron->layer->size;
    if (widx >= PSMatrixLength(neuron->layer->weights[1])) {
        PSErr(__func__, "Layer[%d] Neuron[%d] index is out of bounds",
              neuron->layer->index, neuron->index);
        return NULL;
    }
    return neuron->layer->weights[1] + widx;
}

/* Feedforward Functions */

int PSRecurrentFeedforward(PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    PSNeuralNetwork *network = layer->network;
    PSMathOpts dpopt = {.acceleration = network->acceleration};
    handleLayerFeedforwardDebug(layer, __func__, &dpopt);
    PSLayer *previous = network->layers[layer->index - 1];
    va_list args;
    va_start(args, layer);
    int steps = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    /* Checks */
    if (!PSBeforeSequenceFeedforward(layer, steps, t)) return 0;
    PSFloat *hidden_values = (PSFloat *) layer->extra;
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network);
    int onehot = previous->flags & FLAG_ONEHOT;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int ignore_inputs = 0;
    int feed_previous_step = (t > 0 || layer->initial_states != NULL);
    /* If layer is the first recurrent layer of a one-to-many network, inputs
     * are fed just in the very first step. */
    if (!PSIsRecurrent(previous) && layer == first_recurrent)
        ignore_inputs = (t > 0);
    int prev_t = t - 1;
    PSFloat *inputs = NULL;
    PSFloat *prev_states = NULL;
    PSFloat *outputs = PSGetStates(layer, t);
    if (outputs == NULL) {
        PSErr(NULL,  "Layer[%d]: layer[%d] has no states");
        return 0;
    }
    /* Forward step: feed inputs (from previous layer) */
    PSMatrix input_weights = layer->weights[0],
             hidden_weights = layer->weights[1];
    dpopt.argtype[1] = 'V';
    if (ignore_inputs) goto forward_previous_step;
    /* Feed inputs by multiplying them by layer weights. If there's no previous
     * states, also eventually add biases and activate states with `activate`
     * function. */
    if (onehot) {
        if (!PSOnehotInputsFeedforward(layer, 0, NULL, t, 0, 0)) return 0;
    } else {
        inputs = PSGetStates(previous, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d]: previous layer[%d] has NULL "
                  "outputs", layer->index, previous->index);
            return 0;
        }
        if (!PSDot(input_weights, inputs, outputs, &dpopt)) return 0;
    }
forward_previous_step:
    /* Forward step: feed previous layer's states */
    if (!feed_previous_step) goto final;
    /* Feed previous states by multiplying them with hidden weights, add
     * the result to current states (`outputs`) and eventually add biases and
     * apply `activate` function. */
    prev_states = PSGetStates(layer, prev_t);
    dpopt.tmpdest = hidden_values;
    dpopt.store_mode = PS_STORE_MODE_ADD;
    if (!PSDot(hidden_weights, prev_states, outputs, &dpopt)) return 0;
final:
    dpopt.store_mode = PS_STORE_MODE_SET;
    if (use_bias)
        PSSumVectors(outputs, layer->biases, outputs, layer->size, &dpopt);
    if (layer->activate != NULL)
        layer->activate(outputs, NULL, layer->size, &dpopt);
    return 1;
}

/* Backpropagation Functions */

int PSRecurrentBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *gradients, ...)
{
    va_list args;
    va_start(args, gradients);
    int t = va_arg(args, int);
    int lowest_t = va_arg(args, int);
    va_end(args);
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int lsize = layer->size, i, w, tt;
    PSMatrix prev_layer_delta = previous_layer->delta;
    int do_truncate = (t - lowest_t) > 0;
    int has_initial_states = (layer->initial_states != NULL);
    int onehot = (previous_layer->flags & FLAG_ONEHOT);
    int onehot_vector_size = 0, onehot_idx;
    if (onehot) {
        onehot_vector_size = PSGetOneHotLayerVectorSize(previous_layer);
        if (onehot_vector_size == 0) return 0;
    }
    uint64_t input_weight_size = PSMatrixLength(layer->weights[0]);
    PSMathOpts mopts = {.acceleration = layer->network->acceleration};
    PSMatrix hidden_weights = layer->weights[1];
    PSFloat *gradient_hidden_weights = gradients->weights + input_weight_size;
    /* Cycle over previous time steps until lowest step (`lowest_t`) defined
     * by the window of BPTT_TRUNCATE. */
    for (tt = t; tt >= lowest_t; tt--) {
        int prev_t = (tt - 1), is_first_t = (tt == 0),
            update_delta = !is_first_t;
        PSMatrix delta = layer->delta;
        PSMatrix new_delta = NULL;
        int is_lowest = (tt == lowest_t);
        /* Update gradient */
        mopts.store_mode = PS_STORE_MODE_SET;
        if (!onehot) {
            PSFloat *prev_layer_outputs = PSGetStates(previous_layer, tt);
            assert(prev_layer_outputs != NULL); /* TODO: emit error */
            if (use_bias) PSSumVectors(
                delta, gradients->biases, gradients->biases, layer->size,&mopts
            );
            mopts.store_mode = PS_STORE_MODE_ADD;
            PSOuterProduct(
                delta, prev_layer_outputs, gradients->weights,
                layer->size, previous_layer->size, &mopts
            );
        } else {
            onehot_idx = (int) PSGetState(previous_layer, 0, tt);
            assert(onehot_idx < onehot_vector_size);
            for (i = 0; i < lsize; i++) {
                PSFloat dv = delta[i];
                if (use_bias) gradients->biases[i] += dv;
                w = (i * onehot_vector_size) + onehot_idx;
                gradients->weights[w] += dv;
            }
        }
        if (!is_first_t || has_initial_states) {
            if (update_delta && new_delta == NULL) {
                new_delta = PSMatrixZeros(2, 1, lsize);
                if (new_delta == NULL) return 0;
            }
            /* Update gradients' hidden weights */
            mopts.store_mode = PS_STORE_MODE_ADD;
            PSFloat *previous_states = PSGetStates(layer, prev_t);;
            PSOuterProduct(
                delta, previous_states, gradient_hidden_weights,
                layer->size, layer->size, &mopts
            );
            /* Eventually update new delta. */
            if (update_delta) {
                mopts.store_mode = PS_STORE_MODE_ADD;
                mopts.transpose = 1;
                int ok = PSDotMV(hidden_weights, delta, new_delta, &mopts);
                if (!ok) {
                    PSMatrixDelete(new_delta);
                    return 0;
                }
                mopts.transpose = 0;
            }
            if (do_truncate && update_delta && layer->derivative != NULL) {
                /* If BPTT is truncated, new_delta won't be cumulated to
                 * delta calculated from next layer in next `t` iteration,
                 * while it's used to update gradients during truncated
                 * timesteps tieration.
                 * So, derivative must be applied here. */
                int ok = PSApplyDerivative(
                    layer->derivative, new_delta, PSGetStates(layer, prev_t),
                    layer->size, &mopts
                );
                if (!ok) {
                    PSMatrixDelete(new_delta);
                    return 0;
                }
            }
        }
        if (new_delta != NULL) {
            PSMatrixDelete(delta);
            layer->delta = new_delta;
            new_delta = NULL;
        }
        /* Update previous layer delta */
        if (is_lowest && prev_layer_delta != NULL) {
            mopts.store_mode = PS_STORE_MODE_SET;
            mopts.transpose = 1;
            PSMatrix weights = layer->weights[0];
            int ok = PSDot(weights, layer->delta, previous_layer->delta,&mopts);
            if (!ok) {
                PSErr(NULL, "Layer[%d]: failed backprop (PSDot)", layer->index);
                return 0;
            }
        }
    }
    return 1;
}

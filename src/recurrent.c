/*
 * Copyright (C) 2016-2022 Fabio Nicotra <artix2 at gmail dot com>.
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
#define UNUSED(V) ((void) V)

/* Forward declaration. */

int PSResizeRecurrentHiddenStates(PSLayer *layer, uint32_t steps);
int applyLayerDroput(PSLayer *layer, int t);
int PSRecurrentBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *lgradients, ...);
int PSRecurrentFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);

/* External functions */

int isDroppedOut(PSNeuron *neuron, ...);
int checkLayerForFeedforward(PSLayer *layer);
int onehotInputsFeedforward(PSLayer *layer, PSLayer *previous, int t,
                            int do_activate);

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
                         int size, int ws)
{
    int i;
    layer->on_delete = PSDeleteRNNLayer;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->states = calloc(size, sizeof(PSFloat));
    if (layer->states == NULL) goto memerr;
    layer->weights = calloc(RNN_WEIGHT_TYPES_COUNT, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weights[0] = PSMatrixWithGaussianRandom(1, 2, size, ws);
    if (layer->weights[0] == NULL) goto memerr;
    layer->weight_types_count = 1;
    layer->weights[1] = PSMatrixWithGaussianRandom(1, 2, size, size);
    if (layer->weights[1] == NULL) goto memerr;
    layer->weight_types_count = 2;
    layer->biases = malloc(size * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->extra = calloc(size, sizeof(PSFloat));
    if (layer->extra == NULL) goto memerr;
    PSMatrix weights = layer->weights[0];
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) goto memerr;
        neuron->index = i;
        layer->biases[i] = PSGaussianRandom(0, 1);
        neuron->weights = weights + (i * ws);
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = PSTanhActivation;
    layer->derivative = PSTanhDerivative;
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

int PSRecurrentFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
#ifdef PS_DEBUG_MODE
    PSAddContextualDebug(net, layer, NULL, NULL, "feedforward", 0);
#endif
    PSLayer *previous = layer->network->layers[layer->index - 1];
    va_list args;
    va_start(args, layer);
    int times = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (times < 1) {
        PSErr(__func__, "Layer[%d]: times must be >= 1 (found %d)",
              layer->index, times);
        return 0;
    }
    if (t >= (int) layer->recurrent_states_count) {
        if (!PSResizeRecurrentHiddenStates(layer, t + 1)) {
            if (layer->network) layer->network->status = STATUS_ERROR;
            PSErr(
                NULL, "Could not resize recurrent hidden states for "
                "layer %d", layer->index
            );
            return 0;
        }
    }
    PSFloat *hidden_values = (PSFloat *) layer->extra;
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(net);
    int onehot = previous->flags & FLAG_ONEHOT;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int vector_size = 0, vector_idx = 0;
    if (onehot) {
        vector_size = PSGetOneHotLayerVectorSize(previous);
        vector_idx = (int) PSGetState(previous, 0, t);
        if (vector_size == 0) return 0;
        if (vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                  previous->index, vector_idx, vector_size - 1);
            return 0;
        }
    }
    int ignore_inputs = 0;
    int feed_previous_step = (t > 0 || layer->initial_states != NULL);
    /* If layer is the first recurrent layer of a one-to-many network, inputs
     * are fed just in the very first step. */
    if (!PSIsRecurrent(previous) && layer == first_recurrent)
        ignore_inputs = (t > 0);
    PSMathOpts dpopt = {.acceleration = net->acceleration};
    int prev_t = t - 1;
    PSFloat *inputs = NULL;
    PSFloat *prev_states = NULL;
    PSFloat *outputs = PSGetStates(layer, t);
    if (outputs == NULL) {
        PSErr(NULL,  "Layer[%d]: layer[%d] has no states");
        return 0;
    }
    PSMatrix input_weights = layer->weights[0],
             hidden_weights = layer->weights[1];
    if (ignore_inputs) goto forward_previous_step;
    /* Feed inputs by multiplying them by layer weights. If there's no previous
     * states, also eventually add biases and activate states with `activate`
     * function. */
    if (onehot) {
        if (!onehotInputsFeedforward(layer, previous, t, !feed_previous_step))
            return 0;
    } else {
        inputs = PSGetStates(previous, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d]: previous layer[%d] has NULL "
                  "outputs", layer->index, previous->index);
            return 0;
        }
        if (!feed_previous_step) {
            if (use_bias) dpopt.add_vec = layer->biases;
            dpopt.after = layer->activate;
        }
        PSDot(input_weights, inputs, outputs, &dpopt);
    }
forward_previous_step:
    if (!feed_previous_step) goto final;
    /* Feed previous states by multiplying them with hidden weights, add
     * the result to current states (`outputs`) and eventually add biases and
     * apply `activate` function. */
    prev_states = PSGetStates(layer, prev_t);
    dpopt.tmpdest = hidden_values;
    dpopt.store_mode = MATHS_STORE_MODE_ADD;
    dpopt.after = layer->activate;
    PSDot(hidden_weights, prev_states, outputs, &dpopt);
final:
    if (PSShouldApplyDropout(layer) && !applyLayerDroput(layer, t)) return 0;
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
    PSFloat *prev_layer_delta = previous_layer->delta;
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
    PSMatrix t_hidden_weights = PSMatrixTranspose(hidden_weights, 1, &mopts);
    PSMatrix tweights = PSMatrixTranspose(layer->weights[0], 1, &mopts);
    PSFloat *gradient_hidden_weights = gradients->weights + input_weight_size;
    /* Cycle over previous time steps until lowest step (`lowest_t`) defined
     * by the window of BPTT_TRUNCATE. */
    for (tt = t; tt >= lowest_t; tt--) {
        int prev_t = (tt - 1), is_first_t = (tt == 0),
            update_delta = !is_first_t;
        PSFloat *delta = layer->delta;
        PSFloat *new_delta = NULL;
        int is_lowest = (tt == lowest_t);
        /* Update gradient */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        if (!onehot) {
            PSFloat *prev_layer_outputs = PSGetStates(previous_layer, tt);
            assert(prev_layer_outputs != NULL); /* TODO: emit error */
            if (use_bias) PSSumVectors(
                delta, gradients->biases, gradients->biases, layer->size,&mopts
            );
            mopts.store_mode = MATHS_STORE_MODE_ADD;
            PSVectorProduct(
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
                new_delta = calloc(lsize, sizeof(PSFloat));
                if (new_delta == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
            }
            /* Update gradients' hidden weights */
            mopts.store_mode = MATHS_STORE_MODE_ADD;
            PSFloat *previous_states = PSGetStates(layer, prev_t);;
            PSVectorProduct(
                delta, previous_states, gradient_hidden_weights,
                layer->size, layer->size, &mopts
            );
            /* Eventually update new delta. */
            if (update_delta) {
                mopts.store_mode = MATHS_STORE_MODE_ADD;
                int ok = PSDot(t_hidden_weights, delta, new_delta, &mopts);
                if (!ok) {
                    free(new_delta);
                    return 0;
                }
            }
        }
        if (do_truncate && update_delta && layer->derivative != NULL) {
            /* If BPTT is truncated, new_delta won't be cumulated to
             * delta calculated from next layer in next `t` iteration,
             * while it's used to update gradients during truncated
             * timesteps tieration.
             * So, derivative must be applied here. */
            for (i = 0; i < lsize; i++) {
                PSFloat prev_a = PSGetState(layer, i, prev_t);
                new_delta[i] *= layer->derivative(prev_a);
            }
        }
        if (new_delta != NULL) {
            free(delta);
            layer->delta = new_delta;
            new_delta = NULL;
        }
        /* Update previous layer delta */
        if (is_lowest && prev_layer_delta != NULL) {
            /* TODO: Reimplement dropout*/
            mopts.store_mode = MATHS_STORE_MODE_NORM;
            int ok = PSDot(tweights, delta, previous_layer->delta, &mopts);
            if (!ok) {
                PSErr(NULL, "Layer[%d]: failed backprop (PSDot)", layer->index);
                return 0;
            }
        }
    }
    return 1;
}

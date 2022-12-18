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

#ifdef USE_AVX
#include "avx.h"
#endif

#include "recurrent.h"
#include "utils.h"
#include "log.h"

#define UNUSED(V) ((void) V)

/* Forward declaration. */

int PSRecurrentBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *lgradients, ...);
int PSRecurrentFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);

/* External functions */

int isDroppedOut(PSNeuron *neuron, ...);

/* Recurrent network functions */

PSRecurrentCell *PSCreateRecurrentCell(PSNeuron *neuron, int lsize) {
    PSRecurrentCell *cell = malloc(sizeof(PSRecurrentCell));
    if (cell == NULL) return NULL;
    cell->weights_size = lsize;
    if (!lsize) cell->weights = NULL;
    else cell->weights = neuron->weights + (neuron->weights_size - lsize);
    return cell;
}

/* Init Functions */

int PSInitRecurrentLayer(PSNeuralNetwork *network, PSLayer *layer,
                         int size,int ws)
{
    int i, j;
    ws += size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    layer->activations = calloc(size, sizeof(PSFloat));
    if (layer->activations == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) {
            PSErr(__func__, "Could not allocate neuron!");
            return 0;
        }
        neuron->index = i;
        neuron->weights_size = ws;
        neuron->bias = PSGaussianRandom(0, 1);
        neuron->weights = malloc(sizeof(PSFloat) * ws);
        if (neuron->weights ==  NULL) {
            PSDeleteNeuron(neuron, layer);
            PSErr(__func__, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < ws; j++) {
            neuron->weights[j] = PSGaussianRandom(0, 1);
        }
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->extra = PSCreateRecurrentCell(neuron, size);
        if (neuron->extra == NULL) return 0;
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = PSTanhActivation;
    layer->derivative = PSTanhDerivative;
    layer->feedforward = PSRecurrentFeedforward;
    layer->backprop = PSRecurrentBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
}

/* Feedforward Functions */


int PSRecurrentFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
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
    int size = layer->size;
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer *previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(net);
    int avx_disabled = PSIsAVXDisabled(net);
#ifndef USE_AVX
    UNUSED(avx_disabled);
#endif
    int onehot = previous->flags & FLAG_ONEHOT;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSHyperParameters *params = NULL;
    int vector_size = 0, vector_idx = 0;
    if (onehot) {
        params = previous->hyper_parameters;
        if (params == NULL) {
            PSErr(NULL, "Layer[%d]: prev. onehot layer params are NULL!",
                  layer->index);
            return 0;
        }
        if (params->count < 1) {
            PSErr(NULL, "Layer[%d]: prev. onehot layer params < 1!",
                  layer->index);
            return 0;
        }
        if (params->parameters == NULL) {
            PSErr(NULL, "Layer[%d]: prev. onehot layer "
                  "hyper_parameters->parameters are NULL!", layer->index);
            return 0;
        }
        vector_size = (int) (params->parameters[0]);
        vector_idx = (int) PSGetActivation(previous, 0, t);
        if (vector_size == 0 && vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                  previous->index, vector_idx, vector_size - 1);
            return 0;
        }
    }
    int i, j, w, previous_size = previous->size,
        ignore_previous_activations = 0;
    if (!PSIsRecurrent(previous) && layer == first_recurrent)
        ignore_previous_activations = (t > 0);
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
        if (cell == NULL) {
            PSErr(NULL, "Layer[%d]: neuron[%d] cell is NULL!",
                  layer->index, i);
            return 0;
        }
        PSFloat sum = 0, prev_sum = 0;
        if (ignore_previous_activations) goto forward_previous_step;
        if (onehot) sum = neuron->weights[vector_idx];
        else {
            j = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeDotProduct(
                    previous_size, previous->activations,
                    neuron->weights, sum, j, 1, t
                );
            }
#endif
            for (; j < previous_size; j++) {
                PSNeuron *prev_neuron = previous->neurons[j];
                if (prev_neuron == NULL) return 0;
                PSFloat a = PSGetActivation(previous, j, t);
                sum += (a * neuron->weights[j]);
            }
        }
forward_previous_step:
        if (t > 0 || layer->previous_activations != NULL) {
            int prev_t = t - 1;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                PSFloat *act = layer->activations;
                int avx_t = prev_t;
                if (avx_t < 0) {
                    act = layer->previous_activations;
                    avx_t = 0;
                }
                AVXIterativeDotProduct(
                    size, act, cell->weights, prev_sum, w, 1, avx_t
                );
            }
#endif
            for (; w < size; w++) {
                PSNeuron *n = layer->neurons[w];
                PSRecurrentCell *rc = PSGetRecurrentCell(n);
                if (rc == NULL) return 0;
                PSFloat weight = cell->weights[w];
                PSFloat prev_state = PSGetActivation(layer, w, prev_t);
                prev_sum += (weight * prev_state);
            }
        }
        neuron->z_value = sum + prev_sum;
        if (use_bias) neuron->z_value += neuron->bias;
        PSFloat activation = layer->activate(neuron->z_value);
        int ok = PSSetActivation(layer, activation, i, t);
        if (!ok) {
            layer->network->status = STATUS_ERROR;
            return 0;
        }
    }
    return 1;
}

/* Backpropagation Functions */

int PSRecurrentBackprop(PSLayer *layer, PSLayer *previous_layer,
                        PSGradient *lgradients, ...)
{
    va_list args;
    va_start(args, lgradients);
    int t = va_arg(args, int);
    int lowest_t = va_arg(args, int);
    va_end(args);
    int avx_disabled = PSIsAVXDisabled(layer->network);
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int lsize = layer->size, i, w, tt;
    PSFloat *prev_delta = previous_layer->delta;
    int do_truncate = (t - lowest_t) > 0;
    int has_prev_activations = (layer->previous_activations != NULL);
    /* Cycle over previous time steps until lowest step (`lowest_t`) defined
     * by the window of BPTT_TRUNCATE. */
    for (tt = t; tt >= lowest_t; tt--) {
        PSFloat *delta = layer->delta;
        PSFloat *new_delta = NULL;
        int is_lowest = (tt == lowest_t);
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
            PSGradient *gradient = &(lgradients[i]);
            PSFloat dv = delta[i];
            if (use_bias) gradient->bias += dv;
            int wsize = neuron->weights_size - cell->weights_size;
            /* Update gradients and previous layer delta */
            if (previous_layer->flags & FLAG_ONEHOT) {
                PSHyperParameters *params = previous_layer->hyper_parameters;
                if (params == NULL) {
                    fprintf(
                        stderr, "Layer %d params are NULL!\n",
                        previous_layer->index
                    );
                    if (new_delta != NULL) free(new_delta);
                    return 0;
                }
                int vector_size = (int) params->parameters[0];
                assert(vector_size > 0);
                PSFloat prev_a = PSGetActivation(previous_layer, 0, tt);
                assert(prev_a < vector_size);
                w = (int) prev_a;
                gradient->weights[w] += dv;
            } else {
                for (w = 0; w < wsize; w++) {
                    PSFloat prev_a = PSGetActivation(previous_layer, w, tt);
                    gradient->weights[w] += (dv * prev_a);
                }
            }

            int prev_t = (tt - 1), is_first_t = (tt == 0),
                update_delta = !is_first_t;
            if (!is_first_t || has_prev_activations) {
                if (update_delta && new_delta == NULL) {
                    new_delta = calloc(lsize, sizeof(PSFloat));
                    if (new_delta == NULL) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                }
                /* Update gradients and the new delta for layer with its own
                 * (previous) hidden state */
                w = 0;
#ifdef USE_AVX
                if (!avx_disabled) {
                    PSFloat *act = layer->activations;
                    int avx_t = prev_t;
                    if (avx_t < 0) {
                        act = layer->previous_activations;
                        avx_t = 0;
                    }
                    AVXIterativeMultiplyValue(
                        cell->weights_size, act, dv, gradient->weights + wsize,
                        w, 1, avx_t, AVX_STORE_MODE_ADD
                    );
                }
#endif
                for (; w < cell->weights_size; w++) {
                    PSNeuron *rn = layer->neurons[w];
                    PSRecurrentCell *rc = PSGetRecurrentCell(rn);
                    PSFloat a = PSGetActivation(layer, w, prev_t);
                    gradient->weights[wsize + w] += (dv * a);
                    if (update_delta && avx_disabled && !isDroppedOut(rn, tt)){
                        PSFloat rw = rc->weights[neuron->index];
                        new_delta[neuron->index] += (delta[rn->index] * rw);
                    }
                }
                if (update_delta && !avx_disabled) {
                    for (w = 0; w < cell->weights_size; w++) {
                        PSNeuron *rn = layer->neurons[w];
                        if (isDroppedOut(rn, tt)) continue;
                        PSRecurrentCell *rc = PSGetRecurrentCell(rn);
                        PSFloat rw = rc->weights[neuron->index];
                        new_delta[neuron->index] += (delta[rn->index] * rw);
                    }
                }
                if (do_truncate && update_delta && layer->derivative != NULL) {
                    /* If BPTT is truncated, new_delta won't be cumulated to
                     * delta calculated from next layer in next `t` iteration,
                     * while it's used to update gradients during truncated
                     * timesteps tieration.
                     * So, derivative must be applied here. */
                    PSFloat prev_a = PSGetActivation(layer, i, prev_t);
                    new_delta[neuron->index] *= layer->derivative(prev_a);
                }
            }
            if (is_lowest && prev_delta != NULL && !isDroppedOut(neuron, t)) {
                PSFloat *final_delta = (
                    new_delta != NULL ? new_delta : layer->delta
                );
                for (w = 0; w < wsize; w++) {
                    PSFloat dv = final_delta[neuron->index];
                    prev_delta[w] += (dv * neuron->weights[w]);
                }
            }
        }
        if (new_delta != NULL) {
            free(delta);
            layer->delta = new_delta;
            new_delta = NULL;
        }
    }
    return 1;
}

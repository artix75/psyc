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

#define UNUSED(V) ((void) V)

PSRecurrentCell *PSCreateRecurrentCell(PSNeuron *neuron, int lsize) {
    PSRecurrentCell *cell = malloc(sizeof(PSRecurrentCell));
    if (cell == NULL) return NULL;
    cell->states_count = 0;
    cell->states = NULL;
    cell->weights_size = lsize;
    if (!lsize) cell->weights = NULL;
    else cell->weights = neuron->weights + (neuron->weights_size - lsize);
    return cell;
}

PSFloat *PSAddRecurrentState(PSNeuralNetwork *net, PSNeuron *neuron,
                             PSFloat state, int times, int t)
{
    PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
    if (cell == NULL) {
        cell = PSCreateRecurrentCell(neuron, 0);
        neuron->extra = cell;
        if (cell == NULL) return NULL;
    }
    if (t == 0) {
        cell->states_count = times;
        if (cell->states != NULL) free(cell->states);
        cell->states = malloc(times * sizeof(PSFloat));
        if (cell->states == NULL) {
            neuron->extra = NULL;
            free(cell);
            return NULL;
        }
    }
    cell->states[t] = state;
#ifdef USE_AVX
    if (!PSIsAVXDisabled(net)) {
        PSLayer *layer = neuron->layer;
        assert(layer != NULL);
        int lsize = layer->size;
        if (t == 0 && neuron->index == 0) {
            if (layer->avx_activation_cache != NULL)
                free(layer->avx_activation_cache);
            layer->avx_activation_cache = calloc(lsize *times, sizeof(PSFloat));
        }
        if (layer->avx_activation_cache == NULL) {
            PSPrintMemoryErrorMsg();
            neuron->extra = NULL;
            if (cell->states != NULL) free(cell->states);
            free(cell);
            return NULL;
        }
        layer->avx_activation_cache[(t * lsize) + neuron->index] = state;
    }
#else
    UNUSED(net);
#endif
    return cell->states;
}

/* Init Functions */

int PSInitRecurrentLayer(PSNeuralNetwork *network, PSLayer *layer,
                         int size,int ws)
{
    int i, j;
    ws += size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    /*#ifdef USE_AVX
     layer->avx_activation_cache = calloc(size, sizeof(PSFloat));
     #endif*/
    if (layer->neurons == NULL) {
        PSErr(__func__, "Could not allocate layer neurons!");
        PSAbortLayer(network, layer);
        return 0;
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) {
            PSErr(__func__, "Could not allocate neuron!");
            PSAbortLayer(network, layer);
            return 0;
        }
        neuron->index = i;
        neuron->weights_size = ws;
        neuron->bias = PSGaussianRandom(0, 1);
        neuron->weights = malloc(sizeof(PSFloat) * ws);
        if (neuron->weights ==  NULL) {
            PSAbortLayer(network, layer);
            PSErr(__func__, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < ws; j++) {
            neuron->weights[j] = PSGaussianRandom(0, 1);
        }
        neuron->activation = 0;
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->extra = PSCreateRecurrentCell(neuron, size);
        if (neuron->extra == NULL) {
            PSAbortLayer(network, layer);
            return 0;
        }
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = PSTanhActivation;
    layer->derivative = PSTanhDerivative;
    layer->feedforward = PSRecurrentFeedforward;
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
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#endif
    int onehot = previous->flags & FLAG_ONEHOT;
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
        vector_size = (int) (params->parameters[0]);
        PSNeuron *prev_neuron = previous->neurons[0];
        vector_idx = (int) (prev_neuron->activation);
        if (vector_size == 0 && vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                  previous->index, vector_idx, vector_size - 1);
            return 0;
        }
    }
    int i, j, w, previous_size = previous->size;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
        if (cell == NULL) {
            PSErr(NULL, "Layer[%d]: neuron[%d] cell is NULL!",
                  layer->index, i);
            return 0;
        }
        PSFloat sum = 0, bias = 0;
        if (onehot) sum = neuron->weights[vector_idx];
        else {
            j = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeDotProduct(
                    previous_size, previous->avx_activation_cache,
                    neuron->weights, sum, j, 1, t
                );
            }
#endif
            for (; j < previous_size; j++) {
                PSNeuron *prev_neuron = previous->neurons[j];
                if (prev_neuron == NULL) return 0;
                PSFloat a = prev_neuron->activation;
                sum += (a * neuron->weights[j]);
            }
        }
        if (t > 0) {
            int last_t = t - 1;
            w = 0;
#ifdef USE_AVX
            if (!avx_disabled) {
                AVXIterativeDotProduct(
                    size, layer->avx_activation_cache, cell->weights,
                    bias, w, 1, last_t
                );
            }
#endif
            for (; w < size; w++) {
                PSNeuron *n = layer->neurons[w];
                PSRecurrentCell *rc = PSGetRecurrentCell(n);
                if (rc == NULL) return 0;
                PSFloat weight = cell->weights[w];
                PSFloat last_state = rc->states[last_t];
                bias += (weight *last_state);
            }
        } else {
            if (cell->states != NULL) free(cell->states);
            cell->states_count = times;
            cell->states = calloc(times, sizeof(PSFloat));
#ifdef USE_AVX
            if (!avx_disabled && neuron->index == 0) {
                if (layer->avx_activation_cache != NULL)
                    free(layer->avx_activation_cache);
                layer->avx_activation_cache = calloc(times *size,
                                                     sizeof(PSFloat));
                if (layer->avx_activation_cache == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
            }
#endif
        }
        neuron->z_value = sum + bias;
        neuron->activation = layer->activate(neuron->z_value);
        cell->states[t] = neuron->activation;
#ifdef USE_AVX
        if (!avx_disabled)
            layer->avx_activation_cache[(t * size) + i] = neuron->activation;
#endif
    }
    return 1;
}

/* Backpropagation Functions */

int PSRecurrentBackprop(PSLayer *layer, PSLayer *previousLayer, int lowest_t,
                             PSGradient *lgradients, int t)
{
    PSNeuralNetwork *net = (PSNeuralNetwork *) layer->network;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#else
    UNUSED(net);
#endif
    int lsize = layer->size, i, w, tt;
    for (tt = t; tt >= lowest_t; tt--) {
        PSFloat *delta = layer->delta;
        PSFloat *new_delta = NULL;
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
            PSGradient *gradient = &(lgradients[i]);
            PSFloat dv = delta[i];
            gradient->bias += dv;
            int wsize = neuron->weights_size - cell->weights_size;

            if (previousLayer->flags & FLAG_ONEHOT) {
                PSHyperParameters *params = previousLayer->hyper_parameters;
                if (params == NULL) {
                    fprintf(stderr, "Layer %d params are NULL!\n",
                            previousLayer->index);
                    return 0;
                }
                int vector_size = (int) params->parameters[0];
                assert(vector_size > 0);
                PSNeuron *prev_n = previousLayer->neurons[0];
                PSRecurrentCell *prev_c = PSGetRecurrentCell(prev_n);
                PSFloat prev_a = prev_c->states[tt];
                assert(prev_a < vector_size);
                w = (int) prev_a;
                gradient->weights[w] += dv;
            } else {
                for (w = 0; w < wsize; w++) {
                    PSNeuron *prev_n = previousLayer->neurons[w];
                    PSRecurrentCell *prev_c = PSGetRecurrentCell(prev_n);
                    PSFloat prev_a = prev_c->states[tt];
                    gradient->weights[w] += (dv *prev_a);
                }
            }

            if (tt > 0) {
                if (new_delta == NULL) {
                    new_delta = calloc(lsize, sizeof(PSFloat));
                    if (new_delta == NULL) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                }
                PSFloat rsum = 0.0;
                w = 0;
#ifdef USE_AVX
                if (!avx_disabled) {
                    AVXIterativeMultiplyValue(
                        cell->weights_size, layer->avx_activation_cache, dv,
                        gradient->weights + wsize, w, 1, (tt - 1),
                        AVX_STORE_MODE_ADD
                    );
                }
#endif
                for (; w < cell->weights_size; w++) {
                    PSNeuron *rn = layer->neurons[w];
                    PSRecurrentCell *rc = PSGetRecurrentCell(rn);
                    PSFloat a = rc->states[tt - 1];
                    gradient->weights[wsize + w] += (dv *a);
                }
                for (w = 0; w < cell->weights_size; w++) {
                    PSNeuron *rn = layer->neurons[w];
                    PSRecurrentCell *rc = PSGetRecurrentCell(rn);
                    PSFloat rw = rc->weights[neuron->index];
                    rsum += (delta[rn->index] * rw);
                }
                PSFloat prev_a = cell->states[tt - 1];
                new_delta[neuron->index] = rsum *layer->derivative(prev_a);
            }

        }
        if (new_delta != NULL) {
            free(delta);
            layer->delta = new_delta;
        }
    }
    return 1;
}

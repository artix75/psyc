/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
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

#include "lstm.h"
#include "utils.h"

#define CANDIDATE_IDX   0
#define INPUT_IDX       1
#define OUTPUT_IDX      2
#define FORGET_IDX      3

#define FreeLSTMDeltas() do {\
    if (delta_c != NULL) free(delta_c);\
    if (delta_i != NULL) free(delta_i);\
    if (delta_o != NULL) free(delta_o);\
    if (delta_f != NULL) free(delta_f);\
    if (lstm_delta != NULL) free(lstm_delta);\
} while(0)

static int LSTMCellFeedforward(PSLayer * layer, PSLayer * previous,
                               PSNeuron * neuron, int onehot_idx,
                               int times, int t)
{
    PSLSTMCell * cell = GetLSTMCell(neuron);
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: neuron[%d] cell is NULL!",
              layer->index, neuron->index);
        return 0;
    }
    int wsize = cell->weights_size;
    int prev_size = wsize - layer->size;
    
    double candidate = 0.0;
    double input_gate = 0.0;
    double output_gate = 0.0;
    double forget_gate = 0.0;
    
    double last_z = 0.0;
    
    if (onehot_idx >= 0) {
        candidate = cell->candidate_weights[onehot_idx];
        input_gate = cell->input_weights[onehot_idx];
        output_gate = cell->output_weights[onehot_idx];
        forget_gate = cell->forget_weights[onehot_idx];
    } else {
        int i = 0, j = 0, o = 0, f = 0;
#ifdef USE_AVX
        AVXDotProduct(previous->size, previous->avx_activation_cache,
                      cell->candidate_weights, candidate, i, 1, t);
        AVXDotProduct(previous->size, previous->avx_activation_cache,
                      cell->input_weights, input_gate, j, 1, t);
        AVXDotProduct(previous->size, previous->avx_activation_cache,
                      cell->output_weights, output_gate, o, 1, t);
        AVXDotProduct(previous->size, previous->avx_activation_cache,
                      cell->forget_weights, forget_gate, f, 1, t);
#endif
        for (; i < previous->size; i++) {
            PSNeuron * prev_neuron = previous->neurons[i];
            if (prev_neuron == NULL) return 0;
            double a = prev_neuron->activation;
            candidate += (a * cell->candidate_weights[i]);
            input_gate += (a * cell->input_weights[i]);
            output_gate += (a * cell->output_weights[i]);
            forget_gate += (a * cell->forget_weights[i]);
        }
    }
    
    if (t > 0) {
        int last_t = t - 1;
        last_z = cell->z_values[last_t];
        int i = 0, j = 0, o = 0, f = 0;
#ifdef USE_AVX
        AVXDotProduct(layer->size, layer->avx_activation_cache,
                      cell->candidate_weights + prev_size,
                      candidate, i, 1, last_t);
        AVXDotProduct(layer->size, layer->avx_activation_cache,
                      cell->input_weights + prev_size,
                      input_gate, j, 1, last_t);
        AVXDotProduct(layer->size, layer->avx_activation_cache,
                      cell->output_weights + prev_size,
                      output_gate, o, 1, last_t);
        AVXDotProduct(layer->size, layer->avx_activation_cache,
                      cell->forget_weights + prev_size,
                      forget_gate, f, 1, last_t);
#endif
        for (; i < layer->size; i++) {
            int w = i + prev_size;
            PSNeuron * n = layer->neurons[i];
            PSLSTMCell * c = GetLSTMCell(n);
            if (c == NULL) return 0;
            double last_state = c->states[last_t];
            candidate += (cell->candidate_weights[w] * last_state);
            input_gate += (cell->input_weights[w] * last_state);
            output_gate += (cell->output_weights[w] * last_state);
            forget_gate += (cell->forget_weights[w] * last_state);
        }
    } else {
        if (cell->states != NULL) free(cell->states);
        if (cell->z_values != NULL) free(cell->z_values);
        if (cell->candidates != NULL) free(cell->candidates);
        if (cell->input_gates != NULL) free(cell->input_gates);
        if (cell->output_gates != NULL) free(cell->output_gates);
        if (cell->forget_gates != NULL) free(cell->forget_gates);
        cell->states_count = times;
        cell->states = calloc(times, sizeof(double));
        cell->z_values = calloc(times, sizeof(double));
        cell->candidates = calloc(times, sizeof(double));
        cell->input_gates = calloc(times, sizeof(double));
        cell->output_gates = calloc(times, sizeof(double));
        cell->forget_gates = calloc(times, sizeof(double));
        if (cell->states == NULL) return 0;
        if (cell->z_values == NULL) return 0;
        if (cell->candidates == NULL) return 0;
        if (cell->input_gates == NULL) return 0;
        if (cell->output_gates == NULL) return 0;
        if (cell->forget_gates == NULL) return 0;
#ifdef USE_AVX
        if (neuron->index == 0) {
            if (layer->avx_activation_cache != NULL)
                free(layer->avx_activation_cache);
            layer->avx_activation_cache = calloc(times * layer->size,
                                                 sizeof(double));
            if (layer->avx_activation_cache == NULL) {
                printMemoryErrorMsg();
                return 0;
            }
        }
#endif
    }
    candidate = tanh(candidate + cell->candidate_bias);
    input_gate = sigmoid(input_gate + cell->input_bias);
    output_gate = sigmoid(output_gate + cell->output_bias);
    forget_gate = sigmoid(forget_gate + cell->forget_bias);
    
    cell->candidates[t] = candidate;
    cell->input_gates[t] = input_gate;
    cell->output_gates[t] = output_gate;
    cell->forget_gates[t] = forget_gate;
    
    neuron->z_value = candidate * input_gate + last_z * forget_gate;
    cell->z_values[t] = neuron->z_value;
    
    double activation = output_gate * neuron->z_value;
    if (layer->activate != NULL) activation = layer->activate(activation);
    neuron->activation = activation;
    cell->states[t] = activation;
    return 1;
}

PSLSTMCell * PSCreateLSTMCell(PSNeuron * neuron, int weight_size) {
    
    PSLSTMCell * cell = malloc(sizeof(PSLSTMCell));
    if (cell == NULL) return NULL;
    cell->states_count = 0;
    cell->candidates = NULL;
    cell->input_gates = NULL;
    cell->output_gates = NULL;
    cell->forget_gates = NULL;
    cell->z_values = NULL;
    cell->states = NULL;
    
    cell->candidate_bias = gaussian_random(0, 1);
    cell->input_bias = gaussian_random(0, 1);
    cell->output_bias = gaussian_random(0, 1);
    cell->forget_bias = gaussian_random(0, 1);
    
    cell->weights_size = weight_size;
    cell->candidate_weights = neuron->weights;
    cell->input_weights = neuron->weights + weight_size;
    cell->output_weights = neuron->weights + (weight_size *
                                              OUTPUT_IDX);
    cell->forget_weights = neuron->weights + (weight_size *
                                              FORGET_IDX);
    return cell;
}

void PSDeleteLSTMCell(PSLSTMCell * cell) {
    if (cell->candidates != NULL) free(cell->candidates);
    if (cell->input_gates != NULL) free(cell->input_gates);
    if (cell->output_gates != NULL) free(cell->output_gates);
    if (cell->forget_gates != NULL) free(cell->forget_gates);
    if (cell->z_values != NULL) free(cell->z_values);
    if (cell->states != NULL) free(cell->states);
    free(cell);
}

void PSUpdateLSTMBiases(PSNeuron * neuron, PSGradient * gradient, double rate) {
    double * biases = GetLSTMGradientBiases(neuron, gradient);
    PSLSTMCell *cell = GetLSTMCell(neuron);
    cell->candidate_bias -= (rate * biases[0]);
    cell->input_bias -= (rate * biases[INPUT_IDX]);
    cell->output_bias -= (rate * biases[OUTPUT_IDX]);
    cell->forget_bias -= (rate * biases[FORGET_IDX]);
}

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork * network, PSLayer * layer,
                    int size, int ws) {
    int i, j;
    ws += size;
    int tot_ws = ws * 4; //Weights for candidate, input, output and forget gates
    char * func = "PSInitLSTMLayer";
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(func, "Could not allocate layer neurons!");
        PSAbortLayer(network, layer);
        return 0;
    }
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) {
            PSErr(func, "Could not allocate neuron!");
            PSAbortLayer(network, layer);
            return 0;
        }
        neuron->index = i;
        neuron->weights_size = tot_ws;
        neuron->bias = gaussian_random(0, 1);
        neuron->weights = malloc(sizeof(double) * tot_ws);
        if (neuron->weights ==  NULL) {
            PSAbortLayer(network, layer);
            PSErr(func, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < tot_ws; j++) {
            neuron->weights[j] = gaussian_random(0, 1);
        }
        neuron->activation = 0;
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->extra = PSCreateLSTMCell(neuron, ws);
        if (neuron->extra == NULL) {
            PSAbortLayer(network, layer);
            return 0;
        }
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = tanh;
    layer->derivative = tanh_derivative;
    layer->feedforward = PSLSTMFeedforward;
    network->flags |= FLAG_RECURRENT;
    return 1;
}

/* Feedforward Functions */

int PSLSTMFeedforward(void * _net, void * _layer, ...) {
    PSNeuralNetwork * net = (PSNeuralNetwork*) _net;
    PSLayer * layer = (PSLayer*) _layer;
    char * func = "PSLSTMFeedforward";
    va_list args;
    va_start(args, _layer);
    int times = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (times < 1) {
        PSErr(func, "Layer[%d]: times must be >= 1 (found %d)",
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
    PSLayer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int onehot = previous->flags & FLAG_ONEHOT;
    PSLayerParameters * params = NULL;
    int vector_size = 0, vector_idx = -1;
    if (onehot) {
        params = previous->parameters;
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
        PSNeuron * prev_neuron = previous->neurons[0];
        vector_idx = (int) (prev_neuron->activation);
        if (vector_size == 0 && vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                  previous->index, vector_idx, vector_size - 1);
            return 0;
        }
    }
    int i, j, w, previous_size = previous->size;
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        int ok = LSTMCellFeedforward(layer, previous, neuron,
                                     vector_idx, times, t);
        if (!ok) {
            //TODO: handle
            return 0;
        }
#ifdef USE_AVX
        layer->avx_activation_cache[(t * size) + i] = neuron->activation;
#endif
    }
    return 1;
}

/* Backpropagation Functions */

double * PSLSTMBackprop(PSLayer * layer,
                        PSLayer * previousLayer,
                        int lowest_t,
                        double * last_delta,
                        double * last_lstm_delta,
                        PSGradient * lgradients,
                        int t){
    int lsize = layer->size, i, w, last_t = t - 1;
    //double * delta = malloc(sizeof(double) * lsize);
    double * delta_c = calloc(sizeof(double), lsize);
    double * delta_i = calloc(sizeof(double), lsize);
    double * delta_o = calloc(sizeof(double), lsize);
    double * delta_f = calloc(sizeof(double), lsize);
    double * lstm_delta = calloc(sizeof(double), 2 * lsize);
    if (lstm_delta == NULL || delta_c == NULL || delta_i == NULL ||
        delta_o == NULL || delta_f == NULL) {
        printMemoryErrorMsg();
        FreeLSTMDeltas();
        return NULL;
    }
    double * delta = lstm_delta;
    double * delta_z = lstm_delta + lsize;
    double * last_delta_z = NULL;
    if (last_lstm_delta != NULL) last_delta_z = last_lstm_delta + lsize;

    for (i = 0; i < lsize; i++) {
        PSNeuron * neuron = layer->neurons[i];
        PSLSTMCell * cell = GetLSTMCell(neuron);
        PSGradient * gradient = &(lgradients[i]);
        double * gradient_biases = GetLSTMGradientBiases(neuron, gradient);
        if (last_lstm_delta != NULL) last_delta[i] += last_lstm_delta[i];
        double dv = last_delta[i];
        //gradient->bias += dv;
        int cwsize = cell->weights_size;
        int rwsize = layer->size;
        int wsize = cwsize - rwsize;
        
        double z = cell->z_values[t];
        double last_z = (t > 0 ? cell->z_values[last_t] : 0.0);
        double ig = cell->input_gates[t];
        double og = cell->output_gates[t];
        double fg = cell->forget_gates[t];
        double c = cell->candidates[t];
        
        double last_dz = (last_delta_z != NULL ? last_delta_z[i] : 0.0);
        double dz = og * dv + last_dz;
        double dout = z * dv;
        double di = og * dz;
        double df = last_z * dz;
        double dc = c * dz;
        delta_z[i] = dz * cell->forget_gates[t];
        
        dout *= (1 - dout); // sigmoid_derivative
        di *= (1 - di); // sigmoid_derivative
        df *= (1 - df);
        dc = tanh_derivative(dc);
        
        delta_c[i] = dc;
        delta_i[i] = di;
        delta_o[i] = dout;
        delta_f[i] = df;
        
        gradient_biases[0] += dc;
        gradient_biases[INPUT_IDX] += di;
        gradient_biases[OUTPUT_IDX] += dout;
        gradient_biases[FORGET_IDX] += df;
        
        if (previousLayer->flags & FLAG_ONEHOT) {
            PSLayerParameters * params = previousLayer->parameters;
            if (params == NULL) {
                fprintf(stderr, "Layer %d params are NULL!\n",
                        previousLayer->index);
                FreeLSTMDeltas();
                return NULL;
            }
            int vector_size = (int) params->parameters[0];
            assert(vector_size > 0);
            PSNeuron * prev_n = previousLayer->neurons[0];
            PSLSTMCell * prev_c = GetLSTMCell(prev_n);
            double prev_a = prev_c->states[t];
            assert(prev_a < vector_size);
            w = (int) prev_a;
            gradient->weights[w] += dc;
            gradient->weights[w + cwsize] += di;
            gradient->weights[w + (cwsize * OUTPUT_IDX)] += dout;
            gradient->weights[w + (cwsize * FORGET_IDX)] += df;
        } else {
            for (w = 0; w < wsize; w++) {
                PSNeuron * prev_n = previousLayer->neurons[w];
                PSLSTMCell * prev_c = GetLSTMCell(prev_n);
                double prev_a = prev_c->states[t];
                gradient->weights[w] += (dc * prev_a);
                gradient->weights[w + cwsize] += (di * prev_a);
                gradient->weights[w + (cwsize * OUTPUT_IDX)] +=
                (dout * prev_a);
                gradient->weights[w + (cwsize * FORGET_IDX)] +=
                (df * prev_a);
            }
        }
        
        if (t > 0) {
            int w = 0, i = 0, o = 0, f = 0;
#ifdef USE_AVX
            double * rweights = gradient->weights + wsize;
            AVXMultiplyValue(layer->size,
                             layer->avx_activation_cache,
                             rweights, dc, w,
                             1, (last_t), AVX_STORE_MODE_ADD);
            AVXMultiplyValue(layer->size,
                             layer->avx_activation_cache,
                             rweights + cwsize, di, i,
                             1, (last_t), AVX_STORE_MODE_ADD);
            AVXMultiplyValue(layer->size,
                             layer->avx_activation_cache,
                             rweights + (cwsize * OUTPUT_IDX), dout, o,
                             1, (last_t), AVX_STORE_MODE_ADD);
            AVXMultiplyValue(layer->size,
                             layer->avx_activation_cache,
                             rweights + (cwsize * FORGET_IDX), df, f,
                             1, (last_t), AVX_STORE_MODE_ADD);
#endif
            for (; w < layer->size; w++) {
                PSNeuron * rn = layer->neurons[w];
                PSLSTMCell * rc = GetLSTMCell(rn);
                double a = rc->states[last_t];
                int widx = wsize + w;
                gradient->weights[widx] += (dc * a);
                gradient->weights[widx + cwsize] += (di * a);
                gradient->weights[widx + (cwsize * OUTPUT_IDX)] +=
                (dout * a);
                gradient->weights[widx + (cwsize * FORGET_IDX)] +=
                (df * a);
            }
            
        }
        
    }
    
    if (t > 0) {
        for (i = 0; i < lsize; i++) {
            PSNeuron * neuron = layer->neurons[i];
            PSLSTMCell * cell = GetLSTMCell(neuron);
            int cwsize = cell->weights_size;
            int wsize = cwsize - layer->size;
            double prev_a = cell->states[last_t];
            double rsum = 0.0, csum = 0.0, isum = 0.0, osum = 0.0, fsum = 0.0;
            for (w = 0; w < lsize; w++) {
                PSNeuron * rn = layer->neurons[w];
                PSLSTMCell * rc = GetLSTMCell(rn);
                int widx = neuron->index + wsize;
                double cw = rc->candidate_weights[widx];
                double iw = rc->input_weights[widx];
                double ow = rc->output_weights[widx];
                double fw = rc->forget_weights[widx];
                delta[neuron->index] += delta_c[rn->index] * cw;
                delta[neuron->index] += delta_i[rn->index] * iw;
                delta[neuron->index] += delta_o[rn->index] * ow;
                delta[neuron->index] += delta_f[rn->index] * fw;
            }
            if (layer->derivative != NULL)
                delta[neuron->index] *= layer->derivative(prev_a); //?
        }
    }
    
    free(delta_c);
    free(delta_i);
    free(delta_o);
    free(delta_f);
    return delta;
}

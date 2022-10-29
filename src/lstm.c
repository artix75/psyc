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

#define UNUSED(V) ((void) V)

/* Forward declarations */

int isDroppedOut(PSNeuron *neuron, ...);
PSFloat applyDropout(PSNeuron *neuron, PSFloat value);
int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, ...);
int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);

/* LSTM functions */

PSFloat applyGradientOnParameter(
    int param_type, PSTrainingOptions *options, PSFloat grad, PSFloat param,
    PSGradient *mg, PSGradient *xg, PSFloat rate, int iteration,
    int param_index
);

static int LSTMCellFeedforward(PSLayer *layer, PSLayer *previous,
                               PSNeuron *neuron, int onehot_idx,
                               int times, int t)
{
    PSNeuralNetwork *net = (PSNeuralNetwork *) layer->network;
    PSLSTMCell *cell = PSGetLSTMCell(neuron);
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: neuron[%d] cell is NULL!",
              layer->index, neuron->index);
        return 0;
    }
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#else
    UNUSED(net);
#endif
    int wsize = cell->weights_size;
    int prev_size = wsize - layer->size;

    PSFloat candidate = 0.0;
    PSFloat input_gate = 0.0;
    PSFloat output_gate = 0.0;
    PSFloat forget_gate = 0.0;

    PSFloat last_z = 0.0;

    if (onehot_idx >= 0) {
        candidate = cell->candidate_weights[onehot_idx];
        input_gate = cell->input_weights[onehot_idx];
        output_gate = cell->output_weights[onehot_idx];
        forget_gate = cell->forget_weights[onehot_idx];
    } else {
        int i = 0;
#ifdef USE_AVX
        int j = 0, o = 0, f = 0;
        if (!avx_disabled) {
            AVXIterativeDotProduct(
                previous->size, previous->avx_activation_cache,
                cell->candidate_weights, candidate, i, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->avx_activation_cache,
                cell->input_weights, input_gate, j, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->avx_activation_cache,
                cell->output_weights, output_gate, o, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->avx_activation_cache,
                cell->forget_weights, forget_gate, f, 1, t
            );
        }
#endif
        for (; i < previous->size; i++) {
            PSNeuron *prev_neuron = previous->neurons[i];
            if (prev_neuron == NULL) return 0;
            PSFloat a = prev_neuron->activation;
            candidate += (a * cell->candidate_weights[i]);
            input_gate += (a * cell->input_weights[i]);
            output_gate += (a * cell->output_weights[i]);
            forget_gate += (a * cell->forget_weights[i]);
        }
    }

    if (t > 0) {
        int last_t = t - 1;
        last_z = cell->z_values[last_t];
        int i = 0;
#ifdef USE_AVX
        int j = 0, o = 0, f = 0;
        if (!avx_disabled) {
            AVXIterativeDotProduct(layer->size, layer->avx_activation_cache,
                cell->candidate_weights + prev_size,
                candidate, i, 1, last_t);
            AVXIterativeDotProduct(layer->size, layer->avx_activation_cache,
                cell->input_weights + prev_size,
                input_gate, j, 1, last_t);
            AVXIterativeDotProduct(layer->size, layer->avx_activation_cache,
                cell->output_weights + prev_size,
                output_gate, o, 1, last_t);
            AVXIterativeDotProduct(layer->size, layer->avx_activation_cache,
                cell->forget_weights + prev_size,
                forget_gate, f, 1, last_t);
        }
#endif
        for (; i < layer->size; i++) {
            int w = i + prev_size;
            PSNeuron *n = layer->neurons[i];
            PSLSTMCell *c = PSGetLSTMCell(n);
            if (c == NULL) return 0;
            PSFloat last_state = c->states[last_t];
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
        if (cell->dropped_out != NULL) free(cell->dropped_out);
        cell->states_count = times;
        cell->states = calloc(times, sizeof(PSFloat));
        cell->z_values = calloc(times, sizeof(PSFloat));
        cell->candidates = calloc(times, sizeof(PSFloat));
        cell->input_gates = calloc(times, sizeof(PSFloat));
        cell->output_gates = calloc(times, sizeof(PSFloat));
        cell->forget_gates = calloc(times, sizeof(PSFloat));
        cell->dropped_out = calloc(times, sizeof(int));
        cell->last_step_delta = 0.0;
        if (cell->states == NULL) return 0;
        if (cell->z_values == NULL) return 0;
        if (cell->candidates == NULL) return 0;
        if (cell->input_gates == NULL) return 0;
        if (cell->output_gates == NULL) return 0;
        if (cell->forget_gates == NULL) return 0;
        if (cell->dropped_out == NULL) return 0;
#ifdef USE_AVX
        if (!avx_disabled && neuron->index == 0) {
            if (layer->avx_activation_cache != NULL)
                free(layer->avx_activation_cache);
            layer->avx_activation_cache = calloc(times *layer->size,
                                                 sizeof(PSFloat));
            if (layer->avx_activation_cache == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
        }
#endif
    }
    candidate = PSTanhActivation(candidate + cell->candidate_bias);
    input_gate = PSSigmoid(input_gate + cell->input_bias);
    output_gate = PSSigmoid(output_gate + cell->output_bias);
    forget_gate = PSSigmoid(forget_gate + cell->forget_bias);

    cell->candidates[t] = candidate;
    cell->input_gates[t] = input_gate;
    cell->output_gates[t] = output_gate;
    cell->forget_gates[t] = forget_gate;

    neuron->z_value = candidate *input_gate + last_z *forget_gate;
    cell->z_values[t] = neuron->z_value;

    PSFloat activation = neuron->z_value;
    if (layer->activate != NULL) activation = layer->activate(activation);
    activation = output_gate * activation;
    if (PSShouldApplyDropout(layer))
        activation = applyDropout(neuron, activation);
    neuron->activation = activation;
    cell->dropped_out[t] = neuron->dropped_out;
    cell->states[t] = activation;

    return 1;
}

PSLSTMCell *PSCreateLSTMCell(PSNeuron *neuron, int weight_size) {

    PSLSTMCell *cell = malloc(sizeof(PSLSTMCell));
    if (cell == NULL) return NULL;
    cell->states_count = 0;
    cell->candidates = NULL;
    cell->input_gates = NULL;
    cell->output_gates = NULL;
    cell->forget_gates = NULL;
    cell->z_values = NULL;
    cell->states = NULL;
    cell->last_step_delta = 0.0;

    cell->candidate_bias = PSGaussianRandom(0, 1);
    cell->input_bias = PSGaussianRandom(0, 1);
    cell->output_bias = PSGaussianRandom(0, 1);
    cell->forget_bias = PSGaussianRandom(0, 1);

    cell->weights_size = weight_size;
    cell->candidate_weights = neuron->weights;
    cell->input_weights = neuron->weights + weight_size;
    cell->output_weights = neuron->weights + (weight_size *
                                              OUTPUT_IDX);
    cell->forget_weights = neuron->weights + (weight_size *
                                              FORGET_IDX);
    cell->dropped_out = NULL;
    return cell;
}

void PSDeleteLSTMCell(PSLSTMCell *cell) {
    if (cell->candidates != NULL) free(cell->candidates);
    if (cell->input_gates != NULL) free(cell->input_gates);
    if (cell->output_gates != NULL) free(cell->output_gates);
    if (cell->forget_gates != NULL) free(cell->forget_gates);
    if (cell->z_values != NULL) free(cell->z_values);
    if (cell->states != NULL) free(cell->states);
    if (cell->dropped_out != NULL) free(cell->dropped_out);
    free(cell);
}

void PSUpdateLSTMBiases(PSNeuron *neuron, PSGradient *gradient,
                        PSGradient *mg, PSGradient *xg, PSFloat rate,
                        PSTrainingOptions *opts, int iteration)
{
    PSFloat *biases = PSGetLSTMGradientBiases(neuron, gradient);
    PSLSTMCell *cell = PSGetLSTMCell(neuron);
    cell->candidate_bias = applyGradientOnParameter(
        PARAM_TYPE_BIAS, opts, biases[CANDIDATE_IDX], cell->candidate_bias,
        mg, xg, rate, iteration, 0
    );
    cell->input_bias = applyGradientOnParameter(
        PARAM_TYPE_BIAS, opts, biases[INPUT_IDX], cell->input_bias,
        mg, xg, rate, iteration, 0
    );
    cell->output_bias = applyGradientOnParameter(
        PARAM_TYPE_BIAS, opts, biases[OUTPUT_IDX], cell->output_bias,
        mg, xg, rate, iteration, 0
    );
    cell->forget_bias = applyGradientOnParameter(
        PARAM_TYPE_BIAS, opts, biases[FORGET_IDX], cell->forget_bias,
        mg, xg, rate, iteration, 0
    );
}

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws) {
    int i, j;
    if (size == 0) {
        PSErr(__func__, "Cannot initialize layer with size = 0");
        return 0;
    }
    ws += size;
    int tot_ws = ws * 4; /* Weights for candidate, input, output and
                            forget gates */
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(__func__, "Could not allocate layer neurons!");
        return 0;
    }
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) {
            PSErr(__func__, "Could not allocate neuron!");
            return 0;
        }
        neuron->index = i;
        neuron->weights_size = tot_ws;
        neuron->bias = PSGaussianRandom(0, 1);
        neuron->dropped_out = 0;
        neuron->weights = malloc(sizeof(PSFloat) * tot_ws);
        if (neuron->weights ==  NULL) {
            PSDeleteNeuron(neuron, layer);
            PSErr(__func__, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < tot_ws; j++) {
            neuron->weights[j] = PSGaussianRandom(0, 1);
        }
        neuron->activation = 0;
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->extra = PSCreateLSTMCell(neuron, ws);
        if (neuron->extra == NULL) {
            return 0;
        }
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = PSTanhActivation;
    layer->derivative = PSTanhDerivative;
    layer->feedforward = PSLSTMFeedforward;
    layer->backprop = PSLSTMBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
}

/* Feedforward Functions */

int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
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
    int vector_size = 0, vector_idx = -1;
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
    int i = 0;
    for (; i < size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        int ok = LSTMCellFeedforward(layer, previous, neuron,
                                     vector_idx, times, t);
        if (!ok) {
            /* TODO: handle */
            return 0;
        }
#ifdef USE_AVX
        if (!avx_disabled)
            layer->avx_activation_cache[(t * size) + i] = neuron->activation;
#endif
    }
    return 1;
}

/* Backpropagation Functions */

int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, ...)
{
    va_list args;
    va_start(args, lgradients);
    int t = va_arg(args, int);
    va_end(args);
    PSNeuralNetwork *net = (PSNeuralNetwork *) layer->network;
#ifdef USE_AVX
    int avx_disabled = PSIsAVXDisabled(net);
#else
    UNUSED(net);
#endif
    int onehot = previousLayer->flags & FLAG_ONEHOT;
    int lsize = layer->size, i, w, last_t = t - 1;
    int previous_size = previousLayer->size;
    if (onehot) {
        PSHyperParameters *params = previousLayer->hyper_parameters;
        if (params == NULL) {
            fprintf(stderr, "Layer %d params are NULL!\n",
                    previousLayer->index);
            return 0;
        }
        previous_size = (int) params->parameters[0];
        assert(previous_size > 0);
    }
    PSFloat *delta_c = calloc(sizeof(PSFloat), lsize);
    PSFloat *delta_i = calloc(sizeof(PSFloat), lsize);
    PSFloat *delta_o = calloc(sizeof(PSFloat), lsize);
    PSFloat *delta_f = calloc(sizeof(PSFloat), lsize);

    PSFloat *delta = layer->delta;
    PSFloat *delta_z = delta + lsize;

    for (i = 0; i < lsize; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSLSTMCell *cell = PSGetLSTMCell(neuron);
        PSGradient *gradient = &(lgradients[i]);
        PSFloat *gradient_biases = PSGetLSTMGradientBiases(neuron, gradient);
        PSFloat dv = delta[i];
        int cwsize = cell->weights_size;
        int rwsize = layer->size;
        int wsize = cwsize - rwsize;

        PSFloat z = cell->z_values[t];
        PSFloat last_z = (t > 0 ? cell->z_values[last_t] : 0.0);
        PSFloat ig = cell->input_gates[t];
        PSFloat og = cell->output_gates[t];
        PSFloat fg = cell->forget_gates[t];
        PSFloat c = cell->candidates[t];

        PSFloat last_dz = delta_z[i];
        PSFloat z_multiplier = 1, zz = z;
        if (layer->activate != NULL) {
            z_multiplier = layer->activate(z);
            zz = z_multiplier;
            z_multiplier = layer->derivative(z_multiplier);
        }
        PSFloat dout = zz * dv;
        PSFloat dz = og * dv * z_multiplier + last_dz;
        PSFloat di = c * dz;
        PSFloat df = last_z * dz;
        PSFloat dc = ig * dz;
        delta_z[i] = dz * fg;

        dout *= (og * (1 - og)); /*  PSSigmoidDerivative */
        di *= (ig * (1 - ig)); /*  PSSigmoidDerivative */
        df *= (fg * (1 - fg)); /*  PSSigmoidDerivative */
        dc *= PSTanhDerivative(c);

        delta_c[i] = dc;
        delta_i[i] = di;
        delta_o[i] = dout;
        delta_f[i] = df;

        gradient_biases[0] += dc;
        gradient_biases[INPUT_IDX] += di;
        gradient_biases[OUTPUT_IDX] += dout;
        gradient_biases[FORGET_IDX] += df;

        if (onehot) {
            PSNeuron *prev_n = previousLayer->neurons[0];
            PSLSTMCell *prev_c = PSGetLSTMCell(prev_n);
            PSFloat prev_a = prev_c->states[t];
            assert(prev_a < previous_size);
            w = (int) prev_a;
            gradient->weights[w] += dc;
            gradient->weights[w + cwsize] += di;
            gradient->weights[w + (cwsize *OUTPUT_IDX)] += dout;
            gradient->weights[w + (cwsize *FORGET_IDX)] += df;
        } else {
            for (w = 0; w < wsize; w++) {
                PSNeuron *prev_n = previousLayer->neurons[w];
                PSLSTMCell *prev_c = PSGetLSTMCell(prev_n);
                PSFloat prev_a = prev_c->states[t];
                gradient->weights[w] += (dc *prev_a);
                gradient->weights[w + cwsize] += (di *prev_a);
                gradient->weights[w + (cwsize *OUTPUT_IDX)] +=
                    (dout *prev_a);
                gradient->weights[w + (cwsize *FORGET_IDX)] +=
                    (df *prev_a);
            }
        }

        if (t > 0) {
            int w = 0;
#ifdef USE_AVX
            int i = 0, o = 0, f = 0;
            if (!avx_disabled) {
                PSFloat *rweights = gradient->weights + wsize;
                AVXIterativeMultiplyValue(layer->size,
                    layer->avx_activation_cache,
                    dc, rweights, w,
                    1, (last_t), AVX_STORE_MODE_ADD);
                AVXIterativeMultiplyValue(layer->size,
                    layer->avx_activation_cache,
                    di, rweights + cwsize, i,
                    1, (last_t), AVX_STORE_MODE_ADD);
                AVXIterativeMultiplyValue(layer->size,
                    layer->avx_activation_cache,
                    dout, rweights + (cwsize *OUTPUT_IDX), o,
                    1, (last_t), AVX_STORE_MODE_ADD);
                AVXIterativeMultiplyValue(layer->size,
                    layer->avx_activation_cache,
                    df, rweights + (cwsize *FORGET_IDX), f,
                    1, (last_t), AVX_STORE_MODE_ADD);
            }
#endif
            for (; w < layer->size; w++) {
                PSNeuron *rn = layer->neurons[w];
                PSLSTMCell *rc = PSGetLSTMCell(rn);
                PSFloat a = rc->states[last_t];
                int widx = wsize + w;
                gradient->weights[widx] += (dc *a);
                gradient->weights[widx + cwsize] += (di *a);
                gradient->weights[widx + (cwsize *OUTPUT_IDX)] +=
                    (dout *a);
                gradient->weights[widx + (cwsize *FORGET_IDX)] +=
                    (df *a);
            }

        }

    }

    if (t > 0) {
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            PSLSTMCell *cell = PSGetLSTMCell(neuron);
            int cwsize = cell->weights_size;
            int wsize = cwsize - layer->size;
            /* PSFloat prev_a = cell->states[last_t]; */
            PSFloat d = 0.0;
            for (w = 0; w < lsize; w++) {
                PSNeuron *rn = layer->neurons[w];
                if (isDroppedOut(rn, t)) continue;
                PSLSTMCell *rc = PSGetLSTMCell(rn);
                int widx = neuron->index + wsize;
                PSFloat cw = rc->candidate_weights[widx];
                PSFloat iw = rc->input_weights[widx];
                PSFloat ow = rc->output_weights[widx];
                PSFloat fw = rc->forget_weights[widx];

                d += delta_c[rn->index] * cw;
                d += delta_i[rn->index] * iw;
                d += delta_o[rn->index] * ow;
                d += delta_f[rn->index] * fw;
            }
            delta[neuron->index] = d;
            cell->last_step_delta = d;
        }
    }

    if (previousLayer->delta != NULL) {
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            if (isDroppedOut(neuron, t)) continue;
            PSLSTMCell *cell = PSGetLSTMCell(neuron);
            PSFloat d = delta[neuron->index];
            int cwsize = cell->weights_size;
            int wsize = cwsize - layer->size;
            for (w = 0; w < wsize; w++) {
                PSFloat cw = cell->candidate_weights[w];
                PSFloat iw = cell->input_weights[w];
                PSFloat ow = cell->output_weights[w];
                PSFloat fw = cell->forget_weights[w];
                PSFloat prev_d = 0;
                prev_d += d * cw;
                prev_d += d * iw;
                prev_d += d * ow;
                prev_d += d * fw;
                previousLayer->delta[w] += prev_d;
            }
        }
    }

    free(delta_c);
    free(delta_i);
    free(delta_o);
    free(delta_f);
    return 1;
}

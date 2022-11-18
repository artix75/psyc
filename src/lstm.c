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
#define Z_VAL_IDX       4

#define FreeLSTMDeltas() do {\
    if (delta_c != NULL) free(delta_c);\
    if (delta_i != NULL) free(delta_i);\
    if (delta_o != NULL) free(delta_o);\
    if (delta_f != NULL) free(delta_f);\
    if (lstm_delta != NULL) free(lstm_delta);\
} while(0)


#define getCandidate(layer, i, t) (getLSTMState(layer, i, t, CANDIDATE_IDX))
#define getInputGate(layer, i, t) (getLSTMState(layer, i, t, INPUT_IDX))
#define getOutputGate(layer, i, t) (getLSTMState(layer, i, t, OUTPUT_IDX))
#define getForgetGate(layer, i, t) (getLSTMState(layer, i, t, FORGET_IDX))
#define getZValue(layer, i, t) (getLSTMState(layer, i, t, Z_VAL_IDX))

#define setCandidate(layer, i, s, t) (setLSTMState(layer,i,s,t,CANDIDATE_IDX))
#define setInputGate(layer, i, s, t) (setLSTMState(layer, i, s, t, INPUT_IDX))
#define setOutputGate(layer, i, s, t) (setLSTMState(layer, i, s, t, OUTPUT_IDX))
#define setForgetGate(layer, i, s, t) (setLSTMState(layer, i, s, t, FORGET_IDX))
#define setZValue(layer, i, s, t) (setLSTMState(layer, i, s, t, Z_VAL_IDX))

#define UNUSED(V) ((void) V)

static char *LSTMStateNames[] = {
    "Candidate", "Input gate", "Output gate", "Forget gate", "Z Val"
};

/* Forward declarations */

int isDroppedOut(PSNeuron *neuron, ...);
PSFloat applyDropout(PSNeuron *neuron, PSFloat value);
int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, ...);
int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);

PSFloat applyGradientOnParameter(
    int param_type, PSTrainingOptions *options, PSFloat grad, PSFloat param,
    PSGradient *mg, PSGradient *xg, PSFloat rate, int iteration,
    int param_index
);
PSFloat *initRecurrentStates(PSLayer *layer, uint32_t steps,
                             int retain_previous, PSFloat *current,
                             PSFloat **previous);
PSFloat *resizeRecurrentStates(PSLayer *layer, uint32_t steps,
                               PSFloat *current, PSFloat **previous);
int PSResizeRecurrentHiddenStates(PSLayer *layer, uint32_t steps);

/* LSTM functions */

void PSDeleteLSTMStates(PSLSTMStates *states) {
    if (states->z_values != NULL) free(states->z_values);
    if (states->candidates != NULL) free(states->candidates);
    if (states->input_gates != NULL) free(states->input_gates);
    if (states->output_gates != NULL) free(states->output_gates);
    if (states->forget_gates != NULL) free(states->forget_gates);
    free(states);
}

int PSInitLSTMStates(PSLayer *layer, uint32_t steps, int retain_previous) {
    if (layer->extra == NULL) {
        layer->extra = calloc(1, sizeof(PSLSTMStates));
        if (layer->extra == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
    }
    PSLSTMStates *states = PSGetLSTMStates(layer);

    PSFloat *candidates = initRecurrentStates(
        layer, steps, retain_previous, states->candidates,
        &states->previous_candidates
    );
    if (candidates == NULL) return 0;
    if (states->candidates != NULL) free(states->candidates);
    states->candidates = candidates;

    PSFloat *input_gates = initRecurrentStates(
        layer, steps, retain_previous, states->input_gates,
        &states->previous_input_gates
    );
    if (input_gates == NULL) return 0;
    if (states->input_gates != NULL) free(states->input_gates);
    states->input_gates = input_gates;

    PSFloat *output_gates = initRecurrentStates(
        layer, steps, retain_previous, states->output_gates,
        &states->previous_output_gates
    );
    if (output_gates == NULL) return 0;
    if (states->output_gates != NULL) free(states->output_gates);
    states->output_gates = output_gates;

    PSFloat *forget_gates = initRecurrentStates(
        layer, steps, retain_previous, states->forget_gates,
        &states->previous_forget_gates
    );
    if (forget_gates == NULL) return 0;
    if (states->forget_gates != NULL) free(states->forget_gates);
    states->forget_gates = forget_gates;

    PSFloat *z_values = initRecurrentStates(
        layer, steps, retain_previous, states->z_values,
        &states->previous_z_values
    );
    if (z_values == NULL) return 0;
    if (states->z_values != NULL) free(states->z_values);
    states->z_values = z_values;
    return 1;
}

int PSResizeLSTMStates(PSLayer *layer, uint32_t steps) {
    if (layer->extra == NULL) {
        layer->extra = calloc(1, sizeof(PSLSTMStates));
        if (layer->extra == NULL) return 0;
    }
    PSLSTMStates *states = PSGetLSTMStates(layer);

    PSFloat *candidates = resizeRecurrentStates(
        layer, steps, states->candidates, &states->previous_candidates
    );
    if (candidates == NULL) {
        free(states->candidates);
        states->candidates = NULL;
        states->previous_candidates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    states->candidates = candidates;

    PSFloat *input_gates = resizeRecurrentStates(
        layer, steps, states->input_gates, &states->previous_input_gates
    );
    if (input_gates == NULL) {
        free(states->input_gates);
        states->input_gates = NULL;
        states->previous_input_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    states->input_gates = input_gates;

    PSFloat *output_gates = resizeRecurrentStates(
        layer, steps, states->output_gates, &states->previous_output_gates
    );
    if (output_gates == NULL) {
        free(states->output_gates);
        states->output_gates = NULL;
        states->previous_output_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    states->output_gates = output_gates;

    PSFloat *forget_gates = resizeRecurrentStates(
        layer, steps, states->forget_gates, &states->previous_forget_gates
    );
    if (forget_gates == NULL) {
        free(states->forget_gates);
        states->forget_gates = NULL;
        states->previous_forget_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    states->forget_gates = forget_gates;

    PSFloat *z_values = resizeRecurrentStates(
        layer, steps, states->z_values, &states->previous_z_values
    );
    if (z_values == NULL) {
        free(states->z_values);
        states->z_values = NULL;
        states->previous_z_values = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    states->z_values = z_values;

    return 1;
}

static int getLSTMStatePointers(PSLSTMStates *states, int type,
                                PSFloat **state_ptr, PSFloat **previous_ptr)
{
    if (type == CANDIDATE_IDX) {
        *state_ptr = states->candidates;
        *previous_ptr = states->previous_candidates;
    } else if (type == INPUT_IDX) {
        *state_ptr = states->input_gates;
        *previous_ptr = states->previous_input_gates;
    } else if (type == OUTPUT_IDX) {
        *state_ptr = states->output_gates;
        *previous_ptr = states->previous_output_gates;
    } else if (type == FORGET_IDX) {
        *state_ptr = states->forget_gates;
        *previous_ptr = states->previous_forget_gates;
    } else if (type == Z_VAL_IDX) {
        *state_ptr = states->z_values;
        *previous_ptr = states->previous_z_values;
    } else {
        PSErr(NULL, "Invalid LSTM state type %d", type);
        *state_ptr = NULL;
        *previous_ptr = NULL;
        return 0;
    }
    return 1;
}

static PSFloat getLSTMState(PSLayer *layer, int index, int t, int type) {
    PSLSTMStates *states = PSGetLSTMStates(layer);
    if (states == NULL) return 0.0;
    PSFloat *state_ptr = NULL, *previous_ptr = NULL;
    if (!getLSTMStatePointers(states, type, &state_ptr, &previous_ptr)) {
        PSErr(NULL, "Invalid LSTM state type %d", type);
        abort();
        return 0;
    }
    if (state_ptr == NULL) {
        PSErr(
            NULL, "LSTM %s for layer %d is null",
            LSTMStateNames[type], layer->index
        );
        return 0.0;
    }
    if (t < 0) {
        if (previous_ptr == NULL) return 0.0;
        else return previous_ptr[index];
    } else {
        if (t >= (int) layer->recurrent_states_count) {
            PSErr(
                NULL, "LSTM %s %d is out-of-range: layer %d only "
                "has %d recurrent hidden states",
                "of size %d", LSTMStateNames[type], t, layer->index,
                layer->recurrent_states_count
            );
            return 0.0;
        }
        index = (t * layer->size) + index;
    }
    return state_ptr[index];
}

int setLSTMState(PSLayer *layer, int index, PSFloat state, int t, int type) {
    if (index >= layer->size) {
        PSErr(
            NULL, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0.0;
    }
    PSLSTMStates *states = PSGetLSTMStates(layer);
    if (states == NULL) {
        layer->extra = calloc(1, sizeof(PSLSTMStates));
        if (layer->extra == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        states = (PSLSTMStates *) layer->extra;
    }
    PSFloat *state_ptr = NULL, *previous_ptr = NULL;
    if (t >= (int) layer->recurrent_states_count) {
        if (!PSResizeRecurrentHiddenStates(layer, t + 1)) {
            if (layer->network) layer->network->status = STATUS_ERROR;
            PSErr(
                NULL, "Could not resize recurrent hidden states for "
                "layer %d", layer->index
            );
            return 0;
        }
    } else if (t < 0) {
        PSErr(
            NULL,
            "Invalid recurrent step %d for layer %d, neuron %d",
            layer->index, index
        );
        return 0;
    }
    if (!getLSTMStatePointers(states, type, &state_ptr, &previous_ptr)) {
        PSErr(NULL, "Invalid LSTM state type %d", type);
        abort();
        return 0;
    }
    if (state_ptr == NULL) {
        PSErr(NULL, "Layer %d: null LSTM %s!",
              layer->index,LSTMStateNames[type]);
        return 0;
    }
    index = (t * layer->size) + index;
    state_ptr[index] = state;
    return 1;
}

static int LSTMCellFeedforward(PSLayer *layer, PSLayer *previous,
                               PSNeuron *neuron, int onehot_idx,
                               int times, int t)
{
    UNUSED(times);
    PSNeuralNetwork *net = layer->network;
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
    int ignore_previous_activations = 0;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (!PSIsRecurrent(previous) && layer == PSGetFirstRecurrentLayer(net))
        ignore_previous_activations = (t > 0);

    PSFloat candidate = 0.0;
    PSFloat input_gate = 0.0;
    PSFloat output_gate = 0.0;
    PSFloat forget_gate = 0.0;

    PSFloat prev_z = 0.0;

    if (ignore_previous_activations) goto forward_previous_step;

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
                previous->size, previous->activations,
                cell->candidate_weights, candidate, i, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->activations,
                cell->input_weights, input_gate, j, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->activations,
                cell->output_weights, output_gate, o, 1, t
            );
            AVXIterativeDotProduct(
                previous->size, previous->activations,
                cell->forget_weights, forget_gate, f, 1, t
            );
        }
#endif
        for (; i < previous->size; i++) {
            PSNeuron *prev_neuron = previous->neurons[i];
            if (prev_neuron == NULL) return 0;
            PSFloat a = PSGetActivation(previous, i, t);
            candidate += (a * cell->candidate_weights[i]);
            input_gate += (a * cell->input_weights[i]);
            output_gate += (a * cell->output_weights[i]);
            forget_gate += (a * cell->forget_weights[i]);
        }
    }

forward_previous_step:
    if (t > 0 || layer->previous_activations != NULL) {
        int prev_t = t - 1;
        prev_z = getZValue(layer, neuron->index, prev_t);
        int i = 0;
#ifdef USE_AVX
        int j = 0, o = 0, f = 0;
        if (!avx_disabled) {
            PSFloat *act = layer->activations;
            int avx_t = prev_t;
            if (avx_t < 0) {
                act = layer->previous_activations;
                avx_t = 0;
            }
            AVXIterativeDotProduct(
                layer->size, act, cell->candidate_weights + prev_size,
                candidate, i, 1, avx_t
            );
            AVXIterativeDotProduct(
                layer->size, act, cell->input_weights + prev_size,
                input_gate, j, 1, avx_t
            );
            AVXIterativeDotProduct(
                layer->size, act, cell->output_weights + prev_size,
                output_gate, o, 1, avx_t
            );
            AVXIterativeDotProduct(
                layer->size, act, cell->forget_weights + prev_size,
                forget_gate, f, 1, avx_t
            );
        }
#endif
        for (; i < layer->size; i++) {
            int w = i + prev_size;
            PSNeuron *n = layer->neurons[i];
            PSLSTMCell *c = PSGetLSTMCell(n);
            if (c == NULL) return 0;
            PSFloat prev_state = PSGetActivation(layer, i, prev_t);
            candidate += (cell->candidate_weights[w] * prev_state);
            input_gate += (cell->input_weights[w] * prev_state);
            output_gate += (cell->output_weights[w] * prev_state);
            forget_gate += (cell->forget_weights[w] * prev_state);
        }
    }
    PSFloat candidate_bias = 0.0, input_bias = 0.0, output_bias = 0.0,
            forget_bias = 0.0;
    if (use_bias) {
        candidate_bias = cell->candidate_bias;
        input_bias = cell->input_bias;
        output_bias = cell->output_bias;
        forget_bias = cell->forget_bias;
    }
    candidate = PSTanhActivation(candidate + candidate_bias);
    input_gate = PSSigmoid(input_gate + input_bias);
    output_gate = PSSigmoid(output_gate + output_bias);
    forget_gate = PSSigmoid(forget_gate + forget_bias);

    if (!setCandidate(layer, neuron->index, candidate, t)) goto err;
    if (!setInputGate(layer, neuron->index, input_gate, t)) goto err;
    if (!setOutputGate(layer, neuron->index, output_gate, t)) goto err;
    if (!setForgetGate(layer, neuron->index, forget_gate, t)) goto err;

    neuron->z_value = candidate * input_gate + prev_z * forget_gate;
    if (!setZValue(layer, neuron->index, neuron->z_value, t)) goto err;

    PSFloat activation = neuron->z_value;
    if (layer->activate != NULL) activation = layer->activate(activation);
    activation = output_gate * activation;
    if (!PSSetNeuronActivation(neuron, activation, t)) goto err;
    return 1;
err:
    neuron->layer->network->status = STATUS_ERROR;
    return 0;
}

PSLSTMCell *PSCreateLSTMCell(PSNeuron *neuron, int weight_size) {

    PSLSTMCell *cell = malloc(sizeof(PSLSTMCell));
    if (cell == NULL) return NULL;
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
    return cell;
}

void PSDeleteLSTMCell(PSLSTMCell *cell) {
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
        neuron->weights = malloc(sizeof(PSFloat) * tot_ws);
        if (neuron->weights ==  NULL) {
            PSDeleteNeuron(neuron, layer);
            PSErr(__func__, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < tot_ws; j++) {
            neuron->weights[j] = PSGaussianRandom(0, 1);
        }
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
        vector_idx = (int) PSGetActivation(previous, 0, t);
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
    int lsize = layer->size, i, w, prev_t = t - 1;
    int previous_size = previousLayer->size;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (onehot) {
        PSHyperParameters *params = previousLayer->hyper_parameters;
        if (params == NULL) {
            PSErr(NULL, "Layer %d params are NULL!",
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

        PSFloat z = getZValue(layer, i, t);
        PSFloat prev_z = getZValue(layer, i, prev_t);
        PSFloat ig = getInputGate(layer, i, t);
        PSFloat og = getOutputGate(layer, i, t);
        PSFloat fg = getForgetGate(layer, i, t);
        PSFloat c = getCandidate(layer, i, t);

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
        PSFloat df = prev_z * dz;
        PSFloat dc = ig * dz;
        delta_z[i] = dz * fg;

        dout *= (og * (1 - og)); /*  PSSigmoidDerivative */
        di *= (ig * (1 - ig));   /*  PSSigmoidDerivative */
        df *= (fg * (1 - fg));   /*  PSSigmoidDerivative */
        dc *= PSTanhDerivative(c);

        delta_c[i] = dc;
        delta_i[i] = di;
        delta_o[i] = dout;
        delta_f[i] = df;

        if (use_bias) {
            gradient_biases[CANDIDATE_IDX] += dc;
            gradient_biases[INPUT_IDX] += di;
            gradient_biases[OUTPUT_IDX] += dout;
            gradient_biases[FORGET_IDX] += df;
        }

        if (onehot) {
            PSFloat prev_a = PSGetActivation(previousLayer, 0, t);
            assert(prev_a < previous_size);
            w = (int) prev_a;
            gradient->weights[w] += dc;
            gradient->weights[w + cwsize] += di;
            gradient->weights[w + (cwsize *OUTPUT_IDX)] += dout;
            gradient->weights[w + (cwsize *FORGET_IDX)] += df;
        } else {
            for (w = 0; w < wsize; w++) {
                PSFloat prev_a = PSGetActivation(previousLayer, w, t);
                gradient->weights[w] += (dc *prev_a);
                gradient->weights[w + cwsize] += (di *prev_a);
                gradient->weights[w + (cwsize *OUTPUT_IDX)] +=
                    (dout *prev_a);
                gradient->weights[w + (cwsize *FORGET_IDX)] +=
                    (df *prev_a);
            }
        }

        if (t > 0 || layer->previous_activations != NULL) {
            int w = 0;
#ifdef USE_AVX
            int i = 0, o = 0, f = 0;
            if (!avx_disabled) {
                PSFloat *rweights = gradient->weights + wsize;
                PSFloat *act = layer->activations;
                int avx_t = prev_t;
                if (avx_t < 0) {
                    act = layer->previous_activations;
                    avx_t = 0;
                }
                AVXIterativeMultiplyValue(
                    layer->size, act, dc, rweights, w, 1, avx_t,
                    AVX_STORE_MODE_ADD
                );
                AVXIterativeMultiplyValue(
                    layer->size, act, di, rweights + cwsize, i,
                    1, avx_t, AVX_STORE_MODE_ADD
                );
                AVXIterativeMultiplyValue(
                    layer->size, act, dout, rweights + (cwsize *OUTPUT_IDX),
                    o, 1, avx_t, AVX_STORE_MODE_ADD
                );
                AVXIterativeMultiplyValue(
                    layer->size, act, df, rweights + (cwsize *FORGET_IDX),
                    f, 1, avx_t, AVX_STORE_MODE_ADD
                );
            }
#endif
            for (; w < layer->size; w++) {
                PSFloat a = PSGetActivation(layer, w, prev_t);
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
            /* PSFloat prev_a = cell->states[prev_t]; */
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

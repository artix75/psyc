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

#include "lstm.h"
#include "maths.h"
#include "activation.h"
#include "utils.h"
#include "log.h"

#define CANDIDATE_IDX   PS_LSTM_CANDIDATE_IDX
#define INPUT_IDX       PS_LSTM_INPUT_IDX
#define OUTPUT_IDX      PS_LSTM_OUTPUT_IDX
#define FORGET_IDX      PS_LSTM_FORGET_IDX
#define RAW_STATE_IDX  PS_LSTM_RAWSTATE_IDX

#define LSTM_WEIGHT_TYPES_COUNT (4 * 2)

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
#define getRawState(layer, i, t) (getLSTMState(layer, i, t, RAW_STATE_IDX))

#define getCandidates(layer, t) (PSGetLSTMStates(layer, t, CANDIDATE_IDX))
#define getInputGates(layer, t) (PSGetLSTMStates(layer, t, INPUT_IDX))
#define getOutputGates(layer, t) (PSGetLSTMStates(layer, t, OUTPUT_IDX))
#define getForgetGates(layer, t) (PSGetLSTMStates(layer, t, FORGET_IDX))
#define getRawStates(layer, t) (PSGetLSTMStates(layer, t, RAW_STATE_IDX))

#define setCandidate(layer, i, s, t) (setLSTMState(layer,i,s,t,CANDIDATE_IDX))
#define setInputGate(layer, i, s, t) (setLSTMState(layer, i, s, t, INPUT_IDX))
#define setOutputGate(layer, i, s, t) (setLSTMState(layer, i, s, t, OUTPUT_IDX))
#define setForgetGate(layer, i, s, t) (setLSTMState(layer, i, s, t, FORGET_IDX))
#define setZValue(layer, i, s, t) (setLSTMState(layer, i, s, t, RAW_STATE_IDX))

#define UNUSED(V) ((void) V)

static char *LSTMStateNames[] = {
    "Candidate", "Input gate", "Output gate", "Forget gate", "Z Val"
};

/* Forward declarations */

int isDroppedOut(PSNeuron *neuron, ...);
int applyLayerDroput(PSLayer *layer, int t);
PSVecActivationFunction PSGetVectorActivationFunc(PSActivationFunction func);
int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, ...);
int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);
static int getLSTMStatePointers(PSLSTMCell *cell, int type,
                                PSFloat **state_ptr, PSFloat **previous_ptr);
int checkLayerForFeedforward(PSLayer *layer);
PSLSTMCell *PSCreateLSTMCell(PSLayer *layer);

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

PSLSTMCell *PSGetLSTMCell(PSLayer *layer) {
    if (layer == NULL) return NULL;
    PSLSTMCell *cell = (PSLSTMCell *) layer->extra;
    if (cell == NULL) cell = PSCreateLSTMCell(layer);
    return cell;
}

PSFloat *PSGetLSTMStates(PSLayer *layer, int t, int type) {
    if (layer == NULL) return NULL;
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing LSTM cell");
        return NULL;
    }
    PSFloat *state_ptr = NULL, *previous_ptr = NULL, *states = NULL;
    if (!getLSTMStatePointers(cell, type, &state_ptr, &previous_ptr))
        return NULL;
    if (t < 0) return previous_ptr;
    else {
        if (state_ptr == NULL) return NULL;
        if (t >= (int) layer->recurrent_states_count) {
            PSErr(
                NULL, "LSTM %s at step %d is out-of-range: layer %d only "
                "has %d recurrent hidden states",
                LSTMStateNames[type], t, layer->index,
                layer->recurrent_states_count
            );
            return NULL;
        }
        states = state_ptr + (layer->size * t);
    }
    return states;
}

int PSInitLSTMStates(PSLayer *layer, uint32_t steps, int retain_previous) {
    if (layer == NULL) return 0;
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing LSTM cell", layer->index);
        return 0;
    }
    PSFloat *candidates = initRecurrentStates(
        layer, steps, retain_previous, cell->candidates,
        &cell->initial_candidates
    );
    if (candidates == NULL) return 0;
    if (cell->candidates != NULL) free(cell->candidates);
    cell->candidates = candidates;

    PSFloat *input_gates = initRecurrentStates(
        layer, steps, retain_previous, cell->input_gates,
        &cell->initial_input_gates
    );
    if (input_gates == NULL) return 0;
    if (cell->input_gates != NULL) free(cell->input_gates);
    cell->input_gates = input_gates;

    PSFloat *output_gates = initRecurrentStates(
        layer, steps, retain_previous, cell->output_gates,
        &cell->initial_output_gates
    );
    if (output_gates == NULL) return 0;
    if (cell->output_gates != NULL) free(cell->output_gates);
    cell->output_gates = output_gates;

    PSFloat *forget_gates = initRecurrentStates(
        layer, steps, retain_previous, cell->forget_gates,
        &cell->initial_forget_gates
    );
    if (forget_gates == NULL) return 0;
    if (cell->forget_gates != NULL) free(cell->forget_gates);
    cell->forget_gates = forget_gates;

    PSFloat *raw_states = initRecurrentStates(
        layer, steps, retain_previous, cell->raw_states,
        &cell->initial_raw_states
    );
    if (raw_states == NULL) return 0;
    if (cell->raw_states != NULL) free(cell->raw_states);
    cell->raw_states = raw_states;
    return 1;
}

int PSResizeLSTMStates(PSLayer *layer, uint32_t steps) {
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing LSTM cell");
        return 0;
    }

    PSFloat *candidates = resizeRecurrentStates(
        layer, steps, cell->candidates, &cell->initial_candidates
    );
    if (candidates == NULL) {
        free(cell->candidates);
        cell->candidates = NULL;
        cell->initial_candidates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->candidates = candidates;

    PSFloat *input_gates = resizeRecurrentStates(
        layer, steps, cell->input_gates, &cell->initial_input_gates
    );
    if (input_gates == NULL) {
        free(cell->input_gates);
        cell->input_gates = NULL;
        cell->initial_input_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->input_gates = input_gates;

    PSFloat *output_gates = resizeRecurrentStates(
        layer, steps, cell->output_gates, &cell->initial_output_gates
    );
    if (output_gates == NULL) {
        free(cell->output_gates);
        cell->output_gates = NULL;
        cell->initial_output_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->output_gates = output_gates;

    PSFloat *forget_gates = resizeRecurrentStates(
        layer, steps, cell->forget_gates, &cell->initial_forget_gates
    );
    if (forget_gates == NULL) {
        free(cell->forget_gates);
        cell->forget_gates = NULL;
        cell->initial_forget_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->forget_gates = forget_gates;

    PSFloat *raw_states = resizeRecurrentStates(
        layer, steps, cell->raw_states, &cell->initial_raw_states
    );
    if (raw_states == NULL) {
        free(cell->raw_states);
        cell->raw_states = NULL;
        cell->initial_raw_states = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->raw_states = raw_states;

    return 1;
}

static int getLSTMStatePointers(PSLSTMCell *cell, int type,
                                PSFloat **state_ptr, PSFloat **previous_ptr)
{
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: missing LSTM cell");
        return 0;
    }
    if (type == CANDIDATE_IDX) {
        *state_ptr = cell->candidates;
        *previous_ptr = cell->initial_candidates;
    } else if (type == INPUT_IDX) {
        *state_ptr = cell->input_gates;
        *previous_ptr = cell->initial_input_gates;
    } else if (type == OUTPUT_IDX) {
        *state_ptr = cell->output_gates;
        *previous_ptr = cell->initial_output_gates;
    } else if (type == FORGET_IDX) {
        *state_ptr = cell->forget_gates;
        *previous_ptr = cell->initial_forget_gates;
    } else if (type == RAW_STATE_IDX) {
        *state_ptr = cell->raw_states;
        *previous_ptr = cell->initial_raw_states;
    } else {
        PSErr(NULL, "Invalid LSTM state type %d", type);
        *state_ptr = NULL;
        *previous_ptr = NULL;
        return 0;
    }
    return 1;
}

static PSFloat getLSTMState(PSLayer *layer, int index, int t, int type) {
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) return 0.0;
    PSFloat *state_ptr = NULL, *previous_ptr = NULL;
    if (!getLSTMStatePointers(cell, type, &state_ptr, &previous_ptr)) {
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
                "has %d recurrent hidden cell",
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
        return 0;
    }
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: missing LSTM cell");
        return 0;
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
    if (!getLSTMStatePointers(cell, type, &state_ptr, &previous_ptr)) {
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

static void initLSTMCellParams(PSLayer *layer, PSLSTMCell *cell) {
    cell->candidate_biases = layer->biases + (layer->size * CANDIDATE_IDX);
    cell->input_biases = layer->biases + (layer->size * INPUT_IDX);
    cell->output_biases = layer->biases + (layer->size * OUTPUT_IDX);
    cell->forget_biases = layer->biases + (layer->size * FORGET_IDX);
    cell->candidate_weights = layer->weights[CANDIDATE_IDX];
    cell->input_weights = layer->weights[INPUT_IDX];
    cell->output_weights = layer->weights[OUTPUT_IDX];
    cell->forget_weights = layer->weights[FORGET_IDX];
    PSMatrix *hidden_weights = layer->weights + 4;
    cell->candidate_hidden_weights = hidden_weights[CANDIDATE_IDX];
    cell->input_hidden_weights = hidden_weights[INPUT_IDX];
    cell->output_hidden_weights = hidden_weights[OUTPUT_IDX];
    cell->forget_hidden_weights = hidden_weights[FORGET_IDX];
}

PSLSTMCell *PSCreateLSTMCell(PSLayer *layer) {
    if (layer->extra != NULL) return (PSLSTMCell *) layer->extra;
    if (layer->biases == NULL) {
        PSErr(
            NULL, "Layer[%d]: cannot create LSTM cell, biases are NULL",
            layer->index
        );
        return NULL;
    }
    if (layer->weights == NULL) {
        PSErr(
            NULL, "Layer[%d]: cannot create LSTM cell, weights are NULL",
            layer->index
        );
        return NULL;
    }
    PSLSTMCell *cell = calloc(1, sizeof(PSLSTMCell));
    if (cell == NULL) goto memerr;
    cell->previous_step_delta = calloc(layer->size, sizeof(PSFloat));
    if (cell->previous_step_delta == NULL) goto memerr;
    initLSTMCellParams(layer, cell);
    int i;
    for (i = 0; i < layer->size; i++) {
        cell->candidate_biases[i] = PSGaussianRandom(0, 1);
        cell->input_biases[i] = PSGaussianRandom(0, 1);
        cell->output_biases[i] = PSGaussianRandom(0, 1);
        cell->forget_biases[i] = PSGaussianRandom(0, 1);
    }
    layer->extra = cell;
    return cell;
memerr:
    PSPrintMemoryErrorMsg();
    if (cell != NULL) PSDeleteLSTMCell(cell);
    return NULL;
}

void PSDeleteLSTMCell(PSLSTMCell *cell) {
    if (cell->previous_step_delta != NULL) free(cell->previous_step_delta);
    if (cell->raw_states != NULL) free(cell->raw_states);
    if (cell->candidates != NULL) free(cell->candidates);
    if (cell->input_gates != NULL) free(cell->input_gates);
    if (cell->output_gates != NULL) free(cell->output_gates);
    if (cell->forget_gates != NULL) free(cell->forget_gates);
    free(cell);
}

void PSDeleteLSTMLayer(PSLayer *layer) {
    if (layer->extra != NULL) {
        PSLSTMCell *cell = (PSLSTMCell *) layer->extra;
        PSDeleteLSTMCell(cell);
        layer->extra = NULL;
    }
}

int PSLSTMLayerCopy(PSLayer *layer, PSLayer *src) {
    if (layer == NULL || src == NULL) return 0;
    PSLSTMCell *cell = (PSLSTMCell *) layer->extra;
    PSLSTMCell *srccell = (PSLSTMCell *) src->extra;
    if (cell == NULL || srccell == NULL) return 0;
    size_t size = 0, states_size = 0;
    if (srccell->previous_step_delta != NULL) {
        size = layer->size * sizeof(PSFloat);
        if (cell->previous_step_delta == NULL)
            cell->previous_step_delta = malloc(size);
        if (cell->previous_step_delta == NULL) goto memerr;
        memcpy(cell->previous_step_delta, srccell->previous_step_delta, size);
    }
    if (srccell->candidates != NULL && src->recurrent_states_count > 0) {
        if (srccell->input_gates == NULL || srccell->output_gates == NULL ||
            srccell->forget_gates == NULL || srccell->raw_states == NULL)
        {
            PSErr(NULL, "Layer[%d]: incomplete states", src->index);
            return 0;
        }
        int c = src->recurrent_states_count;
        if (src->initial_states != NULL) c++;
        states_size = c * layer->size * sizeof(PSFloat);
        if (cell->candidates == NULL) cell->candidates = malloc(states_size);
        if (cell->candidates == NULL) goto memerr;
        memcpy(cell->candidates, srccell->candidates, states_size);
        if (cell->input_gates == NULL) cell->input_gates = malloc(states_size);
        if (cell->input_gates == NULL) goto memerr;
        memcpy(cell->input_gates, srccell->input_gates, states_size);
        if (cell->output_gates == NULL)
            cell->output_gates = malloc(states_size);
        if (cell->output_gates == NULL) goto memerr;
        memcpy(cell->output_gates, srccell->output_gates, states_size);
        if (cell->forget_gates == NULL)
            cell->forget_gates = malloc(states_size);
        if (cell->forget_gates == NULL) goto memerr;
        memcpy(cell->forget_gates, srccell->forget_gates, states_size);
        if (cell->raw_states == NULL) cell->raw_states = malloc(states_size);
        if (cell->raw_states == NULL) goto memerr;
        memcpy(cell->raw_states, srccell->raw_states, states_size);
    }
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws) {
    int i;
    layer->on_delete = PSDeleteLSTMLayer;
    layer->on_copy = PSLSTMLayerCopy;
    if (size == 0) {
        PSErr(__func__, "Cannot initialize layer with size = 0");
        return 0;
    }
    if (layer->biases != NULL) free(layer->biases);
    layer->biases = calloc(size, 4 * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->weights = calloc(LSTM_WEIGHT_TYPES_COUNT, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weight_types_count = 0;
    for (i = 0; i < LSTM_WEIGHT_TYPES_COUNT; i++) {
        int hidden = (i >= 4);
        int wsize = (hidden ? size : ws);
        layer->weights[i] = PSMatrixWithGaussianRandom(1, 2, size, wsize);
        if (layer->weights[i] == NULL) goto memerr;
        layer->weight_types_count++;
    }
    layer->delta = calloc(layer->size * 2, sizeof(PSFloat));
    if (layer->delta == NULL) goto memerr;
    if (!PSCreateLSTMCell(layer)) return 0;
    for (i = 0; i < size; i++) {
        PSNeuron *neuron = malloc(sizeof(PSNeuron));
        if (neuron == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        neuron->index = i;
        neuron->bias = NULL;
        neuron->weights = NULL;
        layer->neurons[i] = neuron;
        neuron->extra = NULL;
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    if (layer->activate == NULL) {
        layer->activate = PSTanhActivation;
        layer->derivative = PSTanhDerivative;
    }
    layer->feedforward = PSLSTMFeedforward;
    layer->backprop = PSLSTMBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Feedforward Functions */

int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    va_list args;
    va_start(args, layer);
    int times = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (times < 1) {
        PSErr(NULL, "Layer[%d]: times must be >= 1 (found %d)",
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
    PSLayer *previous = net->layers[layer->index - 1];
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(net);
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) return 0;
    int onehot = previous->flags & FLAG_ONEHOT;
    int vector_size = 0, vector_idx = -1, lsize = layer->size;
    int ignore_inputs = 0;
    int prev_t = t - 1, i;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSMathOpts dfopts = {.acceleration = net->acceleration};
    PSMathOpts dpopt[] = {
        /* Options for candiates */
        {.acceleration = net->acceleration, .after = PSTanhActivation},
        /* Options for input gates */
        {.acceleration = net->acceleration, .after = PSSigmoid},
        /* Options for output gates */
        {.acceleration = net->acceleration, .after = PSSigmoid},
        /* Options for forget gates */
        {.acceleration = net->acceleration, .after = PSSigmoid}
    };
    if (use_bias) {
        dpopt[CANDIDATE_IDX].add_vec = cell->candidate_biases;
        dpopt[INPUT_IDX].add_vec = cell->input_biases;
        dpopt[OUTPUT_IDX].add_vec = cell->output_biases;
        dpopt[FORGET_IDX].add_vec = cell->forget_biases;
    }
    PSMathOpts final_opts = {.acceleration = net->acceleration};
    PSFloat *prev_states = NULL, *prev_z = NULL;
    PSFloat *candidates = getCandidates(layer, t);
    PSFloat *input_gates = getInputGates(layer, t);
    PSFloat *output_gates = getOutputGates(layer, t);
    PSFloat *forget_gates = getForgetGates(layer, t);
    PSFloat *raw_states = getRawStates(layer, t);
    if (candidates == NULL || input_gates == NULL || output_gates == NULL ||
        forget_gates == NULL || raw_states == NULL)
    {
        PSErr(NULL, "Layer[%d]: missing states");
        return 0;
    }
    /* If layer is the first recurrent layer of a one-to-many network, inputs
     * are fed just in the very first step. */
    if (!PSIsRecurrent(previous) && layer == first_recurrent)
        ignore_inputs = (t > 0);
    int feed_previous_step = (t > 0 || layer->initial_states != NULL);
    PSFloat *inputs = NULL;
    PSFloat *outputs = PSGetStates(layer, t);
    if (ignore_inputs) goto forward_previous_step;
    if (onehot) {
        /* Onehot input layers only have one input corresponding to the index
         * of the activated unit. In this case, just take the value of the
         * corresponding weight, since the input should always be considered
         * as it would be 1 */
        vector_size = PSGetOneHotLayerVectorSize(previous);
        vector_idx = (int) PSGetState(previous, 0, t);
        if (vector_size == 0) return 0;
        if (vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                        previous->index, vector_idx, vector_size - 1);
            return 0;
        }
        for (i = 0; i < layer->size; i++) {
	    int offset = (i * vector_size) + vector_idx;
            candidates[i] = cell->candidate_weights[offset];
            input_gates[i] = cell->input_weights[offset];
            output_gates[i] = cell->output_weights[offset];
            forget_gates[i] = cell->forget_weights[offset];
            if (!feed_previous_step) {
                if (use_bias) {
                    candidates[i] += cell->candidate_biases[i];
                    input_gates[i] += cell->input_biases[i];
                    output_gates[i] += cell->output_biases[i];
                    forget_gates[i] += cell->forget_biases[i];
                }
                candidates[i] = PSTanh(candidates[i]);
                input_gates[i] = PSSigmoid(input_gates[i]);
                output_gates[i] = PSSigmoid(output_gates[i]);
                forget_gates[i] = PSSigmoid(forget_gates[i]);
            }
        }
    } else {
        inputs = PSGetStates(previous, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d]: previous layer[%d] has no states",
                  layer->index, previous->index);
            return 0;
        }
        PSMathOpts *c_opts = &dfopts, *ig_opts = &dfopts, *og_opts = &dfopts,
                   *fg_opts = &dfopts;
        if (!feed_previous_step) {
            /* Since previous layer state won't be added, directly use
             * `dpopt` options which will eventually add biases and
             * apply activation function to PSDot results. */
            c_opts = &dpopt[CANDIDATE_IDX];
            ig_opts = &dpopt[INPUT_IDX];
            og_opts = &dpopt[OUTPUT_IDX];
            fg_opts = &dpopt[FORGET_IDX];
        }
        PSDot(cell->candidate_weights, inputs, candidates, c_opts);
        PSDot(cell->input_weights, inputs, input_gates, ig_opts);
        PSDot(cell->output_weights, inputs, output_gates, og_opts);
        PSDot(cell->forget_weights, inputs, forget_gates, fg_opts);
    }
forward_previous_step:
    if (!feed_previous_step) goto final;
    prev_states = PSGetStates(layer, prev_t);
    if (prev_states == NULL) goto final;
    prev_z = getRawStates(layer, prev_t);
    dpopt[CANDIDATE_IDX].store_mode =
    dpopt[INPUT_IDX].store_mode =
    dpopt[OUTPUT_IDX].store_mode =
    dpopt[FORGET_IDX].store_mode = MATHS_STORE_MODE_ADD;
    PSDot(cell->candidate_hidden_weights, prev_states, candidates,
          &dpopt[CANDIDATE_IDX]);
    PSDot(cell->input_hidden_weights, prev_states, input_gates,
          &dpopt[INPUT_IDX]);
    PSDot(cell->output_hidden_weights, prev_states, output_gates,
          &dpopt[OUTPUT_IDX]);
    PSDot(cell->forget_hidden_weights, prev_states, forget_gates,
          &dpopt[FORGET_IDX]);
final:
    PSMultiplyVectors(candidates, input_gates, raw_states, lsize, &final_opts);
    if (prev_z != NULL) {
        final_opts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectors(prev_z, forget_gates, raw_states, lsize, &final_opts);
    }
    final_opts.store_mode = MATHS_STORE_MODE_NORM;
    if (layer->activate != NULL) {
        PSVecActivationFunction activate =
            PSGetVectorActivationFunc(layer->activate);
        if (activate != NULL) activate(raw_states, outputs, lsize, &final_opts);
        PSMultiplyVectors(outputs, output_gates, outputs, lsize, &final_opts);
    } else PSMultiplyVectors(raw_states,output_gates,outputs,lsize,&final_opts);
    if (PSShouldApplyDropout(layer) && !applyLayerDroput(layer, t)) return 0;
    return 1;
}

/* Backpropagation Functions */

int PSLSTMBackprop(PSLayer *layer, PSLayer *previous_layer,
                   PSGradient *lgradients, ...)
{
    PSLSTMCell *cell = (PSLSTMCell *) layer->extra;
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d] missing LSTM cell", layer->index);
        return 0;
    }
    va_list args;
    va_start(args, lgradients);
    int t = va_arg(args, int);
    va_end(args);
    PSMathOpts mopts = {.acceleration = layer->network->acceleration};
    int onehot = previous_layer->flags & FLAG_ONEHOT;
    int lsize = layer->size, i, w, prev_t = t - 1, success = 1;
    int input_size = previous_layer->size;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    if (onehot) {
        input_size = PSGetOneHotLayerVectorSize(previous_layer);
        if (input_size <= 0) return 0;
    }
    PSFloat *delta_c = calloc(lsize, sizeof(PSFloat));
    PSFloat *delta_i = calloc(lsize, sizeof(PSFloat));
    PSFloat *delta_o = calloc(lsize, sizeof(PSFloat));
    PSFloat *delta_f = calloc(lsize, sizeof(PSFloat));
    if (!delta_c || !delta_i || !delta_o || !delta_f) {
        PSPrintMemoryErrorMsg();
        success = 0;
        goto final;
    }

    PSFloat *delta = layer->delta;
    PSFloat *delta_z = delta + lsize;
    PSFloat *gradient_biases_c = lgradients->biases;
    PSFloat *gradient_biases_i = lgradients->biases + layer->size;
    PSFloat *gradient_biases_o = lgradients->biases +
                                 (layer->size * OUTPUT_IDX);
    PSFloat *gradient_biases_f = lgradients->biases +
                                 (layer->size * FORGET_IDX);
    PSFloat *grd_input_weights = lgradients->weights;
    PSFloat *grd_hidden_weights = lgradients->weights + (input_size * 4);
    PSFloat *gradient_weights_c = grd_input_weights;
    PSFloat *gradient_weights_i = grd_input_weights + input_size;
    PSFloat *gradient_weights_o = grd_input_weights + (OUTPUT_IDX * input_size);
    PSFloat *gradient_weights_f = grd_input_weights + (FORGET_IDX * input_size);
    PSFloat *gradient_hweights_c = grd_hidden_weights;
    PSFloat *gradient_hweights_i = grd_hidden_weights + layer->size;
    PSFloat *gradient_hweights_o = grd_hidden_weights +
                                   (OUTPUT_IDX * layer->size);
    PSFloat *gradient_hweights_f = grd_hidden_weights +
                                   (FORGET_IDX * layer->size);

    for (i = 0; i < lsize; i++) {
        PSFloat dv = delta[i];

        PSFloat z = getRawState(layer, i, t);
        PSFloat prev_z = getRawState(layer, i, prev_t);
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
            gradient_biases_c[i] += dc;
            gradient_biases_i[i] += di;
            gradient_biases_o[i] += dout;
            gradient_biases_f[i] += df;
        }

        if (onehot) {
            PSFloat prev_a = PSGetState(previous_layer, 0, t);
            assert(prev_a < input_size);
            w = (int) prev_a;
            gradient_weights_c[w] += dc;
            gradient_weights_i[w] += di;
            gradient_weights_o[w] += dout;
            gradient_weights_f[w] += df;
        } else {
            for (w = 0; w < input_size; w++) {
                PSFloat prev_a = PSGetState(previous_layer, w, t);
                gradient_weights_c[w] += (dc * prev_a);
                gradient_weights_i[w] += (di * prev_a);
                gradient_weights_o[w] += (dout *prev_a);
                gradient_weights_f[w] += (df *prev_a);
            }
        }

        if (t > 0 || layer->initial_states != NULL) {
            int w = 0;
            PSFloat *prev_states = PSGetStates(layer, prev_t);
            mopts.store_mode = MATHS_STORE_MODE_ADD;
            PSMultiplyVectorScalar(
                prev_states, dc, gradient_hweights_c, layer->size, &mopts
            );
            PSMultiplyVectorScalar(
                prev_states, di, gradient_hweights_i, layer->size, &mopts
            );
            PSMultiplyVectorScalar(
                prev_states, dout, gradient_hweights_o, layer->size, &mopts
            );
            PSMultiplyVectorScalar(
                prev_states, df, gradient_hweights_f, layer->size, &mopts
            );
        }

    }

    if (t > 0) {
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            int widx = neuron->index;
            /* PSFloat prev_a = cell->states[prev_t]; */
            PSFloat d = 0.0;
            for (w = 0; w < lsize; w++) {
                PSNeuron *rn = layer->neurons[w];
                if (isDroppedOut(rn, t)) continue;
                PSFloat cw = cell->candidates[widx];
                PSFloat iw = cell->input_gates[widx];
                PSFloat ow = cell->output_gates[widx];
                PSFloat fw = cell->forget_gates[widx];

                d += delta_c[rn->index] * cw;
                d += delta_i[rn->index] * iw;
                d += delta_o[rn->index] * ow;
                d += delta_f[rn->index] * fw;
            }
            delta[i] = d;
            cell->previous_step_delta[i]= d;
        }
    }

    if (previous_layer->delta != NULL) {
        for (i = 0; i < lsize; i++) {
            PSNeuron *neuron = layer->neurons[i];
            if (isDroppedOut(neuron, t)) continue;
            PSFloat d = delta[neuron->index];
            int offset = (input_size * i);
            PSFloat *weights_c = cell->candidate_weights + offset;
            PSFloat *weights_i = cell->input_weights + offset;
            PSFloat *weights_o = cell->output_weights + offset;
            PSFloat *weights_f = cell->forget_weights + offset;
            for (w = 0; w < input_size; w++) {
                PSFloat cw = weights_c[w];
                PSFloat iw = weights_i[w];
                PSFloat ow = weights_o[w];
                PSFloat fw = weights_f[w];
                PSFloat prev_d = 0;
                prev_d += d * cw;
                prev_d += d * iw;
                prev_d += d * ow;
                prev_d += d * fw;
                previous_layer->delta[w] += prev_d;
            }
        }
    }
final:
    free(delta_c);
    free(delta_i);
    free(delta_o);
    free(delta_f);
    return success;
}

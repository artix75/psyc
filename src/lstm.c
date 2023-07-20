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
#define setRawState(layer, i, s, t) \
    (setLSTMState(layer, i, s, t, RAW_STATE_IDX))

#define UNUSED(V) ((void) V)

static char *LSTMStateNames[] = {
    "Candidate", "Input gate", "Output gate", "Forget gate", "Raw State"
};

/* Forward declarations */

PSActivationFunction PSGetActivationDerivative(PSActivationFunction func);
int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, ...);
int PSLSTMForward(PSLayer *layer, ...);
static int getLSTMStatePointers(PSLSTMCell *cell, int type,
                                PSFloat **state_ptr, PSFloat **previous_ptr);
int checkLayerForForward(PSLayer *layer);
PSLSTMCell *PSCreateLSTMCell(PSLayer *layer);
void PSDeleteLSTMCell(PSLSTMCell *cell);
PSFloat applyGradientOnParameter(
    int param_type, PSTrainingOptions *options, PSFloat grad, PSFloat param,
    PSGradient *mg, PSGradient *xg, PSFloat rate, int iteration,
    int param_index
);
PSFloat *initLayerStates(PSLayer *layer, uint32_t steps,
                             int retain_previous, PSFloat *current,
                             PSFloat **previous);
PSFloat *resizeLayerStates(PSLayer *layer, uint32_t steps,
                               PSFloat *current, PSFloat **previous);
int PSResizeLayerStates(PSLayer *layer, uint32_t steps);
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
int PSBeforeSequenceForward(PSLayer *layer, int seqlen, int t);
int PSOnehotInputsForward(PSLayer *layer, int weights_index,
                              PSFloat *outputs, int t, int apply_biases,
                              int do_activate);

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
        if (t >= (int) PSStateSequenceLength(layer)) {
            PSErr(
                NULL, "LSTM %s at step %d is out-of-range: layer %d only "
                "has %d recurrent hidden states",
                LSTMStateNames[type], t, layer->index,
                PSStateSequenceLength(layer)
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
    if (steps == 0) {
        if (cell->candidates != NULL) {
            PSMatrixDelete(cell->candidates);
            cell->candidates = NULL;
        }
        if (cell->input_gates != NULL) {
            PSMatrixDelete(cell->input_gates);
            cell->input_gates = NULL;
        }
        if (cell->output_gates != NULL) {
            PSMatrixDelete(cell->output_gates);
            cell->output_gates = NULL;
        }
        if (cell->forget_gates != NULL) {
            PSMatrixDelete(cell->forget_gates);
            cell->forget_gates = NULL;
        }
        if (cell->raw_states != NULL) {
            PSMatrixDelete(cell->raw_states);
            cell->raw_states = NULL;
        }
        return 1;
    }
    PSMatrix candidates = initLayerStates(
        layer, steps, 0, cell->candidates, NULL
    );
    if (candidates == NULL) return 0;
    if (cell->candidates != NULL) PSMatrixDelete(cell->candidates);
    cell->candidates = candidates;

    PSMatrix input_gates = initLayerStates(
        layer, steps, 0, cell->input_gates, NULL
    );
    if (input_gates == NULL) return 0;
    if (cell->input_gates != NULL) PSMatrixDelete(cell->input_gates);
    cell->input_gates = input_gates;

    PSMatrix output_gates = initLayerStates(
        layer, steps, 0, cell->output_gates, NULL
    );
    if (output_gates == NULL) return 0;
    if (cell->output_gates != NULL) PSMatrixDelete(cell->output_gates);
    cell->output_gates = output_gates;

    PSMatrix forget_gates = initLayerStates(
        layer, steps, 0, cell->forget_gates, NULL
    );
    if (forget_gates == NULL) return 0;
    if (cell->forget_gates != NULL) PSMatrixDelete(cell->forget_gates);
    cell->forget_gates = forget_gates;

    PSMatrix raw_states = initLayerStates(
        layer, steps, retain_previous, cell->raw_states,
        &cell->initial_raw_states
    );
    if (raw_states == NULL) return 0;
    if (cell->raw_states != NULL) PSMatrixDelete(cell->raw_states);
    cell->raw_states = raw_states;
    return 1;
}

int PSResizeLSTMStates(PSLayer *layer, uint32_t steps, uint32_t prev_steps) {
    UNUSED(prev_steps);
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing LSTM cell");
        return 0;
    }

    PSMatrix candidates = resizeLayerStates(
        layer, steps, cell->candidates, NULL
    );
    if (candidates == NULL) {
        PSMatrixDelete(cell->candidates);
        cell->candidates = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
        return 0;
    }
    cell->candidates = candidates;

    PSMatrix input_gates = resizeLayerStates(
        layer, steps, cell->input_gates, NULL
    );
    if (input_gates == NULL) {
        PSMatrixDelete(cell->input_gates);
        cell->input_gates = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
        return 0;
    }
    cell->input_gates = input_gates;

    PSMatrix output_gates = resizeLayerStates(
        layer, steps, cell->output_gates, NULL
    );
    if (output_gates == NULL) {
        PSMatrixDelete(cell->output_gates);
        cell->output_gates = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
        return 0;
    }
    cell->output_gates = output_gates;

    PSMatrix forget_gates = resizeLayerStates(
        layer, steps, cell->forget_gates, NULL
    );
    if (forget_gates == NULL) {
        PSMatrixDelete(cell->forget_gates);
        cell->forget_gates = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
        return 0;
    }
    cell->forget_gates = forget_gates;
    PSMatrix raw_states = resizeLayerStates(
        layer, steps, cell->raw_states, &cell->initial_raw_states
    );
    if (raw_states == NULL) {
        PSMatrixDelete(cell->raw_states);
        cell->raw_states = NULL;
        cell->initial_raw_states = NULL;
        PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
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
        *previous_ptr = NULL;
    } else if (type == INPUT_IDX) {
        *state_ptr = cell->input_gates;
        *previous_ptr = NULL;
    } else if (type == OUTPUT_IDX) {
        *state_ptr = cell->output_gates;
        *previous_ptr = NULL;
    } else if (type == FORGET_IDX) {
        *state_ptr = cell->forget_gates;
        *previous_ptr = NULL;
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

PSFloat getLSTMState(PSLayer *layer, int index, int t, int type) {
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
        if (t >= (int) PSStateSequenceLength(layer)) {
            PSErr(
                NULL, "LSTM %s %d is out-of-range: layer %d only "
                "has %d recurrent hidden cell",
                "of size %d", LSTMStateNames[type], t, layer->index,
                PSStateSequenceLength(layer)
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
    if (t >= (int) PSStateSequenceLength(layer)) {
        if (!PSResizeLayerStates(layer, t + 1)) {
            if (layer->network)
                PSSetNetworkStatus(layer->network, STATUS_ERROR, NULL);
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
    initLSTMCellParams(layer, cell);
    layer->extra = cell;
    return cell;
memerr:
    PSPrintMemoryErrorMsg();
    if (cell != NULL) PSDeleteLSTMCell(cell);
    return NULL;
}

void PSDeleteLSTMCell(PSLSTMCell *cell) {
    if (cell->raw_states != NULL) PSMatrixDelete(cell->raw_states);
    if (cell->candidates != NULL) PSMatrixDelete(cell->candidates);
    if (cell->input_gates != NULL) PSMatrixDelete(cell->input_gates);
    if (cell->output_gates != NULL) PSMatrixDelete(cell->output_gates);
    if (cell->forget_gates != NULL) PSMatrixDelete(cell->forget_gates);
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
    int c = PSStateSequenceLength(src);
    if (srccell->candidates != NULL && c > 0) {
        if (srccell->input_gates == NULL || srccell->output_gates == NULL ||
            srccell->forget_gates == NULL || srccell->raw_states == NULL)
        {
            PSErr(NULL, "Layer[%d]: incomplete states", src->index);
            return 0;
        }
        if (src->initial_states != NULL) c++;
        if (cell->candidates != NULL) PSMatrixDelete(cell->candidates);
        cell->candidates = PSMatrixDup(srccell->candidates);
        if (cell->candidates == NULL) goto memerr;
        if (cell->input_gates != NULL) PSMatrixDelete(cell->input_gates);
        cell->input_gates = PSMatrixDup(srccell->input_gates);
        if (cell->input_gates == NULL) goto memerr;
        if (cell->output_gates != NULL) PSMatrixDelete(cell->output_gates);
        cell->output_gates = PSMatrixDup(srccell->output_gates);
        if (cell->output_gates == NULL) goto memerr;
        if (cell->forget_gates != NULL) PSMatrixDelete(cell->forget_gates);
        cell->forget_gates = PSMatrixDup(srccell->output_gates);
        if (cell->forget_gates == NULL) goto memerr;
        if (cell->raw_states != NULL) PSMatrixDelete(cell->raw_states);
        cell->raw_states = PSMatrixDup(srccell->raw_states);
        if (cell->raw_states == NULL) goto memerr;
    }
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws, PSLayerDef *ldef)
{
    int i, bias_count = size * 4;
    layer->on_delete = PSDeleteLSTMLayer;
    layer->on_copy = PSLSTMLayerCopy;
    layer->on_states_init = PSInitLSTMStates;
    layer->on_states_resize = PSResizeLSTMStates;
    if (size == 0) {
        PSErr(__func__, "Cannot initialize layer with size = 0");
        return 0;
    }
    if (layer->biases != NULL) free(layer->biases);
    layer->biases = calloc(size, bias_count * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->weights = calloc(LSTM_WEIGHT_TYPES_COUNT, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weight_types_count = 0;
    for (i = 0; i < LSTM_WEIGHT_TYPES_COUNT; i++) {
        int hidden = (i >= 4);
        int wsize = (hidden ? size : ws);
        layer->weights[i] = PSInitWeights(layer, size, wsize, ldef, 1, 0);
        if (layer->weights[i] == NULL) goto memerr;
        layer->weight_types_count++;
    }
    int bias_init_mode = (ldef != NULL ? ldef->bias_init_mode : INIT_MODE_AUTO);
    if (bias_init_mode == INIT_MODE_ZERO)
        memset(layer->biases, 0, bias_count * sizeof(PSFloat));
    else {
        for (i = 0; i < bias_count; i++)
            layer->biases[i] = PSInitParam(PARAM_TYPE_BIAS, ldef, 1, 0);
    }
    layer->delta = PSMatrixZeros(2, 1, layer->size * 2);
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
    layer->forward = PSLSTMForward;
    layer->backprop = PSLSTMBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Forward Functions */

int PSLSTMForward(PSLayer *layer, ...) {
    if (!checkLayerForForward(layer)) return 0;
    PSNeuralNetwork *net = layer->network;
    va_list args;
    va_start(args, layer);
    int steps = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (!PSBeforeSequenceForward(layer, steps, t)) return 0;
    PSLayer *previous = net->layers[layer->index - 1];
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(net);
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) return 0;
    int onehot = previous->flags & FLAG_ONEHOT;
    int lsize = layer->size;
    int ignore_inputs = 0;
    int prev_t = t - 1;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    int success = 1;
    PSMathOpts mopts = {.acceleration = net->acceleration};
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
        success = (
            PSOnehotInputsForward(layer, CANDIDATE_IDX, cell->candidates,
                t, 0, 0) &&
            PSOnehotInputsForward(layer, INPUT_IDX, cell->input_gates,
                t, 0, 0) &&
            PSOnehotInputsForward(layer, OUTPUT_IDX, cell->output_gates,
                t, 0, 0) &&
            PSOnehotInputsForward(layer, FORGET_IDX, cell->forget_gates,
                t, 0, 0)
        );
        if (!success) goto final;
    } else {
        inputs = PSGetStates(previous, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d]: previous layer[%d] has no states",
                  layer->index, previous->index);
            return 0;
        }
        mopts.argtype[1] = 'V';
        success = PSDot(cell->candidate_weights, inputs, candidates, &mopts) &&
                  PSDot(cell->input_weights, inputs, input_gates, &mopts) &&
                  PSDot(cell->output_weights, inputs, output_gates, &mopts) &&
                  PSDot(cell->forget_weights, inputs, forget_gates, &mopts);
        if (!success) goto final;
    }
forward_previous_step:
    if (!feed_previous_step) goto final;
    prev_states = PSGetStates(layer, prev_t);
    if (prev_states == NULL) goto final;
    prev_z = getRawStates(layer, prev_t);
    mopts.store_mode = PS_STORE_MODE_ADD;
    mopts.transpose = 0;
    mopts.argtype[0] = 'M';
    mopts.argtype[1] = 'V';
    success = (
        PSDot(cell->candidate_hidden_weights, prev_states,candidates,&mopts) &&
        PSDot(cell->input_hidden_weights, prev_states, input_gates, &mopts) &&
        PSDot(cell->output_hidden_weights, prev_states, output_gates,&mopts) &&
        PSDot(cell->forget_hidden_weights, prev_states, forget_gates, &mopts)
    );
    mopts.argtype[0] = '\0';
    mopts.argtype[1] = '\0';
    if (!success) goto final;
final:
    mopts.store_mode = PS_STORE_MODE_SET;
    /* Add biases */
    if (use_bias) {
        PSSumVectors(candidates, cell->candidate_biases, candidates,
            layer->size, &mopts);
        PSSumVectors(input_gates, cell->input_biases, input_gates,
            layer->size, &mopts);
        PSSumVectors(output_gates, cell->output_biases, output_gates,
            layer->size, &mopts);
        PSSumVectors(forget_gates, cell->forget_biases, forget_gates,
            layer->size, &mopts);
    }
    /* Activate candidates, input, output and forget gates. */
    PSTanhActivation(candidates, NULL, layer->size, &mopts);
    PSSigmoid(input_gates, NULL, layer->size, &mopts);
    PSSigmoid(output_gates, NULL, layer->size, &mopts);
    PSSigmoid(forget_gates, NULL, layer->size, &mopts);
    /* Produce raw states and outputs. */
    PSMultiplyVectors(candidates, input_gates, raw_states, lsize, &mopts);
    if (prev_z != NULL) {
        mopts.store_mode = PS_STORE_MODE_ADD;
        PSMultiplyVectors(prev_z, forget_gates, raw_states, lsize, &mopts);
    }
    mopts.store_mode = PS_STORE_MODE_SET;
    if (layer->activate != NULL) {
        layer->activate(raw_states, outputs, lsize, &mopts);
        PSMultiplyVectors(outputs, output_gates, outputs, lsize, &mopts);
    } else PSMultiplyVectors(raw_states,output_gates,outputs,lsize, &mopts);
    return success;
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
    int lsize = layer->size, prev_t = t - 1, success = 1;
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
    PSFloat *actv_z = calloc(lsize, sizeof(PSFloat));
    PSFloat *dz = calloc(lsize, sizeof(PSFloat));
    if (!delta_c || !delta_i || !delta_o || !delta_f || !actv_z || !dz) {
        PSPrintMemoryErrorMsg();
        success = 0;
        goto final;
    }

    int input_weight_size = input_size * lsize;
    int hidden_weight_size = lsize * lsize;
    PSFloat *delta = layer->delta;
    PSFloat *delta_z = delta + lsize;
    PSFloat *gradient_biases_c = lgradients->biases;
    PSFloat *gradient_biases_i = lgradients->biases + layer->size;
    PSFloat *gradient_biases_o = lgradients->biases +
                                 (layer->size * OUTPUT_IDX);
    PSFloat *gradient_biases_f = lgradients->biases +
                                 (layer->size * FORGET_IDX);
    PSFloat *grd_input_weights = lgradients->weights;
    PSFloat *grd_hidden_weights = lgradients->weights +
                                  (input_weight_size * 4);
    PSFloat *gradient_weights_c = grd_input_weights;
    PSFloat *gradient_weights_i = grd_input_weights + input_weight_size;
    PSFloat *gradient_weights_o = grd_input_weights +
                                  (OUTPUT_IDX * input_weight_size);
    PSFloat *gradient_weights_f = grd_input_weights +
                                  (FORGET_IDX * input_weight_size);
    PSFloat *gradient_hweights_c = grd_hidden_weights;
    PSFloat *gradient_hweights_i = grd_hidden_weights + hidden_weight_size;
    PSFloat *gradient_hweights_o = grd_hidden_weights +
                                   (OUTPUT_IDX * hidden_weight_size);
    PSFloat *gradient_hweights_f = grd_hidden_weights +
                                   (FORGET_IDX * hidden_weight_size);
    PSFloat *candidates = getCandidates(layer, t);
    PSFloat *input_gates = getInputGates(layer, t);
    PSFloat *output_gates = getOutputGates(layer, t);
    PSFloat *forget_gates = getForgetGates(layer, t);

    /* Create deltas for candiate, input, output, forget */
    PSFloat *raw_states = getRawStates(layer, t);
    PSFloat *prev_raw_states = getRawStates(layer, t - 1);
    if (layer->activate != NULL) {
        if (layer->derivative == NULL) {
            layer->derivative = PSGetActivationDerivative(layer->activate);
            if (layer->derivative == NULL) {
                PSErr(NULL, "Layer[%d]: no activation derivative");
                success = 0;
                goto final;
            }
        }
        mopts.store_mode = PS_STORE_MODE_SET;
        layer->activate(raw_states, actv_z, layer->size, &mopts);
        layer->derivative(actv_z, dz, layer->size, &mopts);
    } else {
        PSVectorCopy(actv_z, raw_states, layer->size);
        PSVectorCopy(dz, raw_states, layer->size);
    }
    PSMultiplyVectors(dz, output_gates, dz, layer->size, &mopts);
    PSMultiplyVectors(dz, layer->delta, dz, layer->size, &mopts);
    PSSumVectors(dz, delta_z, dz, layer->size, &mopts);

    /* Update output gates delta (delta_o) */
    PSSigmoidDerivative(output_gates, delta_o, layer->size, &mopts);
    PSMultiplyVectors(delta_o, actv_z, delta_o, layer->size, &mopts);
    PSMultiplyVectors(delta_o, delta, delta_o, layer->size, &mopts);

    /* Update input gates delta (delta_i) */
    PSSigmoidDerivative(input_gates, delta_i, layer->size, &mopts);
    PSMultiplyVectors(delta_i, candidates, delta_i, layer->size, &mopts);
    PSMultiplyVectors(delta_i, dz, delta_i, layer->size, &mopts);

    /* Update candidates delta (delta_c) */
    PSTanhDerivative(candidates, delta_c, layer->size, &mopts);
    PSMultiplyVectors(delta_c, input_gates, delta_c, layer->size, &mopts);
    PSMultiplyVectors(delta_c, dz, delta_c, layer->size, &mopts);

    if (prev_raw_states != NULL) {
        /* Update forget gates delta (delta_f) */
        PSSigmoidDerivative(forget_gates, delta_f, layer->size, &mopts);
        PSMultiplyVectors(delta_f, prev_raw_states, delta_f,layer->size,&mopts);
        PSMultiplyVectors(delta_f, dz, delta_f, layer->size, &mopts);
    }

    /* Update delta_z */
    PSMultiplyVectors(forget_gates, dz, delta_z, layer->size, &mopts);

    /* Update gradient biases */
    if (use_bias) {
        PSSumVectors(delta_c, gradient_biases_c, gradient_biases_c,
                     layer->size, &mopts);
        PSSumVectors(delta_i, gradient_biases_i, gradient_biases_i,
                     layer->size, &mopts);
        PSSumVectors(delta_o, gradient_biases_o, gradient_biases_o,
                     layer->size, &mopts);
        PSSumVectors(delta_f, gradient_biases_f, gradient_biases_f,
                     layer->size, &mopts);
    }

    /* Update gradient input weights */

    if (onehot) {
        PSFloat input = PSGetState(previous_layer, 0, t);
        if (input >= input_size) {
            PSErr(NULL, "Onehot Layer[%d] state %g is out-of-range %d",
                  previous_layer->index, input, input_size);
            success = 0;
            goto final;
        }
        int onehot_idx = (int) input;
        for (int i = 0; i < layer->size; i++) {
            int widx = (i * input_size) + onehot_idx;
            gradient_weights_c[widx] += delta_c[i];
            gradient_weights_i[widx] += delta_i[i];
            gradient_weights_o[widx] += delta_o[i];
            gradient_weights_f[widx] += delta_f[i];
        }
    } else {
        PSFloat *inputs = PSGetStates(previous_layer, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs at step %d",
                  previous_layer->index, t);
            success = 0;
            goto final;
        }
        mopts.store_mode = PS_STORE_MODE_ADD;
        PSOuterProduct(delta_c, inputs, gradient_weights_c,
                       layer->size, previous_layer->size, &mopts);
        PSOuterProduct(delta_i, inputs, gradient_weights_i,
                       layer->size, previous_layer->size, &mopts);
        PSOuterProduct(delta_o, inputs, gradient_weights_o,
                       layer->size, previous_layer->size, &mopts);
        PSOuterProduct(delta_f, inputs, gradient_weights_f,
                       layer->size, previous_layer->size, &mopts);
    }

    if (t > 0 || layer->initial_states != NULL) {
        PSFloat *prev_states = PSGetStates(layer, prev_t);
        /* Update gradient hidden weights */
        mopts.store_mode = PS_STORE_MODE_ADD;
        PSOuterProduct(delta_c, prev_states, gradient_hweights_c,
                       layer->size, layer->size, &mopts);
        PSOuterProduct(delta_i, prev_states, gradient_hweights_i,
                       layer->size, layer->size, &mopts);
        PSOuterProduct(delta_o, prev_states, gradient_hweights_o,
                       layer->size, layer->size, &mopts);
        PSOuterProduct(delta_f, prev_states, gradient_hweights_f,
                       layer->size, layer->size, &mopts);
        /* Update delta */
        if (t > 0) {
            mopts.store_mode = PS_STORE_MODE_SET;
            mopts.transpose = 1;
            success = PSDotMV(cell->candidate_hidden_weights, delta_c,
                              layer->delta, &mopts);
            if (!success) goto final;
            mopts.store_mode = PS_STORE_MODE_ADD;
            success = (
                PSDotMV(cell->input_hidden_weights, delta_i, layer->delta,
                    &mopts) &&
                PSDotMV(cell->output_hidden_weights, delta_o, layer->delta,
                    &mopts) &&
                PSDotMV(cell->forget_hidden_weights, delta_f, layer->delta,
                    &mopts)
            );
            mopts.transpose = 0;
            mopts.argtype[1] = '\0';
            if (!success) goto final;
        }
    }

    if (previous_layer->delta != NULL) {
        /* Update previous layer delta */
        PSMatrix prev_delta = previous_layer->delta;
        mopts.store_mode = PS_STORE_MODE_SET;
        mopts.transpose = 1;
        success = PSDotMV(cell->candidate_weights, delta_c, prev_delta, &mopts);
        if (!success) goto final;
        mopts.store_mode = PS_STORE_MODE_ADD;
        success = (
            PSDotMV(cell->input_weights, delta_i, prev_delta, &mopts) &&
            PSDotMV(cell->output_weights, delta_o, prev_delta, &mopts) &&
            PSDotMV(cell->forget_weights, delta_f, prev_delta, &mopts)
        );
        if (!success) goto final;
    }
final:
    free(delta_c);
    free(delta_i);
    free(delta_o);
    free(delta_f);
    free(actv_z);
    free(dz);
    return success;
}

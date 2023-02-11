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

#include "gru.h"
#include "maths.h"
#include "activation.h"
#include "utils.h"
#include "log.h"

#define CANDIDATE_IDX   PS_GRU_CANDIDATE_IDX
#define UPDATE_IDX      PS_GRU_UPDATE_IDX
#define RESET_IDX       PS_GRU_RESET_IDX

#define GRU_WEIGHT_TYPES_COUNT (3 * 2)

#define getCandidate(layer, i, t) (getGRUState(layer, i, t, CANDIDATE_IDX))
#define getUpdateGate(layer, i, t) (getGRUState(layer, i, t, UPDATE_IDX))
#define getResetGate(layer, i, t) (getGRUState(layer, i, t, RESET_IDX))

#define getCandidates(layer, t) (PSGetGRUStates(layer, t, CANDIDATE_IDX))
#define getUpdateGates(layer, t) (PSGetGRUStates(layer, t, UPDATE_IDX))
#define getResetGates(layer, t) (PSGetGRUStates(layer, t, RESET_IDX))

#define setCandidate(layer, i, s, t) (setGRUState(layer,i,s,t,CANDIDATE_IDX))
#define setUpdateGate(layer, i, s, t) (setGRUState(layer, i, s, t, UPDATE_IDX))
#define setResetGate(layer, i, s, t) (setGRUState(layer, i, s, t, RESET_IDX))

#define UNUSED(V) ((void) V)

static char *GRUStateNames[] = {
    "Candidate", "Update gate", "Reset gate"
};

/* Forward declarations */

int checkLayerForFeedforward(PSLayer *layer);
PSGRUCell *PSCreateGRUCell(PSLayer *layer);
void PSDeleteGRUCell(PSGRUCell *cell);
static int getGRUStatePointers(PSGRUCell *cell, int type,
                               PSFloat **state_ptr, PSFloat **previous_ptr);
PSFloat *initRecurrentStates(PSLayer *layer, uint32_t steps,
                             int retain_previous, PSFloat *current,
                             PSFloat **previous);
PSFloat *resizeRecurrentStates(PSLayer *layer, uint32_t steps,
                               PSFloat *current, PSFloat **previous);
int PSResizeRecurrentHiddenStates(PSLayer *layer, uint32_t steps);
PSVecActivationFunction PSGetVectorActivationFunc(PSActivationFunction func);
PSActivationFunction PSGetActivationDerivative(PSActivationFunction func);
PSMatrix PSInitWeights(PSLayer *layer, int rows, int columns,
                       PSLayerDef *ldef, PSFloat range, PSFloat scale);
PSFloat PSInitParam(int param_type, PSLayerDef *ldef, PSFloat range,
                    PSFloat scale);
int PSGRUFeedforward(PSLayer *layer, ...);
int PSGRUBackprop(PSLayer *layer, PSLayer *previous_layer,
                   PSGradient *lgradients, ...);

/* GRU functions */

PSGRUCell *PSGetGRUCell(PSLayer *layer) {
    if (layer == NULL) return NULL;
    PSGRUCell *cell = (PSGRUCell *) layer->extra;
    if (cell == NULL) cell = PSCreateGRUCell(layer);
    return cell;
}

PSFloat *PSGetGRUStates(PSLayer *layer, int t, int type) {
    if (layer == NULL) return NULL;
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing GRU cell");
        return NULL;
    }
    PSFloat *state_ptr = NULL, *previous_ptr = NULL, *states = NULL;
    if (!getGRUStatePointers(cell, type, &state_ptr, &previous_ptr))
        return NULL;
    if (t < 0) return previous_ptr;
    else {
        if (state_ptr == NULL) return NULL;
        if (t >= (int) layer->recurrent_states_count) {
            PSErr(
                NULL, "GRU %s at step %d is out-of-range: layer %d only "
                "has %d recurrent hidden states",
                GRUStateNames[type], t, layer->index,
                layer->recurrent_states_count
            );
            return NULL;
        }
        states = state_ptr + (layer->size * t);
    }
    return states;
}

int PSInitGRUStates(PSLayer *layer, uint32_t steps, int retain_previous) {
    if (layer == NULL) return 0;
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing GRU cell", layer->index);
        return 0;
    }
    PSFloat *candidates = initRecurrentStates(
        layer, steps, retain_previous, cell->candidates,
        &cell->initial_candidates
    );
    if (candidates == NULL) return 0;
    if (cell->candidates != NULL) free(cell->candidates);
    cell->candidates = candidates;

    PSFloat *update_gates = initRecurrentStates(
        layer, steps, retain_previous, cell->update_gates,
        &cell->initial_update_gates
    );
    if (update_gates == NULL) return 0;
    if (cell->update_gates != NULL) free(cell->update_gates);
    cell->update_gates = update_gates;

    PSFloat *reset_gates = initRecurrentStates(
        layer, steps, retain_previous, cell->reset_gates,
        &cell->initial_reset_gates
    );
    if (reset_gates == NULL) return 0;
    if (cell->reset_gates != NULL) free(cell->reset_gates);
    cell->reset_gates = reset_gates;

    return 1;
}

int PSResizeGRUStates(PSLayer *layer, uint32_t steps) {
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) {
        PSErr(__func__, "Layer[%d]: missing GRU cell");
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

    PSFloat *update_gates = resizeRecurrentStates(
        layer, steps, cell->update_gates, &cell->initial_update_gates
    );
    if (update_gates == NULL) {
        free(cell->update_gates);
        cell->update_gates = NULL;
        cell->initial_update_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->update_gates = update_gates;

    PSFloat *reset_gates = resizeRecurrentStates(
        layer, steps, cell->reset_gates, &cell->initial_reset_gates
    );
    if (reset_gates == NULL) {
        free(cell->reset_gates);
        cell->reset_gates = NULL;
        cell->initial_reset_gates = NULL;
        layer->network->status = STATUS_ERROR;
        return 0;
    }
    cell->reset_gates = reset_gates;

    return 1;
}

static int getGRUStatePointers(PSGRUCell *cell, int type,
                               PSFloat **state_ptr, PSFloat **previous_ptr)
{
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: missing GRU cell");
        return 0;
    }
    if (type == CANDIDATE_IDX) {
        *state_ptr = cell->candidates;
        *previous_ptr = cell->initial_candidates;
    } else if (type == UPDATE_IDX) {
        *state_ptr = cell->update_gates;
        *previous_ptr = cell->initial_update_gates;
    } else if (type == RESET_IDX) {
        *state_ptr = cell->reset_gates;
        *previous_ptr = cell->initial_reset_gates;
    } else {
        PSErr(NULL, "Invalid GRU state type %d", type);
        *state_ptr = NULL;
        *previous_ptr = NULL;
        return 0;
    }
    return 1;
}

PSFloat getGRUState(PSLayer *layer, int index, int t, int type) {
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) return 0.0;
    PSFloat *state_ptr = NULL, *previous_ptr = NULL;
    if (!getGRUStatePointers(cell, type, &state_ptr, &previous_ptr)) {
        PSErr(NULL, "Invalid GRU state type %d", type);
        abort();
        return 0;
    }
    if (state_ptr == NULL) {
        PSErr(
            NULL, "GRU %s for layer %d is null",
            GRUStateNames[type], layer->index
        );
        return 0.0;
    }
    if (t < 0) {
        if (previous_ptr == NULL) return 0.0;
        else return previous_ptr[index];
    } else {
        if (t >= (int) layer->recurrent_states_count) {
            PSErr(
                NULL, "GRU %s %d is out-of-range: layer %d only "
                "has %d recurrent hidden cell",
                "of size %d", GRUStateNames[type], t, layer->index,
                layer->recurrent_states_count
            );
            return 0.0;
        }
        index = (t * layer->size) + index;
    }
    return state_ptr[index];
}

int setGRUState(PSLayer *layer, int index, PSFloat state, int t, int type) {
    if (index >= layer->size) {
        PSErr(
            NULL, "Neuron index %d is out-of-range for layer %d "
            "of size %d", index, layer->index, layer->size
        );
        return 0;
    }
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d]: missing GRU cell");
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
    if (!getGRUStatePointers(cell, type, &state_ptr, &previous_ptr)) {
        PSErr(NULL, "Invalid GRU state type %d", type);
        abort();
        return 0;
    }
    if (state_ptr == NULL) {
        PSErr(NULL, "Layer %d: null GRU %s!",
              layer->index,GRUStateNames[type]);
        return 0;
    }
    index = (t * layer->size) + index;
    state_ptr[index] = state;
    return 1;
}

static void initGRUCellParams(PSLayer *layer, PSGRUCell *cell) {
    cell->candidate_biases = layer->biases + (layer->size * CANDIDATE_IDX);
    cell->update_biases = layer->biases + (layer->size * UPDATE_IDX);
    cell->reset_biases = layer->biases + (layer->size * RESET_IDX);
    cell->candidate_weights = layer->weights[CANDIDATE_IDX];
    cell->update_weights = layer->weights[UPDATE_IDX];
    cell->reset_weights = layer->weights[RESET_IDX];
    PSMatrix *hidden_weights = layer->weights + 3;
    cell->candidate_hidden_weights = hidden_weights[CANDIDATE_IDX];
    cell->update_hidden_weights = hidden_weights[UPDATE_IDX];
    cell->reset_hidden_weights = hidden_weights[RESET_IDX];
}

PSGRUCell *PSCreateGRUCell(PSLayer *layer) {
    if (layer->extra != NULL) return (PSGRUCell *) layer->extra;
    if (layer->biases == NULL) {
        PSErr(
            NULL, "Layer[%d]: cannot create GRU cell, biases are NULL",
            layer->index
        );
        return NULL;
    }
    if (layer->weights == NULL) {
        PSErr(
            NULL, "Layer[%d]: cannot create GRU cell, weights are NULL",
            layer->index
        );
        return NULL;
    }
    PSGRUCell *cell = calloc(1, sizeof(PSGRUCell));
    if (cell == NULL) goto memerr;
    initGRUCellParams(layer, cell);
    layer->extra = cell;
    return cell;
memerr:
    PSPrintMemoryErrorMsg();
    if (cell != NULL) PSDeleteGRUCell(cell);
    return NULL;
}

void PSDeleteGRUCell(PSGRUCell *cell) {
    if (cell->candidates != NULL) free(cell->candidates);
    if (cell->update_gates != NULL) free(cell->update_gates);
    if (cell->reset_gates != NULL) free(cell->reset_gates);
    free(cell);
}

void PSDeleteGRULayer(PSLayer *layer) {
    if (layer->extra != NULL) {
        PSGRUCell *cell = (PSGRUCell *) layer->extra;
        PSDeleteGRUCell(cell);
        layer->extra = NULL;
    }
}

int PSGRULayerCopy(PSLayer *layer, PSLayer *src) {
    if (layer == NULL || src == NULL) return 0;
    PSGRUCell *cell = (PSGRUCell *) layer->extra;
    PSGRUCell *srccell = (PSGRUCell *) src->extra;
    if (cell == NULL || srccell == NULL) return 0;
    size_t states_size = 0;
    if (srccell->candidates != NULL && src->recurrent_states_count > 0) {
        if (srccell->update_gates == NULL || srccell->reset_gates == NULL) {
            PSErr(NULL, "Layer[%d]: incomplete states", src->index);
            return 0;
        }
        int c = src->recurrent_states_count;
        if (src->initial_states != NULL) c++;
        states_size = c * layer->size * sizeof(PSFloat);
        if (cell->candidates == NULL) cell->candidates = malloc(states_size);
        if (cell->candidates == NULL) goto memerr;
        memcpy(cell->candidates, srccell->candidates, states_size);
        if (cell->update_gates == NULL)
            cell->update_gates = malloc(states_size);
        if (cell->update_gates == NULL) goto memerr;
        memcpy(cell->update_gates, srccell->update_gates, states_size);
        if (cell->reset_gates == NULL)
            cell->reset_gates = malloc(states_size);
        if (cell->reset_gates == NULL) goto memerr;
        memcpy(cell->reset_gates, srccell->reset_gates, states_size);
    }
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Init Functions */

int PSInitGRULayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws, PSLayerDef *ldef)
{
    int i, bias_count = size * 3;
    layer->on_delete = PSDeleteGRULayer;
    layer->on_copy = PSGRULayerCopy;
    layer->on_recurrent_states_init = PSInitGRUStates;
    layer->on_recurrent_states_resize = PSResizeGRUStates;
    if (size == 0) {
        PSErr(__func__, "Cannot initialize layer with size = 0");
        return 0;
    }
    if (layer->biases != NULL) free(layer->biases);
    layer->biases = calloc(size, bias_count * sizeof(PSFloat));
    if (layer->biases == NULL) goto memerr;
    layer->neurons = calloc(size, sizeof(PSNeuron*));
    if (layer->neurons == NULL) goto memerr;
    layer->weights = calloc(GRU_WEIGHT_TYPES_COUNT, sizeof(PSMatrix));
    if (layer->weights == NULL) goto memerr;
    layer->weight_types_count = 0;
    for (i = 0; i < GRU_WEIGHT_TYPES_COUNT; i++) {
        int hidden = (i >= 3);
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
    layer->delta = calloc(layer->size, sizeof(PSFloat));
    if (layer->delta == NULL) goto memerr;
    if (!PSCreateGRUCell(layer)) return 0;
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
    layer->feedforward = PSGRUFeedforward;
    layer->backprop = PSGRUBackprop;
    network->flags |= FLAG_RECURRENT;
    return 1;
memerr:
    PSPrintMemoryErrorMsg();
    return 0;
}

/* Feedforward Functions */

int PSGRUFeedforward(PSLayer *layer, ...) {
    if (!checkLayerForFeedforward(layer)) return 0;
    PSNeuralNetwork *net = layer->network;
    int success = 1;
    PSFloat *cache = NULL;
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
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) return 0;
    int onehot = previous->flags & FLAG_ONEHOT;
    int vector_size = 0, vector_idx = -1;
    int ignore_inputs = 0;
    int prev_t = t - 1, i;
    int use_bias = !(layer->flags & FLAG_NO_BIAS);
    PSMathOpts dfopts = {.acceleration = net->acceleration};
    PSMathOpts dpopt[] = {
        /* Options for candiates */
        {.acceleration = net->acceleration, .after = PSTanhActivation},
        /* Options for update gates */
        {.acceleration = net->acceleration, .after = PSSigmoid},
        /* Options for reset gates */
        {.acceleration = net->acceleration, .after = PSSigmoid},
    };
    if (use_bias) {
        dpopt[CANDIDATE_IDX].add_vec = cell->candidate_biases;
        dpopt[UPDATE_IDX].add_vec = cell->update_biases;
        dpopt[RESET_IDX].add_vec = cell->reset_biases;
    }
    PSMathOpts final_opts = {.acceleration = net->acceleration};
    PSFloat *prev_states = NULL;
    PSFloat *candidates = getCandidates(layer, t);
    PSFloat *update_gates = getUpdateGates(layer, t);
    PSFloat *reset_gates = getResetGates(layer, t);
    if (candidates == NULL || update_gates == NULL || reset_gates == NULL) {
        PSErr(NULL, "Layer[%d]: missing states");
        success = 0;
        goto final;
    }
    cache = malloc(layer->size * sizeof(PSFloat));
    if (cache == NULL) {
        PSPrintMemoryErrorMsg();
        success = 0;
        goto final;
    }
    /* If layer is the first recurrent layer of a one-to-many network, updates
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
        success = vector_size > 0;
        if (!success) goto final;
        if (vector_idx >= vector_size) {
            PSErr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                        previous->index, vector_idx, vector_size - 1);
            success = 0;
            goto final;
        }
        for (i = 0; i < layer->size; i++) {
	    int offset = (i * vector_size) + vector_idx;
            candidates[i] = cell->candidate_weights[offset];
            update_gates[i] = cell->update_weights[offset];
            reset_gates[i] = cell->reset_weights[offset];
            if (!feed_previous_step) {
                if (use_bias) {
                    candidates[i] += cell->candidate_biases[i];
                    update_gates[i] += cell->update_biases[i];
                    reset_gates[i] += cell->reset_biases[i];
                }
                candidates[i] = PSTanh(candidates[i]);
                update_gates[i] = PSSigmoid(update_gates[i]);
                reset_gates[i] = PSSigmoid(reset_gates[i]);
            }
        }
    } else {
        inputs = PSGetStates(previous, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d]: previous layer[%d] has no states",
                  layer->index, previous->index);
            success = 0;
            goto final;
        }
        PSMathOpts *c_opts = &dfopts, *ug_opts = &dfopts, *rg_opts = &dfopts;
        if (!feed_previous_step) {
            /* Since previous layer state won't be added, directly use
             * `dpopt` options which will eventually add biases and
             * apply activation function to PSDot results. */
            c_opts = &dpopt[CANDIDATE_IDX];
            ug_opts = &dpopt[UPDATE_IDX];
            rg_opts = &dpopt[RESET_IDX];
        }
        success = PSDot(cell->candidate_weights, inputs, candidates, c_opts) &&
                  PSDot(cell->update_weights, inputs, update_gates, ug_opts) &&
                  PSDot(cell->reset_weights, inputs, reset_gates, rg_opts);
        if (!success) goto final;
    }
forward_previous_step:
    if (!feed_previous_step) goto make_outputs;
    prev_states = PSGetStates(layer, prev_t);
    if (prev_states == NULL) goto make_outputs;
    dpopt[CANDIDATE_IDX].store_mode =
    dpopt[UPDATE_IDX].store_mode =
    dpopt[RESET_IDX].store_mode = MATHS_STORE_MODE_ADD;
    success = (
        PSDot(cell->update_hidden_weights, prev_states, update_gates,
                    &dpopt[UPDATE_IDX]) &&
        PSDot(cell->reset_hidden_weights, prev_states, reset_gates,
                    &dpopt[RESET_IDX])
    );
    if (!success) goto final;
    PSMultiplyVectors(reset_gates, prev_states, cache, layer->size,
                      &final_opts);
    success = PSDot(cell->candidate_hidden_weights, cache, candidates,
                    &dpopt[CANDIDATE_IDX]);
    if (!success) goto final;
    /* Produce outputs */
    PSMultiplyVectors(update_gates, prev_states, outputs, layer->size,
                      &final_opts);
make_outputs:
    PSSubtractScalarVector(1, update_gates, cache, layer->size, &final_opts);
    if (feed_previous_step) final_opts.store_mode = MATHS_STORE_MODE_ADD;
    PSMultiplyVectors(candidates, cache, outputs, layer->size,
                      &final_opts);
final:
    free(cache);
    return success;
}

/* Backpropagation Functions */

int PSGRUBackprop(PSLayer *layer, PSLayer *previous_layer,
                   PSGradient *lgradients, ...)
{
    PSGRUCell *cell = (PSGRUCell *) layer->extra;
    if (cell == NULL) {
        PSErr(NULL, "Layer[%d] missing GRU cell", layer->index);
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
    PSFloat *delta_u = calloc(lsize, sizeof(PSFloat));
    PSFloat *delta_r = calloc(lsize, sizeof(PSFloat));
    PSFloat *delta_rc = calloc(lsize, sizeof(PSFloat));
    PSFloat *cache = calloc(lsize, sizeof(PSFloat));
    if (!delta_c || !delta_u || !delta_r || !cache || !delta_rc) {
        PSPrintMemoryErrorMsg();
        success = 0;
        goto final;
    }
    PSMatrix tr_update_hidden_weights = PSMatrixTranspose(
        cell->update_hidden_weights, 0, &mopts
    );
    PSMatrix tr_candidate_hidden_weights = PSMatrixTranspose(
        cell->candidate_hidden_weights, 0, &mopts
    );
    PSMatrix tr_reset_hidden_weights = PSMatrixTranspose(
        cell->reset_hidden_weights, 0, &mopts
    );
    if (!tr_update_hidden_weights || !tr_candidate_hidden_weights ||
        !tr_reset_hidden_weights)
    {
        PSErr(NULL, "Layer[%d]: failed to transpose weights", layer->index);
        success = 0;
        goto final;
    }

    int input_weight_size = input_size * lsize;
    int hidden_weight_size = lsize * lsize;
    PSFloat *delta = layer->delta;
    PSFloat *gradient_biases_c = lgradients->biases;
    PSFloat *gradient_biases_u = lgradients->biases + layer->size;
    PSFloat *gradient_biases_r = lgradients->biases +
                                 (layer->size * RESET_IDX);
    PSFloat *grd_input_weights = lgradients->weights;
    PSFloat *grd_hidden_weights = lgradients->weights +
                                  (input_weight_size * 3);
    PSFloat *gradient_weights_c = grd_input_weights;
    PSFloat *gradient_weights_u = grd_input_weights + input_weight_size;
    PSFloat *gradient_weights_r = grd_input_weights +
                                  (RESET_IDX * input_weight_size);
    PSFloat *gradient_hweights_c = grd_hidden_weights;
    PSFloat *gradient_hweights_u = grd_hidden_weights + hidden_weight_size;
    PSFloat *gradient_hweights_r = grd_hidden_weights +
                                   (RESET_IDX * hidden_weight_size);
    PSFloat *candidates = getCandidates(layer, t);
    PSFloat *update_gates = getUpdateGates(layer, t);
    PSFloat *reset_gates = getResetGates(layer, t);
    int has_prev_states = (t > 0 || layer->initial_states != NULL);
    PSFloat *prev_states = NULL;
    if (has_prev_states) prev_states = PSGetStates(layer, prev_t);
    if (prev_states == NULL) has_prev_states = 0;
    mopts.store_mode = MATHS_STORE_MODE_NORM;

    /* Build delta_c */
    PSSubtractScalarVector(1, update_gates, delta_c, lsize, &mopts);
    PSMultiplyVectors(delta_c, delta, delta_c, lsize, &mopts);
    PSTanhDerivativeV(candidates, cache, lsize, &mopts);
    PSMultiplyVectors(delta_c, cache, delta_c, lsize, &mopts);

    /* Build delta_rc and delta_r */
    success = PSDot(tr_candidate_hidden_weights, delta_c, delta_rc, &mopts);
    if (!success) goto final;
    if (has_prev_states) {
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectors(delta_rc, prev_states, delta_r, lsize, &mopts);
        PSSigmoidDerivativeV(reset_gates, cache, lsize, &mopts);
        PSMultiplyVectors(delta_r, cache, delta_r, lsize, &mopts);
    }

    /* Build delta_u */
    if (has_prev_states)
        PSSubtractVectors(prev_states, candidates, delta_u, lsize, &mopts);
    else PSSubtractScalarVector(0, candidates, delta_u, lsize, &mopts);
    PSMultiplyVectors(delta_u, delta, delta_u, lsize, &mopts);
    PSSigmoidDerivativeV(update_gates, cache, lsize, &mopts);
    PSMultiplyVectors(delta_u, cache, delta_u, lsize, &mopts);

    /* Update gradient biases */
    if (use_bias) {
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSSumVectors(gradient_biases_c,delta_c,gradient_biases_c,lsize,&mopts);
        PSSumVectors(gradient_biases_u,delta_u,gradient_biases_u,lsize,&mopts);
        if (has_prev_states) {
            PSSumVectors(gradient_biases_r, delta_r, gradient_biases_r,
                         lsize, &mopts);
        }
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
            gradient_weights_u[widx] += delta_u[i];
            if (!has_prev_states) continue;
            gradient_weights_r[widx] += delta_r[i];
        }
    } else {
        PSFloat *inputs = PSGetStates(previous_layer, t);
        if (inputs == NULL) {
            PSErr(NULL, "Layer[%d] has no outputs at step %d",
                  previous_layer->index, t);
            success = 0;
            goto final;
        }
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSVectorProduct(delta_c, inputs, gradient_weights_c, lsize,
                        previous_layer->size, &mopts);
        PSVectorProduct(delta_u, inputs, gradient_weights_u, lsize,
                        previous_layer->size, &mopts);
        if (has_prev_states) {
            PSVectorProduct(delta_r, inputs, gradient_weights_r, lsize,
                            previous_layer->size, &mopts);
        }
    }
    if (has_prev_states) {
        /* Update gradient hidden weights. */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectors(reset_gates, prev_states, cache, lsize, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSVectorProduct(delta_c, cache, gradient_hweights_c, lsize,
                        lsize, &mopts);
        PSVectorProduct(delta_r, prev_states, gradient_hweights_r, lsize,
                        lsize, &mopts);
        PSVectorProduct(delta_u, prev_states, gradient_hweights_u, lsize,
                        lsize, &mopts);
        /* Update Delta */
        if (t > 0) {
            mopts.store_mode = MATHS_STORE_MODE_NORM;
            PSMultiplyVectors(delta, update_gates, delta, lsize, &mopts);
            mopts.store_mode = MATHS_STORE_MODE_ADD;
            success = PSDot(tr_update_hidden_weights, delta_u, delta, &mopts);
            if (!success) goto final;
            PSMultiplyVectors(delta_rc, reset_gates, delta, lsize, &mopts);
            success = PSDot(tr_reset_hidden_weights, delta_r, delta, &mopts);
            if (!success) goto final;
        }
    }
    if (previous_layer->delta != NULL) {
        /* Update previous layer delta */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMatrix tr_weights_c = PSMatrixTranspose(
            cell->candidate_weights, 0, &mopts
        );
        PSMatrix tr_weights_u = PSMatrixTranspose(
            cell->update_weights, 0, &mopts
        );
        PSMatrix tr_weights_r = PSMatrixTranspose(
            cell->reset_weights, 0, &mopts
        );
        if (!tr_weights_c || !tr_weights_u || !tr_weights_r) {
            success = 0;
            goto final;
        }
        success = PSDot(tr_weights_c, delta_c, previous_layer->delta, &mopts);
        if (!success) goto final;
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        success = (
            PSDot(tr_weights_u, delta_u, previous_layer->delta, &mopts) &&
            PSDot(tr_weights_r, delta_r, previous_layer->delta, &mopts)
        );
        if (!success) goto final;
    }
final:
    free(delta_c);
    free(delta_u);
    free(delta_r);
    free(delta_rc);
    free(cache);
    return success;
}

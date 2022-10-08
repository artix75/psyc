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

#ifndef __PS_LSTM_H
#define __PS_LSTM_H

#include "psyc.h"

#define PSGetLSTMCell(neuron) ((PSLSTMCell*) neuron->extra)
#define PSGetLSTMGradientBiases(n, gradient) (gradient->weights +\
 n->weights_size)

typedef struct {
    int states_count;
    int weights_size;
    PSFloat *states;
    PSFloat *z_values;
    PSFloat *candidates;
    PSFloat *input_gates;
    PSFloat *output_gates;
    PSFloat *forget_gates;
    PSFloat candidate_bias;
    PSFloat input_bias;
    PSFloat output_bias;
    PSFloat forget_bias;
    PSFloat *candidate_weights;
    PSFloat *input_weights;
    PSFloat *output_weights;
    PSFloat *forget_weights;
} PSLSTMCell;

PSLSTMCell *PSCreateLSTMCell(PSNeuron *neuron, int lsize);
void PSDeleteLSTMCell(PSLSTMCell *cell);
void PSUpdateLSTMBiases(PSNeuron *neuron, PSGradient *gradient,
                        PSGradient *mg, PSGradient *xg, PSFloat rate,
                        PSTrainingOptions *opts, int iteration);

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork *network, PSLayer *layer,
                    int size, int ws);

/* Feedforward Functions */

int PSLSTMFeedforward(PSNeuralNetwork *net, PSLayer *layer, ...);

/* Backpropagation Functions */

int PSLSTMBackprop(PSLayer *layer, PSLayer *previousLayer,
                   PSGradient *lgradients, int t);


#endif /*  __PS_LSTM_H */

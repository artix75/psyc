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

#ifndef __PS_LSTM_H
#define __PS_LSTM_H

#include "psyc.h"

#define PS_LSTM_CANDIDATE_IDX   0
#define PS_LSTM_INPUT_IDX       1
#define PS_LSTM_OUTPUT_IDX      2
#define PS_LSTM_FORGET_IDX      3
#define PS_LSTM_RAWSTATE_IDX    4

typedef struct {
    /* Biases */
    PSFloat *candidate_biases;
    PSFloat *input_biases;
    PSFloat *output_biases;
    PSFloat *forget_biases;
    /* Weights */
    PSMatrix candidate_weights;
    PSMatrix input_weights;
    PSMatrix output_weights;
    PSMatrix forget_weights;
    PSMatrix candidate_hidden_weights;
    PSMatrix input_hidden_weights;
    PSMatrix output_hidden_weights;
    PSMatrix forget_hidden_weights;

    PSMatrix raw_states;
    PSMatrix candidates;
    PSMatrix input_gates;
    PSMatrix output_gates;
    PSMatrix forget_gates;
    PSFloat *initial_raw_states;
    PSFloat *initial_candidates;   /* TODO: Probabily not needed */
    PSFloat *initial_input_gates;  /* TODO: Probabily not needed */
    PSFloat *initial_output_gates; /* TODO: Probabily not needed */
    PSFloat *initial_forget_gates; /* TODO: Probabily not needed */
} PSLSTMCell;

PSLSTMCell *PSGetLSTMCell(PSLayer *layer);

#endif /*  __PS_LSTM_H */

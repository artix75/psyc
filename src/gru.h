/*
 * Copyright (C) 2016-2024 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#ifndef __PS_GRU_H
#define __PS_GRU_H

#include "psyc.h"

#define PS_GRU_CANDIDATE_IDX    0
#define PS_GRU_UPDATE_IDX       1
#define PS_GRU_RESET_IDX        2

typedef struct {
    /* Biases */
    PSFloat *candidate_biases;
    PSFloat *update_biases;
    PSFloat *reset_biases;
    /* Weights */
    PSMatrix candidate_weights;
    PSMatrix update_weights;
    PSMatrix reset_weights;
    PSMatrix candidate_hidden_weights;
    PSMatrix update_hidden_weights;
    PSMatrix reset_hidden_weights;

    PSMatrix candidates;
    PSMatrix update_gates;
    PSMatrix reset_gates;
} PSGRUCell;

PSGRUCell *PSGetGRUCell(PSLayer *layer);

#endif /*  __PS_GRU_H */

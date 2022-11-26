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

#ifndef __DEBUG_H
#define __DEBUG_H

#include <time.h>
#include <fenv.h>
#include "types.h"

#define DEBUG_PHASE_UPDATE_GRADS     1
#define DEBUG_PHASE_UPDATE_WEIGHTS   2

#define PSShouldDebugDump(network) (network->training != NULL &&\
 network->training->debug_dump_to != NULL &&\
 network->training->current_element == 0 &&\
 network->training->current_epoch == 0)

#define PSAssertWithMessage(expr, fmt, ...) do {\
    if (!(expr)) {\
        printf("\n\n== ASSERTION FAILURE ==\n");\
        fprintf(stderr, fmt, __VA_ARGS__);\
        assert(expr);\
    }\
} while (0);

#define PSAddContextualDebug(net,l,n1,n2,prop,v,...) PSAddDebugInfo(\
    net, __FILE__, __func__, __LINE__, l, n1, n2, prop, v, __VA_ARGS__)

typedef struct PSDebugInfo {
    char *file;
    const char *func;
    int line;
    int status;
    int current_epoch;
    int current_batch;
    int current_element;
    int layer_index;
    int layer_type;
    int neuron_index;
    int neuron2_index;
    int layer2_index;
    int convolutional_feature;
    int timestep;
    PSFloat activation;
    PSFloat activation2;
    PSFloat z_value;
    PSFloat bias;
    PSFloat weight;
    PSFloat delta;
    PSFloat delta2;
    char *custom_prop;
    PSFloat custom_val;
    time_t time;
    int has_info;
} PSDebugInfo;

int PSIsFunctionAvailable(const char *func);
int PSCatchFloatingPointExceptions(int except);

char *PSGetNeuronDebugID(PSNeuron *neuron, PSLayer *layer);
void PSTrainingDebugDump(PSNeuralNetwork *network, char *fmt, ...);
void PSTrainingDebugDumpStep(PSNeuralNetwork *network,
                             int training_phase,
                             const char *func,
                             PSLayer *layer,
                             PSNeuron *neuron,
                             char *format, ...);

void PSTrainingDebugDumpHeader(PSNeuralNetwork *network,
                              int data_size,
                              int test_size,
                              int epochs,
                              PSFloat learning_rate,
                              int batch_size);

void PSTrainingDebugDumpGradient(PSNeuralNetwork *network,
                                 int phase,
                                 const char *func,
                                 PSLayer *layer,
                                 int gradient_idx,
                                 int weight_size,
                                 int weight_idx,
                                 int is_avx,
                                 int avx_len);

void PSResetDebugInfo(void);
void PSAddDebugInfo(PSNeuralNetwork *network, char *file, const char *func,
                    int line, PSLayer *layer, void *neuron1, void *neuron2,
                    char *prop, double val, ...);

extern int PSOriginalStdOutFD;
extern char *PSDumpGradientsPath;
#endif /*  __DEBUG_H */

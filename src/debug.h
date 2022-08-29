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

#ifndef __DEBUG_H
#define __DEBUG_H

char *PSGetNeuronDebugID(PSNeuron *neuron, PSLayer *layer);
void PSTrainingDebugDump(PSNeuralNetwork *network, char *fmt, ...);
void PSTrainingDebugDumpStep(PSNeuralNetwork *network,
                             int training_phase,
                             char *func,
                             PSLayer *layer,
                             PSNeuron *neuron,
                             char *format, ...);

void PSTrainingDebugDumpHeader(PSNeuralNetwork *network,
                              int data_size,
                              int test_size,
                              int epochs,
                              double learning_rate,
                              int batch_size);
#endif // __DEBUG_H

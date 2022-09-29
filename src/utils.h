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

#ifndef __PS_UTILS_H
#define __PS_UTILS_H

#include <string.h>
#include <math.h>
#include "types.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define getNeuronLayer(neuron) ((PSLayer*) neuron->layer)
#define getLayerNetwork(layer) ((PSNeuralNetwork*) layer->network)
#define shouldApplyDerivative(network) (network->loss != PSCrossEntropyLoss)
/* Get elapsed time in milliseconds */
#define PSGetElapsedTimeMS(st, et) ((((et.tv_sec - st.tv_sec) * 1000000) \
/* Get elapsed time in microseconds */
#define PSGetElapsedTimeUS(st, et) (((et.tv_sec - st.tv_sec) * 1000000) \
    + (et.tv_usec - st.tv_usec))

#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[97m"
#define BOLD    "\x1b[1m"
#define DIM     "\x1b[2m"
#define HIDDEN  "\x1b[8m"
#define RESET   "\x1b[0m"
#define RESET_BOLD "\x1b[21m"

#define printMemoryErrorMsg() PSErr(NULL, "Could not allocate memory!")

#ifdef PS_DOUBLE_PRECISION
#define tanh_activation tanh
#define PSSqrt(v) sqrt(v)
#define PSFloor(v) floor(v)
#define PSExp(v) exp(v)
#define PSRound(v) round(v)
#define PSMathLog(v) log(v)
#define PSAbs(v) fabs(v)
#define PSPow(a,b) pow(a, b)
#else
#define tanh_activation tanhf
#define PSSqrt(v) sqrtf(v)
#define PSFloor(v) floorf(v)
#define PSExp(v) expf(v)
#define PSRound(v) roundf(v)
#define PSMathLog(v) logf(v)
#define PSAbs(v) fabsf(v)
#define PSPow(a,b) powf(a, b)
#endif


void PSErr(const char* tag, char* fmt, ...);

/* Activation Functions */

PSFloat sigmoid(PSFloat val);

PSFloat sigmoid_derivative(PSFloat val);

PSFloat relu(PSFloat val);

PSFloat relu_derivative(PSFloat val);

PSFloat tanh_derivative(PSFloat val);

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork * network, PSLayer * layer);

/* Misc */

PSFloat normalized_random();

PSFloat gaussian_random(PSFloat mean, PSFloat stddev);

int get_terminal_columns();

void fill_with_blank(int line_length);

PSFloat *copy_floats(PSFloat *src, size_t size);

#endif //__PS_UTILS_H

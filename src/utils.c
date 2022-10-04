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
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "psyc.h"
#include "platform.h"
#include "utils.h"

#if IS_UNIX
#include <sys/ioctl.h>
#include <unistd.h>
#endif

static unsigned char randomSeeded = 0;

void PSErr(const char* tag, char* fmt, ...) {
    va_list args;
    
    fflush (stdout);
    if (PSGlobalFlags & FLAG_LOG_COLORS) fprintf(stderr, RED);
    fprintf(stderr, "ERROR");
    if (tag != NULL) fprintf(stderr, " [%s]: ", tag);
    else fprintf(stderr, ": ");
    
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    if (PSGlobalFlags & FLAG_LOG_COLORS) fprintf(stderr, WHITE);
}

/* Activation Functions */

PSFloat sigmoid(PSFloat val) {
    return 1.0 / (1.0 + PSExp(-val));
}

PSFloat sigmoid_derivative(PSFloat val) {
    return val * (1 - val);
}

PSFloat relu(PSFloat val) {
    return (val >= 0.0 ? val : 0.0);
}

PSFloat relu_derivative(PSFloat val) {
    return (PSFloat)(val > 0.0);
}

PSFloat tanh_derivative(PSFloat val) {
    return (1 - (val * val));
}

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork * network, PSLayer * layer) {
    if (!network->size) return;
    if (layer->index == (network->size - 1)) {
        network->size--;
        if (!network->size) {
            network->input_size = 0;
            network->output_size = 0;
        } else {
            PSLayer * outputLayer = network->layers[network->size - 1];
            if (outputLayer) network->output_size = outputLayer->size;
            else network->output_size = 0;
            PSLayer * inputLayer = network->layers[0];
            if (inputLayer) network->input_size = inputLayer->size;
            else network->input_size = 0;
        }
        PSDeleteLayer(layer);
    }
}

/* Misc */


PSFloat normalized_random() {
    if (!randomSeeded) {
        randomSeeded = 1;
        srand(time(NULL));
    }
    int r = rand();
    return ((PSFloat) r / (PSFloat) RAND_MAX);
}

PSFloat gaussian_random(PSFloat mean, PSFloat stddev) {
    PSFloat theta = 2 * M_PI * normalized_random();
    PSFloat rho = PSSqrt(-2 * PSMathLog(1 - normalized_random()));
    PSFloat scale = stddev * rho;
    PSFloat x = mean + scale * cos(theta);
    PSFloat y = mean + scale * sin(theta);
    PSFloat r = normalized_random();
    return (r > 0.5 ? y : x);
}

int get_terminal_columns() {
    static int __term_columns = -1;
    if (__term_columns < 0) {
#if IS_UNIX
        struct winsize w = {0};
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        __term_columns = w.ws_col;
#else
        __term_columns = 0;
#endif
        if (__term_columns <= 0) __term_columns = 80;
    }
    return __term_columns;
}

void fill_with_blank(int line_length) {
    int term_w = get_terminal_columns();
    int pad = term_w - line_length, i;
    if (pad <= 0) return;
    for (i = 0; i < pad; i++) printf(" ");
    fflush(stdout);
}

PSFloat *copy_floats(PSFloat *src, size_t length) {
    size_t size = length * sizeof(PSFloat);
    PSFloat *dup = malloc(size);
    if (dup == NULL) return NULL;
    memcpy(dup, src, size);
    return dup;
}

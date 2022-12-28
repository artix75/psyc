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
#include "maths.h"

#if IS_UNIX
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork *network, PSLayer *layer) {
    if (network->size == 0) return;
    if (layer->index == (network->size - 1)) {
        network->size--;
        if (network->size == 0) {
            network->input_size = 0;
            network->output_size = 0;
        } else {
            PSLayer *outputLayer = network->layers[network->size - 1];
            if (outputLayer) network->output_size = outputLayer->size;
            else network->output_size = 0;
            PSLayer *inputLayer = network->layers[0];
            if (inputLayer) network->input_size = inputLayer->size;
            else network->input_size = 0;
        }
        PSDeleteLayer(layer);
    }
}

/* Misc */

int PSGetTerminalColumns() {
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

void PSFillWithBlank(int line_length) {
    int term_w = PSGetTerminalColumns();
    int pad = term_w - line_length, i;
    if (pad <= 0) return;
    for (i = 0; i < pad; i++) printf(" ");
    fflush(stdout);
}

PSFloat *PSCopyFloats(PSFloat *src, size_t length) {
    size_t size = length * sizeof(PSFloat);
    PSFloat *dup = malloc(size);
    if (dup == NULL) return NULL;
    memcpy(dup, src, size);
    return dup;
}

/* Compare version string `vers1` with `vers2`.
 * Returns:
 *  -1 if `vers1` < `vers2`
 *  1 if `vers1` > `vers2`
 *  0 if both versions are equal. */
int PSCompareVersion(const char* vers1, const char* vers2) {
    int major1 = 0, minor1 = 0, patch1 = 0;
    int major2 = 0, minor2 = 0, patch2 = 0;
    sscanf(vers1, "%d.%d.%d", &major1, &minor1, &patch1);
    sscanf(vers2, "%d.%d.%d", &major2, &minor2, &patch2);
    if (major1 < major2) return -1;
    if (major1 > major2) return 1;
    if (minor1 < minor2) return -1;
    if (minor1 > minor2) return 1;
    if (patch1 < patch2) return -1;
    if (patch1 > patch2) return 1;
    return 0;
}

char *PSGetElapsedTimeString(time_t elapsed_us, int opts) {
    static char elapsed_str[256];
    static const char *time_units_short[] = {"us", "ms", "s", "m", "h"};
    static const char *time_units_long[] = {
        "usec.", "msec.", "sec.", "min.", "hour(s)"
    };
    static const char *time_units_full[] = {
        "microsecond(s)", "millisecond(s)", "second(s)", "minute(s)", "hour(s)"
    };
    static const size_t numunits = sizeof(time_units_short) / sizeof(char *);
    int long_format = (opts & OPT_TIME_LONG),
        full_format = (opts & OPT_TIME_FULL),
        human = (opts & OPT_TIME_HUMAN), i = 0;
    const char **units = NULL;
    char *sep = " ";
    if (full_format) units = time_units_full;
    else if (long_format) units = time_units_long;
    else {
        units = time_units_short;
        sep = "";
    }
    double elapsed = (double) elapsed_us, div = 1000, mod = 0;
    while (elapsed >= div) {
        if (++i >= (int) numunits) break;
        mod = fmod(elapsed, div);
        elapsed /= div;
        if (i > 1) div = 60;
    }
    elapsed_str[0] = '\0';
    const char *unit = units[i];
    int round = ((mod > 0) ? 1 : 0);
    if (human) {
        if (mod > 0 && i > 0) {
            const char *lower_unit = units[i - 1];
            snprintf(
                elapsed_str, 255, "%ld%s%s and %ld%s%s",
                (long) elapsed, sep, unit, (long) mod, sep, lower_unit
            );
        } else snprintf(elapsed_str, 255, "%ld%s%s", (long)elapsed, sep, unit);
    } else snprintf(elapsed_str, 255, "%.*f%s%s", round, elapsed, sep, unit);
    return elapsed_str;
}

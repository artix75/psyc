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

#ifndef __PS_UTILS_H
#define __PS_UTILS_H

#include <string.h>
#include <math.h>
#include <time.h>
#include "types.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

/* Get elapsed time in milliseconds */
#define PSGetElapsedTimeMS(st, et) ((((et.tv_sec - st.tv_sec) * 1000000) \
/* Get elapsed time in microseconds */
#define PSGetElapsedTimeUS(st, et) (((et.tv_sec - st.tv_sec) * 1000000) \
    + (et.tv_usec - st.tv_usec))

#define OPT_TIME_LONG        (1 << 0)
#define OPT_TIME_FULL        (1 << 1)
#define OPT_TIME_HUMAN       (1 << 2)

#define PSClipValue(v, max, min) (v > max ? max : (v < min ? min : v))

/* Activation Functions */

PSFloat PSSigmoid(PSFloat val);

PSFloat PSSigmoidDerivative(PSFloat val);

PSFloat PSRelu(PSFloat val);

PSFloat PSReluDerivative(PSFloat val);

PSFloat PSTanhDerivative(PSFloat val);

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork *network, PSLayer *layer);

/* Misc */

int PSGetTerminalColumns();

void PSFillWithBlank(int line_length);

PSFloat *PSCopyFloats(PSFloat *src, size_t size);

char *PSGetElapsedTimeString(time_t elapsed_us, int long_format);

#endif /* __PS_UTILS_H */

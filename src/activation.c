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

#include <stdint.h>
#include <math.h>
#include "types.h"
#include "config.h"
#include "activation.h"
#include "maths.h"

#define UNUSED(V) ((void) V)

/* Activation Functions */

PSFloat PSSigmoid(PSFloat val) {
    return 1.0 / (1.0 + PSExp(-val));
}

PSFloat PSSigmoidDerivative(PSFloat val) {
    return val * (1 - val);
}

PSFloat PSRelu(PSFloat val) {
    return (val >= 0.0 ? val : 0.0);
}

PSFloat PSReluDerivative(PSFloat val) {
    return (PSFloat)(val > 0.0);
}

PSFloat PSTanhDerivative(PSFloat val) {
    return (1 - (val * val));
}

void PSSigmoidV(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    uint8_t acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
    if (acceleration != PSAcceleration_None) {
        PSVectorNeg(vec, dest, len, opts);
        PSVectorExp(dest, dest, len, opts);
        PSSumVectorScalar(dest, 1.0, dest, len, opts);
        PSDivideScalarVector(1.0, dest, dest, len, opts);\
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSSigmoid(vec[i]);
}

void PSReluV(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSVectorThreshold(vec, 0.0, dest, len, opts);
}

void PSSigmoidDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                          PSMathOpts *opts)
{
    if (dest == NULL) dest = vec;
    uint8_t acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
    if (acceleration != PSAcceleration_None) {
        PSSubtractScalarVector(1.0, vec, dest, len, opts);
        PSMultiplyVectors(vec, dest, dest, len, opts);
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSSigmoidDerivative(vec[i]);
}

void PSTanhDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                       PSMathOpts *opts)
{
    if (dest == NULL) dest = vec;
    uint8_t acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
    if (acceleration != PSAcceleration_None) {
        PSMultiplyVectors(vec, vec, dest, len, opts);
        PSSubtractScalarVector(1.0, dest, dest, len, opts);
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSTanhDerivative(vec[i]);
}

void PSReluDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                       PSMathOpts *opts)
{
    UNUSED(opts);
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = (PSFloat)(vec[i] > 0.0);
}

void PSSoftmax(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    uint8_t acceleration = PSGlobalAcceleration;
    PSFloat max = PSFLOAT_MIN, esum = 0.0;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        if (opts->max != NULL) max = *(opts->max);
    }
    if (PSACFEnabled(acceleration)) {
        if (max == PSFLOAT_MIN) max = PSVectorMax(vec, NULL, len, opts);
        PSSubtractVectorScalar(vec, max, dest, len, opts);
        PSVectorExp(dest, NULL, len, opts);
        esum = PSSumVectorElements(dest, len, opts);
        PSDivideVectorScalar(dest, esum, NULL, len, opts);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) {
        PSFloat n = vec[i];
        if (i == 0 || n > max) max = n;
    }
    for (i = 0; i < len; i++) {
        PSFloat e = PSExp(vec[i] - max);
        esum += e;
        dest[i] = e;
    }
    for (i = 0; i < len; i++) dest[i] = dest[i] / esum;
}

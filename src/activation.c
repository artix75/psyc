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

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "types.h"
#include "config.h"
#include "activation.h"
#include "maths.h"

#define UNUSED(V) ((void) V)

void PSInitActivationMathOpts(PSMathOpts *opts, PSMathOpts *srcopts) {
    uint8_t acceleration = PSGlobalAcceleration;
    if (srcopts != NULL) acceleration = srcopts->acceleration;
    opts->acceleration = acceleration;
}

/* Activation Functions */

PSFloat PSSigmoidS(PSFloat val) {
    return 1.0 / (1.0 + PSExp(-val));
}

PSFloat PSSigmoidDerivativeS(PSFloat val) {
    return val * (1 - val);
}

PSFloat PSReluS(PSFloat val) {
    return (val >= 0.0 ? val : 0.0);
}

PSFloat PSReluDerivativeS(PSFloat val) {
    return (PSFloat)(val > 0.0);
}

PSFloat PSTanhDerivativeS(PSFloat val) {
    return (1 - (val * val));
}

void PSSigmoid(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (mopts.acceleration != PSAcceleration_None) {
        PSVectorNeg(vec, dest, len, &mopts);
        PSVectorExp(dest, dest, len, &mopts);
        PSSumVectorScalar(dest, 1.0, dest, len, &mopts);
        PSDivideScalarVector(1.0, dest, dest, len, &mopts);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSSigmoidS(vec[i]);
}

void PSRelu(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    PSVectorThreshold(vec, 0.0, dest, len, &mopts);
}

void PSSigmoidDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                         PSMathOpts *opts)
{
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (mopts.acceleration != PSAcceleration_None) {
        PSSubtractScalarVector(1.0, vec, dest, len, &mopts);
        PSMultiplyVectors(vec, dest, dest, len, &mopts);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSSigmoidDerivativeS(vec[i]);
}

void PSTanhDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts)
{
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (mopts.acceleration != PSAcceleration_None) {
        PSMultiplyVectors(vec, vec, dest, len, &mopts);
        PSSubtractScalarVector(1.0, dest, dest, len, &mopts);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSTanhDerivativeS(vec[i]);
}

void PSReluDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts)
{
    UNUSED(opts);
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = (PSFloat)(vec[i] > 0.0);
}

void PSSoftmax(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSFloat max = PSFLOAT_MIN, esum = 0.0;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (PSACFEnabled(mopts.acceleration)) {
        max = PSVectorMax(vec, NULL, len, &mopts);
        PSSubtractVectorScalar(vec, max, dest, len, &mopts);
        PSVectorExp(dest, NULL, len, &mopts);
        esum = PSSumVectorElements(dest, len, &mopts);
        PSDivideVectorScalar(dest, esum, dest, len, &mopts);
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

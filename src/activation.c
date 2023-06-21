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
#include <string.h>
#include <math.h>
#include "types.h"
#include "config.h"
#include "activation.h"
#include "maths.h"
#include "log.h"

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

PSFloat PSGeluS(PSFloat val) {
    static PSFloat c = 0;
    if (c == 0) c = PSSqrt(2 / M_PI);
    return 0.5 * val * (1 + PSTanhS(c * (val + 0.044715 * PSPow(val, 3))));
}

PSFloat PSGeluDerivativeS(PSFloat val) {
    static PSFloat c1 = 0, c2 = 0, c3 = 0;
    if (c1 == 0) {
        c1 = PSSqrt(2 / M_PI);
        c2 = PSSqrt(2);
        c3 = 2 / PSSqrt(M_PI);
    }
    PSFloat appr = PSTanh(c1 * (val + 0.044715 * PSPow(val, 3)));
    PSFloat erf_prime = c3 * PSExp(-PSPow((val / c2), 2));
    return 0.5 + 0.5 * appr + ((0.5 * val * erf_prime) / c2);
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

void PSGelu(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    static PSFloat c = 0;
    if (c == 0) c = PSSqrt(2 / M_PI);
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (mopts.acceleration != PSAcceleration_None) {
        PSFloat *tmpdest = NULL;
        if (dest == vec) {
            tmpdest = calloc(len, sizeof(PSFloat));
            if (tmpdest == NULL) {
                PSPrintMemoryErrorMsg();
                abort();
            }
            dest = tmpdest;
        }
        int i;
        /* vec ^ 3 */
        for (i = 0; i < 2; i++) {
            PSFloat *a = (i == 0 ? vec : dest);
            PSMultiplyVectors(a, vec, dest, len, &mopts);
        }
        /* 0.044715 * dest */
        PSMultiplyVectorScalar(dest, 0.044715, dest, len, &mopts);
        /* val + dest */
        PSSumVectors(dest, vec, dest, len, &mopts);
        /* 1 + tanh(c * dest) */
        PSMultiplyVectorScalar(dest, c, dest, len, &mopts);
        PSVectorTanh(dest, dest, len, &mopts);
        PSSumVectorScalar(dest, 1, dest, len, &mopts);
        /* 0.5 * vec * dest */
        PSMultiplyVectors(dest, vec, dest, len, &mopts);
        PSMultiplyVectorScalar(dest, 0.5, dest, len, &mopts);
        if (tmpdest != NULL) {
            dest = vec;
            PSVectorCopy(dest, tmpdest, len);
            free(tmpdest);
        }
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSGeluS(vec[i]);
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

void PSGeluDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                     PSMathOpts *opts)
{
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    static PSFloat c1 = 0, c2 = 0, c3 = 0;
    if (mopts.acceleration != PSAcceleration_None) {
        int i;
        if (c1 == 0) {
            c1 = PSSqrt(2 / M_PI);
            c2 = PSSqrt(2);
            c3 = 2 / PSSqrt(M_PI);
        }
        PSFloat *erf_prime = malloc(len * sizeof(PSFloat));
        if (erf_prime == NULL) {
            PSPrintMemoryErrorMsg();
            abort();
        }
        /* vec ^ 3 */
        for (i = 0; i < 2; i++) {
            PSFloat *a = (i == 0 ? vec : dest);
            PSMultiplyVectors(a, vec, dest, len, &mopts);
        }
        /* 0.044715 * dest */
        PSMultiplyVectorScalar(dest, 0.044715, dest, len, &mopts);
        /* val + dest */
        PSSumVectors(dest, vec, dest, len, &mopts);
        /* tanh(c * dest) */
        PSMultiplyVectorScalar(dest, c1, dest, len, &mopts);
        PSVectorTanh(dest, dest, len, &mopts);

        /* erf_prime = x / c2 */
        PSDivideVectorScalar(vec, c2, erf_prime, len, &mopts);
        /* erf_prime = exp(-erf_prime ^ 2) */
        PSMultiplyVectors(erf_prime, erf_prime, erf_prime, len, &mopts);
        PSVectorNeg(erf_prime, erf_prime, len, &mopts);
        PSVectorExp(erf_prime, erf_prime, len, &mopts);
        /* erf_prime = (2 / sqrt(PI)) * erf_prime */
        PSMultiplyVectorScalar(erf_prime, c3, erf_prime, len, &mopts);

        /* 0.5 * vec * erf_prime / sqrt(2) */
        PSMultiplyVectors(vec, erf_prime, erf_prime, len, &mopts);
        PSMultiplyVectorScalar(erf_prime, 0.5, erf_prime, len, &mopts);
        PSDivideVectorScalar(erf_prime, c2, erf_prime, len, &mopts);

        /* 0.5 + (0.5 * dest) + erf_prime */
        PSMultiplyVectorScalar(dest, 0.5, dest, len, &mopts);
        PSSumVectors(dest, erf_prime, dest, len, &mopts);
        PSSumVectorScalar(dest, 0.5, dest, len, &mopts);

        free(erf_prime);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSGeluDerivativeS(vec[i]);
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

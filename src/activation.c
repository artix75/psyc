/*
 * Copyright (C) 2016-2023 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

/*** Activation Functions ***/

/* Sigmoid activation function for scalars. It takes the scalar `val` as
 * argument and returns a `PFloat` scalar.
 * For info about sigmoid:
 *   https://en.wikipedia.org/wiki/Sigmoid_function
 * The equivalent function to be used with vectors/matrices is `PSSigmoid`.
 * The derivative of this function is `PSSigmoidDerivativeS`.
 * Return value: sigmoid scalar result. */
PSFloat PSSigmoidS(PSFloat val) {
    return 1.0 / (1.0 + PSExp(-val));
}

/* Computes derivative for sigmoid activation function (`PSSigmoidS`).
 * This function applies to scalar values, so it takes the scalar `val`
 * as argument and returns a `PFloat` scalar.
 * The equivalent function to be used with vectors/matrices is
 * `PSSigmoidDerivative`.
 * Return value: ReLU derivative scalar result. */
PSFloat PSSigmoidDerivativeS(PSFloat val) {
    return val * (1 - val);
}

/* ReLU (Rectified Linear Unit) activation function for scalars.
 * It takes the scalar `val` as argument and returns a `PFloat` scalar.
 * For info about ReLU:
 *   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 * The equivalent function to be used with vectors/matrices is `PSRelu`.
 * The derivative of this function is `PSReluDerivativeS`.
 * Return value: ReLU scalar result. */
PSFloat PSReluS(PSFloat val) {
    return (val >= 0.0 ? val : 0.0);
}

/* Computes derivative for ReLU activation function (`PSReluS`). This function
 * applies to scalar values, so it takes the scalar `val` as argument and
 * returns a `PFloat` scalar.
 * The equivalent function to be used with vectors/matrices is
 * `PSReluDerivative`.
 * Return value: ReLU derivative scalar result. */
PSFloat PSReluDerivativeS(PSFloat val) {
    return (PSFloat)(val > 0.0);
}

/* GELU (Gaussian Error Linear Units) activation function for scalars.
 * It takes the scalar `val` as argument and returns a `PFloat` scalar.
 * For info about GELU:
 *   https://arxiv.org/abs/1606.08415
 * The equivalent function to be used with vectors/matrices is `PSGelu`.
 * The derivative of this function is `PSGeluDerivativeS`.
 * Return value: GELU scalar result. */
PSFloat PSGeluS(PSFloat val) {
    static PSFloat c = 0;
    if (c == 0) c = PSSqrt(2 / M_PI);
    return 0.5 * val * (1 + PSTanhS(c * (val + 0.044715 * PSPow(val, 3))));
}

/* Computes derivative for GELU activation function (`PSGeluS`). This function
 * applies to scalar values, so it takes the scalar `val` as argument and
 * returns a `PFloat` scalar.
 * The equivalent function to be used with vectors/matrices is
 * `PSGeluDerivative`.
 * Return value: GELU derivative scalar result. */
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

/* Sigmoid activation function for vectors. Sigmoid is computed on vector `vec`
 * of length `len` and stored into vector `dest`. If `dest` is NULL, results
 * will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * For info about sigmoid:
 *   https://en.wikipedia.org/wiki/Sigmoid_function
 * The equivalent function to be used with scalars is `PSSigmoidS`.
 * The derivative of this function is `PSSigmoidDerivative`. */
void PSSigmoid(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (mopts.acceleration != PSAcceleration_None) {
        PSVectorNeg(vec, dest, len, &mopts);
        PSVectorExp(dest, dest, len, &mopts);
        PSAddVectorScalar(dest, 1.0, dest, len, &mopts);
        PSDivideScalarVector(1.0, dest, dest, len, &mopts);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSSigmoidS(vec[i]);
}

/* Tanh (hyperbolic tangent) activation function for vectors. Hyperbolic
 * tangent is computed on vector `vec` of length `len` and stored into vector
 * `dest`.
 * If `dest` is NULL, results will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The derivative of this function is `PSTanhDerivative`. */
void PSTanhActivation(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts)
{
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    PSVectorTanh(vec, dest, len, &mopts);
}

/* ReLU (Rectified Linear Unit) activation function for vectors.
 * ReLU is computed on vector `vec` of length `len` and stored into vector
 * `dest`.
 * If `dest` is NULL, results will be stored
 * into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * For info about ReLU:
 *   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 *
 * The equivalent function to be used with scalars is `PSReluS`.
 * The derivative of this function is `PSReluDerivative`. */
void PSRelu(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    PSVectorThreshold(vec, 0.0, dest, len, &mopts);
}

/* GELU (Gaussian Error Linear Units) activation function for vectors.
 * GELU is computed on vector `vec` of length `len` and stored into vector
 * `dest`.
 * If `dest` is NULL, results will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * For info about GeLU:
 *   https://arxiv.org/abs/1606.08415
 * The equivalent function to be used with scalars is `PSGeLUS`.
 * The derivative of this function is `PSGeLUDerivative`. */
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
        PSAddVectors(dest, vec, dest, len, &mopts);
        /* 1 + tanh(c * dest) */
        PSMultiplyVectorScalar(dest, c, dest, len, &mopts);
        PSVectorTanh(dest, dest, len, &mopts);
        PSAddVectorScalar(dest, 1, dest, len, &mopts);
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

/* Computes the derivative of sigmoid activation function (`PSSigmoid`) for
 * vectors. The sigmoid derivative is computed on vector `vec` of length `len`
 * and stored into vector `dest`. If `dest` is NULL, results will be stored
 * into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The equivalent function to be used with scalars is `PSSigmoidDerivativeS`.*/
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

/* Computes the derivative of tanh (hyperbolic tangent) activation function
 * (`PSTanhActivation`) for vectors. The derivative is computed on vector `vec`
 * of length `len` and stored into vector `dest`.
 * If `dest` is NULL, results will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The equivalent function to be used with scalars is `PSTanhDerivativeS`.*/
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

/* Computes the derivative of ReLU activation function (`PSRelu`) for vectors.
 * The derivative is computed on vector `vec` of length `len` and stored into
 * vector `dest`.
 * If `dest` is NULL, results will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The equivalent function to be used with scalars is `PSReluDerivativeS`.*/
void PSReluDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts)
{
    UNUSED(opts);
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = (PSFloat)(vec[i] > 0.0);
}

/* Computes the derivative of GELU activation function (`PSGelu`) for vectors.
 * The derivative is computed on vector `vec` of length `len` and stored into
 * vector `dest`.
 * If `dest` is NULL, results will be stored into `vec` itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The equivalent function to be used with scalars is `PSGeluDerivativeS`.*/
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
        PSAddVectors(dest, vec, dest, len, &mopts);
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
        PSAddVectors(dest, erf_prime, dest, len, &mopts);
        PSAddVectorScalar(dest, 0.5, dest, len, &mopts);

        free(erf_prime);
        return;
    }
    uint64_t i;
    for (i = 0; i < len; i++) dest[i] = PSGeluDerivativeS(vec[i]);
}

/* Computes Softmax function on vector `vec` of length `len`. Result is stored
 * into vector `dest`. If `dest` is NULL, result will be stored into `vec`
 * itself.
 * The `opts` argument can be used to change default acceleration used to
 * compute results (see `PSMathOpts`).
 * The Softmax function can be used to get the probability distribution from
 * a series of numbers.
 * For more info about Softmax:
 *     https://en.wikipedia.org/wiki/Softmax_function */
void PSSoftmax(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts) {
    if (dest == NULL) dest = vec;
    PSFloat max = PSFLOAT_MIN, esum = 0.0;
    PSMathOpts mopts = {0};
    PSInitActivationMathOpts(&mopts, opts);
    if (PSACFEnabled(mopts.acceleration)) {
        max = PSVectorMax(vec, NULL, len, &mopts);
        PSSubtractVectorScalar(vec, max, dest, len, &mopts);
        PSVectorExp(dest, dest, len, &mopts);
        esum = PSVectorReduceSum(dest, len, &mopts);
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

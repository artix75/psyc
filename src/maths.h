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

#ifndef __PS_MATHS_H__
#define __PS_MATHS_H__

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "types.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanh(v) tanh(v)
#define PSTanhActivation tanh
#define PSSqrt(v) sqrt(v)
#define PSFloor(v) floor(v)
#define PSExp(v) exp(v)
#define PSRound(v) round(v)
#define PSMathLog(v) log(v)
#define PSAbs(v) fabs(v)
#define PSPow(a,b) pow(a, b)
#else
#define PSTanh(v) tanhf(v)
#define PSTanhActivation tanhf
#define PSSqrt(v) sqrtf(v)
#define PSFloor(v) floorf(v)
#define PSExp(v) expf(v)
#define PSRound(v) roundf(v)
#define PSMathLog(v) logf(v)
#define PSAbs(v) fabsf(v)
#define PSPow(a,b) powf(a, b)
#endif

#define PSClipValue(v, min, max) (v > max ? max : (v < min ? min : v))

#define MATHS_STORE_MODE_NORM 0
#define MATHS_STORE_MODE_ADD  1
#define MATHS_STORE_MODE_SUB  2

struct PSMathOpts;
typedef void (*PSDotProductDebug)(int i, PSFloat a, PSFloat b, PSFloat sum,
                                  int using_acceleration,
                                  struct PSMathOpts *opts);

typedef struct PSMathOpts {
    int acceleration;
    int store_mode;
    PSDotProductDebug debug_step;
    void *data;
} PSMathOpts;

/****** Utils *****/

PSFloat PSNormalizedRandom();
PSFloat PSGaussianRandom(PSFloat mean, PSFloat stddev);

/**** PSMatrix ****/

#define PSMatrixDataSize(matrix) (PSMatrixLength(matrix) * sizeof(PSFloat))
#define PSMatrixStrideBytes(matrix,i) \
    (PSMatrixStride(matrix,i) * sizeof(PSFloat))

typedef PSFloat *PSMatrix;
typedef PSFloat (*PSMatrixInitializer)(PSMatrix matrix, int idx, PSFloat n);
PSMatrix PSMatrixCreate(PSFloat init_value, PSMatrixInitializer initializer,
                        int ndims, ...);
PSMatrix PSMatrixZeros(int ndims, ...);
PSMatrix PSMatrixRandom(int ndims, ...);
PSMatrix PSMatrixWithGaussianRandom(PSFloat stddev, int ndims, ...);
int PSMatrixNumDims(PSMatrix matrix);
int PSMatrixDim(PSMatrix matrix, int dim);
size_t PSMatrixLength(PSMatrix matrix);
int PSMatrixStride(PSMatrix matrix, int dim);
PSFloat *PSMatrixValues(PSMatrix matrix, uint32_t *len, int argc, ...);
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result);
int PSMatrixProductMV(PSMatrix a, PSFloat *b, int len, PSMatrix *result);
int PSMatrixProductVM(PSFloat *a, PSMatrix b, int len, PSMatrix *result);
void PSMatrixDelete(PSMatrix matrix);

/**** Operations ***/

void PSSumVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                  PSMathOpts *opts);
void PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts);
void PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts);
void PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts);
void PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts);
void PSSumVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts);
void PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts);
void PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts);
void PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts);
void PSVectorTanh(PSFloat *a, PSFloat *dest, uint64_t length,
                  PSMathOpts *opts);
void PSVectorSqrt(PSFloat *a, PSFloat *dest, uint64_t length,
                  PSMathOpts *opts);
void PSVectorExp(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts);
void PSVectorNeg(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts);
void PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                  uint64_t length, PSMathOpts *opts);
void PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts);
PSFloat PSDotProduct(PSFloat *a, PSFloat *b, uint64_t length, PSMathOpts *opts);
PSFloat PSDotSquare(PSFloat *a, uint64_t length, PSMathOpts *opts);

#endif /* __PS_MATHS_H__ */


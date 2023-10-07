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

#ifndef __PS_MATHS_H__
#define __PS_MATHS_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "types.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanh(v) tanh(v)
#define PSSqrt(v) sqrt(v)
#define PSFloor(v) floor(v)
#define PSExp(v) exp(v)
#define PSRound(v) round(v)
#define PSMathLog(v) log(v)
#define PSMathLog10(v) log10(v)
#define PSAbs(v) fabs(v)
#define PSPow(a,b) pow(a, b)
#define PSSin(a) sin(a)
#define PSCos(a) cos(a)
#else
#define PSTanh(v) tanhf(v)
#define PSSqrt(v) sqrtf(v)
#define PSFloor(v) floorf(v)
#define PSExp(v) expf(v)
#define PSRound(v) roundf(v)
#define PSMathLog(v) logf(v)
#define PSMathLog10(v) log10f(v)
#define PSAbs(v) fabsf(v)
#define PSPow(a,b) powf(a, b)
#define PSSin(a) sinf(a)
#define PSCos(a) cosf(a)
#endif

#define PSClipValue(v, min, max) (v > max ? max : (v < min ? min : v))
#define PSVectorCopy(dest, src, len) memcpy(dest, src, len * sizeof(PSFloat))
#define PSVectorClear(vec, len) memset(vec, 0, len * sizeof(PSFloat))
#define PSVectorZero(len) calloc(len, sizeof(PSFloat))
#define PSVectorCreate(len) malloc(len * sizeof(PSFloat))

#define PS_STORE_MODE_SET 0
#define PS_STORE_MODE_ADD 1
#define PS_STORE_MODE_SUB 2

#define PS_SHAPE_TYPE_NONE   0
#define PS_SHAPE_TYPE_SCALAR 1
#define PS_SHAPE_TYPE_ROW    2
#define PS_SHAPE_TYPE_COL    3
#define PS_SHAPE_TYPE_MATRIX 4

#define PS_MATRIX_MAX_DIMENSIONS 3

struct PSMathOpts;
typedef void (*PSDotProductDebug)(int i, PSFloat a, PSFloat b, PSFloat sum,
                                  int using_acceleration,
                                  struct PSMathOpts *opts);
typedef PSFloat (*PSFloatFunc) (PSFloat n);

/* This structure can be passed to various operations. Not all of its
 * properties are used by all operations.
 * Properties:
 *  - `acceleration`: see `PSAcceleration`
 *  - `store_mode`: specifies how results will be stored into destination:
 *                  - `PS_STORE_MODE_SET`: results will overwrite dest.
 *                  - `PS_STORE_MODE_ADD`: results will be added to dest.
 *                  - `PS_STORE_MODE_SUB`: results will be subtracted from
 *                     dest.
 *  - `transpose`   some operations involving PSMatrix could use this in order
 *                  to transpose one or more matrices. The integer value
 *                  indicates the (1-based) matrix argument position, ie.
 *                  1 for first matrix arg, 2 for second matrix arg, etc.
 *                  More than one matrix can be set (ie. 1 | 2).
 *  - `argtype`     specifies if arguments are `PSMatrix` or `PSFloat *`
 *                  (vector). Functions using this property (such as `PSDot`),
 *                  must have PSMatrix arguments and the property can be used
 *                  to tell the function that one or more arguments must be
 *                  treated as vectors.
 *                  - 'M' or 'm': `PSMatrix` matrix
 *                  - 'V' or 'v': `PSFloat *` vector
 *                  The index indicated argument position (zero-based), ie:
 *                  argtype[1] means that second PSMatrix argument has to be
 *                  treated as vector.
 *  - `vector_len`  optionally pass vector length to functions that cannot
 *                  retrieve this info from matrix arguments, ie. when `PSDot`
 *                  is called with both vectors (`argtype` = {'V', 'V'})
 *  - `tmpdest`:    some operations may use this vector as a cache in order
 *                  to avoid allocating extra memory, for intermediate
 *                  computations.
 *  - `debug_step`: used for debugging by some operations.
 */
typedef struct PSMathOpts {
    int                 acceleration;
    int                 store_mode;
    int                 transpose;
    char                argtype[3];
    int                 vector_len;
    PSFloat             *tmpdest;
    PSDotProductDebug   debug_step;
    void                *data;
} PSMathOpts;

/****** Utils *****/

PSFloat PSNormalizedRandom();
PSFloat PSGaussianRandom(PSFloat mean, PSFloat stddev);
unsigned int PSRandomInt(unsigned int range, PSFloat *weights, int *err,
                         PSMathOpts *opts);

/**** PSMatrix ****/

#define PSMatrixDataSize(matrix) (PSMatrixLength(matrix) * sizeof(PSFloat))
#define PSMatrixStrideBytes(matrix,i) \
    (PSMatrixStride(matrix,i) * sizeof(PSFloat))

/* Basically, `PSMatrix` can be used as a normal array of `PSFloat`.
 * Anyway, it privately holds more info that allow it to be used as a
 * multidimensional matrix, so that matrix operations can be performed on
 * then (transposition, matrix multiplication, etc.).
 * Private data is actually allocated just before the memory address pointed
 * by `PSMatrix`, so you should **NEVER** free PSMatrix by usual `free`,
 * but you have to call PSMatrixDelete instead. */
typedef PSFloat *PSMatrix;
typedef PSFloat (*PSMatrixInitializer)(PSMatrix matrix, int idx, PSFloat n);
PSMatrix PSMatrixCreate(PSFloat init_value, PSMatrixInitializer initializer,
                        int ndims, ...);
PSMatrix PSMatrixCreateWithShape(PSFloat init_value,
                                 PSMatrixInitializer initializer,
                                 int ndims, int *shape);
PSMatrix PSMatrixZeros(int ndims, ...);
PSMatrix PSMatrixRandom(int ndims, ...);
PSMatrix PSMatrixWithGaussianRandom(PSFloat stddev, int ndims, ...);
PSMatrix PSMatrixFromArray(PSFloat *array, int ndims, ...);
PSMatrix PSMatrixExpand(PSMatrix src, int add, int keep_src);
int PSMatrixNumDims(PSMatrix matrix);
int PSMatrixDim(PSMatrix matrix, int dim);
int PSMatrixDimensions(PSMatrix matrix, int *dims);
uint64_t PSMatrixLength(PSMatrix matrix);
int PSMatrixStride(PSMatrix matrix, int dim);
int PSMatrixShapeType(PSMatrix matrix);
void PSMatrixPrintInfo(PSMatrix matrix, const char *name, int newline);
void PSMatrixPrintShape(PSMatrix matrix, int newline);
int PSMatrixWrite(PSMatrix matrix, const char *sep, char bracket,
                  int indent, FILE *out);
void PSMatrixPrint(PSMatrix matrix, const char *sep, int print_shape);
PSFloat *PSMatrixGet(PSMatrix matrix, int ndims, uint32_t *len, ...);
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixProductMV(PSMatrix a, PSFloat *b, int len, PSFloat **result,
                      PSMathOpts *opts);
int PSMatrixProductVM(PSFloat *a, PSMatrix b, int len, PSMatrix *result,
                      PSMathOpts *opts);
int PSMatrixAdd(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixMultiply(PSMatrix a, PSMatrix b, PSMatrix *result,PSMathOpts *opt);
int PSMatrixSubtract(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixDivide(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
PSMatrix PSMatrixReshape(PSMatrix matrix, int num_dims, ...);
PSMatrix PSMatrixFlatten(PSMatrix matrix);
PSMatrix *PSMatrixSplit(PSMatrix matrix, int num_slices, int axis,
                        PSMathOpts *opts);
PSMatrix PSMatrixTranspose(PSMatrix matrix, int rebuild, PSMathOpts *opts);
PSMatrix PSMatrixSwapAxes(PSMatrix matrix, int axis1, int axis2);
void PSMatrixResetTransposed(PSMatrix matrix);
PSMatrix PSMatrixDup(PSMatrix matrix);
PSMatrix PSMatrixDupShape(PSMatrix matrix);
int PSMatrixCopy(PSMatrix src, PSMatrix dst);
int PSMatrixEquals(PSMatrix a, PSMatrix b, int precision, int ignore_shape);
void PSMatrixClear(PSMatrix matrix);
void PSMatrixDelete(PSMatrix matrix);

/**** Operations ***/

PSFloat *PSAddVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                      PSMathOpts *opts);
PSFloat *PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest,
                           uint64_t length, PSMathOpts *opts);
PSFloat *PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest,
                           uint64_t length, PSMathOpts *opts);
PSFloat *PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest,
                         uint64_t length, PSMathOpts *opts);
PSFloat *PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                uint64_t length, PSMathOpts *opts);
PSFloat *PSAddVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                           uint64_t length, PSMathOpts *opts);
PSFloat *PSSubtractVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                uint64_t length, PSMathOpts *opts);
PSFloat *PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                                uint64_t length, PSMathOpts *opts);
PSFloat *PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                              uint64_t length, PSMathOpts *opts);
PSFloat *PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                              uint64_t length, PSMathOpts *opts);
PSFloat *PSVectorTanh(PSFloat *a, PSFloat *dest, uint64_t length,
                      PSMathOpts *opts);
PSFloat *PSVectorSqrt(PSFloat *a, PSFloat *dest, uint64_t length,
                      PSMathOpts *opts);
PSFloat *PSVectorExp(PSFloat *a, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts);
PSFloat *PSVectorNeg(PSFloat *a, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts);
PSFloat *PSVectorAbs(PSFloat *a, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts);
PSFloat *PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                      uint64_t length, PSMathOpts *opts);
PSFloat *PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                           uint64_t length, PSMathOpts *opts);
PSFloat *PSVectorMapWithLimit(PSFloat *a, PSFloat limit, PSFloat mapper,
                              PSFloat *dest, uint64_t length, PSMathOpts *opts);
PSFloat *PSVectorPower(PSFloat *a, PSFloat exp, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts);
PSFloat PSVectorMax(PSFloat *a, uint64_t *index, uint64_t length,
                    PSMathOpts *opts);
PSFloat PSVectorReduceSum(PSFloat *a, uint64_t length, PSMathOpts *opts);
int PSCumulativeSum(PSFloat *a, PSFloat *dest, uint64_t length);
PSFloat PSMean(PSFloat *a, uint64_t length, PSMathOpts *opts);
PSFloat PSVariance(PSFloat *a, uint64_t len, PSMathOpts *opts);
PSFloat PSStdDev(PSFloat *a, uint64_t len, PSMathOpts *opts);
PSFloat PSDotProduct(PSFloat *a, PSFloat *b, uint64_t length, PSMathOpts *opts);
PSFloat PSDotSquare(PSFloat *a, uint64_t length, PSMathOpts *opts);
void PSVectorFill(PSFloat *vec, PSFloat val, uint64_t len, PSMathOpts *opts);
void PSVectorWrite(PSFloat *vec, int len, char* sep, FILE *f);
void PSVectorPrint(PSFloat *vec, int len, char* sep);
PSFloat *PSVectorTranspose(PSFloat *vec, PSFloat *dest, int acceleration,
                           int ndims, ...);
int PSMatMul(PSFloat *a, PSFloat *b, PSFloat *dest, int m, int n, int k,
             PSMathOpts *opts);
int PSDot(PSMatrix a, PSMatrix b, PSFloat *dest, PSMathOpts *opts);
int PSDotMV(PSMatrix a, PSFloat *b, PSFloat *dest, PSMathOpts *opts);
int PSDotVM(PSFloat *a, PSMatrix b, PSMatrix dest, PSMathOpts *opts);
int PSOuterProduct(PSFloat *a, PSFloat *b, PSFloat *dest,
                    uint64_t alen, uint64_t blen, PSMathOpts *opts);
PSMatrix PSDiagonalMask(int size);
PSMatrix PSDiagonalFlatten(PSMatrix matrix);
PSMatrix PSDiagonalFlattenVector(PSFloat *vec, uint64_t len);
PSFloat **PSVectorSplit(PSFloat *vec, int len, int num_slices);
PSFloat *PSVectorDup(PSFloat *src, size_t length);
PSFloat *PSVectorRandom(size_t len);
int PSFloatEquals(PSFloat a, PSFloat b, int precision);
int PSVectorEquals(PSFloat *a, PSFloat *b, uint64_t length, int precision,
                   uint64_t *index);

#endif /* __PS_MATHS_H__ */

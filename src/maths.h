/*
 * Copyright (C) 2016-2024 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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
#define PSLog(v) log(v)
#define PSLog10(v) log10(v)
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
#define PSLog(v) logf(v)
#define PSLog10(v) log10f(v)
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

#define PSMatrixDimensions(matrix, shape) PSMatrixShape(matrix, shape)

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
typedef void (*PSDotProductDebug)(long i, PSFloat a, PSFloat b, PSFloat sum,
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
 *                    dest.
 *  - `transpose`:  some operations involving PSMatrix could use this in order
 *                  to transpose one or more matrices. The integer value
 *                  indicates the (1-based) matrix argument position, ie.
 *                  1 for first matrix arg, 2 for second matrix arg, etc.
 *                  More than one matrix can be set (ie. 1 | 2).
 *  - `argtype`:    specifies if arguments are `PSMatrix` or `PSFloat *`
 *                  (vector). Functions using this property (such as `PSDot`),
 *                  must have PSMatrix arguments and the property can be used
 *                  to tell the function that one or more arguments must be
 *                  treated as vectors.
 *                  - 'M' or 'm': `PSMatrix` matrix
 *                  - 'V' or 'v': `PSFloat *` vector
 *                  The index indicated argument position (zero-based), ie:
 *                  argtype[1] means that second PSMatrix argument has to be
 *                  treated as vector.
 *  - `vector_len`: optionally pass vector length to functions that cannot
 *                  retrieve this info from matrix arguments, ie. when `PSDot`
 *                  is called with both vectors (`argtype` = {'V', 'V'})
 *  - `tmpdest`:    some operations may use this vector as a cache in order
 *                  to avoid allocating extra memory, for intermediate
 *                  computations.
 *  - `debugStep`:  used for debugging by some operations.
 */
typedef struct PSMathOpts {
    int                 acceleration;
    int                 store_mode;
    int                 transpose;
    char                argtype[3];
    long                vector_len;
    PSFloat             *tmpdest;
    PSDotProductDebug   debugStep;
    void                *data;
} PSMathOpts;

/****** Utils *****/

PSFloat PSNormalizedRandom(void);
PSFloat PSGaussianRandom(PSFloat mean, PSFloat stddev);
long PSRandomInt(long range, PSFloat *weights, PSMathOpts *opts);

/**** PSMatrix ****/

#define PSMatrixDataSize(matrix) (PSMatrixLength(matrix) * sizeof(PSFloat))
#define PSMatrixStrideBytes(matrix,i) \
    (PSMatrixStride(matrix,i) * sizeof(PSFloat))

/* Basically, `PSMatrix` can be used as a normal array of `PSFloat`
 * numbers. However, PSMatrix objects created by using the specific functions
 * (`PSMatrixCreate`, `PSMatrixZeros`, and so on) will also contain several
 * private informations about the matrix itself that allow them to be used as
 * multidimensional matrices.
 * Therefore, by using the PSMatrix-related functions provided by PsyC's API,
 * it's possible to get info about the matrix (length, shape, ...) or to
 * perform several operations (ie. transposition, matrix multiplication, etc.)
 * on them.
 * WARN: matrix's private data are actually allocated just before the memory
 * address pointed by `PSMatrix`, so the matrix object should **NEVER** be
 * freed by calling the usual `free` function or similar functions: the
 * dedicated `PSMatrixFree` function should be called instead. */
typedef PSFloat *PSMatrix;
typedef PSFloat (*PSMatrixInitializer)(PSMatrix matrix, long idx, PSFloat n);
PSMatrix PSMatrixCreate(PSFloat init_value, PSMatrixInitializer initializer,
                        int ndims, ...);
PSMatrix PSMatrixCreateWithShape(PSFloat init_value,
                                 PSMatrixInitializer initializer,
                                 int ndims, long *shape);
PSMatrix PSMatrixZeros(int ndims, ...);
PSMatrix PSMatrixRandom(int ndims, ...);
PSMatrix PSMatrixWithGaussianRandom(PSFloat stddev, int ndims, ...);
PSMatrix PSMatrixFromArray(PSFloat *array, int ndims, ...);
PSMatrix PSMatrixExpand(PSMatrix src, long add, int keep_src);
int PSMatrixNumDims(PSMatrix matrix);
long PSMatrixDim(PSMatrix matrix, int dim);
int PSMatrixShape(PSMatrix matrix, long *shape);
long PSMatrixLength(PSMatrix matrix);
long PSMatrixStride(PSMatrix matrix, int dim);
int PSMatrixShapeType(PSMatrix matrix);
void PSMatrixPrintInfo(PSMatrix matrix, const char *name, int newline);
void PSMatrixPrintShape(PSMatrix matrix, int newline);
size_t PSMatrixWrite(PSMatrix matrix, const char *sep, char bracket,
                     int indent, FILE *out);
void PSMatrixPrint(PSMatrix matrix, const char *sep, int print_shape);
PSFloat *PSMatrixGet(PSMatrix matrix, int ndims, long *len, ...);
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixProductMV(PSMatrix a, PSFloat *b, long len, PSFloat **result,
                      PSMathOpts *opts);
int PSMatrixProductVM(PSFloat *a, PSMatrix b, long len, PSMatrix *result,
                      PSMathOpts *opts);
int PSMatrixAdd(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixMultiply(PSMatrix a, PSMatrix b, PSMatrix *result,PSMathOpts *opt);
int PSMatrixSubtract(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
int PSMatrixDivide(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt);
PSMatrix PSMatrixReshape(PSMatrix matrix, int num_dims, ...);
PSMatrix PSMatrixFlatten(PSMatrix matrix);
PSMatrix *PSMatrixSplit(PSMatrix matrix, long num_slices, int axis,
                        PSMathOpts *opts);
PSMatrix PSMatrixTranspose(PSMatrix matrix, int rebuild, PSMathOpts *opts);
int PSMatrixHasTransposedVersion(PSMatrix matrix);
int PSMatrixIsTransposedVersion(PSMatrix matrix);
PSMatrix PSMatrixSwapAxes(PSMatrix matrix, int axis1, int axis2);
void PSMatrixResetTransposed(PSMatrix matrix);
PSMatrix PSMatrixDup(PSMatrix matrix);
PSMatrix PSMatrixDupShape(PSMatrix matrix);
int PSMatrixCopy(PSMatrix src, PSMatrix dst);
int PSMatrixEquals(PSMatrix a, PSMatrix b, int precision, int ignore_shape);
void PSMatrixClear(PSMatrix matrix);
void PSMatrixFree(PSMatrix matrix);

/**** Operations ***/

PSFloat *PSAddVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                      PSMathOpts *opts);
PSFloat *PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                           PSMathOpts *opts);
PSFloat *PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                           PSMathOpts *opts);
PSFloat *PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                         PSMathOpts *opts);
PSFloat *PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                long length, PSMathOpts *opts);
PSFloat *PSAddVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest, long length,
                           PSMathOpts *opts);
PSFloat *PSSubtractVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                long length, PSMathOpts *opts);
PSFloat *PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                                long length, PSMathOpts *opts);
PSFloat *PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                              long length, PSMathOpts *opts);
PSFloat *PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                              long length, PSMathOpts *opts);
PSFloat *PSVectorTanh(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorSqrt(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorExp(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorNeg(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorAbs(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                      long length, PSMathOpts *opts);
PSFloat *PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                           long length, PSMathOpts *opts);
PSFloat *PSVectorMapWithLimit(PSFloat *a, PSFloat limit, PSFloat mapper,
                              PSFloat *dest, long length, PSMathOpts *opts);
PSFloat *PSVectorPower(PSFloat *a, PSFloat exp, PSFloat *dest, long length,
                       PSMathOpts *opts);
PSFloat PSVectorMax(PSFloat *a, long *index, long length, PSMathOpts *opts);
PSFloat PSVectorReduceSum(PSFloat *a, long length, PSMathOpts *opts);
long PSCumulativeSum(PSFloat *a, PSFloat *dest, long length);
PSFloat PSMean(PSFloat *a, long length, PSMathOpts *opts);
PSFloat PSVariance(PSFloat *a, long len, PSMathOpts *opts);
PSFloat PSStdDev(PSFloat *a, long len, PSMathOpts *opts);
PSFloat PSDotProduct(PSFloat *a, PSFloat *b, long length, PSMathOpts *opts);
PSFloat PSDotSquare(PSFloat *a, long length, PSMathOpts *opts);
void PSVectorFill(PSFloat *vec, PSFloat val, long len, PSMathOpts *opts);
size_t PSVectorWrite(PSFloat *vec, long len, char* sep, FILE *f);
void PSVectorPrint(PSFloat *vec, long len, char* sep);
PSFloat *PSVectorTranspose(PSFloat *vec, PSFloat *dest, int acceleration,
                           int ndims, ...);
int PSMatMul(PSFloat *a, PSFloat *b, PSFloat *dest, long m, long n, long k,
             PSMathOpts *opts);
int PSDot(PSMatrix a, PSMatrix b, PSFloat *dest, PSMathOpts *opts);
int PSDotMV(PSMatrix a, PSFloat *b, PSFloat *dest, PSMathOpts *opts);
int PSDotVM(PSFloat *a, PSMatrix b, PSMatrix dest, PSMathOpts *opts);
int PSOuterProduct(PSFloat *a, PSFloat *b, PSFloat *dest,
                   long alen, long blen, PSMathOpts *opts);
PSMatrix PSDiagonalMask(long size);
PSMatrix PSDiagonalFlatten(PSMatrix matrix);
PSMatrix PSDiagonalFlattenVector(PSFloat *vec, long len);
PSFloat **PSVectorSplit(PSFloat *vec, long len, long num_slices);
PSFloat *PSVectorDup(PSFloat *src, long length);
PSFloat *PSVectorRandom(long len);
int PSFloatEquals(PSFloat a, PSFloat b, int precision);
int PSVectorEquals(PSFloat *a, PSFloat *b, long length, int precision,
                   long *index);
PSMatrix PSVectorConvertToMatrix(PSFloat *vec, long len, int ndims,
                                 long *shape);

#endif /* __PS_MATHS_H__ */

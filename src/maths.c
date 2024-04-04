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

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <assert.h>

#include "platform.h"
#include "config.h"
#include "maths.h"
#include "blas.h"
#include "log.h"

#ifdef USE_AVX
#include "avx.h"
#endif

#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#include <Accelerate/Accelerate.h>

#ifdef PS_DOUBLE_PRECISION

#define VDSPAddV(a,b,dest,len) vDSP_vaddD(a, 1, b, 1, dest, 1, len)
#define VDSPSubV(a,b,dest,len) vDSP_vsubD(b, 1, a, 1, dest, 1, len)
#define VDSPMulV(a,b,dest,len) vDSP_vmulD(a, 1, b, 1, dest, 1, len)
#define VDSPMulAddV(a,b,c,d,len) vDSP_vmaD(a, 1, b, 1, c, 1, d, 1, len)
#define VDSPDivV(a,b,dest,len) vDSP_vdivD(b, 1, a, 1, dest, 1, len)
#define VDSPMulVS(a,b,dest,len) vDSP_vsmulD(a, 1, &b, dest, 1, len)
#define VDSPAddVS(a,b,dest,len) vDSP_vsaddD(a, 1, &b, dest, 1, len)
#define VDSPDivVS(a,b,dest,len) vDSP_vsdivD(a, 1, &b, dest, 1, len)
#define VDSPDivSV(a,b,dest,len) vDSP_svdivD(&a, b, 1, dest, 1, len)
#define VDSPVSquare(a, dest, len) vDSP_vsqD(a, 1, dest, 1, len)
#define VDSPNeg(a,dest,len) vDSP_vnegD(a, 1, dest, 1, len)
#define VDSPAbs(a,dest,len) vDSP_vabsD(a, 1, dest, 1, len)
#define VDSPDotProd(a,b,dest,len) vDSP_dotprD(a, 1, b, 1, &dest, len)
#define VDSPSumVecSqr(a,dest,len) vDSP_svesqD(a, 1, &dest, len)
#define VDSPClip(a,min,max,dest,len) vDSP_vclipD(a,1,&min,&max,dest,1,len)
#define VDSPThres(a, min, dest, len) vDSP_vthresD(a,1,&min,dest,1,len)
#define VDSPMax(a, res, len) vDSP_maxvD(a, 1, &res, len)
#define VDSPMin(a, res, len) vDSP_minvD(a, 1, &res, len)
#define VDSPMaxIdx(a, res, idx, len) vDSP_maxviD(a, 1, &res, idx, len)
#define VDSPMinIdx(a, res, idx, len) vDSP_minviD(a, 1, &res, idx, len)
#define VDSPSumElems(a, res, len) vDSP_sveD(a, 1, &res, len)
#define VDSPMTransp(a,dest,m,n) vDSP_mtransD(a, 1, dest, 1, m, n)
#define VDSPMMul(a, b, dest, m, n, p)  vDSP_mmulD(a, 1, b, 1, dest, 1, m, n, p)
#define VDSPVLim(a, limit, i, dest, len) vDSP_vlimD(a, 1, &limit, &i,\
    dest, 1, len)
#define VDSPMean(a, res, len) vDSP_meanvD(a, 1, &res, len)
#define VDSPFill(val, a, len) vDSP_vfillD(&val, a, 1, len)
#define VVSqrt(a,dest,len) vvsqrt(dest, a, (int *)&len)
#define VVTanh(a,dest,len) vvtanh(dest, a, (int *)&len)
#define VVExp(a,dest,len)  vvexp(dest, a, (int *)&len)
#define VVLog(a,dest,len)  vvlog(dest, a, (int *)&len)
#define VVPow(a,exp,dest,len) vvpow(dest, exp, a, (int *)&len)

#else

#define VDSPAddV(a,b,dest,len) vDSP_vadd(a, 1, b, 1, dest, 1, len)
#define VDSPSubV(a,b,dest,len) vDSP_vsub(b, 1, a, 1, dest, 1, len)
#define VDSPMulV(a,b,dest,len) vDSP_vmul(a, 1, b, 1, dest, 1, len)
#define VDSPMulAddV(a,b,c,d,len) vDSP_vma(a, 1, b, 1, c, 1, d, 1, len)
#define VDSPDivV(a,b,dest,len) vDSP_vdiv(b, 1, a, 1, dest, 1, len)
#define VDSPMulVS(a,b,dest,len) vDSP_vsmul(a, 1, &b, dest, 1, len)
#define VDSPAddVS(a,b,dest,len) vDSP_vsadd(a, 1, &b, dest, 1, len)
#define VDSPDivVS(a,b,dest,len) vDSP_vsdiv(a, 1, &b, dest, 1, len)
#define VDSPDivSV(a,b,dest,len) vDSP_svdiv(&a, b, 1, dest, 1, len)
#define VDSPVSquare(a, dest, len) vDSP_vsq(a, 1, dest, 1, len)
#define VDSPClip(a,min,max,dest,len) vDSP_vclip(a,1,&min,&max,dest,1,len)
#define VDSPThres(a, min, dest, len) vDSP_vthres(a,1,&min,dest,1,len)
#define VDSPNeg(a,dest,len) vDSP_vneg(a, 1, dest, 1, len)
#define VDSPAbs(a,dest,len) vDSP_vabs(a, 1, dest, 1, len)
#define VDSPDotProd(a,b,dest,len) vDSP_dotpr(a, 1, b, 1, &dest, len)
#define VDSPSumVecSqr(a,dest,len) vDSP_svesq(a, 1, &dest, len)
#define VDSPMax(a, res, len) vDSP_maxv(a, 1, &res, len)
#define VDSPMin(a, res, len) vDSP_minv(a, 1, &res, len)
#define VDSPMaxIdx(a, res, idx, len) vDSP_maxvi(a, 1, &res, idx, len)
#define VDSPMinIdx(a, res, idx, len) vDSP_minvi(a, 1, &res, idx, len)
#define VDSPSumElems(a, res, len) vDSP_sve(a, 1, &res, len)
#define VDSPMTransp(a,dest,m,n) vDSP_mtrans(a, 1, dest, 1, m, n)
#define VDSPMMul(a, b, dest, m, n, p)  vDSP_mmul(a, 1, b, 1, dest, 1, m, n, p)
#define VDSPVLim(a, limit, i, dest, len) vDSP_vlim(a, 1, &limit, &i,\
    dest, 1, len)
#define VDSPMean(a, res, len) vDSP_meanv(a, 1, &res, len)
#define VDSPFill(val, a, len) vDSP_vfill(&val, a, 1, len)
#define VVSqrt(a,dest,len) vvsqrtf(dest, a, (int *)&len)
#define VVTanh(a,dest,len) vvtanhf(dest, a, (int *)&len)
#define VVExp(a,dest,len)  vvexpf(dest, a, (int *)&len)
#define VVLog(a,dest,len)  vvlogf(dest, a, (int *)&len)
#define VVPow(a,exp,dest,len) vvpowf(dest, exp, a, (int *)&len)

#endif

#endif

#define UNUSED(V) ((void) V)
#define MATHS_OPERATION_PREAMBLE() \
    if (length <= 0) {\
        PSErr(__func__, "argument `length` must be > 0");\
        return NULL;\
    }\
    if (dest == NULL) {\
        dest = PSVectorCreate(length);\
        if (dest == NULL) {\
            PSPrintMemoryErrorMsg();\
            return NULL;\
        }\
    }\
    PSDotProductDebug debugStep = NULL;\
    long i = 0;\
    int acceleration = PSGlobalAcceleration, mode = PS_STORE_MODE_SET;\
    if (opts != NULL) {\
        acceleration = opts->acceleration;\
        mode = opts->store_mode;\
        debugStep = opts->debugStep;\
        assert(mode >= 0 && mode <= PS_STORE_MODE_SUB);\
    }\
    UNUSED(debugStep);

typedef PSFloat * (*PSOpVV) (PSFloat *a, PSFloat *b,PSFloat *res, long len,
                             PSMathOpts *opts);
typedef PSFloat * (*PSOpVS) (PSFloat *a, PSFloat b, PSFloat *res, long len,
                             PSMathOpts *opts);
typedef PSFloat * (*PSOpSV) (PSFloat a, PSFloat *b, PSFloat *res, long len,
                             PSMathOpts *opts);

/* Forward declarations and external functions */

size_t writeSerializedFloatArray(FILE *out, long count, char *sep, int opts,
                                 PSFloat *array);

/**** Utils ****/
static unsigned char randomSeeded = 0;

static void randomSeed(void) {
    if (randomSeeded) return;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srand(tv.tv_usec * tv.tv_sec);
    randomSeeded = 1;
}

/* Generate a random floating number within a range of 0.0 and 1.0.
 * Return value: the random float number. */
PSFloat PSNormalizedRandom(void) {
    randomSeed();
    int r = rand();
    return ((PSFloat) r / (PSFloat) RAND_MAX);
}

/* Generate a random floating number from a gaussian distribution having mean
 * defined by `mean` and standard deviation defined by `stddev`.
 * Return value: the random float number. */
PSFloat PSGaussianRandom(PSFloat mean, PSFloat stddev) {
    PSFloat r1 = PSNormalizedRandom(), r2 = (1 - PSNormalizedRandom());
    if (r2 <= 0) r2 = PSFLOAT_EPS;
    PSFloat theta = 2 * M_PI * r1;
    PSFloat rho = PSSqrt(-2 * PSLog(r2));
    PSFloat scale = stddev * rho;
    PSFloat x = mean + scale * cos(theta);
    PSFloat y = mean + scale * sin(theta);
    PSFloat r = PSNormalizedRandom();
    return (r > 0.5 ? y : x);
}

/* Generate a random unsigned integer number within a range defined by argument
 * `range` (between zero and `range` - 1).
 * If the optional `weights` argument is not NULL, it can be used as a
 * probability distribution the affects the randomness of the result.
 * In this case, `weights` must be an array of `PSFloat` whose length must be
 * equal to `range`: each element of `weights` represents the probability
 * (weight) of its index to be generated (for example, the weights
 * `{0.1, 0.7, 0.2}` with a range of 3 give a probability of 70% to number 1
 * to be generated).
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the random integer number or -1 in case of error. */
long PSRandomInt(long range, PSFloat *weights, PSMathOpts *opts) {
    randomSeed();
    if (range == 0) return 0;
    if (weights == NULL) {
        if (range > RAND_MAX) {
            PSFloat r = PSNormalizedRandom() * (PSFloat) range;
            return (long) r;
        } else return rand() % range;
    } else {
        PSFloat cumulated_weights[range];
        if (!PSCumulativeSum(weights, cumulated_weights, range)) return - 1;
        PSFloat last = cumulated_weights[range - 1];
        PSMathOpts mopts = {.acceleration = PSGlobalAcceleration};
        if (opts != NULL) mopts.acceleration = opts->acceleration;
        PSDivideVectorScalar(
            cumulated_weights, last, cumulated_weights, range, &mopts
        );
        PSFloat r = PSNormalizedRandom();
        long idx = 0;
        for (; idx < range; idx++) {
            if (cumulated_weights[idx] <= r) continue;
            break;
        }
        if (idx >= range) idx = range - 1;
        return idx;
    }
}

/**** PSMatrix ****/

#define PSMatrixGetHeader(matrix) \
    ((PSMatrixHeader *) getMatrixHeadPointer(matrix))

typedef struct PSMatrixHeader {
    int ndims;
    long shape[PS_MATRIX_MAX_DIMENSIONS];
    PSMatrix transposed;
    PSMatrix transposed_from;
    long length;
} PSMatrixHeader;

static const size_t PSMatrixHeaderSize = sizeof(PSMatrixHeader);

static char *getMatrixHeadPointer(PSMatrix matrix) {
    return ((char *) matrix) - PSMatrixHeaderSize;
}

static PSFloat matrixGaussianRandomInitializer(PSMatrix matrix, long idx,
                                               PSFloat stddev)
{
    UNUSED(matrix);
    UNUSED(idx);
    if (stddev == 0.0) stddev = 1.0;
    return PSGaussianRandom(0, stddev);
}

static PSFloat matrixRandomInitializer(PSMatrix matrix, long idx, PSFloat val) {
    UNUSED(matrix);
    UNUSED(idx);
    UNUSED(val);
    return PSNormalizedRandom();
}

static const char *matrixDimensionsToString(int ndims, long *dims) {
    static char dimstr[256] = {0};
    if (dims == NULL || ndims < 0 || ndims > PS_MATRIX_MAX_DIMENSIONS) {
        dimstr[0] = '\0';
        return dimstr;
    }
    int avail = 255;
    char *s = dimstr;
    for (int i = 0; i < ndims; i++) {
        if (avail <= 0) break;
        char *sep = (i > 0 ? "," : "");
        int written = snprintf(s, avail, "%s%ld", sep, dims[i]);
        s += written;
        avail -= written;
    }
    return dimstr;
}

static int getShapeType(int nd, long *shape) {
    if (nd <= 0) return PS_SHAPE_TYPE_NONE;
    else if (nd == 1) {
        if (shape[0] > 1) return PS_SHAPE_TYPE_COL;
        return PS_SHAPE_TYPE_SCALAR;
    } else if (nd == 2) {
        if (shape[0] > 1) {
            if (shape[1] == 1) return PS_SHAPE_TYPE_COL;
            else return PS_SHAPE_TYPE_MATRIX;
        }
        if (shape[1] == 1) return PS_SHAPE_TYPE_SCALAR;
        return PS_SHAPE_TYPE_ROW;
    }
    return PS_SHAPE_TYPE_MATRIX;
}

/* Create a new matrix having number of dimensions defined by `ndims` and
 * shape defined by `shape`. The argument `init_value` can be used to define
 * the initial value of the matrix numbers or, optionally, the `initializer`
 * callback can be used to initialize the matrix values.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * Return value: the allocated matrix or NULL if:
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixCreateWithShape(PSFloat init_value,
                                 PSMatrixInitializer initializer,
                                 int ndims, long *shape)
{
    if (ndims < 1 || ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d",
              PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    long len = 1;
    int i;
    for (i = 0; i < ndims; i++) {
        long d = shape[i];
        if (d <= 0) {
            if (d == 0) {
                PSWarn("%s: invalid dimension[%d] = %d", __func__, i, d);
                return NULL;
            } else {
                PSWarn("%s: invalid dimension[%d] = %d", __func__, i, d);
                ndims = i + 1;
                shape[i] = d;
                break;
            }
        }
        len *= d;
        shape[i] = d;
    }
    if (len <= 0) {
        PSErr(__func__, "Invalid dimensions: %s",
              matrixDimensionsToString(ndims, shape));
        return NULL;
    }
    size_t datasize = ((size_t) len * sizeof(PSFloat));
    size_t size = PSMatrixHeaderSize + datasize;
    char *mem = malloc(size);
    if (mem == NULL) {
        errno = ENOMEM;
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSMatrixHeader *hdr = (PSMatrixHeader *) mem;
    PSMatrix matrix = (PSMatrix)(mem + PSMatrixHeaderSize);
    hdr->transposed = NULL;
    hdr->transposed_from = NULL;
    hdr->ndims = ndims;
    for (i = 0; i < PS_MATRIX_MAX_DIMENSIONS; i++) {
        if (i < ndims) hdr->shape[i] = shape[i];
        else hdr->shape[i] = 0;
    }
    hdr->length = len;
    if (initializer != NULL) {
        for (i = 0; i < len; i++)
            matrix[i] = initializer(matrix, i, init_value);
    } else if (init_value == 0.0) {
        memset(matrix, 0, datasize);
    } else {
        for (i = 0; i < len; i++) matrix[i] = init_value;
    }
    return matrix;
}

PSMatrix PSMatrixCreateV(PSFloat init_value, PSMatrixInitializer initializer,
                         int ndims, va_list args)
{
    if (ndims < 1 || ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d",
              PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    long shape[PS_MATRIX_MAX_DIMENSIONS];
    for (int i = 0; i < ndims; i++) {
        long d = va_arg(args, long);
        shape[i] = d;
    }
    return PSMatrixCreateWithShape(init_value, initializer, ndims, shape);
}

/* Create a new matrix having number of dimensions defined by `ndims`. The
 * shape of the matrix is given by variadic arguments that follow `ndims`.
 * The argument `init_value` can be used to define the initial value of the
 * matrix numbers or, optionally, the `initializer` callback can be used to
 * initialize the matrix values.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * Return value: the allocated matrix or NULL if:
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixCreate(PSFloat init_value, PSMatrixInitializer initializer,
                        int ndims, ...)
{
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(init_value, initializer, ndims, args);
    va_end(args);
    return matrix;
}

/* Create a new, zero-filled, matrix having number of dimensions defined by
 * `ndims`. The shape of the matrix is given by variadic arguments that follows
 * `ndims`.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * Return value: the allocated matrix or NULL if:
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixZeros(int ndims, ...) {
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0.0, NULL, ndims, args);
    va_end(args);
    return matrix;
}

/* Create a new matrix having number of dimensions defined by `ndims`. The
 * shape of the matrix is given by variadic arguments that follow `ndims`.
 * The values of the matrix will be initialized with random numbers from a
 * gaussian distribution having zero mean and the standard deviation defined
 * by `stddev`.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * Return value: the allocated matrix or NULL if:
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixWithGaussianRandom(PSFloat stddev, int ndims, ...) {
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(
        stddev, matrixGaussianRandomInitializer, ndims, args
    );
    va_end(args);
    return matrix;
}

/* Create a new matrix having number of dimensions defined by `ndims`. The
 * shape of the matrix is given by variadic arguments that follow `ndims`.
 * The values of the matrix will be initialized with random numbers from 0.0
 * to 1.0.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * Return value: the allocated matrix or NULL if:
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixRandom(int ndims, ...) {
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0, matrixRandomInitializer, ndims, args);
    va_end(args);
    return matrix;
}

/* Create a new matrix having number of dimensions defined by `ndims`. The
 * shape of the matrix is given by variadic arguments that follow `ndims`.
 * The values of the matrix will be initialized with values of `array`.
 * If the matrix cannot be allocated, `errno` will be set to `ENOMEM`.
 * WARN: the length of `array` must be at least the same of the length of the
 * matrix, so if the matrix has two dimensions of shape [2, 3] (two rows with
 * three columns), the provided array's length cannot be less than six.
 * Return value: the allocated matrix or NULL if:
 *  - `array` is NULL.
 *  - the number of dimensions (`ndims`) is greater than
 *    `PS_MATRIX_MAX_DIMENSIONS` or less than one.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixFromArray(PSFloat *array, int ndims, ...) {
    if (array == NULL) return NULL;
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0, NULL, ndims, args);
    va_end(args);
    if (matrix == NULL) return NULL;
    size_t array_size = PSMatrixLength(matrix) * sizeof(PSFloat);
    memcpy(matrix, array, array_size);
    return matrix;
}

/* Duplicate `matrix` by creating a new matrix having the same shape as
 * `matrix` and by copying all values of `matrix` to the new matrix.
 * Return value: the new matrix or NULL if:
 *  - `matrix` is NULL.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixDup(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    long len = PSMatrixLength(matrix);
    size_t datasize = ((size_t) len * sizeof(PSFloat));
    size_t size = PSMatrixHeaderSize + datasize;
    PSMatrixHeader *dst_hdr = malloc(size);
    if (dst_hdr == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSMatrix clone = (PSMatrix) (dst_hdr + 1);
    PSMatrixHeader *src_hdr = PSMatrixGetHeader(matrix);
    memcpy(dst_hdr, src_hdr, PSMatrixHeaderSize);
    dst_hdr->transposed = NULL;
    dst_hdr->transposed_from = NULL;
    memcpy(clone, matrix, datasize);
    return clone;
}

/* Create a new (zero-filled) matrix having the same shape as `matrix`.
 * Return value: the new matrix or NULL if:
 *  - `matrix` is NULL.
 *  - it's not possible to allocate the matrix in memory.
 * WARN: the address pointed by the returned pointer should never be freed
 * directly. The specific function `PSMatrixFree` should be used instead. */
PSMatrix PSMatrixDupShape(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    long len = PSMatrixLength(matrix);
    size_t datasize = ((size_t) len * sizeof(PSFloat));
    size_t size = PSMatrixHeaderSize + datasize;
    PSMatrixHeader *dst_hdr = malloc(size);
    if (dst_hdr == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSMatrix clone = (PSMatrix) (dst_hdr + 1);
    PSMatrixHeader *src_hdr = PSMatrixGetHeader(matrix);
    memcpy(dst_hdr, src_hdr, PSMatrixHeaderSize);
    dst_hdr->transposed = NULL;
    dst_hdr->transposed_from = NULL;
    memset(clone, 0, datasize);
    return clone;
}

/* Copy values of matrix `src` to matrix `dst`. Both `src` and `dst` must have
 * the same shape.
 * NOTE: if `dst` owns a cached transposed version of itself, the cached
 * version will be cleared. At the same time, if `dst` is the cached transposed
 * version of another matrix, the cached version of the owner matrix will be
 * cleared.
 * Return value: 1 in case of success, 0 if:
 *  - `src` is NULL or `dst` is NULL.
 *  - `src` and `dst` have different shapes. */
int PSMatrixCopy(PSMatrix src, PSMatrix dst) {
    if (src == NULL) {
        PSErr(__func__, "`src` matrix is NULL");
        return 0;
    }
    if (dst == NULL) {
        PSErr(__func__, "`dst` matrix is NULL");
        return 0;
    }
    long src_shape[PS_MATRIX_MAX_DIMENSIONS];
    long dst_shape[PS_MATRIX_MAX_DIMENSIONS];
    int src_ndims = PSMatrixShape(src, src_shape),
        dst_ndims = PSMatrixShape(dst, dst_shape);
    long src_len = PSMatrixLength(src), i;
    if (src_ndims != dst_ndims) {
        PSErr(__func__, "`src` dimensions != `dst` dimensions: %d != %d",
              src_ndims, dst_ndims);
        return 0;
    }
    for (i = 0; i < src_ndims; i++) {
        if (src_shape[i] != dst_shape[i]) {
            PSErr(
                __func__, "`src` shape[%d] != `dst`: %d != %d",
                i, src_shape[i], dst_shape[i]
            );
            return 0;
        }
    }
    PSMatrixHeader *dst_hdr = PSMatrixGetHeader(dst);
    if (dst_hdr->transposed_from != NULL) {
        PSMatrixHeader *owner_hdr = PSMatrixGetHeader(dst_hdr->transposed_from);
        owner_hdr->transposed = NULL;
        dst_hdr->transposed_from = NULL;
    }
    if (dst_hdr->transposed != NULL) {
        PSMatrixFree(dst_hdr->transposed);
        dst_hdr->transposed = NULL;
    }
    memcpy(dst, src, src_len * sizeof(PSFloat));
    return 1;
}

/* Set all values of `matrix` to zero. If `matrix` is NULL, the function does
 * nothing at all.*/
void PSMatrixClear(PSMatrix matrix) {
    if (matrix == NULL) return;
    PSVectorClear(matrix, PSMatrixLength(matrix));
}

/* Create a new matrix having the shape of `src` but with the first dimension
 * increased by the value of `add`. The original values `src` will be copied to
 * the new matrix, and all the new values belonging to thecexpanded dimension
 * will be initialized to zero.
 * If `keep_src` is zero, the original matrix `src` will be freed.
 * Return value: the new expanded matrix or:
 *  - `src` itself if `add` is less that one.
 *  - NULL if `src` is NULL.
 *  - NULL if memory cannot be allocated.
 */
PSMatrix PSMatrixExpand(PSMatrix src, long add, int keep_src) {
    if (src == NULL) return NULL;
    if (add <= 0) return src;
    PSMatrix matrix = NULL;
    PSMatrixHeader *src_hdr = PSMatrixGetHeader(src);
    if (src_hdr->transposed_from != NULL) {
        matrix = PSMatrixExpand(src_hdr->transposed_from, add, keep_src);
        if (matrix == NULL) return NULL;
        PSMathOpts opts = {.acceleration = PSGlobalAcceleration};
        return PSMatrixTranspose(matrix, 1, &opts);
    }
    long new_dims[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixShape(src, new_dims);
    long curlen = PSMatrixLength(src);
    new_dims[0] += add;
    matrix = PSMatrixCreateWithShape(0.0, NULL, ndims, new_dims);
    if (matrix == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSVectorCopy(matrix, src, curlen);
    if (!keep_src) PSMatrixFree(src);
    return matrix;
}

/* Return the number of dimensions of `matrix`. If `matrix` is NULL, the
 * function will return zero. */
int PSMatrixNumDims(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->ndims;
}

/* Return the size of the dimension `dim` of `matrix`. If `dim` is out of
 * bounds or if `matrix` is NULL, the function will return zero. */
long PSMatrixDim(PSMatrix matrix, int dim) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (dim >= hdr->ndims) return 0;
    return hdr->shape[dim];
}

/* Get the shape of `matrix` and store it into `shape` array. The `shape`
 * array must be big enough to hold at least `PS_MATRIX_MAX_DIMENSIONS`
 * elements.
 * If `shape` is NULL, the function will just return the number of dimensions
 * (so, the length of the shape array of `matrix`).
 * Return value: the number of dimensions of `matrix` or zero if `matrix` is
 * NULL. */
int PSMatrixShape(PSMatrix matrix, long *shape) {
    if (matrix == NULL) return 0;
    int ndims = PSMatrixNumDims(matrix);
    if (shape == NULL) return ndims;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    memcpy(shape, hdr->shape, (PS_MATRIX_MAX_DIMENSIONS * sizeof(long)));
    return ndims;
}

/* Get the total number of values belonging to `matrix` (ie. a matrix with
 * shape (2,3) will return 6).
 * Return value: the total number of values belonging to `matrix` or zero if
 * `matrix` is NULL. */
long PSMatrixLength(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->length;
}

/* Get the stride of the dimension `dim` of `matrix`. For example, a matrix
 * with shape (2,3) has a stride of 3 for dimension 0 while a matrix with shape
 * (2,3,3) has a stride of 9 for dimension 0, 3 for dimension 1 and 1 for
 * dimension 2.
 * Return value: the stride of dimension `dim` or zero if `matrix` is NULL. */
long PSMatrixStride(PSMatrix matrix, int dim) {
    if (matrix == NULL) return 0;
    int refdim = dim + 1;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (refdim >= PS_MATRIX_MAX_DIMENSIONS || refdim >= hdr->ndims) return 1;
    long stride = 1;
    while (refdim < hdr->ndims) stride *= hdr->shape[refdim++];
    return stride;
}

/* Get the shape type of `matrix`.
 * Return value: the shape type:
 *  - `PS_SHAPE_TYPE_NONE` if:
 *    - `matrix` is NULL.
 *    - Matrix's shape has zero dimensions.
 *  - `PS_SHAPE_TYPE_SCALAR` if:
 *    - Matrix's shape has one dimension of size 1.
 *    - Matrix's shape has two dimensions, both of size 1.
 *  - `PS_SHAPE_TYPE_COL` if:
 *    - Matrix's shape has one dimension of size greater than 1.
 *    - Matrix's shape has two dimensions and the first dimension is greater
 *      than 1 but the second dimension is 1.
 *  - `PS_SHAPE_TYPE_ROW` if:
 *    - Matrix's shape has two dimensions and the first dimension is 1 but
 *      the second dimension is greater than 1.
 *  - `PS_SHAPE_TYPE_MATRIX`if:
 *    - All other cases. */
int PSMatrixShapeType(PSMatrix matrix) {
    if (matrix == NULL) return PS_SHAPE_TYPE_NONE;
    long dims[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixShape(matrix, dims);
    return getShapeType(ndims, dims);
}

void PSMatrixPrintInfo(PSMatrix matrix, const char *name, int newline) {
    if (matrix == NULL) return;
    if (name == NULL) name = "(unnamed)";
    char *nl = "";
    if (newline) nl = "\n";
    if (matrix == NULL) {
        printf("Matrix %s = (null)%s", name, nl);
        return;
    }
    long dims[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixShape(matrix, dims);
    printf(
        "Matrix %s dimensions = %d, shape = (%s)%s",
        name, ndims, matrixDimensionsToString(ndims, dims), nl
    );
}

void PSMatrixPrintShape(PSMatrix matrix, int newline) {
    if (matrix == NULL) return;
    long shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int nd = PSMatrixShape(matrix, shape), i;
    if (nd > PS_MATRIX_MAX_DIMENSIONS) {
        PSWarn("%s: invalid matrix", __func__);
        return;
    }
    for (i = 0; i < nd; i++) printf("%s%ld", (i > 0 ? "," : ""), shape[i]);
    if (newline) printf("\n");
}

/* Write the string representation of `matrix` to file file stream `out`.
 * If `out` is NULL, the string will be printed to the standard output by
 * default.
 * The optional `sep` argument can be used to specify the separator string
 * for matrix's values: if NULL, the default separator is a comma (,).
 * The optional `bracket` argument can be used to specify the type of
 * brackets enclosing matrix's values, and only the opening bracket is
 * accepted as a valid value:
 *  - '[' to use '[' as opening bracket and ']' as closing bracket.
 *  - '(' to use '(' as opening bracket and ')' as closing bracket.
 *  - '{' to use '{' as opening bracket and '}' as closing bracket.
 * If `bracket` is set to zero, the default bracket is '['. Other values for
 * `bracket` won't be accepted.
 * The `indent` argument can be used to set indentation size (expressed in
 * number of white spaces). If set to zero, no indentation will be used and
 * the matrix string will be written in a single line.
 * Return value: the total number of bytes written or 0 if:
 *  - `matrix` is NULL.
 *  - Invalid value for `bracket` (see above). */
size_t PSMatrixWrite(PSMatrix matrix, const char *sep, char bracket,
                     int indent, FILE *out)
{
    if (matrix == NULL) return 0;
    if (sep == NULL) sep = ",";
    if (out == NULL) out = stdout;
    if (bracket == 0) bracket = '[';
    char end_bracket = 0;
    if (bracket == '[') end_bracket = ']';
    else if (bracket == '{') end_bracket = '}';
    else if (bracket == '(') end_bracket = ')';
    else {
        PSErr(__func__, "invalid bracket '%c'", bracket);
        return 0;
    }
    long shape[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixShape(matrix, shape), i, j, k;
    int last_dim = ndims - 1;
    long index[PS_MATRIX_MAX_DIMENSIONS] = {0};
    long prev_index[PS_MATRIX_MAX_DIMENSIONS];
    for (j = 0; j < ndims; j++) prev_index[j] = -1;
    long strides[PS_MATRIX_MAX_DIMENSIONS] = {0};
    for (j = 0; j < ndims; j++) {
        strides[j] = 1;
        for (k = j + 1; k < ndims; k++) strides[j] *= shape[k];
    }
    long len = PSMatrixLength(matrix), nwritten = 0;
    char *nl = (indent > 0 ? "\n" : "");
    for (i = 0; i < len; i++) {
        PSFloat val = matrix[i];
        for (j = 0; j < ndims; j++) {
            index[j] = (i / strides[j]) % shape[j];
            if (j < last_dim && index[j] == 0 && index[j] != prev_index[j]) {
                nwritten += fprintf(out, "%*c%s", (j * indent) + 1, bracket,nl);
            }
        }
        if (index[last_dim] == 0) {
            if (indent > 0 && ndims > 1) {
                nwritten += fprintf(out, "%-*c%c", last_dim * indent,' ',
                                    bracket);
            } else nwritten += fprintf(out, "%c", bracket);
        } else nwritten += fprintf(out, "%s", sep);
        nwritten += fprintf(out, "%g", val);
        if (index[last_dim] == shape[last_dim] - 1) {
            const char *row_sep = sep;
            if (last_dim > 0) {
                long last_row_idx = shape[last_dim - 1] - 1;
                if (index[last_dim - 1] == last_row_idx) row_sep = "";
            } else if (ndims == 1) row_sep = "";
            nwritten += fprintf(out, "%c%s%s", end_bracket, row_sep, nl);
            for (j = last_dim - 1; j >= 0; j--) {
                long last_idx = shape[j] - 1;
                row_sep = (i == len - 1 ? "" : sep);
                if (index[j] == last_idx) {
                    nwritten += fprintf(
                        out, "%*c%s%s", (j * indent) + 1, end_bracket,
                        row_sep, nl
                    );
                } else break;
            }
        }
        memcpy(prev_index, index, PS_MATRIX_MAX_DIMENSIONS * sizeof(long));
    }
    if (indent > 0) printf("\n");
    return nwritten;
}

/* Print a string representation of `matrix` to the standard output.
 * The optional argument `sep` can be used to specify the separator string for
 * matrix values.
 * If `sep` is NULL, the default separator is a comma (',').
 * If `print_shape` is true, the matrix representation will be preceded
 * by a header describing the shape of `matrix`.
 * If `matrix` is NULL, the function will immediately return. */
void PSMatrixPrint(PSMatrix matrix, const char *sep, int print_shape) {
    if (matrix == NULL) return;
    if (print_shape) {
        long shape[PS_MATRIX_MAX_DIMENSIONS];
        int ndims = PSMatrixShape(matrix, shape);
        printf("Matrix (shape = %s):\n",matrixDimensionsToString(ndims,shape));
    }
    PSMatrixWrite(matrix, sep, '[', 2, stdout);
}

PSFloat *PSMatrixGet(PSMatrix matrix, int ndims, long *len, ...) {
    if (matrix == NULL) return NULL;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    PSFloat *values = matrix;
    long stride = 1;
    if (ndims > hdr->ndims) ndims = hdr->ndims;
    va_list args;
    va_start(args, len);
    for (int i = 0; i < ndims; i++) {
        int refdim = i + 1;
        if (refdim >= hdr->ndims) stride = 1;
        else stride = PSMatrixStride(matrix, i);
        long idx = va_arg(args, long);
        if (idx >= hdr->shape[i]) {
            PSWarn("%s: index %ld is out of bounds for axis[%d] (%ld)",
                   __func__, idx, i, hdr->shape[i]);
            values = NULL;
            stride = 0;
            break;
        }
        values += (idx * stride);
    }
    va_end(args);
    if (len != NULL) *len = stride;
    return values;
}

static int genericMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *out,
                                PSMathOpts *opts)
{
    PSMathOpts dfopts = {.acceleration = PSGlobalAcceleration};
    if (opts == NULL) opts = &dfopts;
    int success = 1;
    if (out == NULL) {
        PSErr(__func__, "`out` is null");
        return 0;
    }
    if (a == NULL) {
        PSErr(__func__, "`a` is null");
        return 0;
    }
    if (b == NULL) {
        PSErr(__func__, "`b` is null");
        return 0;
    }
    if (opts->transpose & 1) {
        a = PSMatrixTranspose(a, 0, opts);
        if (a == NULL) return 0;
    }
    if (opts->transpose & 2) {
        b = PSMatrixTranspose(b, 0, opts);
        if (a == NULL) return 0;
    }
    long shape_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
    long shape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
    long len = 0;
    int ndims_a = PSMatrixShape(a, shape_a);
    int ndims_b = PSMatrixShape(b, shape_b);
    if (ndims_a == 0 || ndims_b == 0) {
        /* `a` or `b` is scalar */
        PSMatrix ma, mb;
        if (ndims_a == 0) {
            ma = b;
            mb = a;
        } else {
            ma = a;
            mb = b;
        }
        len = PSMatrixLength(ma);
        if (*out != NULL) {
            long reslen = PSMatrixLength(*out);
            if (reslen != len) {
                PSErr(__func__, "`out` length != expected: %llu != %llu",
                      reslen, len);
                return 0;
            }
        } else {
            *out = PSMatrixDup(ma);
            if (*out == NULL) return 0;
        }
        PSMultiplyVectorScalar(ma, mb[0], *out, len, opts);
        return 1;
    }
    long l = shape_a[ndims_a - 1];
    int refdim;
    if (ndims_b > 1) refdim = ndims_b - 2;
    else refdim = 0;
    if (shape_b[refdim] != l) {
        PSErr(
            __func__, "Aligment error: `b` dim[%d] != `a` dim[%d] -> %d != %d",
            /*"Matrix a shape: %d,%d %s\n"
            "Matrix b shape: %d,%d %s",*/
            refdim, ndims_a - 1, shape_b[refdim], l/*,
            dims_a[0], dims_a[last_dim_a], (transpose_a ? "(transp.)" : ""),
            dims_b[0], dims_b[last_dim_b], (transpose_b ? "(transp.)" : "")*/
        );
        return 0;
    }
    int nd = ndims_a + ndims_b - 2;
    if (nd > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "result would exceed max dimensions (%d): %d",
              nd, PS_MATRIX_MAX_DIMENSIONS);
        return 0;
    }
    int i, j = 0;
    long niter_a = 1, niter_b = 1;
    long dimensions[PS_MATRIX_MAX_DIMENSIONS * 3] = {0};
    for (i = 0; i < (ndims_a - 1); i++) {
        if (i < (ndims_a - 1)) niter_a *= shape_a[i];
        dimensions[j++] = shape_a[i];
    }
    for (i = 0; i < (ndims_b - 2); i++) dimensions[j++] = shape_b[i];
    if (ndims_b > 1) dimensions[j++] = shape_b[ndims_b - 1];
    niter_b = PSMatrixLength(b) / l;
    if (*out != NULL) {
        long out_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int out_ndims = PSMatrixShape(*out, out_shape);
        if (out_ndims != nd) {
            PSErr(
                __func__, "`out` matrix has %d dimension(s), but "
                "%d dimension(s) needed", out_ndims, nd
            );
            return 0;
        }
        for (i = 0; i < nd; i++) {
            long odim = out_shape[i];
            if (odim != dimensions[i]) {
                PSErr(
                    __func__, "`out` matrix dimension [%d] is %ld, "
                    "but it should be %ld\nResult shape: %ld,%ld",
                    i, odim, dimensions[i]
                );
                return 0;
            }
        }
    } else {
        *out = PSMatrixCreateWithShape(0, NULL, nd, dimensions);
        if (*out == NULL) return 0;
    }
    long outlen = PSMatrixLength(*out);
    PSFloat *out_p = *out, *ap = a, *bp = NULL;
    PSMatrix swap_b = NULL;
    if (ndims_b > 1) {
        b = PSMatrixTranspose(b, 0, opts);
        if (b == NULL) return 0;
        long blen = PSMatrixDim(b, 0);
        if (ndims_b > 2) {
            if ((ndims_b - 1) > 2) {
                PSErr(NULL, "unsupported dimensions for matrix `b`");
                return 0;
            }
            swap_b = PSMatrixDupShape(b);
            if (swap_b == NULL) return 0;
            long bs = PSMatrixStride(swap_b, 0);
            long tshape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
            PSMatrixShape(b, tshape_b);
            for (i = 0; i < blen; i++) {
                long offset = i * bs;
                PSFloat *src = b + offset, *dst = swap_b + offset;
                dst = PSVectorTranspose(
                    src, dst, opts->acceleration, 2,
                    tshape_b[ndims_b - 2], tshape_b[ndims_b - 1]
                );
                success = (dst != NULL);
                if (!success) goto final;
            }
            b = swap_b;
            PSMatrixShape(b, tshape_b);
            long last_db = tshape_b[ndims_b - 1];
            tshape_b[ndims_b - 1] = tshape_b[ndims_b - 2];
            tshape_b[ndims_b - 2] = last_db;
            PSMatrixHeader *hdr = PSMatrixGetHeader(b);
            memcpy(
                hdr->shape, tshape_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(long)
            );
            l = tshape_b[ndims_b - 1];
            long as = PSMatrixStride(a, 0), k;
            for (i = 0; i < niter_a; i++) {
                bp = b;
                for (k = 0; k < tshape_b[ndims_b - 2]; k++) {
                    for (j = 0; j < blen; j++) {
                        bp = (b + (j * bs)) + (l * k);
                        assert((long)(out_p - *out) <= outlen);
                        *(out_p++) = PSDotProduct(ap, bp, l, opts);
                    }
                }
                ap += as;
            }
            goto final;
        }
    }
    for (i = 0; i < niter_a; i++) {
        bp = b;
        for (j = 0; j < niter_b; j++) {
            *(out_p++) = PSDotProduct(ap, bp, l, opts);
            bp += l;
        }
        ap += l;
    }
final:
    PSMatrixFree(swap_b);
    return success;
}

static int genericMatrixOperation(PSMatrix a, PSMatrix b, PSMatrix *result,
                                  PSOpVV vvop, PSOpVS vsop, PSOpSV svop,
                                  const char *func, PSMathOpts *opt)
{
    PSMathOpts dfopts = {.acceleration = PSGlobalAcceleration};
    if (opt == NULL) opt = &dfopts;
    if (result == NULL) {
        PSErr(func, "argument result cannot be null");
        return 0;
    }
    if (a == NULL) {
        PSErr(func, "matrix `a` cannot be null");
        return 0;
    }
    if (b == NULL) {
        PSErr(func, "matrix `b` cannot be null");
        return 0;
    }
    if (func == NULL) func = __func__;
    assert(vvop != NULL);
    assert(vsop != NULL);
    if (opt->transpose & 1) {
        a = PSMatrixTranspose(a, 0, opt);
        if (a == NULL) return 0;
    }
    if (opt->transpose & 2) {
        b = PSMatrixTranspose(b, 0, opt);
        if (b == NULL) return 0;
    }
    /* Original matrix dimensions */
    long dims_a[PS_MATRIX_MAX_DIMENSIONS];
    long dims_b[PS_MATRIX_MAX_DIMENSIONS];
    /* Number of dimensions */
    int ndims_a = PSMatrixShape(a, dims_a);
    int ndims_b = PSMatrixShape(b, dims_b);
    if (ndims_a == 0) {
        PSErr(func, "Invalid matrix `a`");
        return 0;
    }
    if (ndims_b == 0) {
        PSErr(func, "Invalid matrix `b`");
        return 0;
    }
    long *shape_a = dims_a, *shape_b = dims_b, *deepest_shape = NULL;
    int deepest_nd;
    long deepest_len;
    PSMatrix deepest_matrix = NULL;
    int commutative = (svop == NULL);
    if (ndims_b > ndims_a && commutative) {
        /* Swap matrices so that `a` is always the matrix with deepest shape */
        PSMatrix orig_a = a;
        int orig_ndims_a = ndims_a;
        a = b;
        b = orig_a;
        ndims_a = ndims_b;
        ndims_b = orig_ndims_a;
        shape_a = dims_b;
        shape_b = dims_a;
        deepest_shape = shape_a;
        deepest_nd = ndims_a;
        deepest_matrix = a;
    } else {
        if (ndims_b > ndims_a) {
            deepest_shape = shape_b;
            deepest_nd = ndims_b;
            deepest_matrix = b;
        } else {
            deepest_shape = shape_a;
            deepest_nd = ndims_a;
            deepest_matrix = a;
        }
    }
    long len_a = PSMatrixLength(a), len_b = PSMatrixLength(b);
    int same_shape = 0, i;
    deepest_len = PSMatrixLength(deepest_matrix);
    if (len_a == len_b && ndims_a == ndims_b) {
        same_shape = 1;
        for (i = 0; i < ndims_a; i++) {
            same_shape = (shape_a[i] == shape_b[i]);
            if (!same_shape) break;
        }
    }
    int shape_type_a = getShapeType(ndims_a, shape_a),
        shape_type_b = getShapeType(ndims_b, shape_b);
    int a_scalar = (shape_type_a == PS_SHAPE_TYPE_SCALAR),
        b_scalar = (shape_type_b == PS_SHAPE_TYPE_SCALAR),
        use_scalar = (a_scalar || b_scalar);
    int a_vec = (
        shape_type_a == PS_SHAPE_TYPE_COL ||
        shape_type_a == PS_SHAPE_TYPE_ROW
    );
    int b_vec = (
        shape_type_b == PS_SHAPE_TYPE_COL ||
        shape_type_b == PS_SHAPE_TYPE_ROW
    );
    int valid_shapes = (
        same_shape || use_scalar || (a_vec && b_vec && (len_a == len_b)) ||
        (b_vec && (shape_a[ndims_a - 1] == len_b) && (len_a % len_b) == 0)
    );
    if (!valid_shapes) {
        PSErr(func, "operands have incompatible shapes");
        return 0;
    }
    PSMatrix out = *result;
    if (out == NULL) {
        out = PSMatrixCreateWithShape(0, NULL, deepest_nd, deepest_shape);
        if (out == NULL) return 0;
        *result = out;
    } else {
        long shape_o[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int ndims_o = PSMatrixShape(out, shape_o);
        int same_out_shape = (ndims_o == deepest_nd);
        if (same_out_shape) {
            for (i = 0; i < deepest_nd; i++) {
                same_out_shape = deepest_shape[i] == shape_o[i];
                if (!same_out_shape) break;
            }
        }
        if (!same_out_shape) {
            PSErr(func, "result matrix has an invalid shape");
            return 0;
        }
    }
    if (same_shape || (a_vec && b_vec && len_a == len_b))
        vvop(a, b, out, len_a, opt);
    else if (b_scalar) vsop(a, *b, out, len_a, opt);
    else if (a_scalar && !commutative) svop(*a, b, out, deepest_len, opt);
    else {
        PSFloat *ap = a, *op = out;
        long count = len_a / len_b;
        for (i = 0; i < count; i++) {
            vvop(ap, b, op, len_b, opt);
            ap += len_b;
            op += len_b;
        }
    }
    return 1;
}

/* Performs matrix-vector multiplication between matrix `a` and vector `b`.
 * Argument `len` must be the length of the vector `b`.
 * Results are stored into vector pointed by pointer `result`. If pointer
 * pointed by `result` is NULL, a new vector is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * Length of `b` vector must equal matrix `a` second dimension.
 * Length of result vector must equal matrix `a` first dimension.
 * By default, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build, function will compute results by
 * using `PSDotProduct` as fallback.
 * The acceleration method can be changed via the `acceleration` member of
 * the optional `opt` argument.
 * The matrix `a` can be transpose by using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `result` is NULL or `a` is NULL or `b` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Matrix aligment error.
 *  - Invalid result shape.
 *  - Matrix pointed by `result` is not NULL and its shape differs
 *    from resulting output shape.
 *  - Memory allocation failure. */
int PSMatrixProductMV(PSMatrix a, PSFloat *b, long len, PSFloat **result,
                      PSMathOpts *opts)
{
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    PSBLASOrder order = PSBLASRowMajor;
    int acceleration = PSGlobalAcceleration;
    int transpose = 0;
    PSFloat beta = 0.0;
    if (opts != NULL) {
        transpose = opts->transpose;
        acceleration = opts->acceleration;
        if (opts->store_mode == PS_STORE_MODE_ADD) beta = 1.0;
    }
#ifdef HAS_BLAS
    int use_blas = PSBLASEnabled(acceleration);
#else
    int use_blas = 0;
    UNUSED(acceleration);
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    int use_acf = PSAccelerateEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(acceleration);
    UNUSED(use_acf);
#endif
    PSBLASErr blas_err = {.func = __func__};
    long dims_a[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixShape(a, dims_a);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
    if (ndims > 2) {
        PSMatrix mb = PSMatrixZeros(1, len);
        if (mb == NULL) return 0;
        PSMatrix res = NULL;
        int success = genericMatrixProduct(a, mb, &res, opts);
        if (success && res != NULL) {
            long reslen = PSMatrixLength(res);
            if (*result == NULL) *result = calloc(reslen, sizeof(PSFloat));
            if (*result != NULL) PSVectorCopy(*result, res, reslen);
            else success = 0;
        }
        PSMatrixFree(mb);
        PSMatrixFree(res);
        return success;
    }
    int scalar_a = (getShapeType(ndims, dims_a) == PS_SHAPE_TYPE_SCALAR);
    int scalar_b = len == 1;
    int use_scalar = (scalar_a || scalar_b);
    long l = (transpose & 1 ? dims_a[0] : dims_a[ndims - 1]);
    if (len != l && !use_scalar) {
        PSErr(__func__, "Aligment error: vector len != a dim[%d] -> "
              "%ld != %ld (transpose: %d)", (ndims - 1), l, len, transpose);
        PSMatrixPrintInfo(a, "a", 1);
        return 0;
    }
    int nd;
    long ld, outlen, len_a = PSMatrixLength(a);
    if (use_scalar) {
        if (scalar_a && scalar_b) nd = 0;
        else {
            nd = 1;
            if (scalar_a) ld = len;
            else ld = (transpose & 1 ? dims_a[nd] : dims_a[0]);
        }
    } else {
        nd = ndims - 1;
        ld = (transpose & 1 ? dims_a[nd] : dims_a[0]);
    }
    if (nd == 0) outlen = 1;
    else if (nd == 1) outlen = ld;
    else if (nd == 2) outlen = dims_a[ld] * len;
    else {
        PSErr(__func__, "Invalid output dimensions: %d", nd);
        return 0;
    }
    PSFloat *out = *result;
    if (out == NULL) {
        out = *result = calloc(outlen, sizeof(PSFloat));
        if (out == NULL) return 0;
    }
    if (use_scalar) {
        if (!scalar_a && transpose & 1) a = PSMatrixTranspose(a, 0, opts);
        if (scalar_a && scalar_b) *out = *a * *b;
        else if (scalar_a) PSMultiplyVectorScalar(b, *a, out, len, opts);
        else PSMultiplyVectorScalar(a, *b, out, len_a, opts);
        return 1;
    }
    if (ndims == 1) {
        out[0] = PSDotProduct(a, b, len, opts);
        return 1;
    }
    if (use_blas) {
        /* Check whether matrix shape, vector length or resulting matrix
         * length would exceed current BLAS limits, since Psyc used `long`
         * for indices, but the used BLAS library could a used smaller type. */
        int valid = PSBLASCheckLimits(
            &blas_err, "$3", "matrix length", len_a, "vector length", len,
            "result length", outlen
        );
        if (!valid) {
            use_blas = 0;
            PSWarn("%s", PSBLASErrorStr(&blas_err, NULL));
            PSWarn(
                "%s: BLAS has been disabled since some dimensions exceeds "
                "PSBLAS_MAX (%ld)", __func__, PSBLAS_MAX
            );
        }
    }
    long lda = (dims_a[1] > 1 ? dims_a[1] : 1);
    long m = dims_a[0], n = dims_a[1];
    if (!use_blas) {
        int do_add = (beta == 1.0);
        if (transpose & 1) {
            lda = (dims_a[0] > 1 ? dims_a[0] : 1);
            m = dims_a[1];
            n = dims_a[0];
            a = PSMatrixTranspose(a, 0, opts);
        }
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        if (use_acf) {
            if (outlen == 0) return 0;
            PSFloat *dest = out, *tmpdest = (opts ? opts->tmpdest : NULL);
            if (do_add) {
                if (tmpdest == NULL)
                    tmpdest = calloc(outlen, sizeof(PSFloat));
                if (tmpdest == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
                dest = tmpdest;
            }
            VDSPMMul(a, b, dest, m, 1, n);
            if (do_add) {
                PSMathOpts sumopts = {.acceleration = acceleration};
                PSAddVectors(out, dest, out, outlen, &sumopts);
                if (opts == NULL || tmpdest != opts->tmpdest)
                    free(tmpdest);
            }
            return 1;
        }
#endif
        PSMathOpts mopts = {.acceleration = acceleration};
        for (long i = 0; i < m; i++) {
            PSFloat *row = a + (i * n);
            if (!do_add) out[i] = PSDotProduct(row, b, n, &mopts);
            else out[i] += PSDotProduct(row, b, n, &mopts);
        }
        return 1;
    }
    char trans = (transpose & 1) ? 'T' : 'N';
    PSGemv(order, trans, m, n, 1.0, a, lda, b, 1, beta, out, 1);
    if (PSBLASLastError != NULL) return 0;
    return 1;
}

/* Performs vector-matrix multiplication between vector `a` and matrix `b`.
 * Argument `len` must be the length of the vector `a`.
 * Results are stored into matrix pointed by `result`. If pointer
 * pointed by `result` is NULL, a new matrix is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * The function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build, function will compute results by
 * using `PSDotProduct` as fallback.
 * The acceleration method can be changed via the `acceleration` member of
 * the optional `opt` argument.
 * Matrix `b` can be transposed by using the  `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the operand arguments you want to be transposed:
 *  - opt->transpose = 2 (transpose matrix `b`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `result` is NULL or `a` is NULL or `b` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Matrix aligment error.
 *  - Invalid result shape.
 *  - Matrix pointed by `result` is not NULL and its shape differs
 *    from resulting output shape.
 *  - Memory allocation failure. */
int PSMatrixProductVM(PSFloat *a, PSMatrix b, long len, PSMatrix *result,
                      PSMathOpts *opts)
{
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    PSBLASOrder order = PSBLASRowMajor;
    long mdims_b[PS_MATRIX_MAX_DIMENSIONS];
    long tdims_b[PS_MATRIX_MAX_DIMENSIONS];
    long *dims_b = mdims_b;
    int ndims = PSMatrixShape(b, dims_b);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
    if (ndims > 2) {
        PSMatrix ma = PSMatrixZeros(1, len);
        if (ma == NULL) return 0;
        int success = genericMatrixProduct(ma, b, result, opts);
        PSMatrixFree(ma);
        return success;
    }
    int last_dim = ndims - 1;
    long dimensions[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int transpose = 0;
    int acceleration = PSGlobalAcceleration;
    PSFloat beta = 0.0;
    if (opts != NULL) {
        transpose = opts->transpose;
        acceleration = opts->acceleration;
        if (opts->store_mode == PS_STORE_MODE_ADD) beta = 1.0;
        if (transpose & 2) {
            tdims_b[0] = dims_b[last_dim];
            tdims_b[last_dim] = dims_b[0];
            dims_b = tdims_b;
        }
    }
#ifdef HAS_BLAS
    int use_blas = PSBLASEnabled(acceleration);
#else
    int use_blas = 0;
    UNUSED(acceleration);
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    int use_acf = PSAccelerateEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(acceleration);
    UNUSED(use_acf);
#endif
    PSBLASErr blas_err = {.func = __func__};
    int scalar_a = len == 1;
    int scalar_b = (getShapeType(ndims, dims_b) == PS_SHAPE_TYPE_SCALAR);
    int use_scalar = (scalar_a || scalar_b);
    long l = dims_b[0];
    if (l != len && !use_scalar) {
        PSErr(__func__, "Aligment error: b dim[0] != vector length -> "
              "%ld != %ld (transpose: %d)", dims_b[0], len, transpose);
        return 0;
    }
    long len_b = PSMatrixLength(b);
    int nd;
    if (!use_scalar) {
        nd = 1 + ndims - 2;
        if (nd == 1) dimensions[0] = dims_b[1];
        else if (nd == 2) {
            dimensions[0] = len;
            dimensions[1] = dims_b[1];
        } else if (nd == 0) {
            nd = 1;
            dimensions[0] = 1;
        } else {
            PSErr(__func__, "Invalid output dimensions: %d", nd);
            return 0;
        }
    } else {
        nd = 1;
        dimensions[0] = (scalar_a ? len_b : len);
    }
    PSMatrix out = *result;
    long outlen = 0;
    if (out != NULL && (opts == NULL || opts->argtype[1] != 'V')) {
        PSMatrixHeader *hdr = PSMatrixGetHeader(out);
        if (hdr->ndims != nd) {
            PSErr(
                __func__, "`result` matrix has %d dimension(s), but "
                "%d dimension(s) needed", hdr->ndims, nd
            );
            return 0;
        }
        for (int i = 0; i < nd; i++) {
            long odim = PSMatrixDim(out, i);
            if (odim != dimensions[i]) {
                PSErr(__func__, "`result` matrix dimension [%d] is %d, "
                      "but it should be %d", i, odim, dimensions[i]);
                return 0;
            }
        }
        outlen = PSMatrixLength(out);
    } else if (out == NULL) {
        out = PSMatrixCreateWithShape(0, NULL, nd, dimensions);
        *result = out;
        if (out == NULL) return 0;
        outlen = PSMatrixLength(out);
    }
    if (outlen == 0) {
        outlen = 1;
        for (int i = 0; i < nd; i++) outlen *= dimensions[i];
    }
    if (use_scalar) {
        if (!scalar_b && transpose & 2) b = PSMatrixTranspose(b, 0, opts);
        if (scalar_a && scalar_b) *out = *a * *b;
        else if (scalar_b) PSMultiplyVectorScalar(a, *b, out, len, opts);
        else PSMultiplyVectorScalar(b, *a, out, len_b, opts);
        return 1;
    } else if (nd == 1 && dimensions[0] == 1  && len_b == len) {
        *out = PSDotProduct(a, b, len, opts);
        return 1;
    }
    if (use_blas) {
        /* Check whether matrix shape, vector length or resulting matrix
         * length would exceed current BLAS limits, since Psyc used `long`
         * for indices, but the used BLAS library could a used smaller type. */
        int valid = PSBLASCheckLimits(
            &blas_err, "$3", "matrix length", len_b, "vector length", len,
            "result length", outlen
        );
        if (!valid) {
            use_blas = 0;
            PSWarn("%s", PSBLASErrorStr(&blas_err, NULL));
            PSWarn(
                "%s: BLAS has been disabled since some dimensions exceeds "
                "PSBLAS_MAX (%ld)", __func__, PSBLAS_MAX
            );
        }
    }
    long lda = (mdims_b[1] > 1 ? mdims_b[1] : 1);
    long m = mdims_b[0], n = mdims_b[1];
    if (!use_blas) {
        int do_add = (beta == 1.0);
        if (!(transpose & 2)) b = PSMatrixTranspose(b, 0, opts);
        else {
            long tmpm = m;
            m = n;
            n = tmpm;
        }
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        if (use_acf) {
            if (outlen == 0) return 0;
            PSFloat *dest = out, *tmpdest = (opts ? opts->tmpdest : NULL);
            if (do_add) {
                if (tmpdest == NULL)
                    tmpdest = calloc(outlen, sizeof(PSFloat));
                if (tmpdest == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
                dest = tmpdest;
            }
            VDSPMMul(b, a, dest, n, 1, m);
            if (do_add) {
                PSMathOpts sumopts = {.acceleration = acceleration};
                PSAddVectors(dest, out, out, outlen, &sumopts);
                if (opts == NULL || tmpdest != opts->tmpdest)
                    free(tmpdest);
            }
            return 1;
        }
#endif
        PSMathOpts mopts = {.acceleration = acceleration};
        for (long i = 0; i < n; i++) {
            PSFloat *row = b + (i * m);
            if (!do_add) out[i] = PSDotProduct(row, a, m, &mopts);
            else out[i] += PSDotProduct(row, a, m, &mopts);
        }
        return 1;
    }
    char trans = (transpose & 2) ? 'N' : 'T'; /* 'N'; */
    PSGemv(order, trans, m, n, 1.0, b, lda, a, 1, beta, out, 1);
    if (PSBLASLastError != NULL) return 0;
    return 1;
}

/* Performs matrix-matrix multiplication between matrix `a` and matrix `b`.
 * Results are stored into matrix pointed by `result`. If pointer
 * pointed by `result` is NULL, a new matrix is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * The `opt` argument can be NULL.
 * By default, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build and `b` only has one dimension, function
 * will try compute results by using `PSDotProduct`.
 * The acceleration method can be changed via the `acceleration` member of
 * the optional `opt` argument.
 * Matrices can be transpose by using the `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opt`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `result` is NULL or `a` is NULL or `b` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Matrix aligment error.
 *  - Invalid result shape.
 *  - Matrix pointed by `result` is not NULL and its shape differs
 *    from resulting output shape.
 *  - Memory allocation failure.
 */
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt) {
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    /* Original matrix dimensions */
    long mdims_a[PS_MATRIX_MAX_DIMENSIONS];
    long mdims_b[PS_MATRIX_MAX_DIMENSIONS];
    /* Eventually transposed matrix dimensions */
    long tdims_a[PS_MATRIX_MAX_DIMENSIONS];
    long tdims_b[PS_MATRIX_MAX_DIMENSIONS];
    /* Number of dimensions */
    int ndims_a = PSMatrixShape(a, mdims_a);
    int ndims_b = PSMatrixShape(b, mdims_b);
    if (ndims_a == 0) {
        PSErr(__func__, "Invalid matrix `a`");
        return 0;
    }
    if (ndims_b == 0) {
        PSErr(__func__, "Invalid matrix `b`");
        return 0;
    }
    if (ndims_a > 2 || ndims_b > 2)
        return genericMatrixProduct(a, b, result, opt);
    int last_dim_a = ndims_a - 1, last_dim_b = ndims_b - 1, i;
    long *dims_a = mdims_a, *dims_b = mdims_b;
    long lda = 0, ldb = 0, l = 0;
    long dimensions[PS_MATRIX_MAX_DIMENSIONS] = {0};
    char trans_a = 'N', trans_b = 'N';
    int transpose = 0;
    int acceleration = PSGlobalAcceleration;
    PSFloat beta = 0.0;
    if (opt != NULL) {
        transpose = opt->transpose;
        acceleration = opt->acceleration;
        if (opt->store_mode == PS_STORE_MODE_ADD) beta = 1.0;
    }
#ifdef HAS_BLAS
    int use_blas = PSBLASEnabled(acceleration);
#else
    int use_blas = 0;
    UNUSED(acceleration);
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    int use_acf = PSAccelerateEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(use_acf);
    UNUSED(acceleration);
#endif
    PSBLASErr blas_err = {.func = __func__};
    if (use_blas) {
        /* Check whether matrix shape, vector length or resulting matrix
         * length would exceed current BLAS limits, since Psyc used `long`
         * for indices, but the used BLAS library could a used smaller type. */
        int valid = PSBLASCheckLimits(
            &blas_err, "$2", "`a` length", PSMatrixLength(a),
            "`b` length", PSMatrixLength(b)
        );
        if (!valid) {
            use_blas = 0;
            PSWarn("%s", PSBLASErrorStr(&blas_err, NULL));
            PSWarn(
                "%s: BLAS has been disabled since some dimensions exceeds "
                "PSBLAS_MAX (%ld)", __func__, PSBLAS_MAX
            );
        }
    }
    int shape_a = getShapeType(ndims_a, mdims_a);
    int shape_b = getShapeType(ndims_b, mdims_b);
    PSMatrix orig_a = a, orig_b = b;
    int use_scalar = (
        shape_a == PS_SHAPE_TYPE_SCALAR || shape_b == PS_SHAPE_TYPE_SCALAR
    ), reverse_args = 0;
    if (use_scalar) {
        /* One of `a` or `b` is a scalar-line matrix. */
        if (shape_a == PS_SHAPE_TYPE_SCALAR) {
            reverse_args = 1;
            PSMatrix tmp_a = a;
            int tmp_nda = ndims_a;
            long tmp_shape_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
            a = b;
            b = tmp_a;
            shape_a = shape_b;
            shape_b = PS_SHAPE_TYPE_SCALAR;
            ndims_a = ndims_b;
            ndims_b = tmp_nda;
            int tr = 0;
            if (transpose & 2) tr |= 1;
            if (transpose & 1) tr |= 2;
            transpose = tr;
            memcpy(tmp_shape_a, mdims_a, PS_MATRIX_MAX_DIMENSIONS*sizeof(long));
            memcpy(mdims_a, mdims_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(long));
            memcpy(mdims_b, tmp_shape_a, PS_MATRIX_MAX_DIMENSIONS*sizeof(long));
            last_dim_a = ndims_a - 1;
            last_dim_b = ndims_b - 1;
        }
    }
    int a_vector_like = (
        shape_a == PS_SHAPE_TYPE_COL || shape_a == PS_SHAPE_TYPE_ROW
    );
    int b_vector_like = (
        shape_b == PS_SHAPE_TYPE_COL || shape_b == PS_SHAPE_TYPE_ROW
    );
    int transpose_a = transpose & 1 && ndims_a > 1;/* !a_vector_like; */
    int transpose_b = transpose & 2 && ndims_b > 1;/* !b_vector_like; */
    if (transpose_a) {
        tdims_a[0] = mdims_a[last_dim_a];
        tdims_a[last_dim_a] = mdims_a[0];
        dims_a = tdims_a;
        shape_a = getShapeType(ndims_a, tdims_a);
        if (use_blas) trans_a = 'T';
        else {
            a = PSMatrixTranspose(a, 0, opt);
            if (a == NULL) {
                PSErr(__func__, "Failed to transpose matrix `a`");
                return 0;
            }
        }
    }
    if (transpose_b) {
        tdims_b[0] = mdims_b[last_dim_b];
        tdims_b[last_dim_b] = mdims_b[0];
        dims_b = tdims_b;
        shape_b = getShapeType(ndims_b, tdims_b);
        if (use_blas) trans_b = 'T';
        else {
            b = PSMatrixTranspose(b, 0, opt);
            if (a == NULL) {
                PSErr(__func__, "Failed to transpose matrix `b`");
                return 0;
            }
        }
    }
    int nd = 0;
    if (!use_scalar) {
        l = dims_a[last_dim_a];
        if (dims_b[0] != l) goto align_err;
        nd = ndims_a + ndims_b - 2;
        if (nd == 1) dimensions[0] = (ndims_a == 2 ? dims_a[0] : dims_b[1]);
        else if (nd == 2) {
            dimensions[0] = dims_a[0];
            dimensions[1] = dims_b[last_dim_b];
        } else if (nd == 0 && dims_a[0] == dims_b[0]) {
            nd = 1;
            dimensions[0] = 1;
        } else {
            PSErr(__func__, "Invalid output dimensions: %d", nd);
            return 0;
        }
    } else {
        long odims_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
        long odims_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int o_ndims_a = PSMatrixShape(orig_a, odims_a),
            o_ndims_b = PSMatrixShape(orig_b, odims_b);
        if (transpose) {
            int orig_transp = transpose;
            if (reverse_args) {
                orig_transp = 0;
                if (transpose & 2) orig_transp |= 1;
                if (transpose & 1) orig_transp |= 2;
            }
            if (o_ndims_a > 1 && orig_transp & 1) {
                long tmp = odims_a[0];
                odims_a[0] = odims_a[o_ndims_a - 1];
                odims_a[o_ndims_a - 1] = tmp;
            }
            if (o_ndims_b > 1 && orig_transp & 2) {
                long tmp = odims_b[0];
                odims_b[0] = odims_b[o_ndims_b - 1];
                odims_b[o_ndims_b - 1] = tmp;
            }
        }
        l = odims_a[o_ndims_a - 1];
        if (odims_b[0] != l) goto align_err;
        nd = ndims_a + ndims_b - 2;
        if (nd == 1) {
            dimensions[0] = (o_ndims_a == 2 ? odims_a[0] : odims_b[1]);
            l = dimensions[0];
        } else if (nd == 2) {
            dimensions[0] = odims_a[0];
            dimensions[1] = odims_b[1];
            if (shape_a == PS_SHAPE_TYPE_ROW) l = dimensions[1];
            else l = dimensions[0];
        } else if (nd == 0 && dims_a[0] == dims_b[0]) {
            nd = 1;
            dimensions[0] = 1;
        }
        if (odims_a[o_ndims_a - 1] == 0) l = 0;
    }
    long outlen = 0;
    PSMatrix out = *result;
    if (out != NULL && (opt == NULL || opt->argtype[2] != 'V')) {
        PSMatrixHeader *hdr = PSMatrixGetHeader(out);
        if (hdr->ndims != nd) {
            PSErr(
                __func__, "`result` matrix has %d dimension(s), but "
                "%d dimension(s) needed", hdr->ndims, nd
            );
            return 0;
        }
        for (i = 0; i < nd; i++) {
            long odim = PSMatrixDim(out, i);
            if (odim != dimensions[i]) {
                PSErr(
                    __func__, "`result` matrix dimension [%d] is %ld, "
                    "but it should be %ld\nResult shape: %ld,%ld",
                    i, odim, dimensions[i]
                );
                return 0;
            }
        }
        outlen = PSMatrixLength(out);
    } else if (out == NULL) {
        out = PSMatrixCreateWithShape(0, NULL, nd, dimensions);
        *result = out;
        if (out == NULL) return 0;
        outlen = PSMatrixLength(out);
    }
    if (outlen == 0) {
        outlen = 1;
        for (int i = 0; i < nd; i++) outlen *= dimensions[i];
    }
    if (a_vector_like && b_vector_like && nd == 1 && dimensions[0] == 1 &&
        PSMatrixLength(a) == PSMatrixLength(b))
    {
        *out = PSDotProduct(a, b, PSMatrixLength(a), opt);
        return 1;
    }
    if (use_blas && outlen > PSBLAS_MAX) {
        PSErr(__func__, "result matrix length (%ld) would exceed PSBLAS_MAX "
              "(%ld)", outlen, PSBLAS_MAX);
        return 0;
    }
    PSBLASOrder order;
    if (shape_b == PS_SHAPE_TYPE_SCALAR) {
        if (l == 1) {
            *out = *b * *a;
            return 1;
        } else {
            long a_stride;
            if (shape_a == PS_SHAPE_TYPE_ROW) a_stride = PSMatrixStride(a, 1);
            else a_stride = PSMatrixStride(a, 0);
            if (shape_a != PS_SHAPE_TYPE_MATRIX)
                PSAxpy(l, *((PSFloat *) b), a, a_stride, out, 1);
            else {
                int max_dim_idx = (dims_a[0] >= dims_a[1] ? 0 : 1);
                int o_dim_idx = 1 - max_dim_idx;
                l = dims_a[max_dim_idx];
                PSFloat val = *((PSFloat *) b);
                a_stride = PSMatrixStride(a, max_dim_idx);
                long o_stride = PSMatrixStride(out, max_dim_idx);
                long o_dim = dims_a[o_dim_idx];
                PSFloat *aptr = a, *optr = out;
                for (i = 0; i < o_dim; i++) {
                    PSAxpy(l, val, aptr, a_stride, optr, o_stride);
                    aptr += PSMatrixStride(a, o_dim_idx);
                    optr += PSMatrixStride(out, o_dim_idx);
                }
            }
        }
    } else if (!a_vector_like && b_vector_like) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        /* Always use original dimensions here, even if `a` in transposed */
        dims_a = mdims_a;
        dims_b = mdims_b;
        order = PSBLASRowMajor;
        lda = (dims_a[last_dim_a] > 1 ? dims_a[1] : 1);
        long bs = PSMatrixStride(b, (transpose_b ? ndims_b - 1 : 0));
        long m = dims_a[0], n = dims_a[1];
        if (!use_blas) {
            int do_add = (beta == 1.0);
            if (transpose_a) {
                long tmpm = m;
                m = n;
                n = tmpm;
            }
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
            if (use_acf) {
                if (outlen == 0) return 0;
                PSFloat *dest = out, *tmpdest = (opt ? opt->tmpdest : NULL);
                if (do_add) {
                    if (tmpdest == NULL)
                        tmpdest = calloc(outlen, sizeof(PSFloat));
                    if (tmpdest == NULL) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                    dest = tmpdest;
                }
                VDSPMMul(a, b, dest, m, 1, n);
                if (do_add) {
                    PSMathOpts sumopts = {.acceleration = acceleration};
                    PSAddVectors(dest, out, out, outlen, &sumopts);
                    if (opt == NULL || tmpdest != opt->tmpdest)
                        free(tmpdest);
                }
                return 1;
            }
#else
            UNUSED(outlen);
#endif
            PSMathOpts mopts = {.acceleration = acceleration};
            for (long i = 0; i < m; i++) {
                PSFloat *row = a + (i * n);
                if (!do_add) out[i] = PSDotProduct(row, b, n, &mopts);
                else out[i] += PSDotProduct(row, b, n, &mopts);
            }
            return 1;
        }
        PSGemv(order, trans_a, m, n, 1.0, a, lda, b, bs, beta, out, 1);
    } else if (a_vector_like && !b_vector_like) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        /* Always use original dimensions here, even if `a` in transposed */
        dims_a = mdims_a;
        dims_b = mdims_b;
        order = PSBLASRowMajor;
        lda = (dims_b[1] > 1 ? dims_b[1] : 1);
        long m = dims_b[0], n = dims_b[1];
        if (!use_blas) {
            int do_add = (beta == 1.0);
            /* Transposition here needs to work in reveres way: if b should be
             * transposed, use it with its original shape, if no tranposition
             * has been set, use it in its transpodes shape.
             * Since, in the first case (transposed), `b` has already been
             * transposed, always call PSMatrixTranspose:
             *  - if already transposed, just return it's original (cached)
             *    version
             *  - otherwise, transpose it. */
            b = PSMatrixTranspose(b, 0, opt);
            if (transpose_b) {
                long tmpm = m;
                m = n;
                n = tmpm;
            }
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
            if (use_acf) {
                if (outlen == 0) return 0;
                PSFloat *dest = out, *tmpdest = (opt ? opt->tmpdest : NULL);
                if (do_add) {
                    if (tmpdest == NULL)
                        tmpdest = calloc(outlen, sizeof(PSFloat));
                    if (tmpdest == NULL) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                    dest = tmpdest;
                }
                VDSPMMul(b, a, dest, n, 1, m);
                if (do_add) {
                    PSMathOpts sumopt = {.acceleration = acceleration};
                    PSAddVectors(dest, out, out, outlen, &sumopt);
                    if (opt == NULL || tmpdest != opt->tmpdest)
                        free(tmpdest);
                }
                return 1;
            }
#endif
            if (opt) opt->store_mode = PS_STORE_MODE_SET;
            for (long i = 0; i < n; i++) {
                PSFloat *row = b + (i * m);
                if (!do_add) out[i] = PSDotProduct(row, a, m, opt);
                else out[i] += PSDotProduct(row, a, n, opt);
            }
            return 1;
        }
        char trans = (transpose & 2) ? 'N' : 'T'; /* 'N'; */
        PSGemv(order, trans, m, n, 1.0, b, lda, a, 1, beta, out, 1);
    } else {
        /* Matrix matrix multiplication -- Level 3 BLAS */
        order = PSBLASRowMajor;
        long m = dims_a[0];
        long n = dims_b[1];
        long k = dims_a[1];
        if (!use_blas) {
            int do_add = (beta == 1.0);
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
            if (use_acf) {
                if (outlen == 0) return 0;
                PSFloat *dest = out, *tmpdest = (opt ? opt->tmpdest : NULL);
                if (do_add) {
                    if (tmpdest == NULL)
                        tmpdest = calloc(outlen, sizeof(PSFloat));
                    if (tmpdest == NULL) {
                        PSPrintMemoryErrorMsg();
                        return 0;
                    }
                    dest = tmpdest;
                }
                VDSPMMul(a, b, dest, m, n, k);
                if (do_add) {
                    PSMathOpts sumopt = {.acceleration = acceleration};
                    PSAddVectors(dest, out, out, outlen, &sumopt);
                    if (opt == NULL || tmpdest != opt->tmpdest)
                        free(tmpdest);
                }
                return 1;
            }
#endif
            b = PSMatrixTranspose(b, 0, opt);
            m = PSMatrixDim(a, 0);
            n = PSMatrixDim(a, 1);
            l = PSMatrixDim(b, 0);
            k = PSMatrixDim(b, 1);
            if (opt) opt->store_mode = PS_STORE_MODE_SET;
            for (long i = 0; i < dimensions[0]; i++) {
                for (long j = 0; j < dimensions[1]; j++) {
                    long oidx = (i * dimensions[1]) + j;
                    PSFloat *arow = a + (n * i);
                    PSFloat *brow = b + (k * j);
                    if (!do_add) out[oidx] = PSDotProduct(arow, brow, k, opt);
                    else out[oidx] += PSDotProduct(arow, brow, k, opt);
                }
            }
            return 1;
        }
        if (trans_a == 'N') lda =  (k > 1 ? k : 1);
        else lda = (m > 1 ? m : 1);
        if (trans_b == 'N') ldb =  (n > 1 ? n : 1);
        else ldb = (k > 1 ? k : 1);
        /*
        size_t alen = PSMatrixLength(a), blen = PSMatrixLength(b);
        if (alen == blen &&
           dims_a[0] == dims_b[1] &&
           dims_a[1] == dims_b[0] &&
           PSMatrixStride(a, 0) == PSMatrixStride(b, 1) &&
           PSMatrixStride(a, 1) == PSMatrixStride(b, 0) &&
           (trans_a == 'T' ? 1 : 0) ^ (trans_b == 'T' ? 1 : 0) &&
           (trans_a == 'N' ? 1 : 0) ^ (trans_b == 'N' ? 1 : 0)) {
            PSErr(__func__, "Unsupported BLAS Syrc");
            return 0;
        } else {
            long odim1 = PSMatrixDim(out, 1);
            long ldc = ((odim1 > 1) ? odim1 : 1);
            PSGemm(order, trans_a, trans_b, m, n, k, 1.0, a, lda, b, ldb, beta,
                   out, ldc);
        }*/
        long odim1 = PSMatrixDim(out, 1);
        long ldc = ((odim1 > 1) ? odim1 : 1);
        PSGemm(order, trans_a, trans_b, m, n, k, 1.0, a, lda, b, ldb, beta,
               out, ldc);
    }
    if (PSBLASLastError != NULL) return 0;
    return 1;
align_err:
    if (reverse_args) {
        /* `a` and `b` were reversed */
        long tmp_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
        memcpy(tmp_shape, dims_a, PS_MATRIX_MAX_DIMENSIONS * sizeof(long));
        memcpy(dims_a, dims_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(long));
        memcpy(dims_b, dims_a, PS_MATRIX_MAX_DIMENSIONS * sizeof(long));
        int tr_a = transpose_a, last_d_a = last_dim_a;
        transpose_a = transpose_b;
        transpose_b = tr_a;
        last_dim_a = last_dim_b;
        last_dim_b = last_d_a;
    }
    PSErr(
        __func__, "Aligment error: `b` dim[0] != `a` dim[%d] -> %d != %d\n"
        "Matrix a shape: %d,%d %s\n"
        "Matrix b shape: %d,%d %s",
        last_dim_a, dims_b[0], l,
        dims_a[0], dims_a[last_dim_a], (transpose_a ? "(transp.)" : ""),
        dims_b[0], dims_b[last_dim_b], (transpose_b ? "(transp.)" : "")
    );
    return 0;
}

/* Add matrix `a` to matrix `b`. Results are stored into matrix pointed by
 * pointer `result`. If pointer pointed by `result` is NULL, a new matrix is
 * automatically allocated by the function itself and its pointer will be
 * stored into `result`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opt`.
 * Both matrices can be transposed using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result matrix will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result matrix.
 * The function will take in account the shape of both matrices so the
 * operation is performed in different ways depending on the shapes and the
 * shapes' type (see `PSMatrixShape` for more details about the shape type):
 *  - If both `a` and `b` have the same shape or if both their shape types are
 *    vector-like shapes (`PS_SHAPE_TYPE_ROW` or `PS_SHAPE_TYPE_COL`) and
 *    both `a` and `b` have the same length (total number of values),
 *    every element of the resulting matrix will be the sum of every element
 *    of `a` and the corresponding element of `b`.
 *  - If only one of `a` or `b` has a vector-like shape (`PS_SHAPE_TYPE_ROW` or
 *    `PS_SHAPE_TYPE_COL`) and the other matrix has a matrix-like shape and the
 *    length of the vector-like matrix is the same of the last dimension of
 *    the other matrix, the resulting matrix will have the shape of the matrix
 *    with a matrix-like shape and the values from the vector-like matrix will
 *    be added to the values of the "rows" of the matrix-like matrix.
 *    For example: if `a` has a shape of 2,3 and `b` has a shape of 1,3,
 *    the result will be computed as a[0] + b and a[1] + b.
 *  - If `a` or `b` have a scalar-like shape (`PS_SHAPE_TYPE_SCALAR`), the
 *    resulting matrix will have the shape of the non-scalar matrix with the
 *    scalar value of the scalar-like matrix (basically, its first and only
 *    element) added to the all the values of the non-scalar matrix.
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `a` is NULL or `b` is NULL or `result` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Invalid shapes:
 *    - Shapes differ, and
 *    - neither `a` nor `b` have scalar-like shape, and
 *    - both `a` and `b` have vector-like shape but their total length differ
 *    - one of `a` or `b` has vector-like shape whose size differs from the
 *      matrix-like matrix last dimension.
 *  - Memory allocation failure.
 */
int PSMatrixAdd(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt) {
    return genericMatrixOperation(
        a, b, result, PSAddVectors, PSAddVectorScalar, NULL, __func__, opt
    );
}

/* Multiply matrix `a` by matrix `b`. Results are stored into matrix pointed by
 * pointer `result`. If pointer pointed by `result` is NULL, a new matrix is
 * automatically allocated by the function itself and its pointer will be
 * stored into `result`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opt`.
 * Both matrices can be transposed using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result matrix will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result matrix.
 * The function will take in account the shape of both matrices so the
 * operation is performed in different ways depending on the shapes and the
 * shapes' type (see `PSMatrixShape` for more details about the shape type):
 *  - If both `a` and `b` have the same shape or if both their shape types are
 *    vector-like shapes (`PS_SHAPE_TYPE_ROW` or `PS_SHAPE_TYPE_COL`) and
 *    both `a` and `b` have the same length (total number of values),
 *    every element of the resulting matrix will be the multiplication of
 *    every element of `a` by the corresponding element of `b`.
 *  - If only one of `a` or `b` has a vector-like shape (`PS_SHAPE_TYPE_ROW` or
 *    `PS_SHAPE_TYPE_COL`) and the other matrix has a matrix-like shape and the
 *    length of the vector-like matrix is the same of the last dimension of
 *    the other matrix, the resulting matrix will have the shape of the matrix
 *    with a matrix-like shape and the values from the vector-like matrix will
 *    be multiplied by the values of the "rows" of the matrix-like matrix.
 *    For example: if `a` has a shape of 2,3 and `b` has a shape of 1,3,
 *    the result will be computed as a[0] * b and a[1] * b.
 *  - If `a` or `b` have a scalar-like shape (`PS_SHAPE_TYPE_SCALAR`), the
 *    resulting matrix will have the shape of the non-scalar matrix with all
 *    the values of the non-scalar matrix multiplied by the scalar value of
 *    the scalar-like matrix (basically, its first and only element).
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `a` is NULL or `b` is NULL or `result` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Invalid shapes:
 *    - Shapes differ, and
 *    - neither `a` nor `b` have scalar-like shape, and
 *    - both `a` and `b` have vector-like shape but their total length differ
 *    - one of `a` or `b` has vector-like shape whose size differs from the
 *      matrix-like matrix last dimension.
 *  - Memory allocation failure.
 */
int PSMatrixMultiply(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
{
    return genericMatrixOperation(a, b, result, PSMultiplyVectors,
                                  PSMultiplyVectorScalar, NULL, __func__, opt);
}

/* Subtract matrix `b` from matrix `a`. Results are stored into matrix pointed
 * by pointer `result`. If pointer pointed by `result` is NULL, a new matrix is
 * automatically allocated by the function itself and its pointer will be
 * stored into `result`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opt`.
 * Both matrices can be transposed using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result matrix will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result matrix.
 * The function will take in account the shape of both matrices so the
 * operation is performed in different ways depending on the shapes and the
 * shapes' type (see `PSMatrixShape` for more details about the shape type):
 *  - If both `a` and `b` have the same shape or if both their shape types are
 *    vector-like shapes (`PS_SHAPE_TYPE_ROW` or `PS_SHAPE_TYPE_COL`) and
 *    both `a` and `b` have the same length (total number of values),
 *    every element of the resulting matrix will be the subtraction of every
 *    element of `b` from the corresponding element of `a`.
 *  - If only one of `a` or `b` has a vector-like shape (`PS_SHAPE_TYPE_ROW` or
 *    `PS_SHAPE_TYPE_COL`) and the other matrix has a matrix-like shape and the
 *    length of the vector-like matrix is the same of the last dimension of
 *    the other matrix, the resulting matrix will have the shape of the matrix
 *    with a matrix-like shape and the values of the vector-like matrix will
 *    be subtracted from the values of the "rows" of the matrix-like matrix.
 *    For example: if `a` has a shape of 2,3 and `b` has a shape of 1,3,
 *    the result will be computed as a[0] - b and a[1] - b.
 *  - If `a` or `b` have a scalar-like shape (`PS_SHAPE_TYPE_SCALAR`), the
 *    resulting matrix will have the shape of the non-scalar matrix with the
 *    scalar value of the scalar-like matrix (basically, its first and only
 *    element) subtracted from all the values of the non-scalar matrix.
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `a` is NULL or `b` is NULL or `result` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Invalid shapes:
 *    - Shapes differ, and
 *    - neither `a` nor `b` have scalar-like shape, and
 *    - both `a` and `b` have vector-like shape but their total length differ
 *    - one of `a` or `b` has vector-like shape whose size differs from the
 *      matrix-like matrix last dimension.
 *  - Memory allocation failure.
 */
int PSMatrixSubtract(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
{
    return genericMatrixOperation(a, b, result,
                                  PSSubtractVectors,
                                  PSSubtractVectorScalar,
                                  PSSubtractScalarVector,
                                  __func__, opt);
}

/* Divide matrix `a` from matrix `b`. Results are stored into matrix pointed
 * by pointer `result`. If pointer pointed by `result` is NULL, a new matrix is
 * automatically allocated by the function itself and its pointer will be
 * stored into `result`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opt`.
 * Both matrices can be transposed using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result matrix will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result matrix.
 * The function will take in account the shape of both matrices so the
 * operation is performed in different ways depending on the shapes and the
 * shapes' type (see `PSMatrixShape` for more details about the shape type):
 *  - If both `a` and `b` have the same shape or if both their shape types are
 *    vector-like shapes (`PS_SHAPE_TYPE_ROW` or `PS_SHAPE_TYPE_COL`) and
 *    both `a` and `b` have the same length (total number of values),
 *    every element of the resulting matrix will be the division of every
 *    element of `a` by the corresponding element of `b`.
 *  - If only one of `a` or `b` has a vector-like shape (`PS_SHAPE_TYPE_ROW` or
 *    `PS_SHAPE_TYPE_COL`) and the other matrix has a matrix-like shape and the
 *    length of the vector-like matrix is the same of the last dimension of
 *    the other matrix, the resulting matrix will have the shape of the matrix
 *    with a matrix-like shape and the values of the "rows" of the matrix-like
 *    matrix will be divided by the values of the vector-like matrix.
 *    For example: if `a` has a shape of 2,3 and `b` has a shape of 1,3,
 *    the result will be computed as a[0] / b and a[1] / b.
 *  - If `a` or `b` have a scalar-like shape (`PS_SHAPE_TYPE_SCALAR`), the
 *    resulting matrix will have the shape of the non-scalar matrix with all
 *    the values of the non-scalar matrix divded by the scalar value of the
 *    scalar-like matrix (basically, its first and only element).
 * Return value: 1 if operation succeeds, 0 if it fails.
 * Possible failure reasons:
 *  - `a` is NULL or `b` is NULL or `result` is NULL.
 *  - `a` has zero dimensions or `b` has zero dimensions.
 *  - Invalid shapes:
 *    - Shapes differ, and
 *    - neither `a` nor `b` have scalar-like shape, and
 *    - both `a` and `b` have vector-like shape but their total length differ
 *    - one of `a` or `b` has vector-like shape whose size differs from the
 *      matrix-like matrix last dimension.
 *  - Memory allocation failure.
 */
int PSMatrixDivide(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
{
    if (b != NULL) {
        long shape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int nd_b = PSMatrixShape(b, shape_b);
        if (getShapeType(nd_b, shape_b) == PS_SHAPE_TYPE_SCALAR && *b == 0) {
            PSErr(__func__, "division by zero");
            return 0;
        }
    }
    return genericMatrixOperation(a, b, result,
                                  PSDivideVectors,
                                  PSDivideVectorScalar,
                                  PSDivideScalarVector,
                                  __func__, opt);
}

/* Create a new matrix that is the reshaped version of `matrix`. The new matrix
 * will have the same values of `matrix` but a different shape having number of
 * dimensions defined by `num_dims`. The new shape can be declared by using
 * the variadic arguments after `num_dims`.
 * The total number of elements given by the new shape must be equal to the
 * total number of element of `matrix`, so, for example, reshaping a matrix
 * with shape 2,3 to a matrix with shape 1,6 is valid and reshaping a matrix
 * with shape 2,3,3 to a matrix of 1,18 or a matrix of 2,9 is also valid, but
 * reshaping a matrix of 2,3 to a matrix of 1,3 is not valid.
 * Result value: the new reshaped matrix or NULL if:
 *  - `matrix` is NULL.
 *  - `num_dims` is zero or negative.
 *  - `num_dims` is greater than `PS_MATRIX_MAX_DIMENSIONS`.
 *  - The total number of elements of the new matrix would differ from the
 *    total number of elements of `matrix`.
 *  - Memory cannot be allocated. */
PSMatrix PSMatrixReshape(PSMatrix matrix, int num_dims, ...) {
    if (matrix == NULL) return NULL;
    if (num_dims <= 0) return NULL;
    if (num_dims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "max shape dimensions: %d", PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    long new_len = 1;
    long new_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    va_list args;
    va_start(args, num_dims);
    for (int i = 0; i < num_dims; i++) {
        new_shape[i] = va_arg(args, long);
        new_len *= new_shape[i];
    }
    va_end(args);
    if (new_len != hdr->length) {
        PSErr(__func__, "reshaped matrix length would differ from original "
              "matrix length: %llu != %llu");
        return NULL;
    }
    PSMatrix reshaped = PSMatrixCreateWithShape(0, NULL, num_dims, new_shape);
    if (reshaped == NULL) return NULL;
    PSVectorCopy(reshaped, matrix, hdr->length);
    return reshaped;
}

/* Create a new matrix that is the single-dimensioned, flatten version of
 * `matrix`.
 * For example, if `matrix` has a shape of (2,3), the resulting matrix will
 * have a shape of (6).
 * Return value: the new flatten matrix or NULL if:
 *  - `matrix` is NULL.
 *  - Memory connot be allocated. */
PSMatrix PSMatrixFlatten(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    return PSMatrixReshape(matrix, 1, PSMatrixLength(matrix));
}

/* Split `matrix` into smaller matrices whose number is defined by
 * `num_slices`. The matrix will be split on the axis (dimension) defined by
 * the `axis` argument.
 * If the `axis` argument is negative, it will be counted from the last
 * dimension of the shape of `matrix`: for example, an axis of -1 means the
 * last dimension of the shape.
 * NOTE: this function currenlty works only if `matrix` has up-to two
 * dimensions or if `matrix` has more than two dimensions but `axis` is the
 * first dimension or the last dimensions (so it cannot be used to split a
 * matrix with more than two dimensions by an intermediate axis).
 * The optional `opts` argument can be used to change the default acceleration
 * methods (by default, `PSGlobalAcceleration` is used).
 * Return value: an array of `num_slices` sub-matrices whose length is or NULL
 * if:
 *  - `matrix` is NULL.
 *  - `matrix` is empty.
 *  - `axis` is out of bounds.
 *  - `matrix` has more than two dimensions but `axis` is neither the first nor
 *    the last axis.
 *  - The value of `num_slices` would not lead to an equal division
 *    (`shape[axis] % num_slices != 0`).
 *  - Memory cannot be allocate.
 * NOTE: it's up to the developer using this function to free both the
 * sub-matrices (by using `PSMatrixFlatten`) and the returned array containing
 * them. */
PSMatrix *PSMatrixSplit(PSMatrix matrix, long num_slices, int axis,
                        PSMathOpts *opts)
{
    if (matrix == NULL) return NULL;
    long shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixShape(matrix, shape);
    long len = PSMatrixLength(matrix);
    if (len == 0 || ndims <= 0) {
        PSErr(__func__, "cannot split an empty matrix");
        return NULL;
    }
    if (axis < 0) axis = ndims + axis;
    if (axis < 0 || axis >= ndims) {
        if (axis < 0) axis -= ndims;
        PSErr(__func__, "invalid axis %d, matrix only has %d dimension(s)",
              ndims);
        return NULL;
    }
    if (ndims > 2 && axis != 0 && axis != (ndims - 1)) {
        PSErr(__func__, "split not supported for shapes with more that 2 "
              "axes and axis different than first and last");
        return NULL;
    }
    long dimsize = shape[axis];
    long mod = dimsize % num_slices;
    if (mod != 0) {
        PSErr(__func__, "matrix split does not result in equal division");
        return NULL;
    }
    PSMatrix *slices = calloc(num_slices, sizeof(PSMatrix));
    if (slices == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    };
    int success = 1;
    long split_dimsize = dimsize / num_slices, i;
    long num_sizes = 1 + mod + (num_slices - mod);
    long sizes[num_sizes];
    sizes[0] = 0;
    long *size_p = ((long *)sizes) + 1;
    for (i = 0; i < mod; i++) *(size_p++) = split_dimsize + 1;
    for (i = 0; i < (num_slices - mod); i++) *(size_p++) = split_dimsize;
    long points[num_sizes];
    long cumsum = 0;
    for (i = 0; i < num_sizes; i++) {
        cumsum += sizes[i];
        points[i] = cumsum;
    }
    int do_transpose = axis != 0;
    if (do_transpose) {
        matrix = PSMatrixTranspose(matrix, 0, opts);
        success = (matrix != NULL);
        if (!success) goto final;
        PSMatrixShape(matrix, shape);
    }
    long elem_size = 1;
    for (i = 1; i < ndims; i++) elem_size *= shape[i];
    for (i = 0; i < num_slices; i++) {
        long from = points[i], to = points[i + 1], len = to - from;
        long size = len * elem_size;
        PSFloat *data = matrix + (from * elem_size);
        long slice_shape[PS_MATRIX_MAX_DIMENSIONS] = {len};
        for (int j = 1; j < ndims; j++) slice_shape[j] = shape[j];
        PSMatrix subm = PSMatrixCreateWithShape(0, NULL, ndims, slice_shape);
        success = (subm != NULL);
        if (!success) goto final;
        PSVectorCopy(subm, data, size);
        if (do_transpose) {
            subm = PSMatrixTranspose(subm, 0, opts);
            success = (subm != NULL);
            if (!success) goto final;
        }
        slices[i] = subm;
    }
final:
    if (!success) {
        PSErr(__func__, "failed to split matrix with shape %s",
              matrixDimensionsToString(ndims, shape));
        if (slices != NULL) {
            for (i = 0; i < num_slices; i++) PSMatrixFree(slices[i]);
            free(slices);
            slices = NULL;
        }
    }
    return slices;
}

/* Transpose `matrix` by swapping its first dimension with its last
 * dimension. For example, a matrix with a shape of 2,3 will be transposed to
 * a matrix with shape of 3,2.
 * The function won't modify `matrix` but it will create a new matrix that is
 * the transposed version of `matrix`.
 * The trasponsed matrix is cached in the private data of `matrix` so that
 * subsequent calls of this function with the same `matrix` and with `rebuild`
 * argument set to zero will directly return the cached transposed matrix
 * without recomputing the transposition.
 * The `rebuild` argument can be used to invalidate the cached transposed
 * matrix forcing the function to rebuild it.
 * If `matrix` already is the cached transposed matrix of another matrix,
 * the function will directly return the source matrix of `matrix`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * WARN: The cached transposed matrix is automatically freed by freeing its
 * owner (`matrix`) with `PSMatrixFree`, so it should not be freed directly. In 
 * order to free and reset the transposed cached matrix, the function
 * `PSMatrixResetTransposed` should be called on `matrix`.
 * WARN: most of the functions who alter the original matrix (`matrix`)
 * will also invalidate the cached transposed matrix if any. However, manually
 * changing matrix's values would lead to inconsistency between the matrix and
 * its transposed version so the transposed matrix should be invalidated with
 * `PSMatrixResetTransposed` or rebuilt by calling `PSMatrixTranspose` with
 * `rebuild` argument set to true.
 * Return value: the transposed matrix or NULL if:
 *  - `matrix` is NULL.
 *  - Memory canmot be allocated. */
PSMatrix PSMatrixTranspose(PSMatrix matrix, int rebuild, PSMathOpts *opts) {
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    PSMatrixHeader *t_hdr = NULL;
    if (hdr->transposed != NULL) {
        if (!rebuild) return hdr->transposed;
        PSMatrixFree(hdr->transposed);
        hdr->transposed = NULL;
    } else if (hdr->transposed_from != NULL) return hdr->transposed_from;
    int ndims = hdr->ndims;
    long *dims = hdr->shape;
    long ncols = 0, nrows = 0, dlen = 0, t_ncols = 0, t_dlen = 0;
    long x, y, z, idx, i;
    PSMatrix transposed = NULL;
    if (ndims == 3) {
        transposed = PSMatrixZeros(3, dims[2], dims[1], dims[0]);
        if (transposed == NULL) return NULL;
        nrows = dims[1];
        ncols = dims[2];
        t_ncols = dims[0];
        t_dlen = dims[0] * dims[1];
        dlen = nrows * ncols;
        for (i = 0; i < hdr->length; i++) {
            z = i % dlen % ncols;
            y = (i / ncols) % nrows;
            x = i / dlen;
            idx = (z * t_dlen) + (y * t_ncols) + x;
            transposed[idx] = matrix[i];
        }
    } else if (ndims == 2) {
        int acceleration = PSGlobalAcceleration;
        if (opts != NULL) acceleration = opts->acceleration;
        transposed = PSMatrixZeros(2, dims[1], dims[0]);
        if (transposed == NULL) return NULL;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        if (PSAccelerateEnabled(acceleration)) {
            /* Use Apple(R) Accelerate Framework */
            VDSPMTransp(matrix, transposed, dims[1], dims[0]);
            goto final;
        }
#else
        UNUSED(acceleration);
#endif
        ncols = dims[1];
        t_ncols = dims[0];
        for (i = 0; i < hdr->length; i++) {
            y = i % ncols;
            x = i / ncols;
            idx = (y * t_ncols) + x;
            transposed[idx] = matrix[i];
        }
    } else if (ndims == 1) return matrix;
final:
    if (transposed == NULL) return NULL;
    t_hdr = PSMatrixGetHeader(transposed);
    t_hdr->transposed_from = matrix;
    t_hdr->transposed = NULL;
    hdr->transposed = transposed;
    hdr->transposed_from = NULL;
    return transposed;
}

/* Create a new matrix by swapping axes of `matrix`. For example, swapping axes
 * 0 and 1 of a matrix with shape of 2,3 would create a matrix with shape of
 * 3,2 and swaping axes 1 and 2 of a matrix with shape 2,3,4 would create a
 * matrix with a shape of 2,4,3.
 * The axes to be swapped are defined by `axis1` and `axis2` arguments: by
 * using a negative value for an axis, it will be counted from the last
 * dimension of the shape, so, for example, swapping the axes -1 and -2 of
 * a matrix with shape 2,3,4 would create a matrix with shape of 2,4,3.
 * If both `axis1` and `axis2` refer to the same axis, the function will
 * return a duplicated versiob of `matrix`.
 * NOTE: despite calling this function with the first and the last axis would
 * have the same result of `PSMatrixTranspose` in terms of matrix data and
 * shape, the swapped matrix created by `PSMatrixSwapAxes` always is an
 * independent matrix and not the cached tranposed matrix of `matrix`.
 * Return value: the new swapped matrix or NULL if:
 *  - `matrix` is NULL.
 *  - `axis1` is out of bounds or `axis2` is out of bounds.
 *  - Memory cannot be allocated.
 */
PSMatrix PSMatrixSwapAxes(PSMatrix matrix, int axis1, int axis2) {
    if (matrix == NULL) return NULL;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    int orig_axis1 = axis1, orig_axis2 = axis2, i, j;
    if (axis1 < 0) axis1 = hdr->ndims + axis1;
    if (axis1 < 0 || axis1 >= hdr->ndims) {
        PSErr(__func__, "`axis1` is out of bound: %d (matrix has %d "
              "dimension(s))", orig_axis1, hdr->ndims);
    }
    if (axis2 < 0) axis2 = hdr->ndims + axis2;
    if (axis2 < 0 || axis2 >= hdr->ndims) {
        PSErr(__func__, "`axis2` is out of bound: %d (matrix has %d "
              "dimension(s))", orig_axis2, hdr->ndims);
    }
    if (axis1 == axis2) return PSMatrixDup(matrix);
    int last_axis = hdr->ndims - 1;
    int do_transp = (
        (axis1 == 0 && axis2 == last_axis) ||
        (axis1 == last_axis && axis2 == 0)
    );
    if (do_transp) return PSMatrixDup(PSMatrixTranspose(matrix, 0, NULL));
    long shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    for (i = 0; i < hdr->ndims; i++) {
        int idx = i;
        if (axis1 == i) idx = axis2;
        else if (axis2 == i) idx = axis1;
        shape[i] = hdr->shape[idx];
    }
    PSMatrix swapped = PSMatrixCreateWithShape(0, NULL, hdr->ndims, shape);
    if (swapped == NULL) {
        PSErr(__func__, "could not create swapped matrix");
        return NULL;
    }
    PSFloat *src_p = matrix, *dst_p = swapped;
    if ((axis1 == last_axis - 1 && axis2 == last_axis) ||
        (axis1 == last_axis && axis2 == last_axis - 1))
    {
        long stride = PSMatrixStride(swapped, 0);
        long rows = hdr->shape[last_axis - 1], cols = hdr->shape[last_axis];
        for (i = 0; i < shape[0]; i++) {
            PSFloat *transposed = PSVectorTranspose(
                src_p, dst_p, PSGlobalAcceleration, 2, rows, cols
            );
            if (transposed == NULL) {
                PSErr(__func__, "could not swap axes");
                PSMatrixFree(swapped);
                return NULL;
            }
            src_p += stride;
            dst_p += stride;
        }
    } else if ((axis1 == 0 && axis2 == 1) || (axis1 == 1 && axis2 == 0)) {
        long dst_stride = PSMatrixStride(swapped, 0),
            src_stride0 = PSMatrixStride(matrix, 0),
            src_stride1 = PSMatrixStride(matrix, 1);
        for (i = 0; i < shape[0]; i++) {
            for (j = 0; j < shape[1]; j++) {
                src_p = (matrix + (j * src_stride0));
                src_p += i * src_stride1;
                PSVectorCopy(dst_p + (j * src_stride1), src_p, src_stride1);
            }
            dst_p += dst_stride;
        }
    } else {
        PSErr(__func__, "unsupported axes for swapping");
        PSMatrixFree(swapped);
        return NULL;
    }
    return swapped;
}

/* Compare two martrices `a` and `b` having `length` length. Use `precision` to
 * set precision tolerance. Lower precision leads to higher tolerance.
 * By setting `precision` to zero, the two vectors must be perfectly
 * equal (no precision tolerance at all).
 * Returns: 1 if `a` and `b` equal, 0 if they differ at some point. */
int PSMatrixEquals(PSMatrix a, PSMatrix b, int precision, int ignore_shape) {
    if (a == NULL || b == NULL) return 0;
    long alen = PSMatrixLength(a), blen = 0;
    if (!ignore_shape) {
        long shape_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
        long shape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int ndims_a = PSMatrixShape(a, shape_a),
            ndims_b = PSMatrixShape(b, shape_b), i;
        if (ndims_a != ndims_b) return 0;
        for (i = 0; i < ndims_a; i++) {
            if (shape_a[i] != shape_b[i]) return 0;
        }
    } else {
        blen = PSMatrixLength(b);
        if (alen != blen) return 0;
    }
    return PSVectorEquals(a, b, alen, precision, NULL);
}

/* Check whether `matrix` has a cached trasposed version of itself (created
 * via `PSMatrixTranspose`).
 * Return value: 1 if `matrix` is not NULL and has a cached transposed version,
 * 0 otherwise. */
int PSMatrixHasTransposedVersion(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->transposed != NULL;
}

/* Check whether `matrix` is the cached trasposed version of another matrix
 * created by `PSMatrixTranspose`.
 * Return value: 1 if `matrix` is not NULL and is a cached transposed version,
 * 0 otherwise. */
int PSMatrixIsTransposedVersion(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->transposed_from != NULL;
}

static void freeMatrix(PSMatrix matrix) {
    if (matrix == NULL) return;
    void *ptr = (void *) getMatrixHeadPointer(matrix);
    free(ptr);
}

/* Invalidate and free the cached transposed version of `matrix`, if any (see
 * `PSMatrixTranspose`). */
void PSMatrixResetTransposed(PSMatrix matrix) {
    if (matrix == NULL) return;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (hdr->transposed != NULL) {
        freeMatrix(hdr->transposed);
        hdr->transposed = NULL;
    }
}

/* Free `matrix` by also deleting all its private data (including the cached
 * transposed versiob of `matrix` if any).
 * If `matrix` is NULL, the function will directly return. */
void PSMatrixFree(PSMatrix matrix) {
    if (matrix == NULL) return;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (hdr->transposed_from != NULL) {
        PSMatrixHeader *parent_hdr = PSMatrixGetHeader(hdr->transposed_from);
        assert(parent_hdr->transposed == matrix);
        PSMatrixResetTransposed(hdr->transposed_from);
        return;
    }
    if (hdr->transposed != NULL) freeMatrix(hdr->transposed);
    freeMatrix(matrix);
}

/**** Operations ****/

/* Add vector `b` to vector `a`. The argument `length` defines the length of
 * `a` and `b`, so both `a` and `b` must contain at least `length` elements.
 * The resulting vector will have the same length of `a` and `b` and each of
 * its elements will be the sum of the corresponding element of `a` and `b`
 * at the same index (`dest[i] = a[i] + b[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSAddVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                      PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE()
    UNUSED(debugStep);
#ifdef HAS_ACCELERATE_FRAMEWORK
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPAddV(a, b, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeSum(length, a, b, dest, i, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] + b[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] + b[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] + b[i];
            break;
    }
    return dest;
}

/* Subtract vector `b` from vector `a`. The argument `length` defines the
 * length of `a` and `b`, so both `a` and `b` must contain at least `length`
 * elements.
 * The resulting vector will have the same length of `a` and `b` and each of
 * its elements will be the subtraction of the corresponding element of `a`
 * and `b` at the same index (`dest[i] = a[i] - b[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest,
                           long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE()
#ifdef HAS_ACCELERATE_FRAMEWORK
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPSubV(a, b, dest, length);
        return dest;
    }
#elif defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDiff(length, a, b, dest, i, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] - b[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] - b[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] - b[i];
            break;
    }
    return dest;
}

/* Multiply vector `a` by vector `b`. The argument `length` defines the length
 * of `a` and `b`, so both `a` and `b` must contain at least `length`
 * elements.
 * The resulting vector will have the same length of `a` and `b` and each of
 * its elements will be the multiplication of the corresponding element of `a`
 * and `b` at the same index (`dest[i] = a[i] * b[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX, Accelerate Framework) can use some
 * storage modes to speed-up computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest,
                           long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode != PS_STORE_MODE_SUB) {
        if (mode == PS_STORE_MODE_SET)
            VDSPMulV(a, b, dest, length);
        else if (mode == PS_STORE_MODE_ADD)
            VDSPMulAddV(a, b, dest, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        long avx_step_len = AVXGetStepLen(length);
        long avx_steps = (
            avx_step_len > 0 ? length / avx_step_len : 0
        ), avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            PSFloat *x = a + i, *y = b + i, *d = dest + i;
            int c = AVXMultiply(x, y, length, d, mode);
            assert((long) c == avx_step_len);
            i += avx_step_len;
        }
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] * b[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] * b[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] * b[i];
            break;
    }
    return dest;
}

/* Divide vector `a` by vector `b`. The argument `length` defines the length
 * of `a` and `b`, so both `a` and `b` must contain at least `length`
 * elements.
 * The resulting vector will have the same length of `a` and `b` and each of
 * its elements will be the division of the corresponding element of `a`
 * and `b` at the same index (`dest[i] = a[i] / b[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest, long length,
                         PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPDivV(a, b, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        long avx_step_len = AVXGetStepLen(length);
        long avx_steps = (
            avx_step_len > 0 ? length / avx_step_len : 0
        ), avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            PSFloat *x = a + i, *y = b + i, *d = dest + i;
            int c = AVXDivide(x, y, length, d, mode);
            assert((long) c == avx_step_len);
            i += avx_step_len;
        }
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] / b[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] / b[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] / b[i];
            break;
    }
    return dest;
}

/* Multiply vector `a` by scalar `b`. The argument `length` defines the length
 * of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the multiplication of the corresponding element of `a`
 * at the same index by scalar value b (`dest[i] = a[i] * b`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPMulVS(a, b, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeMultiplyValue(length, a, b, dest, i, 0, 0, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] * b;
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] * b;
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] * b;
            break;
    }
    return dest;
}

/* Add scalar `b` to vector `a`. The argument `length` defines the length
 * of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the sum of the corresponding element of `a` at the same
 * index and scalar value b (`dest[i] = a[i] + b`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSAddVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                           long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPAddVS(a, b, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeAddValue(length, a, b, dest, i, 0, 0, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] + b;
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] + b;
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] + b;
            break;
    }
    return dest;
}

/* Subtract scalar `b` from vector `a`. The argument `length` defines the
 * length of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the result of the subtraction of `b` from the corresponding
 * element of `a` at the same index (`dest[i] = a[i] - b`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSSubtractVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                                long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        PSFloat invb = b * -1;
        VDSPAddVS(a, invb, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    /* NOTE: Currently noy supported! */
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] - b;
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] - b;
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] - b;
            break;
    }
    return dest;
}

/* Subtract vector `a` from scalar `b`. The argument `length` defines the
 * length of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the result of the subtraction of the corresponding element
 * of `a` ant the same index from value of `b` (`dest[i] = b - a[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                                long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        PSFloat invb = b * -1;
        VDSPAddVS(a, invb, dest, length);
        VDSPNeg(dest, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    /* NOTE: Currently noy supported! */
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = b - a[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += b - a[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= b - a[i];
            break;
    }
    return dest;
}

/* Divide vector `a` by scalar `b`. The argument `length` defines the length
 * of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the result of the corresponding element of `a` at the same
 * by the scalar value of `b` (`dest[i] = a[i] / b`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                              long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPDivVS(a, b, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDivValue(length, a, b, dest, i, 0, 0, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = a[i] / b;
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] / b;
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] / b;
            break;
    }
    return dest;
}

/* Divide scalar `b` by vector `a`. The argument `length` defines the length
 * of `a`.
 * The resulting vector will have the same length of `a` and each of its
 * elements will be the result of the division of the scalar value of `b` by
 * the corresponding element of `a` ant the same index (`dest[i] = b / a[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Some acceleration systems (ie. AVX) can use some storage modes to speed-up
 * computation.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                              long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK) && defined(__arm64__)
    /* This seems to lead to nan in x86 arch., so only use it with arm64 */
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPDivSV(b, a, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeValueDiv(length, a, b, dest, i, 0, 0, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = b / a[i];
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += b / a[i];
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= b / a[i];
            break;
    }
    return dest;
}

/* Compute hyperbolic tangent (tanh) on every element of vector `a` having
 * length defined by `length`.
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorTanh(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        int maxlen = INT_MAX;
        while (length > maxlen) {
            /* Apple vForce functions use int for index/length, but PsyC
             * uses long. In such cases, apply vForce function to multiple
             * segments of the vector(s). */
            VVTanh(a, dest, maxlen);
            length -= maxlen;
            if (length <= 0) return dest;
            a += maxlen;
            dest += maxlen;
        }
        VVTanh(a, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSTanh(a[i]);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSTanh(a[i]);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSTanh(a[i]);
            break;
    }
    return dest;
}

/* Compute base-e (Euler's number) exponential  on every element of vector `a`
 * having length defined by `length`.
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorExp(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts) {
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        int maxlen = INT_MAX;
        while (length > maxlen) {
            /* Apple vForce functions use int for index/length, but PsyC
             * uses long. In such cases, apply vForce function to multiple
             * segments of the vector(s). */
            VVExp(a, dest, maxlen);
            length -= maxlen;
            if (length <= 0) return dest;
            a += maxlen;
            dest += maxlen;
        }
        VVExp(a, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSExp(a[i]);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSExp(a[i]);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSExp(a[i]);
            break;
    }
    return dest;
}

/* Compute square root on every element of vector `a` having length defined by
 * `length`.
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorSqrt(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        int maxlen = INT_MAX;
        while (length > maxlen) {
            /* Apple vForce functions use int for index/length, but PsyC
             * uses long. In such cases, apply vForce function to multiple
             * segments of the vector(s). */
            VVSqrt(a, dest, maxlen);
            length -= maxlen;
            if (length <= 0) return dest;
            a += maxlen;
            dest += maxlen;
        }
        VVSqrt(a, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSSqrt(a[i]);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSSqrt(a[i]);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSSqrt(a[i]);
            break;
    }
    return dest;
}

/* Compute the negative value of every element of vector `a` having length
 * defined by `length` (`dest[i] = -a[i]`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorNeg(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts) {
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPNeg(a, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeNeg(length, a, dest, i, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = -(a[i]);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += -(a[i]);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= -(a[i]);
            break;
    }
    return dest;
}

/* Compute the absolute value of every element of vector `a` having length
 * defined by `length` (`dest[i] = abs(a[i])`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorAbs(PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts) {
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPAbs(a, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSAbs(a[i]);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSAbs(a[i]);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSAbs(a[i]);
            break;
    }
    return dest;
}

/* Clip values of vector `a` having length defined by `length` to minimum value
 * defined by `min` and maximum value defined by `max`
 * (`dest[i] = (a[i] < min ? min : (a[i] > max ? max : a[i]))`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                      long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPClip(a, min, max, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeClip(length, a, min, max, dest, i, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSClipValue(a[i], min, max);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSClipValue(a[i], min, max);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSClipValue(a[i], min, max);
            break;
    }
    return dest;
}

/* Clip values of vector `a` having length defined by `length` to minimum value
 * defined by `min` and maximum value of PSFloat (`PSFLOAT_MAX`).
 * (`dest[i] = (a[i] < min ? min : (a[i] > PSFLOAT_MAX ? PSFLOAT_MAX : a[i]))`).
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                           long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPThres(a, min, dest, length);
        return dest;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeClip(length, a, min, PSFLOAT_MAX, dest, i, mode);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i]=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i]+=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i]-=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
    }
    return dest;
}

/* Map values of vector `a` having length defined by `length` with the value
 * defined by `mapper`: values greater than `limit` will be represented with
 * the value of `mapper`, while values equal or less than `limit` will be
 * represented with negative value of `mapper` (`-(mapper)`);
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorMapWithLimit(PSFloat *a, PSFloat limit, PSFloat mapper,
                              PSFloat *dest, long length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPVLim(a, limit, mapper, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (i = 0; i < length; i++) {
        if (limit <= a[i]) dest[i] = mapper;
        else dest[i] = -mapper;
    }
    return dest;
}

/* Compute the maximum value among values of vector `a` having length defined
 * by `length`.
 * The optional pointer `index` can be used, if not NULL, to retrieve the index
 * of the maximum value.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the maxium value in the vector `a`. */
PSFloat PSVectorMax(PSFloat *a, long *index, long length,
                    PSMathOpts *opts)
{
    PSFloat max = PSFLOAT_MIN;
    if (a == NULL) {
        if (index != NULL) *index = 0;
        return max;
    }
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        if (index == NULL) VDSPMax(a, max, length);
        else VDSPMaxIdx(a, max, (unsigned long*) index, length);
        return max;
    }
#else
    UNUSED(acceleration);
#endif
    long i;
    if (index != NULL) *index = 0;
    for (i = 0; i < length; i++) {
        PSFloat n = a[i];
        if (n > max) {
            max = n;
            if (index != NULL) *index = i;
        }
    }
    return max;
}

/* Compute the sum of all the elements of vector `a` having length defined by
 * `length`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the sum of all the elements in the vector `a` or zero if
 * `a` is NULL. */
PSFloat PSVectorReduceSum(PSFloat *a, long length, PSMathOpts *opts) {
    if (a == NULL) return 0.0;
    PSFloat sum = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        VDSPSumElems(a, sum, length);
        return sum;
    }
#else
    UNUSED(acceleration);
#endif
    long i;
    for (i = 0; i < length; i++) sum += a[i];
    return sum;
}

/* Raise each value of vector `a` having length `length` to power of `exp`.
 * If the value of `exp` is 2, the function will just call `PSMultiplyVectors`
 * function, mutiplying `a` by itself.
 * Results are stored into the optional `dest` arguments. If `dest` is NULL,
 * a new vector will be allocated and its address will be  returned by the
 * function itself.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * WARN: by using `store_mode`, acceleration will be currently disabled.
 * Return value: the pointer to the address of the vector containing results.
 * If `dest` is not NULL, the return value is `dest` itself, but if `dest`
 * is NULL, the return value is the address of the newly allocated vector.
 * The function returns NULL if `dest` is NULL but the destination vector
 * cannot be allocated in memory. */
PSFloat *PSVectorPower(PSFloat *a, PSFloat exp, PSFloat *dest, long length,
                       PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
    if (exp == 1) {
        if (dest == a) return dest;
        else if (dest != NULL) {
            PSVectorCopy(dest, a, length);
            return dest;
        }
    } else if (exp == 2) return PSMultiplyVectors(a, a, dest, length, opts);
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        PSFloat exps_vec[length];
        PSFloat *exps = exps_vec;
        VDSPFill(exp, exps, length);
        int maxlen = INT_MAX;
        while (length > maxlen) {
            /* Apple vForce functions use int for index/length, but PsyC
             * uses long. In such cases, apply vForce function to multiple
             * segments of the vector(s). */
            VVPow(a, exps, dest, length);
            length -= maxlen;
            if (length <= 0) return dest;
            a += maxlen;
            dest += maxlen;
            exps += maxlen;
        }
        VVPow(a, exps, dest, length);
        return dest;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case PS_STORE_MODE_SET:
            for (; i < length; i++) dest[i] = PSPow(a[i], exp);
            break;
        case PS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSPow(a[i], exp);
            break;
        case PS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSPow(a[i], exp);
            break;
    }
    return dest;
}

/* Compute the cumulative sum on elements of vector `a` having length defined by
 * `length`. The results will be stored into the vector `dest` that must have
 * at least the same length of `a`. The value of each element of the resulting
 * vector will be the sum of the values of `a` up to the index of the current
 * resulting vector element (ie. `dest[2] = a[0] + a[1] + a[2]`).
 * Return value: 1 if the function is successfully executed or 0 if:
 *  - `a` is NULL or `dest` is NULL.
 *  - `length` is zero or negative. */
long PSCumulativeSum(PSFloat *a, PSFloat *dest, long length) {
    if (a == NULL || dest == NULL) {
        PSErr(__func__, "`a` and `dest` cannot be null");
        return 0;
    }
    if (length <= 0) {
        PSErr(__func__, "`length` must be > 0");
        return 0;
    }
    PSFloat sum = 0.0;
    for (long i = 0; i < length; i++) {
        sum += a[i];
        dest[i] = sum;
    }
    return 1;
}

/* Compute the mean value of the elements of vector `a` having length defined
 * by `length`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the mean value of vector `a` values or zero if `a` is NULL. */
PSFloat PSMean(PSFloat *a, long length, PSMathOpts *opts) {
    PSFloat mean = 0.0;
    if (a == NULL || length <= 0) return mean;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        VDSPMean(a, mean, length);
        return mean;
    }
#else
    UNUSED(acceleration);
#endif
    long i;
    for (i = 0; i < length; i++) mean += a[i];
    mean = mean / (PSFloat) length;
    return mean;
}

/* Compute the statistical variance of the elements of vector `a` having
 * length defined by `length`.
 * The variance is the sum of the squared difference of the difference between
 * each value of `a` and the mean value of `a`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the variance of vector `a` values or zero if `a` is NULL. */
PSFloat PSVariance(PSFloat *a, long len, PSMathOpts *opts) {
    if (len <= 0 || a == NULL) return 0;
    PSFloat var = 0.0;
    PSFloat *cache = NULL;
    PSFloat mean = PSMean(a, len, opts);
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        cache = opts->tmpdest;
    }
    int do_free_cache = (cache == NULL);
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        if (cache == NULL) cache = malloc(len * sizeof(PSFloat));
        if (cache != NULL) {
            PSSubtractVectorScalar(a, mean, cache, len, opts);
            PSMultiplyVectors(cache, cache, cache, len, opts);
            var = PSMean(cache, len, opts);
            goto final;
        }
    }
#else
    UNUSED(acceleration);
#endif
    PSFloat sum = 0.0;
    for (long i = 0; i < len; i++) {
        PSFloat d = (a[i] - mean);
        sum += (d * d);
    }
    var = sum / (PSFloat) len;
final:
    if (do_free_cache) free(cache);
    return var;
}

/* Compute the standard deviation of the elements of vector `a` having
 * length defined by `length`.
 * The standard deviation is the square root of the statistical variance (see
 * `PSVariance`).
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the variance of vector `a` values or zero if `a` is NULL. */
PSFloat PSStdDev(PSFloat *a, long len, PSMathOpts *opts) {
    PSFloat variance = PSVariance(a, len, opts);
    return PSSqrt(variance);
}

/* Compute the dot product of vector `a` and vector `b`, both having length
 * defined by `length`.
 * The dot product is the sum of the product of each element of `a` by the
 * corresponding element of `b` at the same index
 * (`a[0] * b[0] + a[1] * b[1] + ... + a[n] * b[n]`).
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Return value: the resulting dot product (scalar) or zero if `a` is NULL
 * or `b` is NULL.*/
PSFloat PSDotProduct(PSFloat *a, PSFloat *b, long length, PSMathOpts *opts)
{
    if (a == NULL || b == NULL) return 0;
    PSDotProductDebug debugStep = NULL;
    long i = 0;
    PSFloat result = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        debugStep = opts->debugStep;
    }
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        VDSPDotProd(a, b, result, length);
        if (debugStep)
            debugStep(length - 1, a[length-1], b[length-1], result, 1, opts);
        return result;
    }
#elif defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDotProduct(length, a, b, result, i, 0, 0);
        if (debugStep) debugStep(i, a[i], b[i], result, 1, opts);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (; i < length; i++) {
        if (debugStep) debugStep(i, a[i], b[i], result, 0, opts);
        result += a[i] * b[i];
    }
    return result;
}

PSFloat PSDotSquare(PSFloat *a, long length, PSMathOpts *opts) {
    PSDotProductDebug debugStep = NULL;
    long i = 0;
    PSFloat result = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        debugStep = opts->debugStep;
    }
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        VDSPSumVecSqr(a, result, length);
        if (debugStep)
            debugStep(length - 1, a[length-1], a[length-1], result, 1, opts);
        return result;
    }
#elif defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDotSquare(length, a, result, i, 0, 0);
        if (debugStep) debugStep(i, a[i], a[i], result, 1, opts);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (; i < length; i++) {
        if (debugStep) debugStep(i, a[i], a[i], result, 0, opts);
        result += a[i] * a[i];
    }
    return result;
}

/* Performs matrix-matrix multiplication, matrix-vector multiplication,
 * vector-matrix multiplication or vector-vector multiplication,
 * depending on the value of `argtype` field in opts (default is matrix-matrix).
 * The resulting vector is stored into `dest`.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * NOTE: if `argtype` for both `a` and `b` is 'V', the function will compute
 * the dot product of the two vectors, assuming that they have the same size.
 * If you need to perform matrix multiplication on two PSFloat arrays,
 * use `PSMatMul` instead.
 * Return value: 1 in case of success, 0 in case of failure.
 * Possible failure reasons:
 *  - `a` is NULL or `b` is NULL or `dest` is NULL.
 *  - Memory allocation failure.
 *  - `a` is vector or `b` is vector and the resulting length would be zero.
 *  - Both `a` and `b` are vectors but `vector_len` member of optional `opts`
 *    argument is zero or `opts` is NULL. */
int PSDot(PSMatrix a, PSMatrix b, PSMatrix dest, PSMathOpts *opts) {
    if (a == NULL || b == NULL || dest == NULL) {
        if (a == NULL) PSErr(__func__, "`a` cannot be null");
        if (b == NULL) PSErr(__func__, "`b` cannot be null");
        if (dest == NULL) PSErr(__func__, "`dest` cannot be null");
        return 0;
    }
    int store_mode = PS_STORE_MODE_SET;
    long dims_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
    long dims_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
    char df_argtype[] = {'M', 'M', 'M'};
    char *argtype = df_argtype;
    int transpose = 0;
    PSFloat *tmpdest = NULL;
    if (opts != NULL) {
        argtype = opts->argtype;
        store_mode = opts->store_mode;
        tmpdest = opts->tmpdest;
        transpose = opts->transpose;
    }
    int a_is_vec = argtype[0] == 'V' || argtype[0] == 'v';
    int b_is_vec = argtype[1] == 'V' || argtype[1] == 'v';
    if (!a_is_vec && !b_is_vec) {
        /* matrix-matrix multiplication */
        PSMatrix tmpmatrix = NULL;
        PSMatrix *dstptr = &dest;
        if (store_mode == PS_STORE_MODE_SUB) {
            opts->store_mode = PS_STORE_MODE_SET;
            tmpmatrix = PSMatrixDupShape(dest);
            if (tmpmatrix == NULL) return 0;
            dstptr = &tmpmatrix;
        }
        int ok = PSMatrixProduct(a, b, dstptr, opts);
        if (!ok) {
            PSMatrixFree(tmpmatrix);
            return 0;
        }
        if (store_mode == PS_STORE_MODE_SUB) {
            PSSubtractVectors(dest, tmpmatrix, dest, PSMatrixLength(dest),opts);
            PSMatrixFree(tmpmatrix);
        }
        return 1;
    } else if (!a_is_vec && b_is_vec) {
        /* matrix-vector multiplication */
        int ndims = PSMatrixShape(a, dims_a);
        long len = (transpose & 1 ? dims_a[0] : dims_a[ndims - 1]);
        int do_free_tmpdest = 0;
        if (len <= 0) {
            PSErr(__func__, "Could not perform matrix-vector multiplication, "
                  "destinaton length would be %d", len);
            return 0;
        }
        PSFloat **dstptr = &dest;
        if (store_mode == PS_STORE_MODE_SUB) {
            opts->store_mode = PS_STORE_MODE_SET;
            if (tmpdest == NULL) {
                tmpdest = calloc(len, sizeof(PSFloat));
                if (tmpdest == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
                do_free_tmpdest = 1;
            }
            dstptr = &tmpdest;
        }
        int ok = PSMatrixProductMV(a, b, len, dstptr, opts);
        if (!ok) {
            if (do_free_tmpdest) free(tmpdest);
            return 0;
        }
        if (store_mode == PS_STORE_MODE_SUB) {
            PSSubtractVectors(dest, tmpdest, dest, len, opts);
            if (do_free_tmpdest) free(tmpdest);
        }
        return 1;
    } else if (a_is_vec && !b_is_vec) {
        /* vector-matrix multiplication */
        PSMatrixShape(b, dims_b);
        long len = dims_b[1];
        int do_free_tmpdest = 0;
        if (len <= 0) {
            PSErr(__func__, "Could not perform vector-matrix multiplication, "
                  "destinaton length would be %d", len);
            return 0;
        }
        PSFloat **dstptr = &dest;
        if (store_mode == PS_STORE_MODE_SUB) {
            opts->store_mode = PS_STORE_MODE_SET;
            if (tmpdest == NULL) {
                tmpdest = calloc(len, sizeof(PSFloat));
                if (tmpdest == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
                do_free_tmpdest = 1;
            }
            dstptr = &tmpdest;
        }
        int ok = PSMatrixProductVM(a, b, len, dstptr, opts);
        if (!ok) {
            if (do_free_tmpdest) free(tmpdest);
            return 0;
        }
        if (store_mode == PS_STORE_MODE_SUB) {
            PSSubtractVectors(dest, tmpdest, dest, len, opts);
            if (do_free_tmpdest) free(tmpdest);
        }
        return 1;
    } else {
        long len = 0;
        if (opts != NULL) len = opts->vector_len;
        if (len <= 0) {
            PSErr(
                __func__, "opts->vector_len is mandatory and must be > 0 when "
                "both `a` and `b` are treated as vectors"
            );
            return 0;
        }
        PSFloat res = PSDotProduct(a, b, len, opts);
        if (store_mode == PS_STORE_MODE_ADD) dest[0] += res;
        else if (store_mode == PS_STORE_MODE_SUB) dest[0] -= res;
        else dest[0] = res;
        return 1;
    }
    return 1;
}

/* Perform matrix-vector multiplication by calling `PSDot` and setting
 * `argtype` member of `opts` to `argtype[0] = 'M', argtype[1] = 'V'`.
 * See `PSDot` for a more detailed description.
 * NOTE: `opts` argument is optional, and if given, it's never overwritten
 * by the function since its values are copied to a local structure.
 * Return value: See `PSDot`. */
int PSDotMV(PSMatrix a, PSFloat *b, PSFloat *dest, PSMathOpts *opts) {
    PSMathOpts myopts = {0};
    if (opts != NULL) myopts = *opts;
    myopts.argtype[0] = 'M';
    myopts.argtype[1] = 'V';
    return PSDot(a, b, dest, &myopts);
}

/* Perform vector-matrix multiplication by calling `PSDot` and setting
 * `argtype` member of `opts` to `argtype[0] = 'V', argtype[1] = 'M'`.
 * See `PSDot` for a more detailed description.
 * NOTE: `opts` argument is optional, and if given, it's never overwritten
 * by the function since its values are copied to a local structure.
 * Return value: See `PSDot`. */
int PSDotVM(PSFloat *a, PSMatrix b, PSMatrix dest, PSMathOpts *opts) {
    PSMathOpts myopts = *opts;
    myopts.argtype[0] = 'V';
    myopts.argtype[1] = 'M';
    return PSDot(a, b, dest, &myopts);
}

/* Multiply every element of vector `a` (having `alen` length) by every
 * element of vector `b` (having `blen` length) and store results into
 * vector `dest` (whose length must be the product of `alen` by `blen`).
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * Aside from acceleration, `opts` can also be used to set the result storage
 * mode (by using the `store_mode` member):
 *  - `PS_STORE_MODE_ADD`: the result is added to the existing values of `dest`.
 *  - `PS_STORE_MODE_SUB`: the result is subtracted from the existing values
 *     of `dest`.
 * Return value: 1 is the function succeeds or zero if:
 *  - `a` is NULL or `b` is NULL or `dest` is NULL.
 *  - BLAS computation error if BLAS acceleration is used. */
int PSOuterProduct(PSFloat *a, PSFloat *b, PSFloat *dest,
                   long alen, long blen, PSMathOpts *opts)
{
    if (a == NULL || b == NULL || dest == NULL) {
        PSErr(__func__, "`a`, `vector` and `dest` cannot be null");
        return 0;
    }
    PSFloat *tmpdest = NULL;
    int store_mode = PS_STORE_MODE_SET;
    int acceleration = PSGlobalAcceleration;
    long i, j;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        store_mode = opts->store_mode;
        tmpdest = opts->tmpdest;
    }
    long dstlen = alen * blen;
    int postprocess = store_mode != PS_STORE_MODE_SET;
#if defined(HAS_BLAS) || defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PSFloat *vpdest = dest;
    int blas_enabled = PSBLASEnabled(acceleration),
        acf_enabled = PSAccelerateEnabled(acceleration),
        do_free_vpdest = 0;
    if (blas_enabled && dstlen > PSBLAS_MAX) {
        blas_enabled = 0;
        PSWarn(
            "%s: BLAS has been disabled since some dimensions exceeds "
            "PSBLAS_MAX (%ld)", __func__, PSBLAS_MAX
        );
    }
    int use_acceleration = (blas_enabled || acf_enabled);
    if (!use_acceleration) goto no_acceleration;
    if (store_mode && (store_mode == PS_STORE_MODE_SUB || !blas_enabled)) {
        vpdest = tmpdest;
        if (vpdest == NULL) vpdest = malloc(dstlen * sizeof(PSFloat));
        if (vpdest == NULL) {
            PSPrintMemoryErrorMsg();
            return 0;
        }
        do_free_vpdest = (vpdest != dest && vpdest != tmpdest);
    }
#ifdef HAS_BLAS
    if (blas_enabled) {
        postprocess = (store_mode == PS_STORE_MODE_SUB);
        PSBLASOrder order = PSBLASRowMajor;
        char trans1 = 'N', trans2 = 'N';
        long m = 1, lda = 1, ldb = blen, ldc = blen;
        PSFloat beta = 0.0;
        if (store_mode == PS_STORE_MODE_ADD) {
            beta = 1.0;
            store_mode = PS_STORE_MODE_SET;
        }
        PSGemm(order, trans1, trans2, alen, blen, m, 1.0, a, lda, b, ldb, beta,
               vpdest, ldc);
        if (PSBLASLastError != NULL) return 0;
        goto acceleration_done;
    }
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    if (acf_enabled) {
        VDSPMMul(a, b, vpdest, alen, blen, 1);
        goto acceleration_done;
    }
#endif
acceleration_done:
    if (postprocess) {
        PSMathOpts ppopts = {.acceleration = acceleration};
        if (store_mode == PS_STORE_MODE_ADD)
            PSAddVectors(dest, vpdest, dest, dstlen, &ppopts);
        else if (store_mode == PS_STORE_MODE_SUB)
            PSSubtractVectors(dest, vpdest, dest, dstlen, &ppopts);
    }
    if (do_free_vpdest) free(vpdest);
    return 1;
#else
    UNUSED(tmpdest);
    UNUSED(acceleration);
    UNUSED(dstlen);
#endif
no_acceleration:
    if (!postprocess) postprocess = store_mode != PS_STORE_MODE_SET;
    for (i = 0; i < alen; i++) {
        for (j = 0; j < blen; j++) {
            long idx = (blen * i) + j;
            PSFloat product = (a[i] * b[j]);
            if (!store_mode) dest[idx] = product;
            if (!postprocess) continue;
            if (store_mode == PS_STORE_MODE_ADD) dest[idx] += product;
            else if (store_mode == PS_STORE_MODE_SUB) dest[idx] -= product;
        }
    }
    return 1;
}

/* Fill vector `vec` having length defined by `len` with `value`. The function
 * will immediately return if `vec` is NULL or `len` is zero.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`. */
void PSVectorFill(PSFloat *vec, PSFloat val, long len, PSMathOpts *opts) {
    if (len <= 0 || vec == NULL) return;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        VDSPFill(val, vec, len);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    if (val == 0.0) {
        PSVectorClear(vec, len);
        return;
    }
    for (long i = 0; i < len; i++) vec[i] = val;
}

/* Write a string representation of vector `vec` having length of `len`
 * to the file stream `f`.
 * The optional `sep` argument can be used to specify a separator string for
 * vector's values (if `sep` is null, by default "," is used as separator).
 * If `vec` is null or `f` is null, the function will immediately return. */
size_t PSVectorWrite(PSFloat *vec, long len, char* sep, FILE *f) {
    if (vec == NULL || f == NULL) return 0;
    if (sep == NULL) sep = ",";
    return writeSerializedFloatArray(f, len, sep, 0, vec);
}

/* Print a string representation of vector `vec` having length of `len` to
 * the standard output.
 * The optional `sep` argument can be used to specify a separator string for
 * vector's values (if `sep` is null, by default "," is used as separator).
 * If `vec` is NULL the function will immediately return. */
void PSVectorPrint(PSFloat *vec, long len, char* sep) {
    PSVectorWrite(vec, len, sep, stdout);
    printf("\n");
}

/* Create a transposed version of `vec`, considering it a matrix with a shape
 * of `ndims` dimensions.
 * Variadic arguments can be used to define the shape of the matrix (that must
 * have max. `PS_MATRIX_MAX_DIMENSIONS` dimensions).
 * If `dest` is not NULL, the transposed vector will be stored into memory
 * pointed by `dest` itself.
 * If `dest` is NULL, the resulting vector will be allocated by the function.
 * The function can take advantage of the available accelerations (both
 * hardwware and software). By default, accelerations set in
 * `PSGlobalAcceleration` are used, if any. However, the used accelerations
 * methods can be changed via the `acceleration` member of the optional
 * argument `opts`.
 * NOTE:
 *  - The shape defined by the variadic dimensions refer to original shape of
 *    the vector (seen as a matrix) and not to the resulting transposed vector.
 *    So, if the defined shape is 2,3, the resulting shape will be 3,2.
 *  - If you need to transpose a `PSMatrix`, directly use `PSMatrixTranspose`
 *    instead.
 * Return value: a pointer to the transposed vector, whose size will be the
 * product of all dimensions of the defined shape or NULL if something goes
 * wrong.
 * If `dest` is not NULL, return value will be `dest` or NULL if
 * something goes wrong.
 * If `ndims` is 1, the function will immediately return `vec` itself.
 * Possible failure reasons:
 *  - `ndims` is greater that `PS_MATRIX_MAX_DIMENSIONS`.
 *  - `ndims` is zero or less than zero.
 *  - One of the shape's dimension in the variadic arguments is zero or
 *    less than zero.
 *  - Memory allocation failure.
 */
PSFloat *PSVectorTranspose(PSFloat *vec, PSFloat *dest, int acceleration,
                           int ndims, ...)
{
    if (ndims == 1) return vec;
    else if (ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "`ndims` must be <= %d", PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    } else if (ndims <= 0) {
        PSErr(__func__, "`ndims` must be > 0");
        return NULL;
    }
    long shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    long x, y, z, idx, i, veclen = 1;
    va_list args;
    va_start(args, ndims);
    for (i = 0; i < (long) ndims; i++) {
        shape[i] = va_arg(args, int);
        veclen *= shape[i];
    }
    va_end(args);
    if (veclen <= 0) {
        PSErr(__func__, "Invalid dimensions: each dimension must be > 0");
        return NULL;
    }
    PSFloat *transposed = dest;
    if (transposed == NULL) transposed = malloc(veclen * sizeof(PSFloat));
    if (transposed == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    long ncols = 0, nrows = 0, dlen = 0, t_ncols = 0, t_dlen = 0;
    if (ndims == 3) {
        nrows = shape[1];
        ncols = shape[2];
        t_ncols = shape[0];
        t_dlen = shape[0] * shape[1];
        dlen = nrows * ncols;
        for (i = 0; i < veclen; i++) {
            z = i % dlen % ncols;
            y = (i / ncols) % nrows;
            x = i / dlen;
            idx = (z * t_dlen) + (y * t_ncols) + x;
            transposed[idx] = vec[i];
        }
    } else if (ndims == 2) {
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        if (PSAccelerateEnabled(acceleration)) {
            /* Use Apple(R) Accelerate Framework */
            VDSPMTransp(vec, transposed, shape[1], shape[0]);
            goto final;
        }
#else
        UNUSED(acceleration);
#endif
        ncols = shape[1];
        t_ncols = shape[0];
        for (i = 0; i < veclen; i++) {
            y = i % ncols;
            x = i / ncols;
            idx = (y * t_ncols) + x;
            transposed[idx] = vec[i];
        }
    }
final:
    return transposed;
}

/* Perform matrix multiplication between vectors (PSFloat arrays) `a` and `b`.
 * If you need to perform matrix multiplication with involve at least one
 * `PSMatrix`, then use `PSMatrixProduct` (matrix-matrix), `PSMatrixProductMV`
 * (matrix-vector) or `PSMatrixProductVM` (vector-matrix) instead.
 * You can set matrix transposition using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the vector arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose `a`)
 * Results will be stored in `dest`, that must be at least `m` * `n` long.
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Other arguments:
 *  - `m`: number of rows in `a` and result `dest`
 *  - `n`: number of columns in `b` and `dest`.
 *  - `k`: number of columns in `a` and rows in `n`.
 * NOTE: if you set transposition for `a` or `b`, `m`,`n` and `k` will
 * refer to rows and columns of the transposed matrix.
 * Return value: 1 in case of success, 0 in case of failure. */
int PSMatMul(PSFloat *a, PSFloat *b, PSFloat *dest, long m, long n, long k,
             PSMathOpts *opts)
{
    if (a == NULL || b == NULL || dest == NULL) {
        PSErr(__func__, "argument `a`,`b` and `dest` cannot be null");
        return 0;
    }
    int transpose = 0;
    int acceleration = PSGlobalAcceleration;
    int store_mode = PS_STORE_MODE_SET;
    if (opts != NULL) {
        transpose = opts->transpose;
        acceleration = opts->acceleration;
        store_mode = opts->store_mode;
    }
    int transpose_a = transpose & 1;
    int transpose_b = transpose & 2;
#ifdef HAS_BLAS
    int use_blas = PSBLASEnabled(acceleration);
#else
    int use_blas = 0;
    UNUSED(acceleration);
#endif
    if (use_blas && (m * n * k) > PSBLAS_MAX) {
        use_blas = 0;
        PSWarn(
            "%s: BLAS has been disabled since some dimensions exceeds "
            "PSBLAS_MAX (%ld)", __func__, PSBLAS_MAX
        );
    }
    if (use_blas) {
        char trans_a = 'N', trans_b = 'N';
        long lda = k, ldb = n, ldc = n;
        if (transpose_a) {
            trans_a = 'T';
            lda = m;
        }
        if (transpose_b) {
            trans_b = 'T';
            ldb = k;
        }
        PSFloat beta = (store_mode == PS_STORE_MODE_ADD ? 1.0 : 0.0);
        PSGemm(PSBLASRowMajor, trans_a, trans_b, m, n, k, 1.0, a, lda, b, ldb,
               beta, dest, ldc);
        return (PSBLASLastError == NULL);
    }
    PSMathOpts mopts = {.acceleration = acceleration};
    int do_add = (store_mode == PS_STORE_MODE_ADD), success = 1;
    PSFloat *transposed_a = NULL, *transposed_b = NULL, *orig_b = b;
    if (transpose_a) {
        transposed_a = PSVectorTranspose(a, NULL, acceleration, 2, k, m);
        if (transposed_a == NULL) return 0;
        a = transposed_a;
    }
    if (transpose_b) {
        transposed_b = PSVectorTranspose(b, NULL, acceleration, 2, n, k);
        if (transposed_b == NULL) return 0;
        b = transposed_b;
    }
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSAccelerateEnabled(acceleration)) {
        long outlen = m * n;
        if (outlen == 0) return 0;
        PSFloat *out = dest, *tmpdest = (opts ? opts->tmpdest : NULL);
        if (do_add) {
            if (tmpdest == NULL)
                tmpdest = calloc(outlen, sizeof(PSFloat));
            if (tmpdest == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            out = tmpdest;
        }
        VDSPMMul(a, b, out, m, n, k);
        if (do_add) {
            PSAddVectors(dest, out, dest, outlen, &mopts);
            if (opts == NULL || tmpdest != opts->tmpdest)
                free(tmpdest);
        }
        goto final;
    }
#endif
    long a_rows = m, a_cols = k, b_rows = k, b_cols = n, out_rows, out_cols;
    if (transpose_b) b = orig_b;
    else {
        transposed_b = PSVectorTranspose(b, NULL, acceleration, 2, k, n);
        b = transposed_b;
    }
    out_rows = a_rows;
    out_cols = b_cols;
    if (a_cols != b_rows) {
        PSErr(__func__, "Aligment error: a columns != b rows -> %d != %d",
              a_cols, b_rows);
    }
    k = b_rows;
    for (long i = 0; i < out_rows; i++) {
        for (long j = 0; j < out_cols; j++) {
            long oidx = (i * out_cols) + j;
            PSFloat *arow = a + (a_cols * i);
            PSFloat *brow = b + (k * j);
            if (!do_add) dest[oidx] = PSDotProduct(arow, brow, k, &mopts);
            else dest[oidx] += PSDotProduct(arow, brow, k, &mopts);
        }
    }
final:
    free(transposed_a);
    free(transposed_b);
    return success;
}

/* Create a matrix with shape `size`, `size` diagonally filled with 1.0 from
 * the top-left side to the bottom-right side, example:
 * ```
 * PSDiagonalMask(4); // ->
 * // {1, 0, 0, 0,
 * //  1, 1, 0, 0,
 * //  1, 1, 1, 0,
 * //  1, 1, 1, 1}
 * ```
 * Return value: the matrix of NULL if:
 *  - `size` is less than 1
 *  - The matrix cannot be allocated in memory. */
PSMatrix PSDiagonalMask(long size) {
    if (size <= 0) {
        PSErr(__func__, "invalid size: %d", size);
        return NULL;
    }
    PSMatrix mask = PSMatrixZeros(2, size, size);
    if (mask == NULL) return NULL;
    for (long r = 0; r < size; r++) {
        for (long c = 0; c < size; c++) {
            if (c <= r) {
                mask[(r * size) + c] = 1.0;
            }
        }
    }
    return mask;
}

/* Create a matrix of shape `len`, `len` where values of vector `vec` having
 * length defined by `len` are distributed over a diagonal line starting from
 * top-left side and ending to bottom-right side.
 * Example:
 * ```
 * PSFLoat vec[4] = {1, 2, 3, 4};
 * PSDiagonalFlattenVector(vec, 4); // ->
 * // {1, 0, 0, 0,
 * //  0, 2, 0, 0,
 * //  0, 0, 3, 0,
 * //  0, 0, 0 ,4}
 * ```
 * Return value: the matrix or NULL if:
 *  - `vec` is NULL or `len` is zero.
 *  - The matrix cannot be allocated in memory. */
PSMatrix PSDiagonalFlattenVector(PSFloat *vec, long len) {
    if (vec == NULL || len == 0) return NULL;
    PSMatrix result = PSMatrixZeros(2, len, len);
    if (result == NULL) {
        PSErr(__func__, "could not create result matrix");
        return NULL;
    }
    PSFloat *res_p = result;
    for (long i = 0; i < len; i++) {
        res_p[i] = vec[i];
        res_p += len;
    }
    return result;
}

/* Create a new squared matrix from source `matrix` having a shape with rows
 * and columns equal to `matrix` length (matrix length x matrix length).
 * The original values of `matrix` are distributed into the new matrix over a
 * diagonal line starting from top-left side and ending to bottom-right side.
 * Example:
 * ```
 * PSFLoat vec[4] = {1, 2, 3, 4};
 * PSMatrix src = PSMatrixFromArray(vec, 2, 2, 2); // 2x2 matrix, total len = 4
 * PSMatrix new = PSDiagonalFlatten(src); // ->
 * // {1, 0, 0, 0,
 * //  0, 2, 0, 0,
 * //  0, 0, 3, 0,
 * //  0, 0, 0 ,4}
 * ```
 * Return value: the matrix or NULL if:
 *  - `matrix` is NULL or `matrix` is empty.
 *  - The matrix cannot be allocated in memory. */
PSMatrix PSDiagonalFlatten(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    long len = PSMatrixLength(matrix);
    if (len == 0) return NULL;
    return PSDiagonalFlattenVector(matrix, len);
}

/* Split vector `vec` having length defined by `len` into `num_slices` vectors.
 * For example, a vector of 10 elements split into two slices will create two
 * vectors of size 5.
 * Return value: an array of `num_slices` vectors (PSFloat *) or NULL if:
 *  - `vec` is NULL.
 *  - `len` is zero or negative.
 *  - `num_slices` is zero or negative.
 *  - `len` / `num_slices` does not result in equal division.
 *  - Memory allocation issues. */
PSFloat **PSVectorSplit(PSFloat *vec, long len, long num_slices) {
    if (vec == NULL) return NULL;
    if (len <= 0) {
        PSErr(__func__, "`len` must be > 0");
        return NULL;
    }
    if (num_slices <= 0) {
        PSErr(__func__, "`num_slices` must be > 0");
        return NULL;
    }
    if ((len % num_slices) != 0) {
        PSErr(__func__, "vector split does not result in equal division");
        return NULL;
    }
    long slice_size = len / num_slices;
    PSFloat **vectors = calloc(num_slices, sizeof(PSFloat *));
    if (vectors == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSFloat *vec_p = vec;
    for (int i = 0; i < num_slices; i++) {
        PSFloat *slice = malloc(slice_size * sizeof(PSFloat));
        if (slice == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        vectors[i] = slice;
        PSVectorCopy(slice, vec_p, slice_size);
        vec_p += slice_size;
    }
    return vectors;
fail:
    if (vectors != NULL) {
        for (int i = 0; i < num_slices; i++) {
            free(vectors[i]);
        }
        free(vectors);
    }
    return NULL;
}

/* Duplicate vector `vec` having length defined by `length`.
 * Return value: the duplicated vector or NULL is memory cannot be allocated.*/
PSFloat *PSVectorDup(PSFloat *src, long length) {
    size_t size = (size_t) length * sizeof(PSFloat);
    PSFloat *dup = malloc(size);
    if (dup == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    memcpy(dup, src, size);
    return dup;
}

/* Allocate a new vector having length defined by `len` and fill it with random
 * values within a range of 0.0 and 1.0.
 * Return value: the allocated vector or NULL if memory cannot be allocated. */
PSFloat *PSVectorRandom(long len) {
    PSFloat *vec = malloc((size_t) len * sizeof(PSFloat));
    if (vec == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    long i;
    for (i = 0; i < len; i++) vec[i] = PSNormalizedRandom();
    return vec;
}

/* Compare two floats `a` and `b`. Use `precision` to set precision tolerance.
 * Lower precision leads to higher tolerance.
 * By setting `precision` to zero, the two numbers must be perfectly
 * equal (no precision tolerance at all).
 * Returns: 1 if `a` and `b` equal, 0 if they differ. */
int PSFloatEquals(PSFloat a, PSFloat b, int precision) {
    static PSFloat precision_table[23] = {0};
    if (precision > 0) {
        if (precision > 22) precision = 22;
        PSFloat diff = fabs(a - b), maxdiff;
        maxdiff = precision_table[precision];
        if (maxdiff == 0) {
            maxdiff = PSPow(10, precision * -1);
            precision_table[precision] = maxdiff;
        }
        return diff <= maxdiff;
    }
    return a == b;
}

/* Compare two vectors `a` and `b` having `length` length. Use `precision` to
 * set precision tolerance. Lower precision leads to higher tolerance.
 * By setting `precision` to zero, the two vectors must be perfectly
 * equal (no precision tolerance at all).
 * Use `index` pointer if you need to know the index of the first
 * non-equal elements.
 * Returns: 1 if `a` and `b` equal, 0 if they differ at some point. */
int PSVectorEquals(PSFloat *a, PSFloat *b, long length, int precision,
                   long *index)
{
    long i;
    for (i = 0; i < length; i++) {
        if (!PSFloatEquals(a[i], b[i], precision)) {
            if (index != NULL) *index = i;
            return 0;
        }
    }
    return 1;
}

/* Convert the vector `vec` of length `len` to a `PSMatrix`. This function
 * differs from `PSMatrixFromArray` since it reallocates the vector in order
 * to make room for the matrix header that will contain matrix's properties.
 * So the vector is reallocated and its memory is moved by the size of the
 * matrix header.
 * It's possible to specify the matrix's shape by using the `ndims` argument
 * and the `shape` argument:
 *  - `ndims`: number of dimensions (axes) of the matrix shape.
 *  - `shape`: the shape itself.
 * If `shape` is NULL or `ndims` is zero, the function will use a default shape
 * of {`len`} (if `ndims` is 0 or 1) or {1, `len`} (if `ndims` is 2).
 * The function will fail if `shape` is NULL and `ndims` is greater than 2.
 * WARN: if the function succeeds, it's not possible to use the source vector
 * `vec` anymore, since its data have been moved in memory and the original
 * address could have been reallocated.
 * WARN: the vector `vec` must be an array of `PSFloat` that was previously
 * allocated (ie. by using `PSVectorCreate`, `PSVectorDup`, `malloc`, `calloc`
 * or `realloc`). Using global/static arrays or arrays from the stack frame
 * will lead to memory corruption.
 * Return value: the matrix or NULL if the function fails.
 * Possible failure reasons:
 *  - `vec` is NULL.
 *  - `len` is zero.
 *  - `ndims` is greater than `PS_MATRIX_MAX_DIMENSIONS`.
 *  - `shape` is NULL but `ndims` is greater than 2.
 *  - `len` mismatches `shape` (`len` must equals the product of shape axes).
 *  - Memory allocation failure */
PSMatrix PSVectorConvertToMatrix(PSFloat *vec, long len, int ndims,
                                 long *shape)
{
    if (vec == NULL) {
        PSErr(__func__, "argument`vec` cannot be null");
        return NULL;
    }
    if (len == 0) {
        PSErr(__func__, "vector is empty");
        return NULL;
    }
    if (ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "shape would exceed max dimensions %d",
              PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    long dfshape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int i;
    if (shape == NULL) {
        if (ndims > 2) {
            PSErr(__func__, "argument `shape` cannot be null if `ndims` > 2");
            return NULL;
        }
        shape = dfshape;
    }
    if (ndims <= 0) {
        shape = dfshape;
        ndims = 1;
    }
    if (shape == dfshape) {
        if (len > LONG_MAX) {
            PSErr(NULL, "vector length exceeds shape dimensions");
            return NULL;
        }
        if (ndims == 1) shape[0] = len;
        else if (ndims == 2) {
            shape[0] = 1;
            shape[1] = len;
        }
    } else {
        long shape_len = 1;
        for (i = 0; i < ndims; i++) shape_len *= shape[i];
        if (shape_len != len) {
            PSErr(NULL, "invalid shape for length %" PRIu64, len);
            return NULL;
        }
    }
    size_t vecsize = len * sizeof(PSFloat);
    size_t newsize = PSMatrixHeaderSize + vecsize;
    uint8_t *new = realloc(vec, newsize);
    if (new == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSMatrix matrix = (PSMatrix)(new + PSMatrixHeaderSize);
    memmove(matrix, new, vecsize);
    PSMatrixHeader *hdr = (PSMatrixHeader *) new;
    memset(hdr, 0, PSMatrixHeaderSize);
    hdr->length = len;
    hdr->ndims = ndims;
    for (i = 0; i < ndims; i++) hdr->shape[i] = shape[i];
    hdr->transposed = NULL;
    hdr->transposed_from = NULL;
    return matrix;
}

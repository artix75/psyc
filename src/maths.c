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

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <assert.h>

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
#define VVSqrt(a,dest,len) vvsqrt(dest, a, (int *)&len)
#define VVTanh(a,dest,len) vvtanh(dest, a, (int *)&len)
#define VVExp(a,dest,len)  vvexp(dest, a, (int *)&len)

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
#define VVSqrt(a,dest,len) vvsqrtf(dest, a, (int *)&len)
#define VVTanh(a,dest,len) vvtanhf(dest, a, (int *)&len)
#define VVExp(a,dest,len)  vvexpf(dest, a, (int *)&len)

#endif

#endif

#define UNUSED(V) ((void) V)
#define MATHS_OPERATION_PREAMBLE() \
    if (dest == NULL) dest = a;\
    PSDotProductDebug debug_step = NULL;\
    uint64_t i = 0;\
    int acceleration = PSGlobalAcceleration, mode = MATHS_STORE_MODE_NORM;\
    if (opts != NULL) {\
        acceleration = opts->acceleration;\
        mode = opts->store_mode;\
        debug_step = opts->debug_step;\
        assert(mode >= 0 && mode <= MATHS_STORE_MODE_SUB);\
        if (opts->after != NULL) {\
            PSWarn("PSMathOpts `after` is only used by `PSDot` and "\
                   "`PSVectorProduct`");\
        }\
    }\
    UNUSED(debug_step);

#define MAX_DIMENSIONS 3

/**** Utils ****/
static unsigned char randomSeeded = 0;

PSFloat PSNormalizedRandom() {
    if (!randomSeeded) {
        randomSeeded = 1;
        srand(time(NULL));
    }
    int r = rand();
    return ((PSFloat) r / (PSFloat) RAND_MAX);
}

PSFloat PSGaussianRandom(PSFloat mean, PSFloat stddev) {
    PSFloat theta = 2 * M_PI * PSNormalizedRandom();
    PSFloat rho = PSSqrt(-2 * PSMathLog(1 - PSNormalizedRandom()));
    PSFloat scale = stddev *rho;
    PSFloat x = mean + scale *cos(theta);
    PSFloat y = mean + scale *sin(theta);
    PSFloat r = PSNormalizedRandom();
    return (r > 0.5 ? y : x);
}

/**** PSMatrix ****/

#define PSMatrixGetHeader(matrix) \
    ((PSMatrixHeader *) getMatrixHeadPointer(matrix))

typedef struct PSMatrixHeader {
    int ndims;
    int dims[MAX_DIMENSIONS];
    PSMatrix transposed;
    PSMatrix transposed_from;
    uint64_t length;
} PSMatrixHeader;

static const size_t PSMatrixHeaderSize = sizeof(PSMatrixHeader);

static char *getMatrixHeadPointer(PSMatrix matrix) {
    return ((char *) matrix) - PSMatrixHeaderSize;
}

static PSFloat matrixGaussianRandomInitializer(PSMatrix matrix, int idx,
                                               PSFloat stddev)
{
    UNUSED(matrix);
    UNUSED(idx);
    if (stddev == 0.0) stddev = 1.0;
    return PSGaussianRandom(0, stddev);
}

static PSFloat matrixRandomInitializer(PSMatrix matrix, int idx, PSFloat val) {
    UNUSED(matrix);
    UNUSED(idx);
    UNUSED(val);
    return PSNormalizedRandom();
}

PSMatrix PSMatrixCreateWithDims(PSFloat init_value,
                                PSMatrixInitializer initializer,
                                int ndims, int *dims)
{
    if (ndims < 1 || ndims > MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d", MAX_DIMENSIONS);
        return NULL;
    }
    int len = 1, i;
    for (i = 0; i < ndims; i++) {
        int d = dims[i];
        if (d <= 0) {
            if (d == 0) {
                PSWarn("%s: invalid dimension[%d] = %d", __func__, i, d);
                return NULL;
            } else {
                PSWarn("%s: invalid dimension[%d] = %d", __func__, i, d);
                ndims = i + 1;
                dims[i] = d;
                break;
            }
        }
        len *= d;
        dims[i] = d;
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
    for (i = 0; i < MAX_DIMENSIONS; i++) {
        if (i < ndims) hdr->dims[i] = dims[i];
        else hdr->dims[i] = 0;
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
    if (ndims < 1 || ndims > MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d", MAX_DIMENSIONS);
        return NULL;
    }
    int dims[MAX_DIMENSIONS];
    for (int i = 0; i < ndims; i++) {
        int d = va_arg(args, int);
        dims[i] = d;
    }
    return PSMatrixCreateWithDims(init_value, initializer, ndims, dims);
}

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

PSMatrix PSMatrixZeros(int ndims, ...) {
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0.0, NULL, ndims, args);
    va_end(args);
    return matrix;
}

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

PSMatrix PSMatrixRandom(int ndims, ...) {
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0, matrixRandomInitializer, ndims, args);
    va_end(args);
    return matrix;
}

PSMatrix PSMatrixFromArray(PSFloat *array, int ndims, ...) {
    if (array == NULL) return NULL;
    PSMatrix matrix = NULL;
    va_list args;
    va_start(args, ndims);
    matrix = PSMatrixCreateV(0, matrixRandomInitializer, ndims, args);
    va_end(args);
    size_t array_size = PSMatrixLength(matrix) * sizeof(PSFloat);
    memcpy(matrix, array, array_size);
    return matrix;
}

PSMatrix PSMatrixDup(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    uint64_t len = PSMatrixLength(matrix);
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

int PSMatrixCopy(PSMatrix src, PSMatrix dst) {
    if (src == NULL) {
        PSErr(__func__, "`src` matrix is NULL");
        return 0;
    }
    if (dst == NULL) {
        PSErr(__func__, "`dst` matrix is NULL");
        return 0;
    }
    int src_dims[MAX_DIMENSIONS];
    int dst_dims[MAX_DIMENSIONS];
    int src_ndims = PSMatrixDimensions(src, src_dims),
        dst_ndims = PSMatrixDimensions(dst, dst_dims),
        src_len = PSMatrixLength(src), i;
    if (src_ndims != dst_ndims) {
        PSErr(__func__, "`src` dimensions != `dst` dimensions: %d != %d",
              src_ndims, dst_ndims);
        return 0;
    }
    for (i = 0; i < src_ndims; i++) {
        if (src_dims[i] != dst_dims[i]) {
            PSErr(
                __func__, "`src` dimension[%d] != `dst`: %d != %d",
                i, src_dims[i], dst_dims[i]
            );
            return 0;
        }
    }
    PSMatrixHeader *dst_hdr = PSMatrixGetHeader(dst);
    dst_hdr->transposed_from = NULL;
    if (dst_hdr->transposed != NULL) {
        PSMatrixDelete(dst_hdr->transposed);
        dst_hdr->transposed = NULL;
    }
    memcpy(dst, src, src_len * sizeof(PSFloat));
    return 1;
}

/* Expand matrix `src` by adding `add` to its first dimension. Added data will
 * be set to zero.
 * Beware of the fact that `src` matrix could be freed after the process,
 * so always assing the return value of this function to a new variable, since
 * it could lead to memory leaks in case of a NULL return value. Also beware of
 * the fact that the original variable holding `src` could point to freed
 * memry after function returns. */
PSMatrix PSMatrixExpand(PSMatrix src, int add) {
    if (src == NULL) return NULL;
    if (add <= 0) return src;
    PSMatrix matrix = NULL;
    PSMatrixHeader *src_hdr = PSMatrixGetHeader(src);
    if (src_hdr->transposed_from != NULL) {
        matrix = PSMatrixExpand(src_hdr->transposed_from, add);
        if (matrix == NULL) return NULL;
        PSMathOpts opts = {.acceleration = PSGlobalAcceleration};
        return PSMatrixTranspose(matrix, 1, &opts);
    }
    if (src_hdr->transposed != NULL) {
        PSMatrixDelete(src_hdr->transposed);
        src_hdr->transposed = NULL;
    }
    int new_dim = src_hdr->dims[0] + add;
    uint64_t cur_len = src_hdr->length, new_len = (uint64_t) new_dim;
    for (int i = 1; i < src_hdr->ndims; i++) new_len *= src_hdr->dims[i];
    size_t datasize = ((size_t) new_len * sizeof(PSFloat));
    size_t size = PSMatrixHeaderSize + datasize;
    PSMatrixHeader *new_hdr = realloc(src_hdr, size);
    if (new_hdr == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    new_hdr->dims[0] = new_dim;
    new_hdr->length = new_len;
    new_hdr->transposed = NULL;
    new_hdr->transposed_from = NULL;
    matrix = (PSMatrix) (new_hdr + 1);
    size_t added_size = ((size_t)(new_len - cur_len) * sizeof(PSFloat));
    memset(((PSFloat *) matrix) + cur_len, 0, added_size);
    return matrix;
}

int PSMatrixNumDims(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->ndims;
}

int PSMatrixDim(PSMatrix matrix, int dim) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (dim >= hdr->ndims) return 0;
    return hdr->dims[dim];
}

int PSMatrixDimensions(PSMatrix matrix, int *dims) {
    if (matrix == NULL) return 0;
    int ndims = PSMatrixNumDims(matrix);
    if (dims == NULL) return ndims;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    memcpy(dims, hdr->dims, (MAX_DIMENSIONS * sizeof(int)));
    return ndims;
}

size_t PSMatrixLength(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->length;
}

int PSMatrixStride(PSMatrix matrix, int dim) {
    if (matrix == NULL) return 0;
    int refdim = dim + 1;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (refdim >= MAX_DIMENSIONS || refdim >= hdr->ndims) return 1;
    return hdr->dims[refdim];
}

PSFloat *PSMatrixGet(PSMatrix matrix, int ndims, uint32_t *len, ...) {
    if (matrix == NULL) return NULL;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    PSFloat *values = matrix;
    int stride = 1;
    if (ndims > hdr->ndims) ndims = hdr->ndims;
    va_list args;
    va_start(args, len);
    for (int i = 0; i < ndims; i++) {
        int refdim = i + 1;
        if (refdim >= hdr->ndims) stride = 1;
        else stride = hdr->dims[refdim];
        int idx = va_arg(args, int);
        if (idx >= hdr->dims[i]) {
            PSWarn("%s: index %d (%d) is out of bounds", i, idx, hdr->dims[i]);
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

/* Performs matrix-vector multiplication between matrix `a` and vector `b`.
 * Results are stored into vector pointed by pointer `result`. If pointer
 * pointed by `result` is NULL, a new vector is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * By defaults, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build, function will compute results by
 * using `PSMultiplyVectors` as fallback.
 * By default, data in result vector will be overwritten. Anyaway, if
 * `MATHS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if ti fails. */
int PSMatrixProductMV(PSMatrix a, PSFloat *b, int len, PSFloat **result,
                      PSMathOpts *opts)
{
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    PSBLASOrder order = PSBLASRowMajor;
    int dims_a[MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(a, dims_a);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
#ifndef HAS_BLAS
    if (ndims > 2) {
        PSErr(__func__, "BLAS is disabled: max. 2 dimensions allowed");
        return 0;
    }
#endif
    int l = dims_a[ndims - 1];
    if (len != l) {
        PSErr(__func__, "Aligment error: vector len != a dim[%d] -> "
              "%d != %d", len, l);
        return 0;
    }
    int nd = ndims - 1, outlen;
    if (nd == 1 || nd == 0) outlen = (ndims == 2 ? dims_a[0] : len);
    else if (nd == 2) outlen = dims_a[0] * len;
    else {
        PSErr(__func__, "Invalid output dimensions: %d", nd);
        return 0;
    }
    PSFloat *out = *result;
    if (out == NULL) {
        out = *result = calloc(outlen, sizeof(PSFloat));
        if (out == NULL) return 0;
    }
    if (ndims == 1) {
        /* Just multiply two vectors */
        PSMultiplyVectors(a, b, out, len, NULL);
        return 1;
    }
    int lda = (dims_a[1] > 1 ? dims_a[1] : 1);
    int m = dims_a[0], n = dims_a[1];
    PSFloat beta = 0.0;
    if (opts != NULL && opts->store_mode == MATHS_STORE_MODE_ADD)
        beta = 1.0;
#ifndef HAS_BLAS
    int do_add = (beta == 1.0);
    for (int i = 0; i < m; i += lda) {
        if (!do_add) out[i] = PSDotProduct(a, b, n, opts);
        else out[i] += PSDotProduct(a, b, n, opts);
    }
    return 1;
#endif
    PSGemv(order, 'N', m, n, 1.0, a, lda, b, 1, beta, out, 1);
    if (PSBLASLastError != NULL) return 0;
    return 1;
}

/* Performs vector-matrix multiplication between vector `a` and matrix `b`.
 * Results are stored into matrix pointed by `result`. If pointer
 * pointed by `result` is NULL, a new matrix is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * The function uses BLAS to compute the result, so, if BLAS support is
 * missing in PsyC build, function will fail.
 * By default, data in result vector will be overwritten. Anyaway, if
 * `MATHS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if ti fails. */
int PSMatrixProductVM(PSFloat *a, PSMatrix b, int len, PSMatrix *result,
                      PSMathOpts *opts)
{
#ifndef HAS_BLAS
    PSErr(__func__, "BLAS is disabled");
    return 0;
#endif
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    PSBLASOrder order = PSBLASRowMajor;
    int dims_b[MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(b, dims_b);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
    int dimensions[MAX_DIMENSIONS] = {0};
    if (dims_b[0] != len) {
        PSErr(__func__, "Aligment error: b dim[0] != vector length -> "
              "%d != %d", dims_b[0], len);
        return 0;
    }
    int nd = 1 + ndims - 2;
    if (nd == 1) dimensions[0] = dims_b[1];
    else if (nd == 2) {
        dimensions[0] = len;
        dimensions[1] = dims_b[1];
    } else {
        PSErr(__func__, "Invalid output dimensions: %d", nd);
        return 0;
    }
    PSMatrix out = *result;
    if (out != NULL) {
        PSMatrixHeader *hdr = PSMatrixGetHeader(out);
        if (hdr->ndims != nd) {
            PSErr(
                __func__, "`result` matrix has %d dimension(s), but "
                "%d dimension(s) needed", hdr->ndims, nd
            );
            return 0;
        }
        for (int i = 0; i < nd; i++) {
            int odim = PSMatrixDim(out, i);
            if (odim != dimensions[i]) {
                PSErr(__func__, "`result` matrix dimension [%d] is %d, "
                      "but it should be %d", i, odim, dimensions[i]);
                return 0;
            }
        }
    } else {
        out = PSMatrixCreateWithDims(0, NULL, nd, dimensions);
        *result = out;
        if (out == NULL) return 0;
    }
    int lda = (dims_b[1] > 1 ? dims_b[1] : 1);
    int m = dims_b[0], n = dims_b[1];
    PSFloat beta = 0.0;
    if (opts != NULL && opts->store_mode == MATHS_STORE_MODE_ADD) beta = 1.0;
    PSGemv(order, 'N', m, n, 1.0, b, lda, a, 1, beta, out, 1);
    if (PSBLASLastError != NULL) return 0;
    return 1;
}

/* Performs matrix-matrix multiplication between matrix `a` and vector `b`.
 * Results are stored into matrix pointed by `result`. If pointer
 * pointed by `result` is NULL, a new matrix is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * By defaults, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build and `b` only has one dimension, function
 * will try compute results by using `PSMultiplyVectors` as fallback (for all
 * other cases, it will fail).
 * By default, data in result vector will be overwritten. Anyaway, if
 * `MATHS_STORE_MODE_ADD` is set as `store_mode` into `opt`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if ti fails. */
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt) {
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    int dims_a[MAX_DIMENSIONS];
    int dims_b[MAX_DIMENSIONS];
    int ndims_a = PSMatrixDimensions(a, dims_a);
    int ndims_b = PSMatrixDimensions(b, dims_b);
    if (ndims_a == 0) {
        PSErr(__func__, "Invalid matrix `a`");
        return 0;
    }
    if (ndims_b == 0) {
        PSErr(__func__, "Invalid matrix `b`");
        return 0;
    }
    int lda = 0, ldb = 0, l = 0, i;
    int dimensions[MAX_DIMENSIONS] = {0};
    l = dims_a[ndims_a - 1];
    if (dims_b[0] != l) {
        PSErr(__func__, "Aligment error: b dim[0] != a dim[%d] -> "
              "%d != %d", (ndims_a - 1), dims_b[0], l);
        return 0;
    }
    int nd = ndims_a + ndims_b - 2;
    if (nd == 1) dimensions[0] = (ndims_a == 2 ? dims_a[0] : dims_b[1]);
    else if (nd == 2) {
        dimensions[0] = dims_a[0];
        dimensions[1] = dims_b[1];
    } else {
        PSErr(__func__, "Invalid output dimensions: %d", nd);
        return 0;
    }
    PSMatrix out = *result;
    if (out != NULL) {
        PSMatrixHeader *hdr = PSMatrixGetHeader(out);
        if (hdr->ndims != nd) {
            PSErr(
                __func__, "`result` matrix has %d dimension(s), but "
                "%d dimension(s) needed", hdr->ndims, nd
            );
            return 0;
        }
        for (i = 0; i < nd; i++) {
            int odim = PSMatrixDim(out, i);
            if (odim != dimensions[i]) {
                PSErr(__func__, "`result` matrix dimension [%d] is %d, "
                      "but it should be %d", i, odim, dimensions[i]);
                return 0;
            }
        }
    } else {
        out = PSMatrixCreateWithDims(0, NULL, nd, dimensions);
        *result = out;
        if (out == NULL) return 0;
    }
    int a_vector_like = (ndims_a == 1),
        b_vector_like = (ndims_b == 1);
    PSBLASOrder order;
    PSFloat beta = 0.0;
    if (opt != NULL && opt->store_mode == MATHS_STORE_MODE_ADD) beta = 1.0;
    if (!a_vector_like && b_vector_like) {
        /* Matrix vector multiplication -- Level 2 BLAS */
#ifndef HAS_BLAS
        int do_add = (beta == 1.0);
        for (int i = 0; i < m; i += lda) {
            if (!do_add) out[i] = PSDotProduct(a, b, n, NULL);
            else out[i] += PSDotProduct(a, b, n, NULL);
        }
        return 1;
#endif
        order = PSBLASRowMajor;
        lda = (dims_a[1] > 1 ? dims_a[1] : 1);
        int bs = PSMatrixStride(b, 0);
        int m = dims_a[0], n = dims_a[1];
        PSGemv(order, 'N', m, n, 1.0, a, lda, b, bs, beta, out, 1);
    } else if (a_vector_like && !b_vector_like) {
        /* Vector matrix multiplication -- Level 2 BLAS */
#ifndef HAS_BLAS
        PSErr(__func__, "BLAS disabled");
        return 0;
#endif
        order = PSBLASRowMajor;
        lda = (dims_b[1] > 1 ? dims_b[1] : 1);
        int as = PSMatrixStride(a, 0);
        int m = dims_b[0], n = dims_b[1];
        PSGemv(order, 'N', m, n, 1.0, b, lda, a, as, beta, out, 1);
    } else {
        /* Matrix matrix multiplication -- Level 3 BLAS */
#ifndef HAS_BLAS
        PSErr(__func__, "BLAS disabled");
        return 0;
#endif
        order = PSBLASRowMajor;
        char trans1 = 'N', trans2 = 'N';
        int m = dims_a[0];
        int n = dims_b[1];
        int k = dims_b[0];
        lda = (dims_a[1] > 1 ? dims_a[1] : 1);
        ldb = (dims_b[1] > 1 ? dims_b[1] : 1);
        size_t alen = PSMatrixLength(a), blen = PSMatrixLength(b);
        if (alen == blen &&
           dims_a[0] == dims_b[1] &&
           dims_a[1] == dims_b[0] &&
           PSMatrixStride(a, 0) == PSMatrixStride(b, 1) &&
           PSMatrixStride(a, 1) == PSMatrixStride(b, 0) &&
           (trans1 == 'T' ? 1 : 0) ^ (trans2 == 'T' ? 1 : 0) &&
           (trans1 == 'N' ? 1 : 0) ^ (trans2 == 'N' ? 1 : 0)) {
            PSErr(__func__, "Unsupported BLAS Syrc");
        } else {
            int odim1 = PSMatrixDim(out, 1);
            int ldc = ((odim1 > 1) ? odim1 : 1);
            PSGemm(order, trans1, trans2, m, n, k, 1.0, a, lda, b, ldb, beta,
                   out, ldc);
        }
    }
    if (PSBLASLastError != NULL) return 0;
    return 1;
}

PSMatrix PSMatrixTranspose(PSMatrix matrix, int rebuild, PSMathOpts *opts) {
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    PSMatrixHeader *t_hdr = NULL;
    if (hdr->transposed != NULL) {
        if (!rebuild) return hdr->transposed;
        PSMatrixDelete(hdr->transposed);
        hdr->transposed = NULL;
    } else if (hdr->transposed_from != NULL) return hdr->transposed_from;
    int ndims = hdr->ndims;
    int *dims = hdr->dims;
    int ncols = 0, nrows = 0, dlen = 0, t_ncols = 0, t_dlen = 0;
    uint64_t x, y, z, idx, i;
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
        if (PSACFEnabled(acceleration)) {
            /* Use Apple Accelerate Framework */
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

void PSMatrixResetTransposed(PSMatrix matrix) {
    if (matrix == NULL) return;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (hdr->transposed != NULL) {
        PSMatrixDelete(hdr->transposed);
        hdr->transposed = NULL;
    }
}

void PSMatrixDelete(PSMatrix matrix) {
    if (matrix == NULL) return;
    void *ptr = (void *) getMatrixHeadPointer(matrix);
    PSMatrixHeader *hdr = (PSMatrixHeader *) ptr;
    if (hdr->transposed != NULL) PSMatrixDelete(hdr->transposed);
    free(ptr);
}

/**** Operations ****/

void PSSumVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                  PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE()
    UNUSED(debug_step);
#ifdef HAS_ACCELERATE_FRAMEWORK
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPAddV(a, b, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] + b[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] + b[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] + b[i];
            break;
    }
}

void PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE()
#ifdef HAS_ACCELERATE_FRAMEWORK
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPSubV(a, b, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] - b[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] - b[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] - b[i];
            break;
    }
}

void PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode != MATHS_STORE_MODE_SUB) {
        if (mode == MATHS_STORE_MODE_NORM)
            VDSPMulV(a, b, dest, length);
        else if (mode == MATHS_STORE_MODE_ADD)
            VDSPMulAddV(a, b, dest, dest, length);
        return;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        uint64_t avx_step_len = AVXGetStepLen(length);
        uint64_t avx_steps = (
            avx_step_len > 0 ? length / avx_step_len : 0
        ), avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            PSFloat *x = a + i, *y = b + i;
            int c = AVXMultiply(x, y, length, dest, mode);
            assert((uint64_t) c == avx_step_len);
            i += avx_step_len;
        }
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] * b[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] * b[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] * b[i];
            break;
    }
}

void PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPDivV(a, b, dest, length);
        return;
    }
#endif
#if defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        uint64_t avx_step_len = AVXGetStepLen(length);
        uint64_t avx_steps = (
            avx_step_len > 0 ? length / avx_step_len : 0
        ), avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            PSFloat *x = a + i, *y = b + i;
            int c = AVXDivide(x, y, length, dest, mode);
            assert((uint64_t) c == avx_step_len);
            i += avx_step_len;
        }
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] / b[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] / b[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] / b[i];
            break;
    }
}

void PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPMulVS(a, b, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] * b;
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] * b;
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] * b;
            break;
    }
}

void PSSumVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPAddVS(a, b, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] + b;
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] + b;
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] + b;
            break;
    }
}

void PSSubtractVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        PSFloat invb = b * -1;
        VDSPAddVS(a, invb, dest, length);
        return;
    }
#endif
#if defined(USE_AVX)
    /* NOTE: Currently noy supported! */
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i]- b;
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] - b;
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] - b;
            break;
    }
}

void PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        PSFloat invb = b * -1;
        VDSPAddVS(a, invb, dest, length);
        VDSPNeg(dest, dest, length);
        return;
    }
#endif
#if defined(USE_AVX)
    /* NOTE: Currently noy supported! */
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = b - a[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += b - a[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= b - a[i];
            break;
    }
}

void PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPDivVS(a, b, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = a[i] / b;
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += a[i] / b;
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= a[i] / b;
            break;
    }
}

void PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPDivSV(b, a, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = b / a[i];
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += b / a[i];
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= b / a[i];
            break;
    }
}

void PSVectorTanh(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VVTanh(a, dest, length);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = PSTanh(a[i]);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSTanh(a[i]);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSTanh(a[i]);
            break;
    }
}

void PSVectorExp(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VVExp(a, dest, length);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = PSExp(a[i]);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSExp(a[i]);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSExp(a[i]);
            break;
    }
}

void PSVectorSqrt(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VVSqrt(a, dest, length);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = PSSqrt(a[i]);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSSqrt(a[i]);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSSqrt(a[i]);
            break;
    }
}

void PSVectorNeg(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPNeg(a, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = -(a[i]);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += -(a[i]);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= -(a[i]);
            break;
    }
}

void PSVectorAbs(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPAbs(a, dest, length);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    switch (mode) {
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = PSAbs(a[i]);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSAbs(a[i]);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSAbs(a[i]);
            break;
    }
}

void PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                  uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPClip(a, min, max, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i] = PSClipValue(a[i], min, max);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i] += PSClipValue(a[i], min, max);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i] -= PSClipValue(a[i], min, max);
            break;
    }
}

void PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPThres(a, min, dest, length);
        return;
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
        case MATHS_STORE_MODE_NORM:
            for (; i < length; i++) dest[i]=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
        case MATHS_STORE_MODE_ADD:
            for (; i < length; i++) dest[i]+=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
        case MATHS_STORE_MODE_SUB:
            for (; i < length; i++) dest[i]-=PSClipValue(a[i],min,PSFLOAT_MAX);
            break;
    }
}

void PSVectorMapWithLimit(PSFloat *a, PSFloat limit, PSFloat mapper,
                          PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == MATHS_STORE_MODE_NORM) {
        VDSPVLim(a, limit, mapper, dest, length);
        return;
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (i = 0; i < length; i++) {
        if (limit <= a[i]) dest[i] = mapper;
        else dest[i] = -mapper;
    }
}

PSFloat PSVectorMax(PSFloat *a, uint64_t *index, uint64_t length,
                    PSMathOpts *opts)
{
    PSFloat max = PSFLOAT_MIN;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
        if (index == NULL) VDSPMax(a, max, length);
        else VDSPMaxIdx(a, max, (unsigned long*) index, length);
        return max;
    }
#else
    UNUSED(acceleration);
#endif
    uint64_t i;
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

PSFloat PSSumVectorElements(PSFloat *a, uint64_t length, PSMathOpts *opts) {
    PSFloat sum = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
        VDSPSumElems(a, sum, length);
        return sum;
    }
#else
    UNUSED(acceleration);
#endif
    uint64_t i;
    for (i = 0; i < length; i++) sum += a[i];
    return sum;
}

PSFloat PSMean(PSFloat *a, uint64_t length, PSMathOpts *opts) {
    PSFloat mean = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
        VDSPMean(a, mean, length);
        return mean;
    }
#else
    UNUSED(acceleration);
#endif
    uint64_t i;
    for (i = 0; i < length; i++) mean += a[i];
    mean = mean / (PSFloat) length;
    return mean;
}

PSFloat PSVariance(PSFloat *a, uint64_t len, PSMathOpts *opts) {
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
    if (PSACFEnabled(acceleration)) {
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
    for (uint64_t i = 0; i < len; i++) {
        PSFloat d = (a[i] - mean);
        sum += (d * d);
    }
    var = sum / (PSFloat) len;
final:
    if (do_free_cache) free(cache);
    return var;
}

PSFloat PSStdDev(PSFloat *a, uint64_t len, PSMathOpts *opts) {
    PSFloat variance = PSVariance(a, len, opts);
    return PSSqrt(variance);
}

PSFloat PSDotProduct(PSFloat *a, PSFloat *b, uint64_t length, PSMathOpts *opts)
{
    PSDotProductDebug debug_step = NULL;
    uint64_t i = 0;
    PSFloat result = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        debug_step = opts->debug_step;
    }
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
        VDSPDotProd(a, b, result, length);
        if (debug_step)
            debug_step(length - 1, a[length-1], b[length-1], result, 1, opts);
        return result;
    }
#elif defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDotProduct(length, a, b, result, i, 0, 0);
        if (debug_step) debug_step(i, a[i], b[i], result, 1, opts);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (; i < length; i++) {
        if (debug_step) debug_step(i, a[i], b[i], result, 0, opts);
        result += a[i] * b[i];
    }
    return result;
}

PSFloat PSDotSquare(PSFloat *a, uint64_t length, PSMathOpts *opts) {
    PSDotProductDebug debug_step = NULL;
    uint64_t i = 0;
    PSFloat result = 0.0;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        debug_step = opts->debug_step;
    }
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
        VDSPSumVecSqr(a, result, length);
        if (debug_step)
            debug_step(length - 1, a[length-1], a[length-1], result, 1, opts);
        return result;
    }
#elif defined(USE_AVX)
    if (PSAVXEnabled(acceleration)) {
        AVXIterativeDotSquare(length, a, result, i, 0, 0);
        if (debug_step) debug_step(i, a[i], a[i], result, 1, opts);
    }
#else
    UNUSED(acceleration);
#endif
    /* No Acceleration */
    for (; i < length; i++) {
        if (debug_step) debug_step(i, a[i], a[i], result, 0, opts);
        result += a[i] * a[i];
    }
    return result;
}

int PSDot(PSMatrix matrix, PSFloat *vector, PSFloat *dest, PSMathOpts *opts) {
    if (matrix == NULL || vector == NULL || dest == NULL) {
        if (matrix == NULL) PSErr(__func__, "`matrix` cannot be null");
        if (vector == NULL) PSErr(__func__, "`vector` cannot be null");
        if (dest == NULL) PSErr(__func__, "`dest` cannot be null");
        return 0;
    }
    int dims[MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(matrix, dims);
    if (ndims != 2) {
        PSErr(__func__, "Invalid matrix: only 2D matrix allowed");
        return 0;
    }
    int rows = dims[0], len = dims[1], i;
    int acceleration = PSGlobalAcceleration;
    PSFloat *vec2add = NULL, *max = NULL, *tmpdest = NULL;
    PSFloatFunc after = NULL;
    int store_mode = MATHS_STORE_MODE_NORM;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        vec2add = opts->add_vec;
        after = opts->after;
        max = opts->max;
        store_mode = opts->store_mode;
        tmpdest = opts->tmpdest;
    }
    int do_process = (
        vec2add != NULL || after != NULL || max != NULL
    );
    /* TODO: implement "auto" acceleration type selection */
#ifdef HAS_BLAS
    if (PSBLASEnabled(acceleration)) {
        PSFloat *dpdest = dest;
        if (store_mode && store_mode != MATHS_STORE_MODE_ADD) {
            dpdest = tmpdest;
            if (dpdest == NULL) dpdest = malloc(rows * sizeof(PSFloat));
            if (dpdest == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            do_process = 1;
        }
        int do_free_dpdest = (dpdest != dest && dpdest != tmpdest);
        if (!PSMatrixProductMV(matrix, vector, len, &dpdest, opts)) {
            if (do_free_dpdest) free(dpdest);
            return 0;
        }
        if (do_process) {
            for (i = 0; i < rows; i++) {
                if (store_mode == MATHS_STORE_MODE_SUB) dest[i]-=dpdest[i];
                if (vec2add != NULL) dest[i] += vec2add[i];
                if (after != NULL) dest[i] = after(dest[i]);
                if (max != NULL && (i == 0 || dest[i] > *max)) *max = dest[i];
            }
        }
        if (do_free_dpdest) free(dpdest);
        return 1;
    }
#else
    UNUSED(tmpdest);
    UNUSED(acceleration);
    if (!do_process) do_process = store_mode != MATHS_STORE_MODE_NORM;
#endif
    PSFloat *mptr = matrix;
    for (i = 0; i < rows; i++) {
        PSFloat dp = PSDotProduct(mptr, vector, len, opts);
        if (!store_mode) dest[i] = dp;
        mptr += len;
        if (!do_process) continue;
        if (store_mode == MATHS_STORE_MODE_ADD) dest[i] += dp;
        else if (store_mode == MATHS_STORE_MODE_SUB) dest[i] -= dp;
        if (vec2add != NULL) dest[i] += vec2add[i];
        if (after != NULL) dest[i] = after(dest[i]);
        if (max != NULL && (i == 0 || dest[i] > *max)) *max = dest[i];
    }
    return 1;
}

/* Multiply every element of vector `a` (having `alen` length) by every
 * element of vector `b` (having `blen` length) and store results into
 * vector `dest` (whose length must be `alen` * `blen`). */
int PSVectorProduct(PSFloat *a, PSFloat *b, PSFloat *dest,
                    uint64_t alen, uint64_t blen, PSMathOpts *opts)
{
    if (a == NULL || b == NULL || dest == NULL) {
        PSErr(__func__, "`a`, `vector` and `dest` cannot be null");
        return 0;
    }
    PSFloat *vec2add = NULL, *max = NULL, *tmpdest = NULL;
    PSFloatFunc after = NULL;
    int store_mode = MATHS_STORE_MODE_NORM;
    int acceleration = PSGlobalAcceleration;
    uint64_t i, j;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        vec2add = opts->add_vec;
        after = opts->after;
        max = opts->max;
        store_mode = opts->store_mode;
        tmpdest = opts->tmpdest;
    }
    int do_process = (
        vec2add != NULL || after != NULL || max != NULL ||
        store_mode != MATHS_STORE_MODE_NORM
    );
#if defined(HAS_BLAS) || defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    uint64_t dstlen = alen * blen;
    PSFloat *vpdest = dest;
    int blas_enabled = PSBLASEnabled(acceleration),
        acf_enabled = PSACFEnabled(acceleration),
        do_free_vpdest = 0, use_acceleration = (blas_enabled || acf_enabled);
    if (!use_acceleration) goto no_acceleration;
    if (store_mode) {
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
        PSBLASOrder order = PSBLASRowMajor;
        char trans1 = 'N', trans2 = 'N';
        int m = 1, lda = 1, ldb = blen, ldc = blen;
        PSGemm(order, trans1, trans2, alen, blen, m, 1.0, a, lda, b, ldb, 0.0,
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
    if (do_process) {
        for (i = 0; i < dstlen; i++) {
            if (store_mode == MATHS_STORE_MODE_ADD) dest[i] += vpdest[i];
            else if (store_mode == MATHS_STORE_MODE_SUB) dest[i]-=vpdest[i];
            if (vec2add != NULL) dest[i] += vec2add[i];
            if (after != NULL) dest[i] = after(dest[i]);
            if (max != NULL && (i == 0 || dest[i] > *max)) *max = dest[i];
        }
    }
    if (do_free_vpdest) free(vpdest);
    return 1;
#else
    UNUSED(tmpdest);
    UNUSED(acceleration);
#endif
no_acceleration:
    for (i = 0; i < alen; i++) {
        for (j = 0; j < blen; j++) {
            uint64_t idx = (blen * i) + j;
            PSFloat product = (a[i] * b[j]);
            if (!store_mode) dest[idx] = product;
            if (!do_process) continue;
            if (store_mode == MATHS_STORE_MODE_ADD) dest[idx] += product;
            else if (store_mode == MATHS_STORE_MODE_SUB) dest[idx] -= product;
            if (vec2add != NULL) dest[idx] += vec2add[idx];
            if (after != NULL) dest[idx] = after(dest[idx]);
            if (max != NULL && (i == 0 || dest[idx] > *max)) *max = dest[idx];
        }
    }
    return 1;
}

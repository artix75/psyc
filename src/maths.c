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

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#include "config.h"
#include "maths.h"
#include "blas.h"
#include "log.h"
#ifdef USE_AVX
#include "avx.h"
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#include <Accelerate/Accelerate.h>
#endif

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
#define UNUSED(V) ((void) V)

typedef struct PSMatrixHeader {
    int ndims;
    int dims[MAX_DIMENSIONS];
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

PSFloat *PSMatrixValues(PSMatrix matrix, uint32_t *len, int argc, ...) {
    if (matrix == NULL) return NULL;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    PSFloat *values = matrix;
    int stride = 1;
    if (argc > hdr->ndims) argc = hdr->ndims;
    va_list args;
    va_start(args, argc);
    for (int i = 0; i < argc; i++) {
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

int PSMatrixProductMV(PSMatrix a, PSFloat *b, int len, PSMatrix *result) {
    PSBlasOrder order = PSBlasRowMajor;
    int dims_a[MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(a, dims_a);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
    int dimensions[MAX_DIMENSIONS] = {0};
    int l = dims_a[ndims - 1];
    if (len != l) {
        PSErr(__func__, "Aligment error: vector len != a dim[%d] -> "
              "%d != %d", len, l);
        return 0;
    }
    int nd = ndims - 1;
    if (nd == 1) dimensions[0] = (ndims == 2 ? dims_a[0] : len);
    else if (nd == 2) {
        dimensions[0] = dims_a[0];
        dimensions[1] = len;
    } else {
        PSErr(__func__, "Invalid output dimensions: %d", nd);
        return 0;
    }
    PSMatrix out = NULL;
    if (result != NULL) {
        out = *result;
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
        if (out == NULL) return 0;
    }
    int lda = (dims_a[1] > 1 ? dims_a[1] : 1);
    int m = dims_a[0], n = dims_a[1];
    PSGemv(order, 'N', m, n, 1.0, a, lda, b, 1, 0.0, out, 1);
    return 1;
}

int PSMatrixProductVM(PSFloat *a, PSMatrix b, int len, PSMatrix *result) {
    PSBlasOrder order = PSBlasRowMajor;
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
    PSMatrix out = NULL;
    if (result != NULL) {
        out = *result;
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
        if (out == NULL) return 0;
    }
    int lda = (dims_b[1] > 1 ? dims_b[1] : 1);
    int m = dims_b[0], n = dims_b[1];
    PSGemv(order, 'N', m, n, 1.0, b, lda, a, 1, 0.0, out, 1);
    return 1;
}

int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result) {
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
    PSMatrix out = NULL;
    if (result != NULL) {
        out = *result;
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
        if (out == NULL) return 0;
    }
    int a_vector_like = (ndims_a == 1),
        b_vector_like = (ndims_b == 1);
    PSBlasOrder order;
    if (!a_vector_like && b_vector_like) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        order = PSBlasRowMajor;
        lda = (dims_a[1] > 1 ? dims_a[1] : 1);
        int bs = PSMatrixStride(b, 0);
        int m = dims_a[0], n = dims_a[1];
        PSGemv(order, 'N', m, n, 1.0, a, lda, b, bs, 0.0, out, 1);
    } else if (a_vector_like && !b_vector_like) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        order = PSBlasRowMajor;
        lda = (dims_b[1] > 1 ? dims_b[1] : 1);
        int as = PSMatrixStride(a, 0);
        int m = dims_b[0], n = dims_b[1];
        PSGemv(order, 'N', m, n, 1.0, b, lda, a, as, 0.0, out, 1);
    } else {
        /* Matrix matrix multiplication -- Level 3 BLAS */
        order = PSBlasRowMajor;
        char trans1 = 'N', trans2 = 'N';
        int l = dims_a[0];
        int n = dims_b[1];
        int m = dims_b[0];
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
            PSGemm(order, trans1, trans2, l, n, m, 1.0, a, lda, b, ldb, 0.0,
                   out, ldc);
        }
    }
    return 1;
}

void PSMatrixDelete(PSMatrix matrix) {
    if (matrix == NULL) return;
    void *ptr = (void *) getMatrixHeadPointer(matrix);
    free(ptr);
}

/**** Operations ****/

PSFloat PSDotProduct(PSFloat *a, PSFloat *b, uint64_t length, PSDotOpts *opts)
{
    PSDotProductDebug debug_step = NULL;
    if (opts != NULL) debug_step = opts->debug_step;
    uint64_t i = 0;
    PSFloat result = 0.0;
    int acceleration =
        (opts != NULL ? opts->acceleration : PSGlobalAcceleration);
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSDSPEnabled(acceleration)) {
#ifndef PS_DOUBLE_PRECISION
        vDSP_dotpr(a, 1, b, 1, &result, length);
#else
        vDSP_dotprD(a, 1, b, 1, &result, length);
#endif
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
    for (; i < length; i++) {
        if (debug_step) debug_step(i, a[i], b[i], result, 0, opts);
        result += a[i] * b[i];
    }
    return result;
}

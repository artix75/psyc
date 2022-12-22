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

#include "maths.h"
#include "log.h"
#ifdef USE_AVX
#include "avx.h"
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

PSMatrix PSMatrixCreateV(PSFloat init_value, PSMatrixInitializer initializer,
                         int ndims, va_list args)
{
    if (ndims < 1 || ndims > MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d", MAX_DIMENSIONS);
        return NULL;
    }
    int dims[MAX_DIMENSIONS];
    int len = 1, i;
    for (i = 0; i < ndims; i++) {
        int d = va_arg(args, int);
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
    int stride = 1, i = 0;
    if (argc > hdr->ndims) argc = hdr->ndims;
    va_list args;
    va_start(args, argc);
    for (i = 0; i < argc; i++) {
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
#ifdef USE_AVX
    if (opts != NULL && (opts->acceleration & PS_ACCELERATION_AVX)) {
        AVXIterativeDotProduct(length, a, b, result, i, 0, 0);
        if (debug_step) debug_step(i, a[i], b[i], result, 1, opts);
    }
#endif
    for (; i < length; i++) {
        if (debug_step) debug_step(i, a[i], b[i], result, 0, opts);
        result += a[i] * b[i];
    }
    return result;
}

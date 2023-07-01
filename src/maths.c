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
#define VDSPFill(val, a, len) vDSP_vfillD(&val, a, 1, len)
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
#define VDSPFill(val, a, len) vDSP_vfill(&val, a, 1, len)
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
    int acceleration = PSGlobalAcceleration, mode = PS_STORE_MODE_SET;\
    if (opts != NULL) {\
        acceleration = opts->acceleration;\
        mode = opts->store_mode;\
        debug_step = opts->debug_step;\
        assert(mode >= 0 && mode <= PS_STORE_MODE_SUB);\
    }\
    UNUSED(debug_step);

/* Forward declarations and external functions */

int writeSerializedFloatArray(FILE *out, int count, char *sep, int opts,
                              PSFloat *array);

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
    PSFloat scale = stddev * rho;
    PSFloat x = mean + scale * cos(theta);
    PSFloat y = mean + scale * sin(theta);
    PSFloat r = PSNormalizedRandom();
    return (r > 0.5 ? y : x);
}

unsigned int PSRandomInt(unsigned int range, PSFloat *weights, int *err,
                         PSMathOpts *opts)
{
    if (!randomSeeded) {
        randomSeeded = 1;
        srand(time(NULL));
    }
    if (err != NULL) *err = 0;
    if (range == 0) return 0;
    if (weights == NULL) {
        if (range > RAND_MAX) {
            PSFloat r = PSNormalizedRandom() * (PSFloat) range;
            return (int) r;
        } else return rand() % range;
    } else {
        PSFloat cumulated_weights[range];
        if (!PSCumulativeSum(weights, cumulated_weights, range)) {
            if (err != NULL) *err = 1;
            return 0;
        }
        PSFloat last = cumulated_weights[range - 1];
        PSMathOpts mopts = {.acceleration = PSGlobalAcceleration};
        if (opts != NULL) mopts.acceleration = opts->acceleration;
        PSDivideVectorScalar(
            cumulated_weights, last, cumulated_weights, range, &mopts
        );
        PSFloat r = PSNormalizedRandom();
        unsigned int idx = 0;
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
    int dims[PS_MATRIX_MAX_DIMENSIONS];
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

static const char *matrixDimensionsToString(int ndims, int *dims) {
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
        int written = snprintf(s, avail, "%s%d", sep, dims[i]);
        s += written;
        avail -= written;
    }
    return dimstr;
}

static int getShapeType(int nd, int *dims) {
    if (nd <= 0) return PS_SHAPE_TYPE_NONE;
    else if (nd == 1) {
        if (dims[0] > 1) return PS_SHAPE_TYPE_COL;
        return PS_SHAPE_TYPE_SCALAR;
    } else if (nd == 2) {
        if (dims[0] > 1) {
            if (dims[1] == 1) return PS_SHAPE_TYPE_COL;
            else return PS_SHAPE_TYPE_MATRIX;
        }
        if (dims[1] == 1) return PS_SHAPE_TYPE_SCALAR;
        return PS_SHAPE_TYPE_ROW;
    }
    return PS_SHAPE_TYPE_MATRIX;
}

PSMatrix PSMatrixCreateWithShape(PSFloat init_value,
                                 PSMatrixInitializer initializer,
                                 int ndims, int *shape)
{
    if (ndims < 1 || ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d",
              PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    int len = 1, i;
    for (i = 0; i < ndims; i++) {
        int d = shape[i];
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
    if (len == 0) {
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
        if (i < ndims) hdr->dims[i] = shape[i];
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
    if (ndims < 1 || ndims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "ndims must be between 1 and %d",
              PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    int dims[PS_MATRIX_MAX_DIMENSIONS];
    for (int i = 0; i < ndims; i++) {
        int d = va_arg(args, int);
        dims[i] = d;
    }
    return PSMatrixCreateWithShape(init_value, initializer, ndims, dims);
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
    matrix = PSMatrixCreateV(0, NULL, ndims, args);
    va_end(args);
    if (matrix == NULL) return NULL;
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

PSMatrix PSMatrixDupShape(PSMatrix matrix) {
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
    memset(clone, 0, datasize);
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
    int src_dims[PS_MATRIX_MAX_DIMENSIONS];
    int dst_dims[PS_MATRIX_MAX_DIMENSIONS];
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

void PSMatrixClear(PSMatrix matrix) {
    if (matrix == NULL) return;
    PSVectorClear(matrix, PSMatrixLength(matrix));
}

/* Expand matrix `src` by adding `add` to its first dimension. Added data will
 * be set to zero.
 * Beware of the fact that `src` matrix could be freed after the process,
 * so always assing the return value of this function to a new variable, since
 * it could lead to memory leaks in case of a NULL return value. Also beware of
 * the fact that the original variable holding `src` could point to freed
 * memry after function returns. */
PSMatrix PSMatrixExpand(PSMatrix src, int add, int keep_src) {
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
    int new_dims[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixDimensions(src, new_dims);
    int curlen = PSMatrixLength(src);
    new_dims[0] += add;
    matrix = PSMatrixCreateWithShape(0.0, NULL, ndims, new_dims);
    if (matrix == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    PSVectorCopy(matrix, src, curlen);
    if (!keep_src) PSMatrixDelete(src);
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
    memcpy(dims, hdr->dims, (PS_MATRIX_MAX_DIMENSIONS * sizeof(int)));
    return ndims;
}

uint64_t PSMatrixLength(PSMatrix matrix) {
    if (matrix == NULL) return 0;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    return hdr->length;
}

int PSMatrixStride(PSMatrix matrix, int dim) {
    if (matrix == NULL) return 0;
    int refdim = dim + 1;
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    if (refdim >= PS_MATRIX_MAX_DIMENSIONS || refdim >= hdr->ndims) return 1;
    int stride = 1;
    while (refdim < hdr->ndims) stride *= hdr->dims[refdim++];
    return stride;
}

int PSMatrixShapeType(PSMatrix matrix) {
    if (matrix == NULL) return PS_SHAPE_TYPE_NONE;
    int dims[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(matrix, dims);
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
    int dims[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixDimensions(matrix, dims);
    printf(
        "Matrix %s dimensions = %d, shape = (%s)%s",
        name, ndims, matrixDimensionsToString(ndims, dims), nl
    );
}

void PSMatrixPrintShape(PSMatrix matrix, int newline) {
    if (matrix == NULL) return;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int nd = PSMatrixDimensions(matrix, shape), i;
    if (nd > PS_MATRIX_MAX_DIMENSIONS) {
        PSWarn("%s: invalid matrix", __func__);
        return;
    }
    for (i = 0; i < nd; i++) printf("%s%d", (i > 0 ? "," : ""), shape[i]);
    if (newline) printf("\n");
}

int PSMatrixWrite(PSMatrix matrix, const char *sep, char bracket,
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
    int shape[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(matrix, shape), i, j, k;
    int last_dim = ndims - 1;
    int index[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int prev_index[PS_MATRIX_MAX_DIMENSIONS];
    for (j = 0; j < ndims; j++) prev_index[j] = -1;
    int strides[PS_MATRIX_MAX_DIMENSIONS] = {0};
    for (j = 0; j < ndims; j++) {
        strides[j] = 1;
        for (k = j + 1; k < ndims; k++) strides[j] *= shape[k];
    }
    int len = PSMatrixLength(matrix), nwritten = 0;
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
                int last_row_idx = shape[last_dim - 1] - 1;
                if (index[last_dim - 1] == last_row_idx) row_sep = "";
            } else if (ndims == 1) row_sep = "";
            nwritten += fprintf(out, "%c%s%s", end_bracket, row_sep, nl);
            for (j = last_dim - 1; j >= 0; j--) {
                int last_idx = shape[j] - 1;
                row_sep = (i == len - 1 ? "" : sep);
                if (index[j] == last_idx) {
                    nwritten += fprintf(
                        out, "%*c%s%s", (j * indent) + 1, end_bracket,
                        row_sep, nl
                    );
                } else break;
            }
        }
        memcpy(prev_index, index, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
    }
    if (indent > 0) printf("\n");
    return nwritten;
}

void PSMatrixPrint(PSMatrix matrix, const char *sep, int print_shape) {
    if (matrix == NULL) return;
    if (print_shape) {
        int shape[PS_MATRIX_MAX_DIMENSIONS];
        int ndims = PSMatrixDimensions(matrix, shape);
        printf("Matrix (shape = %s):\n",matrixDimensionsToString(ndims,shape));
    }
    PSMatrixWrite(matrix, sep, '[', 2, stdout);
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
            PSWarn("%s: index %d is out of bounds for dim[%d] (%d)",
                   __func__, idx, i, hdr->dims[i]);
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

int genericMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *out,
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
    int shape_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int shape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims_a = PSMatrixDimensions(a, shape_a);
    int ndims_b = PSMatrixDimensions(b, shape_b);
    uint64_t len = 0;
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
            uint64_t reslen = PSMatrixLength(*out);
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
    int l = shape_a[ndims_a - 1], refdim;
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
    int i, j = 0, niter_a = 1, niter_b = 1;
    int dimensions[PS_MATRIX_MAX_DIMENSIONS * 3] = {0};
    for (i = 0; i < (ndims_a - 1); i++) {
        if (i < (ndims_a - 1)) niter_a *= shape_a[i];
        dimensions[j++] = shape_a[i];
    }
    for (i = 0; i < (ndims_b - 2); i++) dimensions[j++] = shape_b[i];
    if (ndims_b > 1) dimensions[j++] = shape_b[ndims_b - 1];
    niter_b = PSMatrixLength(b) / l;
    if (*out != NULL) {
        int out_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int out_ndims = PSMatrixDimensions(*out, out_shape);
        if (out_ndims != nd) {
            PSErr(
                __func__, "`out` matrix has %d dimension(s), but "
                "%d dimension(s) needed", out_ndims, nd
            );
            return 0;
        }
        for (i = 0; i < nd; i++) {
            int odim = out_shape[i];
            if (odim != dimensions[i]) {
                PSErr(
                    __func__, "`out` matrix dimension [%d] is %d, "
                      "but it should be %d\nResult shape: %d,%d",
                      i, odim, dimensions[i]
                );
                return 0;
            }
        }
    } else {
        *out = PSMatrixCreateWithShape(0, NULL, nd, dimensions);
        if (*out == NULL) return 0;
    }
    uint64_t outlen = PSMatrixLength(*out);
    PSFloat *out_p = *out, *ap = a, *bp = NULL;
    PSMatrix swap_b = NULL;
    if (ndims_b > 1) {
        b = PSMatrixTranspose(b, 0, opts);
        if (b == NULL) return 0;
        int blen = PSMatrixDim(b, 0);
        if (ndims_b > 2) {
            if ((ndims_b - 1) > 2) {
                PSErr(NULL, "unsupported dimensions for matrix `b`");
                return 0;
            }
            swap_b = PSMatrixDupShape(b);
            if (swap_b == NULL) return 0;
            int bs = PSMatrixStride(swap_b, 0);
            int tshape_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
            PSMatrixDimensions(b, tshape_b);
            for (i = 0; i < blen; i++) {
                int offset = i * bs;
                PSFloat *src = b + offset, *dst = swap_b + offset;
                dst = PSVectorTranspose(
                    src, dst, opts->acceleration, 2,
                    tshape_b[ndims_b - 2], tshape_b[ndims_b - 1]
                );
                success = (dst != NULL);
                if (!success) goto final;
            }
            b = swap_b;
            PSMatrixDimensions(b, tshape_b);
            int last_db = tshape_b[ndims_b - 1];
            tshape_b[ndims_b - 1] = tshape_b[ndims_b - 2];
            tshape_b[ndims_b - 2] = last_db;
            PSMatrixHeader *hdr = PSMatrixGetHeader(b);
            memcpy(hdr->dims, tshape_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
            l = tshape_b[ndims_b - 1];
            int as = PSMatrixStride(a, 0), k;
            for (i = 0; i < niter_a; i++) {
                bp = b;
                for (k = 0; k < tshape_b[ndims_b - 2]; k++) {
                    for (j = 0; j < blen; j++) {
                        bp = (b + (j * bs)) + (l * k);
                        assert((uint64_t)(out_p - *out) <= outlen);
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
    PSMatrixDelete(swap_b);
    return success;
}

/* Performs matrix-vector multiplication between matrix `a` and vector `b`.
 * Argument `len` must be the length of the vector `b`.
 * Results are stored into vector pointed by pointer `result`. If pointer
 * pointed by `result` is NULL, a new vector is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * Length of `b` vector must equal matrix `a` second dimension.
 * Length of result vector must equal matrix `a` first dimension.
 * By defaults, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build, function will compute results by
 * using `PSMultiplyVectors` as fallback.
 * You can set matrix transposition using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
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
    int use_acf = PSACFEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(acceleration);
    UNUSED(use_acf);
#endif
    int dims_a[PS_MATRIX_MAX_DIMENSIONS];
    int ndims = PSMatrixDimensions(a, dims_a);
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
            int reslen = PSMatrixLength(res);
            if (*result == NULL) *result = calloc(reslen, sizeof(PSFloat));
            if (*result != NULL) PSVectorCopy(*result, res, reslen);
            else success = 0;
        }
        PSMatrixDelete(mb);
        PSMatrixDelete(res);
        return success;
    }
    int l = (transpose & 1 ? dims_a[0] : dims_a[ndims - 1]);
    if (len != l) {
        PSErr(__func__, "Aligment error: vector len != a dim[%d] -> "
              "%d != %d (transpose: %d)", (ndims - 1), l, len, transpose);
        PSMatrixPrintInfo(a, "a", 1);
        return 0;
    }
    int scalar_a = (getShapeType(ndims, dims_a) == PS_SHAPE_TYPE_SCALAR);
    int scalar_b = len == 1;
    int use_scalar = (scalar_a || scalar_b);
    int nd, ld, outlen;
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
        else PSMultiplyVectorScalar(a, *b, out, PSMatrixLength(a), opts);
        return 1;
    }
    if (ndims == 1) {
        out[0] = PSDotProduct(a, b, len, opts);
        return 1;
    }
    int lda = (dims_a[1] > 1 ? dims_a[1] : 1);
    int m = dims_a[0], n = dims_a[1];
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
                PSSumVectors(out, dest, out, outlen, &sumopts);
                if (opts == NULL || tmpdest != opts->tmpdest)
                    free(tmpdest);
            }
            return 1;
        }
#endif
        PSMathOpts mopts = {.acceleration = acceleration};
        for (int i = 0; i < m; i++) {
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
 * The function uses BLAS to compute the result, so, if BLAS support is
 * missing in PsyC build, function will fail.
 * You can set matrix transposition using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the operand arguments you want to be transposed:
 *  - opt->transpose = 2 (transpose matrix `b`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opts`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if ti fails. */
int PSMatrixProductVM(PSFloat *a, PSMatrix b, int len, PSMatrix *result,
                      PSMathOpts *opts)
{
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    PSBLASOrder order = PSBLASRowMajor;
    int mdims_b[PS_MATRIX_MAX_DIMENSIONS];
    int tdims_b[PS_MATRIX_MAX_DIMENSIONS];
    int *dims_b = mdims_b;
    int ndims = PSMatrixDimensions(b, dims_b);
    if (ndims == 0) {
        PSErr(__func__, "Invalid matrix");
        return 0;
    }
    if (ndims > 2) {
        PSMatrix ma = PSMatrixZeros(1, len);
        if (ma == NULL) return 0;
        int success = genericMatrixProduct(ma, b, result, opts);
        PSMatrixDelete(ma);
        return success;
    }
    int last_dim = ndims - 1;
    int dimensions[PS_MATRIX_MAX_DIMENSIONS] = {0};
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
    int use_acf = PSACFEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(acceleration);
    UNUSED(use_acf);
#endif
    int l = dims_b[0];
    if (l != len) {
        PSErr(__func__, "Aligment error: b dim[0] != vector length -> "
              "%d != %d (transpose: %d)", dims_b[0], len, transpose);
        return 0;
    }
    int scalar_a = len == 1;
    int scalar_b = (getShapeType(ndims, dims_b) == PS_SHAPE_TYPE_SCALAR);
    int use_scalar = (scalar_a || scalar_b);
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
        dimensions[0] = (scalar_a ? PSMatrixLength(b) : len);
    }
    PSMatrix out = *result;
    int outlen = 0;
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
            int odim = PSMatrixDim(out, i);
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
        else PSMultiplyVectorScalar(b, *a, out, PSMatrixLength(b), opts);
    } else if (nd==1 && dimensions[0]==1  && PSMatrixLength(b)==(uint64_t)len) {
        *out = PSDotProduct(a, b, len, opts);
        return 1;
    }
    int lda = (mdims_b[1] > 1 ? mdims_b[1] : 1);
    int m = mdims_b[0], n = mdims_b[1];
    if (!use_blas) {
        int do_add = (beta == 1.0);
        if (!(transpose & 2)) b = PSMatrixTranspose(b, 0, opts);
        else {
            int tmpm = m;
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
                PSSumVectors(dest, out, out, outlen, &sumopts);
                if (opts == NULL || tmpdest != opts->tmpdest)
                    free(tmpdest);
            }
            return 1;
        }
#endif
        PSMathOpts mopts = {.acceleration = acceleration};
        for (int i = 0; i < n; i++) {
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

/* Performs matrix-matrix multiplication between matrix `a` and vector `b`.
 * Results are stored into matrix pointed by `result`. If pointer
 * pointed by `result` is NULL, a new matrix is automatically allocated
 * by the function itself and its pointer will be stored into `result`.
 * By defaults, function uses BLAS to compute the result. Anyway, if BLAS
 * support is missing in PsyC build and `b` only has one dimension, function
 * will try compute results by using `PSMultiplyVectors` as fallback (for all
 * other cases, it will fail!).
 * The `opt` argument can be NULL.
 * You can set matrix transposition using `transpose` field in the `opt`
 * argument. In that case, `transpose` will contain the (1-based) indices
 * of the matrix arguments you want to be transposed:
 *  - opt->transpose = 1 (transpose matrix `a`)
 *  - opt->transpose = 2 (transpose matrix `b`)
 *  - opt->transpose = (1 | 2) (transpose both matrix `a` and `b`)
 * By default, data in result vector will be overwritten. Anyway, if
 * `PS_STORE_MODE_ADD` is set as `store_mode` into `opt`, result will
 * be added to data already present in the result vector.
 * Return value: 1 if operation succeeds, 0 if ti fails. */
int PSMatrixProduct(PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt) {
    if (result == NULL) {
        PSErr(__func__, "argument result cannot be null");
        return 0;
    }
    /* Original matrix dimensions */
    int mdims_a[PS_MATRIX_MAX_DIMENSIONS];
    int mdims_b[PS_MATRIX_MAX_DIMENSIONS];
    /* Eventually transposed matrix dimensions */
    int tdims_a[PS_MATRIX_MAX_DIMENSIONS];
    int tdims_b[PS_MATRIX_MAX_DIMENSIONS];
    /* Number of dimensions */
    int ndims_a = PSMatrixDimensions(a, mdims_a);
    int ndims_b = PSMatrixDimensions(b, mdims_b);
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
    int last_dim_a = ndims_a - 1, last_dim_b = ndims_b - 1;
    int *dims_a = mdims_a, *dims_b = mdims_b;
    int lda = 0, ldb = 0, l = 0, i;
    int dimensions[PS_MATRIX_MAX_DIMENSIONS] = {0};
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
    int use_acf = PSACFEnabled(acceleration);
#else
    int use_acf = 0;
    UNUSED(use_acf);
    UNUSED(acceleration);
#endif
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
            int tmp_shape_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
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
            memcpy(tmp_shape_a, mdims_a, PS_MATRIX_MAX_DIMENSIONS*sizeof(int));
            memcpy(mdims_a, mdims_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
            memcpy(mdims_b, tmp_shape_a, PS_MATRIX_MAX_DIMENSIONS*sizeof(int));
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
        int odims_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int odims_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
        int o_ndims_a = PSMatrixDimensions(orig_a, odims_a),
            o_ndims_b = PSMatrixDimensions(orig_b, odims_b);
        if (transpose) {
            int orig_transp = transpose;
            if (reverse_args) {
                orig_transp = 0;
                if (transpose & 2) orig_transp |= 1;
                if (transpose & 1) orig_transp |= 2;
            }
            if (o_ndims_a > 1 && orig_transp & 1) {
                int tmp = odims_a[0];
                odims_a[0] = odims_a[o_ndims_a - 1];
                odims_a[o_ndims_a - 1] = tmp;
            }
            if (o_ndims_b > 1 && orig_transp & 2) {
                int tmp = odims_b[0];
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
    int outlen = 0;
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
            int odim = PSMatrixDim(out, i);
            if (odim != dimensions[i]) {
                PSErr(
                    __func__, "`result` matrix dimension [%d] is %d, "
                      "but it should be %d\nResult shape: %d,%d",
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
    PSBLASOrder order;
    if (shape_b == PS_SHAPE_TYPE_SCALAR) {
        if (l == 1) {
            *out = *b * *a;
            return 1;
        } else {
            int a_stride;
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
                int o_stride = PSMatrixStride(out, max_dim_idx);
                int o_dim = dims_a[o_dim_idx];
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
        int bs = PSMatrixStride(b, (transpose_b ? ndims_b - 1 : 0));
        int m = dims_a[0], n = dims_a[1];
        if (!use_blas) {
            int do_add = (beta == 1.0);
            if (transpose_a) {
                int tmpm = m;
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
                    PSSumVectors(dest, out, out, outlen, &sumopts);
                    if (opt == NULL || tmpdest != opt->tmpdest)
                        free(tmpdest);
                }
                return 1;
            }
#else
            UNUSED(outlen);
#endif
            PSMathOpts mopts = {.acceleration = acceleration};
            for (int i = 0; i < m; i++) {
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
        int m = dims_b[0], n = dims_b[1];
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
                int tmpm = m;
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
                    PSMathOpts sumopt = {.acceleration = opt->acceleration};
                    PSSumVectors(dest, out, out, outlen, &sumopt);
                    if (opt == NULL || tmpdest != opt->tmpdest)
                        free(tmpdest);
                }
                return 1;
            }
#endif
            if (opt) opt->store_mode = PS_STORE_MODE_SET;
            for (int i = 0; i < n; i++) {
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
        int m = dims_a[0];
        int n = dims_b[1];
        int k = dims_a[1];
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
                    PSMathOpts sumopt = {.acceleration = opt->acceleration};
                    PSSumVectors(dest, out, out, outlen, &sumopt);
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
            for (int i = 0; i < dimensions[0]; i++) {
                for (int j = 0; j < dimensions[1]; j++) {
                    int oidx = (i * dimensions[1]) + j;
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
            int odim1 = PSMatrixDim(out, 1);
            int ldc = ((odim1 > 1) ? odim1 : 1);
            PSGemm(order, trans_a, trans_b, m, n, k, 1.0, a, lda, b, ldb, beta,
                   out, ldc);
        }*/
        int odim1 = PSMatrixDim(out, 1);
        int ldc = ((odim1 > 1) ? odim1 : 1);
        PSGemm(order, trans_a, trans_b, m, n, k, 1.0, a, lda, b, ldb, beta,
               out, ldc);
    }
    if (PSBLASLastError != NULL) return 0;
    return 1;
align_err:
    if (reverse_args) {
        /* `a` and `b` were reversed */
        int tmp_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
        memcpy(tmp_shape, dims_a, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
        memcpy(dims_a, dims_b, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
        memcpy(dims_b, dims_a, PS_MATRIX_MAX_DIMENSIONS * sizeof(int));
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

PSMatrix PSMatrixReshape(PSMatrix matrix, int num_dims, ...) {
    if (matrix == NULL) return NULL;
    if (num_dims <= 0) return NULL;
    if (num_dims > PS_MATRIX_MAX_DIMENSIONS) {
        PSErr(__func__, "max shape dimensions: %d", PS_MATRIX_MAX_DIMENSIONS);
        return NULL;
    }
    uint64_t new_len = 1;
    int new_shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrixHeader *hdr = PSMatrixGetHeader(matrix);
    va_list args;
    va_start(args, num_dims);
    for (int i = 0; i < num_dims; i++) {
        new_shape[i] = va_arg(args, int);
        new_len *= (uint64_t) new_shape[i];
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

PSMatrix PSMatrixFlatten(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    return PSMatrixReshape(matrix, 1, PSMatrixLength(matrix));
}

PSMatrix *PSMatrixSplit(PSMatrix matrix, int num_slices, int axis,
                        PSMathOpts *opts)
{
    if (matrix == NULL) return NULL;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int ndims = PSMatrixDimensions(matrix, shape);
    uint64_t len = PSMatrixLength(matrix);
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
    int dimsize = shape[axis];
    int mod = dimsize % num_slices;
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
    int split_dimsize = dimsize / num_slices, i;
    int num_sizes = 1 + mod + (num_slices - mod);
    int sizes[num_sizes];
    sizes[0] = 0;
    int *size_p = ((int *)sizes) + 1;
    for (i = 0; i < mod; i++) *(size_p++) = split_dimsize + 1;
    for (i = 0; i < (num_slices - mod); i++) *(size_p++) = split_dimsize;
    int points[num_sizes];
    int cumsum = 0;
    for (i = 0; i < num_sizes; i++) {
        cumsum += sizes[i];
        points[i] = cumsum;
    }
    int do_transpose = axis != 0;
    if (do_transpose) {
        matrix = PSMatrixTranspose(matrix, 0, opts);
        success = (matrix != NULL);
        if (!success) goto final;
        PSMatrixDimensions(matrix, shape);
    }
    int elem_size = 1;
    for (i = 1; i < ndims; i++) elem_size *= shape[i];
    for (i = 0; i < num_slices; i++) {
        int from = points[i], to = points[i + 1], len = to - from;
        int size = len * elem_size;
        PSFloat *data = matrix + (from * elem_size);
        int slice_shape[PS_MATRIX_MAX_DIMENSIONS] = {len};
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
            for (i = 0; i < num_slices; i++) PSMatrixDelete(slices[i]);
            free(slices);
            slices = NULL;
        }
    }
    return slices;
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
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSSubtractVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE()
#ifdef HAS_ACCELERATE_FRAMEWORK
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSMultiplyVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                       PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode != PS_STORE_MODE_SUB) {
        if (mode == PS_STORE_MODE_SET)
            VDSPMulV(a, b, dest, length);
        else if (mode == PS_STORE_MODE_ADD)
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
            PSFloat *x = a + i, *y = b + i, *d = dest + i;
            int c = AVXMultiply(x, y, length, d, mode);
            assert((uint64_t) c == avx_step_len);
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
}

void PSDivideVectors(PSFloat *a, PSFloat *b, PSFloat *dest, uint64_t length,
                     PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
            PSFloat *x = a + i, *y = b + i, *d = dest + i;
            int c = AVXDivide(x, y, length, d, mode);
            assert((uint64_t) c == avx_step_len);
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
}

void PSMultiplyVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSSumVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSSubtractVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSSubtractScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                            uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSDivideVectorScalar(PSFloat *a, PSFloat b, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSDivideScalarVector(PSFloat b, PSFloat *a, PSFloat *dest,
                          uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK) && defined(__arm64__)
    /* This seems to lead to nan in x86 arch., so only use it with arm64 */
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSVectorTanh(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VVTanh(a, dest, length);
        return;
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
}

void PSVectorExp(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VVExp(a, dest, length);
        return;
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
}

void PSVectorSqrt(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VVSqrt(a, dest, length);
        return;
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
}

void PSVectorNeg(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSVectorAbs(PSFloat *a, PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
        VDSPAbs(a, dest, length);
        return;
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
}

void PSVectorClip(PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest,
                  uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSVectorThreshold(PSFloat *a, PSFloat min, PSFloat *dest,
                       uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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
}

void PSVectorMapWithLimit(PSFloat *a, PSFloat limit, PSFloat mapper,
                          PSFloat *dest, uint64_t length, PSMathOpts *opts)
{
    MATHS_OPERATION_PREAMBLE();
#if defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration) && mode == PS_STORE_MODE_SET) {
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

int PSCumulativeSum(PSFloat *a, PSFloat *dest, uint64_t length) {
    if (a == NULL || dest == NULL) {
        PSErr(__func__, "`a` and `dest` cannot be null");
        return 0;
    }
    PSFloat sum = 0.0;
    for (uint64_t i = 0; i < length; i++) {
        sum += a[i];
        dest[i] = sum;
    }
    return 1;
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
    if (len == 0) return 0;
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

/* Performs matrix-matrix multiplication, matrix-vector multiplication,
 * vector-matrix multiplication or vector-vector multiplication,
 * depending on the value of `argtype` field in opts (default is
 * matrix-matrix).
 * Store result is `dest`.
 * Return value: 1 in case of success, 0 in case of failure.
 * NOTE: if `argtype` for both `a` and `b` is 'V', the function will compute
 * the dot product of the two vectors, assuming that they have the same size.
 * If you need to perform matrix multiplication on two PSFloat arrays,
 * use `PSMatMul` instead. */
int PSDot(PSMatrix a, PSMatrix b, PSMatrix dest, PSMathOpts *opts) {
    if (a == NULL || b == NULL || dest == NULL) {
        if (a == NULL) PSErr(__func__, "`a` cannot be null");
        if (b == NULL) PSErr(__func__, "`b` cannot be null");
        if (dest == NULL) PSErr(__func__, "`dest` cannot be null");
        return 0;
    }
    int store_mode = PS_STORE_MODE_SET;
    int dims_a[PS_MATRIX_MAX_DIMENSIONS] = {0};
    int dims_b[PS_MATRIX_MAX_DIMENSIONS] = {0};
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
            PSMatrixDelete(tmpmatrix);
            return 0;
        }
        if (store_mode == PS_STORE_MODE_SUB) {
            PSSubtractVectors(dest, tmpmatrix, dest, PSMatrixLength(dest),opts);
            PSMatrixDelete(tmpmatrix);
        }
        return 1;
    } else if (!a_is_vec && b_is_vec) {
        /* matrix-vector multiplication */
        int ndims = PSMatrixDimensions(a, dims_a);
        int len = (transpose & 1 ? dims_a[0] : dims_a[ndims - 1]);
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
        PSMatrixDimensions(b, dims_b);
        int len = dims_b[1];
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
        int len = 0;
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

int PSDotMV(PSMatrix a, PSFloat *b, PSFloat *dest, PSMathOpts *opts) {
    PSMathOpts myopts = *opts;
    myopts.argtype[0] = 'M';
    myopts.argtype[1] = 'V';
    return PSDot(a, b, dest, &myopts);
}

int PSDotVM(PSFloat *a, PSMatrix b, PSMatrix dest, PSMathOpts *opts) {
    PSMathOpts myopts = *opts;
    myopts.argtype[0] = 'V';
    myopts.argtype[1] = 'M';
    return PSDot(a, b, dest, &myopts);
}

/* Multiply every element of vector `a` (having `alen` length) by every
 * element of vector `b` (having `blen` length) and store results into
 * vector `dest` (whose length must be `alen` * `blen`). */
int PSOuterProduct(PSFloat *a, PSFloat *b, PSFloat *dest,
                   uint64_t alen, uint64_t blen, PSMathOpts *opts)
{
    if (a == NULL || b == NULL || dest == NULL) {
        PSErr(__func__, "`a`, `vector` and `dest` cannot be null");
        return 0;
    }
    PSFloat *tmpdest = NULL;
    int store_mode = PS_STORE_MODE_SET;
    int acceleration = PSGlobalAcceleration;
    uint64_t i, j;
    if (opts != NULL) {
        acceleration = opts->acceleration;
        store_mode = opts->store_mode;
        tmpdest = opts->tmpdest;
    }
    uint64_t dstlen = alen * blen;
    int postprocess = store_mode != PS_STORE_MODE_SET;
#if defined(HAS_BLAS) || defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PSFloat *vpdest = dest;
    int blas_enabled = PSBLASEnabled(acceleration),
        acf_enabled = PSACFEnabled(acceleration),
        do_free_vpdest = 0, use_acceleration = (blas_enabled || acf_enabled);
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
        int m = 1, lda = 1, ldb = blen, ldc = blen;
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
            PSSumVectors(dest, vpdest, dest, dstlen, &ppopts);
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
            uint64_t idx = (blen * i) + j;
            PSFloat product = (a[i] * b[j]);
            if (!store_mode) dest[idx] = product;
            if (!postprocess) continue;
            if (store_mode == PS_STORE_MODE_ADD) dest[idx] += product;
            else if (store_mode == PS_STORE_MODE_SUB) dest[idx] -= product;
        }
    }
    return 1;
}

void PSVectorFill(PSFloat *vec, PSFloat val, uint64_t len, PSMathOpts *opts) {
    if (len == 0 || vec == NULL) return;
    int acceleration = PSGlobalAcceleration;
    if (opts != NULL) acceleration = opts->acceleration;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    if (PSACFEnabled(acceleration)) {
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
    for (uint64_t i = 0; i < len; i++) vec[i] = val;
}

void PSVectorWrite(PSFloat *vec, int len, char* sep, FILE *f) {
    if (vec == NULL || f == NULL) return;
    if (sep == NULL) sep = ",";
    writeSerializedFloatArray(f, len, sep, 0, vec);
}

void PSVectorPrint(PSFloat *vec, int len, char* sep) {
    PSVectorWrite(vec, len, sep, stdout);
    printf("\n");
}

/* Create a transposed version of `vec`, considering it a matrix with a shape
 * of `ndims` dimensions.
 * Use variadic arguments to set up-to 3 dimensions in the shape,
 * (ie rows, columns for 2-D array).
 * If `dest` is not NULL, transposed vector will be stored into it.
 * Return value: the transposed array, with size of dim1*dim12*dim3, or NULL
 * if something goes wrong. If `dest` is not NULL, return value will be
 * `dest` or NULL if something goes wrong.
 * NOTES:
 * - Variadic dimensions refer to original matrix shape, and not to the
 *   resulting transposed matrix.
 * - If you need to transpose a `PSMatrix`, use `PSMatrixTranspose` instead. */
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
    int dims[PS_MATRIX_MAX_DIMENSIONS] = {0};
    uint64_t x, y, z, idx, i, veclen = 1;
    va_list args;
    va_start(args, ndims);
    for (i = 0; i < (uint64_t) ndims; i++) {
        dims[i] = va_arg(args, int);
        veclen *= dims[i];
    }
    va_end(args);
    if (veclen <= 0) {
        PSErr(__func__, "Invalid dimensions: eachdimension must be > 0");
        return NULL;
    }
    PSFloat *transposed = dest;
    if (transposed == NULL) transposed = malloc(veclen * sizeof(PSFloat));
    if (transposed == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int ncols = 0, nrows = 0, dlen = 0, t_ncols = 0, t_dlen = 0;
    if (ndims == 3) {
        nrows = dims[1];
        ncols = dims[2];
        t_ncols = dims[0];
        t_dlen = dims[0] * dims[1];
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
        if (PSACFEnabled(acceleration)) {
            /* Use Apple Accelerate Framework */
            VDSPMTransp(vec, transposed, dims[1], dims[0]);
            goto final;
        }
#else
        UNUSED(acceleration);
#endif
        ncols = dims[1];
        t_ncols = dims[0];
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
int PSMatMul(PSFloat *a, PSFloat *b, PSFloat *dest, int m, int n, int k,
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
    if (use_blas) {
        char trans_a = 'N', trans_b = 'N';
        int lda = k, ldb = n, ldc = n;
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
    if (PSACFEnabled(acceleration)) {
        int outlen = m * n;
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
            PSSumVectors(dest, out, dest, outlen, &mopts);
            if (opts == NULL || tmpdest != opts->tmpdest)
                free(tmpdest);
        }
        goto final;
    }
#endif
    int a_rows = m, a_cols = k, b_rows = k, b_cols = n, out_rows, out_cols;
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
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            int oidx = (i * out_cols) + j;
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

PSMatrix PSDiagonalMask(int size) {
    if (size <= 0) {
        PSErr(__func__, "invalid size: %d", size);
        return NULL;
    }
    PSMatrix mask = PSMatrixZeros(2, size, size);
    if (mask == NULL) return NULL;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            if (c <= r) {
                mask[(r * size) + c] = 1.0;
            }
        }
    }
    return mask;
}

PSMatrix PSDiagonalFlattenVector(PSFloat *vec, uint64_t len) {
    if (vec == NULL || len == 0) return NULL;
    PSMatrix result = PSMatrixZeros(2, len, len);
    if (result == NULL) {
        PSErr(__func__, "could not create result matrix");
        return NULL;
    }
    PSFloat *res_p = result;
    for (uint64_t i = 0; i < len; i++) {
        res_p[i] = vec[i];
        res_p += len;
    }
    return result;
}

PSMatrix PSDiagonalFlatten(PSMatrix matrix) {
    if (matrix == NULL) return NULL;
    uint64_t len = PSMatrixLength(matrix);
    if (len == 0) return NULL;
    return PSDiagonalFlattenVector(matrix, len);
}

PSFloat **PSVectorSplit(PSFloat *vec, int len, int num_slices) {
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
    int slice_size = len / num_slices;
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

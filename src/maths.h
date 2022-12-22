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

#include "types.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanhActivation tanh
#define PSSqrt(v) sqrt(v)
#define PSFloor(v) floor(v)
#define PSExp(v) exp(v)
#define PSRound(v) round(v)
#define PSMathLog(v) log(v)
#define PSAbs(v) fabs(v)
#define PSPow(a,b) pow(a, b)
#else
#define PSTanhActivation tanhf
#define PSSqrt(v) sqrtf(v)
#define PSFloor(v) floorf(v)
#define PSExp(v) expf(v)
#define PSRound(v) roundf(v)
#define PSMathLog(v) logf(v)
#define PSAbs(v) fabsf(v)
#define PSPow(a,b) powf(a, b)
#endif

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
PSMatrix PSMatrixWithGaussianRandom(PSFloat stddev, int ndims, ...);
int PSMatrixNumDims(PSMatrix matrix);
int PSMatrixDim(PSMatrix matrix, int dim);
size_t PSMatrixLength(PSMatrix matrix);
int PSMatrixStride(PSMatrix matrix, int dim);
PSFloat *PSMatrixValues(PSMatrix matrix, uint32_t *len, int argc, ...);
void PSMatrixDelete(PSMatrix matrix);

#endif /* __PS_MATHS_H__ */


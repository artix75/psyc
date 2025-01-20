/*
 * Copyright (C) 2016-present Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#ifndef __PS_BLAS_H__
#define __PS_BLAS_H__

#include <stdint.h>
#include "types.h"

#ifdef PSBLAS_INT_SIZE
#if PSBLAS_INT_SIZE == 8 && !defined(PS_LAPACK_I32)
typedef long PSBLAS_int;
#define PSBLAS_MAX INT64_MAX
#else
typedef int32_t PSBLAS_int;
#define PSBLAS_MAX (long) INT32_MAX
#endif
#else
typedef int32_t PSBLAS_int;
#define PSBLAS_MAX (long) INT32_MAX
#endif

typedef enum {
    PSBLASRowMajor,
    PSBLASColMajor
} PSBLASOrder;

typedef struct PSBLASErr {
    const char *func;
    const char *param;
    long param_pos;
    long param_value;
    long array_index;
} PSBLASErr;

extern PSBLASErr *PSBLASLastError;
const char *PSBLASErrorStr(PSBLASErr *err, const char **param_names);
int PSBLASCheckLimits(PSBLASErr *err, const char *argformat, ...);
void PSAxpy(PSBLAS_int n, PSFloat alpha, PSFloat *x, PSBLAS_int incx,
            PSFloat *y, PSBLAS_int incy);
void PSGemv(PSBLASOrder order, char trans, PSBLAS_int m, PSBLAS_int n,
            PSFloat alpha, PSFloat *a, PSBLAS_int lda, PSFloat *x,
            PSFloat incx, PSFloat beta, PSFloat *y, PSBLAS_int incy);
void PSGemm(PSBLASOrder order, char trans_a, char trans_b, PSBLAS_int m,
            PSBLAS_int n, PSBLAS_int k, PSFloat alpha, PSFloat *a,
            PSBLAS_int lda, PSFloat *b, PSBLAS_int ldb, PSFloat beta,
            PSFloat *c, PSBLAS_int ldc);

#endif /* __PS_BLAS_H__ */

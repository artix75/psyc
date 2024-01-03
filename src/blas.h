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

#ifndef __PS_BLAS_H__
#define __PS_BLAS_H__

#include "types.h"

typedef enum {
    PSBLASRowMajor,
    PSBLASColMajor
} PSBLASOrder;

typedef struct PSBlasErr {
    const char *func;
    const char *param;
    int param_pos;
    int param_value;
} PSBLASErr;

extern PSBLASErr *PSBLASLastError;
void PSAxpy(int n, PSFloat alpha, PSFloat *x, int incx, PSFloat *y, int incy);
void PSGemv(PSBLASOrder order, char trans, int m, int n, PSFloat alpha,
            PSFloat *a, int lda, PSFloat *x, PSFloat incx, PSFloat beta,
            PSFloat *y, int incy);
void PSGemm(PSBLASOrder order, char trans_a, char trans_b, int m, int n, int k,
            PSFloat alpha, PSFloat *a, int lda, PSFloat *b, int ldb,
            PSFloat beta, PSFloat *c, int ldc);

#endif /* __PS_BLAS_H__ */

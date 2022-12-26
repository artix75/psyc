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

#include "blas.h"
#include "log.h"
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#include <Accelerate/Accelerate.h>
#ifndef HAS_CBLAS
#define HAS_CBLAS 1
#endif
#elif defined(HAS_CBLAS)
#include <cblas.h>
#endif

#ifdef HAS_CBLAS
#define PSBLAS_CONVERT_ORDER(order, var) do {\
    switch (order) {\
    case PSBlasRowMajor: var = CblasRowMajor; break;\
    case PSBlasColMajor: var = CblasColMajor; break;\
    default: PSErr(__func__, "Invalid order"); return;\
    }\
} while(0);
#define PSBLAS_CONVERT_TRANS(trans, var) do {\
    switch (trans) {\
    case 'N': var = CblasNoTrans; break;\
    case 'T': var = CblasTrans; break;\
    case 'C': var = CblasConjTrans; break;\
    case 'A': var = AtlasConj; break;\
    default: PSErr(__func__, "Invalid trans"); return;\
    }\
} while(0);
#endif

#define PS_BLAS_OFFSET(inc,len) (inc > 0 ?  0 : (len - 1) * -(inc))
#define UNUSED(V) ((void) V)

/* Native BLAS functions, loosely inspired by GNU Scientific Library (GSL):
 * https://www.gnu.org/software/gsl/doc/html/ */

static void psyc_gemv(PSBlasOrder order, char trans, int m, int n,
                      PSFloat alpha, PSFloat *a, int lda, PSFloat *x,
                      PSFloat incx, PSFloat beta, PSFloat *y, int incy)
{
    trans = (trans != 'C' ? trans : 'T');
    int info = 0;
    if (trans != 'N' && trans != 'T' && trans !='C') info = 2;
    else if (m < 0) info = 3;
    else if (n < 0) info = 4;
    else if (order == PSBlasColMajor) {
        int max_m = (m > 1 ? m : 1);
        if (lda < max_m) info = 7;
    } else if (order == PSBlasRowMajor) {
        int max_n = (n > 1 ? n : 1);
        if (lda < max_n) info = 7;
    }
    else if (incx == 0) info = 9;
    else if (incy == 0) info = 12;
    if (info != 0) {
        PSErr("PSGemv", "Invalid parameter at %d", info);
        return;
    }

    if (m == 0 || n == 0) return;
    if (alpha == 0.0 && beta == 1.0) return;

    int lenx = 0, leny = 0;
    if (trans == 'N') {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }

    int i, j, iy, ix;
    if (beta == 0.0) {
        iy = PS_BLAS_OFFSET(leny, incy);
        for (i = 0; i < leny; i++) {
            y[iy] = 0.0;
            iy += incy;
        }
    } else if (beta != 1.0) {
        iy = PS_BLAS_OFFSET(leny, incy);
        for (i = 0; i < leny; i++) {
            y[iy] *= beta;
            iy += incy;
        }
    }

    if (alpha == 0.0) return;

    if ((order == PSBlasRowMajor && trans == 'N') ||
        (order == PSBlasColMajor && trans == 'T')) {
        iy = PS_BLAS_OFFSET(leny, incy);
        for (i = 0; i < leny; i++) {
            PSFloat temp = 0.0;
            ix = PS_BLAS_OFFSET(lenx, incx);
            for (j = 0; j < lenx; j++) {
                temp += x[ix] * a[lda * i + j];
                ix += incx;
            }
            y[iy] += alpha * temp;
            iy += incy;
        }
    } else if ((order == PSBlasRowMajor && trans == 'T') ||
               (order == PSBlasColMajor && trans == 'N')) {
        ix = PS_BLAS_OFFSET(lenx, incx);
        for (j = 0; j < lenx; j++) {
            PSFloat temp = alpha * x[ix];
            if (temp != 0.0) {
                iy = PS_BLAS_OFFSET(leny, incy);
                for (i = 0; i < leny; i++) {
                    y[iy] += temp * a[lda * j + i];
                    iy += incy;
                }
            }
            ix += incx;
        }
    } else PSErr("PSGemv", "ERROR: unrecognized operation");
}

static void psyc_gemm(PSBlasOrder order, char trans_a, char trans_b, int m,
            int n, int k, PSFloat alpha, PSFloat *a, int lda, PSFloat *b,
            int ldb, PSFloat beta, PSFloat *c, int ldc) {
    char trans_f, trans_g;
    if (order == PSBlasRowMajor) {
        trans_f = (trans_a != 'C' ? trans_a : 'T');
        trans_g = (trans_b != 'C' ? trans_b : 'T');
    } else {
        trans_f = (trans_b != 'C' ? trans_b : 'T');
        trans_g = (trans_a != 'C' ? trans_a : 'T');
    }
    int info = 0, maxk = (k > 1 ? k : 1), maxm = (m > 1 ? m : 1),
         maxn = (n > 1 ? n : 1);
    if (order == PSBlasRowMajor) {
        if (trans_f == 'N' && lda < maxk) info = 9;
        else if (lda < maxm) info = 9;
        if (trans_g == 'N' && ldb < maxn) info = 11;
        else if (ldb < maxk) info = 11;
        if (ldc < maxn) info = 14;
    } else {
        if (trans_f == 'N' && ldb < maxk) info = 11;
        else if (ldb < maxn) info = 11;
        if (trans_g == 'N' && lda < maxm) info = 9;
        else if (lda < maxk) info = 9;
        if (ldc < maxm) info = 14;
    }
    if (info != 0) {
        PSErr("PSGemm", "Invalid parameter at %d", info);
        return;
    }
    if (alpha == 0.0 && beta == 1.0) return;
    int n1, n2, ldf, ldg, i, j, _k;
    PSFloat *f, *g;
    if (order == PSBlasRowMajor) {
        n1 = m;
        n2 = n;
        f = a;
        ldf = lda;
        trans_f = (trans_a == 'C' ? 'T' : trans_a);
        g = b;
        ldg = ldb;
        trans_g = (trans_b == 'C' ? 'T'  : trans_b);
    } else {
        n1 = n;
        n2 = m;
        f = b;
        ldf = ldb;
        trans_f = (trans_b == 'C' ? 'T' : trans_b);
        g = a;
        ldg = lda;
        trans_g = (trans_a == 'C' ? 'T' : trans_a);
    }
    if (beta == 0.0) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) c[ldc * i + j] = 0.0;
        }
    } else if (beta != 1.0) {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) c[ldc * i + j] *= beta;
        }
    }
    if (alpha == 0.0) return;

    if (trans_f == 'N' && trans_g == 'N') {
        for (_k = 0; _k < k; _k++) {
            for (i = 0; i < n1; i++) {
                PSFloat temp = alpha * f[ldf * i + _k];
                for (j = 0; j < n2; j++)
                    c[ldc * i + j] += temp * g[ldg * _k + j];
            }
        }
    } else if (trans_f == 'N' && trans_g == 'T') {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                PSFloat temp = 0.0;
                for (_k = 0; _k < k; k++) {
                    temp += g[ldf * i + _k] * g[ldg * j + _k];
                }
                c[ldc * i + j] += alpha * temp;
            }
        }
    } else if (trans_f == 'T' && trans_g == 'N') {

        for (_k = 0; _k < k; _k++) {
            for (i = 0; i < n1; i++) {
                PSFloat temp = alpha * f[ldf * _k + i];
                if (temp != 0.0) {
                    for (j = 0; j < n2; j++) {
                        c[ldc * i + j] += temp * g[ldg * _k + j];
                    }
                }
            }
        }
    } else if (trans_f == 'T' && trans_g == 'T') {
        for (i = 0; i < n1; i++) {
            for (j = 0; j < n2; j++) {
                PSFloat temp = 0.0;
                for (_k = 0; _k < k; _k++)
                     temp += g[ldf * _k + i] * g[ldg * j + _k];
                c[ldc * i + j] += alpha * temp;
            }
        }
    } else PSErr("PSGemv", "ERROR: unrecognized operation");
}

/* Wrapper public functions */

void PSGemv(PSBlasOrder order, char trans, int m, int n, PSFloat alpha,
            PSFloat *a, int lda, PSFloat *x, PSFloat incx, PSFloat beta,
            PSFloat *y, int incy)
{
#if defined(HAS_CBLAS) && !defined(USE_PSYC_BLAS)
    UNUSED(psyc_gemv);
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;
    PSBLAS_CONVERT_ORDER(order, cblas_order);
    PSBLAS_CONVERT_TRANS(trans, cblas_trans);
#ifdef PS_DOUBLE_PRECISION
    cblas_dgemv(cblas_order, cblas_trans, m, n, alpha, a, lda, x, incx,
                beta, y, incy);
#else
    cblas_sgemv(cblas_order, cblas_trans, m, n, alpha, a, lda, x, incx,
                beta, y, incy);
#endif
#else
    psyc_gemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}

void PSGemm(PSBlasOrder order, char trans_a, char trans_b, int m, int n, int k,
            PSFloat alpha, PSFloat *a, int lda, PSFloat *b, int ldb,
            PSFloat beta, PSFloat *c, int ldc)
{
#if defined(HAS_CBLAS) && !defined(USE_PSYC_BLAS)
    UNUSED(psyc_gemm);
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans_a, cblas_trans_b;
    PSBLAS_CONVERT_ORDER(order, cblas_order);
    PSBLAS_CONVERT_TRANS(trans_a, cblas_trans_a);
    PSBLAS_CONVERT_TRANS(trans_b, cblas_trans_b);
#ifdef PS_DOUBLE_PRECISION
    cblas_dgemm(cblas_order, cblas_trans_a, cblas_trans_b, m, n, k, alpha, a,
                lda, b, ldb, beta, c, ldc);
#else
    cblas_sgemm(cblas_order, cblas_trans_a, cblas_trans_b, m, n, k, alpha, a,
                lda, b, ldb, beta, c, ldc);
#endif
#else
    psyc_gemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
              beta, c, ldc);
#endif
}


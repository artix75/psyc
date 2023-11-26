/*
 * Copyright (C) 2016-2023 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#include "platform.h"
#include "blas.h"
#include "log.h"
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#include <Accelerate/Accelerate.h>
#ifndef HAS_CBLAS
#define HAS_CBLAS 1
#endif
#elif defined(HAS_CBLAS)

#ifdef HAS_GSL_CBLAS
#include <gsl_cblas.h>
#else
#include <cblas.h>
#endif

#endif

#ifdef HAS_CBLAS
#define PSBLAS_CONVERT_ORDER(order, var) do {\
    switch (order) {\
    case PSBLASRowMajor: var = CblasRowMajor; break;\
    case PSBLASColMajor: var = CblasColMajor; break;\
    default: PSErr(__func__, "Invalid order"); return;\
    }\
} while(0);


#ifndef HAS_GSL_CBLAS
#define _ATLAS_CONJ_A AtlasConj
#else
#define _ATLAS_CONJ_A 0; PSErr(NULL, "BLAS Trans 'A' not supported");abort();
#endif

#define PSBLAS_CONVERT_TRANS(trans, var) do {\
    switch (trans) {\
    case 'N': var = CblasNoTrans; break;\
    case 'T': var = CblasTrans; break;\
    case 'C': var = CblasConjTrans; break;\
    case 'A': var = _ATLAS_CONJ_A; break;\
    default: PSErr(__func__, "Invalid trans"); return;\
    }\
} while(0);
#endif

#define PS_BLAS_OFFSET(inc,len) (inc > 0 ?  0 : (len - 1) * -(inc))
#define UNUSED(V) ((void) V)

PSBLASErr *PSBLASLastError = NULL;
static PSBLASErr _BLASLastErr = {0};

static void HandleBLASError(const char *func, const char *param,
                            const int *param_pos, const int *param_val)
{
    if (func == NULL) return;
    int pos = -1, val = -1;
    if (param_pos != NULL) pos = *param_pos;
    if (param_val != NULL) val = *param_val;
    PSErr(
        NULL, "\nBLAS Error in func '%s': invalid value %d for param '%s'",
        func, val, param
    );
    _BLASLastErr.func = func;
    _BLASLastErr.param = param;
    _BLASLastErr.param_pos = pos;
    _BLASLastErr.param_value = val;
    PSBLASLastError = &_BLASLastErr;
}

/* Native BLAS functions, loosely inspired by GNU Scientific Library (GSL):
 * https://www.gnu.org/software/gsl/doc/html/ */

static void psyc_axpy(int n, PSFloat alpha, PSFloat *x, int incx,
                      PSFloat *y, int incy)
{
    int i;

    if (alpha == 0.0) return;

    if (incx == 1 && incy == 1) {
        const int m = n % 4;

        for (i = 0; i < m; i++) y[i] += alpha * x[i];

        for (i = m; i + 3 < n; i += 4) {
            y[i] += alpha * x[i];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
    } else {
        int ix = (incx > 0 ?  0 : (n - 1) * -incx);
        int iy = (incy > 0 ?  0 : (n - 1) * -incy);

        for (i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

static void psyc_gemv(PSBLASOrder order, char trans, int m, int n,
                      PSFloat alpha, PSFloat *a, int lda, PSFloat *x,
                      PSFloat incx, PSFloat beta, PSFloat *y, int incy)
{
    static char *params[] = {
        "order", "trans", "m", "n", "alpha", "a", "lda", "x", "incx",
        "beta", "y", "incy"
    };
    PSBLASLastError = NULL;
    trans = (trans != 'C' ? trans : 'T');
    int info = 0;
    if (trans != 'N' && trans != 'T' && trans !='C') info = 2;
    else if (m < 0) info = 3;
    else if (n < 0) info = 4;
    else if (order == PSBLASColMajor) {
        int max_m = (m > 1 ? m : 1);
        if (lda < max_m) info = 7;
    } else if (order == PSBLASRowMajor) {
        int max_n = (n > 1 ? n : 1);
        if (lda < max_n) info = 7;
    }
    else if (incx == 0) info = 9;
    else if (incy == 0) info = 12;
    if (info > 0) {
        char *param = NULL;
        int pos = info - 1, val = -1;
        if ((size_t) pos < (sizeof(params) / sizeof(char*))) {
            param = params[pos];
            if (info == 2) val = (int) trans;
            else if (info == 3) val = m;
            else if (info == 4) val = n;
            else if (info == 7) val = lda;
            else if (info == 9) val = incx;
            else if (info == 12) val = incy;
        }
        HandleBLASError("GEMV", param, &info, &val);
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

    if ((order == PSBLASRowMajor && trans == 'N') ||
        (order == PSBLASColMajor && trans == 'T')) {
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
    } else if ((order == PSBLASRowMajor && trans == 'T') ||
               (order == PSBLASColMajor && trans == 'N')) {
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
    } else {
        PSErr("PSGemv", "ERROR: unrecognized operation");
        HandleBLASError("GEMV", NULL, NULL, NULL);
    }
}

static void psyc_gemm(PSBLASOrder order, char trans_a, char trans_b, int m,
                      int n, int k, PSFloat alpha, PSFloat *a, int lda,
                      PSFloat *b, int ldb, PSFloat beta, PSFloat *c, int ldc)
{
    static char *params[] = {
        "order", "trans_a", "trans_b", "m", "n", "k", "alpha", "a", "lda", "b",
        "ldb", "beta", "c", "ldc"
    };
    PSBLASLastError = NULL;
    char trans_f, trans_g;
    if (order == PSBLASRowMajor) {
        trans_f = (trans_a != 'C' ? trans_a : 'T');
        trans_g = (trans_b != 'C' ? trans_b : 'T');
    } else {
        trans_f = (trans_b != 'C' ? trans_b : 'T');
        trans_g = (trans_a != 'C' ? trans_a : 'T');
    }
    int info = 0, maxk = (k > 1 ? k : 1), maxm = (m > 1 ? m : 1),
         maxn = (n > 1 ? n : 1);
    if (order == PSBLASRowMajor) {
        if (trans_f == 'N' && lda < maxk) info = 9;
        else if (trans_f != 'N' && lda < maxm) info = 9;
        if (trans_g == 'N' && ldb < maxn) info = 11;
        else if (trans_g != 'N' && ldb < maxk) info = 11;
        if (ldc < maxn) info = 14;
    } else {
        if (trans_f == 'N' && ldb < maxk) info = 11;
        else if (trans_f != 'N' && ldb < maxn) info = 11;
        if (trans_g == 'N' && lda < maxm) info = 9;
        else if (trans_g != 'N' && lda < maxk) info = 9;
        if (ldc < maxm) info = 14;
    }
    if (info != 0) {
        char *param = NULL;
        int pos = info - 1, val = -1;
        if ((size_t) pos < (sizeof(params) / sizeof(char*))) {
            param = params[pos];
            if (info == 9) val = lda;
            else if (info == 11) val = ldb;
            else if (info == 14) val = ldc;
        }
        HandleBLASError("GEMM", param, &info, &val);
        return;
    }
    if (alpha == 0.0 && beta == 1.0) return;
    int n1, n2, ldf, ldg, i, j, _k;
    PSFloat *f, *g;
    if (order == PSBLASRowMajor) {
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
                for (_k = 0; _k < k; _k++) {
                    temp += f[ldf * i + _k] * g[ldg * j + _k];
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
    } else {
        PSErr("PSGemm", "ERROR: unrecognized operation");
        HandleBLASError("GEMM", NULL, NULL, NULL);
    }
}

/* Wrapper public functions */

void PSAxpy(int n, PSFloat alpha, PSFloat *x, int incx, PSFloat *y, int incy) {
#if defined(HAS_CBLAS) && !defined(USE_PSYC_BLAS)
    UNUSED(psyc_axpy);
#ifdef PS_DOUBLE_PRECISION
    cblas_daxpy(n, alpha, x, incx, y, incy);
#else
    cblas_saxpy(n, alpha, x, incx, y, incy);
#endif
#else
    psyc_axpy(n, alpha, x, incx, y, incy);
#endif
}

void PSGemv(PSBLASOrder order, char trans, int m, int n, PSFloat alpha,
            PSFloat *a, int lda, PSFloat *x, PSFloat incx, PSFloat beta,
            PSFloat *y, int incy)
{
    PSBLASLastError = NULL;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    SetBLASParamErrorProc(HandleBLASError);
#endif
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

void PSGemm(PSBLASOrder order, char trans_a, char trans_b, int m, int n, int k,
            PSFloat alpha, PSFloat *a, int lda, PSFloat *b, int ldb,
            PSFloat beta, PSFloat *c, int ldc)
{
    PSBLASLastError = NULL;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    SetBLASParamErrorProc(HandleBLASError);
#endif
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

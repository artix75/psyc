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

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>
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
                            const PSBLAS_int *param_pos,
                            const PSBLAS_int *param_val)
{
    if (func == NULL) return;
    int pos = -1;
    PSBLAS_int val = -1;
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

/* Get a string representing the BLAS error `err`. If `err->param` is NULL,
 * the optional `param_names` argument can be used to pass the names of the
 * parameters (determined by 1-based err->param_pos value).
 * NOTE: the returned string is a `static` string: it should never be freed
 * and it's always overwritten by subsequent calls.
 * Return value: the error string of NULL if `err` is NULL.*/
const char *PSBLASErrorStr(PSBLASErr *err, const char **param_names) {
    static char errstr[1024] = {0};
    if (err == NULL) return NULL;
    char dfparam[55] = {0};
    char location[255] = {0};
    char idxstr[25] = {0};
    const char *param = err->param;
    if (param == NULL && param_names != NULL && err->param_pos > 0)
        param = param_names[err->param_pos - 1];
    if (param == NULL) {
        if (err->param_pos > 0) {
            snprintf(dfparam, 55, "param %ld", err->param_pos);
            param = dfparam;
        } else param = "param 'unknown'";
    }
    if (err->func != NULL)
        snprintf(location, 255, " in function `%s`", err->func);
    if (err->array_index >= 0)
        snprintf(idxstr, 25, "[%ld]", err->array_index);
    snprintf(
        errstr, 1024, "BLAS Error%s: invalid value %ld for %s%s",
        location, err->param_value, param, idxstr
    );
    return errstr;
}

/* Check variadic arguments integer values against ensuring that they're
 * not greater than PSBLAS_MAX.
 * Checked values can be either `long` scalars or `long` arrays.
 * The mandatory `argformat` argument is a string used to specify how variadic
 * arguments must be intepreted.
 * The format string for every variadic argument is:
 *  - zero or more of the following flag characters:
 *    - '@': indicates that the next variadic argument is an array of `long`
 *    - '$': indicates that the next variadic argument(s) are precedeed
 *           by their names. Names are strings passed as variadic arguments
 *           just before the scalar and/or vector argument.
 *  - The number of the next variadic argument (scalar) or the length of the
 *    next array argument (if '@' was set):
 *    - As a fixed number string representation (between 1 and 9).
 *    - '*': indicates that argument count/array length must be read from
 *          the next variadic argument (casted to `int`).
 * The optional `err` argument can be used to retrieve more info about the
 * value that exceeds PSBLAS_MAX:
 *   - `param` string is automatically set if argument names is provided via
 *     the '$' flag.
 *   - `param_pos` indicates the (1-based) position of the invalid argument.
 *   - `param_value` contains the value of the invalid argument that exceeded
 *      PSBLAS_MAX.
 * WARN: the count/length argument (if '*' is used in the format string) must
 * always preceed the argument name (if '$' flag has also used in the
 * format string).
 *
 * Examples:
 * ```
 * PSBLASErr err = {0};
 * long a = 10, b = 1000;
 * long nums[3] = {10, 100, 1000};
 * int valid = PSBLASCheckLimits(&err, "2", a, b);
 * valid = PSBLASCheckLimits(&err, "@3", nums);
 * valid = PSBLASCheckLimits(&err, "2@3", a, b, nums);
 * valid = PSBLASCheckLimits(&err, "*@*", 2, a, b, 3, nums);
 * valid = PSBLASCheckLimits(&err, "$*$@*", 2, "a", a, "b", b, 3,
 *                           "numbers", nums);
 * ```
 *
 * Return value: 1 if all arguments are less or equal than PSBLAS_MAX,
 * zero if one of the arguments exceeded PSBLAS_MAX.
 */
int PSBLASCheckLimits(PSBLASErr *err, const char *argformat, ...) {
    if (argformat == NULL) return 1;
    int valid = 1;
    const char *p = argformat, *param_name = NULL;
    char c = 0;
    long val = 0;
    int argidx = -1, arridx = -1;
    int is_array = 0, get_param_name = 0;
    int len = 0;
    if (err != NULL) err->array_index = -1;
    va_list args;
    va_start(args, argformat);
    while ((c = *(p++))) {
        if (c == '@') {
            assert(!is_array);
            is_array = 1;
            continue;
        } else if (c == '$') {
            get_param_name = 1;
            continue;
        } else {
            if (c == '*') {
                len = va_arg(args, int);
                argidx++;
            } else {
                assert(isdigit(c));
                char digit[2];
                digit[0] = c;
                digit[1] = '\0';
                len = atoi(digit);
            }
            long *array = NULL;
            if (is_array) {
                if (get_param_name) {
                    param_name = va_arg(args, char *);
                    argidx++;
                    get_param_name = 0;
                }
                array = va_arg(args, long *);
                assert(array != NULL);
                is_array = 0;
                argidx++;
                arridx = -1;
            }
            while (len-- > 0) {
                if (array != NULL) {
                    val = *(array++);
                    arridx++;
                } else {
                    if (get_param_name) {
                        param_name = va_arg(args, char *);
                        argidx++;
                    }
                    val = va_arg(args, long);
                    argidx++;
                }
                valid = (val <= PSBLAS_MAX);
                if (!valid) break;
            }
            get_param_name = 0;
        }
        if (!valid) {
            if (err != NULL) {
                err->param = param_name;
                err->param_pos = argidx + 1;
                err->param_value = val;
                err->array_index = arridx;
            }
            break;
        }
    }
    va_end(args);
    return valid;
}

/* Native BLAS functions, loosely inspired by GNU Scientific Library (GSL):
 * https://www.gnu.org/software/gsl/doc/html/ */

static void psyc_axpy(int n, PSFloat alpha, PSFloat *x, PSBLAS_int incx,
                      PSFloat *y, PSBLAS_int incy)
{
    PSBLAS_int i;
    if (alpha == 0.0) return;
    if (incx == 1 && incy == 1) {
        const PSBLAS_int m = n % 4;
        for (i = 0; i < m; i++) y[i] += alpha * x[i];
        for (i = m; i + 3 < n; i += 4) {
            y[i] += alpha * x[i];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
    } else {
        PSBLAS_int ix = (incx > 0 ?  0 : (n - 1) * -incx);
        PSBLAS_int iy = (incy > 0 ?  0 : (n - 1) * -incy);
        for (i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

static void psyc_gemv(PSBLASOrder order, char trans, PSBLAS_int m, int n,
                      PSFloat alpha, PSFloat *a, PSBLAS_int lda, PSFloat *x,
                      PSFloat incx, PSFloat beta, PSFloat *y, PSBLAS_int incy)
{
    static char *params[] = {
        "order", "trans", "m", "n", "alpha", "a", "lda", "x", "incx",
        "beta", "y", "incy"
    };
    PSBLASLastError = NULL;
    trans = (trans != 'C' ? trans : 'T');
    PSBLAS_int info = 0;
    if (trans != 'N' && trans != 'T' && trans !='C') info = 2;
    else if (m < 0) info = 3;
    else if (n < 0) info = 4;
    else if (order == PSBLASColMajor) {
        PSBLAS_int max_m = (m > 1 ? m : 1);
        if (lda < max_m) info = 7;
    } else if (order == PSBLASRowMajor) {
        PSBLAS_int max_n = (n > 1 ? n : 1);
        if (lda < max_n) info = 7;
    }
    else if (incx == 0) info = 9;
    else if (incy == 0) info = 12;
    if (info > 0) {
        char *param = NULL;
        PSBLAS_int pos = info - 1;
        PSBLAS_int val = -1;
        if ((size_t) pos < (sizeof(params) / sizeof(char*))) {
            param = params[pos];
            if (info == 2) val = (PSBLAS_int) trans;
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

    PSBLAS_int lenx = 0, leny = 0;
    if (trans == 'N') {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }

    PSBLAS_int i, j, iy, ix;
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

static void psyc_gemm(PSBLASOrder order, char trans_a, char trans_b,
                      PSBLAS_int m, PSBLAS_int n, PSBLAS_int k, PSFloat alpha,
                      PSFloat *a, PSBLAS_int lda, PSFloat *b, PSBLAS_int ldb,
                      PSFloat beta, PSFloat *c, PSBLAS_int ldc)
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
    PSBLAS_int info = 0;
    PSBLAS_int maxk = (k > 1 ? k : 1),
         maxm = (m > 1 ? m : 1),
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
        PSBLAS_int pos = info - 1;
        PSBLAS_int val = -1;
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
    PSBLAS_int n1, n2, ldf, ldg, i, j, _k;
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

void PSAxpy(PSBLAS_int n, PSFloat alpha, PSFloat *x, PSBLAS_int incx,
            PSFloat *y, PSBLAS_int incy)
{
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

void PSGemv(PSBLASOrder order, char trans, PSBLAS_int m, PSBLAS_int n,
            PSFloat alpha, PSFloat *a, PSBLAS_int lda, PSFloat *x,
            PSFloat incx, PSFloat beta, PSFloat *y, PSBLAS_int incy)
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

void PSGemm(PSBLASOrder order, char trans_a, char trans_b, PSBLAS_int m,
            PSBLAS_int n, PSBLAS_int k, PSFloat alpha, PSFloat *a,
            PSBLAS_int lda, PSFloat *b, PSBLAS_int ldb,
            PSFloat beta, PSFloat *c, PSBLAS_int ldc)
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

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

#ifndef __PS_AVX_H
#define __PS_AVX_H

#include <assert.h>
#include "types.h"

#ifdef USE_AVX

#define AVXGetStepLen(s) AVXComputeStepLength(s, 0, NULL)
#define AVXGetDotStepLen(s) AVXComputeStepLength(s, 1, NULL)

#define AVX_STORE_MODE_NORM 0
#define AVX_STORE_MODE_ADD  1
#define AVX_STORE_MODE_SUB  2

/* Iteratively compute AVX dot product (sum of multiplication of two arrays)
 * up to N elements of the the array `x` and array `y` of size `size` and
 * store results in variable `res`.
 * Note that the function will process elements divided in steps until
 * every step fills the AVX registers, so the array could not be completely
 * multiplied. The argument `i` will keep count of the processed elements,
 * so that you can manually complete the operation.
 * If `is_recurrent`, use `t` for recurrent network values. */

#define AVXIterativeDotProduct(size, x, y, res, i, is_recurrent, t) do { \
    int avx_step_len = AVXGetDotStepLen(size);\
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) {\
        PSFloat *x_vector = x + i;\
        if (is_recurrent) x_vector += (t * size);\
        PSFloat *y_vector = y + i;\
        int c = 0;\
        res += AVXDotProduct(x_vector, y_vector, avx_step_len, &c);\
        assert(c == avx_step_len);\
        i += avx_step_len; \
    }\
} while (0)


/* Same as AVXDotProduct, but multply array `x` by itself (square). */
#define AVXIterativeDotSquare(size, x, res, i, is_recurrent, t) do {\
    int avx_step_len = AVXGetDotStepLen(size);\
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) {\
        PSFloat *x_vector = x + i;\
        if (is_recurrent) x_vector += (t * size);\
        int c = 0;\
        res += AVXDotProduct(x_vector, x_vector, avx_step_len, &c);\
        assert(c == avx_step_len);\
        i += avx_step_len;\
    }\
} while (0)

/* Iteratively multply via AVX up to N elements of the the array `x` of size
 * `size` by value `val` and store results in array `dest`.
 * Note that the function will multiply elements divided in steps until
 * every step fills the AVX registers, so the array could not be completely
 * multiplied. The argument `i` will keep count of the processed elements,
 * so that you can manually complete the operation.
 * Results are stored in `dest` using the AVX operator `mode` (NORM, SUB,
 * etc.).
 * If `is_recurrent`, use `t` for recurrent network values. */

#define AVXIterativeMultiplyValue(size, x, val, dest, i, is_recurrent, t, mode) do { \
    int avx_step_len = AVXGetStepLen(size);\
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) {\
        PSFloat *x_vector = x + i;\
        if (is_recurrent) x_vector += (t * size);\
        int c = AVXMultiplyValue(x_vector, val, avx_step_len, dest + i, mode);\
        assert(c == avx_step_len);\
        i += avx_step_len;\
    } \
} while (0)

/* Iteratively multply via AVX up to N elements of the the array `x1` of size
 * `size` by value `v1` and elements of array `x2` with value `v2` and then
 * store results in array `dest`.
 * Note that the function will multiply elements divided in steps until
 * every step fills the AVX registers, so the array could not be completely
 * multiplied. The argument `i` will keep count of the processed elements,
 * so that you can manually complete the operation.
 * Results of `x1` multiplication are stored in `d` using the AVX operator
 * `m1` (NORM, SUB, etc.), while results of `x2` are stored using operator
 * `m2`.
 * If `is_rec` (recurrent), use `t` for recurrent network values. */

#define AVXIterativeMultiplyValues(size, x1, v1, x2, v2, dest, i,\
 is_recurrent, t, mode1, mode2) do {\
    int avx_step_len = AVXGetStepLen(size); \
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        PSFloat *xv1 = x1 + i;\
        PSFloat *xv2 = x2 + i;\
        PSFloat *dd = dest + i;\
        if (is_recurrent) {\
            xv1 += (t * size);\
            xv2 += (t * size);\
        }\
        int c1 = AVXMultiplyValue(xv1, v1, avx_step_len, dd, mode1);\
        assert(c1 == avx_step_len);\
        int c2 = AVXMultiplyValue(xv2, v2, avx_step_len, dd, mode2);\
        assert(c2 == c1);\
        i += avx_step_len;\
    }\
} while (0)

/* Iteratively sum via AVX up to N elements of the the array `x` with array
 * `y` and store results into array `dest` using operator `mode`. */
#define AVXIterativeSum(size, x, y, dest, i, mode) do {\
    int avx_step_len = AVXGetStepLen(size);\
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    int x_is_dest = (x == dest);\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) {\
        int doffs = (x_is_dest ? i : 0);\
        int c = AVXSum(x + i, y + i, avx_step_len, dest + doffs, mode);\
        assert(c == avx_step_len);\
        i += avx_step_len;\
    }\
} while (0)

/* Iteratively subtract via AVX up to N elements of the the array `y` from
 *array `x` and store results into array `dest` using operator `mode`. */
#define AVXIterativeDiff(size, x, y, dest, i, mode) do {\
    int avx_step_len = AVXGetStepLen(size);i\
    int avx_steps = (avx_step_len > 0 ? size / avx_step_len : 0), avx_step;\
    int x_is_dest = (x == dest);\
    for (avx_step = 0; avx_step < avx_steps; avx_step++) {\
        int doffs = (x_is_dest ? i : 0);\
        int c = AVXDiff(x + i, y + i, avx_step_len, dest + doffs, mode);\
        assert(c == avx_step_len);\
        i += avx_step_len;\
    }\
} while (0)

extern const int AVX_VECTOR_SIZE;
extern const int AVX128_VECTOR_SIZE;
extern const int AVX256_VECTOR_SIZE;
extern const int AVX_MIN_VECTOR_SIZE;
extern const int AVX_MAX_VECTOR_SIZE;

int AVXComputeStepLength(int size, int allow_multiple_vectors, int *bits);
PSFloat AVXDotProduct(PSFloat *x, PSFloat *y, int size, int *count);
int AVXMultiplyValue(PSFloat *x, PSFloat value, int size, PSFloat *dest,
                     int mode);
int AVXMultiply(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode);
int AVXSum(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode);
int AVXDiff(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode);

#endif /* USE_AVX*/
#endif /*__PS_AVX_H*/

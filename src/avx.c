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

#ifdef USE_AVX
#include <stdio.h>
#include <assert.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

#include "avx.h"

#define _AVX_VECTOR_SIZE (256 / (8 * sizeof(PSFloat)))
#define MAX_AVX_VECTORS 4

#ifdef PS_DOUBLE_PRECISION
#define AVX128LoadUnalign(v)        _mm_loadu_pd(v)
#define AVX128Multiply(a, b)        _mm_mul_pd(a, b)
#define AVX128Divide(a, b)          _mm_div_pd(a, b)
#define AVX128HorizontalAdd(a, b)   _mm_hadd_pd(a, b)
#define AVX128Add(a, b)             _mm_add_pd(a, b)
#define AVX128Sub(a, b)             _mm_sub_pd(a, b)
#define AVX128SetVal(v)             _mm_set_pd(v, v)
#define AVX128StoreUnalign(dst,src) _mm_storeu_pd(dst, src)
#define AVX128Neg(a)                _mm_xor_pd(a, _mm_set1_pd(-0.0))
#define AVX128Min(a,b)              _mm_min_pd(a,b)
#define AVX128Max(a,b)              _mm_max_pd(a,b)

#define AVX256LoadUnalign(v)        _mm256_loadu_pd(v)
#define AVX256Multiply(a, b)        _mm256_mul_pd(a, b)
#define AVX256Divide(a, b)          _mm256_div_pd(a, b)
#define AVX256HorizontalAdd(a, b)   _mm256_hadd_pd(a, b)
#define AVX256Extract128(v, i)      _mm256_extractf128_pd(v, i)
#define AVX256Add(a, b)             _mm256_add_pd(a, b)
#define AVX256Sub(a, b)             _mm256_sub_pd(a, b)
#define AVX256Permute2F128(a, b, i) _mm256_permute2f128_pd(a, b, i)
#define AVX256Blend(a, b, mask)     _mm256_blend_pd(a, b, mask)
#define AVX256SetVal(v)             _mm256_set_pd(v, v, v, v)
#define AVX256StoreUnalign(dst,src) _mm256_storeu_pd(dst, src)
#define AVX256Neg(a)                _mm256_xor_pd(a, _mm256_set1_pd(-0.0))
#define AVX256Min(a,b)              _mm256_min_pd(a,b)
#define AVX256Max(a,b)              _mm256_max_pd(a,b)
#define AVX256_BLEND_MASK           0x0C /* 0b1100 */
typedef __m128d AVX128;
typedef __m256d AVX256;
#else
#define AVX128LoadUnalign(v)        _mm_loadu_ps(v)
#define AVX128Multiply(a, b)        _mm_mul_ps(a, b)
#define AVX128Divide(a, b)          _mm_div_ps(a, b)
#define AVX128HorizontalAdd(a, b)   _mm_hadd_ps(a, b)
#define AVX128Add(a, b)             _mm_add_ps(a, b)
#define AVX128Sub(a, b)             _mm_sub_ps(a, b)
#define AVX128SetVal(v)             _mm_set_ps(v, v, v, v)
#define AVX128StoreUnalign(dst,src) _mm_storeu_ps(dst, src)
#define AVX128Neg(a)                _mm_xor_ps(a, _mm_set1_ps(-0.0))
#define AVX128Min(a,b)              _mm_min_ps(a,b)
#define AVX128Max(a,b)              _mm_max_ps(a,b)

#define AVX256LoadUnalign(v)        _mm256_loadu_ps(v)
#define AVX256Multiply(a, b)        _mm256_mul_ps(a, b)
#define AVX256Divide(a, b)          _mm256_div_ps(a, b)
#define AVX256HorizontalAdd(a, b)   _mm256_hadd_ps(a, b)
#define AVX256Extract128(v, i)      _mm256_extractf128_ps(v, i)
#define AVX256Add(a, b)             _mm256_add_ps(a, b)
#define AVX256Sub(a, b)             _mm256_sub_ps(a, b)
#define AVX256Permute2F128(a, b, i) _mm256_permute2f128_ps(a, b, i)
#define AVX256Blend(a, b, mask)     _mm256_blend_ps(a, b, mask)
#define AVX256SetVal(v)             _mm256_set_ps(v, v, v, v, v, v, v, v)
#define AVX256StoreUnalign(dst,src) _mm256_storeu_ps(dst, src)
#define AVX256Neg(a)                _mm256_xor_ps(a, _mm256_set1_ps(-0.0))
#define AVX256Min(a,b)              _mm256_min_ps(a,b)
#define AVX256Max(a,b)              _mm256_max_ps(a,b)
#define AVX256_BLEND_MASK           0xF0 /* 0b11110000 */
typedef __m128 AVX128;
typedef __m256 AVX256;
#endif

const int AVX_VECTOR_SIZE = _AVX_VECTOR_SIZE;
const int AVX256_VECTOR_SIZE = _AVX_VECTOR_SIZE;
const int AVX128_VECTOR_SIZE = _AVX_VECTOR_SIZE / 2;
const int AVX_MIN_VECTOR_SIZE = _AVX_VECTOR_SIZE / 2;
const int AVX_MAX_VECTOR_SIZE = MAX_AVX_VECTORS * _AVX_VECTOR_SIZE;

int AVXComputeStepLength(int size, int allow_multiple_vectors, int *bits) {
    if (size < AVX_MIN_VECTOR_SIZE) return 0;
    if (size > AVX_MAX_VECTOR_SIZE) size = AVX_MAX_VECTOR_SIZE;
    int regbits = (size < AVX256_VECTOR_SIZE ? 128 : 256);
    if (bits != NULL) *bits = regbits;
    int reglen = regbits / (8 * sizeof(PSFloat));
    if (!allow_multiple_vectors) return reglen;
    int num_vectors = size / reglen;
    assert(num_vectors <= MAX_AVX_VECTORS);
    while (num_vectors > 1 && (num_vectors % 2) != 0) num_vectors--;
    size = num_vectors * reglen; /* Ensure vector_len is multiple of reglen */
    return size;
}

/* Simulatenously compute dot product between two arrays (`x` and `y`) using
 * AVX. Argument `size` is the size of the arrays (they must have the same
 * length). Since only multiple of AVX vectors will be computed, only a
 * part of the array could be used for computation. Use the pointer
 * `count` ito get the exact number of elements in the arrays used
 * for computation.
 * The function returns the result of the dot product for the computed
 * elements (that is the sum of x * y used elements). */
PSFloat AVXDotProduct(PSFloat *x, PSFloat *y, int size, int *count) {
    if (size < AVX_MIN_VECTOR_SIZE) return 0;
    if (size > AVX_MAX_VECTOR_SIZE) size = AVX_MAX_VECTOR_SIZE;
    PSFloat res = 0;
    int regbits = (size < AVX256_VECTOR_SIZE ? 128 : 256);
    int reglen = regbits / (8 * sizeof(PSFloat));
    int num_vectors = size / reglen;
    assert(num_vectors <= MAX_AVX_VECTORS);
    size = num_vectors * reglen; /* Ensure vector_len is multiple of reglen */
    if (count != NULL) *count = size;
    int sumv_len = reglen, count_divisor = 1, i;
    void *sumv = NULL;
    AVX128 dp128;
    AVX256 dp256;
    /* Use native _mm_dp_ps/_mm256_dp_ps if PSFloat is single precision,
     * since _mm256_dp_ps is only available for 32bit float. */
    int use_instrinsic = (sizeof(PSFloat) == 4);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128LoadUnalign(y);
        if (use_instrinsic) {
            AVX128 xy = _mm_dp_ps(xv, yv, 0xFF);
            return xy[0];
        }
        AVX128 xy = AVX128Multiply(xv, yv);
        AVX128 zeros = AVX128SetVal(0);
        dp128 = AVX128HorizontalAdd(xy, zeros);
        count_divisor = 2;
        sumv = &dp128;
    } else {
        AVX256 xv[MAX_AVX_VECTORS];
        AVX256 yv[MAX_AVX_VECTORS];
        AVX256 xy[MAX_AVX_VECTORS];
        AVX256 tempv[MAX_AVX_VECTORS];
        /* Load array elements into AVX vectors `xv` && `yv` and multiply
         * their elements storing them into `xy` vectors. */
        for (i = 0; i < num_vectors; i++) {
            int idx = i * AVX256_VECTOR_SIZE;
            xv[i] = AVX256LoadUnalign(x + idx);
            yv[i] = AVX256LoadUnalign(y + idx);
            if (use_instrinsic) {
                xy[i] = _mm256_dp_ps(xv[i], yv[i], 0xFF);
                res += (xy[i][0] + xy[i][4]);
            } else xy[i] = AVX256Multiply(xv[i], yv[i]);
        }
        if (use_instrinsic) return res;
        int templen = (num_vectors / 2);
        if (templen < 1) templen = 1;
        for (i = 0; i < templen; i++) {
            assert(i < num_vectors);
            int idx1 = i * 2;
            int idx2 = idx1 + 1;
            AVX256 xy1 = xy[idx1], xy2;
            if (idx2 < num_vectors) xy2 = xy[idx2];
            else {
                xy2 = AVX256SetVal(0);
                count_divisor = 2;
            }
            /* Horizontal sum adiacent pairs of elements in vectors `xy1` and
             * `xy2` storing them packed in `tempv[i]`. Example for vectors
             * composed of four elements:
             * xy1[0]+xy[1], xy2[0]+xy2[1], xy1[2]+xy1[3], xy2[2]+xy2[3] */
            tempv[i] = AVX256HorizontalAdd(xy1, xy2);
        }
        if (templen == 1) {
            AVX256 temp = tempv[0];
            /* Since the order generated by AVX256HorizontalAdd is
             * xy1, xy1, xy2, xy2, we split it into two 128 bit vectors,
             * and then we sum them into a 128 bit vector. */
            AVX128 lo128 = AVX256Extract128(temp, 0);
            AVX128 hi128 = AVX256Extract128(temp, 1);
            dp128 = AVX128Add(lo128, hi128);
            sumv = &dp128;
            sumv_len = AVX128_VECTOR_SIZE;
        } else {
            assert(templen <= 2);
            /* low to high:
             * xy0[2]+xy0[3] xy1[2]+xy1[3] xy2[0]+xy2[1] xy3[0]+xy3[1] */
            AVX256 swapped = AVX256Permute2F128(tempv[0], tempv[1], 0x21);
            /* low to high:
             * xy0[0]+xy0[1] xy1[0]+xy1[1] xy2[2]+xy2[3] xy3[2]+xy3[3] */
            AVX256 blended = AVX256Blend(tempv[0], tempv[1], AVX256_BLEND_MASK);
            dp256 = AVX256Add(swapped, blended);
            sumv = &dp256;
        }
    }
    int numbers_to_sum = sumv_len / count_divisor;
    assert(numbers_to_sum > 0);
    if (numbers_to_sum == 1) return *((PSFloat *) sumv);
    int vidx = 0;
    while (numbers_to_sum > 0) {
        PSFloat n = ((PSFloat *) sumv)[vidx];
        res += n;
        vidx += 1;
        numbers_to_sum--;
    }
    return res;
}

void AVX128StoreWithMode(PSFloat *dest, AVX128 src, int mode) {
    if (mode != AVX_STORE_MODE_SET) {
        AVX128 temp = AVX128LoadUnalign(dest);
        if (mode == AVX_STORE_MODE_ADD) src = AVX128Add(temp, src);
        else if (mode == AVX_STORE_MODE_SUB) src = AVX128Sub(temp, src);
    }
    AVX128StoreUnalign(dest, src);
}

void AVX256StoreWithMode(PSFloat *dest, AVX256 src, int mode) {
    if (mode != AVX_STORE_MODE_SET) {
        AVX256 temp = AVX256LoadUnalign(dest);
        if (mode == AVX_STORE_MODE_ADD) src = AVX256Add(temp, src);
        else if (mode == AVX_STORE_MODE_SUB) src = AVX256Sub(temp, src);
    }
    AVX256StoreUnalign(dest, src);
}

/* Simulatenously multiply values in array `x` by `value` using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXMultiplyValue(PSFloat *x, PSFloat value, int size, PSFloat *dest,
                       int mode)
{
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128SetVal(value);
        AVX128 xy = AVX128Multiply(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256SetVal(value);
        AVX256 xy = AVX256Multiply(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously add `value` to values in array `x` using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXAddValue(PSFloat *x, PSFloat value, int size, PSFloat *dest, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128SetVal(value);
        AVX128 xy = AVX128Add(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256SetVal(value);
        AVX256 xy = AVX256Add(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously divide elements in array `x` by `value` using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXDivideValue(PSFloat *x, PSFloat value, int size, PSFloat *dest,
                   int mode)
{
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128SetVal(value);
        AVX128 xy = AVX128Divide(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256SetVal(value);
        AVX256 xy = AVX256Divide(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously divide `value` by values in array `x` with `value` using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXValueDivide(PSFloat value, PSFloat *x, int size, PSFloat *dest,
                   int mode)
{
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128SetVal(value);
        AVX128 xy = AVX128Divide(yv, xv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256SetVal(value);
        AVX256 xy = AVX256Divide(yv, xv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously multiply values in array `x` by values in array `y`,
 * using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXMultiply(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128LoadUnalign(y);
        AVX128 xy = AVX128Multiply(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256LoadUnalign(y);
        AVX256 xy = AVX256Multiply(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously divide values in array `x` by values in array `y`,
 * using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were multiplied. */
int AVXDivide(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128LoadUnalign(y);
        AVX128 xy = AVX128Divide(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256LoadUnalign(y);
        AVX256 xy = AVX256Divide(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously add values in array `x` to values in array `y`, using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `x` that were added to `y`. */
int AVXSum(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128LoadUnalign(y);
        AVX128 xy = AVX128Add(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256LoadUnalign(y);
        AVX256 xy = AVX256Add(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously subtract values in array `y` from values in array `y`,
 * using AVX.
 * Argument `size` is the size of the array. Since only multiple of AVX
 * vectors will be computed, only a part of the array could be used for
 * computation.
 * The results of the operation will be stored into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - AVX_STORE_MODE_SET: the result will directly stored into `dest`.
 * Return value: count of elements in `y` that were subtracted from `x`. */
int AVXDiff(PSFloat *x, PSFloat *y, int size, PSFloat *dest, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 yv = AVX128LoadUnalign(y);
        AVX128 xy = AVX128Sub(xv, yv);
        AVX128StoreWithMode(dest, xy, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 yv = AVX256LoadUnalign(y);
        AVX256 xy = AVX256Sub(xv, yv);
        AVX256StoreWithMode(dest, xy, mode);
    }
    return size;
}

/* Simulatenously invert sign (negate) elements of  array `x` of size `size`
 * and store results into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - mode: the result will directly stored into `dest`.
 * Return value: count of processed elements. */
int AVXNegate(PSFloat *x, PSFloat *dest, int size, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 res = AVX128Neg(xv);
        AVX128StoreWithMode(dest, res, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 res = AVX256Neg(xv);
        AVX256StoreWithMode(dest, res, mode);
    }
    return size;
}

/* Simulatenously clip elements of  array `x` of size `size`
 * within `min` and `max` and store results into array `dest`.
 * The argument `mode` can be used to specify how the result should be
 * stored:
 *  - AVX_STORE_MODE_ADD: the result will be added to values of `dest`.
 *  - AVX_STORE_MODE_SUB: the result will be subtracted from values of `dest`.
 *  - mode: the result will directly stored into `dest`.
 * Return value: count of processed elements. */
int AVXClip(PSFloat *x, PSFloat min, PSFloat max, PSFloat *dest,
            int size, int mode) {
    int regbits = 0;
    size = AVXComputeStepLength(size, 0, &regbits);
    if (size == 0) return 0;
    assert(regbits != 0);
    if (regbits == 128) {
        AVX128 xv = AVX128LoadUnalign(x);
        AVX128 minv = AVX128SetVal(min);
        AVX128 maxv = AVX128SetVal(max);
        AVX128 res = AVX128Min(AVX128Max(xv, minv), maxv);
        AVX128StoreWithMode(dest, res, mode);
    } else {
        AVX256 xv = AVX256LoadUnalign(x);
        AVX256 minv = AVX256SetVal(min);
        AVX256 maxv = AVX256SetVal(max);
        AVX256 res = AVX256Min(AVX256Max(xv, minv), maxv);
        AVX256StoreWithMode(dest, res, mode);
    }
    return size;
}

#endif /* USE_AVX */

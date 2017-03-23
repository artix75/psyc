/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

#include "avx.h"

#define _AVX_VECTOR_SIZE (256 / (8 * sizeof(double)))

int AVX_VECTOR_SIZE = _AVX_VECTOR_SIZE;

int AVX_IDX1 = _AVX_VECTOR_SIZE;
int AVX_IDX2 = _AVX_VECTOR_SIZE * 2;
int AVX_IDX3 = _AVX_VECTOR_SIZE * 3;

double _avx_zeros[2] = {0.0, 0.0};

double avx_dot_product2(double * x, double * y) {
    __m128d xv = _mm_load_pd(x);
    __m128d yv = _mm_load_pd(y);
    __m128d xy = _mm_mul_pd( xv, yv );
    __m128d dotproduct = _mm_hadd_pd( xy, xy );
    double * d = (double*) &dotproduct;
    return *d;
}

double avx_dot_product4(double * x, double * y) {
    __m256d xv = _mm256_load_pd(x);
    __m256d yv = _mm256_load_pd(y);
    __m256d xy = _mm256_mul_pd( xv, yv );
    __m256d temp = _mm256_hadd_pd( xy, xy );
    __m128d lo128 = _mm256_extractf128_pd( temp, 0 );
    __m128d hi128 = _mm256_extractf128_pd( temp, 1 );
    __m128d dotproduct = _mm_add_pd( lo128, hi128 );
    double * d = (double*) &dotproduct;
    return *d;
}

double avx_dot_product8(double * x, double * y) {
    __m256d xv = _mm256_load_pd(x);
    __m256d yv = _mm256_load_pd(y);
    __m256d wv = _mm256_load_pd(x + AVX_IDX1);
    __m256d zv = _mm256_load_pd(y + AVX_IDX1);
    __m256d xy = _mm256_mul_pd( xv, yv );
    __m256d zw = _mm256_mul_pd( zv, wv );
    __m256d temp = _mm256_hadd_pd( xy, zw );
    __m128d lo128 = _mm256_extractf128_pd( temp, 0 );
    __m128d hi128 = _mm256_extractf128_pd( temp, 1 );
    __m128d dotproduct = _mm_add_pd( lo128, hi128 );
    double * d = (double*) &dotproduct;
    return d[0] + d[1];
}

// Computes Dot Product between 2 arrays of 16 doubles at time

double avx_dot_product16(double * x, double * y) {
    __m256d xv0 = _mm256_load_pd(x);
    __m256d yv0 = _mm256_load_pd(y);
    __m256d xv1 = _mm256_load_pd(x + AVX_IDX1);
    __m256d yv1 = _mm256_load_pd(y + AVX_IDX1);
    __m256d xv2 = _mm256_load_pd(x + AVX_IDX2);
    __m256d yv2 = _mm256_load_pd(y + AVX_IDX2);
    __m256d xv3 = _mm256_load_pd(x + AVX_IDX3);
    __m256d yv3 = _mm256_load_pd(y + AVX_IDX3);
    
    __m256d xy0 = _mm256_mul_pd( xv0, yv0 );
    __m256d xy1 = _mm256_mul_pd( xv1, yv1 );
    __m256d xy2 = _mm256_mul_pd( xv2, yv2 );
    __m256d xy3 = _mm256_mul_pd( xv3, yv3 );
    
    // low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
    __m256d temp01 = _mm256_hadd_pd( xy0, xy1 );
    
    // low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
    __m256d temp23 = _mm256_hadd_pd( xy2, xy3 );
    
    // low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
    __m256d swapped = _mm256_permute2f128_pd( temp01, temp23, 0x21 );
    
    // low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
    __m256d blended = _mm256_blend_pd(temp01, temp23, 0b1100);
    
    __m256d dotproduct = _mm256_add_pd( swapped, blended );
    
    double * d = (double*) &dotproduct;
    //printf("LO: %lf\n", d[0]);
    //printf("HI: %lf\n", d[1]);
    return d[0] + d[1] + d[2] + d[3];
}

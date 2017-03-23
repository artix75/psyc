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

#ifndef __AVX_H
#define __AVX_H

#define AVXGetDotStepLen(s) (s >= 16 ? 16 : (s >= 8 ? 8 : (s >= 4 ? 4 : 2)))
#define AVXGetDotProductFunc(s) (s >= 16 ? avx_dot_product16 : \
    (s >= 8 ? avx_dot_product8 : \
    (s >= 4 ? avx_dot_product4 : avx_dot_product2)))

extern int AVX_VECTOR_SIZE;

typedef double (* avx_dot_product)(double * x, double * y);

double avx_dot_product2(double * x, double * y);
double avx_dot_product4(double * x, double * y);
double avx_dot_product8(double * x, double * y);
double avx_dot_product16(double * x, double * y);

#endif //__AVX_H

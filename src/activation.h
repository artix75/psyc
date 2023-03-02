/*
 * Copyright (C) 2016-2023 Fabio Nicotra <artix2 at gmail dot com>.
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

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
/* Activation Functions */

#include "maths.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanhS tanh
#else
#define PSTanhS tanhf
#endif

#define PSTanhActivation PSVectorTanh

typedef void     (*PSActivationFunction) (PSFloat *vec, PSFloat *dest,
                                          uint64_t len, PSMathOpts *opts);
typedef PSFloat  (*PSScalarActivationFunction) (PSFloat);

PSFloat PSSigmoidS(PSFloat val);
PSFloat PSSigmoidDerivativeS(PSFloat val);
PSFloat PSReluS(PSFloat val);
PSFloat PSReluDerivativeS(PSFloat val);
PSFloat PSTanhDerivativeS(PSFloat val);

void PSSigmoid(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts);
void PSRelu(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts);
void PSSigmoidDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                         PSMathOpts *opts);
void PSTanhDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts);
void PSReluDerivative(PSFloat *vec, PSFloat *dest, uint64_t len,
                      PSMathOpts *opts);
void PSSoftmax(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts);

#endif /* __ACTIVATION_H__ */

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

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
/* Activation Functions */

#include "maths.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanhActivation tanh
#else
#define PSTanhActivation tanhf
#endif

#define PSTanhV PSVectorTanh

PSFloat PSSigmoid(PSFloat val);
PSFloat PSSigmoidDerivative(PSFloat val);
PSFloat PSRelu(PSFloat val);
PSFloat PSReluDerivative(PSFloat val);
PSFloat PSTanhDerivative(PSFloat val);

void PSSigmoidV(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts);
void PSReluV(PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts);
void PSSigmoidDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                          PSMathOpts *opts);
void PSTanhDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                       PSMathOpts *opts);
void PSReluDerivativeV(PSFloat *vec, PSFloat *dest, uint64_t len,
                       PSMathOpts *opts);

#endif /* __ACTIVATION_H__ */

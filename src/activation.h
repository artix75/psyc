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

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
/* Activation Functions */

#include "maths.h"

#ifdef PS_DOUBLE_PRECISION
#define PSTanhS tanh
#else
#define PSTanhS tanhf
#endif

typedef void     (*PSActivationFunction) (PSFloat *vec, PSFloat *dest,
                                          long len, int acceleration);
typedef PSFloat  (*PSScalarActivationFunction) (PSFloat);

PSFloat PSSigmoidS(PSFloat val);
PSFloat PSReluS(PSFloat val);
PSFloat PSGeluS(PSFloat val);
PSFloat PSSigmoidDerivativeS(PSFloat val);
PSFloat PSReluDerivativeS(PSFloat val);
PSFloat PSGeluDerivativeS(PSFloat val);
PSFloat PSTanhDerivativeS(PSFloat val);

void PSSigmoid(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSTanhActivation(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSRelu(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSGelu(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSSigmoidDerivative(PSFloat *vec, PSFloat *dest, long len,
                         int acceleration);
void PSTanhDerivative(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSReluDerivative(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSGeluDerivative(PSFloat *vec, PSFloat *dest, long len, int acceleration);
void PSSoftmax(PSFloat *vec, PSFloat *dest, long len, int acceleration);

#endif /* __ACTIVATION_H__ */

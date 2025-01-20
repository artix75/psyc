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

#include <stdint.h>
#include <float.h>

#ifndef __PS_TYPES_H__
#define __PS_TYPES_H__

#ifndef FLT_DECIMAL_DIG
#define FLT_DECIMAL_DIG (FLT_DIG + 2)
#endif

#ifndef DBL_DECIMAL_DIG
#define DBL_DECIMAL_DIG (DBL_DIG + 2)
#endif

#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_FORMAT "%lg"
#define PSFLOAT_DIG DBL_DECIMAL_DIG
#define PSFLOAT_MIN DBL_MIN
#define PSFLOAT_MAX DBL_MAX
#define PSFLOAT_EPS DBL_EPSILON
typedef double PSFloat;
#else
#define PSFLOAT_FORMAT "%g"
#define PSFLOAT_DIG FLT_DECIMAL_DIG
#define PSFLOAT_MIN FLT_MIN
#define PSFLOAT_MAX FLT_MAX
#define PSFLOAT_EPS FLT_EPSILON
typedef float PSFloat;
#endif

#endif /* __PS_TYPES_H__ */

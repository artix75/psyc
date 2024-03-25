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

#ifndef __PLATFORM_H
#define __PLATFORM_H


#include <stdint.h>
#include <limits.h>
#ifdef __APPLE__
#include <AvailabilityMacros.h>
#if defined(HAS_ACCELERATE_FRAMEWORK)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#if LONG_MAX==LLONG_MAX && !defined(PS_LAPACK_I32)
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64
#endif
#endif
#endif
#endif

#ifdef __linux__
#include <linux/version.h>
#include <features.h>
#endif

#if defined(__APPLE__) || (defined(__linux__) && defined(__GLIBC__)) || \
    defined(__FreeBSD__) || (defined(__OpenBSD__) && defined(USE_BACKTRACE))\
 || defined(__DragonFly__)
#define BACKTRACE_AVAILABLE 1
#endif

#if defined (__unix__) || defined(__linux__) || \
    (defined (__APPLE__) && defined (__MACH__)) || \
    defined(__FreeBSD__) || defined(__OpenBSD__)
    #define IS_UNIX 1
#else
    #define IS_UNIX 0
#endif

/* PS_IEC_559 (IEE 754 conformity)
 *   0: No
 *   1: Yes
 *   2: Maybe */
#ifdef __GCC_IEC_559

#if __GCC_IEC_559 > 0
#define PS_IEC_559 1
#else
#define PS_IEC_559 0
#endif

#else

#ifdef __STDC_IEC_559__
#define PS_IEC_559 1
#elif __DBL_DIG__ == 15 && __DBL_MANT_DIG__ == 53 && __DBL_MAX_10_EXP__ == 308\
    && __DBL_MAX_EXP__ == 1024 && __DBL_MIN_10_EXP__ == -307 && \
    __DBL_MIN_EXP__ == -1021
#define PS_IEC_559 2
#else
#define PS_IEC_559 0
#endif

#endif

/* Endianness */
#define PS_IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

#endif /* __PLATFORM_H  */

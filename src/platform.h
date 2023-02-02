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

#ifndef __PLATFORM_H
#define __PLATFORM_H


#ifdef __APPLE__
#include <AvailabilityMacros.h>
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

#endif /* __PLATFORM_H  */

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

#include "types.h"
#include "config.h"

int PSGlobalFlags = 0;

#ifdef HAS_BLAS

#if !defined(USE_AVX)
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#define PS_UNAVAILABLE_ACCEL PSAcceleration_AVX
#else
#define PS_UNAVAILABLE_ACCEL (PSAcceleration_AVX | PSAcceleration_ACF)
#endif
#elif !defined(__APPLE__) || !defined(HAS_ACCELERATE_FRAMEWORK)
#define PS_UNAVAILABLE_ACCEL PSAcceleration_ACF
#else
#define PS_UNAVAILABLE_ACCEL 0
#endif

#else

#if !defined(USE_AVX)
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
#define PS_UNAVAILABLE_ACCEL PSAcceleration_BLAS | PSAcceleration_AVX
#else
#define PS_UNAVAILABLE_ACCEL \
    (PSAcceleration_AVX | PSAcceleration_ACF | PSAcceleration_BLAS)
#endif
#elif !defined(__APPLE__) || !defined(HAS_ACCELERATE_FRAMEWORK)
#define PS_UNAVAILABLE_ACCEL PSAcceleration_ACF | PSAcceleration_BLAS
#else
#define PS_UNAVAILABLE_ACCEL PSAcceleration_BLAS
#endif

#endif

static uint8_t unavailableAccelerations = PS_UNAVAILABLE_ACCEL;
static uint8_t unavailableAccelerationsMask = ~(PS_UNAVAILABLE_ACCEL);
uint8_t PSGlobalAcceleration = (PSAcceleration_All & ~(PS_UNAVAILABLE_ACCEL));

int PSIsAccelerationAvailable(PSAcceleration acceleration) {
    return !(unavailableAccelerations & acceleration);
}

int PSIsAccelerationEnabled(uint8_t config, PSAcceleration acceleration) {
    return (config & unavailableAccelerationsMask) & acceleration;
}

int PSEnableAcceleration(uint8_t *config, PSAcceleration acceleration) {
    if (!PSIsAccelerationAvailable(acceleration)) return 0;
    return (*config = *config | acceleration);
}

void PSDisableAcceleration(uint8_t *config, PSAcceleration acceleration) {
    *config = (*config & ~((unsigned) acceleration));
}

const char *PSGetAccelerationName(PSAcceleration acceleration) {
    switch (acceleration) {
        case PSAcceleration_None: return "None";
        case PSAcceleration_AVX: return "AVX";
        case PSAcceleration_ACF: return "Accelerate Framework";
        case PSAcceleration_BLAS: return "BLAS";
        case PSAcceleration_All: return "All";
    }
    return "Unknown";
}

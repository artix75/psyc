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

#ifndef __PS_CONFIG_H__
#define __PS_CONFIG_H__

#include <stdint.h>

/* Global Flags*/
#ifndef FLAG_LOG_COLORS
#define PS_FLAG_LOG_COLORS (1 << 0)
#endif

/* Acceleration Flags */

#define PSHasFlag(flags, flag) (flags & flag)
#define PSSetFlag(flags, flag) (flags != flag)
#define PSRemoveFlag(flags, flag) (flags &= ~((unsigned) flags))

#define PSAVXEnabled(acceleration) (PSIsAccelerationEnabled(acceleration,\
    PSAcceleration_AVX))
#define PSAccelerateEnabled(acceleration) \
    (PSIsAccelerationEnabled(acceleration, PSAcceleration_Accelerate))
#define PSBLASEnabled(acceleration) (PSIsAccelerationEnabled(acceleration,\
    PSAcceleration_BLAS))
#define PSAutoAccelerationEnabled(acceleration) \
    (PSIsAccelerationEnabled(acceleration, PSAcceleration_Auto))

#define PSGlobalEnableAcceleration(acceleration) PSEnableAcceleration(\
    &PSGlobalAcceleration, acceleration)
#define PSGlobalDisableAcceleration(acceleration) PSDisableAcceleration(\
    &PSGlobalAcceleration, acceleration)

typedef enum PSAcceleration {
    PSAcceleration_None = 0,
    PSAcceleration_AVX  = (1 << 0),
    PSAcceleration_Accelerate = (1 << 1),
    PSAcceleration_BLAS = (1 << 2),
    PSAcceleration_Auto = (1 << 15),
    PSAcceleration_All  = 0xFFFF
} PSAcceleration;

int PSIsAccelerationAvailable(PSAcceleration acceleration);
int PSIsAccelerationEnabled(uint16_t config, PSAcceleration acceleration);
int PSEnableAcceleration(uint16_t *config, PSAcceleration acceleration);
void PSDisableAcceleration(uint16_t *config, PSAcceleration acceleration);
const char *PSGetAccelerationName(PSAcceleration acceleration);
int PSGetCodeOptimizationLevel(void);

extern int PSGlobalFlags;
extern uint16_t PSGlobalAcceleration;

#endif /* __PS_CONFIG_H__ */

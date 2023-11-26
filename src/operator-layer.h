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

#ifndef __PS_OPERATOR_H
#define __PS_OPERATOR_H

#include "types.h"
#include "psyc.h"

#define PS_MAX_PROVIDERS 1024

typedef enum {
    PSConcatenateOperator,
    PSAddOperator,
    PSMultiplyOperator,
    PSInvalidOperator = 999
} PSOperatorType;

PSOperatorType PSGetOperatorLayerType(PSLayer *layer);
const char *PSGetOperatorLayerTypeLabel(PSOperatorType operator);
PSLayer **PSGetOperatorLayerProviders(PSLayer *layer, int *count);

#endif /* __PS_OPERATOR_H */


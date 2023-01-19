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

#ifndef __PS_CONVOLUTIONAL_H
#define __PS_CONVOLUTIONAL_H

#include "psyc.h"
#include "utils.h"

#define PARAM_FEATURE_COUNT     0
#define PARAM_REGION_SIZE       1
#define PARAM_STRIDE            2
#define PARAM_INPUT_WIDTH       3
#define PARAM_INPUT_HEIGHT      4
#define PARAM_OUTPUT_WIDTH      5
#define PARAM_OUTPUT_HEIGHT     6
#define PARAM_PADDING           7
#define PARAM_USE_RELU          8

#define CONV_PARAMETER_COUNT 9

#define PSGetColumn(index, width) (index % width)
#define PSGetRow(index, width) ((int) ((int) index / (int) width))

#endif /* __PS_CONVOLUTIONAL_H */

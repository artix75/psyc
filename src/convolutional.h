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

#ifndef __PS_CONVOLUTIONAL_H
#define __PS_CONVOLUTIONAL_H

#include "psyc.h"
#include "utils.h"

/* Automatically determine padding in order to make the output size the same
 * as the input size (padding = filter_width / 2). Stride must be 1,
 * filter_height must be 1 or the same of filter_width and filter_width must be
 * odd. */
#define PS_PADDING_SAME -1
/* Automatically determine padding so that the output size is always bigger
 * than the input size (padding = filter_width - 1). Stride must be 1 and
 * filter_height must be 1 or the same of filer_width. */
#define PS_PADDING_FULL -2

#define PSGetConvolutionalSettings(layer) \
    ((PSConvolutionalSettings *) layer->extra)
#define PSGetColumn(index, width) (index % width)
#define PSGetRow(index, width) ((int) ((int) index / (int) width))

typedef struct PSConvolutionalSettings {
    int stride;
    int padding;
    long filter_width;
    long filter_height;
    long filter_depth;
    long input_width;
    long input_height;
    long input_depth;
} PSConvolutionalSettings;

#endif /* __PS_CONVOLUTIONAL_H */

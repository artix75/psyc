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

#ifndef __PS_CIFAR_H
#define __PS_CIFAR_H

#include "types.h"

#ifndef DATA_TYPE_TRAINING
#define DATA_TYPE_TRAINING   0
#define DATA_TYPE_TEST       1
#endif

#define CIFAR_IMAGE_SIZE (32 * 32 * 3)

int loadCIFARData(int type, int classes, const char *dataset_path,
                  PSFloat **data, int max_files, int max_elements);

#endif /*  __PS_CIFAR_H */

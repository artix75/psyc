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

#ifndef __PS_EMBEDDING_H
#define __PS_EMBEDDING_H

#include "psyc.h"

typedef enum PSEmbeddingType {
    Word2Vec
} PSEmbeddingType;

PSFloat *PSCreateWord2VecTrainingData(PSFloat *tokens, size_t token_count,
                                      int window_size, int vocabulary_size,
                                      int onehot, int *num_elements_ptr);

#endif /*  __PS_EMBEDDING_H */

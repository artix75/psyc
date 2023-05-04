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

#ifndef __PS_ATTENTION_H
#define __PS_ATTENTION_H

#include "psyc.h"

#define PS_TRAINABLE_QUERY      (1 << 0)
#define PS_TRAINABLE_KEYS       (1 << 1)
#define PS_TRAINABLE_VALUES     (1 << 2)
#define PS_TRAINABLE_PROJECTION (1 << 3)
#define PS_TRAINABLE_SCORES     (1 << 4)

#define PS_QUERY_IDX        0
#define PS_KEYS_IDX         1
#define PS_VALUES_IDX       2
#define PS_PROJECTION_IDX   3
#define PS_SCORES_IDX       4

#define PSIsMultiHeadAttention(layer) (PSGetAttentionHeadCount(layer) > 1)

typedef enum {
    PSDotAttention,
    PSAdditiveAttention,
    PSInvalidAttention = 9999
} PSAttentionType;

const char *PSGetAttentionTypeLabel(PSAttentionType type);
PSAttentionType PSGetAttentionType(PSLayer *layer);
PSFloat PSGetAttentionScale(PSLayer *layer);
int PSGetAttentionHeadCount(PSLayer *layer);
int PSGetAttentionProviders(PSLayer *layer, PSLayer **query_provider,
                            PSLayer **keys_provider, PSLayer **values_provider);
int PSGetAttnetionTrainableParameters(PSLayer *layer);
int PSIsCausalAttention(PSLayer *layer);
int PSSetAttentionQueryProvider(PSLayer *layer, PSLayer *provider);

#endif /* __PS_ATTENTION_H */

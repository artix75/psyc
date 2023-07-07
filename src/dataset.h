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

#ifndef __PS_DATASET_H
#define __PS_DATASET_H

#include "types.h"
#include "psyc.h"
#include "utils.h"

#ifndef DATA_TYPE_TRAINING
#define DATA_TYPE_TRAINING   0
#define DATA_TYPE_TEST       1
#endif

/* Text processing */
#define PS_DEFAULT_TOKEN_SEPARATOR  " ,.:!?-'\"\n\t\r"

#define PS_INVALID_TOKEN_ID -1
#define PS_TOKEN_NOT_FOUND  -2

#define PS_PARSER_MODE_TOKENS  0
#define PS_PARSER_MODE_CHARS   1

#define PS_PARSER_FLAG_NO_NORMALIZATION (1 << 0)
#define PS_PARSER_FLAG_PRESERVE_STRING  (1 << 1)
#define PS_PARSER_FLAG_READONLY_VOCAB   (1 << 2)

typedef char *(*PSTokenNormalizer) (char *token, int len);
typedef int   (*PSTokenMatch) (char *str, int *len);

typedef struct {
    int mode;
    int flags;
    int64_t max_vocabulary_size; /* Except <unknown> token */
    int64_t sequence_length;
    const char *separator;
    /*const char *keep;*/
    const char *unkown_token;
    int capacity;
    int buffer_size;
    PSTokenNormalizer normalizer;
    PSTokenMatch match_token;
} PSTextParserOptions;

typedef struct {
    int64_t         size;
    int64_t         capacity;
    PSDict          *token_map;
    const char      **tokens;
} PSVocabulary;

PSVocabulary *PSVocabularyCreate(int64_t initial_capacity);
int64_t PSVocabularyAdd(PSVocabulary *vocabulary, char *token);
int64_t PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token);
const char *PSVocabularyGetTokenByID(PSVocabulary *vocabulary, int64_t id);
const char *PSVocabularyErrorString(int err);
void PSVocabularyRelease(PSVocabulary *vocabulary);

char *PSNormalizeToken(char *token, int len);
PSFloat *PSLoadDataFromString(char *str, PSTextParserOptions *opts,
                              PSFloat *existing_data, int64_t *datalen,
                              PSVocabulary **vocabulary);
PSFloat *PSLoadDataFromTextFile(const char *filepath,
                                PSTextParserOptions *opts,
                                int64_t *datalen,
                                PSVocabulary **vocabulary);

/* MNIST Dataset */
#define MNIST_INPUT_SIZE (28 * 28)

int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data);

/* CIFAR Dataset */
#define CIFAR_IMAGE_SIZE (32 * 32 * 3)

int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_elements);
PSLayer *PSAddCIFARInputLayer(PSNeuralNetwork *network);

#endif /* __PS_DATASET_H */

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

#ifndef PS_DATA_TYPE_TRAINING
#define PS_DATA_TYPE_TRAINING   0
#define PS_DATA_TYPE_TEST       1
#endif

/*** Text processing ***/

#define PS_DEFAULT_TOKEN_SEPARATOR  " ,.:!?-'\"\n\t\r"

#define PS_INVALID_TOKEN_ID -1
#define PS_TOKEN_NOT_FOUND  -2

/* Parse text as separate tokens*/
#define PS_PARSER_MODE_TOKENS  0
/* Parse text as individual characters. */
#define PS_PARSER_MODE_CHARS   1

#define PS_PARSER_FLAG_NO_NORMALIZATION (1 << 0)
#define PS_PARSER_FLAG_PRESERVE_STRING  (1 << 1)
#define PS_PARSER_FLAG_READONLY_VOCAB   (1 << 2)

#define PS_DEFAULT_PARSER_CAPACITY  50
#define PS_DEFAULT_MAX_VOCAB_SIZE   15000
#define PS_DEFAULT_UNKNOWN_TOKEN    "<unknown>"

/* Pointer to a function that can be used to normalize tokens. It takes the
 * input `token` of length `len` and returns the normalized token.
 * Return value: the normalized token. */
typedef char *(*PSTokenNormalizer) (char *token, int len);

/* Pointer to a function that can be used by parsing functions (ie.
 * `PSLoadDataFromString`) to match the token `token`.
 * The length of the matched token must be stored into address pointed by
 * `len`.
 * Return value: 1 if token has been matched, 0 if no token has been matched.
 */
typedef int   (*PSTokenMatch) (char *str, int *len);

/* Options for text parsing:
 *  - `mode`: text parsing mode:
 *            - `PS_PARSER_MODE_TOKENS`: parse text as tokens (each token
 *              will be converted to a number).
 *            - `PS_PARSER_MODE_CHARS`: parse text as characters (each
 *              individual character will be converted to a number).
 *  - `flags`: text parsing flags:
 *             - `PS_PARSER_FLAG_NO_NORMALIZATION`: do not perform token
 *               normalization on parsed text.
 *             - `PS_PARSER_FLAG_PRESERVE_STRING`: prevent string from being
 *               modified during parsing.
 *             - `PS_PARSER_FLAG_READONLY_VOCAB`: by enabling this flag, the
 *               vocabulary will be treated as read-only. Any parsed token
 *               that is not present in the vocabulary will not be added and
 *               will be considered <unknown> (see the `unknown_token` option).
 *  - `max_vocabulary_size`: maximum number of tokens that can be added to
 *                           the vocabulary, except for the <unknown> token.
 *                           Every new parsed token will be automatically
 *                           converted to the <unknown> token (see the
 *                           `unknown_token` option).
 *                           If the value of this option is zero, the
 *                           default value will be `PS_DEFAULT_MAX_VOCAB_SIZE`.
 *  - `separator`: a set of characters that should be used as separators to
 *                 split string into individual tokens (ie: ".," would split
 *                 by using both '.' and ',' as separators).
 *  - `unknown_token`: string to be used for unmatched tokens.
 *  - `capacity`: initial capacity of vocabularies allocated by parsing
 *                functions (ie. `PSLoadDataFromString`).
 *  - `buffer_size`: parsing buffer size.
 *  - `normalizer`: pointer to function to be used to normalize tokens
 *                  (see `PSTokenNormalizer`)
 *  - `match_token`: pointer to function to be used to match individual tokens
 *                   (it usually overrides the usage of `separator` to split
 *                   string).
*/
typedef struct {
    int mode;
    int flags;
    int64_t max_vocabulary_size; /* Except <unknown> token */
    const char *separator;
    const char *unknown_token;
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

/* Text datasets */
PSVocabulary *PSVocabularyCreate(int64_t initial_capacity);
int64_t PSVocabularyAdd(PSVocabulary *vocabulary, char *token);
int64_t PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token);
const char *PSVocabularyGetTokenByID(PSVocabulary *vocabulary, int64_t id);
const char *PSVocabularyErrorString(int err);
void PSVocabularyFree(PSVocabulary *vocabulary);

char *PSNormalizeToken(char *token, int len);
PSFloat *PSLoadDataFromString(char *str, PSTextParserOptions *opts,
                              PSFloat *existing_data, int64_t *datalen,
                              PSVocabulary **vocabulary);
PSFloat *PSLoadDataFromTextFile(const char *filepath,
                                PSTextParserOptions *opts,
                                int64_t *datalen,
                                PSVocabulary **vocabulary);

/* Generic datasets */


PSFloat *PSLoadDataFromFile(const char *filepath, uint64_t *datalen);
int PSSaveDataToFile(const char *filepath, PSFloat *data, uint64_t datalen);

/* MNIST Dataset */
#define PS_MNIST_INPUT_SIZE (28 * 28)

int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data);

/* CIFAR Dataset */
#define PS_CIFAR_IMAGE_SIZE (32 * 32 * 3)

int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_elements);
PSLayer *PSAddCIFARInputLayer(PSModel *model);

#endif /* __PS_DATASET_H */

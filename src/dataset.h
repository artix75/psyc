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

#ifndef __PS_DATASET_H
#define __PS_DATASET_H

#include "types.h"
#include "psyc.h"
#include "utils.h"

#ifndef PS_DATA_TYPE_TRAINING
#define PS_DATA_TYPE_TRAINING   0
#define PS_DATA_TYPE_TEST       1
#endif

#define PS_DATA_SHUFFLE         (1 << 0)
#define PS_DATA_EVENLY_SPREAD   (1 << 1)
#define PS_DATA_SEQUENCES       PS_FLAG_USE_SEQUENCES
#define PS_DATA_SEQ2SEQ         PS_TRAINING_FLAG_SEQ2SEQ
#define PS_DATA_ALWAYS_ALLOC    (1 << 16)

/*** Text processing ***/

#define PS_DEFAULT_TOKEN_SEPARATOR  " ,.:!?-'\"\n\t\r"

#define PS_INVALID_TOKEN_ID -1
#define PS_TOKEN_NOT_FOUND  -2

#define PS_IO_MAX_TOKEN_SIZE 1024

/* Parse text as separate tokens*/
#define PS_PARSER_MODE_TOKENS  0
/* Parse text as individual characters. */
#define PS_PARSER_MODE_CHARS   1

/* Disable token normalization (keep original token string). */
#define PS_PARSER_FLAG_NO_NORMALIZATION (1 << 0)
/* Preserve parsed string from being modified by parsing a duplicated string.*/
#define PS_PARSER_FLAG_PRESERVE_STRING  (1 << 1)
/* Prevent adding new tokens to vocabulary used for generating a dataset from
 * a parsed string. */
#define PS_PARSER_FLAG_READONLY_VOCAB   (1 << 2)
/* Just generate a dataset that only consist of parsed tokens, with no
 * metadata, no sequence splitting and not targets. */
#define PS_PARSER_FLAG_ENCODE_ONLY      (1 << 3)
/* Add a 'starting' token to the dataset:
 *  - If the string is being parsed as collection of fixed length sequences,
 *    the starting token will be prepended to the first sequence only.
 *  - If the string is being split into variable-length sequences (ie. by
 *    using a sequence separator), but the target sequence has the same length
 *    of the input sequence, the starting token will be prepended to
 *    every input sequence.
 *  - If the dataset has target sequences whose length can differ from the
 *    related input sequences (ie. targets come from another dataset),
 *    the starting token is prepended to every target sequence and, unless
 *    the `PS_PARSER_FLAG_EXACT_INPUTS` is set, to every input sequence.
 *  - If no string is being provided as the starting token with `start_token`
 *    member of `PSTextParserOptions`, by default `PS_DEFAULT_START_TOKEN` is
 *    used. */
#define PS_PARSER_FLAG_START_TOKEN      (1 << 4)
/* Add an 'ending' token to the dataset:
 *  - If the string is being parsed as collection of fixed length sequences,
 *    the ending token will be appended to the last sequence only.
 *  - If the string is being split into variable-length sequences (ie. by
 *    using a sequence separator), but the target sequence has the same length
 *    of the input sequence, the ending token will be appended to
 *    every target sequence.
 *  - If the dataset has target sequences whose length can differ from the
 *    related input sequences (ie. targets come from another dataset),
 *    the ending token is appended to every target sequence and, unless
 *    the `PS_PARSER_FLAG_EXACT_INPUTS` is set, to every input sequence.
 *  - If no string is being provided as the starting token with `end_token`
 *    member of `PSTextParserOptions`, by default `PS_DEFAULT_END_TOKEN` is
 *    used. */
#define PS_PARSER_FLAG_END_TOKEN        (1 << 5)
/* Let text parsing functions (ie. PSDataFromText) also generate the
 * target sequence for every input sequence. */
#define PS_PARSER_FLAG_MAKE_TARGETS     (1 << 6)
/* When the dataset has target sequences whose length can differ from the
*  related input sequences (ie. targets come from another dataset), this
*  flag prevents start/end tokens (see `PS_PARSER_FLAG_START_TOKEN` and
*  `PS_PARSER_FLAG_START_TOKEN` to be added to the input sequences. */
#define PS_PARSER_FLAG_EXACT_INPUTS     (1 << 7)

#define PS_DEFAULT_PARSER_CAPACITY  50
#define PS_DEFAULT_MAX_VOCAB_SIZE   15000
#define PS_DEFAULT_UNKNOWN_TOKEN    "<unknown>"
#define PS_DEFAULT_START_TOKEN      "<start>"
#define PS_DEFAULT_END_TOKEN        "<end>"

/* Pointer to a function that can be used to normalize tokens. It takes the
 * input `token` of length `len` and returns the normalized token.
 * Return value: the normalized token. */
typedef char *(*PSTokenNormalizer) (char *token, size_t len);

/* Pointer to a function that can be used by parsing functions (ie.
 * `PSDataFromText`) to match the token `token`.
 * The length of the matched token must be stored into address pointed by
 * `len`.
 * Return value: 1 if token has been matched, 0 if no token has been matched.
 */
typedef int   (*PSTokenMatch) (char *str, size_t *len);

struct PSVocabulary;

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
 *             - See also: `PS_PARSER_FLAG_ENCODE_ONLY`,
 *               `PS_PARSER_FLAG_MAKE_TARGETS`, `PS_PARSER_FLAG_ENCODE_ONLY`,
 *               `PS_PARSER_FLAG_START_TOKEN`, `PS_DEFAULT_END_TOKEN`,
 *               `PS_PARSER_FLAG_EXACT_INPUTS`.
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
 *                functions (ie. `PSDataFromText`).
 *  - `buffer_size`: parsing buffer size.
 *  - `normalizer`: pointer to function to be used to normalize tokens
 *                  (see `PSTokenNormalizer`)
 *  - `match_token`: pointer to function to be used to match individual tokens
 *                   (it usually overrides the usage of `separator` to split
 *                   string).
 *  - `sequence_length`: split text into multiple fixed-length sequences.
 *  - `match_sequence_end`: pointer to function to be used to match the ending
 *                          token of the sequence. If the callback returns 1,
 *                          the current token will be the ending token of the
 *                          current sequence. It produces variable length
 *                          sequences.
 *  - `sequence_separator`: string that can be used to split text into multiple
 *                          sequences. If the current token is equal to
 *                          `sequence_separator`, it will be the ending token
 *                          of the current sequence. It produces variable
 *                          length sequences.
 *  - `max_sequences`: max number of sequences to be parsed.
 *  - `start_token`: a string to be used as the sequence starting token (by
 *                   default, when needed, `PS_DEFAULT_START_TOKEN` is used).
 *  - `end_token`: a string to be used as the sequence endining token (by
 *                 default, when needed, `PS_DEFAULT_END_TOKEN` is used).
 *  - `target_dataset`: an already existing dataset tha can be used to produce
 *                      the target sequences. For each input sequence, a
 *                      sequence from `target_dataset` will be taken and used
 *                      as target sequence. This can be useful to build
 *                      datasets for sequence-to-sequence models, such as
 *                      natural language translation models (neural machine
 *                      translation).
 *                      The `target_dataset` must have at least the same number
 *                      of sequences of the dataset being generated.
 *  - `target_datalen`: the length (number of `PSFloat` elements) of the
 *                      `target_dataset`, if any.
 *  - `target_vocabulary`: the vocabulary associated to the `target_dataset`,
 *                         if any. If NULL, the same vocabulary used for the
 *                         dataset being generated will be used.
 *                         Example: neural machine translation use different
 *                         vocabularies for different natural languages.
*/
typedef struct {
    int mode;
    int flags;
    long max_vocabulary_size; /* Except <unknown> token */
    const char *separator;
    const char *unknown_token;
    int capacity;
    int buffer_size;
    PSTokenNormalizer normalizer;
    PSTokenMatch match_token;
    int sequence_length;
    PSTokenMatch match_sequence_end;
    const char *sequence_separator;
    long max_sequences;
    const char *start_token;
    const char *end_token;
    PSFloat *target_dataset;
    long target_datalen;
    struct PSVocabulary *target_vocabulary;
} PSTextParserOptions;

typedef struct PSVocabulary {
    long            size;
    long            capacity;
    PSDict          *token_map;
    const char      **tokens;
} PSVocabulary;

/* Text datasets */
PSVocabulary *PSVocabularyCreate(long initial_capacity);
long PSVocabularyAdd(PSVocabulary *vocabulary, char *token);
long PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token);
const char *PSVocabularyGetTokenByID(PSVocabulary *vocabulary, long id);
PSVocabulary *PSVocabularyLoad(const char *path);
int PSVocabularySave(PSVocabulary *vocabulary, const char *path);
const char *PSVocabularyErrorString(long err);
void PSVocabularyFree(PSVocabulary *vocabulary);

char *PSNormalizeToken(char *token, size_t len);
PSFloat *PSDataFromText(char *str, PSTextParserOptions *opts, long *datalen,
                        PSVocabulary **vocabulary);
PSFloat *PSDataFromTextFile(const char *filepath, PSTextParserOptions *opts,
                            long *datalen, PSVocabulary **vocabulary);

/* Generic datasets */


PSFloat *PSDataLoad(const char *filepath, long *datalen);
int PSDataSave(const char *path, PSFloat *data, long len, int opts);
int PSDataSplit(PSFloat *data, long datalen, float percentage,
                long input_size, long target_size,
                PSFloat **left, PSFloat **right,
                long *left_length, long *right_length,
                int opts);

/* MNIST Dataset */
#define PS_MNIST_INPUT_SIZE (28 * 28)

int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data);

/* CIFAR Dataset */
#define PS_CIFAR_IMAGE_SIZE (32 * 32 * 3)

int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_examples);
PSLayer *PSAddCIFARInputLayer(PSModel *model);

#endif /* __PS_DATASET_H */

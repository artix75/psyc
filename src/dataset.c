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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>
#include <limits.h>
#include <zlib.h>
#include <errno.h>

#include "dataset.h"
#include "psyc.h"
#include "utils.h"
#include "log.h"
#include "platform.h"

#define PS_PARSER_BUFFER_SIZE 4096
#define PS_IS_MULTISEQ(opts) (opts->sequence_length > 0 || \
    opts->sequence_separator != NULL || opts->match_sequence_end != NULL)

typedef struct {
    PSFloat *nseq_p;
    PSFloat *xlen_p;
    PSFloat *ylen_p;
    PSFloat *x;
    PSFloat *y;
    int incomplete_target;
    long capacity;
    size_t target_dataset_offset;
} PSDataSequenceState;

/* Forward declarations */
uint32_t swap_uint32(uint32_t val);
PSFloat *readSerializedFloatArray(FILE *in, char *sep, size_t *length,
                                  size_t maxlen, size_t capacity);
size_t writeSerializedFloatArray(FILE *out, size_t count, char *sep,
                                 int opts, PSFloat *array);
PSFloat *loadBinaryVector(const char *filepath, FILE *f, size_t *len);
int saveBinaryVector(FILE *f, PSFloat *vec, size_t len);

/**** PSVocabulary ****/

/* Create a `PSVocabulary` with initial capacity of `initial_capacity`.
 * Return value: the vocabulary or NULL if memory cannot be allocated. */
PSVocabulary *PSVocabularyCreate(long initial_capacity) {
    PSVocabulary *vocabulary = malloc(sizeof(*vocabulary));
    if (vocabulary == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    vocabulary->size = 0;
    vocabulary->capacity = 0;
    vocabulary->token_map = PSDictCreate(PSDICT_UPDATE_DISABLED);
    if (vocabulary->token_map == NULL) {
        free(vocabulary);
        return NULL;
    }
    vocabulary->tokens = NULL;
    if (initial_capacity > 0) {
        vocabulary->tokens = calloc(initial_capacity, sizeof(char *));
        if (vocabulary->tokens != NULL)
            vocabulary->capacity = initial_capacity;
    }
    return vocabulary;
}

/* Add token `token` to `vocabulary`. The token is added to the internal
 * dictionary of `vocabulary` and a numeric index (ID) is assigned to it.
 * The index is a progressive number. If the token already exists, the
 * internal dictionary won't be updated and the token's numeric value (id)
 * is immediately returned.
 * Return value: the numeric index (ID) of the token. If token could not be
 * added to the dictionary or `token` is NULL, the function will return
 * `PS_INVALID_TOKEN_ID`. */
long PSVocabularyAdd(PSVocabulary *vocabulary, char *token) {
    if (token == NULL) return PS_INVALID_TOKEN_ID;
    long id = vocabulary->token_map->length, size;
    PSDict *token_map = vocabulary->token_map;
    /* Set id as value for token into token_map dictionary. Since token_map
     * has flag PSDICT_UPDATE_DISABLED enabled, value won't be updated if it
     * already exists for token, so, in this case, existing value will be
     * returned. */
    PSDictItem *item = PSDictSet(token_map, token, PSDictItemFromInt(id));
    if (item == NULL) return PS_INVALID_TOKEN_ID;
    id = item->value.as_int;
    size = token_map->length;
    if (vocabulary->tokens == NULL || size > vocabulary->capacity) {
        char **tokens = realloc(vocabulary->tokens, size * sizeof(char *));
        if (tokens == NULL) {
            PSPrintMemoryErrorMsg();
            PSDictRemove(token_map, token);
            return PS_INVALID_TOKEN_ID;
        }
        long added_size = size - vocabulary->capacity;
        char **added = tokens + vocabulary->capacity;
        memset(added, 0, added_size * sizeof(char *));
        vocabulary->tokens = (const char**) tokens;
        vocabulary->capacity = size;
    }
    vocabulary->size = size;
    vocabulary->tokens[id] = item->key;
    return id;
}

/* Get the ID of the token `token` from vocabulary `vocabulary`.
 * Return value: the numeric index (ID) of the token. If token is not found
 * into `vocabulary`, the function will return `PS_TOKEN_NOT_FOUND`.
 * If `vocabulary` is NULL or `token` is NULL, the function will return
 * `PS_INVALID_TOKEN_ID`. */
long PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token) {
    if (vocabulary == NULL || token == NULL || vocabulary->token_map == NULL)
        return PS_INVALID_TOKEN_ID;
    PSDictItem *item = PSDictGet(vocabulary->token_map, token);
    if (item == NULL) return PS_TOKEN_NOT_FOUND;
    return item->value.as_int;
}

/* Get the token associated with `id` from `vocabulary`.
 * Return value: the token associated with `id` or NULL if no `token` is found
 * with `id`. Also return NULL if `vocabulary` is NULL. */
const char *PSVocabularyGetTokenByID(PSVocabulary *vocabulary, long id) {
    if (vocabulary == NULL) return NULL;
    if (id >= vocabulary->size || vocabulary->tokens == NULL) return NULL;
    return vocabulary->tokens[id];
}

/* Save `vocabulary` to file located at `path`. Vocabulary tokens are written
 * sequentially to the file as an ordered sequence of NULL-terminated strings.
 * Return values: 1 if the vocabulary has been successfully saved, 0 if:
 *  - `vocabulary` is NULL or `path` is NULL.
 *  - `path` cannot be opened for writing.
 *  - Size of some token exceeds max. size (`PS_IO_MAX_TOKEN_SIZE`).
 *  - Some writing error occurs. */
int PSVocabularySave(PSVocabulary *vocabulary, const char *path) {
    if (vocabulary == NULL) {
        PSErr(__func__, "mandatory argument `vocabulary` is null");
        return 0;
    }
    if (path == NULL) {
        PSErr(__func__, "mandatory argument `path` is null");
        return 0;
    }
    if (vocabulary->size == 0) {
        PSErr(__func__, "empty vocabulary");
        return 0;
    }
    int success = 1;
    long i;
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        PSErr(__func__, "could not open '%s' for writing", path);
        return 0;
    }
    errno = 0;
    int nwritten = fprintf(f, "%ld:", vocabulary->size);
    success = (nwritten >= 0);
    if (!success) {
        PSErr(__func__, "failed to write to file '%s'", path);
        goto final;
    }
    for (i = 0; i < vocabulary->size; i++) {
        const char *token = vocabulary->tokens[i];
        success = (token != NULL);
        if (!success) {
            PSErr(__func__, "token[%ld] is null", i);
            goto final;
        }
        nwritten = fprintf(f, "%s%c", token, '\0');
        success = (nwritten >= 0);
        if (!success) {
            PSErr(__func__, "failed to write token[%" PRIi64 "] to `%s`", i,
                  path);
            goto final;
        }
        success = (nwritten <= PS_IO_MAX_TOKEN_SIZE);
        if (!success) {
            PSErr(__func__, "token[%" PRIi64 "] exceeds max. size %d in "
                  "'%s'", i, PS_IO_MAX_TOKEN_SIZE, path);
            goto final;
        }
    }
final:
    if (!success && errno > 0) PSErr(__func__, "%s", strerror(errno));
    fclose(f);
    return success;
}

/* Load vocabulary from file located at `path`.
 * Return value: the pointer to vocabulary or NULL if:
 *  - `path` is NULL.
 *  - `path` does not exists.
 *  - `path` cannot be opened for reading.
 *  - File at `path` is not a valid PsyC vocabulary file.
 *  - Vocabulary would have zero tokens.
 *  - Memory cannot be allocated.
 *  - Size of some token exceeds max. size (`PS_IO_MAX_TOKEN_SIZE`).
 *  - Some token cannot be added to vocabulary. */
PSVocabulary *PSVocabularyLoad(const char *path) {
    if (path == NULL) {
        PSErr(__func__, "argument `path` cannot be null");
        return NULL;
    }
    if (!PSFileExists(path)) {
        PSErr(__func__, "file not found: '%s'", path);
        return NULL;
    }
    PSVocabulary *vocabulary = NULL;
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        PSErr(__func__, "could not open '%s' for reading", path);
        return NULL;
    }
    int64_t size = 0;
    char sep[2] = {0};
    int matched = fscanf(f, "%" SCNi64 "%1[:]", &size, sep);
    if (matched < 2 || sep[0] != ':') {
        PSErr(__func__, "invalid file '%s'", path);
        goto final;
    }
    if ((uint64_t) size > LONG_MAX) {
        PSErr(__func__, "vocabulary size exceeds limits");
        goto final;
    }
    if (size <= 0) {
        PSErr(__func__, "empty vocabulary at path '%s'", path);
        goto final;
    }
    vocabulary = PSVocabularyCreate((long) size);
    if (vocabulary == NULL) {
        PSErr(__func__, "cannot create vocabulary");
        goto final;
    }
    char token[PS_IO_MAX_TOKEN_SIZE] = {0};
    char c = 0;
    char *p = token;
    int toklen = 0;
    while ((c = getc(f)) != EOF) {
        if (++toklen > PS_IO_MAX_TOKEN_SIZE) {
            PSErr(__func__, "token[%ld] exceeds max. size %d in "
                  "'%s'", vocabulary->size, PS_IO_MAX_TOKEN_SIZE, path);
            PSVocabularyFree(vocabulary);
            vocabulary = NULL;
            goto final;
        }
        *(p++) = c;
        if (c == '\0') {
            long id = PSVocabularyAdd(vocabulary, token);
            if (id < 0) {
                PSErr(__func__, "could not add token from '%s'", path);
                PSVocabularyFree(vocabulary);
                vocabulary = NULL;
                goto final;
            }
            p = token;
            *p = '\0';
            toklen = 0;
        }
    }
final:
    fclose(f);
    return vocabulary;
}

const char *PSVocabularyErrorString(long err) {
    switch (err) {
        case PS_INVALID_TOKEN_ID: return "generic error";
        case PS_TOKEN_NOT_FOUND: return "token not found";
    }
    return "";
}

void PSVocabularyFree(PSVocabulary *vocabulary) {
    if (vocabulary == NULL) return;
    free(vocabulary->tokens);
    PSDictFree(vocabulary->token_map);
    free(vocabulary);
}

/**** Text Datasets ****/

static PSFloat *makeRoomForNewData(PSFloat *data, long *current_capacity,
                                   long new_length, long capacity_incr,
                                   int num_pointers, ...)
{
    if (new_length > *current_capacity) {
        long min_capacity = (new_length - *current_capacity);
        if (capacity_incr < min_capacity) capacity_incr = min_capacity;
        long new_capacity = *current_capacity + capacity_incr;
        PSFloat *new_data = realloc(data, new_capacity * sizeof(PSFloat));
        if (new_data == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
        if (data != new_data) {
            va_list args;
            va_start(args, num_pointers);
            while (num_pointers-- > 0) {
                PSFloat **p = va_arg(args, PSFloat **);
                if (p == NULL) break;
                PSFloat *ptr = *p;
                if (ptr == NULL) continue;
                size_t offs = ptr - data;
                *p = new_data + offs;
            }
            va_end(args);
        }
        data = new_data;
        *current_capacity = new_capacity;
    }
    return data;
}

static PSFloat *handleTruncatedTextSequenceData(PSFloat *data,
                                                long *datalen,
                                                long *current_capacity,
                                                long end_token_id,
                                                PSTextParserOptions *opts,
                                                PSDataSequenceState *state)
{
    assert(data != NULL);
    assert(datalen != NULL);
    assert(opts != NULL);
    assert(state != NULL);
    assert(current_capacity != NULL);
    int do_drop = 0, make_targets = (
        opts->flags & PS_PARSER_FLAG_MAKE_TARGETS ||
        (opts->target_dataset != NULL && opts->target_datalen > 0)
    );
    long count = *datalen, cur_capacity = *current_capacity;
    int xlen = 0;
    if (state->xlen_p) xlen = (int)(*state->xlen_p);
    if (state->incomplete_target) {
        /* Target sequence (y) is not complete. */
        if (end_token_id > 0) {
            /* Try to fill target sequence with the provided end token. */
            long newlen = ++count;
            PSFloat *new_data = makeRoomForNewData(
                data, &cur_capacity, newlen, newlen, 3,
                &(state->nseq_p), &(state->x), &(state->xlen_p)
            );
            if (new_data == NULL) return NULL;
            data = new_data;
            *current_capacity = cur_capacity;
            data[count - 1] = (PSFloat) end_token_id;
        } else do_drop = 1;
    } else if (opts->sequence_length > 0) {
        /* Still reading input sequence (x). */
        if (state->xlen_p != NULL && xlen < opts->sequence_length)
            do_drop = 1;
    } else {
        /*if (make_targets || xlen_p == NULL) {
            do_drop = 1;
        } else if (xlen_p != NULL) *xlen_p = *xlen_p + 1; */
        if (state->xlen_p != NULL) {
            long slen = (data + count - 1) - state->xlen_p;
            assert(slen >= 0); /* TODO: handle? */
            do_drop = (
                slen == 0 || xlen == 0 || (make_targets && slen == xlen)
            );
            if (!do_drop && !make_targets && xlen < slen)
                *(state->xlen_p) = slen;
        } else do_drop = 1;
    }
    long nseq = *(state->nseq_p);
    if (do_drop) {
        /* If it was reading target sequence, decrease sequence count. */
        if (state->incomplete_target && nseq > 0) *(state->nseq_p) = nseq - 1;
        if (state->xlen_p != NULL) count = state->xlen_p - data;
    } else if (!make_targets && count > 0) {
        *(state->nseq_p) = nseq + 1;
    }
    if (opts->flags & PS_PARSER_FLAG_END_TOKEN && end_token_id >= 0)
        if (data[count - 1] != end_token_id) data[count - 1] = end_token_id;
    *datalen = count;
    return data;
}

PSFloat *loadDatasetFromString(char *str, PSTextParserOptions *opts,
                               long *datalen, PSVocabulary **vocabulary,
                               PSFloat *existing_data,
                               PSDataSequenceState *state, const char *func)
{
    static PSTextParserOptions default_opts = {0};
    assert(state != NULL);
    char *tmpstr = NULL;
    const char *seq_separator = NULL, *start_token = NULL, *end_token = NULL;
    int new_vocab = 0, use_y_dataset = 0,
        multi_seq = 0, incomplete_target = 0, i;
    PSVocabulary *vocab = NULL;
    PSFloat *data = NULL;
    if (datalen == NULL) {
        PSErr(func, "Missing mandatory argument `datalen`");
        goto fail;
    }
    if (vocabulary == NULL) {
        PSErr(func, "Missing mandatory argument `vocabulary`");
        goto fail;
    }
    if (str == NULL) {
        if (existing_data == NULL) *datalen = 0;
        return NULL;
    }
    size_t len = strlen(str);
    if (len == 0) {
        if (existing_data == NULL) *datalen = 0;
        return NULL;
    }
    size_t end_idx = len - 1;
    long y_nseq = 0;
    PSTokenNormalizer normalize = NULL;
    if (opts == NULL) opts = &default_opts;
    int do_normalize = !(opts->flags & PS_PARSER_FLAG_NO_NORMALIZATION),
        read_only = (opts->flags & PS_PARSER_FLAG_READONLY_VOCAB),
        char_mode = (opts->mode == PS_PARSER_MODE_CHARS),
        encode_only = (opts->flags & PS_PARSER_FLAG_ENCODE_ONLY),
        make_targets = (opts->flags & PS_PARSER_FLAG_MAKE_TARGETS),
        fixed_seqlen = opts->sequence_length;
    seq_separator = opts->sequence_separator;
    PSFloat *y_dataset = opts->target_dataset;
    long y_datalen = opts->target_datalen;
    start_token = opts->start_token;
    end_token = opts->end_token;
    if (!encode_only) {
        multi_seq = PS_IS_MULTISEQ(opts);
        if ((use_y_dataset = (y_dataset != NULL && y_datalen > 0))) {
            make_targets = 1;
            if (!multi_seq) {
                PSErr(func, "no sequence splitting provided");
                goto fail;
            }
            /* Read target number of sequences from the first element of
             * target dataset. */
            y_nseq = (long) *(y_dataset++);
            y_datalen--;
            if (y_nseq == 0 || y_datalen == 0) {
                PSErr(func, "empty `y_dataset`");
                goto fail;
            }
        }
    }
    if (do_normalize) {
        normalize = opts->normalizer;
        if (normalize == NULL) normalize = PSNormalizeToken;
    }
    if (opts->flags & PS_PARSER_FLAG_PRESERVE_STRING) {
        /* Duplicate str, since the function could modifiy the parsed string.*/
        tmpstr = strdup(str);
        if (tmpstr == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        str = tmpstr;
    }
    long max_vocab_size = opts->max_vocabulary_size;
    if (max_vocab_size <= 0) max_vocab_size = PS_DEFAULT_MAX_VOCAB_SIZE;
    PSTokenMatch match_token = opts->match_token;
    const char *separator = opts->separator;
    if (separator == NULL && !char_mode)
        separator = PS_DEFAULT_TOKEN_SEPARATOR;
    long count = 0, start_token_id = -1, end_token_id = -1,
         y_start_token_id = -1, y_end_token_id = -1,
         current_capacity = 0, capacity_increment = 0;
    vocab = *vocabulary;
    if (vocab != NULL) capacity_increment = vocab->capacity;
    else {
        new_vocab = 1;
        capacity_increment = opts->capacity;
        vocab = PSVocabularyCreate(opts->capacity);
        if (vocab == NULL) goto fail;
        *vocabulary = vocab;
    }
    PSVocabulary *y_vocab = opts->target_vocabulary;
    if (capacity_increment <= 0)
        capacity_increment = PS_DEFAULT_PARSER_CAPACITY;
    /* Eventually initialize start and end tokens. */
    const char *initial_tokens[] = {start_token, end_token};
    const char *initial_tokens_defaults[] = {
        PS_DEFAULT_START_TOKEN, PS_DEFAULT_END_TOKEN
    };
    long *initial_token_ids[] = {&start_token_id, &end_token_id};
    long *y_token_ids[] = {&y_start_token_id, &y_end_token_id};
    int initial_token_flags[] = {
        PS_PARSER_FLAG_START_TOKEN, PS_PARSER_FLAG_END_TOKEN
    };
    for (i = 0; i < 2; i++) {
        const char *tok = initial_tokens[i];
        int flag = initial_token_flags[i];
        if (tok == NULL) {
            if (opts->flags & flag) tok = initial_tokens_defaults[i];
            else continue;
        }
        long *tok_id_p = initial_token_ids[i], *y_tok_id_p = y_token_ids[i];
        char *toktype = (i == 0 ? "start" : "end");
        if (read_only || (vocab->size >= max_vocab_size)) {
            *tok_id_p = PSVocabularyGetTokenID(vocab, (char *) tok);
            if (y_vocab != NULL)
                *y_tok_id_p = PSVocabularyGetTokenID(y_vocab, (char *) tok);
        } else {
            *tok_id_p = PSVocabularyAdd(vocab, (char *) tok);
            if (y_vocab != NULL)
                *y_tok_id_p = PSVocabularyAdd(y_vocab, (char *) tok);
        }
        if (*tok_id_p < 0) {
            if (!read_only) {
                PSErr(func, "failed to add %s token to vocabulary: %s",
                      toktype, PSVocabularyErrorString(*tok_id_p));
            } else {
                PSErr(func, "%s token not found: %s",
                      toktype, PSVocabularyErrorString(*tok_id_p));
            }
            goto fail;
        }
        if (y_vocab != NULL && *y_tok_id_p < 0) {
            if (!read_only) {
                PSErr(func, "failed to add %s token to target vocabulary: "
                      "%s", toktype, PSVocabularyErrorString(*tok_id_p));
            } else {
                PSErr(func, "%s token not found in target vocabulary: %s",
                      toktype, PSVocabularyErrorString(*tok_id_p));
            }
            goto fail;
        }
    }
    if (y_start_token_id < 0) y_start_token_id = start_token_id;
    if (y_end_token_id < 0) y_end_token_id = end_token_id;
    int use_start_token = (opts->flags & PS_PARSER_FLAG_START_TOKEN) &&
                           start_token_id >= 0;
    int use_end_token = (opts->flags & PS_PARSER_FLAG_END_TOKEN) &&
                         end_token_id >= 0;
    int exact_inputs = opts->flags & PS_PARSER_FLAG_EXACT_INPUTS;

    PSFloat *nseq_p = NULL, *xlen_p = NULL, *ylen_p = NULL,
            *x = NULL, *y = NULL;
    if (existing_data == NULL) {
        /* Create dataset from scratch. */
        current_capacity = capacity_increment;
        data = malloc(current_capacity * sizeof(PSFloat));
        if (data == NULL) goto memerr;
        count = 0;
        if (multi_seq) data[count++] = 0; /* Sequence count. */
        if (!encode_only) data[count++] = 0; /* First sequence length. */
    } else {
        /* Loaded data must be appended to already existing dataset. */
        data = existing_data;
        count = *datalen;
        /*nseq_p = state->nseq_p;*/
        xlen_p = state->xlen_p;
        ylen_p = state->ylen_p;
        x = state->x;
        y = state->y;
        incomplete_target = state->incomplete_target;
        current_capacity = state->capacity;
        if (count == 0 && multi_seq && !encode_only) {
            /* Make room for first two elements containing sequence count and
             * first sequence length. */
            data = makeRoomForNewData(
                data, &current_capacity, 2, capacity_increment, 3,
                &nseq_p, &x, &xlen_p
            );
            count = 2;
            if (multi_seq) data[0] = 0; /* Sequence count. */
            if (!encode_only) data[1] = 0; /* First sequence length. */
        }
        if (y_dataset != NULL && state->target_dataset_offset > 0) {
            y_dataset = opts->target_dataset + state->target_dataset_offset;
            y_datalen = (opts->target_datalen - state->target_dataset_offset);
        }
    }
    if (multi_seq) {
        nseq_p = data;
        if (opts->max_sequences > 0 && (long)*nseq_p > opts->max_sequences)
            return data;
    }
    char *token = str, *p = str, *sep_p = NULL;
    char ctoken[2] = {0};
    int do_free_token = 0;

    /* Parse text */
    while ((p - str) <= (long) end_idx) {
        size_t wlen = 0;
        if (*p == 0) break;
        if (char_mode) {
            /* Character parsing mode: every character is a token. */
            ctoken[0] = *p++;
            token = ctoken;
            if (separator && strchr(separator, ctoken[0])) continue;
            wlen = 1;
            goto add_to_vocab;
        }
        if (match_token != NULL) {
            /* Use the match_token callback. */
            int matched = match_token(p, &wlen);
            if (!matched || wlen <= 0) {
                p++;
                continue;
            }
            token = malloc(wlen + 1);
            if (token == NULL) {
                PSErr(func, "could not allocate token of length: %d",wlen);
                goto memerr;
            }
            memcpy(token, p, wlen);
            token[wlen] = '\0';
            do_free_token = 1;
            p += wlen;
        } else {
            /* Use separator. */
            token = p;
            sep_p = strpbrk(p, separator);
            if (sep_p != NULL) {
                wlen = sep_p - p;
                *sep_p = '\0';
                p = sep_p + 1;
            } else {
                wlen = strlen(p);
                p += wlen;
            }
        }
add_to_vocab:
        if (wlen == 0) continue;
        int is_seq_start = 0, is_seq_end = 0, add_start_token = 0;
        int do_check_seq_end = (
            !encode_only && !incomplete_target && multi_seq
        );
        /* Token normalization */
        if (do_normalize) {
            /* Also check for sequence ending before normalization. */
            int match_seq_end = (
                do_check_seq_end && fixed_seqlen == 0 &&
                opts->match_sequence_end != NULL
            );
            if (match_seq_end)
                is_seq_end = opts->match_sequence_end(token, NULL);
            char *normalized = normalize(token, wlen);
            if (normalized == NULL) {
                PSErr(func, "could not normalize token '%s'", token);
                if (do_free_token) free(token);
                token = NULL;
                goto fail;
            } else if (normalized != token) {
                if (do_free_token) free(token);
                token = normalized;
                do_free_token = 1;
            }
        }
        long id = -1;
        /* Get token ID (by adding/reading it to/from vocabulary. */
        if (read_only || (vocab->size >= max_vocab_size)) {
            id = PSVocabularyGetTokenID(vocab, token);
            if (id == PS_TOKEN_NOT_FOUND) {
                /* Vocabulary is already full or read-only and token was not
                   found, so set it to unknown. */
                if (do_free_token) {
                    free(token);
                    do_free_token = 0;
                }
                token = (char *) opts->unknown_token;
                if (token == NULL) token = PS_DEFAULT_UNKNOWN_TOKEN;
                /* Set or get <unknown> token. */
                if (!read_only) id = PSVocabularyAdd(vocab, token);
                else id = PSVocabularyGetTokenID(vocab, token);
            }
        } else id = PSVocabularyAdd(vocab, token);
        if (id < 0) {
            if (do_free_token) free(token);
            if (read_only) {
                PSErr(func, "failed to add token to vocabulary: %s",
                      PSVocabularyErrorString(id));
            } else {
                PSErr(func, "token not found: %s",
                      PSVocabularyErrorString(id));
            }
            goto fail;
        }
        PSFloat token_id = (PSFloat) id;

        /* Prepare dataset: increase count, ventually make room for the new
         * data and check for sequence termination. */
        long newlen = ++count;
        int xlen = 0, ylen = 0;
        if (!encode_only) {
            if (!incomplete_target) {
                if (x == NULL) {
                    x = data + count - 1;
                    xlen_p = x - 1;
                    is_seq_start = 1;
                    add_start_token = (
                        use_start_token &&
                        !exact_inputs &&
                        (use_y_dataset || *nseq_p == 0 || fixed_seqlen <= 0)
                    );
                    if (add_start_token) {
                        if (use_y_dataset) {
                            newlen++;
                            if (use_end_token) newlen++;
                        }
                        xlen++;
                    }
                } else if (xlen_p != NULL) xlen = *xlen_p;
                /* Increase input (x) sequence length. */
                *xlen_p = ++xlen;
                if (multi_seq && !is_seq_end) {
                    /* Check whether the current token represents the end
                     * of the input sequence. */
                    if (fixed_seqlen > 0)
                        is_seq_end = (xlen >= fixed_seqlen);
                    else if (opts->match_sequence_end != NULL)
                        is_seq_end = opts->match_sequence_end(token, NULL);
                    else if (seq_separator != NULL)
                        is_seq_end = (strcmp(seq_separator, token));
                }
            } else {
                /* Make room for first input (x) token and new sequence length
                 * element. */
                newlen += 2;
            }
        }
        if (do_free_token) free(token);
        if (is_seq_end) {
            /* Increase the sequence count. */
            *nseq_p += 1;
            if (make_targets) {
                int len_elems_count = 1;
                if (use_y_dataset) {
                    /* Take in account 2 elements for input (x) seq. length
                     * and target (y) sequence length and enough elements
                     * for the target sequence by reading the first element
                     * of target dataset containing the sequence length. */
                    len_elems_count = 2;
                    ylen = *(y_dataset++);
                    y_datalen--;
                    if (ylen <= 0 || y_datalen == 0) {
                        PSErr(func, "`y_dataset` is truncated");
                        goto fail;
                    }
                    if (use_start_token) newlen++;
                    if (use_end_token) newlen++;
                } else ylen = (int) *xlen_p;
                newlen += (ylen + len_elems_count);
            } else newlen++;
        }
        PSFloat *new_data = makeRoomForNewData(
            data, &current_capacity, newlen, capacity_increment, 3,
            &nseq_p, &x, &xlen_p
        );
        if (new_data == NULL) goto memerr;
        data = new_data;

        /* Check whether to add the start token. */
        if (is_seq_start && add_start_token) {
            data[count - 1] = start_token_id;
            count++;
        }
        /* Add the current token. */
        data[count - 1] = token_id;
        if (incomplete_target) {
            /* Target sequence (y) was still incomplete, but the current token
             * has already been added as the last token of the target sequence,
             * so set incomplete_target to 0 and initialize the new input
             * sequence by using the current token as the first input (x)
             * token. */
            incomplete_target = 0;
            xlen_p = data + count++;
            *xlen_p = 1;   /* New sequence length */
            x = data + count++;
            *x = token_id; /* New sequence starting token id */
        } else if (is_seq_end) {
            /* Sequence end. */
            if (make_targets) {
                if (!use_y_dataset) {
                    /* No separate dataset for targets, so just inputs (x)
                     * translated by one position as targets:
                     * x[1], ...x[length - 1].
                     * If no 'end' token is to be used, the first element
                     * of the next input sequence will be added to targets
                     * (by setting `incomplete_target` to true), otherwise
                     * just append the 'end' token. */
                    long ycount = ylen - 1;
                    if (ylen > 1)
                        PSVectorCopy((data + count), (x + 1), ycount);
                    if (use_end_token && fixed_seqlen <= 0)
                        data[count + ycount++] = end_token_id;
                    else incomplete_target = 1;
                    count += ycount;
                    if (!incomplete_target) {
                        /* By using the 'end' token, target sequence is
                         * complete, so initialize the next input sequence (x)
                         * counter. */
                        xlen_p = data + count++;
                        *xlen_p = 0;
                    }
                } else {
                    /* Take targets from the separate target dataset. */
                    if (ylen > y_datalen) {
                        PSErr(func, "`y_dataset` is truncated");
                        goto fail;
                    }
                    long ylen_add = 0;
                    if (use_end_token && !exact_inputs && xlen_p != NULL) {
                        *xlen_p = *xlen_p + 1;
                        data[count++] = (PSFloat) end_token_id;
                    }
                    if (use_start_token) ylen_add++;
                    if (use_end_token) ylen_add++;
                    data[count++] = (PSFloat) ylen + ylen_add;
                    if (use_start_token) data[count++] = y_start_token_id;
                    PSVectorCopy(data + count, y_dataset, ylen);
                    y_dataset += ylen;
                    y_datalen -= ylen;
                    count += ylen;
                    if (use_end_token) data[count++] = y_end_token_id;
                    xlen_p = data + count++;
                    *xlen_p = 0;
                }
            } else {
                if (use_end_token && !exact_inputs && xlen_p != NULL) {
                    *xlen_p = *xlen_p + 1;
                    data[count++] = (PSFloat) end_token_id;
                }
                xlen_p = data + count++;
                *xlen_p = 0; /* New sequence length */
            }
            x = NULL;
        }
        if (opts->max_sequences > 0 && (long)*nseq_p >= opts->max_sequences)
            if (multi_seq && !incomplete_target) break;
    }
    if (state != NULL) {
        state->nseq_p = nseq_p;
        state->xlen_p = xlen_p;
        state->ylen_p = ylen_p;
        state->x = x;
        state->y = y;
        state->incomplete_target = incomplete_target;
        state->capacity = current_capacity;
        if (y_dataset != NULL) {
            state->target_dataset_offset = (
                y_dataset - opts->target_dataset
            );
        }
    }
    if (multi_seq && existing_data == NULL) {
        PSDataSequenceState tmpstate = {0};
        if (state == NULL) {
            state = &tmpstate;
            state->nseq_p = nseq_p;
            state->xlen_p = xlen_p;
            state->ylen_p = ylen_p;
            state->x = x;
            state->y = y;
            state->incomplete_target = incomplete_target;
            state->capacity = current_capacity;
            if (y_dataset != NULL) {
                state->target_dataset_offset = (
                    y_dataset - opts->target_dataset
                );
            }
        }
        data = handleTruncatedTextSequenceData(
            data, &count, &current_capacity, end_token_id, opts, state
        );
        if (data == NULL) goto memerr;
    }
    *datalen = count;
    return data;
memerr:
    PSPrintMemoryErrorMsg();
fail:
    if (datalen != NULL) *datalen = 0;
    if (data != NULL) free(data);
    if (new_vocab) {
        if (vocabulary != NULL) *vocabulary = NULL;
        if (vocab != NULL) PSVocabularyFree(vocab);
    }
    free(tmpstr);
    return NULL;
}

char *PSNormalizeToken(char *token, size_t len) {
    if (token == NULL) return NULL;
    for (size_t i = 0; i < len; i++) token[i] = tolower(token[i]);
    return token;
}

/**** Generic Dataset Functions. ****/

/* Split `data` into two separated datasets. This function can be useful to
 * separate validation data used for testing models from data used for
 * training them.
 * The `datalen` argument must contain the length (number of elements) of the
 * `data` array, while `input_size` and `target_size` must contain the number
 * of elements of each single input (`input_size`) and each single target
 * (`target_size`). For example, if the dataset consists of 28x28 images
 * the value for `input_size` must be 784 and if the targets are composed of
 * ten classes (such in popular MNIST dataset), the value of `target_size`
 * should be 10. If `data` contains no targets, the value of `target_size`
 * must be zero.
 * The size of the resulting datasets (called left and right dataset), are
 * defined by the value of `percentage` that is the percentage of `data` that
 * goes to the "left" datasets and must be expressed with a value between 0.0
 * and 1.0, so, for example, a percentage of 0.8 means that the left dataset
 * will receive the 80% of the examples  from `data` and, consequently, the
 * right dataset will receive the 20% of the examples from `data`.
 * The way the data is distributed between the two datasets can vary depending
 * on the value of `opts`. By default, the examples at the beginning of `data`
 * go to the left dataset until the selected percentage is reached and then
 * the remaining examples go to the right dataset.
 * If `opts` has the `PS_DATA_EVENLY_SPREAD` flag enabled, data will be
 * evenly distributed to both datasets in an uniform way.
 * If `opts` has the `PS_DATA_SHUFFLE` flag enabled, data will be randomly
 * assigned to both datasets.
 * If `data` is made up of sequences, the `PS_DATA_SEQUENCES` flag must be
 * enabled into the `opts` argument. If input sequences and target sequences
 * may have different lengths, the flag `PS_DATA_SEQ2SEQ` must be enabled
 * into the `opts` argument.
 * The addresses of the resulting datasets will be stored into the `left`
 * and `right` arguments and their lengths (number of their respective elements)
 * will be stored into the `left_length` and `right_length` arguments.
 * NOTE: if `data` has no sequences and none of `PS_DATA_ALWAYS_ALLOC`,
 * `PS_DATA_SHUFFLE` and `PS_DATA_EVENLY_SPREAD` flags are set, the function
 * won't allocate the resulting datasets, unless `PS_DATA_ALWAYS_ALLOC`. In
 * this case, `left` will contain the pointer to the original address of `data`
 * and `right` will contain the pointer to the first element of `data` that
 * will belong to the right dataset. This means that the right dataset should
 * **never be freed** by its own. In all the other cases, memory for both the
 * left and the right dataset will be allocated and must be freed when not
 * used anymore.
 * Return value: 1 in case of success, 0 in case of failure.
 * Possible failure reasons:
 *  - at least one of `data`, `left`, `right`, `left_length` or `right_length`
 *    is NULL.
 *  - `datalen` is zero.
 *  - `input_size` is zero.
 *  - `opts` has the flags `PS_DATA_SEQUENCES` or `PS_DATA_SEQ2SEQ` enabled
 *    but `target_size` is zero.
 *  - `percentage` is less than or equal to 0 or greater than or equal to 1.
 *  - `opts` has the flags `PS_DATA_SEQUENCES` or `PS_DATA_SEQ2SEQ` enabled
 *    but the dataset contains no sequences (the value of the first element
 *    of `data` is less than or equal to zero.
 *  - There's no enough memory to be allocated for left and right datasets. */
int PSDataSplit(PSFloat *data, long datalen, float percentage,
                long input_size, long target_size,
                PSFloat **left, PSFloat **right,
                long *left_length, long *right_length,
                int opts)
{
    PSFloat *left_data = NULL, *right_data = NULL;
    int success = (data != NULL);
    if (!success) {
        PSErr(__func__, "missing mandatory argument `data`");
        goto final;
    }
    success = (datalen > 0);
    if (!success) {
        PSErr(__func__, "empty dataset");
        goto final;
    }
    success = (left != NULL);
    if (!success) {
        PSErr(__func__, "missing mandatory argument `left`");
        goto final;
    }
    success = (right != NULL);
    if (!success) {
        PSErr(__func__, "missing mandatory argument `right`");
        goto final;
    }
    success = (left_length != NULL);
    if (!success) {
        PSErr(__func__, "missing mandatory argument `left_length`");
        goto final;
    }
    success = (right_length != NULL);
    if (!success) {
        PSErr(__func__, "missing mandatory argument `right_length`");
        goto final;
    }
    success = (percentage > 0 && percentage < 1);
    if (!success) {
        PSErr(__func__, "`percentage` must be greater 0 and less than 1");
        goto final;
    }
    int seq2seq = opts & PS_DATA_SEQ2SEQ,
        has_targets = target_size > 0 || seq2seq,
        has_seqs = (opts & PS_DATA_SEQUENCES) || seq2seq,
        shuffle = opts & PS_DATA_SHUFFLE,
        evenly_split = opts & PS_DATA_EVENLY_SPREAD;
    success = !has_targets || target_size > 0;
    if (!success) {
        PSErr(__func__, "`target_size` must be greater than zero");
        goto final;
    }
    success = input_size > 0;
    if (!success) {
        PSErr(__func__, "`input_size` must be greater than zero");
        goto final;
    }
    long example_size = 0, n_examples = 0;
    PSFloat *data_p = data;
    if (has_seqs) n_examples = (long) *(data_p++);
    else {
        example_size = input_size + target_size;
        success = example_size > 0;
        if (!success) {
            PSErr(__func__, "invalid example size");
            goto final;
        }
        n_examples = datalen / example_size;
    }
    success = n_examples > 0;
    if (!success) {
        PSErr(__func__, "no examples found in dataset");
        goto final;
    }
    long left_examples = (long) lroundf(percentage * (float) n_examples);
    long right_examples = n_examples - left_examples;
    long llen = 0, rlen = 0, llen_alloc = 0, rlen_alloc = 0;
    if (!has_seqs) {
        llen_alloc = left_examples * example_size;
        rlen_alloc = right_examples * example_size;
        if (!shuffle && !evenly_split) {
            /* Trivial situation. */
            *left_length = llen_alloc;
            *right_length = rlen_alloc;
            if (opts & PS_DATA_ALWAYS_ALLOC) {
                *left = PSVectorDup(data, llen_alloc);
                *right = PSVectorDup(data + llen_alloc, rlen_alloc);
            } else {
                *left = data;
                *right = data + llen_alloc;
            }
            return 1;
        }
    } else {
        /* Heuristically predict left and right size. */
        llen_alloc = (long) (percentage * (float) datalen);
        rlen_alloc = (long) ((1 - percentage) * (float) datalen);
    }
    left_data = malloc(llen_alloc * sizeof(PSFloat));
    success = (left_data != NULL);
    if (!success) {
        PSPrintMemoryErrorMsg();
        goto final;
    }
    right_data = malloc(rlen_alloc * sizeof(PSFloat));
    success = (right_data != NULL);
    if (!success) {
        PSPrintMemoryErrorMsg();
        goto final;
    }
    PSFloat **big = NULL, **small = NULL;
    long small_examples = 0, big_examples = 0, slice_examples = 0, i;
    long *small_len = NULL, *big_len = NULL, *small_len_alloc = NULL,
         *big_len_alloc = NULL;
    if (shuffle || evenly_split) {
        if (percentage >= 0.5) {
            big = &left_data;
            big_examples = left_examples;
            big_len = &llen;
            big_len_alloc = &llen_alloc;
            small = &right_data;
            small_examples = right_examples;
            small_len = &rlen;
            small_len_alloc = &rlen_alloc;
        } else {
            big = &right_data;
            big_examples = right_examples;
            big_len = &rlen;
            big_len_alloc = &rlen_alloc;
            small = &left_data;
            small_examples = left_examples;
            small_len = &llen;
            small_len_alloc = &llen_alloc;
        }
        slice_examples = (big_examples / small_examples) + 1;
    }
    if (has_seqs) {
        left_data[llen++] = 0;
        right_data[rlen++] = 0;
    }
    long example2pick = -1;
    for (i = 0; i < n_examples; i++) {
        /* Get current example. */
        PSFloat *example = data_p;
        if (has_seqs) {
            int xlen = (int) *(data_p++), ylen = xlen;
            data_p += (xlen * input_size);
            if (has_targets) {
                if (seq2seq) ylen = (int) *(data_p++);
                data_p += ylen * input_size;
            }
            example_size = data_p - example;
        } else data_p += example_size;
        /* Determine target (destination) dataset. */
        PSFloat **target_dataset = NULL;
        long *target_len = NULL, *target_len_alloc = NULL;
        if (!shuffle && !evenly_split) {
            if (i < left_examples) {
                target_dataset = &left_data;
                target_len = &llen;
                target_len_alloc = &llen_alloc;
            } else {
                target_dataset = &right_data;
                target_len = &rlen;
                target_len_alloc = &rlen_alloc;
            }
        } else {
            long slice_idx = i / slice_examples;
            long slice_example_idx = i % slice_examples;
            long remaining_examples = (
                n_examples - (slice_idx * slice_examples)
            );
            long slice_len = (
                remaining_examples >= slice_examples ? slice_examples :
                                                       remaining_examples
            );
            long small_count, big_count;
            if (has_seqs) {
                small_count = (long) ((*small)[0]);
                big_count = (long) ((*big)[0]);
            } else {
                small_count = *small_len / example_size;
                big_count = *big_len / example_size;
            }
            if (slice_example_idx == 0) {
                /* First example of the slice, determine example2pick if
                 * small dataset is not full. */
                 if (small_count < small_examples) {
                    if (shuffle)
                        example2pick = PSRandomInt(slice_len, NULL, NULL);
                    else
                        example2pick = (percentage >= 0.5 ? slice_len - 1 : 0);
                 } else example2pick = -1;
            }
            if (example2pick == slice_example_idx) {
                target_dataset = small;
                target_len = small_len;
                target_len_alloc = small_len_alloc;
            } else if (big_count < big_examples) {
                target_dataset = big;
                target_len = big_len;
                target_len_alloc = big_len_alloc;
            } else {
                PSErr(NULL, "cannot determine dataset for example %llu", i);
                success = 0;
                goto final;
            }
        }
        long curlen = *target_len;
        *target_len += example_size;
        if (*target_len > *target_len_alloc) {
            PSFloat *resized = realloc(
                *target_dataset, *target_len * sizeof(PSFloat)
            );
            success = (resized != NULL);
            if (!success) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            *target_len_alloc = *target_len;
            *target_dataset = resized;
        }
        PSVectorCopy(*target_dataset + curlen, example, example_size);
        if (has_seqs) (*target_dataset)[0] += 1;
    }
    if (llen < llen_alloc) {
        PSFloat *resized = realloc(left_data, llen * sizeof(PSFloat));
        if (resized != NULL) left_data = resized;
    }
    if (rlen < rlen_alloc) {
        PSFloat *resized = realloc(right_data, rlen * sizeof(PSFloat));
        if (resized != NULL) right_data = resized;
    }
    *left = left_data;
    *right = right_data;
    *left_length = llen;
    *right_length = rlen;
final:
    if (!success) {
        if (left_length != NULL) *left_length = 0;
        if (right_length != NULL) *right_length = 0;
        if (left != NULL) *left = NULL;
        if (right != NULL) *right = NULL;
        free(left_data);
        free(right_data);
    }
    return success;
}

/* Load a dataset (an array of `PSFloat` numbers) from a string. Depending
 * on the parsing mode, each token or character found in the string will be
 * converted to a numeric representation of itself. The dataset can be used to
 * train a model (`PSModel`) or it can provide inputs to the model.
 * Numeric representation of tokens/characters is defined by key-value pairs
 * contained into `vocabulary` and indices (ids) related to token are cast to
 * `PSFloat`.
 * The function can use an existing vocabulary or create a new one from
 * scratch.
 * By default, the string will be parsed as a sequence of tokens
 * (`PS_PARSER_MODE_TOKENS`) and each token will be converted to a number.
 * The above behavior can be changed by using `PS_PARSER_MODE_CHARS` in
 * the (optional) `opts` argument (see `PSTextParserOptions`).
 *
 * When `PS_PARSER_MODE_TOKENS` mode is used, tokens can be matched in two ways:
 *  - By using a separator: in this case the string will be split by using the
 *    separators defined into `separator` member of `opts`.
 *    If `separator` is NULL or `opts` is NULL, the default separators
 *    will be those defined by the macro `PS_DEFAULT_TOKEN_SEPARATOR`.
 *  - By using a callback that let the developers to define their own logic
 *    for identifying and extracting tokens from the input string.
 *    The callback function can be set into the `match_token` member of `opts`,
 *    and it's a function of type `PSTokenMatch`.
 *    The callback receives a token to match and a pointer to an integer to
 *    store the length of the matched token. If the callback returns a non-zero
 *    value (true), it indicates a successful match and a token of the matched
 *    length will be extracted from the input string.
 *    In this case, the parsing position is moved forward by the matched length.
 *    If the callback returns 0, it signals that the token was not matched, and
 *    the parsing position will be advanced to the next byte.
 *
 * If the flag `PS_PARSER_FLAG_ENCODE_ONLY` is set, the resulting dataset will
 * only consist of converted token values, with no further info or metadata
 * about the dataset itself. Otherwise, other data is added to the dataset,
 * such as sequence length, number of sequences, and so on.
 *
 * By default, the dataset is generated as a whole single sequence, and
 * the sequence length is prepended as the first element of the dataset.
 * In order to split text into multiple sequences, it's possible to use
 * the properties of `opts`:
 *  - If `sequence_length` is greater than zero, the text will be split into
 *    into multiple fixed-length sequence having `sequence_length` tokens.
 *  - If the `match_sequence_end` callback is not NULL, it will be called
 *    with the current token and, if its return value is true, the token will
 *    be the ending token of the current sequence. It produces variable-length
 *    sequences.
 *  - If the `sequence_separator` string is not NULL and the current token
 *    is equal to it, the current token will be the ending token of the
 *    current sequence. It produces variable-length sequences.
 * When the dataset is split into multiple sequences, the total number of
 * sequence is set into the first element of the dataset itself.
 *
 * By default, no target data will be generated, unless the flag
 * `PS_PARSER_FLAG_MAKE_TARGETS` is set into `flags` member of `opts` or a
 * target dataset is specified into `opts` by using the `target_dataset` and
 * the `target_datalen` members of `opts`.
 * If no `target_dataset` is provided, by default the target sequences will be
 * generated by shifting the related input sequence by one. For example, if
 * we have an the following sequence "hello world have a nice day" and
 * a `sequence_length` of five tokens, the input sequence will be
 * "hello world have a nice" and the target sequence will be "world have a
 * nice day".
 * Incomplete sequences are dropped, but it's possible to "pad" them by using
 * 'start' and 'end' tokens that can be set with the
 * `PS_PARSER_FLAG_START_TOKEN` and `PS_PARSER_FLAG_END_TOKEN` flags into
 * `flags` member of `opts`.
 * If not specified by `start_token` and `end_token` members of `opts`, the
 * function will use the default string defined by `PS_DEFAULT_START_TOKEN` and
 * `PS_DEFAULT_END_TOKEN`.
 * If a `target_dataset` is provided, target sequences will be taken from it
 * (the target dataset must contain at least the same number of sequences of
 * the current dataset being generated).
 *
 * WARN: The parsed string `str` may be modified during text parsing. By
 * setting the `PS_PARSER_FLAG_PRESERVE_STRING` flag into `opts->flags`, the
 * function will work on a copy of the string, preventing the original string
 * from being altered.
 *
 * Arguments:
 *  - `str`: the (null-terminated) string to be parsed (mandatory).
 *  - `opts`: parsing options, it can be NULL.
 *  - `existing_data`: optional argument that can be used to append
 *     parsed data to an existing dataset.
 *     WARN: Since data can be reallocated, always use the returned dataset
 *     after calling the function.
 *  - `datalen`: pointer to `int64_t` where the final length of the dataset
 *     will be stored. If `existing_data` is not `NULL`, the address pointed
 *     by `datalen` must contain the current length of the existing dataset.
 *     If NULL is returned by the function, the pointed address will contain
 *     zero.
 *  - `vocabulary`: pointer to pointer to a `PSVocabulary` struct. The
 *     argument is mandatory and cannot be `NULL`.
 *     If the pointer pointed by `vocabulary` is NULL, a new `PSVocabulary`
 *     will be allocated and it will be filled with parsed tokens|characters.
 *     If the pointer pointed by `vocabulary` points to an already existing
 *     vocabulary, its numeric values will be used for parsed tokens.
 *     If a token or character is not found in the existing vocabulary,
 *     it will be automatically added, unless `PS_PARSER_FLAG_READONLY_VOCAB`
 *     flag is set into `opts`.
 * Return value: the dataset (`PSFloat` array) or NULL is something goes
 * wrong. */
PSFloat *PSDataFromText(char *str, PSTextParserOptions *opts, long *datalen,
                        PSVocabulary **vocabulary)
{
    PSDataSequenceState state = {0};
    PSFloat *data = loadDatasetFromString(
        str, opts, datalen, vocabulary, NULL, &state, __func__
    );
    if (data == NULL) return NULL;
    if (*datalen < state.capacity) {
        PSFloat *resized = realloc(data, *datalen * sizeof(PSFloat));
        if (resized != NULL) data = resized;
    }
    return data;
}

/* Load a dataset (an array of PSFloat numbers) from the text file found at
 * `filepath`.
 * For parsing options and other arguments, see `PSDataFromText`.
 * Return value: the dataset (`PSFloat` array) or NULL is something goes
 * wrong. */
PSFloat *PSDataFromTextFile(const char *filepath, PSTextParserOptions *opts,
                            long *datalen, PSVocabulary **vocabulary)
{
    static PSTextParserOptions default_opts = {0};
    FILE *file = NULL;
    PSVocabulary *vocab = NULL;
    PSFloat *data = NULL;
    if (opts == NULL) opts = &default_opts;
    int buffer_size = opts->buffer_size, new_vocab = 0;
    if (buffer_size <= 0) buffer_size = PS_PARSER_BUFFER_SIZE;
    char buf[buffer_size];
    buf[0] = '\0';
    if (filepath == NULL) {
        PSErr(__func__, "Missing mandatory argument `filepath`");
        goto fail;
    }
    if (datalen == NULL) {
        PSErr(__func__, "Missing mandatory argument `datalen`");
        goto fail;
    }
    if (vocabulary == NULL) {
        PSErr(__func__, "Missing mandatory argument `vocabulary`");
        goto fail;
    }
    vocab = *vocabulary;
    const char *separator = opts->separator;
    if (separator == NULL) separator = PS_DEFAULT_TOKEN_SEPARATOR;
    long capacity = 0;
    if (vocab != NULL) capacity = vocab->capacity;
    else {
        new_vocab = 1;
        capacity = opts->capacity;
        if (capacity <= 0) capacity = PS_DEFAULT_PARSER_CAPACITY;
        vocab = PSVocabularyCreate(capacity);
        if (vocab == NULL) goto fail;
        *vocabulary = vocab;
    }
    file = fopen(filepath, "r");
    if (file == NULL) {
        PSErr(__func__, "Could not open file '%s'", filepath);
        goto fail;
    }
    data = malloc(capacity * sizeof(PSFloat));
    if (data == NULL) goto memerr;
    int multi_seq = (
        !(opts->flags & PS_PARSER_FLAG_ENCODE_ONLY) &&
        PS_IS_MULTISEQ(opts)
    );
    size_t nread = 0, buflen = sizeof(buf) - 1;
    *datalen = 0;
    int err = 0;
    PSDataSequenceState state = {0};
    while ((nread = fread(buf, 1, buflen, file))) {
        err = ferror(file);
        if (err != 0 || nread <= 0) break;
        buf[nread] = '\0';
        size_t last_idx = nread - 1;
        if (!feof(file)) {
            size_t idx = last_idx;
            while (strchr(separator, buf[idx]) == NULL) {
                /* Buffer probabily ends with a broken word, so buffer must
                 * be truncated to last separator found. */
                if (--idx == 0) {
                    PSErr(__func__, "word is too long (file: '%s', "
                          "offset: %lld), increase buffer size", filepath,
                          ftello(file) - nread);
                    goto fail;
                }
            }
            if (idx < last_idx) {
                /* Truncate buffer to index of last separator found and
                 * reset file stream offset to first byte after truncation. */
                buf[idx] = '\0';
                long truncated_len = (long) (last_idx - idx);
                fseek(file, -truncated_len, SEEK_CUR);
                last_idx = idx;
            }
        }
        data = loadDatasetFromString(
            buf, opts, datalen, &vocab, data, &state, __func__
        );
        if (data == NULL) goto fail;
        if (multi_seq && opts->max_sequences > 0) {
            if ((long) *data >= opts->max_sequences) break;
        }
    }
    if (err != 0) {
        PSErr(__func__, "Error while reading file '%s' (%d): '%s'",
              filepath, err, strerror(err));
        goto fail;
    }
    fclose(file);
    int encode_only = (opts->flags & PS_PARSER_FLAG_ENCODE_ONLY);
    if (!encode_only && PS_IS_MULTISEQ(opts)) {
        long end_token_id = -1;
        if (opts->end_token != NULL || opts->flags & PS_PARSER_FLAG_END_TOKEN)
        {
            PSVocabulary *y_vocab = opts->target_vocabulary;
            if (y_vocab == NULL) y_vocab = vocab;
            char *end_tok = (char*) opts->end_token;
            if (end_tok == NULL) end_tok = PS_DEFAULT_END_TOKEN;
            end_token_id = PSVocabularyGetTokenID(y_vocab, end_tok);
        }
        data = handleTruncatedTextSequenceData(
            data, datalen, &(state.capacity), end_token_id, opts, &state
        );
        if (data == NULL) goto memerr;
    }
    if (*datalen < state.capacity) {
        PSFloat *resized = realloc(data, *datalen * sizeof(PSFloat));
        if (resized != NULL) data = resized;
    }
    return data;
memerr:
    PSPrintMemoryErrorMsg();
fail:
    if (datalen != NULL) *datalen = 0;
    if (file != NULL) fclose(file);
    if (data != NULL) free(data);
    if (new_vocab) {
        if (vocabulary != NULL) *vocabulary = NULL;
        if (vocab != NULL) PSVocabularyFree(vocab);
    }
    return NULL;
}

/**** Generic Datasets ****/

/* Load dataset from file located at `filepath`. Dataset is returned as an
 * array of PSFloat elements whose length (number of elements) is stored into
 * mandatory argument `datalen`.
 * The file must be an ASCII file where every number of the dataset is written
 * as a string representation of floating point numbers and separated by a
 * comma character.
 * Optionally, the whole dataset can be prefixed with its length written
 * as a string representation of a decimal number followed by a colon separator
 * caharcter (':').
 * Example: 3:1.25,2,-0.15 (dataset of three elements 1.25, 2.0 and -0.15)
 * Return value: the loaded dataset or NULL is somethign goes wrong.
 * Possible failure reasons:
 *  - Mandatory arguments `filepath` or `datalen` are NULL.
 *  - File is not found at `filepath`.
 *  - File at `filepath` cannot be opened or read.
 *  - Dataset cannot be allocated into memory. */
PSFloat *PSDataLoad(const char *filepath, long *datalen) {
    FILE *file = NULL;
    PSFloat *data = NULL;
    if (filepath == NULL) {
        PSErr(__func__, "Missing mandatory argument `filepath`");
        goto fail;
    }
    if (datalen == NULL) {
        PSErr(__func__, "Missing mandatory argument `datalen`");
        goto fail;
    }
    long capacity = 0;
    file = fopen(filepath, "r");
    if (file == NULL) {
        PSErr(__func__, "Could not open file '%s'", filepath);
        goto fail;
    }
    int first_byte = fgetc(file);
    fseek(file, 0, SEEK_SET);
    if (first_byte == EOF) {
        PSErr(__func__, "read error while reading file '%s'", filepath);
        return 0;
    }
    if (first_byte == 0xFF) {
        /* Probabily the dataset file is in binary format. */
        size_t veclen = 0;
        data = loadBinaryVector(filepath, file, &veclen);
        if (data == NULL) {
            PSErr(
                __func__, "could not load dataset from binary file '%s'",
                filepath
            );
            *datalen = 0;
        }
        *datalen = (long) veclen;
        fclose(file);
        return data;
    }
    char prfx_sep[2] = {0};
    int matched = fscanf(file, "%ld%[:]", &capacity, prfx_sep);
    if (matched < 2 ||  prfx_sep[0] != ':') {
        capacity = 1024;
        fseek(file, 0, SEEK_SET);
    }
    size_t size = 0;
    data = readSerializedFloatArray(file, ",", &size, 0, capacity);
    *datalen = (long) size;
    if (data == NULL || *datalen <= 0) goto fail;
    if (*datalen < capacity) {
        PSFloat *resized = realloc(data, *datalen * sizeof(PSFloat));
        if (resized != NULL) data = resized;
    }
    fclose(file);
    return data;
fail:
    if (file != NULL) fclose(file);
    free(data);
    return NULL;
}

/* Save dataset `data` to the file located at `path`. The dataset must be
 * an array of PSFloat elements whose length (number of elements) defined by
 * argument `len`.
 * By default, the dataset is saved as a comma-separated list of its values
 * written as string representations of floating point numbers.
 * The datasets itself is prefixed with its length written as a string
 * representation of a decimal number followed by a colon separator
 * caharcter (':').
 * Example: 3:1.25,2,-0.15 (dataset of three elements 1.25, 2.0 and -0.15)
 * If flag `PS_IO_BINARY_MODE` is set into `opts`, the dataset will be saved in
 * binary format.
 * Return value: 1 if datasets is successfully saved, 0 in case of failure.
 * Possible failure reasons:
 *  - Mandatory arguments `path` or `data` are NULL.
 *  - File at `path` cannot be opened for writing.
 *  - Some error occurs while writing to the file.  */
int PSDataSave(const char *path, PSFloat *data, long len, int opts) {
    FILE *file = NULL;
    int success = (path != NULL);
    if (!success) {
        PSErr(__func__, "Missing mandatory argument `path`");
        goto final;
    }
    success = (data != NULL);
    if (!success) {
        PSErr(__func__, "Missing mandatory argument `data`");
        goto final;
    }
    if (len == 0) return 0;
    file = fopen(path, "w");
    success = (file != NULL);
    if (!success) {
        PSErr(__func__, "could not open '%s' for writing", path);
        goto final;
    }
    /* Save in binary format */
    if (opts & PS_IO_BINARY_MODE) {
        success = saveBinaryVector(file, data, len);
        goto final;
    }
    fprintf(file, "%ld:", len);
    size_t wlen = writeSerializedFloatArray(file, len, ",", 0, data);
    success = (wlen > 0);
final:
    if (file != NULL) fclose(file);
    return success;
}

/**** MNIST Dataset ****/

#define MNIST_CHUNK 16384
#define IMAGES_MAGIC_NUM 2051
#define LABELS_MAGIC_NUM 2049

int decompressGZip(FILE *source, FILE *dest) {
    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[MNIST_CHUNK];
    unsigned char out[MNIST_CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit2(&strm, 16+MAX_WBITS);
    if (ret != Z_OK)
        return ret;
    /* decompress until deflate stream ends or end of file */
    do {
        strm.avail_in = fread(in, 1, MNIST_CHUNK, source);
        if (ferror(source)) {
            (void)inflateEnd(&strm);
            return Z_ERRNO;
        }
        if (strm.avail_in == 0)
            break;
        strm.next_in = in;
        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = MNIST_CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR;
                /* fall through */
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    (void)inflateEnd(&strm);
                    return ret;
            }
            have = MNIST_CHUNK - strm.avail_out;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                (void)inflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);
    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}


/* report a zlib or i/o error */
void zerr(int ret) {
    fputs("zpipe: ", stderr);
    switch (ret) {
        case Z_ERRNO:
            if (ferror(stdin))
                fputs("error reading stdin\n", stderr);
            if (ferror(stdout))
                fputs("error writing stdout\n", stderr);
            break;
        case Z_STREAM_ERROR:
            fputs("invalid compression level\n", stderr);
            break;
        case Z_DATA_ERROR:
            fputs("invalid or incomplete deflate data\n", stderr);
            break;
        case Z_MEM_ERROR:
            fputs("out of memory\n", stderr);
            break;
        case Z_VERSION_ERROR:
            fputs("zlib version mismatch!\n", stderr);
    }
}

void getTempFileName(const char *prefix, char *buffer) {
    char buff[4] = {0};
    if (prefix == NULL || buffer == NULL) return;
    FILE *urand = fopen("/dev/urandom", "r");
    if (urand != NULL) {
        fgets(buff, 4, urand);
        fclose(urand);
    }
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x",
            prefix,
            (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3]);
}

/* Load the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database) from
 * files.
 * The dataset files must be in gzip format (.gz): images data must be loaded
 * from `images_file` path and labels data must be loaded from `labels_file`
 * path.
 * The dataset is allocated by the function itself and its pointer is stored
 * into `data` pointer-to-pointer (it cannot be NULL). The length of the
 * resulting dataset is returned by the function.
 * The `type` argument can be used to tell if the dataset is a training
 * dataset (`PS_DATA_TYPE_TRAINING`) or a test dataset (`PS_DATA_TYPE_TEST`).
 * Return value: the length of the dataset (number of `PSFloat` elements) or
 * zero if some error occurs.
 * Possibile errors:
 *  - The data argument is NULL
 *  - Dataset files are NULL, they don't exist or they cannot be opened.
 *  - Dataset cannot be allocated into memory
 *  - Dataset files are not in gzip format or some error occurs whil unzipping
 *    them.
 *  - The internal format of the dataset files format is not valid.
 */
int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data)
{
    if (data == NULL) {
        PSErr(__func__, "argument `data` cannot be null");
        return 0;
    }
    char tmpImagesFileName[PATH_MAX] = {0};
    char tmpLabelsFileName[PATH_MAX] = {0};
    char *prefixImg = NULL, *prefixLbl = NULL;
    int data_len = 0, err;
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO);
    *data = NULL;
    if (type == PS_DATA_TYPE_TRAINING) {
        if (do_log) printf("Loading MNIST Data for training...\n");
        prefixImg = "train-images";
        prefixLbl = "train-labels";
    } else {
        if (do_log) printf("Loading MNIST Data for testing...\n");
        prefixImg = "test-images";
        prefixLbl = "test-labels";
    }
    getTempFileName(prefixImg, tmpImagesFileName);
    getTempFileName(prefixLbl, tmpLabelsFileName);
    FILE *images = fopen(images_file, "r");
    if (images == NULL) {
        PSErr(__func__, "Cannot open images '%s'", images_file);
        *data = NULL;
        return 0;
    }
    FILE *labels = fopen(labels_file, "r");
    if (labels == NULL) {
        PSErr(__func__, "Cannot open labels '%s'", labels_file);
        *data = NULL;
        fclose(images);
        return 0;
    }
    FILE *tmpimages = fopen(tmpImagesFileName, "w");
    FILE *tmplabels = fopen(tmpLabelsFileName, "w");
    if (tmpimages == NULL || tmplabels == NULL) {
        PSErr(__func__, "Cannot open temporary files for writing");
        *data = NULL;
        if (tmpimages) fclose(tmpimages);
        if (tmplabels) fclose(tmplabels);
        return 0;
    }
    if (do_log) printf("Loading images...\n");
    err = decompressGZip(images, tmpimages);
    if (err) {zerr(err); *data = NULL; goto final;}
    if (do_log) printf("Loading labels...\n");
    err = decompressGZip(labels, tmplabels);
    if (err) {zerr(err); *data = NULL; goto final;}
    fclose(tmpimages);
    fclose(tmplabels);
    tmpimages = fopen(tmpImagesFileName, "r");
    tmplabels = fopen(tmpLabelsFileName, "r");
    if (tmpimages == NULL || tmplabels == NULL) {
        PSErr(__func__, "Cannot open temporary files for writing");
        *data = NULL;
        if (tmpimages) fclose(tmpimages);
        if (tmplabels) fclose(tmplabels);
        return 0;
    }
    fseek(tmpimages, 0, SEEK_SET);
    fseek(tmplabels, 0, SEEK_SET);
    uint32_t magic_num = 0, image_count = 0, label_count = 0;
    int i = 0, j = 0, do_swap = !PS_IS_BIG_ENDIAN;
    fread(&magic_num, 1, 4, tmpimages);
    if (do_swap) magic_num = swap_uint32(magic_num);
    if (magic_num != IMAGES_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for image file: %d", magic_num);
        *data = NULL;
        goto final;
    }
    fread(&image_count, 1, 4, tmpimages);
    if (do_swap) image_count = swap_uint32(image_count);
    if (image_count == 0) {
        PSErr(__func__, "Image count is zero");
        *data = NULL;
        goto final;
    }
    if (do_log) printf("Found %d images.\n", image_count);
    fread(&magic_num, 1, 4, tmplabels);
    if (do_swap) magic_num = swap_uint32(magic_num);
    if (magic_num != LABELS_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for labels file: %d",magic_num);
        *data = NULL;
        goto final;
    }
    fread(&label_count, 1, 4, tmplabels);
    if (do_swap) label_count = swap_uint32(label_count);
    if (label_count == 0) {
        PSErr(__func__, "Label count is zero");
        *data = NULL;
        goto final;
    }
    if (do_log) printf("Found %d labels.\n", label_count);
    if (label_count != image_count) {
        PSErr(__func__, "Image count and label count do not match!");
        *data = NULL;
        goto final;
    }
    uint32_t rows = 0, cols = 0;
    fread(&rows, 1, 4, tmpimages);
    fread(&cols, 1, 4, tmpimages);
    if (do_swap) {
        rows = swap_uint32(rows);
        cols = swap_uint32(cols);
    }
    if (do_log) printf("Image size: %dx%d\n", rows, cols);
    int img_area = rows * cols;
    if (img_area == 0) {
        PSErr(__func__, "Invalid image size");
        *data = NULL;
        goto final;
    }
    data_len = (img_area * image_count) + (label_count * 10);
    *data = malloc(data_len * sizeof(PSFloat));
    if (*data == NULL) {
        PSPrintMemoryErrorMsg();
        data_len = 0;
        goto final;
    }
    PSFloat *data_p = *data;
    for (i = 0; i < (int) image_count; i++) {
        if (do_log) printf("\rLoading image %d/%d", i + 1, image_count);
        for (j = 0; j < img_area; j++) {
            int pixel = fgetc(tmpimages);
            PSFloat d = (PSFloat) pixel / (PSFloat) 255;
            *(data_p++) = d;
        }
        int label = fgetc(tmplabels);
        /* printf("Label: %d", label); */
        for (j = 0; j < 10; j++) {
            *(data_p++) = (j == label);
        }
    }
    if (do_log) printf("\n");
final:
    if (images != NULL) fclose(images);
    if (labels != NULL) fclose(labels);
    if (tmpimages != NULL) fclose(tmpimages);
    if (tmplabels != NULL) fclose(tmplabels);
    unlink(tmpImagesFileName);
    unlink(tmpLabelsFileName);
    /* printf("Datalen: %d\n", data_len); */
    /* printf("Allocated data size: %d\n", data_p - *data); */
    return data_len;
}

/*** CIFAR Dataset ****/
#define CIFAR_FILE_IMG_COUNT 10000
#define CIFAR_IMAGE_BYTESIZE 3072
#define CIFAR_DATAFILE_COUNT 6

static int compareFilenames(const void* a, const void* b) {
    return strcmp((const char*)a, (const char*)b);
}

/* Load the CIFAR dataset (https://www.cs.toronto.edu/~kriz/cifar.html) from
 * files.
 * The dataset files must be in gzip format (.gz): dataset files must be
 * in binary version and located into `dataset_path` directory.
 * The dataset is allocated by the function itself and its pointer is stored
 * into `data` pointer-to-pointer (it cannot be NULL). The length of the
 * resulting dataset is returned by the function.
 * The CIFAR dataset comes in two fashions:
 *  - CIFAR-10:  each image can be classified with 10 classes.
 *  - CIFAR-100: each image can be classified with 100 classes.
 * The `classes` argument can be used to tell the function which kind of
 * dataset is going to be loaded.
 * The `type` argument can be used to tell if the dataset is a training
 * dataset (`PS_DATA_TYPE_TRAINING`) or a test dataset (`PS_DATA_TYPE_TEST`).
 * Return value: the length of the dataset (number of `PSFloat` elements) or
 * zero if some error occurs.
 * Possibile errors:
 *  - The data argument is NULL
 *  - The value for `class` is neither 10 not 100.
 *  - Dataset directory is NULL, or dataset files cannot be opened cannot be
 *    opened.
 *  - Dataset cannot be allocated into memory
 *  - Dataset file format is not valid
 */
int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_examples)
{
    if (data == NULL) {
        PSErr(__func__, "argument `data` cannot be null");
        return 0;
    }
    if (classes != 10 && classes != 100) {
        PSErr(__func__, "Invalid classes %d: only 10 or 100 allowed.", classes);
        return 0;
    }
    int label_size = (classes == 100 ? 2 : 1);
    int img_count = CIFAR_FILE_IMG_COUNT;
    if (max_examples > 0) img_count = max_examples;
    int expected_fsize = (CIFAR_IMAGE_BYTESIZE + label_size) * img_count;
    int fcount = 0, dataset_size = 0, i, j, k;
    char datafiles[CIFAR_DATAFILE_COUNT][255];
    *data = NULL;

    DIR *dir;
    struct dirent *finfo;
    dir = opendir(dataset_path);
    if (dir == NULL) {
        PSErr(__func__, "Invalid path %s", dataset_path);
        return 0;
    }

    char *prfx = (type == PS_DATA_TYPE_TRAINING ? "data_batch" : "test_batch");
    while ((finfo = readdir(dir))) {
        if (strstr(finfo->d_name, prfx) == NULL) continue;
        sprintf(datafiles[fcount++], "%s/%s", dataset_path, finfo->d_name);
    }
    if (fcount == 0) {
        PSErr(__func__, "Empty directory: %s", dataset_path);
        return 0;
    }
    qsort(datafiles, fcount, 255, compareFilenames);
    if (max_files > 0 && max_files < fcount) fcount = max_files;

    int datasize = (fcount * img_count * (classes + CIFAR_IMAGE_BYTESIZE));
    dataset_size = datasize * sizeof(PSFloat);
    *data = calloc(dataset_size, 1);
    if (*data == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSFloat *data_p = *data;
    for (i = 0; i < fcount; i++) {
        char *fname = datafiles[i];
        PSInfo("Reading %s", fname);
        FILE *f = fopen(fname, "r");
        if (f == NULL) {
            PSErr(__func__, "Could not open file %s", fname);
            free(*data);
            *data = NULL;
            return 0;
        }
        fseek(f, 0, SEEK_END);
        long pos = ftell(f);
        if (pos < expected_fsize) {
            PSErr(
                __func__, "Invalid file size: %d != %d (expected)", pos,
                expected_fsize
            );
            free(*data);
            *data = NULL;
            fclose(f);
            return 0;
        }
        fseek(f, 0, SEEK_SET);
        for (j = 0; j < img_count; j++) {
            int label = 0;
            if (classes == 10) {
                label = fgetc(f);
            } else {
                label = 100 * fgetc(f);
                label += fgetc(f);
            }
            /* printf("Label: %d\n", label); */
            for (k = 0; k < CIFAR_IMAGE_BYTESIZE; k++) {
                float b = (float)(fgetc(f)) / 255.0f;
                /* printf("%d ", b); */
                *(data_p++) = (PSFloat) b;
            }
            for (k = 0; k < classes; k++) {
                PSFloat y = (k == label ? 1.0 : 0.0);
                *(data_p++) = y;
            }
        }
        fclose(f);
    }
    (void) closedir (dir);
    return datasize;
}

PSLayer *PSAddCIFARInputLayer(PSModel *model) {
    if (model->size > 0) {
        PSErr(__func__, "CIFAR layer must be input layer!\n");
        return NULL;
    }
    PSLayerDef ldef = {
        .output_depth = 3, .output_columns = 32, .output_rows = 32
    };
    return PSAddLayer(model, FullyConnected, PS_CIFAR_IMAGE_SIZE, &ldef);
}

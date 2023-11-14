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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <sys/types.h>
#include <dirent.h>
#include <zlib.h>

#include "dataset.h"
#include "psyc.h"
#include "utils.h"
#include "log.h"
#include "platform.h"

#define PS_PARSER_BUFFER_SIZE 4096

/* Forward declarations */
uint32_t swap_uint32(uint32_t val);

/**** PSVocabulary ****/

/* Create a `PSVocabulary` with initial capacity of `initial_capacity`.
 * Return value: the vocabulary or NULL if memory cannot be allocated. */
PSVocabulary *PSVocabularyCreate(int64_t initial_capacity) {
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
int64_t PSVocabularyAdd(PSVocabulary *vocabulary, char *token) {
    if (token == NULL) return PS_INVALID_TOKEN_ID;
    int64_t id = vocabulary->token_map->length, size;
    PSDict *token_map = vocabulary->token_map;
    /* Set id as value for token into token_map dictionary. Since token_map
     * has flag PSDICT_UPDATE_DISABLED enabled, value won't be updated if it
     * already exists for token, so, in this case, existing value will be
     * returned. */
    PSDictItem *item = PSDictSet(token_map, token, PSDictFromInt(id));
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
        int64_t added_size = size - vocabulary->capacity;
        char **added = tokens + vocabulary->capacity;
        memset(added, 0, added_size * sizeof(char *));
        vocabulary->tokens = (const char**) tokens;
        vocabulary->capacity = size;
    }
    vocabulary->size = size;
    vocabulary->tokens[id] = item->key;
    return id;
}

/* Get the ID the token `token` from vocabulary `vocabulary`.
 * Return value: the numeric index (ID) of the token. If token is not found
 * into `vocabulary`, the function will return `PS_TOKEN_NOT_FOUND`.
 * If `vocabulary` is NULL or `token` is NULL, the function will return
 * `PS_INVALID_TOKEN_ID`. */
int64_t PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token) {
    if (vocabulary == NULL || token == NULL || vocabulary->token_map == NULL)
        return PS_INVALID_TOKEN_ID;
    PSDictItem *item = PSDictGet(vocabulary->token_map, token);
    if (item == NULL) return PS_TOKEN_NOT_FOUND;
    return item->value.as_int;
}

/* Get the token associated with `id` from `vocabulary`.
 * Return value: the token associated with `id` or NULL if no `token` is found
 * with `id`. Also return NULL if `vocabulary` is NULL. */
const char *PSVocabularyGetTokenByID(PSVocabulary *vocabulary, int64_t id) {
    if (vocabulary == NULL) return NULL;
    if (id >= vocabulary->size || vocabulary->tokens == NULL) return NULL;
    return vocabulary->tokens[id];
}

const char *PSVocabularyErrorString(int err) {
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
}

/**** Text Datasets ****/

char *PSNormalizeToken(char *token, int len) {
    for (int i = 0; i < len; i++) token[i] = tolower(token[i]);
    return token;
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
 *  - `datalen`: pointer to `uint64_t` where the final length of the dataset
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
PSFloat *PSLoadDataFromString(char *str, PSTextParserOptions *opts,
                              PSFloat *existing_data, int64_t *datalen,
                              PSVocabulary **vocabulary)
{
    static PSTextParserOptions default_opts = {0};
    char *tmpstr = NULL;
    int new_vocab = 0;
    PSVocabulary *vocab = NULL;
    PSFloat *data = NULL;
    if (datalen == NULL) {
        PSErr(__func__, "Missing mandatory argument `datalen`");
        goto fail;
    }
    if (vocabulary == NULL) {
        PSErr(__func__, "Missing mandatory argument `vocabulary`");
        goto fail;
    }
    if (str == NULL) {
        if (existing_data == NULL) *datalen = 0;
        return NULL;
    }
    int len = strlen(str);
    if (len == 0) {
        if (existing_data == NULL) *datalen = 0;
        return NULL;
    }
    int last_idx = len - 1;
    if (opts == NULL) opts = &default_opts;
    int do_normalize = !(opts->flags & PS_PARSER_FLAG_NO_NORMALIZATION),
        read_only = (opts->flags & PS_PARSER_FLAG_READONLY_VOCAB),
        char_mode = (opts->mode == PS_PARSER_MODE_CHARS);
    PSTokenNormalizer normalize = NULL;
    if (do_normalize) {
        normalize = opts->normalizer;
        if (normalize == NULL) normalize = PSNormalizeToken;
    }
    if (opts->flags & PS_PARSER_FLAG_PRESERVE_STRING) {
        tmpstr = strdup(str);
        if (tmpstr == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        str = tmpstr;
    }
    int64_t max_vocab_size = opts->max_vocabulary_size;
    if (max_vocab_size <= 0) max_vocab_size = PS_DEFAULT_MAX_VOCAB_SIZE;
    PSTokenMatch match_token = opts->match_token;
    const char *separator = opts->separator;
    if (separator == NULL && !char_mode)
        separator = PS_DEFAULT_TOKEN_SEPARATOR;
    int64_t capacity = 0, count = 0;
    vocab = *vocabulary;
    if (vocab != NULL) capacity = vocab->capacity;
    else {
        new_vocab = 1;
        capacity = opts->capacity;
        if (capacity <= 0) capacity = PS_DEFAULT_PARSER_CAPACITY;
        vocab = PSVocabularyCreate(capacity);
        if (vocab == NULL) goto fail;
        *vocabulary = vocab;
    }
    if (existing_data == NULL) {
        data = malloc(capacity * sizeof(PSFloat));
        if (data == NULL) goto memerr;
        count = 0;
    } else {
        data = existing_data;
        count = *datalen;
    }
    int64_t current_capacity = capacity;
    char *token = str, *p = str, *sep_p = NULL;
    char ctoken[2] = {0};
    int do_free_token = 0;
    while ((p - str) <= (long) last_idx) {
        size_t wlen = 0;
        if (*p == 0) break;
        if (char_mode) {
            ctoken[0] = *p++;
            token = ctoken;
            if (separator && strchr(separator, ctoken[0])) continue;
            wlen = 1;
            goto add_to_vocab;
        }
        if (match_token != NULL) {
            int matched = match_token(p, (int *) &wlen);
            if (!matched || wlen <= 0) {
                p++;
                continue;
            }
            token = malloc(wlen + 1);
            if (token == NULL) {
                PSErr(__func__, "could not allocate token of length: %d",wlen);
                goto memerr;
            }
            memcpy(token, p, wlen);
            token[wlen] = '\0';
            do_free_token = 1;
            p += wlen;
        } else {
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
        if (do_normalize) {
            char *normalized = normalize(token, wlen);
            if (normalized == NULL) {
                PSErr(__func__, "could not normalize token '%s'", token);
                if (do_free_token) free(token);
                token = NULL;
                goto fail;
            } else if (normalized != token) {
                if (do_free_token) free(token);
                token = normalized;
                do_free_token = 1;
            }
        }
        int64_t id = -1;
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
        if (do_free_token) free(token);
        if (id < 0) {
            if (read_only) {
                PSErr(__func__, "Failed to add vocabulary: %s",
                      PSVocabularyErrorString(id));
            } else {
                PSErr(__func__, "Token not found: %s",
                      PSVocabularyErrorString(id));
            }
            goto fail;
        }
        PSFloat token_id = (PSFloat) id;
        if (++count >= current_capacity) {
            current_capacity += capacity;
            PSFloat *new_data = realloc(
                data, current_capacity * sizeof(PSFloat)
            );
            if (new_data == NULL) goto memerr;
            data = new_data;
        }
        data[count - 1] = token_id;
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

/* Load a dataset (an array of PSFloat numbers) from the text file found at
 * `filepath`.
 * For parsing options and other arguments, see `PSLoadDataFromString`.
 * Return value: the dataset (`PSFloat` array) or NULL is something goes
 * wrong. */
PSFloat *PSLoadDataFromTextFile(const char *filepath,
                                PSTextParserOptions *opts,
                                int64_t *datalen,
                                PSVocabulary **vocabulary)
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
    int64_t capacity = 0;
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
    size_t nread = 0, buflen = sizeof(buf) - 1;
    *datalen = 0;
    int err = 0;
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
                    PSErr(__func__, "Word is too long (file: '%s', "
                          "offset: %lld)", filepath, ftello(file) - nread);
                    goto fail;
                }
            }
            if (idx < last_idx) {
                /* Truncate buffer to index of last separator found and
                 * reset file stream offset to first byte after truncation. */
                buf[idx] = '\0';
                int truncated_len = (int) (last_idx - idx);
                fseek(file, -truncated_len, SEEK_CUR);
                last_idx = idx;
            }
        }
        data = PSLoadDataFromString(
            buf, opts, data, datalen, &vocab
        );
        if (data == NULL) goto fail;
    }
    if (err != 0) {
        PSErr(__func__, "Error while reading file '%s' (%d): '%s'",
              filepath, err, strerror(err));
        goto fail;
    }
final:
    fclose(file);
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
    if (buffer == NULL) return;
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

int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data)
{
    char tmpImagesFileName[255] = {0};
    char tmpLabelsFileName[255] = {0};
    char *prefixImg = NULL, *prefixLbl = NULL;
    int data_len = 0, err;
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO);
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
        data = NULL;
        return 0;
    }
    FILE *labels = fopen(labels_file, "r");
    if (labels == NULL) {
        PSErr(__func__, "Cannot open labels '%s'", labels_file);
        data = NULL;
        fclose(images);
        return 0;
    }
    FILE *tmpimages = fopen(tmpImagesFileName, "w");
    FILE *tmplabels = fopen(tmpLabelsFileName, "w");
    if (do_log) printf("Loading images...\n");
    err = decompressGZip(images, tmpimages);
    if (err) {zerr(err); data = NULL; goto final;}
    if (do_log) printf("Loading labels...\n");
    err = decompressGZip(labels, tmplabels);
    if (err) {zerr(err); data = NULL; goto final;}
    fclose(tmpimages);
    fclose(tmplabels);
    tmpimages = fopen(tmpImagesFileName, "r");
    tmplabels = fopen(tmpLabelsFileName, "r");
    fseek(tmpimages, 0, SEEK_SET);
    fseek(tmplabels, 0, SEEK_SET);
    uint32_t magic_num = 0, image_count = 0, label_count = 0;
    int i = 0, j = 0, do_swap = !PS_IS_BIG_ENDIAN;
    fread(&magic_num, 1, 4, tmpimages);
    if (do_swap) magic_num = swap_uint32(magic_num);
    if (magic_num != IMAGES_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for image file: %d", magic_num);
        data = NULL;
        goto final;
    }
    fread(&image_count, 1, 4, tmpimages);
    if (do_swap) image_count = swap_uint32(image_count);
    if (image_count == 0) {
        PSErr(__func__, "Image count is 0!");
        data = NULL;
        goto final;
    }
    if (do_log) printf("Found %d images.\n", image_count);
    fread(&magic_num, 1, 4, tmplabels);
    if (do_swap) magic_num = swap_uint32(magic_num);
    if (magic_num != LABELS_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for labels file: %d",magic_num);
        data = NULL;
        goto final;
    }
    fread(&label_count, 1, 4, tmplabels);
    if (do_swap) label_count = swap_uint32(label_count);
    if (label_count == 0) {
        PSErr(__func__, "Label count is 0!");
        data = NULL;
        goto final;
    }
    if (do_log) printf("Found %d labels.\n", label_count);
    if (label_count != image_count) {
        PSErr(__func__, "Image count and label count do not match!");
        data = NULL;
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
        PSErr(__func__, "Invalid image size!");
        data = NULL;
        goto final;
    }
    data_len = (img_area * image_count) + (label_count * 10);
    *data = malloc(data_len * sizeof(PSFloat));
    PSFloat *data_p = *data;
    for (i = 0; i < (int) image_count; i++) {
        if (do_log) printf("\rLoading image %d/%d", i + 1, image_count);
        for (j = 0; j < img_area; j++) {
            int pixel = fgetc(tmpimages);
            PSFloat d = (PSFloat) pixel / (PSFloat) 255;
            *data_p = d;
            data_p++;
        }
        int label = fgetc(tmplabels);
        /* printf("Label: %d", label); */
        for (j = 0; j < 10; j++) {
            *data_p = (j == label);
            data_p++;
        }
    }
    printf("\n");
final:
    if (images != NULL) fclose(images);
    if (labels != NULL) fclose(labels);
    if (tmpimages != NULL) fclose(tmpimages);
    if (tmplabels != NULL) fclose(tmplabels);
    remove(tmpImagesFileName);
    remove(tmpLabelsFileName);
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

int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_elements)
{
    if (classes != 10 && classes != 100) {
        PSErr(__func__, "Invalid classes %d: only 10 or 100 allowed.", classes);
        return 0;
    }
    int label_size = (classes == 100 ? 2 : 1);
    int img_count = CIFAR_FILE_IMG_COUNT;
    if (max_elements > 0) img_count = max_elements;
    int expected_fsize = (CIFAR_IMAGE_BYTESIZE + label_size) * img_count;
    int fcount = 0, dataset_size = 0, i, j, k;
    char datafiles[CIFAR_DATAFILE_COUNT][255];

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
    if (*data == NULL) return 0;
    PSFloat *data_p = *data;
    for (i = 0; i < fcount; i++) {
        char *fname = datafiles[i];
        printf("Reading %s\n", fname);
        FILE *f = fopen(fname, "r");
        if (f == NULL) {
            PSErr(__func__, "Could not open file %s", fname);
            free(*data);
            data = NULL;
            return 0;
        }
        fseek(f, 0, SEEK_END);
        int pos = ftell(f);
        if (pos < expected_fsize) {
            PSErr(
                __func__, "Invalid file size: %d != %d (expected)", pos,
                expected_fsize
            );
            free(*data);
            data = NULL;
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
    return dataset_size;
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

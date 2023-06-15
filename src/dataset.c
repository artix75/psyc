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

#define PS_DEFAULT_TOKEN_SEPARATOR  " ,.:!?-'\"\n\t\r"
#define PS_DEFAULT_PARSER_CAPACITY  50
#define PS_DEFAULT_MAX_VOCAB_SIZE   15000
#define PS_DEFAULT_UNKOWN_TOKEN     "<unknown>"
#define PS_PARSER_MODE_TOKENS  0
#define PS_PARSER_MODE_CHARS   1

#define PS_PARSER_BUFFER_SIZE 4096

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

int64_t PSVocabularyAdd(PSVocabulary *vocabulary, char *token) {
    if (token == NULL) return PS_INVALID_TOKEN_ID;
    int64_t id = vocabulary->token_map->length, size;
    PSDict *token_map = vocabulary->token_map;
    PSDictItem *item = PSDictSet(token_map, token, PSDictFromInt(id));
    if (item == NULL) return PS_INVALID_TOKEN_ID;
    id = item->value.as_int;
    size = token_map->length;
    if (vocabulary->tokens == NULL || size > vocabulary->capacity) {
        char **tokens = realloc(vocabulary->tokens, size * sizeof(char *));
        if (tokens == NULL) {
            PSPrintMemoryErrorMsg();
            PSDictDelete(token_map, token);
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

int64_t PSVocabularyGetTokenID(PSVocabulary *vocabulary, char *token) {
    if (vocabulary == NULL || token == NULL || vocabulary->token_map == NULL)
        return PS_INVALID_TOKEN_ID;
    PSDictItem *item = PSDictGet(vocabulary->token_map, token);
    if (item == NULL) return PS_TOKEN_NOT_FOUND;
    return item->value.as_int;
}

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

void PSVocabularyRelease(PSVocabulary *vocabulary) {
    if (vocabulary == NULL) return;
    free(vocabulary->tokens);
    PSDictRelease(vocabulary->token_map);
}

void PSNormalizeToken(char *token, int len) {
    for (int i = 0; i < len; i++) token[i] = tolower(token[i]);
}

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
    int buffer_size = opts->buffer_size;
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
    const char *separator = opts->separator;
    if (separator == NULL) separator = PS_DEFAULT_TOKEN_SEPARATOR;
    int64_t max_vocab_size = opts->max_vocabulary_size;
    if (max_vocab_size <= 0) max_vocab_size = PS_DEFAULT_MAX_VOCAB_SIZE;
    int capacity = opts->capacity;
    if (capacity <= 0) capacity = PS_DEFAULT_PARSER_CAPACITY;
    int current_capacity = capacity;
    vocab = PSVocabularyCreate(capacity);
    if (vocab == NULL) goto fail;
    *vocabulary = vocab;
    data = malloc(capacity * sizeof(PSFloat));
    if (data == NULL) goto memerr;
    file = fopen(filepath, "r");
    if (file == NULL) {
        PSErr(__func__, "Could not open file '%s'", filepath);
        goto fail;
    }
    size_t nread = 0, buflen = sizeof(buf) - 1;
    int err = 0, count = 0;
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
                buf[idx] = '\0';
                int truncated_len = (int) (last_idx - idx);
                fseek(file, -truncated_len, SEEK_CUR);
                last_idx = idx;
            }
        }
        char *token = buf, *p = buf, *sep_p = NULL;
        while ((p - buf) < (long) last_idx) {
            token = p;
            sep_p = strpbrk(p, separator);
            size_t wlen = 0;
            if (sep_p != NULL) {
                wlen = sep_p - p;
                *sep_p = '\0';
                p = sep_p + 1;
            } else {
                wlen = p - buf;
                p += (wlen + 1);
            }
            if (wlen == 0) continue;
            PSNormalizeToken(token, wlen);
            int64_t id = -1;
            if (vocab->size >= max_vocab_size) {
                id = PSVocabularyGetTokenID(vocab, token);
                if (id == PS_TOKEN_NOT_FOUND) {
                    token = (char *) opts->unkown_token;
                    if (token == NULL) token = PS_DEFAULT_UNKOWN_TOKEN;
                    id = PSVocabularyAdd(vocab, token);
                }
            } else id = PSVocabularyAdd(vocab, token);
            if (id < 0) {
                PSErr(__func__, "Failed to add vocabulary: %s",
                      PSVocabularyErrorString(id));
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
    if (vocabulary != NULL) *vocabulary = NULL;
    if (file != NULL) fclose(file);
    if (data != NULL) free(data);
    if (vocab != NULL) PSVocabularyRelease(vocab);
    return NULL;
}

/**** MNIST Dataset ****/

#define MNIST_CHUNK 16384
#define IMAGES_MAGIC_NUM 2051
#define LABELS_MAGIC_NUM 2049
#define le2be(x) ((x >> 24 & 0x000000FF) | \
    (x >> 8 & 0x0000FF00) | \
    (x << 8 & 0x00FF0000) | \
    (x << 24))

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
void zerr(int ret)
{
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
    FILE *urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x",
            prefix,
            (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3]);
    fclose(urand);
}

int PSLoadMNISTData(int type, const char *images_file, const char *labels_file,
                    PSFloat **data)
{
    char tmpImagesFileName[255];
    char tmpLabelsFileName[255];
    char *prefixImg;
    char *prefixLbl;
    int data_len = 0, err;
    int do_log = (PSLogLevel <= PSLOGLEVEL_INFO);
    if (type == DATA_TYPE_TRAINING) {
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
        PSErr(__func__, "Cannot open %s", images_file);
        data = NULL;
        return 0;
    }
    FILE *labels = fopen(labels_file, "r");
    if (labels == NULL) {
        PSErr(__func__, "Cannot open %s", labels_file);
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
    int i = 0, j = 0;
    fread(&magic_num, 1, 4, tmpimages);
    magic_num = le2be(magic_num);
    if (magic_num != IMAGES_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for image file: %d", magic_num);
        data = NULL;
        goto final;
    }
    fread(&image_count, 1, 4, tmpimages);
    image_count = le2be(image_count);
    if (image_count == 0) {
        PSErr(__func__, "Image count is 0!");
        data = NULL;
        goto final;
    }
    if (do_log) printf("Found %d images.\n", image_count);
    fread(&magic_num, 1, 4, tmplabels);
    magic_num = le2be(magic_num);
    if (magic_num != LABELS_MAGIC_NUM) {
        PSErr(__func__, "Invalid magic number for labels file: %d",magic_num);
        data = NULL;
        goto final;
    }
    fread(&label_count, 1, 4, tmplabels);
    label_count = le2be(label_count);
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
    rows = le2be(rows);
    cols = le2be(cols);
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

    char *prfx = (type == DATA_TYPE_TRAINING ? "data_batch" : "test_batch");
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

PSLayer *PSAddCIFARInputLayer(PSNeuralNetwork *network) {
    if (network->size > 0) {
        PSErr(__func__, "CIFAR layer must be input layer!\n");
        return NULL;
    }
    PSLayerDef ldef = {
        .output_depth = 3, .output_columns = 32, .output_rows = 32
    };
    return PSAddLayer(network, FullyConnected, CIFAR_IMAGE_SIZE, &ldef);
}

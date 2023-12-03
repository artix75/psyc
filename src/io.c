/*
 * Copyright (C) 2016-2023 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <inttypes.h>
#include <sys/utsname.h>
#include <stdint.h>
#include <errno.h>

#include "psyc.h"
#include "config.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "gru.h"
#include "dropout.h"
#include "normalization.h"
#include "attention.h"
#include "operator-layer.h"
#include "log.h"
#include "buildinfo.h"
#include "optimization.h"
#include "platform.h"

#define DATA_LAYER_MIN_ARGC 3
#define OPT_FLOAT_FORMAT_HEX (1 << 0)
#define OPT_FLOAT_FORMAT_DBL (1 << 1)
#define MODEL_TRAINING_DATA_SEP "------ training data ------\n"

#define CONV_PARAM_FEATURE_COUNT     0
#define CONV_PARAM_REGION_SIZE       1
#define CONV_PARAM_STRIDE            2
#define CONV_PARAM_INPUT_WIDTH       3
#define CONV_PARAM_INPUT_HEIGHT      4
#define CONV_PARAM_OUTPUT_WIDTH      5
#define CONV_PARAM_OUTPUT_HEIGHT     6
#define CONV_PARAM_PADDING           7
#define CONV_PARAM_USE_RELU          8

#define CONV_PARAMETER_COUNT 9

#define PS_BINARY_FTYPE_LAYER   0
#define PS_BINARY_FTYPE_MODEL   1
#define PS_BINARY_FTYPE_VECTOR  2
#define PS_BINARY_FTYPE_MATRIX  3

#define UNUSED(V) ((void) V)

PSOptimization optimizationsByIndex[] = {
    PSDefaultOptimization,
    PSAdamOptimization,
    PSAdaGradOptimization,
    PSAdaDeltaOptimization,
    PSWindowGradOptimization,
    PSNesterovOptimization
};

static int optimization_count = sizeof(optimizationsByIndex) /
                                sizeof(PSOptimization);

typedef struct PSModelFileHeader {
    char    git_sha1[9];
    int     git_dirty;
    char    git_branch[256];
    size_t  float_size;
    int     archbits;
    int     avx;
    int     accelerate;
    int     global_flags;
    time_t  time;
    char    sysname[256];
    char    sysvers[12];
    char    sysmachine[12];
    int     acceleration;
} PSModelFileHeader;

typedef struct PSBinaryFileHeader {
    int type;
    int float_size;
    int iee_754_conformity;
    int big_endian;
    char version[255];
} PSBinaryFileHeader;

PSTrainingOptions *PSGetModelTrainingOptions(PSModel *model);
int PSGetTrainingMemoryGradients(PSModel *model,
                                 PSGradient ***grads_p);
int initTrainingContext(PSModel *model, PSTrainingOptions *opts,
                        int mem_gradients_count);
int PSCompareVersion(const char* vers1, const char* vers2);
int getLossFunctionIndex(PSLossFunction function);
PSLossFunction getLossFunctionAtIndex(int index);
void PSPrintLayerInfo(PSLayer *layer);
const char *PSGetActivationName(PSActivationFunction func);
PSFloat *PSSetSequenceStart(PSModel *model, PSFloat *start, int len);
PSLayer *PSMakeLayerPlaceholder(int layer_index, int model_index);

uint16_t swap_uint16(uint16_t val) {
    return (val << 8) | (val >> 8 );
}

int16_t swap_int16(int16_t val) {
    return (val << 8) | ((val >> 8) & 0xFF);
}

uint32_t swap_uint32(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int32_t swap_int32(int32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | ((val >> 16) & 0xFFFF);
}

int64_t swap_int64(int64_t val) {
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
          ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) |
          ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | ((val >> 32) & 0xFFFFFFFFULL);
}

uint64_t swap_uint64(uint64_t val) {
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
          ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) |
          ((val >> 16) & 0x0000FFFF0000FFFFULL );
    return (val << 32) | (val >> 32);
}

uint16_t readUInt16(FILE *f, int swap) {
    if (f == NULL) return 0;
    uint16_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_uint16(val);
    return val;
}

uint16_t readInt16(FILE *f, int swap) {
    if (f == NULL) return 0;
    int16_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_int16(val);
    return val;
}

uint32_t readUInt32(FILE *f, int swap) {
    if (f == NULL) return 0;
    uint32_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_uint32(val);
    return val;
}

uint32_t readInt32(FILE *f, int swap) {
    if (f == NULL) return 0;
    int32_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_int32(val);
    return val;
}

uint64_t readUInt64(FILE *f, int swap) {
    if (f == NULL) return 0;
    uint64_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_uint64(val);
    return val;
}

uint64_t readInt64(FILE *f, int swap) {
    if (f == NULL) return 0;
    int64_t val = 0;
    int nread = fread(&val, sizeof(val), 1, f);
    if (nread < 1) return 0;
    if (swap) val = swap_int64(val);
    return val;
}

PSFloat string2float(char *str, int *valid) {
    char *endptr = NULL;
#ifdef PS_DOUBLE_PRECISION
    PSFloat num = strtod(str, &endptr);
#else
    PSFloat num = strtof(str, &endptr);
#endif
    if (valid != NULL) *valid = !isnan(num) && (endptr != str);
    return num;
}

int string2int(char *str, int *valid) {
    char *endptr = NULL;
    int num = (int) strtol(str, &endptr, 10);
    if (valid != NULL) *valid = (endptr != str);
    return num;
}

static PSLayer *layerByIndex(PSModel *model, PSModel *parent,
                             int nidx, int lidx)
{
    if (parent == NULL || PSIsModelChain(model))
        return PSGetLayerByIndex(model, lidx, nidx);
    int last_idx = 0;
    if (PSIsModelChain(parent)) {
        PSModel *tail = PSModelChainTail(parent);
        if (tail != NULL) last_idx = tail->index;
    }
    if (nidx > last_idx) return PSGetLayerByIndex(model, lidx, 0);
    else return PSGetLayerByIndex(parent, lidx, nidx);
}

static int getOptimizationIndex(PSOptimization optimization) {
    if (optimization == NULL) return 0;
    int idx = 0, i;
    for (i = 0; i < optimization_count; i++) {
        if (optimizationsByIndex[i] == optimization) {
            idx = i;
            break;
        }
    }
    return idx;
}

void printModelHeaderInfo(PSModelFileHeader *hdr) {
    printf(
        "Git: %s/%d (branch: '%s')\n",
        hdr->git_sha1, hdr->git_dirty, hdr->git_branch
    );
    printf("Host: %s %s %s\n", hdr->sysname, hdr->sysvers, hdr->sysmachine);
    printf("Arch. Bits: %d\n", hdr->archbits);
    printf("Double Precision: %s\n", (hdr->float_size == 8 ? "yes" : "no"));
    printf("AVX: %s\n", (hdr->avx ? "yes" : "no"));
    printf("Saved on: %s\n", ctime(&hdr->time));
}

static void loadErr(const char *fname, FILE *f, const char *fmt, ...) {
    PSLog(PSLOGLEVEL_ERROR, "ERROR: while loading file '%s'", fname);
    if (f != NULL) {
        off_t offset = ftello(f), chars = 0, last_line_offset = 0;
        char buf[25] = {0};
        fgets(buf, 25, f);
        fseeko(f, 0, SEEK_SET);
        int line = 1;
        while (chars++ < offset) {
            if (fgetc(f) == '\n') {
                line++;
                last_line_offset = chars;
            }
        }
        fseeko(f, offset, SEEK_SET);
        PSLog(
            PSLOGLEVEL_ERROR, " (offset: %ld, line: %d, col: %ld):\n", offset,
            line, (offset - last_line_offset) + 1
        );
        PSLog(PSLOGLEVEL_ERROR, "Near: '%s'\n", buf);
    } else PSLog(PSLOGLEVEL_ERROR, "\n");
    va_list args;
    va_start(args, fmt);
    PSVLog(PSLOGLEVEL_ERROR, fmt, args);
    va_end(args);
    PSLog(PSLOGLEVEL_ERROR, "\n");
}

static char *getFormatStringEnd(char *fmt, int *invalid) {
    assert(invalid != NULL);
    *invalid = 0;
    char *p = strrchr(fmt, '%');
    if (p == NULL) return NULL;
    if (p > fmt && *(p - 1) == '%') return p;
    char c = '\0', prev_char = '%';
    int is_chr_class = 0, has_size_modifier = 0, has_type = 0;
    while ((c = *(++p)) != '\0') {
        if (c == '*') {
            if (prev_char == '%') goto next_char;
            else return p;
        }
        if (c == '\0') return p;
        if (c == 'H' || c == 'L' || c == 'D') {
            if (has_type) return p;
            if (!has_size_modifier) {
                has_size_modifier = 1;
                goto next_char;
            } else {
                *invalid = 1;
                return NULL;
            }
        } else if (c == 'l') {
            if (has_type) return p;
            if (!has_size_modifier) {
                has_size_modifier = 1;
                goto next_char;
            } else if (prev_char != 'l') {
                *invalid = 1;
                return NULL;
            }
        } else if (c >= '0' && c <= '9') {
            if (has_type) return p;
            goto next_char;
        } else if (strchr("idouxXfFgGeEcsnp", c) != NULL) {
            if (has_type) return p;
            has_type = 1;
            goto next_char;
        } else if (c == '[') {
            if (has_type) return p;
            if (is_chr_class) {
                *invalid = 1;
                return NULL;
            } else {
                is_chr_class = 1;
                goto next_char;
            }
        } else if (c == ']') {
            if (is_chr_class) {
                has_type = 1;
                is_chr_class = 0;
                goto next_char;
            } else return p;
        }
        else if (is_chr_class) goto next_char;
        else return p;
next_char:
        prev_char = c;
    }
    return p;
}

static int scanFileVar(FILE *f, char *fmt, int match, int *matched,
                       va_list args)
{
    long initial_pos = ftell(f);
    long minlen = 0;
    int match_count = 0, success = 0;
    char *modp = strchr(fmt, '%');
    if (match > 0) {
        assert(modp != NULL);
        match_count = vfscanf(f, fmt, args);
        success = (match_count == match);
        if (!success) goto final;
        int invalid = 0;
        char *endstr = getFormatStringEnd(fmt, &invalid);
        if (endstr == NULL) {
            if (invalid) {
                fprintf(stderr, "WARN: invalid format string '%s'\n", fmt);
                return 0;
            }
            goto final;
        }
        int endlen = strlen(endstr);
        if (endlen == 0) goto final;
        long curpos = ftell(f);
        fseek(f, -endlen, SEEK_CUR);
        char buf[endlen + 1];
        buf[endlen] = '\0';
        int read = fread(buf, sizeof(char), (size_t) endlen, f);
        if (read != endlen) {
            success = 0;
            goto final;
        }
        success = (strcmp(buf, endstr) == 0);
        if (!success) goto final;
        fseek(f, curpos, SEEK_SET);
    } else {
        if (modp != NULL)
            fprintf(stderr, "\nFATAL: Invalid fmt '%s' (%s)\n", fmt, __func__);
        assert(modp == NULL);
        int unused = 0;
        match_count = fscanf(f, fmt, &unused);
        success = (match_count == 0);
        if (!success) goto final;
        if (modp != NULL) minlen = (long) ((modp - fmt) + 1);
        else minlen = (long) strlen(fmt);
        if (minlen == 0) goto final;
        long curpos = ftell(f);
        success = (curpos - initial_pos) >= (long) minlen;
    }
final:
    if (!success) fseek(f, initial_pos, SEEK_SET);
    if (matched != NULL) *matched = match_count;
    return success;
}

static int scanFile(FILE *f, char *fmt, int match, int *matched, ...) {
    int ok = 0;
    va_list args;
    va_start(args, matched);
    ok = scanFileVar(f, fmt, match, matched, args);
    va_end(args);
    return ok;
}

static int scanFileNoMatch(FILE *f, char *fmt) {
    return scanFile(f, fmt, 0, NULL);
}

/* Binary format */

/* Write PsyC binary header. Argument `type` can be:
   0 - Layer definition file
   1 - Model definition file

   Format specifications:

   BYTES                 DESCR                  CONTENT             TYPE
   --------------------------------------------------------------------------
   4                     Filler                 FFFFFFFF
   4                     Fixed String           PSYC
   1                     File Type              0 - Layer file
                                                1 - Model file
                                                2 - Data (PSFloat vector)
                                                3 - Data (PSMatrix)
   1                     Version Len                                unsigned
   [Version Len]         PsyC Version                               string
   1                     Size of PSFloat                            unsigned
   1                     Float IEEE 754 Conform. 0 = false          signed
                                                 1 = true
                                                 2 = maybe
   1                     Big Endian             0|1                 unsigned
*/

static int readBinaryFileHeader(FILE *f, PSBinaryFileHeader *hdr) {
    if (hdr == NULL || f == NULL) return 0;
    if (ftell(f) != 0) return 0;
    uint32_t filler = 0;
    char hdr_s[5] = {0};
    int nread = fread(&filler, sizeof(filler), 1, f);
    int valid_hdr = (nread == 1 && filler == 0xFFFFFFFF);
    if (!valid_hdr) {
        if (nread < 1 && !feof(f)) valid_hdr = 1;
        goto err;
    }
    nread = fread(hdr_s, 1, 4, f);
    if (nread < 4) {
        valid_hdr = !feof(f);
        goto err;
    }
    valid_hdr = (strcmp("PSYC", hdr_s) == 0);
    if (!valid_hdr) goto err;
    hdr->type = fgetc(f);
    if (hdr->type == EOF) goto err;
    int verslen = fgetc(f);
    if (verslen == EOF) goto err;
    if (verslen > 254) {
        PSErr(NULL, "invalid version string length");
        goto err;
    }
    hdr->version[0] = '\0';
    nread = fread(hdr->version, 1, verslen, f);
    if (nread < verslen) goto err;
    hdr->float_size = fgetc(f);
    if (hdr->float_size == EOF) goto err;
    int valid_sz = (
        hdr->float_size == sizeof(float) ||
        hdr->float_size == sizeof(double)
    );
    if (!valid_sz) {
        PSErr(NULL, "invalid PSFloat size %d", hdr->float_size);
        goto err;
    }
    hdr->iee_754_conformity = fgetc(f);
    if (hdr->iee_754_conformity == EOF) goto err;
    hdr->big_endian = fgetc(f);
    if (hdr->big_endian == EOF) goto err;
    return 1;
err:
    fseek(f, 0, SEEK_SET);
    if (valid_hdr) return 0;
    PSErr(NULL, "invalid PsyC binary file");
    return 0;
}

static int checkBinaryFile(PSBinaryFileHeader *hdr, const char *filepath) {
    int iee_754_conformity = PS_IEC_559;
    int ok = (
        (hdr->iee_754_conformity && iee_754_conformity) ||
        (!hdr->iee_754_conformity && !iee_754_conformity)
    );
    if (!ok) {
        loadErr(filepath, NULL, "float representation used in file does "
                "not match with executable float representation");
        return 0;
    }
    if (sizeof(PSFloat) != hdr->float_size) {
        loadErr(filepath, NULL, "floats in file have different size from "
                "psyc PSFloat size: %zu != %zu. Try to rebuild psyc with "
                "proper float size (ie. `make DOUBLE_PRECISION=%s)",
                (size_t)hdr->float_size, sizeof(PSFloat),
                (sizeof(PSFloat) == sizeof(float) ? "on" : "off"));
        return 0;
    }
    return 1;
}

static int writeBinaryFileHeader(FILE *f, int type) {
    if (f == NULL) return 0;
    if (ftell(f) != 0) return 0;
    static uint32_t hdr_i = 0xFFFFFFFF;
    static char *hdr_s = "PSYC";
    int success = 1;
    int nwritten = fwrite(&hdr_i, 4, 1, f);
    success = (nwritten == 1);
    if (!success) return 0;
    nwritten = fwrite(hdr_s, 1, 4, f);
    success = (nwritten == 4);
    if (!success) return 0;
    success = fputc(type, f) != EOF;
    if (!success) return 0;
    int verslen = strlen(PSYC_VERSION);
    if (verslen > UINT8_MAX) verslen = UINT8_MAX;
    success = fputc((uint8_t) verslen, f) != EOF;
    if (!success) return 0;
    nwritten = fwrite(PSYC_VERSION, 1, verslen, f);
    success = (nwritten == verslen);
    if (!success) return 0;
    success = fputc((uint8_t) sizeof(PSFloat), f) != EOF;
    if (!success) return 0;
    success = fputc((uint8_t) PS_IEC_559, f) != EOF;
    if (!success) return 0;
    success = fputc((uint8_t) PS_IS_BIG_ENDIAN, f) != EOF;
    if (!success) return 0;
    return success;
}

static int writeBinaryFloats(PSFloat *floats, uint64_t len, FILE *f) {
    if (f == NULL || floats == NULL) return 0;
    if (len == 0) return 1;
    int max_chunk_len = 4096 / sizeof(PSFloat);
    int nwritten = 0, totwritten = 0, remaining = len;
    errno = 0;
    PSFloat *floats_p = floats;
    while ((uint64_t) totwritten < len) {
        int buflen = (remaining >= max_chunk_len ? max_chunk_len : remaining);
        nwritten = fwrite(floats_p, sizeof(PSFloat), buflen, f);
        totwritten += nwritten;
        remaining -= nwritten;
        if (nwritten < buflen) {
            char *err = "an error occurred";
            if (errno > 0) err = strerror(errno);
            PSErr(NULL, "failed to write binary floats: %s", err);
            return 0;
        }
        floats_p += nwritten;
    }
    return 1;
}

static int readBinaryFloats(PSFloat *floats, uint64_t len, FILE *f) {
    if (f == NULL || floats == NULL) return 0;
    int max_chunk_len = 4096 / sizeof(PSFloat);
    int nread = 0, totread = 0, remaining = len;
    errno = 0;
    PSFloat *floats_p = floats;
    while ((uint64_t) totread < len) {
        int buflen = (remaining >= max_chunk_len ? max_chunk_len : remaining);
        nread = fread(floats_p, sizeof(PSFloat), buflen, f);
        totread += nread;
        remaining -= nread;
        if (nread < buflen) {
            char *err = "an error occurred";
            if (errno > 0) err = strerror(errno);
            PSErr(NULL, "failed to read binary floats: %s", err);
            return 0;
        }
        floats_p += nread;
    }
    return 1;
}

/* Format: LEN[8] + VECTOR[LEN * sizef(PSFloat)] */
static int writeBinaryFloatArray(PSFloat *floats, uint64_t len, FILE *f) {
    if (f == NULL) return 0;
    if (floats == NULL) len = 0;
    int nwritten = fwrite(&len, sizeof(len), 1, f);
    if (nwritten < 1) return 0;
    if (floats == NULL) return 1;
    return writeBinaryFloats(floats, len, f);
}

PSFloat *readBinaryFloatArray(PSFloat *floats, uint64_t *len, FILE *f) {
    assert(len != NULL);
    if (f == NULL) return NULL;
    *len = 0;
    int nread = fread(&len, sizeof(len), 1, f);
    if (nread < 1) return NULL;
    PSFloat *allocd = NULL;
    if (floats == NULL) {
        allocd = calloc(*len, sizeof(PSFloat));
        if (allocd == NULL) {
            PSPrintMemoryErrorMsg();
            return NULL;
        }
        floats = allocd;
    }
    if (!readBinaryFloats(floats, *len, f)) {
        floats = NULL;
        free(allocd);
    }
    return floats;
}

/* Format: LEN[8] + MATRIX[LEN * sizef(PSFloat)] */
static int writeBinaryMatrixAsVector(PSMatrix matrix, FILE *f) {
    return writeBinaryFloatArray(matrix, PSMatrixLength(matrix), f);
}

/* Format:
    SHAPELEN[1] + SHAPE[SHAPELEN] + LEN[8] + MATRIX[LEN * sizef(PSFloat)]
*/
int writeBinaryMatrix(PSMatrix matrix, FILE *f) {
    if (f == NULL) return 0;
    if (matrix == NULL) return fputc(0, f) != EOF;
    int shape[3];
    int shape_len = PSMatrixShape(matrix, shape), i;
    if (fputc((uint8_t) shape_len, f) == EOF) return 0;
    for (i = 0; i < shape_len; i++) {
        if (fputc((uint8_t) shape[i], f) == EOF) return 0;
    }
    return writeBinaryMatrixAsVector(matrix, f);
}

/* Format: HEADER (see writeBinaryFileHeader) +
           INDEX[4] + BIAS_COUNT[8] + BIAS[BIAS_COUNT * sizeof(PSFloat)] +
           WEIGHT_TYPES[1] + TOT_WEIGHT_COUNT[8] +
           For each weight matrix:
           WEIGHT_COUNT[8] + WEIGHTS[WEIGHT_COUNT * sizeof(PSFloat)] */
static int writeBinaryLayerParameters(PSLayer *layer, int opts, FILE *f,
                                      const char*func)
{
    UNUSED(opts);
    if (layer == NULL || f == NULL) return 0;
    PSLayerType ltype = layer->type;
    if (Pooling == ltype || layer->type == Dropout) return 0;
    uint64_t bias_count = PSGetLayerParametersCount(layer, PS_PARAM_BIAS),
             weights_count = PSGetLayerParametersCount(layer,PS_PARAM_WEIGHT);
    int i = layer->index, j;
    if (!writeBinaryFileHeader(f, PS_BINARY_FTYPE_LAYER)) return 0;
    int nwritten = fwrite(&layer->index, sizeof(layer->index), 1, f);
    if (nwritten < 1) return 0;
    nwritten = fwrite(&bias_count, sizeof(bias_count), 1, f);
    if (nwritten < 1) return 0;
    if (bias_count > 0) {
        if (layer->biases == NULL) {
            PSErr(func, "Layer[%d]: biases are NULL", i);
            return 0;
        }
        if (!writeBinaryFloats(layer->biases, bias_count, f)) {
            PSErrNN(func, NULL, layer, "could not write binary biases");
            return 0;
        }
    }
    int partial_trainable_parameters = (layer->type == Attention);
    if (fputc((uint8_t) layer->weight_types, f) == EOF) return 0;
    nwritten = fwrite(&weights_count, sizeof(weights_count), 1, f);
    if (nwritten < 1) return 0;
    if (layer->weight_types > 0) {
        if (layer->weights == NULL) {
            PSErr(func, "Layer[%d]: weights are NULL", i);
            return 0;
        }
        for (j = 0; j < layer->weight_types; j++) {
            PSMatrix weights = layer->weights[j];
            int ok = (weights != NULL || partial_trainable_parameters);
            if (!ok) {
                PSErr(func, "Layer[%d]: weights[%d] are NULL", i, j);
                return 0;
            }
            uint64_t len = 0;
            if (weights == NULL) {
                nwritten = fwrite(&len, sizeof(len), 1, f);
                if (nwritten < 0) return 0;
                continue;
            }
            len = PSMatrixLength(weights);
            nwritten = fwrite(&len, sizeof(len), 1, f);
            if (nwritten < 0) return 0;
            if (!writeBinaryFloats(weights, len, f)) {
                PSErrNN(func, NULL, layer, "could not write binary "
                        "weights[%d]", j);
                return 0;
            }
        }
    }
    return 1;
}

static int loadBinaryLayerParameters(PSLayer *layer, const char *filepath,
                                     FILE *f, int check_index, int verbose)
{
    if (layer->type == Pooling || layer->type == Dropout) return 0;
    errno = 0;
    int lidx = -1, wtype_count = 0, ok = 1,
        i = layer->index;
    uint64_t bias_count = 0, wcount = 0;
    PSBinaryFileHeader hdr = {0};
    ok = readBinaryFileHeader(f, &hdr);
    if (!ok) {
        loadErr(filepath, f, "file is not a valid psyc binary file");
        return 0;
    }
    if (!checkBinaryFile(&hdr, filepath)) return 0;
    int do_swap = (
        (hdr.big_endian && !PS_IS_BIG_ENDIAN) ||
        (!hdr.big_endian && PS_IS_BIG_ENDIAN)
    );
    lidx = readUInt32(f, do_swap);
    if (errno > 0) goto read_err;
    ok = (!check_index || i == lidx);
    if (!ok) {
        loadErr(
            filepath, f, "Invalid layer index %u, expected: %llu", lidx, i
        );
        return 0;
    }
    uint64_t expected_bias_count = PSGetLayerParametersCount(
        layer, PS_PARAM_BIAS
    );
    uint64_t expected_weights_count = PSGetLayerParametersCount(
        layer, PS_PARAM_WEIGHT
    );
    bias_count = readUInt64(f, do_swap);
    if (errno > 0) goto read_err;
    ok = (bias_count == expected_bias_count);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%u]: found %llu biases, expected: %llu",
            i, bias_count, expected_bias_count
        );
        return 0;
    }
    if (layer->biases == NULL && bias_count > 0) {
        loadErr(
            filepath, f, "Layer[%u]: found %llu biases, but "
            "layer->biases is NULL", i, bias_count
        );
        return 0;
    }
    if (bias_count > 0 && layer->biases != NULL)
        ok = readBinaryFloats(layer->biases, bias_count, f);
    if (!ok) {
        loadErr(filepath, f, "could not load biases");
        return 0;
    }
    wtype_count = fgetc(f);
    if (wtype_count == EOF) goto read_err;
    ok = (wtype_count == layer->weight_types);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%u]: found %d weight types, "
            "expected: %d", i, wtype_count, layer->weight_types
        );
        return 0;
    }
    wcount = readUInt64(f, do_swap);
    if (errno > 0) goto read_err;
    ok = (wcount == expected_weights_count);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%d]: found %llu weights, expected: %llu",
            i, wcount, expected_weights_count
        );
        return 0;
    }
    if (layer->weights == NULL && wtype_count > 0) {
        loadErr(
            filepath, f, "Layer[%u]: found %llu weights, but "
            "layer->weights is NULL", i, wcount
        );
        ok = 0;
        return 0;
    }
    int partial_trainable_parameters = (layer->type == Attention);
    for (int j = 0; j < layer->weight_types; j++) {
        PSMatrix weights = layer->weights[j];
        uint64_t wlen = readUInt64(f, do_swap);
        if (errno > 0) goto read_err;
        ok = (weights != NULL);
        if (!ok && partial_trainable_parameters) ok = wlen == 0;
        if (!ok) {
            loadErr(filepath, NULL, "Layer[%d]: weights[%d] is NULL", i, j);
            return 0;
        }
        if (wlen == 0 && weights == NULL) continue;
        ok = (wlen == PSMatrixLength(weights));
        if (!ok) {
            loadErr(filepath, NULL, "Layer[%u]: weights[%d] size expected to "
                    "be %llu, but files states %llu", i, j,
                    PSMatrixLength(weights), wlen);
            return 0;
        }
        ok = readBinaryFloats(weights, wlen, f);
        if (!ok) {
            loadErr(filepath, f, "could not load weights[%d]", j);
            return 0;
        }
    }
    if (verbose) {
        int llen = printf("\rLayer[%d]: Loaded", i);
        PSFillWithBlank(llen - 1);
        printf("\n");
        PSPrintLayerInfo(layer);
    }
    return ok;
read_err:
    if (errno > 0)
        loadErr(filepath, f, "failed to read file: %s", strerror(errno));
    else loadErr(filepath, f, "failed to read file");
    return 0;
}

PSFloat *loadBinaryVector(const char *filepath, FILE *f, uint64_t *len) {
    assert(len != NULL);
    errno = 0;
    PSBinaryFileHeader hdr = {0};
    if (!readBinaryFileHeader(f, &hdr)) {
        loadErr(filepath, f, "file is not a valid psyc binary file");
        return 0;
    }
    if (!checkBinaryFile(&hdr, filepath)) return 0;
    if (hdr.type != PS_BINARY_FTYPE_VECTOR) {
        PSErr(NULL, "file '%s' is not a vector binary file");
        return 0;
    }
    int do_swap = (
        (hdr.big_endian && !PS_IS_BIG_ENDIAN) ||
        (!hdr.big_endian && PS_IS_BIG_ENDIAN)
    );
    *len = readUInt64(f, do_swap);
    if (*len == 0) return NULL;
    uint64_t maxsize = SIZE_MAX;
    if (*len > maxsize) {
        PSErr(
            NULL, "vector size from file '%s' exceeds maximum size",
            filepath
        );
        return 0;
    }
    PSFloat *data = malloc((size_t) len * sizeof(PSFloat));
    if (data == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    if (!readBinaryFloats(data, *len, f)) {
        PSErr(NULL, "failed to load vector from binary file: %s", filepath);
        free(data);
        data = NULL;
    }
    return data;
}

int saveBinaryVector(FILE *f, PSFloat *vec, uint64_t len) {
    assert(f != NULL);
    if (!writeBinaryFileHeader(f, PS_BINARY_FTYPE_VECTOR)) return 0;
    if (!writeBinaryFloatArray(vec, len, f)) return 0;
    return 1;
}

/* Scan model file header (version >= 0.3) until new line (included). */
static int scanModelFileHeader(FILE *f, PSModelFileHeader *header,
                               const char *fname)
{
    assert(header != NULL);
    int ok = 0;
    char propname[255];
    char val[4096];
    propname[0] = '\0';
    val[0] = '\0';
    char sep[2];
    sep[0] = '\0';
    while ((ok = scanFile(f, "%254[a-zA-Z_-]=", 1, NULL, propname))) {
        sep[0] = '\0';
        ok = scanFile(f, "%4095[^;\n]%1[;\n]", 2, NULL, val, sep);
        if (!ok) goto fail;
        if (sep[0] != ';' && sep[0] != '\n') goto fail;
        if (strcmp("git", propname) == 0) {
            int matched = sscanf(
                val, "%8s/%d-%255s", header->git_sha1, &(header->git_dirty),
                header->git_branch
            );
            if (matched != 3) goto fail;
        } else if (strcmp("float_size", propname) == 0) {
            header->float_size = atoi(val);
            if (header->float_size <= 0 || (header->float_size % 4) != 0)
                goto fail;
        } else if (strcmp("archbits", propname) == 0) {
            header->archbits = atoi(val);
            if (header->archbits <= 0 || (header->archbits % 8) != 0)
                goto fail;
        } else if (strcmp("avx", propname) == 0) {
            header->avx = atoi(val);
            if (header->avx < 0 || header->avx > 1) goto fail;
        } else if (strcmp("accelerate", propname) == 0) {
            header->accelerate = atoi(val);
            if (header->accelerate < 0 || header->accelerate > 1) goto fail;
        } else if (strcmp("acceleration", propname) == 0) {
            header->acceleration = atoi(val);
            if (header->acceleration < 0) goto fail;
        } else if (strcmp("sys", propname) == 0) {
            int matched = sscanf(
                val, "%256[^,],%12[^,],%12s", header->sysname,
                header->sysvers, header->sysmachine
            );
            if (matched != 3) goto fail;
        } else if (strcmp("global_flags", propname) == 0) {
            header->global_flags = atoi(val);
            if (header->global_flags < 0) goto fail;
        } else if (strcmp("savetime", propname) == 0) {
            header->time = (time_t) strtol(val, NULL, 10);
            if (header->time == 0) goto fail;
        } else {
            PSWarn(
                "unknown header property `%s` in file '%s'",
                propname, fname
            );
        }
    }
    if (!ok && sep[0] == '\n') ok = 1;
    return ok;
fail:
    PSErr(
        NULL, "Invalid value for property '%s' in file '%s': '%s'",
        propname, fname, val
    );
    return 0;
}

static int scanTrainingOptions(FILE *f, PSTrainingOptions *opts,
                               const char *fname)
{
    assert(opts != NULL);
    int ok = 0;
    char propname[255];
    char val[4096];
    propname[0] = '\0';
    val[0] = '\0';
    char sep[2];
    sep[0] = '\0';
    while ((ok = scanFile(f, "%254[a-zA-Z0-9_-]=", 1, NULL, propname))) {
        sep[0] = '\0';
        ok = scanFile(f, "%4095[^,\n]%1[,\n]", 2, NULL, val, sep);
        if (!ok) goto fail;
        if (sep[0] != ',' && sep[0] != '\n') goto fail;
        if (strcmp("flags", propname) == 0) {
            int flags = string2int(val, &ok);
            if (!ok) goto fail;
            opts->flags = flags;
        } else if (strcmp("bptt_truncate", propname) == 0) {
            int bptt_truncate = string2int(val, &ok);
            if (!ok || bptt_truncate < 0) goto fail;
            opts->bptt_truncate = bptt_truncate;
        } else if (strcmp("optimization", propname) == 0) {
            int optimization_idx = string2int(val, &ok);
            if (!ok || (int) optimization_idx < 0) goto fail;
            if (optimization_idx >= optimization_count) goto fail;
            opts->optimization = optimizationsByIndex[optimization_idx];
        } else if (strcmp("l1_decay", propname) == 0) {
            PSFloat l1_decay = string2float(val, &ok);
            if (!ok) goto fail;
            opts->l1_decay = l1_decay;
        } else if (strcmp("l2_decay", propname) == 0) {
            PSFloat l2_decay = string2float(val, &ok);
            if (!ok) goto fail;
            opts->l2_decay = l2_decay;
        } else if (strcmp("momentum", propname) == 0) {
            PSFloat momentum = string2float(val, &ok);
            if (!ok) goto fail;
            opts->momentum = momentum;
        } else if (strcmp("rho", propname) == 0) {
            PSFloat rho = string2float(val, &ok);
            if (!ok) goto fail;
            opts->rho = rho;
        } else if (strcmp("eps", propname) == 0) {
            PSFloat eps = string2float(val, &ok);
            if (!ok) goto fail;
            opts->eps = eps;
        } else if (strcmp("beta1", propname) == 0) {
            PSFloat beta1 = string2float(val, &ok);
            if (!ok) goto fail;
            opts->beta1 = beta1;
        } else if (strcmp("beta2", propname) == 0) {
            PSFloat beta2 = string2float(val, &ok);
            if (!ok) goto fail;
            opts->beta2 = beta2;
        } else if (strcmp("clip", propname) == 0) {
            PSFloat clip = string2float(val, &ok);
            if (!ok) goto fail;
            opts->clip = clip;
        } else {
            PSWarn("unknown header property `%s` in file '%s'",propname,fname);
        }
    }
    if (!ok && sep[0] == '\n') ok = 1;
    return ok;
fail:
    PSErr(
        NULL, "Invalid value for training option '%s' in file '%s': '%s'",
        propname, fname, val
    );
    return 0;
}

int writeSerializedFloat(FILE *out, PSFloat fnum, int opts) {
    if (opts & OPT_FLOAT_FORMAT_DBL)
        return fprintf(out, "%.*g", DBL_DECIMAL_DIG, (double) fnum);
    else if (opts & OPT_FLOAT_FORMAT_HEX) return fprintf(out, "%a", fnum);
    else return fprintf(out, "%.*g", PSFLOAT_DIG, fnum);
}

int writeSerializedFloats(FILE *out, uint64_t count, char *sep, int opts, ...)
{
    int has_sep = (sep != NULL), len = 0;
    uint64_t i;
    va_list args;
    va_start(args, opts);
    for (i = 0; i < count; i++) {
        if (has_sep && i > 0) len += fprintf(out, "%s", sep);
        PSFloat fnum = (PSFloat) va_arg(args, double);
        len += writeSerializedFloat(out, fnum, opts);
    }
    va_end(args);
    return len;
}

int writeSerializedFloatArray(FILE *out, uint64_t count, char *sep, int opts,
                              PSFloat *array)
{
    int has_sep = (sep != NULL), len = 0;
    uint64_t i;
    for (i = 0; i < count; i++) {
        if (has_sep && i > 0) len += fprintf(out, "%s", sep);
        len += writeSerializedFloat(out, array[i], opts);
    }
    return len;
}

PSFloat *readSerializedFloatArray(FILE *in, char *sep, uint64_t *length,
                                  uint64_t maxlen, uint64_t capacity)
{
    PSFloat *array = NULL;
    if (length == NULL) {
        PSErr(__func__, "`length` cannot be NULL");
        return NULL;
    }
    *length = 0;
    if (capacity == 0) capacity = 1;
    if (sep == NULL) sep = ",";
    int seplen = strlen(sep);
    if (seplen == 0) {
        PSErr(__func__, "`sep` is empty");
        return NULL;
    }
    uint64_t arraylen = capacity;
    array = calloc(arraylen, sizeof(PSFloat));
    if (array == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    char fmt[256];
    char matched_sep[512] = {0};
    snprintf(fmt, 255, "%s%%%d[%s]", PSFLOAT_FORMAT, 511, sep);
    int matched = 0;
    PSFloat val = 0;
    while ((matched = fscanf(in, fmt, &val, matched_sep))) {
        uint64_t idx = *length;
        if (idx >= arraylen) {
            arraylen += capacity;
            PSFloat *new_array = realloc(array, arraylen * sizeof(PSFloat));
            if (new_array == NULL) {
                PSPrintMemoryErrorMsg();
                free(array);
                return NULL;
            }
            array = new_array;
        }
        array[idx] = val;
        *length += 1;
        if (matched < 2) break;
        if (maxlen > 0 && *length >= maxlen) break;
    }
    if (arraylen > *length) {
        memset(array + *length, 0, (arraylen - *length) * sizeof(PSFloat));
    }
final:
    return array;
}

int writeGradients(PSModel *model, PSGradient **gradients,
                   int opts, FILE *f)
{
    if (model == NULL || model->size == 0) return 0;
    for (int i = 1; i < model->size; i++) {
        PSGradient *gradient = gradients[i - 1];
        if (gradient == NULL) continue;
        if (gradient->bias_count > 0 && gradient->biases == NULL) {
            PSErr("PSModelSave", "invalid gradient biases");
            return 0;
        }
        if (gradient->weight_count > 0 && gradient->weights == NULL) {
            PSErr("PSModelSave", "invalid gradient weights");
            return 0;
        }
        fprintf(
            f, "--- Gradient[%d] Biases: %" PRIu64 " ---\n", i - 1,
            gradient->bias_count
        );
        if (gradient->bias_count > 0) {
            writeSerializedFloatArray(
                f, gradient->bias_count, ",", opts, gradient->biases
            );
            fprintf(f, "\n");
        }
        fprintf(
            f, "--- Gradient[%d] Weights: %" PRIu64 " ---\n", i - 1,
            gradient->weight_count
        );
        if (gradient->weight_count > 0) {
            writeSerializedFloatArray(
                f, gradient->weight_count, ",", opts, gradient->weights
            );
            fprintf(f, "\n");
        }
    }
    return 1;
}

int writeLayerDefinition(PSLayer *layer, FILE *f) {
    char *activation = (char *) PSGetActivationName(layer->activate);
    if (activation == NULL) activation = "null";
    fprintf(
        f, "layer[%d]:%d,%d,%d,activation=%s,"
        "output_depth=%d,output_cols=%d,output_rows=%d",
        layer->index, (int) layer->type, layer->size,
        layer->flags, activation, layer->output_depth,
        layer->output_columns, layer->output_rows
    );
    if (layer->flags & PS_FLAG_ONEHOT && layer->index == 0)
        fprintf(f, ",onehot_size=%d", layer->onehot_vector_size);
    if (Convolutional == layer->type || Pooling == layer->type) {
        int stride = 0, padding = 0, filter_w = 0, filter_h = 0;
        PSConvolutionalSettings *csettings = PSGetConvolutionalSettings(layer);
        if (csettings != NULL) {
            stride = csettings->stride;
            padding = csettings->padding;
            filter_w = csettings->filter_width;
            filter_h = csettings->filter_height;
        }
        fprintf(f, ",stride=%d,padding=%d,filter_width=%d,filter_height=%d",
                stride, padding, filter_w, filter_h);
    } else if (Dropout == layer->type) {
        PSFloat dropout = PSGetDropout(layer);
        fprintf(f, ",dropout=" PSFLOAT_FORMAT, dropout);
    } else if (Normalization == layer->type) {
        PSNormalizationLayerSettings *normsettings =
            PSGetNormalizationSettings(layer);
        PSFloat eps = PSDEFAULT_NORM_EPSILON;
        if (normsettings != NULL) eps = normsettings->epsilon;
        if (eps == 0) eps = PSDEFAULT_NORM_EPSILON;
        fprintf(f, ",epsilon=" PSFLOAT_FORMAT, eps);
    } else if (Attention == layer->type) {
        fprintf(f,",attention_type=%d,causal=%d,n_heads=%d,attention_scale=%g,"
                "enabled_projections=%d",
                PSGetAttentionType(layer), PSIsCausalAttention(layer),
                PSGetAttentionHeadCount(layer),
                PSGetAttentionScale(layer),
                PSGetAttentionEnabledProjections(layer));
        PSLayer *qprovider = NULL, *kprovider = NULL, *vprovider = NULL;
        PSGetAttentionProviders(layer, &qprovider, &kprovider, &vprovider);
        if (qprovider != NULL && qprovider->model != NULL) {
            fprintf(f, ",query_provider=%d:%d", qprovider->model->index,
                    qprovider->index);
        }
        if (kprovider != NULL && kprovider->model != NULL) {
            fprintf(f, ",keys_provider=%d:%d", kprovider->model->index,
                    kprovider->index);
        }
        if (vprovider != NULL && vprovider->model != NULL) {
            fprintf(f, ",values_provider=%d:%d", vprovider->model->index,
                    vprovider->index);
        }
    } else if (OperatorLayer == layer->type) {
        fprintf(f, ",operator=%d", PSGetOperatorLayerType(layer));
        int prvcount = 0;
        PSLayer **providers = PSGetOperatorLayerProviders(layer, &prvcount);
        fprintf(f, ",providers_count=%d", prvcount);
        if (providers != NULL && prvcount > 0) {
            fprintf(f, ",providers=");
            for (int i = 0; i < prvcount; i++) {
                PSLayer *provider = providers[i];
                if (provider == NULL) {
                    PSErrNN(NULL, NULL, layer, "provider[%d] is null", i);
                    return 0;
                }
                if (provider->model == NULL) return 0;
                fprintf(
                    f, "%s%d:%d", (i > 0 ? "-" : ""), provider->model->index,
                    provider->index
                );
            }
        }
    }
    if (layer->pretrain != NULL)
        fprintf(f, ",pretrained=%d", layer->pretrained);
    fprintf(f, "\n");
    return 1;
}

int writeLayerParameters(PSLayer *layer, int opts, FILE *f, const char *func) {
    PSLayerType ltype = layer->type;
    if (Pooling == ltype || layer->type == Dropout) return 0;
    int bias_count = PSGetLayerParametersCount(layer, PS_PARAM_BIAS),
        weights_count = PSGetLayerParametersCount(layer, PS_PARAM_WEIGHT);
    int i = layer->index, k, j;
    fprintf(f, "--- Layer[%d] Biases: %d ---\n", i, bias_count);
    if (bias_count > 0) {
        if (layer->biases == NULL) {
            PSErr(func, "Layer[%d]: biases are NULL", i);
            fclose(f);
            return 0;
        }
        for (j = 0; j < bias_count; j++) {
            if (j > 0) fprintf(f, ",");
            writeSerializedFloat(f, layer->biases[j], opts);
        }
        fprintf(f, "\n");
    }
    int partial_trainable_parameters = (layer->type == Attention);
    fprintf(f, "--- Layer[%d] Weights: %d,%d ---\n",
            i, layer->weight_types, weights_count);
    if (layer->weight_types > 0) {
        if (layer->weights == NULL) {
            PSErr(func, "Layer[%d]: weights are NULL", i);
            fclose(f);
            return 0;
        }
        for (j = 0; j < layer->weight_types; j++) {
            PSMatrix weights = layer->weights[j];
            int ok = (weights != NULL);
            if (!ok && partial_trainable_parameters) {
                fprintf(f, "---\n");
                continue;
            }
            if (!ok) {
                PSErr(func, "Layer[%d]: weights[%d] are NULL", i, j);
                fclose(f);
                return 0;
            }
            int ws = PSMatrixLength(weights);
            for (k = 0; k < ws; k++) {
                if (k > 0) fprintf(f, ",");
                writeSerializedFloat(f, weights[k], opts);
            }
            fprintf(f, "\n");
        }
    }
    return 1;
}

static int loadLegacyLayerDefinitions(PSModel *model, char *vers,
                                      int nlayers, int empty,
                                      const char *filepath, FILE *f)
{
    int min_argc = 1, i, ok;
    if (PSCompareVersion(vers, "0.2.2") == 1) min_argc = DATA_LAYER_MIN_ARGC;
    else if (PSCompareVersion(vers, "0.0.0") == 1) min_argc = 2;
    PSLayer *layer = NULL;
    for (i = 0; i < nlayers; i++) {
        int lsize = 0;
        int lflags = 0;
        PSFloat dropout = 0.0;
        PSLayerType ltype = FullyConnected;
        int args[20];
        int argc = 0, aidx = 0;
        char sep[2] = {0};
        /* Simple FullyConnected layers with no flags, no dropout and no
         * hyper-parameters are only saved as a single integer (layer->size) */
        /* fputs(fmt, stderr); */
        int matched = scanFile(f, "%d%1[,\n]", 2, NULL, &lsize, sep);
        if (!matched) {
            /* Try to parse more complex layer definitions declared as an
             * array of numeric values: [type, argc, args...]. */
            int type = 0, arg = 0;
            PSFloat argf = 0.0;
            argc = 0;
            ok = scanFile(f, "[%d,%d", 2, NULL, &type, &argc);
            if (!ok) {
                loadErr(filepath, f, "Invalid layer def: layer[%d]", i);
                return 0;
            }
            if (argc == 0) {
                loadErr(
                    filepath, NULL,
                    "Layer %d must have at least 1 argument (size)", i
                );
                ok = 0;
                return 0;
            }
            /* For backward compatibility, data here can be parsed in different
             * ways, and `min_argc` is used to indicate the minumum number of
             * fixed arguments, that can be:
             * - flags (min_argc == 2)
             * - flags, dropout (min_argc == 3)
             * The rest of the arguments is used in case of  eventual
             * PSHyperParameters. */
            ltype = (PSLayerType) type;
            for (aidx = 0; aidx < argc; aidx++) {
                if (min_argc == 3 && aidx == 2)
                    ok = scanFile(f, "," PSFLOAT_FORMAT, 1, NULL, &argf);
                else ok = scanFile(f, ",%d", 1, NULL, &arg);
                if (!ok) {
                    loadErr(
                        filepath, f,
                        "Invalid layer def: l%d, arg. %d",
                        i, aidx
                    );
                    return 0;
                }
                if (aidx == 0) lsize = arg;
                else if (min_argc > 1 && aidx == 1) lflags = arg;
                else if (min_argc > 2 && aidx == 2) dropout = argf;
                else {
                    int arg_aidx = aidx - min_argc;
                    if (arg_aidx >= 20) {
                        loadErr(filepath, f, "Argument is out-of-bounds");
                        return 0;
                    }
                    args[arg_aidx] = arg;
                }
            }
            argc -= min_argc;
            ok = scanFile(f, "]%1[,\n]", 1, NULL, sep);
            if (!ok) return 0;
        }
        if (!empty) {
            layer = model->layers[i];
            if (layer->size != lsize) {
                loadErr(filepath, NULL, "Layer %d size %d differs from %d!",
                        i, layer->size, lsize);
                return 0;
            }
            if (ltype != layer->type) {
                loadErr(filepath, NULL, "Layer %d type %d differs from %d!",
                        i, (int) (layer->type), (int) ltype);
                return 0;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                PSConvolutionalSettings *settings =
                    PSGetConvolutionalSettings(layer);
                if (settings == NULL) {
                    PSErr(
                        __func__, "Layer %d: missing convolutional settings", i
                    );
                    return 0;
                }
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= CONV_PARAMETER_COUNT) break;
                    int arg = args[aidx], val;
                    char *argname = NULL;
                    if (aidx == CONV_PARAM_FEATURE_COUNT) {
                        val = layer->output_depth;
                        argname = "output_depth";
                    } else if (aidx == CONV_PARAM_INPUT_WIDTH) {
                        val = settings->input_width;
                        argname = "input_width";
                    } else if (aidx == CONV_PARAM_INPUT_HEIGHT) {
                        val = settings->input_height;
                        argname = "input_height";
                    } else if (aidx == CONV_PARAM_OUTPUT_WIDTH) {
                        val = layer->output_columns;
                        argname = "output_columns";
                    } else if (aidx == CONV_PARAM_OUTPUT_HEIGHT) {
                        val = layer->output_rows;
                        argname = "output_rows";
                    } else if (aidx == CONV_PARAM_REGION_SIZE) {
                        val = settings->filter_width;
                        if (val != arg) {
                            loadErr(
                                filepath, f, "Layer[%d] filter_width is %d, "
                                "but file spcifies %d", layer->index, val, arg
                            );
                            return 0;
                        }
                        val = settings->filter_height;
                        if (val != arg && val > 0) {
                            loadErr(
                                filepath, f, "Layer[%d] filter_height is %d, "
                                "but file spcifies %d", layer->index, val, arg
                            );
                            return 0;
                        }
                        continue;
                    } else if (aidx == CONV_PARAM_STRIDE) {
                        val = settings->stride;
                        argname = "stride";
                    } else if (aidx == CONV_PARAM_PADDING) {
                        val = settings->padding;
                        argname = "padding";
                    } else if (aidx == CONV_PARAM_USE_RELU) {
                        int use_relu = (arg == 1);
                        if (use_relu && layer->activate != PSRelu) {
                            loadErr(filepath, f, "Layer[%d] activation is %s"
                                    ", but file specifies relu", layer->index,
                                    PSGetActivationName(layer->activate));
                            return 0;
                        } else if (!use_relu && layer->activate == PSRelu) {
                            loadErr(filepath, f, "Layer[%d] activation is relu"
                                    ", but file activation isn't",
                                    layer->index);
                            return 0;
                        }
                        continue;
                    } else {
                        loadErr(filepath, f, "Layer[%d]: unknown argument[%d]",
                                layer->index, aidx);
                        return 0;
                    }
                    if (arg != val) {
                        loadErr(
                            filepath, f,
                            "Layer %d: loaded arg[%d] = %d differs from "
                            "%s = %d",
                            i, aidx, arg, argname, val
                        );
                        return 0;
                    }
                }
            }
            PSSetDropout(layer, dropout);
        } else {
            layer = NULL;
            PSLayerDef ldef = {.flags = lflags};
            if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                for (aidx = 0; aidx < param_c; aidx++) {
                    int arg = (aidx < argc ? args[aidx] : 0);
                    if (aidx == CONV_PARAM_FEATURE_COUNT)
                        ldef.output_depth = arg;
                    else if (aidx == CONV_PARAM_OUTPUT_WIDTH)
                        ldef.output_columns = arg;
                    else if (aidx == CONV_PARAM_OUTPUT_HEIGHT)
                        ldef.output_rows = arg;
                    else if (aidx == CONV_PARAM_STRIDE)
                        ldef.stride = arg;
                    else if (aidx == CONV_PARAM_PADDING)
                        ldef.padding = arg;
                    else if (aidx == CONV_PARAM_USE_RELU && arg)
                        ldef.activation = PSRelu;
                    else if (aidx == CONV_PARAM_REGION_SIZE) {
                        ldef.filter_width = arg;
                        ldef.filter_height = arg;
                    }
                }
            } else {
                if (model->size == 0 && (lflags & PS_FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    model->flags |= PS_FLAG_ONEHOT;
                } else if (argc > 0) {
                    /*loadErr(filepath, f, "Unknown arguments");
                    return 0;*/
                    for (aidx = 0; aidx < argc; aidx++) {
                        int arg = args[aidx];
                        if (aidx == CONV_PARAM_FEATURE_COUNT)
                            ldef.output_depth = arg;
                        else if (aidx == CONV_PARAM_OUTPUT_WIDTH)
                            ldef.output_columns = arg;
                        else if (aidx == CONV_PARAM_OUTPUT_HEIGHT)
                            ldef.output_rows = arg;
                    }
                }
            }
            layer = PSAddLayer(model, ltype, lsize, &ldef);
            if (layer == NULL) {
                PSErr(__func__, "Could not create layer %d", i);
                return 0;
            }
            layer->flags |= lflags;
            if (dropout > 0) {
                PSLayerDef dropout_ldef = {.dropout = dropout};
                PSLayer *dropout_layer = PSAddLayer(
                    model, Dropout, lsize, &dropout_ldef
                );
                if (dropout_layer == NULL) {
                    PSErr(__func__, "Could not create dropout layer %d", i + 1);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int loadLayerDefinitions(PSModel *model, char *vers,
                                int nlayers, int empty,
                                PSModel *parent,
                                const char *filepath, FILE *f)
{
    UNUSED(vers);
    PSLayer *layer = NULL;
    for (int i = 0; i < nlayers; i++) {
        int idx = 0, lsize = 0, lflags = 0, type = 0;
        PSLayerType ltype = FullyConnected;
        PSLayer *providers[PS_MAX_PROVIDERS];
        char sep[2] = {0};
        char propname[31];
        int ok = scanFile(
            f, "layer[%d]:%d,%d,%d%1[,\n]", 5, NULL,
            &idx, &type, &lsize, &lflags, sep
        );
        if (!ok) {
            loadErr(filepath, f, "Invalid layer %d definition");
            return 0;
        }
        if (i != idx) {
            loadErr(filepath, f, "Expected layer %d, got %d",
                    i, idx);
            return 0;
        }
        if (!empty) {
            layer = model->layers[i];
            if (layer == NULL) {
                loadErr(filepath, NULL, "model has no layer at index %d", i);
                return 0;
            }
            if (layer->size != lsize) {
                loadErr(filepath, NULL, "Layer %d size %d differs from %d!",
                        i, layer->size, lsize);
                return 0;
            }
            if (ltype != layer->type) {
                loadErr(filepath, NULL, "Layer %d type %d differs from %d!",
                        i, (int) (layer->type), (int) ltype);
                return 0;
            }
        }
        PSLayerDef ldef = {.flags = lflags};
        while (sep[0] != '\n') {
            ok = scanFile(f, "%30[a-zA-Z_-]=", 1, NULL, propname);
            if (!ok) {
                loadErr(filepath, f, "Invalid layer property def.");
                return 0;
            }
            if (strcmp("activation", propname) == 0) {
                char actvname[31] = {0};
                ok = scanFile(f, "%30[a-z]%1[,\n]", 2, NULL, actvname, sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid layer property value");
                    return 0;
                }
                if (strcmp("sigmoid", actvname) == 0)
                    ldef.activation = PSSigmoid;
                else if (strcmp("tanh", actvname) == 0)
                    ldef.activation = PSTanhActivation;
                else if (strcmp("relu", actvname) == 0)
                    ldef.activation = PSRelu;
                else if (strcmp("null", actvname) == 0)
                    ldef.activation = NULL;
                else {
                    loadErr(filepath, f, "Invalid activation function: '%s'",
                            actvname);
                    return 0;
                }
            } else if (strcmp("output_depth", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_depth), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid output_depth");
                    return 0;
                }
            } else if (strcmp("output_cols", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_columns), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid output_cols");
                    return 0;
                }
            } else if (strcmp("output_rows", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_rows), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid output_rows");
                    return 0;
                }
            } else if (strcmp("output_cols", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_columns), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid output_cols");
                    return 0;
                }
            } else if (strcmp("dropout", propname) == 0) {
                ok = scanFile(
                    f, PSFLOAT_FORMAT "%1[,\n]", 2, NULL,
                    &(ldef.dropout), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid dropout");
                    return 0;
                }
            } else if (strcmp("stride", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.stride), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid stride");
                    return 0;
                }
            } else if (strcmp("padding", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.padding), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid padding");
                    return 0;
                }
            } else if (strcmp("filter_width", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.filter_width), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid filter_width");
                    return 0;
                }
            } else if (strcmp("filter_height", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.filter_height), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid filter_height");
                    return 0;
                }
            } else if (strcmp("onehot_size", propname) == 0) {
                int onehot_size = 0;
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &onehot_size, sep
                );
                if (!ok || onehot_size < 0) {
                    loadErr(filepath, f, "Invalid onehot_size");
                    return 0;
                }
                ldef.flags |= PS_FLAG_ONEHOT;
                lsize = onehot_size;
            } else if (strcmp("pretrained", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.pretrained), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid 'pretrained' value");
                    return 0;
                }
            } else if (strcmp("epsilon", propname) == 0) {
                ok = scanFile(
                    f, PSFLOAT_FORMAT "%1[,\n]", 2, NULL,
                    &(ldef.epsilon), sep
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid epsilon");
                    return 0;
                }
            } else if (strcmp("attention_type", propname) == 0) {
                int attention_type = PSInvalidAttention;
                ok = scanFile(f, "%d%1[,\n]", 2, NULL, &attention_type, sep);
                if (ok) ok = (
                    attention_type >= 0 && attention_type <= PSAdditiveAttention
                );
                if (!ok) {
                    loadErr(filepath, f, "Invalid attention_type");
                    return 0;
                }
                ldef.attention_type = (PSAttentionType) attention_type;
            } else if (strcmp("n_heads", propname) == 0) {
                ok = scanFile(f, "%d%1[,\n]", 2, NULL,
                              &(ldef.attention_heads), sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid n_heads");
                    return 0;
                }
            } else if (strcmp("causal", propname) == 0) {
                ok = scanFile(f, "%d%1[,\n]", 2, NULL,
                              &(ldef.causal_attention), sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid causal value");
                    return 0;
                }
            } else if (strcmp("attention_scale", propname) == 0) {
                ok = scanFile(f, PSFLOAT_FORMAT "%1[,\n]", 2, NULL,
                              &(ldef.attention_scale), sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid attention_scale");
                    return 0;
                }
            } else if (strcmp("query_provider", propname) == 0) {
                int nidx = -1, lidx = -1;
                ok = scanFile(f, "%d:%d%1[,\n]", 3, NULL, &nidx, &lidx, sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid query_provider value");
                    return 0;
                }
                PSLayer *provider = layerByIndex(model, parent, nidx, lidx);
                if (provider == NULL)
                    provider = PSMakeLayerPlaceholder(lidx, nidx);
                if (provider == NULL) {
                    loadErr(filepath, f, "Invalid query_provider %d:%d",
                            nidx, lidx);
                    return 0;
                }
                ldef.query_provider = provider;
            } else if (strcmp("keys_provider", propname) == 0) {
                int nidx = -1, lidx = -1;
                ok = scanFile(f, "%d:%d%1[,\n]", 3, NULL, &nidx, &lidx, sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid keys_provider value");
                    return 0;
                }
                PSLayer *provider = layerByIndex(model, parent, nidx, lidx);
                if (provider == NULL)
                    provider = PSMakeLayerPlaceholder(lidx, nidx);
                if (provider == NULL) {
                    loadErr(filepath, f, "Invalid keys_provider %d:%d",
                            nidx, lidx);
                    return 0;
                }
                ldef.keys_provider = provider;
            } else if (strcmp("values_provider", propname) == 0) {
                int nidx = -1, lidx = -1;
                ok = scanFile(f, "%d:%d%1[,\n]", 3, NULL, &nidx, &lidx, sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid values_provider value");
                    return 0;
                }
                PSLayer *provider = layerByIndex(model, parent, nidx, lidx);
                if (provider == NULL)
                    provider = PSMakeLayerPlaceholder(lidx, nidx);
                if (provider == NULL) {
                    loadErr(filepath, f, "Invalid values_provider %d:%d",
                            nidx, lidx);
                    return 0;
                }
                ldef.values_provider = provider;
            } else if (strcmp("enabled_projections", propname) == 0 ||
                       strcmp("trainable_params", propname) == 0)
            {
                ok = scanFile(f, "%d%1[,\n]", 2, NULL,
                              &(ldef.enabled_projections), sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid %s", propname);
                    return 0;
                }
            } else if (strcmp("operator", propname) == 0) {
                ok = scanFile(f, "%d%1[,\n]", 2, NULL,
                              &(ldef.operator), sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid operator");
                    return 0;
                }
            } else if (strcmp("providers_count", propname) == 0) {
                ok = scanFile(f, "%d%1[,\n]", 2, NULL,
                              &(ldef.providers_count), sep);
                if (!ok || ldef.providers_count < 0) {
                    loadErr(filepath, f, "Invalid providers_count");
                    return 0;
                } else if (ldef.providers_count > PS_MAX_PROVIDERS) {
                    loadErr(filepath, f, "providers_count must be <= %d",
                            PS_MAX_PROVIDERS);
                    return 0;
                }
            } else if (strcmp("providers", propname) == 0) {
                if (ldef.providers_count <= 0) {
                    loadErr(filepath, f, "providers_count must be > 0");
                    return 0;
                }
                for (int j = 0; j < ldef.providers_count; j++) {
                    int nidx = -1, lidx = -1;
                    ok = scanFile(f, "%d:%d%1[,-\n]", 3, NULL,
                                  &nidx, &lidx, sep);
                    if (!ok) {
                        loadErr(filepath, f, "invalid provider[%d]", j);
                        return 0;
                    }
                    PSLayer *provider = layerByIndex(model, parent,nidx,lidx);
                    /*if (provider == NULL)
                        provider = PSMakeLayerPlaceholder(lidx, nidx);*/
                    if (provider == NULL) {
                        loadErr(filepath, f, "invalid provider[%d]: %d:%d",
                                j, nidx, lidx);
                        return 0;
                    }
                    providers[j] = provider;
                    ldef.providers = providers;
                }
            } else {
                ok = scanFile(f, "%*[^,\n]%1[,\n]", 1, NULL, sep);
                if (!ok) {
                    loadErr(filepath, f, "Invalid layer property value");
                    return 0;
                }
                continue;
            }
        }
        if (!empty) {
            /* TODO: perform checks */
            continue;
        }
        ltype = (PSLayerType) type;
        layer = PSAddLayer(model, ltype, lsize, &ldef);
        if (layer == NULL) {
            PSErr(__func__, "Could not create layer %d", i);
            return 0;
        }
    }
    return 1;
}

static int loadLegacyLayersParameters(PSModel *model,
                                      const char *filepath,
                                      FILE *f, int verbose)
{
    int i;
    char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
        "," PSFLOAT_FORMAT "|";
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        int lsize = 0;
        if (layer->type == Convolutional) {
            lsize = layer->output_depth;
        } else if (layer->type == Pooling || layer->type == Dropout) {
            continue;
        } else lsize = layer->size;
        if (GRU == layer->type) {
            loadErr(filepath, NULL, "GRU layers not supported in models "
                    "saved with version < 0.9.0");
            return 0;
        }
        int is_lstm = (LSTM == layer->type);
        int llen = 0, ok;
        uint64_t wsize = PSGetLayerInputWeightsCount(layer, 1);
        int input_size = wsize, widx;
        if (RNNLayer == layer->type) wsize += layer->size;
        for (int j = 0; j < lsize; j++) {
            PSFloat bias = 0;
            /* LSTM biases */
            PSFloat cb = 0.0, ib = 0.0, ob = 0.0, fb = 0.0;
            int matched = 0;
            if (!is_lstm) ok = scanFile(f, PSFLOAT_FORMAT "|", 1, NULL, &bias);
            else ok = scanFile(f, lstm_fmt, 4, &matched, &cb, &ib, &ob, &fb);
            if (!ok || (is_lstm && matched < 4)) {
                if (verbose) printf("\n");
                loadErr(
                    filepath, f, "Layer %d, neuron %d: invalid bias!", i, j
                );
                return 0;
            }
            if (layer->biases == NULL) {
                if (verbose) printf("\n");
                loadErr(filepath, f, "Layer %d biases are NULL");
                return 0;
            }
            if (layer->weights == NULL || layer->weights[0] == NULL) {
                if (verbose) printf("\n");
                loadErr(filepath, f, "Layer %d weights are NULL");
                return 0;
            }
            PSFloat *weights = NULL;
            if (!is_lstm) {
                layer->biases[j] = bias;
                if (Convolutional == layer->type) {
                    weights = layer->weights[j];
                } else {
                    weights = layer->weights[0];
                }
            } else {
                layer->biases[j] = cb;
                layer->biases[(PS_LSTM_INPUT_IDX * layer->size) + j] = ib;
                layer->biases[(PS_LSTM_OUTPUT_IDX * layer->size) + j] = ob;
                layer->biases[(PS_LSTM_FORGET_IDX * layer->size) + j] = fb;
                PSLSTMCell *cell = (PSLSTMCell *) layer->extra;
                if (cell == NULL) {
                    loadErr(filepath, f, "Layer %d LSTM cell is NULL");
                    return 0;
                }
                if (cell->candidate_weights == NULL ||
                    cell->input_weights == NULL ||
                    cell->output_weights == NULL ||
                    cell->forget_weights == NULL ||
                    cell->candidate_hidden_weights == NULL ||
                    cell->input_hidden_weights == NULL ||
                    cell->output_hidden_weights == NULL ||
                    cell->forget_hidden_weights == NULL)
                {
                    loadErr(filepath, f, "Layer %d incomplete LSTM weights");
                    return 0;
                }
                input_size = PSMatrixLength(cell->candidate_weights) / lsize;
                wsize = (input_size + layer->size) * 4;
            }
            /* if (Convolutional == layer->type) weights = layer->weights[j];*/
            for (uint64_t k = 0; k < wsize; k++) {
                if (Convolutional == layer->type) widx = k;
                else widx = k + (j * input_size);
                PSFloat w = 0;
                ok = scanFile(f, PSFLOAT_FORMAT "%*[,\n]", 1, NULL, &w);
                if (!ok) {
                    if (verbose) printf("\n");
                    loadErr(
                        filepath, f,"Layer %d neuron %d: invalid weight[%d]",
                        i, j, k
                    );
                    return 0;
                }
                if (RNNLayer == layer->type) {
                    if (k >= (uint64_t) input_size) {
                        weights = layer->weights[1];
                        widx = (j * layer->size) + (k - input_size);
                    } else weights = layer->weights[0];
                } else if (is_lstm) {
                    int matrix_idx = k / (input_size + layer->size);
                    widx = k % (input_size + layer->size);
                    if (widx >= input_size) {
                        matrix_idx += 4;
                        widx -= input_size;
                    }
                    weights = layer->weights[matrix_idx];
                }
                if (weights == NULL) {
                    loadErr(
                        filepath, f,"Layer %d neuron %d weight %d: "
                        "could not determine weights", i, j, k
                    );
                    return 0;
                }
                uint64_t matrix_len = PSMatrixLength(weights);
                if ((uint64_t) widx >= matrix_len) {
                    loadErr(
                        filepath, f,"Layer %d neuron %d weight %d: "
                        "invalid weight index %d (max index: %llu)",
                        i, j, k, widx, matrix_len - 1
                    );
                    return 0;
                }
                weights[widx] = w;
                if (verbose) {
                    llen = printf("\rLoading layer %d, neuron %d", i, j);
                    PSFillWithBlank(llen - 1);
                }
            }
        }
        if (verbose) {
            llen = printf("\rLayer[%d]: Loaded %d neurons", i, lsize);
            PSFillWithBlank(llen - 1);
            printf("\n");
            PSPrintLayerInfo(layer);
        }
    }
    return 1;
}

static int loadLayerParameters(PSLayer *layer, const char *filepath, FILE *f,
                               int check_index, int verbose)
{

    if (layer->type == Pooling || layer->type == Dropout) return 0;
    int bias_count = 0, lidx = -1, wtype_count = 0, wcount = 0, ok = 1,
        i = layer->index;
    ok = scanFile(
        f, "--- Layer[%d] Biases: %d ---\n", 2, NULL, &lidx, &bias_count
    );
    if (!ok) {
        loadErr(filepath, f, "Missing layer %d biases header", i);
        return 0;
    }
    ok = (!check_index || i == lidx);
    if (!ok) {
        loadErr(
            filepath, f, "Invalid layer index %d, expected: %d", lidx, i
        );
        return 0;
    }
    int expected_bias_count = PSGetLayerParametersCount(
        layer, PS_PARAM_BIAS
    );
    int expected_weights_count = PSGetLayerParametersCount(
        layer, PS_PARAM_WEIGHT
    );
    ok = (bias_count == expected_bias_count);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%d]: found %d biases, expected: %d",
            i, bias_count, expected_bias_count
        );
        return 0;
    }
    if (layer->biases == NULL && bias_count > 0) {
        loadErr(
            filepath, f, "Layer[%d]: found %d biases, but "
            "layer->biases is NULL", i, bias_count
        );
        return 0;
    }
    for (int j = 0; j < bias_count; j++) {
        char *fmt = PSFLOAT_FORMAT ",";
        if (j == (bias_count - 1)) fmt = PSFLOAT_FORMAT "\n";
        PSFloat bias = 0;
        ok = scanFile(f, fmt, 1, NULL, &bias);
        if (!ok) {
            loadErr(
                filepath, f, "Layer[%d]: invalid bias %d", layer->index, j
            );
            return 0;
        }
        layer->biases[j] = bias;
    }
    ok = scanFile(f, "--- Layer[%d] Weights: %d,%d ---\n", 3, NULL,
                  &lidx, &wtype_count, &wcount);
    if (!ok) {
        loadErr(filepath, f, "Missing layer %d weights header", i);
        return 0;
    }
    ok = (i == lidx);
    if (!ok) {
        loadErr(
            filepath, f, "Invalid layer index %d, expected: %d", lidx, i
        );
        return 0;
    }
    ok = (wtype_count == layer->weight_types);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%d]: found %d weight types, "
            "expected: %d", i, wtype_count, layer->weight_types
        );
        return 0;
    }
    ok = (wcount == expected_weights_count);
    if (!ok) {
        loadErr(
            filepath, f, "Layer[%d]: found %d weights, "
            "expected: %d", i, wcount, expected_weights_count
        );
        return 0;
    }
    if (layer->weights == NULL && wtype_count > 0) {
        loadErr(
            filepath, f, "Layer[%d]: found %d weights, but "
            "layer->weights is NULL",
            i, wcount
        );
        ok = 0;
        return 0;
    }
    int partial_trainable_parameters = (layer->type == Attention);
    for (int j = 0; j < layer->weight_types; j++) {
        PSMatrix weights = layer->weights[j];
        ok = (weights != NULL);
        if (!ok && partial_trainable_parameters)
            ok = scanFileNoMatch(f, "---\n");
        if (!ok) {
            loadErr(filepath, NULL, "Layer[%d]: weights[%d] is NULL", j);
            ok = 0;
            return 0;
        } else if (weights == NULL) continue;
        uint64_t wlen = PSMatrixLength(weights), widx;
        for (widx = 0; widx < wlen; widx++) {
            char *fmt = PSFLOAT_FORMAT ",";
            if (widx == (wlen - 1))
                fmt = PSFLOAT_FORMAT "\n";
            PSFloat w = 0;
            ok = scanFile(f, fmt, 1, NULL, &w);
            if (!ok) {
                loadErr(
                    filepath, f, "Layer[%d]: invalid weight[%d][%d]",
                    layer->index, j, widx
                );
                return 0;
            }
            weights[widx] = w;
        }
    }
    if (verbose) {
        int llen = printf("\rLayer[%d]: Loaded", i);
        PSFillWithBlank(llen - 1);
        printf("\n");
        PSPrintLayerInfo(layer);
    }
    return ok;
}

static int loadLayersParameters(PSModel *model,
                                const char *filepath,
                                FILE *f, int verbose)
{
    for (int i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer->type == Pooling || layer->type == Dropout) continue;
        if (!loadLayerParameters(layer, filepath, f, 1, verbose)) return 0;
    }
    return 1;
}

static int loadLegacyGradients(PSModel *model, const char *filepath,
                               FILE *f, PSGradient **gradients, int i)
{
    char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
        "," PSFLOAT_FORMAT "|";
    char sep[2];
    sep[0] = '\0';
    for (int j = 1; j < model->size; j++) {
        PSGradient *lgradients = gradients[j - 1];
        if (lgradients == NULL) continue;
        PSLayer *layer = model->layers[j];
        assert(layer != NULL);
        int lsize = 0, wsize = 0;
        if (layer->type == Pooling || layer->type == Dropout) continue;
        if (GRU == layer->type) {
            loadErr(filepath, NULL, "GRU layers not supported in models "
                    "saved with version < 0.9.0");
            return 0;
        }
        int is_lstm = (LSTM == layer->type);
        assert(layer->weights != NULL);
        assert(layer->weights[0] != NULL);
        wsize = (int) PSMatrixLength(layer->weights[0]);
        if (layer->type == Convolutional) {
            lsize = layer->output_depth;
        } else {
            lsize = layer->size;
            if (is_lstm) {
                wsize += lsize;
                wsize *= 4;
            } else if (RNNLayer == layer->type) wsize += lsize;
        }
        for (int k = 0; k < lsize; k++) {
            int matched = 0, ok;
            PSFloat *bias_p = lgradients->biases + k;
            if (!is_lstm) ok = scanFile(f, PSFLOAT_FORMAT "|", 1, NULL,bias_p);
            else {
                PSFloat *cbias_p = lgradients->biases + j,
                        *ibias_p = lgradients->biases + lsize + j,
                        *obias_p = lgradients->biases + (lsize * 2) + j,
                        *fbias_p = lgradients->biases + (lsize * 4) + j;
                ok = scanFile(
                    f, lstm_fmt, 4, &matched,
                    cbias_p, ibias_p, obias_p, fbias_p
                );
            }
            if (!ok || (is_lstm && matched < 4)) {
                printf("\n");
                loadErr(
                    filepath, f,
                    "Memory gradients %d, Layer %d, Gradient %d: "
                    "invalid bias!", i, j, k
                );
                ok = 0; return 0;
            }
            for (int w = 0; w < wsize; w++) {
                PSFloat gw = 0;
                ok = scanFile(
                    f, PSFLOAT_FORMAT "%1[,\n]", 2, NULL, gw, sep
                );
                if (!ok) {
                    loadErr(
                        filepath, f,
                        "Memory gradients %d, Layer %d, "
                        "Gradient %d: invalid weight %d",
                        i, j, k, w
                    );
                    return 0;
                }
                int idx = 0;
                if (RNNLayer == layer->type) {
                    int input_size = (wsize - lsize);
                    int is_hidden = (w >= input_size);
                    if (is_hidden)
                        idx = (input_size * lsize) + (k * lsize) + w;
                    else idx = (k * input_size) + w;
                } else if (is_lstm) {
                    int wtype_idx = k / (wsize / 4);
                    int input_size = (wsize - lsize);
                    int is_hidden = (w >= input_size);
                    if (!is_hidden) idx = (wtype_idx * input_size * lsize) + w;
                    else {
                        idx = (input_size * lsize * 4) +
                              (wtype_idx *k * lsize) + w;
                    }
                } else idx = (k * wsize) + w;
                assert((uint64_t) idx < lgradients->weight_count);
                lgradients->weights[idx] = gw;
            }
        }
    }
    return 1;
}

static int loadGradients(PSModel *model, const char *filepath,
                         FILE *f, PSGradient **gradients, int i)
{
    for (int j = 1; j < model->size; j++) {
        int grad_idx = j - 1, gidx, bias_count, weight_count, ok, k;
        PSGradient *lgradients = gradients[grad_idx];
        if (lgradients == NULL) continue;
        ok = scanFile(
            f, "--- Gradient[%d] Biases: %d ---\n", 2, NULL,
            &gidx, &bias_count
        );
        if (!ok) {
            loadErr(
                filepath, f, "Missing memory gradients[%d] "
                "biases[%d] header", i, j
            );
            return 0;
        }
        ok = (gidx == grad_idx);
        if (!ok) {
            loadErr(
                filepath, f, "Invalid memory gradients[%d] index "
                "%d, expected %d (biases[%d])", i, gidx, grad_idx,j
            );
            return 0;
        }
        if (bias_count > 0 && lgradients->biases == NULL) {
            lgradients->biases = malloc(bias_count * sizeof(PSFloat));
            if (lgradients->biases == NULL) {
                PSPrintMemoryErrorMsg();
                loadErr(
                    filepath, NULL, "Failed to allocate gradient "
                    "biases"
                );
                return 0;
            }
        }
        for (k = 0; k < bias_count; k++) {
            char *fmt = PSFLOAT_FORMAT ",";
            if (k == (bias_count - 1)) fmt = PSFLOAT_FORMAT "\n";
            PSFloat bias;
            ok = scanFile(f, fmt, 1, NULL, &bias);
            if (!ok) {
                loadErr(
                    filepath, f, "Failed to load memory "
                    "gradients[%d] bias[%d][%d]", i, j, k
                );
                return 0;
            }
            lgradients->biases[k] = bias;
        }
        ok = scanFile(
            f, "--- Gradient[%d] Weights: %d ---\n", 2, NULL,
            &gidx, &weight_count
        );
        if (!ok) {
            loadErr(
                filepath, f, "Missing memory gradients[%d] "
                "weights[%d] header", i, j
            );
            return 0;
        }
        ok = (gidx == grad_idx);
        if (!ok) {
            loadErr(
                filepath, f, "Invalid memory gradients[%d] index "
                "%d, expected %d (weights[%d])", i, gidx, grad_idx,j
            );
            return 0;
        }
        if (weight_count > 0 && lgradients->weights == NULL) {
            lgradients->weights = malloc(weight_count * sizeof(PSFloat));
            if (lgradients->weights == NULL) {
                PSPrintMemoryErrorMsg();
                loadErr(
                    filepath, NULL, "Failed to allocate gradient "
                    "weights"
                );
                return 0;
            }
        }
        for (k = 0; k < weight_count; k++) {
            char *fmt = PSFLOAT_FORMAT ",";
            if (k == (weight_count - 1)) fmt = PSFLOAT_FORMAT "\n";
            PSFloat w;
            ok = scanFile(f, fmt, 1, NULL, &w);
            if (!ok) {
                loadErr(
                    filepath, f, "Failed to load memory "
                    "gradients[%d] weight[%d][%d]", i, j, k
                );
                return 0;
            }
            lgradients->weights[k] = w;
        }
    }
    return 1;
}

int PSLayerLoad(PSLayer *layer, const char *filepath) {
    if (layer == NULL) return 0;
    FILE *f = fopen(filepath, "r");
    PSInfo("Loading layer %d from %s", layer->index, filepath);
    if (f == NULL) {
        PSErr(__func__, "Could not open '%s'", filepath);
        return 0;
    }
    int first_byte = fgetc(f);
    int loaded = 0, lidx = 0;
    fseek(f, 0, SEEK_SET);
    if (first_byte == EOF) {
        PSErr(__func__, "read error");
        return 0;
    }
    if (first_byte == 0xFF) {
        loaded = loadBinaryLayerParameters(layer, filepath, f, 0, 0);
        goto final;
    }
    if (scanFile(f, "layer[%d]:", 1, NULL, &lidx)) {
        /* Ignore layer definition */
        signed char c = fgetc(f);
        while (c != '\n') {
            if (c == EOF) break;
            c = fgetc(f);
        }
        if (c == EOF) {
            loadErr(filepath, NULL, "Invalid layer file");
            goto final;
        }
    }
    loaded = loadLayerParameters(layer, filepath, f, 0, 0);
final:
    fclose(f);
    return loaded;
}

/* Save `layer` to file located at `filepath`. By default, only the layer's
 * trainable parameters (ie. weights, biases) are saved and the layer is saved
 * in ASCII format.
 * However, this behavior can be changed by setting the following flags into
 * the `opts` argument:
 *  - `PS_IO_BINARY_MODE`: save the layer data in binary format.
 *  - `PS_IO_SAVE_DEFINITION`: also save layer's properties (ie. type,
 *    size, ...). This option cannot be used along with `PS_IO_BINARY_MODE`.
 * Return value: 1 if the layer is saved, 0 if somethign goes wrong.
 * Possible failure reasons:
 *  - The `layer` argument is NULL.
 *  - Both `PS_IO_BINARY_MODE` and `PS_IO_SAVE_DEFINITION` are set.
 *  - The file at `filepath` cannot be opened for writing.
 *  - Some error occurs qhile writing data. */
int PSLayerSave(PSLayer *layer, const char *filepath, int opts) {
    if (layer == NULL) return 0;
    int save_definition = (opts & PS_IO_SAVE_DEFINITION),
        binary = (opts & PS_IO_BINARY_MODE), saved = 1;
    if (save_definition && binary) {
        PSErr(__func__, "cannot save layer definition in binary mode");
        return 0;
    }
    FILE *f = fopen(filepath, "w");
    PSInfo("Saving layer %d to %s", layer->index, filepath);
    if (f == NULL) {
        PSErr(__func__, "Cannot open %s for writing!", filepath);
        return 0;
    }
    if (save_definition) saved = writeLayerDefinition(layer, f);
    if (saved) {
        int (*writeParams) (PSLayer *, int, FILE *, const char *) = NULL;
        if (!binary) writeParams = writeLayerParameters;
        else writeParams = writeBinaryLayerParameters;
        saved = writeParams(layer, 0, f, __func__);
    }
    fclose(f);
    return saved;
}

int loadModelDefinition(PSModel *model, FILE *f, char *vers) {
    int ok = 1;
    int idx = 0, val = 0;
    int epochs = 0, batch_count = 0, elements = 0, status = PS_STATUS_UNTRAINED,
        batch_size = 0, rnn_mode = NonRecurrent,
        max_sequence_len = PS_MAX_SEQUENCE_LENGTH,
        sequence_end = -1, is_built = 0,
        acceleration = PSGlobalAcceleration;
    char sep[2];
    sep[0] = '\0';
    while (scanFile(f, "%d%1[,\n]", 2, NULL, &val, sep)) {
        switch (idx++) {
            case 0:  model->flags |= val; break;
            case 1:  model->loss = getLossFunctionAtIndex(val); break;
            case 2:  epochs = val; break;
            case 3:  batch_count = val; break;
            case 4:  status = val; break;
            case 5:  elements = val; break;
            case 6:  batch_size = val; break;
            case 7:  rnn_mode = (PSRecurrentNetworkMode) val; break;
            case 8:  max_sequence_len = val; break;
            case 9:  sequence_end = val; break;
            case 10: is_built = val; break;
            case 11: acceleration = val; break;
            default: break;
        }
    }
    UNUSED(is_built);
    if (rnn_mode != NonRecurrent)
        PSSetRecurrentNetworkMode(model, rnn_mode);
    if (max_sequence_len > 0 || sequence_end >= 0) {
        model->sequence_settings.max_length = max_sequence_len;
        model->sequence_settings.end = sequence_end;
    }
    model->status = status;
    if (status != PS_STATUS_UNTRAINED) {
        if (model->training == NULL) {
            model->training = malloc(sizeof(PSTrainingInfo));
            model->training->requested_action = PS_ACTION_NONE;
            model->training->debug_dump_to = NULL;
        }
        model->training->current_epoch = epochs;
        model->training->current_batch = batch_count;
        model->training->current_element = elements;
        model->training->batch_size = batch_size;
    }
    if (PSCompareVersion(vers, "0.4.0") >= 0) {
        PSEnableAcceleration(&(model->acceleration), acceleration);
        if (acceleration != model->acceleration)
            PSWarn("Could not enable all saved accelerations");
    } else if (model->flags & PS_FLAG_ACCEL_DISABLED)
        model->acceleration = 0;
    return ok;
}

int loadModelName(PSModel *model, FILE *f, const char* filepath) {
    int ok = 1;
    char *name = NULL, *p = NULL;
    if (scanFileNoMatch(f, "name")) {
        uint32_t len = 0;
        ok = scanFile(f, "(%u):", 1, NULL, &len);
        if (!ok) {
            loadErr(filepath, f, "Missing model name length");
            goto final;
        }
        if (len == 0) goto eos;
        name = malloc(len + 1);
        ok = (name != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        p = name;
        while (len--) {
            signed char c = fgetc(f);
            ok = (c != EOF);
            if (!ok) {
                loadErr(filepath, f, "Model name is shorter than declared "
                        "length %u", len);
                goto final;
            }
            *(p++) = c;
        }
        *p = '\0';
        ok = PSModelSetName(model, name);
        if (!ok) {
            loadErr(filepath, NULL, "could not set model name");
            goto final;
        }
eos:
        ok = scanFileNoMatch(f, "\n");
    }
final:
    free(name);
    return ok;
}

int loadModelTrainingData(PSModel *model, FILE *f, const char* filepath,
                          int legacy_model)
{
    int ok = 1;
    if (scanFileNoMatch(f, MODEL_TRAINING_DATA_SEP)) {
        /* Model file has training data */
        int numgradients = 0;
        ok = scanFile(f, "memory_gradients:%d\n", 1, NULL, &numgradients);
        if (!ok) {
            loadErr(filepath, f, "Invalid or missing 'memory_gradients'");
            goto final;
        }
        ok = initTrainingContext(model, NULL, numgradients);
        if (!ok) {
            PSErr(__func__, "Failed to initialize model training data");
            goto final;
        }
        PSTrainingOptions *topts = PSGetModelTrainingOptions(model);
        if (topts == NULL) {
            PSErr(__func__, "Invalid model training data (missing options)");
            goto final;
        }
        PSGradient **memory_gradients[PS_MAX_MEMORY_GRADIENTS] = {0};
        if (numgradients > 0) {
            int foundgradients = PSGetTrainingMemoryGradients(
                model, memory_gradients
            );
            ok = (foundgradients == numgradients);
            if (ok) {
                ok = memory_gradients[0] != NULL;
                if (ok && numgradients >= 2) ok = memory_gradients[1] != NULL;
            }
            if (!ok) {
                loadErr(
                    filepath, NULL, "invalid model training data (missing "
                    "gradients)"
                );
                goto final;
            }
        }
        ok = scanFileNoMatch(f, "training_options:");
        if (!ok) {
            loadErr(filepath, f, "Missing 'training_options'");
            goto final;
        }
        ok = scanTrainingOptions(f, topts, filepath);
        if (!ok) goto final;
        for (int i = 0; i < numgradients; i++) {
            PSGradient **memg = memory_gradients[i];
            int gidx = -1;
            ok = scanFile(f, "memory_gradients[%d]:\n", 1, NULL, &gidx);
            if (ok && gidx != i) ok = 0;
            if (!ok) {
                loadErr(filepath, f, "invalid memory gradient[%d] header", i);
                goto final;
            }
            if (legacy_model)
                ok = loadLegacyGradients(model, filepath, f, memg, i);
            else
                ok = loadGradients(model, filepath, f, memg, i);
            if (!ok) {
                loadErr(
                    filepath, NULL, "failed to load memory gradients %s", i
                );
                goto final;
            }
        }
    }
final:
    return ok;
}

int loadLegacyModel(PSModel *model, FILE *f, const char* filepath,
                    char *vers, int has_model_def, int empty)
{
    int nlayers, ok = 1;
    int verbose = PSLogLevel == PSLOGLEVEL_DEBUG;
    if (has_model_def) {
        ok = loadModelDefinition(model, f, vers);
        if (!ok) return 0;
    }
    ok = scanFile(f, "%d:", 1, NULL, &nlayers);
    if (!ok) {
        loadErr(filepath, f, "Missing model size definition");
        goto final;
    }
    if (nlayers == 0) {
        loadErr(filepath, NULL, "Empty model model!");
        ok = 0;
        goto final;
    }
    if (!empty && model->size != nlayers) {
        loadErr(filepath, NULL, "model size differs!");
        ok = 0;
        goto final;
    }
    ok = loadLegacyLayerDefinitions(model, vers, nlayers, empty, filepath,f);
    if (!ok) goto final;
    ok = loadLegacyLayersParameters(model, filepath, f, verbose);
    if (verbose) printf("\n");
    if (!ok) goto final;
    /* Check for traing data */
    ok = loadModelTrainingData(model, f, filepath, 1);
    if (!ok) goto final;
    PSModelBuild(model);
final:
    return ok;
}

int readModel(PSModel *model, FILE *f, const char* filepath,
              char *vers, int empty, PSModel *parent)
{
    int ok = 1, nlayers = 0;
    int verbose = PSLogLevel == PSLOGLEVEL_DEBUG;
    ok = loadModelDefinition(model, f, vers);
    if (!ok) goto final;
    ok = loadModelName(model, f, filepath);
    if (!ok) goto final;
    ok = scanFile(f, "layers:%d\n", 1, NULL, &nlayers);
    if (!ok) {
        loadErr(filepath, f, "Missing model size definition");
        goto final;
    }
    if (nlayers == 0) {
        loadErr(filepath, NULL, "Empty model");
        ok = 0;
        goto final;
    }
    if (!empty && model->size != nlayers) {
        loadErr(filepath, NULL, "Model size differs!");
        ok = 0;
        goto final;
    }
    ok = loadLayerDefinitions(model, vers, nlayers, empty, parent,
                              filepath, f);
    if (!ok) goto final;
    ok = loadLayersParameters(model, filepath, f, verbose);
    if (verbose) printf("\n");
    if (!ok) goto final;
    PSModelLink link = {0};
    PSModelLink *prev_link = NULL;
    char *link_name = "model_link";
    if (PSCompareVersion(vers, "0.9.3") < 0) link_name = "network_link";
    char link_prop[255] = {0};
    snprintf(link_prop, 255, "%s:", link_name);
    if (parent != NULL && empty) {
        link.layer = NULL;
        link.previous_layer = NULL;
        if (scanFileNoMatch(f, link_prop)) {
            int layer_idx = -1, prev_net_idx = -1, prev_layer_idx = -1;
            ok = scanFile(f, "%d,%d,%d\n", 3, NULL, &layer_idx, &prev_net_idx,
                          &prev_layer_idx);
            if (!ok) {
                loadErr(filepath, f, "Invalid %d definition", link_name);
                goto final;
            }
            ok = layer_idx >= 0 || layer_idx < model->size;
            if (!ok) {
                loadErr(filepath, f, "Invalid layer index: %d", layer_idx);
                goto final;
            }
            link.layer = model->layers[layer_idx];
            ok = link.layer != NULL;
            if (!ok) {
                loadErr(filepath, f, "Invalid layer at %d", layer_idx);
                goto final;
            }
            PSModel *prevn = PSGetModelAtIndex(parent, prev_net_idx);
            ok = prevn != NULL;
            if (!ok) {
                loadErr(filepath, f, "No previous model at index %d",
                        prev_net_idx);
                goto final;
            }
            ok = prev_layer_idx >= 0 && prev_layer_idx < prevn->size;
            if (!ok) {
                loadErr(filepath, f, "Invalid layer index: %d", prev_layer_idx);
                goto final;
            }
            link.previous_layer = prevn->layers[prev_layer_idx];
            ok = link.previous_layer != NULL;
            if (!ok) {
                loadErr(filepath, f, "Invalid layer at %d", prev_layer_idx);
                goto final;
            }
            prev_link = &link;
        }
        ok = PSAddModel(parent, model, prev_link);
        if (!ok) {
            loadErr(filepath, NULL, "failed to add model");
            goto final;
        }
    }
    if (scanFileNoMatch(f, "sequence_start:")) {
        uint64_t seqstartlen = 0;
        PSFloat *seqstart = readSerializedFloatArray(
            f, ",\n", &seqstartlen, 0, model->input_size
        );
        ok = seqstart != NULL && seqstartlen == (uint64_t) model->input_size;
        if (!ok) {
            free(seqstart);
            loadErr(filepath, f, "Invalid sequence_start");
            goto final;
        }
        if (!PSSetSequenceStart(model, seqstart, seqstartlen)) {
            ok = 0;
            free(seqstart);
            loadErr(filepath, NULL, "Failed to set sequence start");
            goto final;
        }
    }
    /* Check for training data */
    ok = loadModelTrainingData(model, f, filepath, 0);
    if (!ok) goto final;
    ok = PSModelBuild(model);
final:
    return ok;
}

/* Load model data (including layers and their parameters) from file
 * located at `filepath` into `model`.
 * This function requires an already existing model. In order to load a new
 * model from scratch from, `PSLoadModel` should be used instead.
 * If `model` is empty (it has no layers), both the model structure and data
 * such as layer parameters will be loaded into the model itself).
 * If `model` is not empty (it already has layers), only data such as layer
 * parameters will be loaded into model. In this case, the structure of `model`
 * must match the structure declared by the file.
 * If the file defines a multi-model chain, the whole chain will be loaded (
 * in this case, if `model` is not empty, the `model` chain structure must
 * match the structure that has to be loaded from the file).
 * Return value: 1 if `model` is successfully loaded, 0 if:
 *  - `model` is NULL or `filepath` is NULL.
 *  - The file at `filepath` could not be opened for reading.
 *  - The file at `filepath` is not a valid PsyC model file.
 *  - PsyC version is lower than version declared in the file.
 *  - `model` is not empty and its structure differs from the one declared
 *    by the file (ie. different number of layers or models, different layer
 *    types, and so on).
 *  - Some error occurred while reading data from file. */
int PSModelLoad(PSModel *model, const char* filepath) {
    if (model == NULL || filepath == NULL) return 0;
    FILE *f = fopen(filepath, "r");
    PSInfo("Loading model from %s", filepath);
    if (f == NULL) {
        PSErr(__func__, "Could not open '%s'", filepath);
        return 0;
    }
    int verbose = PSLogLevel == PSLOGLEVEL_DEBUG;
    int empty = model->size == 0;
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int ok = 1, has_model_def = 0, legacy_model = 0;
    /* Search for header */
    if (scanFile(f, "--v%d.%d.%d", 3, NULL, &v0, &v1, &v2)) {
        sprintf(vers, "%d.%d.%d", v0, v1, v2);
        PSInfo("Model PsyC version is %s (current: %s).", vers, PSYC_VERSION);
        ok = (PSCompareVersion(vers, PSYC_VERSION) <= 0);
        if (!ok) {
            PSErr(
                __func__,
                "File version is higher than current PsyC version: %s > %s\n"
                "PsyC %s (or higher) is required to open '%s'",
                vers, PSYC_VERSION, vers, filepath
            );
            goto final;
        }
        /* Older versions directly print model definition after version */
        has_model_def = scanFileNoMatch(f, ",");
        if (has_model_def) legacy_model = 1;
        else if (scanFileNoMatch(f, ":")) {
            /* Scan header info */
            PSModelFileHeader header = {{0}};
            ok = scanModelFileHeader(f, &header, filepath);
            if (!ok) {
                loadErr(filepath, NULL, "Invalid file header");
                goto final;
            }
            if (verbose) {
                PSInfo("Info for file '%s':", filepath);
                printModelHeaderInfo(&header);
            }
        }
    }
    if (!legacy_model) legacy_model = (PSCompareVersion(vers, "0.9.0") < 0);
    if (legacy_model) {
        ok = loadLegacyModel(model,f,filepath,vers,has_model_def,empty);
        goto final;
    }
    ok = scanFileNoMatch(f, "model:");
    if (!ok) {
        loadErr(filepath, f, "Missing `model:` definition");
        goto final;
    }
    ok = readModel(model, f, filepath, vers, empty, NULL);
    if (!ok) goto final;
    int num_models = 1, loaded_models = 1;
    if (!empty) num_models = PSModelChainLength(model);
    PSModel *current = model;
    while (scanFileNoMatch(f, "model:")) {
        if (!empty) {
            current = model->next;
            ok = (current != NULL);
            if (!ok) {
                loadErr(
                    filepath, f, "Non-empty model only has %d model(s), "
                    "but model file still has models to load",
                    num_models
                );
                goto final;
            }
        } else {
            current = PSModelCreate(NULL);
            ok = current != NULL;
            if (!ok) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
        }
        ok = readModel(current, f, filepath, vers, empty, model);
        if (!ok) {
            if (empty) PSModelFree(current);
            goto final;
        }
        loaded_models++;
    }
final:
    if (f != NULL) fclose(f);
    return ok;
}

/* Load a new model from the file located at `filepath` into `model`.
 * In order to load model's data into an already existing model, `PSModelLoad`
 * should be used instead.
 * If the file defines a multi-model chain, the whole chain will be loaded.
 * Return value: 1 if model is successfully loaded, 0 if:
 *  - `filepath` is NULL.
 *  - The file at `filepath` could not be opened for reading.
 *  - The file at `filepath` is not a valid PsyC model file.
 *  - PsyC version is lower than version declared in the file.
 *  - Memory allocation issues.
 *  - Some error occurred while reading data from file. */
PSModel *PSLoadModel(const char* filepath) {
    if (filepath == NULL) {
        PSErr(__func__, "`filepath` is null");
        return NULL;
    }
    PSModel *model = PSModelCreate(NULL);
    if (model == NULL) {
        PSErr(__func__, "could not create model");
        return NULL;
    }
    int loaded = PSModelLoad(model, filepath);
    if (!loaded) {
        PSModelFree(model);
        return NULL;
    }
    return model;
}

static int writeModel(PSModel *model, FILE *f) {
    int ok = 1, opts = 0, i;
    int current_epoch = 0, current_batch = 0, current_element = 0,
        batch_size = 0;
    if (model->training != NULL) {
        current_epoch = model->training->current_epoch;
        current_batch = model->training->current_batch;
        current_element = model->training->current_element;
        batch_size = model->training->batch_size;
    }
    PSRecurrentNetworkMode rnn_mode = model->rnn_mode;
    int max_steps = model->sequence_settings.max_length;
    int eos = model->sequence_settings.end;
    int loss_function = getLossFunctionIndex(model->loss);
    fprintf(f, "model:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", model->flags,
            loss_function, current_epoch, current_batch, model->status,
            current_element, batch_size, (int) rnn_mode,
            max_steps, eos, PSModelIsBuilt(model));
    if (model->name != NULL) {
        int namelen = strlen(model->name);
        fprintf(f, "name(%d):%s\n", namelen, model->name);
    }
    fprintf(f, "layers:%d\n", model->size);
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (!writeLayerDefinition(layer, f)) return 0;
    }
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        PSLayerType ltype = layer->type;
        if (Pooling == ltype || layer->type == Dropout) continue;
        if (!writeLayerParameters(layer, opts, f, __func__)) return 0;
    }
    if (model->previous != NULL && model->previous_model_link != NULL) {
        PSModelLink *link = model->previous_model_link;
        if (link->layer == NULL || link->previous_layer == NULL ||
            link->layer->model == NULL ||
            link->previous_layer->model == NULL ||
            link->layer->model != model)
        {
            PSErr(NULL, "invalid previous_model_link for model %d",
                  model->index);
            return 0;
        }
        fprintf(f, "model_link:%d,%d,%d\n",
                link->layer->index, link->previous_layer->model->index,
                link->previous_layer->index);
    }
    if (model->sequence_settings.start != NULL) {
        fprintf(f, "sequence_start:");
        ok = writeSerializedFloatArray(f, model->input_size, ",", 0,
                                       model->sequence_settings.start);
        if (!ok) return 0;
        fprintf(f, "\n");
    }
    PSTrainingOptions *topts = PSGetModelTrainingOptions(model);
    PSGradient **memory_gradients[PS_MAX_MEMORY_GRADIENTS] = {0};
    int numgradients = PSGetTrainingMemoryGradients(model, memory_gradients);
    if (topts != NULL || numgradients > 0) {
        fprintf(f, MODEL_TRAINING_DATA_SEP);
        fprintf(f, "memory_gradients:%d\n", numgradients);
        fprintf(f, "training_options:");
        if (topts != NULL) {
            fprintf(f, "flags=%d,", topts->flags);
            fprintf(f, "l1_decay=");
            writeSerializedFloat(f, topts->l1_decay, opts);
            fprintf(f, ",l2_decay=");
            writeSerializedFloat(f, topts->l2_decay, opts);
            fprintf(f, ",momentum=");
            writeSerializedFloat(f, topts->momentum, opts);
            fprintf(f, ",rho=");
            writeSerializedFloat(f, topts->rho, opts);
            fprintf(f, ",eps=");
            writeSerializedFloat(f, topts->eps, opts);
            fprintf(f, ",beta1=");
            writeSerializedFloat(f, topts->beta1, opts);
            fprintf(f, ",beta2=");
            writeSerializedFloat(f, topts->beta2, opts);
            fprintf(f, ",clip=");
            writeSerializedFloat(f, topts->clip, opts);
            fprintf(
                f, ",optimization=%d",
                getOptimizationIndex(topts->optimization)
            );
            fprintf(f, ",bptt_truncate=%d", topts->bptt_truncate);
        }
        fprintf(f, "\n");
        for (i = 0; i < numgradients; i++) {
            PSGradient **memg = memory_gradients[i];
            if (memg == NULL) {
                PSErr(__func__, "memory gradient[%d] is NULL", i);
                return 0;
            }
            fprintf(f, "memory_gradients[%d]:\n", i);
            if (!writeGradients(model, memg, opts, f)) return 0;
        }
    }
    return 1;
}

/* Save `model` to file located at `filepath`. The function will save both
 * model's structure (ie layer propeties) and data (ie. parameters).
 * If `model` is part of a multi-model chain, the whole chain will be saved.
 * Return value: 1 if `model` is successfully saved, 0 if:
 *  - `model` is NULL or `filepath` is NULL.
 *  - `model` is empty (it has no layers).
 *  - The file at `filepath` could not be opened for writing.
 *  - Some error occurred while writing data to file. */
int PSModelSave(PSModel *model, const char* filepath) {
    if (model == NULL) {
        PSErr(__func__, "`model` is NULL");
        return 0;
    }
    if (filepath == NULL) {
        PSErr(__func__, "`filepath` is NULL");
        return 0;
    }
    if (model->size == 0) {
        PSErr(__func__, "Empty model!");
        return 0;
    }
    FILE *f = fopen(filepath, "w");
    PSInfo("Saving model to %s", filepath);
    if (f == NULL) {
        PSErr(__func__, "Cannot open %s for writing!", filepath);
        return 0;
    }
    int ok = 1;
    /*  Header */
    static struct utsname sysinfo;
    static int sysinfo_read = 0;
    if (!sysinfo_read) {
       uname(&sysinfo);
       sysinfo_read = 1;
    }
    int avx_available = (
        PSIsAccelerationAvailable(PSAcceleration_AVX) ? 1 : 0
    );
    int acf_available = (
        PSIsAccelerationAvailable(PSAcceleration_ACF) ? 1 : 0
    );
    fprintf(
        f, "--v%s:git=%s/%s-%s;float_size=%zu;archbits=%d;avx=%d;"
        "accelerate=%d,sys=%s,%s,%s;global_flags=%d;acceleration=%d;"
        "savetime=%ld\n",
        PSYC_VERSION, PSYC_GIT_SHA, PSYC_GIT_DIRTY, PSYC_GIT_BRANCH,
        sizeof(PSFloat), ((sizeof(long) == 8) ? 64 : 32), avx_available,
        acf_available, sysinfo.sysname, sysinfo.release, sysinfo.machine,
        PSGlobalFlags, PSGlobalAcceleration, time(NULL)
    );
    while (model != NULL) {
        ok = writeModel(model, f);
        if (!ok) goto final;
        model = model->next;
    }
final:
    fclose(f);
    return ok;
}

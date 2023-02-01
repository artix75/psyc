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

#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <inttypes.h>
#include <sys/utsname.h>

#include "psyc.h"
#include "config.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "gru.h"
#include "dropout.h"
#include "log.h"
#include "buildinfo.h"
#include "optimization.h"

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

PSTrainingOptions *PSGetNetworkTrainingOptions(PSNeuralNetwork *network);
int PSGetTrainingMemoryGradients(PSNeuralNetwork *network,
                                 PSGradient ***mg1, PSGradient ***mg2);
int initTrainingContext(PSNeuralNetwork *network, int mem_gradients_count);
int PSCompareVersion(const char* vers1, const char* vers2);
int getLossFunctionIndex(PSLossFunction function);
PSLossFunction getLossFunctionAtIndex(int index);
void PSPrintLayerInfo(PSLayer *layer);
const char *PSGetActivationName(PSActivationFunction func);

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
    PSLog(PSLOGLEVEL_ERROR, "ERROR: ");
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
    while ((ok = scanFile(f, "%[a-zA-Z_-]=", 1, NULL, propname))) {
        sep[0] = '\0';
        ok = scanFile(f, "%[^;\n]%[;\n]", 2, NULL, val, sep);
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
    while ((ok = scanFile(f, "%[a-zA-Z0-9_-]=", 1, NULL, propname))) {
        sep[0] = '\0';
        ok = scanFile(f, "%[^,\n]%[,\n]", 2, NULL, val, sep);
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

int writeSerializedFloats(FILE *out, int count, char *sep, int opts, ...) {
    int has_sep = (sep != NULL), len = 0, i;
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

int writeSerializedFloatArray(FILE *out, int count, char *sep, int opts,
                              PSFloat *array)
{
    int has_sep = (sep != NULL), len = 0, i;
    for (i = 0; i < count; i++) {
        if (has_sep && i > 0) len += fprintf(out, "%s", sep);
        len += writeSerializedFloat(out, array[i], opts);
    }
    return len;
}

int writeGradients(PSNeuralNetwork *network, PSGradient **gradients,
                   int opts, FILE *f)
{
    if (network == NULL || network->size == 0) return 0;
    for (int i = 1; i < network->size; i++) {
        PSGradient *gradient = gradients[i - 1];
        if (gradient == NULL) continue;
        if (gradient->bias_count > 0 && gradient->biases == NULL) {
            PSErr("PSSaveNetwork", "Invalid gradient biases");
            return 0;
        }
        if (gradient->weight_count > 0 && gradient->weights == NULL) {
            PSErr("PSSaveNetwork", "Invalid gradient weights");
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

void writeLayerDefinition(PSLayer *layer, FILE *f) {
    char *activation = (char *) PSGetActivationName(layer->activate);
    if (activation == NULL) activation = "null";
    fprintf(
        f, "layer[%d]:%d,%d,%d,activation=%s,"
        "output_depth=%d,output_cols=%d,output_rows=%d",
        layer->index, (int) layer->type, layer->size,
        layer->flags, activation, layer->output_depth,
        layer->output_columns, layer->output_rows
    );
    if (layer->flags & FLAG_ONEHOT && layer->index == 0)
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
        fprintf(f, ",stride=%d,padding=%d,filter_width=%d,filter_height=%d\n",
                stride, padding, filter_w, filter_h);
    } else if (Dropout == layer->type) {
        PSFloat dropout = PSGetDropout(layer);
        fprintf(f, ",dropout=" PSFLOAT_FORMAT "\n", dropout);
    } else fprintf(f, "\n");
}

static int loadLegacyLayerDefinitions(PSNeuralNetwork *network, char *vers,
                                      int netsize, int empty,
                                      const char *filename, FILE *f)
{
    int min_argc = 1, i, ok;
    if (PSCompareVersion(vers, "0.2.2") == 1) min_argc = DATA_LAYER_MIN_ARGC;
    else if (PSCompareVersion(vers, "0.0.0") == 1) min_argc = 2;
    PSLayer *layer = NULL;
    for (i = 0; i < netsize; i++) {
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
                loadErr(filename, f, "Invalid layer def: layer[%d]", i);
                return 0;
            }
            if (argc == 0) {
                loadErr(
                    filename, NULL,
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
                        filename, f,
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
                        loadErr(filename, f, "Argument is out-of-bounds");
                        return 0;
                    }
                    args[arg_aidx] = arg;
                }
            }
            argc -= min_argc;
            ok = scanFile(f, "]%[,\n]", 1, NULL, sep);
            if (!ok) return 0;
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                loadErr(filename, NULL, "Layer %d size %d differs from %d!",
                    i, layer->size, lsize);
                return 0;
            }
            if (ltype != layer->type) {
                loadErr(filename, NULL, "Layer %d type %d differs from %d!",
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
                                filename, f, "Layer[%d] filter_width is %d, "
                                "but file spcifies %d", layer->index, val, arg
                            );
                            return 0;
                        }
                        val = settings->filter_height;
                        if (val != arg && val > 0) {
                            loadErr(
                                filename, f, "Layer[%d] filter_height is %d, "
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
                            loadErr(filename, f, "Layer[%d] activation is %s"
                                    ", but file specifies relu", layer->index,
                                    PSGetActivationName(layer->activate));
                            return 0;
                        } else if (!use_relu && layer->activate == PSRelu) {
                            loadErr(filename, f, "Layer[%d] activation is relu"
                                    ", but file activation isn't",
                                    layer->index);
                            return 0;
                        }
                        continue;
                    } else {
                        loadErr(filename, f, "Layer[%d]: unknown argument[%d]",
                                layer->index, aidx);
                        return 0;
                    }
                    if (arg != val) {
                        loadErr(
                            filename, f,
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
                if (network->size == 0 && (lflags & FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    network->flags |= FLAG_ONEHOT;
                } else if (argc > 0) {
                    /*loadErr(filename, f, "Unknown arguments");
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
            layer = PSAddLayer(network, ltype, lsize, &ldef);
            if (layer == NULL) {
                PSErr(__func__, "Could not create layer %d", i);
                return 0;
            }
            layer->flags |= lflags;
            if (dropout > 0) {
                PSLayerDef dropout_ldef = {.dropout = dropout};
                PSLayer *dropout_layer = PSAddLayer(
                    network, Dropout, lsize, &dropout_ldef
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

static int loadLayerDefinitions(PSNeuralNetwork *network, char *vers,
                                int netsize, int empty, const char *filename,
                                FILE *f)
{
    UNUSED(vers);
    PSLayer *layer = NULL;
    for (int i = 0; i < netsize; i++) {
        int idx = 0, lsize = 0, lflags = 0, type = 0;
        PSLayerType ltype = FullyConnected;
        char sep[2] = {0};
        char propname[31];
        int ok = scanFile(
            f, "layer[%d]:%d,%d,%d%1[,\n]", 5, NULL,
            &idx, &type, &lsize, &lflags, sep
        );
        if (!ok) {
            loadErr(filename, f, "Invalid layer %d definition");
            return 0;
        }
        if (i != idx) {
            loadErr(filename, f, "Expected layer %d, got %d",
                    i, idx);
            return 0;
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer == NULL) {
                loadErr(filename, NULL, "Network has no layer at index %d", i);
                return 0;
            }
            if (layer->size != lsize) {
                loadErr(filename, NULL, "Layer %d size %d differs from %d!",
                    i, layer->size, lsize);
                return 0;
            }
            if (ltype != layer->type) {
                loadErr(filename, NULL, "Layer %d type %d differs from %d!",
                        i, (int) (layer->type), (int) ltype);
                return 0;
            }
        }
        PSLayerDef ldef = {.flags = lflags};
        while (sep[0] != '\n') {
            ok = scanFile(f, "%30[a-zA-Z_-]=", 1, NULL, propname);
            if (!ok) {
                loadErr(filename, f, "Invalid layer property def.");
                return 0;
            }
            if (strcmp("activation", propname) == 0) {
                char actvname[31] = {0};
                ok = scanFile(f, "%30[a-z]%1[,\n]", 2, NULL, actvname, sep);
                if (!ok) {
                    loadErr(filename, f, "Invalid layer property value");
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
                    loadErr(filename, f, "Invalid activation function: '%s'",
                            actvname);
                    return 0;
                }
            } else if (strcmp("output_depth", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_depth), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid output_depth");
                    return 0;
                }
            } else if (strcmp("output_cols", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_columns), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid output_cols");
                    return 0;
                }
            } else if (strcmp("output_rows", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_rows), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid output_rows");
                    return 0;
                }
            } else if (strcmp("output_cols", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.output_columns), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid output_cols");
                    return 0;
                }
            } else if (strcmp("dropout", propname) == 0) {
                ok = scanFile(
                    f, PSFLOAT_FORMAT "%1[,\n]", 2, NULL,
                    &(ldef.dropout), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid output_cols");
                    return 0;
                }
            } else if (strcmp("stride", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.stride), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid stride");
                    return 0;
                }
            } else if (strcmp("padding", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.padding), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid padding");
                    return 0;
                }
            } else if (strcmp("filter_width", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.filter_width), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid filter_width");
                    return 0;
                }
            } else if (strcmp("filter_height", propname) == 0) {
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &(ldef.filter_height), sep
                );
                if (!ok) {
                    loadErr(filename, f, "Invalid filter_height");
                    return 0;
                }
            } else if (strcmp("onehot_size", propname) == 0) {
                int onehot_size = 0;
                ok = scanFile(
                    f, "%d%1[,\n]", 2, NULL, &onehot_size, sep
                );
                if (!ok || onehot_size < 0) {
                    loadErr(filename, f, "Invalid onehot_size");
                    return 0;
                }
                ldef.flags |= FLAG_ONEHOT;
                lsize = onehot_size;
            } else {
                ok = scanFile(f, "%*[^,\n]%1[,\n]", 1, NULL, sep);
                if (!ok) {
                    loadErr(filename, f, "Invalid layer property value");
                    return 0;
                }
                continue;
            }
        }
        if (!empty) {
            //TODO: perform checks
            continue;
        }
        ltype = (PSLayerType) type;
        layer = PSAddLayer(network, ltype, lsize, &ldef);
        if (layer == NULL) {
            PSErr(__func__, "Could not create layer %d", i);
            return 0;
        }
    }
    return 1;
}

static int loadLegacyLayerParameters(PSNeuralNetwork *network,
                                     const char * filename,
                                     FILE *f, int verbose)
{
    int i;
    char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
        "," PSFLOAT_FORMAT "|";
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int lsize = 0;
        if (layer->type == Convolutional) {
            lsize = layer->output_depth;
        } else if (layer->type == Pooling || layer->type == Dropout) {
            continue;
        } else lsize = layer->size;
        if (GRU == layer->type) {
            loadErr(filename, NULL, "GRU layers not supported in models "
                    "saved with version < 0.9.0");
            return 0;
        }
        int is_lstm = (LSTM == layer->type);
        int llen = 0, ok;
        uint64_t wsize = PSGetLayerInputWeightsCount(layer, 1);
        int input_size = wsize, widx;
        if (Recurrent == layer->type) wsize += layer->size;
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
                    filename, f, "Layer %d, neuron %d: invalid bias!", i, j
                );
                return 0;
            }
            if (layer->biases == NULL) {
                if (verbose) printf("\n");
                loadErr(filename, f, "Layer %d biases are NULL");
                return 0;
            }
            if (layer->weights == NULL || layer->weights[0] == NULL) {
                if (verbose) printf("\n");
                loadErr(filename, f, "Layer %d weights are NULL");
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
                    loadErr(filename, f, "Layer %d LSTM cell is NULL");
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
                    loadErr(filename, f, "Layer %d incomplete LSTM weights");
                    return 0;
                }
                input_size = PSMatrixLength(cell->candidate_weights) / lsize;
                wsize = (input_size + layer->size) * 4;
            }
            //if (Convolutional == layer->type) weights = layer->weights[j];
            for (uint64_t k = 0; k < wsize; k++) {
                if (Convolutional == layer->type) widx = k;
                else widx = k + (j * input_size);
                PSFloat w = 0;
                ok = scanFile(f, PSFLOAT_FORMAT "%*[,\n]", 1, NULL, &w);
                if (!ok) {
                    if (verbose) printf("\n");
                    loadErr(
                        filename, f,"Layer %d neuron %d: invalid weight[%d]",
                        i, j, k
                    );
                    return 0;
                }
                if (Recurrent == layer->type) {
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
                         filename, f,"Layer %d neuron %d weight %d: "
                         "could not determine weights", i, j, k
                    );
                    return 0;
                }
                uint64_t matrix_len = PSMatrixLength(weights);
                if ((uint64_t) widx >= matrix_len) {
                    loadErr(
                         filename, f,"Layer %d neuron %d weight %d: "
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

static int loadLayerParameters(PSNeuralNetwork *network,
                               const char * filename,
                               FILE *f, int verbose)
{
    int ok, i;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (layer->type == Pooling || layer->type == Dropout) continue;
        int bias_count = 0, lidx = -1, wtype_count = 0, wcount = 0;
        ok = scanFile(
            f, "--- Layer[%d] Biases: %d ---\n", 2, NULL, &lidx, &bias_count
        );
        if (!ok) {
            loadErr(filename, f, "Missing layer %d biases header", i);
            return 0;
        }
        ok = (i == lidx);
        if (!ok) {
            loadErr(filename, f, "Invalid layer index %d, expected: %d",
                lidx, i
            );
            return 0;
        }
        int expected_bias_count = PSGetLayerParametersCount(
            layer, PARAM_TYPE_BIAS
        );
        int expected_weights_count = PSGetLayerParametersCount(
            layer, PARAM_TYPE_WEIGHT
        );
        ok = (bias_count == expected_bias_count);
        if (!ok) {
            loadErr(filename, f, "Layer[%d]: found %d biases, expected: %d",
                i, bias_count, expected_bias_count
            );
            return 0;
        }
        if (layer->biases == NULL && bias_count > 0) {
            loadErr(filename, f, "Layer[%d]: found %d biases, but "
                "layer->biases is NULL",
                i, bias_count
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
                    filename, f, "Layer[%d]: invalid bias %d", layer->index, j
                );
                return 0;
            }
            layer->biases[j] = bias;
        }
        ok = scanFile(f, "--- Layer[%d] Weights: %d,%d ---\n", 3, NULL,
                      &lidx, &wtype_count, &wcount);
        if (!ok) {
            loadErr(filename, f, "Missing layer %d weights header", i);
            return 0;
        }
        ok = (i == lidx);
        if (!ok) {
            loadErr(filename, f, "Invalid layer index %d, expected: %d",
                lidx, i
            );
            return 0;
        }
        ok = (wtype_count == layer->weight_types_count);
        if (!ok) {
            loadErr(filename, f, "Layer[%d]: found %d weight types, "
                "expected: %d", i, wtype_count, layer->weight_types_count
            );
            return 0;
        }
        ok = (wcount == expected_weights_count);
        if (!ok) {
            loadErr(filename, f, "Layer[%d]: found %d weights, "
                "expected: %d", i, wcount, expected_weights_count
            );
            return 0;
        }
        if (layer->weights == NULL && wtype_count > 0) {
            loadErr(filename, f, "Layer[%d]: found %d weights, but "
                "layer->weights is NULL",
                i, wcount
            );
            ok = 0;
            return 0;
        }
        for (int j = 0; j < layer->weight_types_count; j++) {
            PSMatrix weights = layer->weights[j];
            if (weights == NULL) {
                loadErr(filename, NULL, "Layer[%d]: weights[%d] is NULL", j);
                ok = 0;
                return 0;
            }
            uint64_t wlen = PSMatrixLength(weights), widx;
            for (widx = 0; widx < wlen; widx++) {
                char *fmt = PSFLOAT_FORMAT ",";
                if (widx == (wlen - 1))
                    fmt = PSFLOAT_FORMAT "\n";
                PSFloat w = 0;
                ok = scanFile(f, fmt, 1, NULL, &w);
                if (!ok) {
                    loadErr(
                        filename, f, "Layer[%d]: invalid weight[%d][%d]",
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
    }
    return 1;
}

static int loadLegacyGradients(PSNeuralNetwork *network, const char * filename,
                               FILE *f, PSGradient **gradients, int i)
{
    char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
        "," PSFLOAT_FORMAT "|";
    char sep[2];
    sep[0] = '\0';
    for (int j = 1; j < network->size; j++) {
        PSGradient *lgradients = gradients[j - 1];
        if (lgradients == NULL) continue;
        PSLayer *layer = network->layers[j];
        assert(layer != NULL);
        int lsize = 0, wsize = 0;
        if (layer->type == Pooling || layer->type == Dropout) continue;
        if (GRU == layer->type) {
            loadErr(filename, NULL, "GRU layers not supported in models "
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
            } else if (Recurrent == layer->type) wsize += lsize;
        }
        for (int k = 0; k < lsize; k++) {
            PSNeuron *n = layer->neurons[k];
            int matched = 0, ok;
            PSFloat *bias_p = lgradients->biases + k;
            if (!is_lstm) ok = scanFile(f, PSFLOAT_FORMAT "|", 1, NULL,bias_p);
            else {
                assert(n != NULL);
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
                    filename, f,
                    "Memory gradients %d, Layer %d, Gradient %d: "
                    "invalid bias!", i, j, k
                );
                ok = 0; return 0;
            }
            for (int w = 0; w < wsize; w++) {
                PSFloat gw = 0;
                ok = scanFile(
                    f, PSFLOAT_FORMAT "%[,\n]", 2, NULL, gw, sep
                );
                if (!ok) {
                    loadErr(
                        filename, f,
                        "Memory gradients %d, Layer %d, "
                        "Gradient %d: invalid weight %d",
                        i, j, k, w
                    );
                    return 0;
                }
                int idx = 0;
                if (Recurrent == layer->type) {
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

static int loadGradients(PSNeuralNetwork *network, const char * filename,
                         FILE *f, PSGradient **gradients, int i)
{
    for (int j = 1; j < network->size; j++) {
        int grad_idx = j - 1, gidx, bias_count, weight_count, ok, k;
        PSGradient *lgradients = gradients[grad_idx];
        if (lgradients == NULL) continue;
        ok = scanFile(
            f, "--- Gradient[%d] Biases: %d ---\n", 2, NULL,
            &gidx, &bias_count
        );
        if (!ok) {
            loadErr(
                filename, f, "Missing memory gradients[%d] "
                "biases[%d] header", i, j
            );
            return 0;
        }
        ok = (gidx == grad_idx);
        if (!ok) {
            loadErr(
                filename, f, "Invalid memory gradients[%d] index "
                "%d, expected %d (biases[%d])", i, gidx, grad_idx,j
            );
            return 0;
        }
        if (bias_count > 0 && lgradients->biases == NULL) {
            lgradients->biases = malloc(bias_count * sizeof(PSFloat));
            if (lgradients->biases == NULL) {
                PSPrintMemoryErrorMsg();
                loadErr(
                    filename, NULL, "Failed to allocate gradient "
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
                    filename, f, "Failed to load memory "
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
                filename, f, "Missing memory gradients[%d] "
                "weights[%d] header", i, j
            );
            return 0;
        }
        ok = (gidx == grad_idx);
        if (!ok) {
            loadErr(
                filename, f, "Invalid memory gradients[%d] index "
                "%d, expected %d (weights[%d])", i, gidx, grad_idx,j
            );
            return 0;
        }
        if (weight_count > 0 && lgradients->weights == NULL) {
            lgradients->weights = malloc(weight_count * sizeof(PSFloat));
            if (lgradients->weights == NULL) {
                PSPrintMemoryErrorMsg();
                loadErr(
                    filename, NULL, "Failed to allocate gradient "
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
                    filename, f, "Failed to load memory "
                    "gradients[%d] weight[%d][%d]", i, j, k
                );
                return 0;
            }
            lgradients->weights[k] = w;
        }
    }
    return 1;
}

int PSLoadNetwork(PSNeuralNetwork *network, const char* filename) {
    if (network == NULL) return 0;
    FILE *f = fopen(filename, "r");
    PSInfo("Loading network from %s", filename);
    if (f == NULL) {
        PSErr(__func__, "Could not open '%s'", filename);
        return 0;
    }
    int netsize, i;
    int empty = (network->size == 0);
    int verbose = PSLogLevel == PSLOGLEVEL_DEBUG;
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int epochs = 0, batch_count = 0, elements = 0, status = STATUS_UNTRAINED,
        batch_size = 0, rnn_mode = NonRecurrent,
        max_recurrent_output_steps = MAX_RECURRENT_OUTPUT_STEPS,
        eos_recurrent_output_index = -1, is_built = 0,
        acceleration = PSGlobalAcceleration;
    int ok = 1, has_model_def = 0;
    char sep[2];
    sep[0] = '\0';
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
                vers, PSYC_VERSION, vers, filename
            );
            goto final;
        }
        /* Older versions directly print model definition after version */
        has_model_def = scanFileNoMatch(f, ",");
        if (has_model_def) goto scan_model_def;
        else if (scanFileNoMatch(f, ":")) {
            /* Scan header info */
            PSModelFileHeader header = {{0}};
            ok = scanModelFileHeader(f, &header, filename);
            if (!ok) {
                loadErr(filename, NULL, "Invalid file header");
                goto final;
            }
            ok = scanFileNoMatch(f, "model:");
            if (!ok) {
                loadErr(filename, f, "Missing `model:` definition");
                goto final;
            }
            if (verbose) {
                PSInfo("Info for file '%s':", filename);
                printModelHeaderInfo(&header);
            }
            has_model_def = 1;
        }
    }
scan_model_def:
    if (has_model_def) {
        int idx = 0, val = 0;
        while (scanFile(f, "%d%[,\n]", 2, NULL, &val, sep)) {
            switch (idx++) {
                case 0:
                    network->flags |= val; break;
                case 1:
                    network->loss = getLossFunctionAtIndex(val);
                    break;
                case 2:  epochs = val; break;
                case 3:  batch_count = val; break;
                case 4:  status = val; break;
                case 5:  elements = val; break;
                case 6:  batch_size = val; break;
                case 7:  rnn_mode = (PSRecurrentNetworkMode) val; break;
                case 8:  max_recurrent_output_steps = val; break;
                case 9:  eos_recurrent_output_index = val; break;
                case 10: is_built = val; break;
                case 11: acceleration = val; break;
                default:
                    break;
            }
        }
        if (rnn_mode != NonRecurrent)
            PSSetRecurrentNetworkMode(network, rnn_mode);
        if (max_recurrent_output_steps > 0 || eos_recurrent_output_index >= 0) {
            if (network->rnn_options == NULL) {
                network->rnn_options =
                    calloc(1, sizeof(PSRecurrentNetworkOptions));
                if (network->rnn_options == NULL) {
                    PSPrintMemoryErrorMsg();
                    return 0;
                }
            }
            network->rnn_options->sequence_stop_criterion.max_steps =
                max_recurrent_output_steps;
            network->rnn_options->sequence_stop_criterion.eos =
                eos_recurrent_output_index;
        }
        network->status = status;
        if (status != STATUS_UNTRAINED) {
            if (network->training == NULL) {
                network->training = malloc(sizeof(PSTrainingInfo));
                network->training->requested_action = ACTION_NONE;
                network->training->debug_dump_to = NULL;
            }
            network->training->current_epoch = epochs;
            network->training->current_batch = batch_count;
            network->training->current_element = elements;
            network->training->batch_size = batch_size;
        }
        if (PSCompareVersion(vers, "0.4.0") >= 0) {
            PSEnableAcceleration(&(network->acceleration), acceleration);
            if (acceleration != network->acceleration)
                PSWarn("Could not enable all saved accelerations");
        } else if (network->flags & FLAG_ACCEL_DISABLED)
            network->acceleration = 0;
    }
    int legacy_model = (PSCompareVersion(vers, "0.9.0") < 0);
    if (!legacy_model) ok = scanFile(f, "layers:%d\n", 1, NULL, &netsize);
    else ok = scanFile(f, "%d:", 1, NULL, &netsize);
    if (!ok) {
        loadErr(filename, f, "Missing network size definition");
        goto final;
    }
    if (netsize == 0) {
        loadErr(filename, NULL, "Empty network model!");
        ok = 0;
        goto final;
    }
    if (!empty && network->size != netsize) {
        loadErr(filename, NULL, "Network size differs!");
        ok = 0;
        goto final;
    }
    if (legacy_model)
        ok = loadLegacyLayerDefinitions(network,vers,netsize,empty,filename, f);
    else ok = loadLayerDefinitions(network, vers, netsize, empty, filename, f);
    if (!ok) goto final;
    /* Load layer parameters */
    if (legacy_model)
        ok = loadLegacyLayerParameters(network, filename, f, verbose);
    else
        ok = loadLayerParameters(network, filename, f, verbose);
    if (!ok) goto final;

    if (verbose) printf("\n");
    /* Check for traing data */
    if (scanFileNoMatch(f, MODEL_TRAINING_DATA_SEP)) {
        /* Model file has training data */
        int numgradients = 0;
        ok = scanFile(f, "memory_gradients:%d\n", 1, NULL, &numgradients);
        if (!ok) {
            loadErr(filename, f, "Invalid or missing 'memory_gradients'");
            goto final;
        }
        ok = initTrainingContext(network, numgradients);
        if (!ok) {
            PSErr(__func__, "Failed to initialize network training data");
            goto final;
        }
        PSTrainingOptions *topts = PSGetNetworkTrainingOptions(network);
        if (topts == NULL) {
            PSErr(__func__, "Invalid network training data (missing options)");
            goto final;
        }
        PSGradient **memg1 = NULL, **memg2 = NULL;
        if (numgradients > 0) {
            int foundgradients = PSGetTrainingMemoryGradients(
                network, &memg1, &memg2
            );
            ok = (foundgradients == numgradients);
            if (ok) {
                ok = memg1 != NULL;
                if (ok && numgradients >= 2) ok = memg2 != NULL;
            }
            if (!ok) {
                loadErr(
                    filename, NULL, "Invalid network training data (missing "
                    "gradients)"
                );
                goto final;
            }
        }
        ok = scanFileNoMatch(f, "training_options:");
        if (!ok) {
            loadErr(filename, f, "Missing 'training_options'");
            goto final;
        }
        ok = scanTrainingOptions(f, topts, filename);
        if (!ok) goto final;
        for (i = 0; i < numgradients; i++) {
            PSGradient **memg = NULL;
            if (i == 0) memg = memg1;
            else if (i == 1) memg = memg2;
            else break;
            int gidx = -1;
            ok = scanFile(f, "memory_gradients[%d]:\n", 1, NULL, &gidx);
            if (ok && gidx != i) ok = 0;
            if (!ok) {
                loadErr(filename, f, "Invalid memory gradient header");
                goto final;
            }
            if (legacy_model)
                ok = loadLegacyGradients(network, filename, f, memg, i);
            else
                ok = loadGradients(network, filename, f, memg, i);
            if (!ok) {
                loadErr(
                    filename, NULL, "Failed to load memory gradients %s", i
                );
                goto final;
            }
        }
    }
    if (is_built) PSBuildNetwork(network);
final:
    if (f != NULL) fclose(f);
    return ok;
}

int PSSaveNetwork(PSNeuralNetwork *network, const char* filename) {
    if (network->size == 0) {
        PSErr(__func__, "Empty network!");
        return 0;
    }
    FILE *f = fopen(filename, "w");
    PSInfo("Saving network to %s", filename);
    if (f == NULL) {
        PSErr(__func__, "Cannot open %s for writing!", filename);
        return 0;
    }
    int i, j, k, opts = 0, ok = 1;
    int loss_function = getLossFunctionIndex(network->loss);
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
    int current_epoch = 0, current_batch = 0, current_element = 0,
        batch_size = 0;
    if (network->training != NULL) {
        current_epoch = network->training->current_epoch;
        current_batch = network->training->current_batch;
        current_element = network->training->current_element;
        batch_size = network->training->batch_size;
    }
    PSRecurrentNetworkMode rnn_mode = PSGetRecurrentNetworkMode(network);
    int max_steps = 0, eos = -1;
    if (network->rnn_options != NULL) {
        max_steps = network->rnn_options->sequence_stop_criterion.max_steps;
        eos = network->rnn_options->sequence_stop_criterion.eos;
    }
    fprintf(f, "model:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", network->flags,
            loss_function, current_epoch, current_batch, network->status,
            current_element, batch_size, (int) rnn_mode,
            max_steps, eos, PSIsNetworkBuilt(network));
    fprintf(f, "layers:%d\n", network->size);
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        writeLayerDefinition(layer, f);
    }
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        if (Pooling == ltype || layer->type == Dropout) continue;
        int bias_count = PSGetLayerParametersCount(layer, PARAM_TYPE_BIAS),
            weights_count = PSGetLayerParametersCount(layer, PARAM_TYPE_WEIGHT);
        fprintf(f, "--- Layer[%d] Biases: %d ---\n", i, bias_count);
        if (bias_count > 0) {
            if (layer->biases == NULL) {
                PSErr(__func__, "Layer[%d]: biases are NULL", i);
                fclose(f);
                return 0;
            }
            for (j = 0; j < bias_count; j++) {
                if (j > 0) fprintf(f, ",");
                writeSerializedFloat(f, layer->biases[j], opts);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "--- Layer[%d] Weights: %d,%d ---\n",
                i, layer->weight_types_count, weights_count);
        if (layer->weight_types_count > 0) {
            if (layer->weights == NULL) {
                PSErr(__func__, "Layer[%d]: weights are NULL", i);
                fclose(f);
                return 0;
            }
            for (j = 0; j < layer->weight_types_count; j++) {
                PSMatrix weights = layer->weights[j];
                if (weights == NULL) {
                    PSErr(__func__, "Layer[%d]: weights[%d] are NULL", i, j);
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
    }
    PSTrainingOptions *topts = PSGetNetworkTrainingOptions(network);
    PSGradient **memg1 = NULL, **memg2 = NULL;
    int numgradients = PSGetTrainingMemoryGradients(network, &memg1, &memg2);
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
        if (numgradients > 0) {
            if (memg1 == NULL) {
                PSErr(__func__, "Training memory gradient 1 is NULL");
                ok = 0; goto final;
            }
            fprintf(f, "memory_gradients[0]:\n");
            ok = writeGradients(network, memg1, opts, f);
            if (!ok) goto final;
            if (memg2 != NULL) {
                fprintf(f, "memory_gradients[1]:\n");
                ok = writeGradients(network, memg2, opts, f);
                if (!ok) goto final;
            }
        }
    }
final:
    fclose(f);
    return ok;
}

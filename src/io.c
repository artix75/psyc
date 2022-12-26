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
#include <sys/utsname.h>

#include "psyc.h"
#include "config.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"
#include "log.h"
#include "buildinfo.h"

#define DATA_LAYER_MIN_ARGC 3
#define OPT_FLOAT_FORMAT_HEX (1 << 0)
#define OPT_FLOAT_FORMAT_DBL (1 << 1)
#define MODEL_TRAINING_DATA_SEP "------ training data ------\n"

typedef struct PSModelFileHeader {
    char    git_sha1[9];
    int     git_dirty;
    char    git_branch[256];
    size_t  float_size;
    int     archbits;
    int     avx;
    int     vdsp;
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
    if (f != NULL) PSLog(PSLOGLEVEL_ERROR, " (offset: %ld):\n", ftello(f));
    else PSLog(PSLOGLEVEL_ERROR, "\n");
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
        } else if (strcmp("vdsp", propname) == 0) {
            header->vdsp = atoi(val);
            if (header->vdsp < 0 || header->vdsp > 1) goto fail;
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
            PSTrainingOptimization optimization =
                (PSTrainingOptimization) string2int(val, &ok);
            if (!ok || (int) optimization < 0) goto fail;
            opts->optimization = optimization;
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
    int i, j, size;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        PSHyperParameters *lparams = layer->hyper_parameters;
        int fcount = 1, weights_size = 0;
        if ((ltype == Convolutional || ltype == Pooling ||
            ltype == FullyConnected) && lparams != NULL)
        {
            PSFloat *params = lparams->parameters;
            fcount = (int) (params[PARAM_FEATURE_COUNT]);
        }
        if (fcount < 0) fcount = 1;
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        if (ltype == Convolutional || ltype == Pooling) size = fcount;
        else size = layer->size;
        PSSharedParams *shared = NULL;
        if (ltype == Convolutional) {
            shared = PSGetConvSharedParams(layer);
            if (!shared) {
                PSErr(
                    NULL, "Shared params for Convolutional layer %d "
                    "are NULL!\n", i
                );
                return 0;
            }
            weights_size = shared->weights_size;
        } else {
            PSNeuron *n = layer->neurons[0];
            assert(n != NULL);
            weights_size = n->weights_size;
            /*if (LSTM == ltype) weights_size += 4;*/
        }
        for(j = 0; j < size; j++) {
            PSGradient *gradient = &(lgradients[j]);
            if (gradient == NULL) {
                PSErr(NULL, "Layer[%d] Gradient[%d] is NULL", i, j);
                return 0;
            }
            if (gradient->weights == NULL) {
                PSErr(NULL, "Layer[%d] Gradient[%d] weights is NULL", i, j);
                return 0;
            }
            if (ltype != LSTM) writeSerializedFloat(f, gradient->bias, 0);
            else {
                PSNeuron *neuron = layer->neurons[j];
                PSFloat *gbiases = PSGetLSTMGradientBiases(neuron, gradient);
                writeSerializedFloatArray(f, 4, ",", opts, gbiases);
            }
            fprintf(f, "|");
            writeSerializedFloatArray(
                f, weights_size, ",", opts, gradient->weights
            );
            fprintf(f, "\n");
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
    char *lstm_fmt = PSFLOAT_FORMAT "," PSFLOAT_FORMAT "," PSFLOAT_FORMAT
        "," PSFLOAT_FORMAT "|";
    int netsize, i, j, k;
    int empty = (network->size == 0);
    int verbose = PSLogLevel == PSLOGLEVEL_DEBUG;
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int epochs = 0, batch_count = 0, elements = 0, status = STATUS_UNTRAINED,
        batch_size = 0, rnn_mode = NonRecurrent,
        max_recurrent_output_steps = MAX_RECURRENT_OUTPUT_STEPS,
        eos_recurrent_output_index = -1, is_built = 0,
        acceleration = PSGlobalAcceleration;
    int matched = 0, ok = 1, has_model_def = 0;
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
            PSModelFileHeader header = {0};
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
    ok = scanFile(f, "%d:", 1, NULL, &netsize);
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
    int min_argc = 1;
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
        /* Simple FullyConnected layers with no flags, no dropout and no
         * hyper-parameters are only saved as a single integer (layer->size) */
        /* fputs(fmt, stderr); */
        matched = scanFile(f, "%d%[,\n]", 2, NULL, &lsize, sep);
        if (!matched) {
            /* Try to parse more complex layer definitions declared as an
             * array of numeric values: [type, argc, args...]. */
            int type = 0, arg = 0;
            PSFloat argf = 0.0;
            argc = 0;
            ok = scanFile(f, "[%d,%d", 2, NULL, &type, &argc);
            if (!ok) {
                loadErr(filename, f, "Invalid layer def: layer[%d]", i);
                goto final;
            }
            if (argc == 0) {
                loadErr(
                    filename, NULL,
                    "Layer %d must have at least 1 argument (size)", i
                );
                ok = 0;
                goto final;
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
                    goto final;
                }
                if (aidx == 0) lsize = arg;
                else if (min_argc > 1 && aidx == 1) lflags = arg;
                else if (min_argc > 2 && aidx == 2) dropout = argf;
                else args[aidx - min_argc] = arg;
            }
            argc -= min_argc;
            ok = scanFile(f, "]%[,\n]", 1, NULL, sep);
            if (!ok) goto final;
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                loadErr(filename, NULL, "Layer %d size %d differs from %d!",
                    i, layer->size, lsize);
                ok = 0; goto final;
            }
            if (ltype != layer->type) {
                loadErr(filename, NULL, "Layer %d type %d differs from %d!",
                        i, (int) (layer->type), (int) ltype);
                ok = 0; goto final;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                PSHyperParameters *params = layer->hyper_parameters;
                if (params == NULL) {
                    PSErr(__func__, "Layer %d params are NULL!", i);
                    ok = 0; goto final;
                }
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= params->count) break;
                    int arg = args[aidx];
                    PSFloat val = params->parameters[aidx];
                    if (arg != (int) val) {
                        loadErr(
                            filename, NULL,
                            "Layer %d arg[%d] %d diff. from %d!",
                            i, aidx,(int) val, arg
                        );
                        ok = 0; goto final;
                    }
                }
            }
            layer->dropout = dropout;
        } else {
            layer = NULL;
            PSHyperParameters *params = NULL;
            if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                params = PSCreateHyperParamenters(param_c);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= param_c) break;
                    int arg = args[aidx];
                    params->parameters[aidx] = (PSFloat) arg;
                }
                if (aidx < (param_c - 1)) {
                    for (; aidx < (param_c - 1); aidx++)
                        params->parameters[aidx] = 0.0;
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            } else {
                if (network->size == 0 && (lflags & FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    network->flags |= FLAG_ONEHOT;
                } else if (argc > 0) {
                    params = PSCreateHyperParamenters(argc);
                    for (aidx = 0; aidx < argc; aidx++) {
                        int arg = args[aidx];
                        params->parameters[aidx] = (PSFloat) arg;
                    }
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            }
            if (layer == NULL) {
                PSErr(__func__, "Could not create layer %d", i);
                ok = 0; goto final;
            }
            layer->flags |= lflags;
            layer->dropout = dropout;
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = 0;
        PSSharedParams *shared = NULL;
        if (layer->type == Convolutional) {
            shared = PSGetConvSharedParams(layer);
            if (shared == NULL) {
                loadErr(filename, NULL, "Layer %d, missing shared params!",i);
                ok = 0; goto final;
            }
            lsize = shared->feature_count;
        } else if (layer->type == Pooling) {
            continue;
        } else lsize = layer->size;
        int is_lstm = (LSTM == layer->type);
        int llen = 0;
        for (j = 0; j < lsize; j++) {
            PSFloat bias = 0;
            int wsize = 0;
            PSFloat *weights = NULL;
            /* LSTM biases */
            PSFloat cb = 0.0, ib = 0.0, ob = 0.0, fb = 0.0;
            matched = 0;
            if (!is_lstm) ok = scanFile(f, PSFLOAT_FORMAT "|", 1, NULL, &bias);
            else ok = scanFile(f, lstm_fmt, 4, &matched, &cb, &ib, &ob, &fb);
            if (!ok || (is_lstm && matched < 4)) {
                if (verbose) printf("\n");
                loadErr(
                    filename, f, "Layer %d, neuron %d: invalid bias!", i, j
                );
                ok = 0; goto final;
            }
            if (shared == NULL) {
                PSNeuron *neuron = layer->neurons[j];
                wsize = neuron->weights_size;
                neuron->bias = bias;
                weights = neuron->weights;
                if (is_lstm) {
                    PSLSTMCell *cell = PSGetLSTMCell(neuron);
                    assert(cell != NULL);
                    cell->candidate_bias = cb;
                    cell->input_bias = ib;
                    cell->output_bias = ob;
                    cell->forget_bias = fb;
                }
            } else {
                shared->biases[j] = bias;
                wsize = shared->weights_size;
                weights = shared->weights[j];
            }
            for (k = 0; k < wsize; k++) {
                PSFloat w = 0;
                ok = scanFile(f, PSFLOAT_FORMAT "%*[,\n]", 1, NULL, &w);
                if (!ok) {
                    if (verbose) printf("\n");
                    loadErr(
                        filename, f,"Layer %d neuron %d: invalid weight[%d]",
                        i, j, k
                    );
                    goto final;
                }
                weights[k] = w;
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
    if (verbose) printf("\n");
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
            for (j = 1; j < netsize; j++) {
                PSLayer *layer = network->layers[j];
                assert(layer != NULL);
                PSSharedParams *shared = NULL;
                int lsize = 0, weights_size = 0, w;
                if (layer->type == Convolutional) {
                    shared = PSGetConvSharedParams(layer);
                    assert(shared != NULL);
                    lsize = shared->feature_count;
                    weights_size = shared->weights_size;
                } else if (layer->type == Pooling) {
                    continue;
                } else lsize = layer->size;
                PSGradient *lgradients = memg[j - 1];
                assert(lgradients != NULL);
                int is_lstm = (LSTM == layer->type);
                for (k = 0; k < lsize; k++) {
                    PSNeuron *n = layer->neurons[k];
                    if (weights_size == 0) weights_size = n->weights_size;
                    PSGradient *gradient = &(lgradients[k]);
                    assert(gradient != NULL);
                    matched = 0;
                    if (!is_lstm) {
                        ok = scanFile(
                            f, PSFLOAT_FORMAT "|", 1, NULL, &(gradient->bias)
                        );
                    } else {
                        assert(n != NULL);
                        PSFloat *biases = PSGetLSTMGradientBiases(n, gradient);
                        ok = scanFile(
                            f, lstm_fmt, 4, &matched,
                            biases, biases + 1, biases + 2, biases + 3
                        );
                    }
                    if (!ok || (is_lstm && matched < 4)) {
                        printf("\n");
                        loadErr(
                            filename, f,
                            "Memory gradients %d, Layer %d, Gradient %d: "
                            "invalid bias!", i, j, k
                        );
                        ok = 0; goto final;
                    }
                    for (w = 0; w < weights_size; w++) {
                        PSFloat *gw = gradient->weights + w;
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
                            goto final;
                        }
                    }
                }
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
    int vdsp_available = (
        PSIsAccelerationAvailable(PSAcceleration_vDSP) ? 1 : 0
    );
    fprintf(
        f, "--v%s:git=%s/%s-%s;float_size=%zu;archbits=%d;avx=%d;"
        "vdsp=%d,sys=%s,%s,%s;global_flags=%d;acceleration=%d;savetime=%ld\n",
        PSYC_VERSION, PSYC_GIT_SHA, PSYC_GIT_DIRTY, PSYC_GIT_BRANCH,
        sizeof(PSFloat), ((sizeof(long) == 8) ? 64 : 32), avx_available,
        vdsp_available, sysinfo.sysname, sysinfo.release, sysinfo.machine,
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
    fprintf(f, "%d:", network->size);
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        if (i > 0) fprintf(f, ",");
        int flags = layer->flags;
        PSFloat dropout = layer->dropout;
        PSHyperParameters *params = layer->hyper_parameters;
        if (FullyConnected == ltype && !flags && !params && dropout <= 0.0)
            fprintf(f, "%d", layer->size);
        else if (params) {
            int argc = params->count;
            fprintf(
                f, "[%d,%d,%d,%d,%g",
                (int) ltype, DATA_LAYER_MIN_ARGC + argc, layer->size,
                layer->flags, dropout
            );
            for (j = 0; j < argc; j++) {
                fprintf(f, ",%d", (int) (params->parameters[j]));
            }
            fprintf(f, "]");
        } else {
            fprintf(
                f, "[%d,%d,%d,%d,%g]",
                (int) ltype, DATA_LAYER_MIN_ARGC, layer->size, flags, dropout
            );
        }
    }
    fprintf(f, "\n");
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        int lsize = layer->size;
        if (Convolutional == ltype) {
            PSSharedParams *shared = PSGetConvSharedParams(layer);
            if (shared == NULL) {
                PSErr(__func__, "Layer[%d]: shared params are NULL!", i);
                fclose(f);
                return 0;
            }
            int feature_count = shared->feature_count;
            if (feature_count < 1) {
                PSErr(__func__, "Layer[%d]: feature count must be >= 1!", i);
                fclose(f);
                return 0;
            }
            for (j = 0; j < feature_count; j++) {
                PSFloat bias = shared->biases[j];
                PSFloat *weights = shared->weights[j];
                writeSerializedFloat(f, bias, opts);
                fprintf(f, "|");
                for (k = 0; k < shared->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    PSFloat w = weights[k];
                    writeSerializedFloat(f, w, opts);
                }
                fprintf(f, "\n");
            }
        }
        else if (Pooling == ltype) continue;
        else {
            int is_lstm = (LSTM == ltype);
            for (j = 0; j < lsize; j++) {
                PSNeuron *neuron = layer->neurons[j];
                if (!is_lstm) {
                    writeSerializedFloat(f, neuron->bias, opts);
                } else {
                    PSLSTMCell *cell = PSGetLSTMCell(neuron);
                    assert(cell != NULL);
                    writeSerializedFloats(
                        f, 4, ",", opts,
                        cell->candidate_bias,
                        cell->input_bias,
                        cell->output_bias,
                        cell->forget_bias
                    );
                }
                fprintf(f, "|");
                for (k = 0; k < neuron->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    PSFloat w = neuron->weights[k];
                    writeSerializedFloat(f, w, opts);
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
            fprintf(f, ",optimization=%d", (int) topts->optimization);
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

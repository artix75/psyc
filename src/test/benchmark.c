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
#include <math.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/utsname.h>

#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../dropout.h"
#include "../lstm.h"
#include "../gru.h"
#include "../normalization.h"
#include "../dataset.h"
#include "../maths.h"
#include "../activation.h"
#include "../optimization.h"
#include "../utils.h"
#include "../debug.h"
#include "../log.h"
#ifdef USE_AVX
#include "../avx.h"
#endif

#define UNUSED(V) ((void) V)

#define DEFAULT_BENCHMARK_SAMPLES 10

#define PS_BENCHMARK_PREAMBLE(cfg, results) do {\
    assert(num_results != NULL);\
    assert(results != NULL);\
    if (cfg->samples <= 0) cfg->samples = DEFAULT_BENCHMARK_SAMPLES;\
    results->samples = cfg->samples;\
} while (0)

#define PS_INIT_BENCHMARK(cfg, index_ptr, results, bm_name) do {\
    if (bm_name != NULL) sprintf(res->name, "%s", bm_name);\
    else res->name[0] = '\0';\
    if (cfg->samples <= 0) cfg->samples = DEFAULT_BENCHMARK_SAMPLES;\
    results->samples = cfg->samples;\
    results->index = *index_ptr;\
} while(0)

#define INTARGS(...) ((void *)((int []){__VA_ARGS__}))
#define FLOATARGS(...) ((void *)((PSFloat []){__VA_ARGS__}))

#define PSBenchmarkMeasure(results, code) do {\
    struct timeval st, et;\
    results->tot_time_us = 0;\
    results->performed = 0;\
    if (results->samples <= 0) results->samples = DEFAULT_BENCHMARK_SAMPLES;\
    if (!disable_prerun && results->index == 0) {\
        code;\
    }\
    while (results->performed++ < results->samples) {\
        gettimeofday(&st, NULL);\
        code;\
        gettimeofday(&et, NULL);\
        results->tot_time_us += PSGetElapsedTimeUS(st, et);\
        if (verbose) {\
            printf(\
                "%s: sample %2d = %ldus\n", results->name, results->performed,\
                PSGetElapsedTimeUS(st, et)\
            );\
        }\
    }\
    results->avg_time_us = results->tot_time_us / results->samples;\
} while (0)

struct PSBenchmarkConfig;
struct PSBenchmarkResults;
typedef int (* PSBenchmarkFunction) (struct PSBenchmarkConfig *,
                                     int *num_results,
                                     struct PSBenchmarkResults *);

typedef struct PSBenchmarkResults {
    char name[256];
    int index;
    int samples;
    int performed;
    time_t tot_time_us;
    time_t avg_time_us;
} PSBenchmarkResults;

typedef struct PSBenchmarkConfig {
    const char *name;
    int *tag;
    int samples;
    int max_results;
    PSBenchmarkFunction do_benchmark;
    int argc;
    void *argv;
    int size_args;
    void *default_argv;
} PSBenchmarkConfig;

/* Globals */

char *exclude_str = NULL;
char *match_str = NULL;
int cache_enabled = 1;
int auto_accel = 0;
int min_size = 0, max_size = 0;
int int_argc = 0, flt_argc = 0;
int int_argv[50] = {0};
PSFloat flt_argv[50] = {0};
void *user_argv = NULL;
char *csv_output = NULL;
char *json_output = NULL;
int disable_prerun = 0;
int verbose = 0;

/* Forward decl. */
PSGradient **backprop(PSModel *model, PSFloat *x, PSFloat *y,
                      PSTrainingOptions *opts, PSGradient **gradients);

/* Helpers */
static int compareResults(const void * a, const void * b) {
    assert(a != NULL);
    assert(b != NULL);
    PSBenchmarkResults *res_a = (PSBenchmarkResults *) a;
    PSBenchmarkResults *res_b = (PSBenchmarkResults *) b;
    return res_a->avg_time_us - res_b->avg_time_us;
}

static int parseUserArgv(char *argvstr, void *argv, int is_int) {
    assert(argvstr != NULL);
    assert(argv != NULL);
    int argc = 0;
    char *p = argvstr, *valstr = p, *errptr = NULL, remaining = strlen(argvstr);
    while (p != NULL && ((p = strchr(p, ',')) || remaining > 0)) {
        if (argc >= 50) break;
        if (p != NULL) *(p++) = '\0';
        remaining -= strlen(valstr);
        if (is_int) {
            int val = strtol(valstr, &errptr, 10);
            if (val == 0 && errptr == valstr) {
                fprintf(stderr, "ERROR: invalid integer arg. '%s'\n", valstr);
                exit(1);
            }
            ((int *) argv)[argc++] = val;
        } else {
#ifdef PS_DOUBLE_PRECISION
            PSFloat val = strtod(valstr, &errptr);
#else
            PSFloat val = strtof(valstr, &errptr);
#endif
            if (val == 0 && errptr == valstr) {
                fprintf(stderr, "ERROR: invalid float arg. '%s'\n", valstr);
                exit(1);
            }
            ((PSFloat *) argv)[argc++] = val;
        }
        if (p != NULL) valstr = p;
    }
    return argc;
}

static int setDefaultArgv(PSBenchmarkConfig *cfg, int user_argc,
                          void *user_argv, int is_int)
{
    int default_argc = cfg->argc - user_argc;
    if (default_argc <= 0) return 0;
    if (cfg->default_argv == NULL) return 0;
    size_t elem_sz = (is_int ? sizeof(int) : sizeof(PSFloat));
    cfg->argv = calloc(cfg->argc, elem_sz);
    if (cfg->argv == NULL) {
        PSWarn("Benchmark '%s': failed to allocate %s argv of size %d",
               cfg->name, (is_int ? "int" : "PSFloat"), cfg->argc);
        return 0;
    }
    size_t default_sz = user_argc * elem_sz;
    uint8_t *default_src = ((uint8_t *) cfg->default_argv) + default_sz;
    uint8_t *default_dst = ((uint8_t *) cfg->argv) + default_sz;
    if (user_argc > 0) memcpy(cfg->argv, user_argv, user_argc * elem_sz);
    memcpy(default_dst, default_src, default_argc);
    return default_argc;
}

char *PSBenchmarkName(PSBenchmarkConfig *cfg) {
    static char fmtname[255] = {0};
    if (cfg == NULL || cfg->name == NULL) return NULL;
    char *name = (char *) cfg->name;
    int newlen = 0;
    if (cfg->argv != NULL && cfg->argc > 0) {
        void *argv = cfg->argv;
        int available = 254, remaining = strlen(name), i = 0;
        char *p = name, *cur = p, *fmtname_p = fmtname;
        while ((p = strchr(cur, '%')) || remaining > 0) {
            if (p != NULL && i < cfg->argc) {
                char *fmt = NULL;
                int len = (p - cur);
                remaining -= len;
                if (len == 0) break;
                char c = *(++p);
                int is_int = 0;
                if ((is_int = (c == 'd'))) fmt = "%d";
                else if (c == 'f') fmt = "%f";
                else if (c == 0) break;
                else goto next;
                p++;
                remaining -= 2;

                if (remaining < 1) break;
                newlen += len;
                if (newlen >= 254) break;
                available -= len;
                if (available <= 0) break;
                strncpy(fmtname_p, cur, len);
                fmtname_p += len;
                if (is_int) {
                    len = snprintf(
                        fmtname_p, available, fmt, ((int *)argv)[i++]
                    );
                } else {
                    len = snprintf(
                        fmtname_p, available, fmt, ((PSFloat *) argv)[i++]
                    );
                }
                available -= len;
                fmtname_p += len;
                newlen += len;
            } else {
                if (cur != name && remaining > 0) {
                    if (remaining > available) remaining = available;
                    strncpy(fmtname_p, cur, remaining);
                    newlen += remaining;
                    remaining = 0;
                }
                break;
            }
next:
            cur = p;
        }
    }
    if (newlen > 0) {
        name = fmtname;
        if ((newlen + 1) < 255) fmtname[newlen + 1] = '\0';
    }
    return name;
}

void PSPrintBenchmarkResults(PSBenchmarkResults *results, int num_results) {
    assert(results != NULL);
    if (num_results <= 0) return;
    qsort(results, (size_t) num_results, sizeof(*results), compareResults);
    PSBenchmarkResults *res = results;
    printf("   %40s %15s\n", "AVG", "AVG (human)");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < num_results; i++) {
        const char *name = res->name;
        if (name[0] == '\0') name = "Main";
        int len = strlen(name), pad = 35 - len;
        char *elapsed_str = PSGetElapsedTimeString(res->avg_time_us, 0);
        printf(
            PSCOLOR_CYAN " -> %s:" PSCOLOR_RESET " %*ldus %15s\n",
            name, pad, res->avg_time_us, elapsed_str
        );
        free(elapsed_str);
        res++;
    }
}

void PSWriteBenchmarkResultsToCSV(PSBenchmarkConfig *cfg,
                                  PSBenchmarkResults *results,
                                  int num_results,
                                  FILE *f)
{
    assert(results != NULL);
    if (num_results <= 0) return;
    PSBenchmarkResults *res = results;
    char *cfgname = PSBenchmarkName(cfg);
    for (int i = 0; i < num_results; i++) {
        const char *name = res->name;
        if (name[0] == '\0') name = "Main";
        char *elapsed_str = PSGetElapsedTimeString(res->avg_time_us, 0);
        fprintf(f, "\"%s\",\"%s\",%ld,\"%s\"\n", cfgname, name,
                res->avg_time_us, elapsed_str);
        free(elapsed_str);
        res++;
    }
}

void PSWriteBenchmarkResultsToJSON(PSBenchmarkConfig *cfg,
                                   PSBenchmarkResults *results,
                                   int num_results, int index,
                                   FILE *f)
{
    assert(results != NULL);
    if (num_results <= 0) return;
    PSBenchmarkResults *res = results;
    char *cfgname = PSBenchmarkName(cfg);
    char *sep = (index > 0 ? ",\n" : "");
    fprintf(f, "%s    {\n        \"name\": ", sep);
    if (cfgname != NULL) fprintf(f, "\"%s\",\n", cfgname);
    else fprintf(f, "null,\n");
    fprintf(f, "        \"results\": [\n");
    for (int i = 0; i < num_results; i++) {
        sep = (i < (num_results - 1) ? ",\n" : "\n");
        const char *name = res->name;
        if (name[0] == '\0') name = "Main";
        char *elapsed_str = PSGetElapsedTimeString(res->avg_time_us, 0);
        fprintf(f, "            {\n");
        fprintf(f, "                \"name\": \"%s\",\n", name);
        fprintf(f, "                \"avg_time_us\": %ld,\n",
                res->avg_time_us);
        fprintf(f, "                \"avg_time_human\": \"%s\"\n",
                elapsed_str);
        fprintf(f, "            }%s", sep);
        res++;
        free(elapsed_str);
    }
    fprintf(f, "        ]\n");
    fprintf(f, "    }");
}

int PSBecnhmarkSize(PSBenchmarkConfig *cfg) {
    if (cfg->argv == NULL || cfg->argc <= 0 || cfg->size_args == 0) return 0;
    int size = 0, i;
    int *argv = (int *) cfg->argv;
    for (i = 0; i < cfg->argc; i++) {
        int is_size = ((i + 1) & cfg->size_args);
        if (!is_size) continue;
        int n = argv[i];
        if (n <= 0) continue;
        if (i == 0) size = 1;
        size *= n;
    }
    return size;
}

/* Benchmark tags */
static int maths_tag = 1, activation_tag = 1,
           optimization_tag = 1, fullnet_tag = 1, convnet_tag = 1,
           rnn_tag = 1, lstm_tag = 1, gru_tag = 1,
           normalization_tag = 1;

static int *tag_ptrs[] = {
    NULL, &maths_tag, &activation_tag, &optimization_tag,
    &fullnet_tag, &convnet_tag, &rnn_tag, &lstm_tag, &gru_tag,
    &normalization_tag
};

static char*tag_ids[] = {
    NULL, "maths", "activation", "optimization", "fully-connected",
    "convolutional", "rnn", "lstm", "gru", "normalization"
};

static void printTagList(void) {
    size_t i;
    for (i = 0; i < (sizeof(tag_ids) / sizeof(char*)); i++) {
        char *tag_id = tag_ids[i];
        if (tag_id == NULL) continue;
        printf("%s\n", tag_id);
    }
}

static int *tagEnabledPointerByID(char *id) {
    int *ptr = NULL;
    size_t i;
    for (i = 0; i < (sizeof(tag_ids) / sizeof(char*)); i++) {
        char *tag_id = tag_ids[i];
        if (tag_id == NULL) continue;
        if (strcasecmp(tag_id, id) == 0) {
            ptr = tag_ptrs[i];
            break;
        }
    }
    return ptr;
}

static int setTagEnabledStatus(char *tag_id, int enabled) {
    int *ptr = tagEnabledPointerByID(tag_id);
    if (ptr == NULL) {
        fprintf(stderr, "ERROR: invalid tag ID: '%s'\n", tag_id);
        return 0;
    }
    *ptr = enabled;
    return 1;
}

static void disableAllTags(void) {
    size_t i;
    for (i = 0; i < (sizeof(tag_ptrs) / sizeof(int*)); i++) {
        int *ptr = tag_ptrs[i];
        if (ptr == NULL) continue;
        *ptr = 0;
    }
}

static PSModel *makeCIFARLikeCNN(void) {
    PSModel *model = PSModelCreate("CIFAR CNN");
    if (model == NULL) return NULL;
    PSLayer *l = NULL;
    l = PSAddLayer(model, FullyConnected, PS_CIFAR_IMAGE_SIZE, PSLDEF(
        .output_depth = 3,
        .output_columns = 32,
        .output_rows = 32
    ));
    if (!l) goto fail;
    l = PSAddConvolutionalLayer(model, PSLDEF(
        .output_depth = 16,
        .filter_width = 5,
        .filter_height = 5,
        .padding = 2,
        .stride = 1,
        .activation = PSRelu
    ));
    if (!l) goto fail;
    l = PSAddPoolingLayer(model, PSLDEF(
        .stride = 2,
        .filter_width = 2,
        .filter_height = 2
    ));
    if (!l) goto fail;
    l = PSAddConvolutionalLayer(model, PSLDEF(
        .output_depth = 20,
        .filter_width = 5,
        .filter_height = 5,
        .padding = 2,
        .stride = 1,
        .activation = PSRelu
    ));
    if (!l) goto fail;
    l = PSAddPoolingLayer(model, PSLDEF(
        .stride = 2,
        .filter_width = 2,
        .filter_height = 2
    ));
    if (!l) goto fail;
    l = PSAddConvolutionalLayer(model, PSLDEF(
        .output_depth = 20,
        .filter_width = 5,
        .filter_height = 5,
        .padding = 2,
        .stride = 1,
        .activation = PSRelu
    ));
    if (!l) goto fail;
    l = PSAddPoolingLayer(model, PSLDEF(
        .stride = 2,
        .filter_width = 2,
        .filter_height = 2
    ));
    if (!l) goto fail;
    l = PSAddLayer(model, SoftMax, 10, NULL);
    if (!l) goto fail;
    return model;
fail:
    if (model != NULL) PSModelFree(model);
    return NULL;
}

/* Benchmark functions */
int dummyBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                   PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    PSBenchmarkMeasure(results, sleep(1));
    *num_results = 1;
    return 1;
}

int mathsDotProductBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsDotBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                      PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc == 2 && cfg->argv != NULL);
    int *intargs = (int *) cfg->argv;
    int rows = intargs[0], cols = intargs[1], ok = 1;
    assert(rows > 0);
    assert(cols > 0);
    PSFloat dest[cols];
    PSMatrix x = PSMatrixWithGaussianRandom(1, 2, rows, cols);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, cols);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
    opts.argtype[1] = 'V';
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "BLAS");
    opts.acceleration = PSAcceleration_BLAS;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;

    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsVecProdBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                          PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc == 2 && cfg->argv != NULL);
    int *intargs = (int *) cfg->argv;
    int len_a = intargs[0], len_b = intargs[1], ok = 1;
    assert(len_a > 0);
    assert(len_b > 0);
    PSMatrix a = PSMatrixWithGaussianRandom(1, 1, len_a);
    PSMatrix b = PSMatrixWithGaussianRandom(1, 1, len_b);
    PSFloat *dest = malloc((len_a * len_b) * sizeof(PSFloat));
    if (a == NULL || b == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#ifdef HAS_BLAS
    opts.acceleration = PSAcceleration_BLAS;
    PS_INIT_BENCHMARK(cfg, num_results, res, "BLAS");
    PSBenchmarkMeasure(
        res, (ok = PSOuterProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_Accelerate;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, (ok = PSOuterProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, (ok = PSOuterProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (a != NULL) PSMatrixFree(a);
    if (b != NULL) PSMatrixFree(b);
    free(dest);
    return ok;
}

int mathsSumVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSAddVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSAddVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSAddVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(dest);
    return ok;
}

int mathsSubVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(dest);
    return ok;
}

int mathsMulVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(dest);
    return ok;
}

int mathsAddVSBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSAddVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSAddVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSAddVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsMulVSBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSMultiplyVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSMultiplyVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSMultiplyVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsDivVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSMathOpts opts = {.acceleration = PSGlobalAcceleration};
    /* Avoid division by zero */
    PSAddVectorScalar(y, PSFLOAT_EPS, y, size, &opts);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(dest);
    return ok;
}

int mathsDivVSBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    if (y == 0) y += PSFLOAT_EPS;
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSDivideVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDivideVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDivideVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsDivSVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    if (y == 0) y += PSFLOAT_EPS;
    PSAddVectorScalar(x, PSFLOAT_EPS, x, size, NULL);
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSDivideScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDivideScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDivideScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsSubVSBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    if (y == 0) y += PSFLOAT_EPS;
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSSubtractVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSSubtractVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSSubtractVectorScalar(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsSubSVBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    if (cfg->argc > 1) mode = argv[1];
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = PSGaussianRandom(0, 1);
    if (y == 0) y += PSFLOAT_EPS;
    PSAddVectorScalar(x, PSFLOAT_EPS, x, size, NULL);
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSSubtractScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSSubtractScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSSubtractScalarVector(y, x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsReduceBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                         PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorReduceSum(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorReduceSum(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorReduceSum(x, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    return ok;
}

int mathsMeanBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSMean(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSMean(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSMean(x, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    return ok;
}

int mathsVarianceBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                           PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVariance(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVariance(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVariance(x, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    return ok;
}

int mathsStdDevBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                           PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSStdDev(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSStdDev(x, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSStdDev(x, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    return ok;
}

int mathsSqrtBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorSqrt(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorSqrt(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorSqrt(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsTanhBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorTanh(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorTanh(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorTanh(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsExpBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorExp(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorExp(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorExp(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsNegBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorNeg(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorNeg(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorNeg(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsAbsBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(y) == (long) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorAbs(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorAbs(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorAbs(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    return ok;
}

int mathsVecPowBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                         PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int *argv = (int *) cfg->argv;
    int size = argv[0], mode = PS_STORE_MODE_SET, ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat y = (PSFloat) argv[1];
    PSMatrix dest = PSMatrixWithGaussianRandom(1, 1, size), tmpdest = NULL;
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    assert(PSMatrixLength(dest) == (long) size);
    if (mode != PS_STORE_MODE_SET && cache_enabled)
        tmpdest = PSMatrixDupShape(dest);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {.store_mode = mode, .tmpdest = tmpdest};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    opts.acceleration = PSAcceleration_Accelerate;
    PSBenchmarkMeasure(res, PSVectorPower(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSVectorPower(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSVectorPower(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    PSMatrixFree(x);
    PSMatrixFree(dest);
    PSMatrixFree(tmpdest);
    return ok;
}

int mathsMatrixProdBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc == 3 && cfg->argv != NULL);
    int *intargs = (int *) cfg->argv;
    int m = intargs[0], n = intargs[1], k = intargs[2], ok = 1;
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    PSMatrix a = PSMatrixWithGaussianRandom(1, 2, m, n);
    PSMatrix b = PSMatrixWithGaussianRandom(1, 2, n, k);
    PSMatrix dest = PSMatrixZeros(2, m, k);
    if (a == NULL || b == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#ifdef HAS_BLAS
    opts.acceleration = PSAcceleration_BLAS;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration |= PSAcceleration_Accelerate;
    PS_INIT_BENCHMARK(cfg, num_results, res, "BLAS|Accelerate");
#else
    PS_INIT_BENCHMARK(cfg, num_results, res, "BLAS");
#endif
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_Accelerate;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (a != NULL) PSMatrixFree(a);
    if (b != NULL) PSMatrixFree(b);
    if (dest != NULL) PSMatrixFree(dest);
    return ok;
}

int actSigmoidBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSSigmoid(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, PSSigmoid(x, dest, size, PSAcceleration_AVX));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, PSSigmoid(x, dest, size, PSAcceleration_None));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actSigmoidDerivBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSSigmoidDerivative(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(
        res, PSSigmoidDerivative(x, dest, size, PSAcceleration_AVX)
    );
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, PSSigmoidDerivative(x, dest, size, PSAcceleration_None)
    );
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actTanhBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSTanhActivation(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(
        res, PSTanhActivation(x, dest, size,PSAcceleration_AVX)
    );
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, PSTanhActivation(x, dest, size,PSAcceleration_None)
    );
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actTanhDerivBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSTanhDerivative(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(
        res, PSTanhDerivative(x, dest, size, PSAcceleration_AVX)
    );
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, PSTanhDerivative(x, dest, size, PSAcceleration_None)
    );
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actReluBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, PSRelu(x, dest, size, PSAcceleration_Accelerate));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, PSRelu(x, dest, size, PSAcceleration_AVX));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, PSRelu(x, dest, size, PSAcceleration_None));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actGeluBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, PSGelu(x, dest, size, PSAcceleration_Accelerate));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, PSGelu(x, dest, size, PSAcceleration_AVX));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, PSGelu(x, dest, size, PSAcceleration_None));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actReluDerivBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSReluDerivative(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(
        res, PSReluDerivative(x, dest, size, PSAcceleration_AVX)
    );
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, PSReluDerivative(x, dest, size, PSAcceleration_None)
    );
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actGeluDerivBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSGeluDerivative(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(
        res, PSGeluDerivative(x, dest, size, PSAcceleration_AVX)
    );
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, PSGeluDerivative(x, dest, size, PSAcceleration_None)
    );
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int actSoftmaxBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                        PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    if (x == NULL || dest == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(
        res, PSSoftmax(x, dest, size, PSAcceleration_Accelerate)
    );
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, PSSoftmax(x, dest, size, PSAcceleration_AVX));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, PSSoftmax(x, dest, size, PSAcceleration_None));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    free(dest);
    return ok;
}

int optimDefaultBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                          PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    int use_momentum = 0;
    if (cfg->argc > 1) use_momentum = *(((int *) cfg->argv) + 1);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    PSFloat *tmp = NULL;
    if (x == NULL || y == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    if (cache_enabled) tmp = PSVectorCreate(size);
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSSGDOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSSGDOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSSGDOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem);
    free(tmp);
    return ok;
}

int optimNesterovBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                           PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    PSFloat *tmp = NULL;
    if (x == NULL || y == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    if (cache_enabled) tmp = PSVectorCreate(size);
    PSFloat rate = 0.1, momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSNesterovOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSNesterovOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSNesterovOptimization(
            x, y, mem, NULL, tmp, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem);
    free(tmp);
    return ok;
}

int optimWindowGradBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                          PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSFloat rate = 0.1, momentum = 0.0;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PSBenchmarkMeasure(res, (
        ok = PSWindowGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSWindowGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSWindowGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem);
    return ok;
}

int optimAdaGradBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                          PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    int use_momentum = 0;
    if (cfg->argc > 1) use_momentum = *(((int *) cfg->argv) + 1);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSAdaGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSAdaGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSAdaGradOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem);
    return ok;
}

int optimRMSPropBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                          PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    int use_momentum = 0;
    if (cfg->argc > 1) use_momentum = *(((int *) cfg->argv) + 1);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSRMSPropOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSRMSPropOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSRMSPropOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem);
    return ok;
}

int optimAdaDeltaBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv), ok = 1;
    assert(size > 0);
    int use_momentum = 0;
    if (cfg->argc > 1) use_momentum = *(((int *) cfg->argv) + 1);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem1 = malloc(size * sizeof(PSFloat));
    PSFloat *mem2 = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || mem1 == NULL || mem2 == NULL)
    {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSAdaDeltaOptimization(
            x, y, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSAdaDeltaOptimization(
            x, y, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSAdaDeltaOptimization(
            x, y, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem1);
    free(mem2);
    return ok;
}

int optimAdamBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                       PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv);
    assert(size > 0);
    int use_momentum = 0, ok = 1;
    if (cfg->argc > 1) use_momentum = *(((int *) cfg->argv) + 1);
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *mem1 = malloc(size * sizeof(PSFloat));
    PSFloat *mem2 = malloc(size * sizeof(PSFloat));
    PSFloat *tmp1 = NULL, *tmp2 = NULL, *tmp3 = NULL;
    if (x == NULL || y == NULL || mem1 == NULL || mem2 == NULL)
    {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    if (cache_enabled) {
        tmp1 = PSVectorCreate(size);
        tmp2 = PSVectorCreate(size);
        tmp3 = PSVectorCreate(size);
    }
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSAdamOptimization(
            x, y, mem1, mem2, tmp1, tmp2, tmp3, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSAdamOptimization(
            x, y, mem1, mem2, tmp1, tmp2, tmp3, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSAdamOptimization(
            x, y, mem1, mem2, tmp1, tmp2, tmp3, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(mem1);
    free(mem2);
    free(tmp1);
    free(tmp2);
    free(tmp3);
    return ok;
}

int LRegularizationBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                             PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc > 0 && cfg->argv != NULL);
    int size = *((int *) cfg->argv);
    assert(size > 0);
    int types = 3, batches = 4, wdecay = 0, ok = 1;
    if (cfg->argc >= 2) types = *(((int *) cfg->argv) + 1);
    if (cfg->argc >= 3) batches = *(((int *) cfg->argv) + 2);
    if (cfg->argc >= 4) wdecay = *(((int *) cfg->argv) + 3);
    if (types <= 0) types = 3;
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, size);
    PSMatrix y = PSMatrixWithGaussianRandom(1, 1, size);
    PSFloat *tmp = NULL;
    if (x == NULL || y == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (long) size);
    if (cache_enabled) {
        tmp = PSVectorCreate(size);
    }
    PSBenchmarkResults *res = results;
    int acceleration = 0;
    PSFloat l1 = (types & 1 ? 0.0001 : 0),
            l2 = (types & 2 ? 0.00001 : 0);
    PSFloat l1_loss = 0, l2_loss = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_Accelerate;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSLRegularization(
            l1, l2, x, y, tmp, size, &l1_loss, &l2_loss,
            batches, wdecay, acceleration
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    if (auto_accel) acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSLRegularization(
            l1, l2, x, y, tmp, size, &l1_loss, &l2_loss,
            batches, wdecay, acceleration
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSLRegularization(
            l1, l2, x, y, tmp, size, &l1_loss, &l2_loss,
            batches, wdecay, acceleration
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (y != NULL) PSMatrixFree(y);
    free(tmp);
    return ok;
}

int fullnetForwardBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                            PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc == 2 && cfg->argv != NULL);
    int *intargs = (int *) cfg->argv;
    int input_size = intargs[0], layer_size = intargs[1], ok = 1;
    assert(input_size > 0);
    assert(layer_size > 0);
    PSModel *model = PSModelCreate("Fullly connected benchmark");
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, input_size);
    if (x == NULL || model == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    ok = PSAddLayer(model, FullyConnected, input_size, NULL) != NULL;
    if (!ok) goto final;
    ok = PSAddLayer(model, FullyConnected, layer_size, NULL) != NULL;
    if (!PSModelIsBuilt(model)) PSModelBuild(model);
    ok = PSModelIsBuilt(model);
    if (!ok) {
        PSErr(__func__, "Could not build model");
        goto final;
    }
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    model->acceleration = PSAcceleration_Accelerate;
    if (auto_accel) model->acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSForward(model, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    model->acceleration = PSAcceleration_AVX;
    if (auto_accel) model->acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSForward(model, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    model->acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSForward(model, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixFree(x);
    if (model != NULL) PSModelFree(model);
    return ok;
}

int cifarCNNBackpropBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                              PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    int ok = 1;
    PSMatrix x = NULL;
    PSModel *model = makeCIFARLikeCNN();
    PSGradient **gradients = NULL;
    ok = model != NULL;
    if (!ok) goto final;
    x = PSMatrixWithGaussianRandom(1, 1, PS_CIFAR_IMAGE_SIZE);
    ok = (x != NULL);
    if (!ok) goto final;
    PSFloat y[10] = {0};
    y[9] = 1.0;
    if (!PSModelIsBuilt(model)) PSModelBuild(model);
    ok = PSModelIsBuilt(model);
    if (!ok) {
        PSErr(__func__, "Could not build model");
        goto final;
    }
    PSBenchmarkResults *res = results;
    model->acceleration = PSGlobalAcceleration;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Default Acceleration");
    PSBenchmarkMeasure(res, (
        gradients = backprop(model, x, y, NULL, NULL)
    ));
    ok = gradients != NULL;
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#if HAS_BLAS
    model->acceleration = PSAcceleration_BLAS;
    if (auto_accel) model->acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "BLAS");
    PSBenchmarkMeasure(res, (
        gradients = backprop(model, x, y, NULL, NULL)
    ));
    ok = gradients != NULL;
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    model->acceleration = PSAcceleration_Accelerate;
    if (auto_accel) model->acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "Accelerate Framework");
    PSBenchmarkMeasure(res, (
        gradients = backprop(model, x, y, NULL, NULL)
    ));
    ok = gradients != NULL;
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    model->acceleration = PSAcceleration_AVX;
    if (auto_accel) model->acceleration |= PSAcceleration_Auto;
    PS_INIT_BENCHMARK(cfg, num_results, res, "AVX");
    PSBenchmarkMeasure(res, (
        gradients = backprop(model, x, y, NULL, NULL)
    ));
    ok = gradients != NULL;
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    model->acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, num_results, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        gradients = backprop(model, x, y, NULL, NULL)
    ));
    ok = gradients != NULL;
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (gradients != NULL && model != NULL)
        PSDeleteModelGradients(gradients, model);
    if (model != NULL) PSModelFree(model);
    PSMatrixFree(x);
    return ok;
}

/* Benchmark configurations */
PSBenchmarkConfig bechmarks[] = {
    /*{"Dummy", NULL, 3, 1, dummyBenchmark, 0, NULL},*/

    /********************** Maths **********************/

    /* PSDotProduct */
    {"PSDotProduct (1000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(1000)},
    {"PSDotProduct (10000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(10000)},
    {"PSDotProduct (100000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(100000)},
    {"PSDotProduct (1000000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(1000000)},
    {"PSDotProduct (%d)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, int_argv},

    /* PSDot */
    {"PSDot ((100,200) * 200)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(100, 200)},
    {"PSDot ((1000,3000) * 3000)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(1000, 3000)},
    {"PSDot ((1000,50000) * 50000)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(10000, 50000)},

    /* PSOuterProduct */
    {"PSOuterProduct (100,200)", &maths_tag, 0, 10, mathsVecProdBenchmark,
     2, INTARGS(100, 200)},
    {"PSOuterProduct (1000,3000)", &maths_tag, 0, 10,
     mathsVecProdBenchmark, 2, INTARGS(1000, 3000)},
    {"PSOuterProduct (1000,50000)", &maths_tag, 0, 10,
     mathsVecProdBenchmark, 2, INTARGS(10000, 50000)},

    /* PSAddVectors */
    {"PSAddVectors (1000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(1000)},
    {"PSAddVectors (10000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(10000)},
    {"PSAddVectors (100000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(100000)},
    {"PSAddVectors (1000000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(1000000)},
    {"PSAddVectors (%d)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, int_argv},

    /* PSSubtractVectors */
    {"PSSubtractVectors (1000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(1000)},
    {"PSSubtractVectors (10000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(10000)},
    {"PSSubtractVectors (100000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(100000)},
    {"PSSubtractVectors (1000000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(1000000)},
    {"PSSubtractVectors (%d)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, int_argv},

    /* PSMultiplyVectors */
    {"PSMultiplyVectors (1000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(1000)},
    {"PSMultiplyVectors (10000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(10000)},
    {"PSMultiplyVectors (100000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(100000)},
    {"PSMultiplyVectors (1000000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(1000000)},
    {"PSMultiplyVectors (%d)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, int_argv},

    /* PSDivideVectors */
    {"PSDivideVectors (1000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(1000)},
    {"PSDivideVectors (10000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(10000)},
    {"PSDivideVectors (100000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(100000)},
    {"PSDivideVectors (1000000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(1000000)},
    {"PSDivideVectors (%d)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, int_argv},

    /* PSAddVectorScalar */
    {"PSAddVectorScalar (1000)", &maths_tag, 0, 10, mathsAddVSBenchmark,
     1, INTARGS(1000)},
    {"PSAddVectorScalar (10000)", &maths_tag, 0, 10, mathsAddVSBenchmark,
     1, INTARGS(10000)},
    {"PSAddVectorScalar (100000)", &maths_tag, 0, 10, mathsAddVSBenchmark,
     1, INTARGS(100000)},
    {"PSAddVectorScalar (1000000)", &maths_tag, 0, 10, mathsAddVSBenchmark,
     1, INTARGS(1000000)},
    {"PSAddVectorScalar (%d)", &maths_tag, 0, 10, mathsAddVSBenchmark,
     1, int_argv},

    /* PSMultiplyVectorScalar */
    {"PSMultiplyVectorScalar (1000)", &maths_tag, 0, 10, mathsMulVSBenchmark,
     1, INTARGS(1000)},
    {"PSMultiplyVectorScalar (10000)", &maths_tag, 0, 10, mathsMulVSBenchmark,
     1, INTARGS(10000)},
    {"PSMultiplyVectorScalar (100000)", &maths_tag, 0, 10, mathsMulVSBenchmark,
     1, INTARGS(100000)},
    {"PSMultiplyVectorScalar (1000000)", &maths_tag, 0, 10, mathsMulVSBenchmark,
     1, INTARGS(1000000)},
    {"PSMultiplyVectorScalar (%d)", &maths_tag, 0, 10, mathsMulVSBenchmark,
     1, int_argv},
    {"PSMultiplyVectorScalar (1000, ADD)", &maths_tag, 0, 10,
        mathsMulVSBenchmark, 2, INTARGS(1000, PS_STORE_MODE_ADD), 1},
    {"PSMultiplyVectorScalar (10000, ADD)", &maths_tag, 0, 10,
        mathsMulVSBenchmark, 2, INTARGS(10000, PS_STORE_MODE_ADD), 1},
    {"PSMultiplyVectorScalar (100000, ADD)", &maths_tag, 0, 10,
        mathsMulVSBenchmark, 2, INTARGS(100000, PS_STORE_MODE_ADD), 1},
    {"PSMultiplyVectorScalar (1000000, ADD)", &maths_tag, 0, 10,
        mathsMulVSBenchmark, 2, INTARGS(1000000, PS_STORE_MODE_ADD), 1},
    {"PSMultiplyVectorScalar (%d, ADD)", &maths_tag, 0, 10,
        mathsMulVSBenchmark, 2, int_argv, 1, INTARGS(0, PS_STORE_MODE_ADD)},

    /* PSDivideVectorScalar */
    {"PSDivideVectorScalar (1000)", &maths_tag, 0, 10, mathsDivVSBenchmark,
     1, INTARGS(1000)},
    {"PSDivideVectorScalar (10000)", &maths_tag, 0, 10, mathsDivVSBenchmark,
     1, INTARGS(10000)},
    {"PSDivideVectorScalar (100000)", &maths_tag, 0, 10, mathsDivVSBenchmark,
     1, INTARGS(100000)},
    {"PSDivideVectorScalar (1000000)", &maths_tag, 0, 10, mathsDivVSBenchmark,
     1, INTARGS(1000000)},
    {"PSDivideVectorScalar (%d)", &maths_tag, 0, 10, mathsDivVSBenchmark,
     1, int_argv},
    {"PSDivideVectorScalar (1000, ADD)", &maths_tag, 0, 10,
        mathsDivVSBenchmark, 2, INTARGS(1000, PS_STORE_MODE_ADD), 1},
    {"PSDivideVectorScalar (10000, ADD)", &maths_tag, 0, 10,
        mathsDivVSBenchmark, 2, INTARGS(10000, PS_STORE_MODE_ADD), 1},
    {"PSDivideVectorScalar (100000, ADD)", &maths_tag, 0, 10,
        mathsDivVSBenchmark, 2, INTARGS(100000, PS_STORE_MODE_ADD), 1},
    {"PSDivideVectorScalar (1000000, ADD)", &maths_tag, 0, 10,
        mathsDivVSBenchmark, 2, INTARGS(1000000, PS_STORE_MODE_ADD), 1},
    {"PSDivideVectorScalar (%d, ADD)", &maths_tag, 0, 10,
        mathsDivVSBenchmark, 2, int_argv, 1, INTARGS(0, PS_STORE_MODE_ADD)},

    /* PSDivideScalarVector */
    {"PSDivideScalarVector (1000)", &maths_tag, 0, 10, mathsDivSVBenchmark,
     1, INTARGS(1000)},
    {"PSDivideScalarVector (10000)", &maths_tag, 0, 10, mathsDivSVBenchmark,
     1, INTARGS(10000)},
    {"PSDivideScalarVector (100000)", &maths_tag, 0, 10, mathsDivSVBenchmark,
     1, INTARGS(100000)},
    {"PSDivideScalarVector (1000000)", &maths_tag, 0, 10, mathsDivSVBenchmark,
     1, INTARGS(1000000)},
    {"PSDivideScalarVector (%d)", &maths_tag, 0, 10, mathsDivSVBenchmark,
     1, int_argv},
    {"PSDivideScalarVector (1000, ADD)", &maths_tag, 0, 10,
        mathsDivSVBenchmark, 2, INTARGS(1000, PS_STORE_MODE_ADD), 1},
    {"PSDivideScalarVector (10000, ADD)", &maths_tag, 0, 10,
        mathsDivSVBenchmark, 2, INTARGS(10000, PS_STORE_MODE_ADD), 1},
    {"PSDivideScalarVector (100000, ADD)", &maths_tag, 0, 10,
        mathsDivSVBenchmark, 2, INTARGS(100000, PS_STORE_MODE_ADD), 1},
    {"PSDivideScalarVector (1000000, ADD)", &maths_tag, 0, 10,
        mathsDivSVBenchmark, 2, INTARGS(1000000, PS_STORE_MODE_ADD), 1},
    {"PSDivideScalarVector (%d, ADD)", &maths_tag, 0, 10,
        mathsDivSVBenchmark, 2, int_argv, 1, INTARGS(0, PS_STORE_MODE_ADD)},

    /* PSSubtractVectorScalar */
    {"PSSubtractVectorScalar (1000)", &maths_tag, 0, 10, mathsSubVSBenchmark,
     1, INTARGS(1000)},
    {"PSSubtractVectorScalar (10000)", &maths_tag, 0, 10, mathsSubVSBenchmark,
     1, INTARGS(10000)},
    {"PSSubtractVectorScalar (100000)", &maths_tag, 0, 10, mathsSubVSBenchmark,
     1, INTARGS(100000)},
    {"PSSubtractVectorScalar (1000000)", &maths_tag, 0, 10, mathsSubVSBenchmark,
     1, INTARGS(1000000)},
    {"PSSubtractVectorScalar (%d)", &maths_tag, 0, 10, mathsSubVSBenchmark,
     1, int_argv},
    {"PSSubtractVectorScalar (1000, ADD)", &maths_tag, 0, 10,
        mathsSubVSBenchmark, 2, INTARGS(1000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractVectorScalar (10000, ADD)", &maths_tag, 0, 10,
        mathsSubVSBenchmark, 2, INTARGS(10000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractVectorScalar (100000, ADD)", &maths_tag, 0, 10,
        mathsSubVSBenchmark, 2, INTARGS(100000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractVectorScalar (1000000, ADD)", &maths_tag, 0, 10,
        mathsSubVSBenchmark, 2, INTARGS(1000000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractVectorScalar (%d, ADD)", &maths_tag, 0, 10,
        mathsSubVSBenchmark, 2, int_argv, 1, INTARGS(0, PS_STORE_MODE_ADD)},

    /* PSSubtractScalarVector */
    {"PSSubtractScalarVector (1000)", &maths_tag, 0, 10, mathsSubSVBenchmark,
     1, INTARGS(1000)},
    {"PSSubtractScalarVector (10000)", &maths_tag, 0, 10, mathsSubSVBenchmark,
     1, INTARGS(10000)},
    {"PSSubtractScalarVector (100000)", &maths_tag, 0, 10, mathsSubSVBenchmark,
     1, INTARGS(100000)},
    {"PSSubtractScalarVector (1000000)", &maths_tag, 0, 10, mathsSubSVBenchmark,
     1, INTARGS(1000000)},
    {"PSSubtractScalarVector (%d)", &maths_tag, 0, 10, mathsSubSVBenchmark,
     1, int_argv},
    {"PSSubtractScalarVector (1000, ADD)", &maths_tag, 0, 10,
        mathsSubSVBenchmark, 2, INTARGS(1000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractScalarVector (10000, ADD)", &maths_tag, 0, 10,
        mathsSubSVBenchmark, 2, INTARGS(10000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractScalarVector (100000, ADD)", &maths_tag, 0, 10,
        mathsSubSVBenchmark, 2, INTARGS(100000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractScalarVector (1000000, ADD)", &maths_tag, 0, 10,
        mathsSubSVBenchmark, 2, INTARGS(1000000, PS_STORE_MODE_ADD), 1},
    {"PSSubtractScalarVector (%d, ADD)", &maths_tag, 0, 10,
        mathsSubSVBenchmark, 2, int_argv, 1, INTARGS(0, PS_STORE_MODE_ADD)},

    /* PSVectorReduceSum */
    {"PSVectorReduceSum (1000)", &maths_tag, 0, 10, mathsReduceBenchmark,
     1, INTARGS(1000)},
    {"PSVectorReduceSum (10000)", &maths_tag, 0, 10, mathsReduceBenchmark,
     1, INTARGS(10000)},
    {"PSVectorReduceSum (100000)", &maths_tag, 0, 10, mathsReduceBenchmark,
     1, INTARGS(100000)},
    {"PSVectorReduceSum (1000000)", &maths_tag, 0, 10, mathsReduceBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorReduceSum (%d)", &maths_tag, 0, 10, mathsReduceBenchmark,
     1, int_argv},

    /* PSMean */
    {"PSMean (1000)", &maths_tag, 0, 10, mathsMeanBenchmark,
     1, INTARGS(1000)},
    {"PSMean (10000)", &maths_tag, 0, 10, mathsMeanBenchmark,
     1, INTARGS(10000)},
    {"PSMean (100000)", &maths_tag, 0, 10, mathsMeanBenchmark,
     1, INTARGS(100000)},
    {"PSMean (1000000)", &maths_tag, 0, 10, mathsMeanBenchmark,
     1, INTARGS(1000000)},
    {"PSMean (%d)", &maths_tag, 0, 10, mathsMeanBenchmark,
     1, int_argv},

    /* PSVariance */
    {"PSVariance (1000)", &maths_tag, 0, 10, mathsVarianceBenchmark,
     1, INTARGS(1000)},
    {"PSVariance (10000)", &maths_tag, 0, 10, mathsVarianceBenchmark,
     1, INTARGS(10000)},
    {"PSVariance (100000)", &maths_tag, 0, 10, mathsVarianceBenchmark,
     1, INTARGS(100000)},
    {"PSVariance (1000000)", &maths_tag, 0, 10, mathsVarianceBenchmark,
     1, INTARGS(1000000)},
    {"PSVariance (%d)", &maths_tag, 0, 10, mathsVarianceBenchmark,
     1, int_argv},

    /* PSStdDev */
    {"PSStdDev (1000)", &maths_tag, 0, 10, mathsStdDevBenchmark,
     1, INTARGS(1000)},
    {"PSStdDev (10000)", &maths_tag, 0, 10, mathsStdDevBenchmark,
     1, INTARGS(10000)},
    {"PSStdDev (100000)", &maths_tag, 0, 10, mathsStdDevBenchmark,
     1, INTARGS(100000)},
    {"PSStdDev (1000000)", &maths_tag, 0, 10, mathsStdDevBenchmark,
     1, INTARGS(1000000)},
    {"PSStdDev (%d)", &maths_tag, 0, 10, mathsStdDevBenchmark,
     1, int_argv},

    /* PSVectorSqrt */
    {"PSVectorSqrt (1000)", &maths_tag, 0, 10, mathsSqrtBenchmark,
     1, INTARGS(1000)},
    {"PSVectorSqrt (10000)", &maths_tag, 0, 10, mathsSqrtBenchmark,
     1, INTARGS(10000)},
    {"PSVectorSqrt (100000)", &maths_tag, 0, 10, mathsSqrtBenchmark,
     1, INTARGS(100000)},
    {"PSVectorSqrt (1000000)", &maths_tag, 0, 10, mathsSqrtBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorSqrt (%d)", &maths_tag, 0, 10, mathsSqrtBenchmark,
     1, int_argv},

    /* PSVectorTanh */
    {"PSVectorTanh (1000)", &maths_tag, 0, 10, mathsTanhBenchmark,
     1, INTARGS(1000)},
    {"PSVectorTanh (10000)", &maths_tag, 0, 10, mathsTanhBenchmark,
     1, INTARGS(10000)},
    {"PSVectorTanh (100000)", &maths_tag, 0, 10, mathsTanhBenchmark,
     1, INTARGS(100000)},
    {"PSVectorTanh (1000000)", &maths_tag, 0, 10, mathsTanhBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorTanh (%d)", &maths_tag, 0, 10, mathsTanhBenchmark,
     1, int_argv},

    /* PSVectorExp */
    {"PSVectorExp (1000)", &maths_tag, 0, 10, mathsExpBenchmark,
     1, INTARGS(1000)},
    {"PSVectorExp (10000)", &maths_tag, 0, 10, mathsExpBenchmark,
     1, INTARGS(10000)},
    {"PSVectorExp (100000)", &maths_tag, 0, 10, mathsExpBenchmark,
     1, INTARGS(100000)},
    {"PSVectorExp (1000000)", &maths_tag, 0, 10, mathsExpBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorExp (%d)", &maths_tag, 0, 10, mathsExpBenchmark,
     1, int_argv},

    /* PSVectorNeg */
    {"PSVectorNeg (1000)", &maths_tag, 0, 10, mathsNegBenchmark,
     1, INTARGS(1000)},
    {"PSVectorNeg (10000)", &maths_tag, 0, 10, mathsNegBenchmark,
     1, INTARGS(10000)},
    {"PSVectorNeg (100000)", &maths_tag, 0, 10, mathsNegBenchmark,
     1, INTARGS(100000)},
    {"PSVectorNeg (1000000)", &maths_tag, 0, 10, mathsNegBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorNeg (%d)", &maths_tag, 0, 10, mathsNegBenchmark,
     1, int_argv},

    /* PSVectorAbs */
    {"PSVectorAbs (1000)", &maths_tag, 0, 10, mathsAbsBenchmark,
     1, INTARGS(1000)},
    {"PSVectorAbs (10000)", &maths_tag, 0, 10, mathsAbsBenchmark,
     1, INTARGS(10000)},
    {"PSVectorAbs (100000)", &maths_tag, 0, 10, mathsAbsBenchmark,
     1, INTARGS(100000)},
    {"PSVectorAbs (1000000)", &maths_tag, 0, 10, mathsAbsBenchmark,
     1, INTARGS(1000000)},
    {"PSVectorAbs (%d)", &maths_tag, 0, 10, mathsAbsBenchmark,
     1, int_argv},

    /* PSVectorPower */
    {"PSVectorPower (1000, 2)", &maths_tag, 0, 10, mathsVecPowBenchmark,
     2, INTARGS(1000, 2), 1},
    {"PSVectorPower (10000, 2)", &maths_tag, 0, 10, mathsVecPowBenchmark,
     2, INTARGS(10000, 2), 1},
    {"PSVectorPower (100000, 200)", &maths_tag, 0, 10, mathsVecPowBenchmark,
     2, INTARGS(100000, 200), 1},
    {"PSVectorPower (1000000, 200)", &maths_tag, 0, 10, mathsVecPowBenchmark,
     2, INTARGS(1000000, 200), 1},
    {"PSVectorPower (%d, 200)", &maths_tag, 0, 10, mathsVecPowBenchmark,
     2, int_argv, 1},

    /* PSMatrixProduct */
    {"PSMatrixProduct (100,300,200)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(100, 300, 200)},
    {"PSMatrixProduct (1000,3000,2000)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(1000, 3000, 2000)},
    /*{"PSMatrixProduct (1000,10000,5000)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(1000, 10000, 5000)},*/

    /********************** Activation **********************/

    /* PSSigmoid */
    {"PSSigmoid (1000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(1000)},
    {"PSSigmoid (10000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(10000)},
    {"PSSigmoid (100000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(100000)},
    {"PSSigmoid (1000000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(1000000)},
    {"PSSigmoid (%d)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, int_argv},

    /* PSTanhActivation */
    {"PSTanhActivation (1000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(1000)},
    {"PSTanhActivation (10000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(10000)},
    {"PSTanhActivation (100000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(100000)},
    {"PSTanhActivation (1000000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(1000000)},
    {"PSTanhActivation (%d)", &activation_tag, 0, 10, actTanhBenchmark,
     1, int_argv},

    /* PSRelu */
    {"PSRelu (1000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(1000)},
    {"PSRelu (10000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(10000)},
    {"PSRelu (100000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(100000)},
    {"PSRelu (1000000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(1000000)},
    {"PSRelu (%d)", &activation_tag, 0, 10, actReluBenchmark,
     1, int_argv},

    /* PSGelu */
    {"PSGelu (1000)", &activation_tag, 0, 10, actGeluBenchmark,
     1, INTARGS(1000)},
    {"PSGelu (10000)", &activation_tag, 0, 10, actGeluBenchmark,
     1, INTARGS(10000)},
    {"PSGelu (100000)", &activation_tag, 0, 10, actGeluBenchmark,
     1, INTARGS(100000)},
    {"PSGelu (1000000)", &activation_tag, 0, 10, actGeluBenchmark,
     1, INTARGS(1000000)},
    {"PSGelu (%d)", &activation_tag, 0, 10, actGeluBenchmark,
     1, int_argv},

    /* PSSoftmax */
    {"PSSoftmax (1000)", &activation_tag, 0, 10, actSoftmaxBenchmark,
     1, INTARGS(1000)},
    {"PSSoftmax (10000)", &activation_tag, 0, 10, actSoftmaxBenchmark,
     1, INTARGS(10000)},
    {"PSSoftmax (100000)", &activation_tag, 0, 10, actSoftmaxBenchmark,
     1, INTARGS(100000)},
    {"PSSoftmax (1000000)", &activation_tag, 0, 10, actSoftmaxBenchmark,
     1, INTARGS(1000000)},
    {"PSSoftmax (%d)", &activation_tag, 0, 10, actSoftmaxBenchmark,
     1, int_argv},

    /* PSSigmoidDerivative */
    {"PSSigmoidDerivative (1000)", &activation_tag, 0, 10,
      actSigmoidDerivBenchmark, 1, INTARGS(1000)},
    {"PSSigmoidDerivative (10000)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, INTARGS(10000)},
    {"PSSigmoidDerivative (100000)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, INTARGS(100000)},
    {"PSSigmoidDerivative (1000000)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, INTARGS(1000000)},
    {"PSSigmoidDerivative (%d)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, int_argv},

    /* PSTanhDerivative */
    {"PSTanhDerivative (1000)", &activation_tag, 0, 10,
      actTanhDerivBenchmark, 1, INTARGS(1000)},
    {"PSTanhDerivative (10000)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, INTARGS(10000)},
    {"PSTanhDerivative (100000)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, INTARGS(100000)},
    {"PSTanhDerivative (1000000)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, INTARGS(1000000)},
    {"PSTanhDerivative (%d)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, int_argv},

    /* PSReluDerivative */
    {"PSReluDerivative (1000)", &activation_tag, 0, 10,
      actReluDerivBenchmark, 1, INTARGS(1000)},
    {"PSReluDerivative (10000)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, INTARGS(10000)},
    {"PSReluDerivative (100000)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, INTARGS(100000)},
    {"PSReluDerivative (1000000)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, INTARGS(1000000)},
    {"PSReluDerivative (%d)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, int_argv},

    /* PSGeluDerivative */
    {"PSGeluDerivative (1000)", &activation_tag, 0, 10,
      actGeluDerivBenchmark, 1, INTARGS(1000)},
    {"PSGeluDerivative (10000)", &activation_tag, 0, 10,
     actGeluDerivBenchmark, 1, INTARGS(10000)},
    {"PSGeluDerivative (100000)", &activation_tag, 0, 10,
     actGeluDerivBenchmark, 1, INTARGS(100000)},
    {"PSGeluDerivative (1000000)", &activation_tag, 0, 10,
     actGeluDerivBenchmark, 1, INTARGS(1000000)},
    {"PSGeluDerivative (%d)", &activation_tag, 0, 10,
     actGeluDerivBenchmark, 1, int_argv},

    /********************** Optimization **********************/

    /* PSSGDOptimization */
    {"PSSGDOptimization (10000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(10000)},
    {"PSSGDOptimization (100000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(100000)},
    {"PSSGDOptimization (1000000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(1000000)},
    {"PSSGDOptimization (%d)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, int_argv},

    /* PSSGDOptimization (with momentum)*/
    {"PSSGDOptimization (mom.) (10000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 2, INTARGS(10000,1), 1},
    {"PSSGDOptimization (mom.) (100000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 2, INTARGS(100000,1), 1},
    {"PSSGDOptimization (mom.) (1000000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 2, INTARGS(1000000,1), 1},
    {"PSSGDOptimization (mom.) (%d)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 2, int_argv, 1},

    /* PSNesterovOptimization */
    {"PSNesterovOptimization (10000)", &optimization_tag, 0, 10,
     optimNesterovBenchmark, 1, INTARGS(10000)},
    {"PSNesterovOptimization (100000)", &optimization_tag, 0, 10,
     optimNesterovBenchmark, 1, INTARGS(100000)},
    {"PSNesterovOptimization (1000000)", &optimization_tag, 0, 10,
     optimNesterovBenchmark, 1, INTARGS(1000000)},
    {"PSNesterovOptimization (%d)", &optimization_tag, 0, 10,
     optimNesterovBenchmark, 1, int_argv},

    /* PSWindowGradOptimization */
    {"PSWindowGradOptimization (10000)", &optimization_tag, 0, 10,
     optimWindowGradBenchmark, 1, INTARGS(10000)},
    {"PSWindowGradOptimization (100000)", &optimization_tag, 0, 10,
     optimWindowGradBenchmark, 1, INTARGS(100000)},
    {"PSWindowGradOptimization (1000000)", &optimization_tag, 0, 10,
     optimWindowGradBenchmark, 1, INTARGS(1000000)},
    {"PSWindowGradOptimization (%d)", &optimization_tag, 0, 10,
     optimWindowGradBenchmark, 1, int_argv},

    /* PSAdaGradOptimization */
    {"PSAdaGradOptimization (10000)", &optimization_tag, 0, 10,
     optimAdaGradBenchmark, 1, INTARGS(10000)},
    {"PSAdaGradOptimization (100000)", &optimization_tag, 0, 10,
     optimAdaGradBenchmark, 1, INTARGS(100000)},
    {"PSAdaGradOptimization (1000000)", &optimization_tag, 0, 10,
     optimAdaGradBenchmark, 1, INTARGS(1000000)},
    {"PSAdaGradOptimization (%d)", &optimization_tag, 0, 10,
     optimAdaGradBenchmark, 1, int_argv},

    /* PSAdaDeltaOptimization */
    {"PSAdaDeltaOptimization (10000)", &optimization_tag, 0, 10,
     optimAdaDeltaBenchmark, 1, INTARGS(10000)},
    {"PSAdaDeltaOptimization (100000)", &optimization_tag, 0, 10,
     optimAdaDeltaBenchmark, 1, INTARGS(100000)},
    {"PSAdaDeltaOptimization (1000000)", &optimization_tag, 0, 10,
     optimAdaDeltaBenchmark, 1, INTARGS(1000000)},
    {"PSAdaDeltaOptimization (%d)", &optimization_tag, 0, 10,
     optimAdaDeltaBenchmark, 1, int_argv},

    /* PSAdamOptimization */
    {"PSAdamOptimization (10000)", &optimization_tag, 0, 10,
     optimAdamBenchmark, 1, INTARGS(10000)},
    {"PSAdamOptimization (100000)", &optimization_tag, 0, 10,
     optimAdamBenchmark, 1, INTARGS(100000)},
    {"PSAdamOptimization (1000000)", &optimization_tag, 0, 10,
     optimAdamBenchmark, 1, INTARGS(1000000)},
    {"PSAdamOptimization (%d)", &optimization_tag, 0, 10,
     optimAdamBenchmark, 1, int_argv},

    /* PSRMSPropOptimization */
    {"PSRMSPropOptimization (10000)", &optimization_tag, 0, 10,
     optimRMSPropBenchmark, 1, INTARGS(10000)},
    {"PSRMSPropOptimization (100000)", &optimization_tag, 0, 10,
     optimRMSPropBenchmark, 1, INTARGS(100000)},
    {"PSRMSPropOptimization (1000000)", &optimization_tag, 0, 10,
     optimRMSPropBenchmark, 1, INTARGS(1000000)},
    {"PSRMSPropOptimization (%d)", &optimization_tag, 0, 10,
     optimRMSPropBenchmark, 1, int_argv},

    /* PSLRegularization */
    {"PSLRegularization (L1,10000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(10000,1), 1},
    {"PSLRegularization (L1,100000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(100000,1), 1},
    {"PSLRegularization (L1,1000000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(1000000,1), 1},
    {"PSLRegularization (L1,%d)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, int_argv, 1, INTARGS(0, 1)},

    {"PSLRegularization (L2,10000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(10000,2), 1},
    {"PSLRegularization (L2,100000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(100000,2), 1},
    {"PSLRegularization (L2,1000000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(1000000,2), 1},
    {"PSLRegularization (L2,%d)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, int_argv, 1, INTARGS(0, 2)},

    {"PSLRegularization (L1|L2,10000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(10000,3), 1},
    {"PSLRegularization (L1|L2,100000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(100000,3), 1},
    {"PSLRegularization (L1|L2,1000000)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, INTARGS(1000000,3), 1},
    {"PSLRegularization (L1|L2,%d)", &optimization_tag, 0, 10,
     LRegularizationBenchmark, 2, int_argv, 1, INTARGS(0, 3)},


    /**** Forward step ****/

    /* FullyConnected Forward */
    {"FullyConnected Forward (500,1000)", &fullnet_tag, 0, 10,
     fullnetForwardBenchmark, 2, INTARGS(500,1000)},
    {"FullyConnected Forward (5000,10000)", &fullnet_tag, 0, 10,
     fullnetForwardBenchmark, 2, INTARGS(5000,10000)},
    {"FullyConnected Forward (10000,20000)", &fullnet_tag, 0, 10,
     fullnetForwardBenchmark, 2, INTARGS(10000,20000)},
    {"Convolutional Backprop", &convnet_tag, 0, 10,
     cifarCNNBackpropBenchmark, 0, NULL},
};

void printHelp(char *executable) {
    fprintf(stderr, "Usage: %s [OPTIONS] [TEST_ID, ...]\n", executable);
    fprintf(stderr, "\nOPTIONS:\n\n");
    fprintf(stderr, "   --no-cache              Disable math caches\n");
    fprintf(stderr, "   --auto-acceleration     Enable PSAcceleration_Auto for "
            "tests using it\n");
    fprintf(stderr, "   --prerun-disabled       Disable pre-running benchmark "
                    "code.\n"
                    "                           (by default, benchmark code "
                    "is executed once \n"
                    "                           before actual measure "
                    "in order to avoid code\n"
                    "                           optimization confounding "
                    "effects.)\n");
    fprintf(stderr, "   --int-argv ARGV         Custom int arguments "
            "(comma-separated)\n");
    fprintf(stderr, "   --float-argv ARGV       Custom float arguments "
            "comma-separated)\n");
    fprintf(stderr, "   --min-size MIN          Skip tests having size less "
            "than MIN\n");
    fprintf(stderr, "   --max-size MAX          Skip tests having size greater "
            "than MAX\n");
    fprintf(stderr, "   --match SUBSTR          Only perform tests whose name "
            "matches SUBSTR\n");
    fprintf(stderr, "   --exclude SUBSTR        Skip tests whose name "
            "matches SUBSTR\n");
    fprintf(stderr, "   --skip TEST_ID          Skip test (can be used "
        "multiple times\n");
    fprintf(stderr, "   --csv PATH              CSV Output\n");
    fprintf(stderr, "   --json PATH             JSON Output\n");
    fprintf(stderr, "   --list-tests            List all test IDS\n");
    fprintf(stderr, "   -v, --verbose           Verbose output\n");
    fprintf(stderr, "   -h, --help              Print this help\n");
}

int parseOptions(int argc, char **argv) {
    int i, is_last = 0, last_arg_idx = argc - 1;
    for (i = 1; i < argc; i++) {
        is_last = (last_arg_idx == i);
        char *arg = argv[i];
        if (strcmp("--skip", arg) == 0 && !is_last) {
            char *test_id = argv[++i];
            if (!setTagEnabledStatus(test_id, 0)) exit(1);
        } else if (strcmp("--match", arg) == 0 && !is_last) {
            match_str = argv[++i];
        } else if (strcmp("--exclude", arg) == 0 && !is_last) {
            exclude_str = argv[++i];
        } else if (strcmp("--min-size", arg) == 0 && !is_last) {
            min_size = atoi(argv[++i]);
        } else if (strcmp("--max-size", arg) == 0 && !is_last) {
            max_size = atoi(argv[++i]);
        } else if (strcmp("--int-argv", arg) == 0 && !is_last) {
            int_argc = parseUserArgv(argv[++i], int_argv, 1);
        } else if (strcmp("--float-argv", arg) == 0 && !is_last) {
            flt_argc = parseUserArgv(argv[++i], flt_argv, 0);
        } else if (strcmp("--csv", arg) == 0 && !is_last) {
            csv_output = argv[++i];
        } else if (strcmp("--json", arg) == 0 && !is_last) {
            json_output = argv[++i];
        } else if (strcmp("--no-cache", arg) == 0) {
            cache_enabled = 0;
        } else if (strcmp("--prerun-disabled", arg) == 0) {
            disable_prerun = 1;
        } else if (strcmp("--auto-acceleration", arg) == 0) {
            auto_accel = 1;
        } else if (strcmp("--list-tests", arg) == 0) {
            printTagList();
            exit(1);
        } else if ((strcmp("-v", arg) == 0) ||
                   (strcmp("--verbose", arg) == 0))
        {
            verbose = 1;
        } else if ((strcmp("-h", arg) == 0) || (strcmp("--help", arg) == 0)) {
            printHelp(argv[0]);
            exit(1);
        } else if (arg[0] == '-') {
            fprintf(
                stderr, "ERROR: invalid option '%s'. Use '-h' to see all "
                "available options\n", arg
            );
            exit(1);
        } else break;
    }
    return i;
}

/* Main function */

int main(int argc, char **argv) {
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    int return_val = 0;
    int tot_benchmarks = sizeof(bechmarks) / sizeof(PSBenchmarkConfig),
        bm_idx = 0, i;
    int argidx = parseOptions(argc, argv), all_disabled = 0;
    while (argidx < argc) {
        if (!all_disabled) disableAllTags();
        all_disabled = 1;
        char *test_id = argv[argidx++];
        if (!setTagEnabledStatus(test_id, 1)) return 1;
    }
    FILE *csvf = NULL, *jsonf = NULL;
    if (csv_output != NULL) {
        csvf = fopen(csv_output, "w");
        if (csvf == NULL) {
            PSErr(NULL, "could not open '%s' for writing", csv_output);
            return_val = 1;
            goto final;
        }
        fprintf(csvf, "BENCHMARK,TEST,\"ELAPSED (MICROSEC.)\","
                "\"ELAPSED (HUMAN)\"\n");
    }
    if (json_output != NULL) {
        jsonf = fopen(json_output, "w");
        if (jsonf == NULL) {
            PSErr(NULL, "could not open '%s' for writing", json_output);
            return_val = 1;
            goto final;
        }
        fprintf(jsonf, "[\n");
    }
    struct utsname sysinfo;
    uname(&sysinfo);
    char os[31] = {0};
    snprintf(os, 30, "%s %s %s", sysinfo.sysname, sysinfo.release,
        sysinfo.machine);
    printf(
        PSCOLOR_BOLD PSCOLOR_CYAN
        "============ PsyC Benchmarks ============\n"
        PSCOLOR_RESET
    );
    int optlevel = PSGetCodeOptimizationLevel();
    printf("%-20s %30s\n", "Version:", PSYC_VERSION);
    printf("%-20s %30s\n", "OS:", os);
    printf("%-20s %30d\n", "Arch:", (sizeof(long) == 8 ? 64 : 32));
    if (optlevel >= 0)
        printf("%-20s %30d\n", "Code Optimization:", optlevel);
    else
        printf("%-20s %30s\n", "Code Optimization:", "unknown");
    printf("%-20s %30s\n", "Cache Enabled:", (cache_enabled ? "yes" : "no"));
    printf("Default Accelerations:\n");
    if (PSAVXEnabled(PSGlobalAcceleration))
        printf(" - AVX\n");
    if (PSAccelerateEnabled(PSGlobalAcceleration))
        printf(" - Accelerate Framework\n");
    if (PSBLASEnabled(PSGlobalAcceleration)) {
        printf(" - BLAS");
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
        printf(" (Accelerate Framework)");
#elif defined(HAS_GSL_CBLAS)
        printf(" (GNU Scientific Library)");
#endif
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < tot_benchmarks; i++) {
        PSBenchmarkConfig *cfg = &(bechmarks[i]);
        if (cfg->tag != NULL && !(*cfg->tag)) continue;
        if (match_str != NULL && strstr(cfg->name, match_str) == NULL)
            continue;
        if (exclude_str != NULL && strstr(cfg->name, exclude_str) != NULL)
            continue;
        if (int_argc > 0 && cfg->argv != int_argv) continue;
        if (int_argc <= 0 && cfg->argv == int_argv) continue;
        if (flt_argc > 0 && cfg->argv != flt_argv) continue;
        if (flt_argc <= 0 && cfg->argv == flt_argv) continue;
        if (cfg->argv == int_argv && int_argc < cfg->argc)
            if (!setDefaultArgv(cfg, int_argc, int_argv, 1)) continue;
        if (cfg->argv == flt_argv && flt_argc < cfg->argc)
            if (!setDefaultArgv(cfg, flt_argc, flt_argv, 0)) continue;
        int test_size = 0;
        if (min_size || max_size) {
            if (cfg->size_args == 0) cfg->size_args = 0xFF;
            test_size = PSBecnhmarkSize(cfg);
            if (min_size && test_size < min_size) continue;
            if (max_size && test_size > max_size) continue;
        }
        if (cfg->do_benchmark != NULL) {
            char *name = PSBenchmarkName(cfg);
            printf(
                PSCOLOR_BOLD "Performing Benchmark \"%s\"\n" PSCOLOR_RESET,
                name
            );
            int num_results = 0, max_results = cfg->max_results;
            if (max_results <= 0) max_results = 1;
            PSBenchmarkResults *results = calloc(max_results, sizeof(*results));
            if (results == NULL) {
                PSPrintMemoryErrorMsg();
                return_val = 1;
                goto final;
            }
            struct timeval st, et;
            gettimeofday(&st, NULL);
            int ok = cfg->do_benchmark(cfg, &num_results, results);
            gettimeofday(&et, NULL);
            if (!ok) {
                PSErr(NULL, "Failed to perform benchmark \"%s\"", cfg->name);
                return_val = 1;
                goto final;
            }
            if (num_results == 0) {
                free(results);
                continue;
            }
            PSPrintBenchmarkResults(results, num_results);
            if (csvf != NULL) PSWriteBenchmarkResultsToCSV(
                cfg, results, num_results, csvf
            );
            if (jsonf != NULL) PSWriteBenchmarkResultsToJSON(
                cfg, results, num_results, bm_idx++, jsonf
            );
            free(results);
            time_t elapsed = PSGetElapsedTimeUS(st, et);
            char *elapsed_str = PSGetElapsedTimeString(elapsed, 1);
            printf(
                " -> %d benchmark(s) performed in %s\n\n",
                num_results, elapsed_str
            );
            free(elapsed_str);
        }
    }
final:
    if (csvf != NULL) fclose(csvf);
    if (jsonf != NULL) {
        fprintf(jsonf, "\n]");
        fclose(jsonf);
    }
    return return_val;
}

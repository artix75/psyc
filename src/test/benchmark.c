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
#include <math.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../dropout.h"
#include "../lstm.h"
#include "../gru.h"
#include "../normalization.h"
#include "../mnist.h"
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

#define PS_INIT_BENCHMARK(cfg, results, bm_name) do {\
    if (bm_name != NULL) sprintf(res->name, "%s", bm_name);\
    else res->name[0] = '\0';\
    if (cfg->samples <= 0) cfg->samples = DEFAULT_BENCHMARK_SAMPLES;\
    results->samples = cfg->samples;\
} while(0)

#define INTARGS(...) ((void *)((int []){__VA_ARGS__}))
#define FLOATARGS(...) ((void *)((PSFloat []){__VA_ARGS__}))

#define PSBenchmarkMeasure(results, code) do {\
    struct timeval st, et;\
    results->tot_time_us = 0;\
    results->performed = 0;\
    if (results->samples <= 0) results->samples = DEFAULT_BENCHMARK_SAMPLES;\
    while (results->performed++ < results->samples) {\
        gettimeofday(&st, NULL);\
        code;\
        gettimeofday(&et, NULL);\
        results->tot_time_us += PSGetElapsedTimeUS(st, et);\
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
} PSBenchmarkConfig;

static int compareResults(const void * a, const void * b) {
    assert(a != NULL);
    assert(b != NULL);
    PSBenchmarkResults *res_a = (PSBenchmarkResults *) a;
    PSBenchmarkResults *res_b = (PSBenchmarkResults *) b;
    return res_a->avg_time_us - res_b->avg_time_us;
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
        res++;
    }
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
        *ptr = 0;
    }
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
    assert(PSMatrixLength(x) == (size_t) size);
    assert(PSMatrixLength(y) == (size_t) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDotProduct(x, y, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
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
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "BLAS");
    opts.acceleration = PSAcceleration_BLAS;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;

    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, (ok = PSDot(x, y, dest, &opts)));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
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
    PS_INIT_BENCHMARK(cfg, res, "BLAS");
    PSBenchmarkMeasure(
        res, (ok = PSVectorProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    PSBenchmarkMeasure(
        res, (ok = PSVectorProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, (ok = PSVectorProduct(a, b, dest, len_a, len_b, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (a != NULL) PSMatrixDelete(a);
    if (b != NULL) PSMatrixDelete(b);
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
    assert(PSMatrixLength(x) == (size_t) size);
    assert(PSMatrixLength(y) == (size_t) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSumVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSSumVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSSumVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
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
    assert(PSMatrixLength(x) == (size_t) size);
    assert(PSMatrixLength(y) == (size_t) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSSubtractVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
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
    assert(PSMatrixLength(x) == (size_t) size);
    assert(PSMatrixLength(y) == (size_t) size);
    PSBenchmarkResults *res = results;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSMultiplyVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
    free(dest);
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
    assert(PSMatrixLength(x) == (size_t) size);
    assert(PSMatrixLength(y) == (size_t) size);
    PSMathOpts opts = {.acceleration = PSGlobalAcceleration};
    /* Avoid division by zero */
    PSSumVectorScalar(y, PSFLOAT_EPS, y, size, &opts);
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_AVX;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_None;
    PSBenchmarkMeasure(res, PSDivideVectors(x, y, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
    free(dest);
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
    opts.acceleration |= PSAcceleration_ACF;
#endif
    PS_INIT_BENCHMARK(cfg, res, "BLAS");
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    PSBenchmarkMeasure(
        res, (ok = PSMatrixProduct(a, b, &dest, &opts))
    );
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (a != NULL) PSMatrixDelete(a);
    if (b != NULL) PSMatrixDelete(b);
    if (dest != NULL) PSMatrixDelete(dest);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSSigmoidDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSTanhDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
#endif
    opts.acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    PSBenchmarkMeasure(res, PSReluDerivativeV(x, dest, size, &opts));
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
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
    PSFloat *dest = malloc(size * sizeof(PSFloat));
    PSFloat *mem = malloc(size * sizeof(PSFloat));
    if (x == NULL || y == NULL || dest == NULL || mem == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    assert(PSMatrixLength(x) == (size_t) size);
    PSMathOpts opts = {0};
    PSFloat rate = 0.1, momentum = 0.0;
    if (use_momentum) momentum = 0.9;
    PSBenchmarkResults *res = results;
    int acceleration = 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    opts.acceleration = PSAcceleration_ACF;
    memcpy(dest, x, size * sizeof(PSFloat));
    PSBenchmarkMeasure(res, (
        ok = PSDefaultOptimization(
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
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    opts.acceleration = PSAcceleration_ACF;
    memcpy(dest, x, size * sizeof(PSFloat));
    PSBenchmarkMeasure(res, (
        ok = PSDefaultOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    opts.acceleration = PSAcceleration_ACF;
    memcpy(dest, x, size * sizeof(PSFloat));
    PSBenchmarkMeasure(res, (
        ok = PSDefaultOptimization(
            x, y, mem, NULL, NULL, NULL, NULL, rate, momentum,
            size, acceleration, 0, NULL
        )
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (y != NULL) PSMatrixDelete(y);
    free(dest);
    free(mem);
    return ok;
}

int fullnetFeedforwardBenchmark(PSBenchmarkConfig *cfg, int *num_results,
                                PSBenchmarkResults *results)
{
    PS_BENCHMARK_PREAMBLE(cfg, results);
    assert(cfg->argc == 2 && cfg->argv != NULL);
    int *intargs = (int *) cfg->argv;
    int input_size = intargs[0], layer_size = intargs[1], ok = 1;
    assert(input_size > 0);
    assert(layer_size > 0);
    PSNeuralNetwork *network = PSCreateNetwork("Fullly connected benchmark");
    PSMatrix x = PSMatrixWithGaussianRandom(1, 1, input_size);
    if (x == NULL || network == NULL) {
        PSPrintMemoryErrorMsg();
        ok = 0;
        goto final;
    }
    ok = PSAddLayer(network, FullyConnected, input_size, NULL) != NULL;
    if (!ok) goto final;
    ok = PSAddLayer(network, FullyConnected, layer_size, NULL) != NULL;
    if (!PSIsNetworkBuilt(network)) PSBuildNetwork(network);
    ok = PSIsNetworkBuilt(network);
    if (!ok) {
        PSErr(__func__, "Could not build network");
        goto final;
    }
    PSBenchmarkResults *res = results;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    network->acceleration = PSAcceleration_ACF;
    PS_INIT_BENCHMARK(cfg, res, "Apple Accelerate Framework");
    PSBenchmarkMeasure(res, (
        ok = PSFeedforward(network, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
#ifdef USE_AVX
    network->acceleration = PSAcceleration_AVX;
    PS_INIT_BENCHMARK(cfg, res, "AVX");
    PSBenchmarkMeasure(res, (
        ok = PSFeedforward(network, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
#endif
    network->acceleration = PSAcceleration_None;
    PS_INIT_BENCHMARK(cfg, res, "No Acceleration");
    PSBenchmarkMeasure(res, (
        ok = PSFeedforward(network, x)
    ));
    if (!ok) goto final;
    *num_results += 1;
    res += 1;
final:
    if (x != NULL) PSMatrixDelete(x);
    if (network != NULL) PSDeleteNetwork(network);
    return ok;
}

/* Benchmark configurations */
PSBenchmarkConfig bechmarks[] = {
    /*{"Dummy", NULL, 3, 1, dummyBenchmark, 0, NULL},*/
    {"PSDotProduct (1000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(1000)},
    {"PSDotProduct (10000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(10000)},
    {"PSDotProduct (100000)", &maths_tag, 0, 10, mathsDotProductBenchmark,
     1, INTARGS(100000)},
    {"PSDot ((100,200) * 200)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(100, 200)},
    {"PSDot ((1000,3000) * 3000)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(1000, 3000)},
    {"PSDot ((1000,50000) * 50000)", &maths_tag, 0, 10, mathsDotBenchmark,
     2, INTARGS(10000, 50000)},
    {"PSVectorProduct (100,200)", &maths_tag, 0, 10, mathsVecProdBenchmark,
     2, INTARGS(100, 200)},
    {"PSVectorProduct (1000,3000)", &maths_tag, 0, 10,
     mathsVecProdBenchmark, 2, INTARGS(1000, 3000)},
    {"PSVectorProduct (1000,50000)", &maths_tag, 0, 10,
     mathsVecProdBenchmark, 2, INTARGS(10000, 50000)},
    {"PSSumVectors (1000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(1000)},
    {"PSSumVectors (10000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(10000)},
    {"PSSumVectors (100000)", &maths_tag, 0, 10, mathsSumVBenchmark,
     1, INTARGS(100000)},
    {"PSSubtractVectors (1000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(1000)},
    {"PSSubtractVectors (10000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(10000)},
    {"PSSubtractVectors (100000)", &maths_tag, 0, 10, mathsSubVBenchmark,
     1, INTARGS(100000)},
    {"PSMultiplyVectors (1000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(1000)},
    {"PSMultiplyVectors (10000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(10000)},
    {"PSMultiplyVectors (100000)", &maths_tag, 0, 10, mathsMulVBenchmark,
     1, INTARGS(100000)},
    {"PSDivideVectors (1000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(1000)},
    {"PSDivideVectors (10000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(10000)},
    {"PSDivideVectors (100000)", &maths_tag, 0, 10, mathsDivVBenchmark,
     1, INTARGS(100000)},
    {"PSMatrixProduct (100,300,200)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(100, 300, 200)},
    {"PSMatrixProduct (1000,3000,2000)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(1000, 3000, 2000)},
    /*{"PSMatrixProduct (1000,10000,5000)", &maths_tag, 0, 10,
     mathsMatrixProdBenchmark, 3, INTARGS(1000, 10000, 5000)},*/
    {"PSSigmoidV (1000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(1000)},
    {"PSSigmoidV (10000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(10000)},
    {"PSSigmoidV (100000)", &activation_tag, 0, 10, actSigmoidBenchmark,
     1, INTARGS(100000)},
    {"PSTanhV (1000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(1000)},
    {"PSTanhV (10000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(10000)},
    {"PSTanhV (100000)", &activation_tag, 0, 10, actTanhBenchmark,
     1, INTARGS(100000)},
    {"PSReluV (1000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(1000)},
    {"PSReluV (10000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(10000)},
    {"PSReluV (100000)", &activation_tag, 0, 10, actReluBenchmark,
     1, INTARGS(100000)},
    {"PSSigmoidDerivativeV (1000)", &activation_tag, 0, 10,
      actSigmoidDerivBenchmark, 1, INTARGS(1000)},
    {"PSSigmoidDerivativeV (10000)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, INTARGS(10000)},
    {"PSSigmoidDerivativeV (100000)", &activation_tag, 0, 10,
     actSigmoidDerivBenchmark, 1, INTARGS(100000)},
    {"PSTanhDerivativeV (1000)", &activation_tag, 0, 10,
      actTanhDerivBenchmark, 1, INTARGS(1000)},
    {"PSTanhDerivativeV (10000)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, INTARGS(10000)},
    {"PSTanhDerivativeV (100000)", &activation_tag, 0, 10,
     actTanhDerivBenchmark, 1, INTARGS(100000)},
    {"PSReluDerivativeV (1000)", &activation_tag, 0, 10,
      actReluDerivBenchmark, 1, INTARGS(1000)},
    {"PSReluDerivativeV (10000)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, INTARGS(10000)},
    {"PSReluDerivativeV (100000)", &activation_tag, 0, 10,
     actReluDerivBenchmark, 1, INTARGS(100000)},
    {"PSDefaultOptimization (1000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(1000)},
    {"PSDefaultOptimization (10000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(10000)},
    {"PSDefaultOptimization (100000)", &optimization_tag, 0, 10,
     optimDefaultBenchmark, 1, INTARGS(100000)},
    {"FullyConnected Feed (500,1000)", &fullnet_tag, 0, 10,
     fullnetFeedforwardBenchmark, 2, INTARGS(500,1000)},
    {"FullyConnected Feed (5000,10000)", &fullnet_tag, 0, 10,
     fullnetFeedforwardBenchmark, 2, INTARGS(5000,10000)},
    {"FullyConnected Feed (10000,20000)", &fullnet_tag, 0, 10,
     fullnetFeedforwardBenchmark, 2, INTARGS(10000,20000)},
};

/* Main functions */

void printHelp(char *executable) {
    fprintf(stderr, "Usage: %s [OPTIONS] [TEST_ID, ...]\n", executable);
    fprintf(stderr, "\nOPTIONS:\n\n");
    fprintf(stderr, "   --skip TEST_ID          Skip test (can be used "
        "multiple times\n");
    fprintf(stderr, "   --list-tests            List all test IDS\n");
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
        } else if (strcmp("--list-tests", arg) == 0) {
            printTagList();
            exit(1);
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

int main(int argc, char **argv) {
    int return_val = 0;
    int tot_benchmarks = sizeof(bechmarks) / sizeof(PSBenchmarkConfig),
        performed_benchmarks = 0, i;
    int argidx = parseOptions(argc, argv), all_disabled = 0;
    while (argidx < argc) {
        if (!all_disabled) disableAllTags();
        char *test_id = argv[argidx++];
        if (!setTagEnabledStatus(test_id, 1)) return 1;
    }
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    for (i = 0; i < tot_benchmarks; i++) {
        PSBenchmarkConfig *cfg = &(bechmarks[i]);
        if (cfg->tag != NULL && !(*cfg->tag)) continue;
        if (cfg->do_benchmark != NULL) {
            printf(
                PSCOLOR_BOLD "Performing Benchmark \"%s\"\n" PSCOLOR_RESET,
                cfg->name
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
            performed_benchmarks++;
            free(results);
            time_t elapsed = PSGetElapsedTimeUS(st, et);
            printf(
                " -> %d benchmark(s) performed in %s\n\n",
                num_results, PSGetElapsedTimeString(elapsed, 1)
            );
        }
    }
final:
    return return_val;
}

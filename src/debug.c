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

 /*
  *  Crash handler code (segvHandler, getEip, logStackTrace, logHexDump,
  *  dumpX86Calls) is (totally or partially) based on Redis (R)
  *  (https://github.com/redis/redis).
  *  Redis is an open-source (BSD-license) software owned by Redis Ltd.
  *  Code has been based on functions found inside Redis (R) debug.c source.
  *  Copyright (c) 2009-2020, Salvatore Sanfilippo <antirez at gmail dot com>
  *  Copyright (c) 2020, Redis Labs, Inc
 */

#define _XOPEN_SOURCE 1
#if defined(__linux__)
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <stdlib.h>
#include <arpa/inet.h>
#include <signal.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "buildinfo.h"
#include "platform.h"
#include "config.h"
#include "buildinfo.h"
#include "psyc.h"
#include "maths.h"
#include "debug.h"
#include "convolutional.h"
#include "lstm.h"
#include "log.h"
#include "utils.h"
#include "dataset.h"
#define UNUSED(V) ((void) V)

#ifdef BACKTRACE_AVAILABLE
#include <execinfo.h>
#ifndef __OpenBSD__
#include <ucontext.h>
#else
typedef ucontext_t sigcontext_t;
#endif
#include <fcntl.h>
#include <unistd.h>
#include <fenv.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif
#endif /* BACKTRACE_AVAILABLE */

#ifdef __CYGWIN__
#ifndef SA_ONSTACK
#define SA_ONSTACK 0x08000000
#endif
#endif

#if defined(__APPLE__) && defined(__arm64__)
#include <mach/mach.h>
#endif

int PSOriginalStdOutFD = -999;
PSDebugInfo last_debug_info = {0};
char *PSDumpGradientsPath = NULL;
static void printLastDebugInfo(void);
int writeSerializedFloat(FILE *out, PSFloat fnum, int opts);
void DumpLayerInfo(PSLayer *layer, FILE *dump_file, int add_new_line);
int (*PSShouldDumpGradientsCallback) (PSModel *model) = NULL;
int writeSerializedFloat(FILE *out, PSFloat fnum, int opts);
const char *PSGetActivationName(PSActivationFunction func);

#ifdef BACKTRACE_AVAILABLE
static void *getEip(ucontext_t *uc) {
#if defined(__APPLE__) && !defined(MAC_OS_X_VERSION_10_6)
    /* OSX < 10.6 */
    #if defined(__x86_64__)
    return (void*) uc->uc_mcontext->__ss.__rip;
    #elif defined(__i386__)
    return (void*) uc->uc_mcontext->__ss.__eip;
    #else
    return (void*) uc->uc_mcontext->__ss.__srr0;
    #endif
#elif defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_6)
    /* OSX >= 10.6 */
    #if defined(_STRUCT_X86_THREAD_STATE64) && !defined(__i386__)
    return (void*) uc->uc_mcontext->__ss.__rip;
    #elif defined(__i386__)
    return (void*) uc->uc_mcontext->__ss.__eip;
    #else
    return (void*) arm_thread_state64_get_pc(uc->uc_mcontext->__ss);
    #endif
#elif defined(__linux__)
    /* Linux */
    #if defined(__i386__) || defined(__ILP32__)
    return (void*) uc->uc_mcontext.gregs[14]; /* Linux 32 */
    #elif defined(__X86_64__) || defined(__x86_64__)
    return (void*) uc->uc_mcontext.gregs[16]; /* Linux 64 */
    #elif defined(__ia64__) /* Linux IA64 */
    return (void*) uc->uc_mcontext.sc_ip;
    #elif defined(__arm__) /* Linux ARM */
    return (void*) uc->uc_mcontext.arm_pc;
    #elif defined(__aarch64__) /* Linux AArch64 */
    return (void*) uc->uc_mcontext.pc;
    #else
    return NULL;
    #endif
#elif defined(__FreeBSD__)
    /* FreeBSD */
    #if defined(__i386__)
    return (void*) uc->uc_mcontext.mc_eip;
    #elif defined(__x86_64__)
    return (void*) uc->uc_mcontext.mc_rip;
    #else
    return NULL;
    #endif
#elif defined(__OpenBSD__)
    /* OpenBSD */
    #if defined(__i386__)
    return (void*) uc->sc_eip;
    #elif defined(__x86_64__)
    return (void*) uc->sc_rip;
    #else
    return NULL;
    #endif
#elif defined(__DragonFly__)
    return (void*) uc->uc_mcontext.mc_rip;
#else
    return NULL;
#endif
}

void logStackTrace(ucontext_t *uc) {
    void *trace[101];
    int trace_size = 0, fd = fileno(stdout);/* STDOUT_FILENO;*/
    trace_size = backtrace(trace + 1, 100);

    if (getEip(uc) != NULL) {
        char *msg1 = "EIP:\n";
        char *msg2 = "\nBacktrace:\n";
        if (write(fd,msg1,strlen(msg1)) == -1) {};
        trace[0] = getEip(uc);
        backtrace_symbols_fd(trace, 1, fd);
        if (write(fd,msg2,strlen(msg2)) == -1) {};
    }

    backtrace_symbols_fd(trace+1, trace_size, fd);
}

static void logHexDump(char *descr, void *value, size_t len) {
    char buf[65], *b;
    unsigned char *v = value;
    char charset[] = "0123456789abcdef";

    printf("%s (hexdump of %zu bytes):\n", descr, len);
    b = buf;
    while(len) {
        b[0] = charset[(*v)>>4];
        b[1] = charset[(*v)&0xf];
        b[2] = '\0';
        b += 2;
        len--;
        v++;
        if (b-buf == 64 || len == 0) {
            printf("%s", buf);
            b = buf;
        }
    }
    printf("\n");
}

void dumpX86Calls(void *addr, size_t len) {
    size_t j;
    unsigned char *p = addr;
    Dl_info info;
    /* Hash table to best-effort avoid printing the same symbol
     * multiple times. */
    unsigned long ht[256] = {0};

    if (len < 5) return;
    for (j = 0; j < len-4; j++) {
        if (p[j] != 0xE8) continue; /* Not an E8 CALL opcode. */
        unsigned long target = (unsigned long)addr+j+5;
        target += *((int32_t*)(p+j+1));
        if (dladdr((void*)target, &info) != 0 && info.dli_sname != NULL) {
            if (ht[target&0xff] != target) {
                printf("Function at 0x%lx is %s\n",target,info.dli_sname);
                ht[target&0xff] = target;
            }
            j += 4; /* Skip the 32 bit immediate. */
        }
    }
}

#pragma fenv_access(on)
static void dumpfloatinPointExecption() {
    printf("\n\n-- FLOATING POINT EXC. --\n");
    if (fetestexcept(FE_DIVBYZERO)) printf("FE_DIVBYZERO catched\n");
    if (fetestexcept(FE_OVERFLOW)) printf("FE_OVERFLOW catched\n");
    if (fetestexcept(FE_UNDERFLOW)) printf("FE_UNDERFLOW catched\n");
    if (fetestexcept(FE_INEXACT)) printf("FE_INEXACT catched\n");
    if (fetestexcept(FE_INVALID)) printf("FE_INVALID catched\n");
}

void segvHandler(int sig, siginfo_t *info, void *secret) {
    ucontext_t *uc = (ucontext_t*) secret;
    void *eip = getEip(uc);
    struct sigaction act;

    if (PSOriginalStdOutFD >= 0 && fileno(stdout) != PSOriginalStdOutFD) {
        /* STDOUT has been redirected, restore it. */
        fflush(stdout);
        fclose(stdout);
        stdout = fdopen(PSOriginalStdOutFD, "w");
        PSOriginalStdOutFD = -999;
    }

    fflush(stdout);
    printf("\n\n=== BUG REPORT ===\n");
    fflush(stdout);
    printf("Psyc %s crashed by signal: %d\n", PSYC_VERSION, sig);
    if (eip != NULL)
        printf("Running instruction at: %p\n", eip);
    if (sig == SIGSEGV || sig == SIGBUS)
        printf("Accessing address: %p\n", (void*)info->si_addr);

    struct utsname sysinfo;
    uname(&sysinfo);
    printf("\n\n------ MISC. INFO -------\n");
    printf("Git SHA:            %s\n", PSYC_GIT_SHA);
    printf("Git Dirty:          %s\n", PSYC_GIT_DIRTY);
    printf("Git Branch:         %s\n", PSYC_GIT_BRANCH);
    printf("OS:                 %s %s %s\n",
        sysinfo.sysname, sysinfo.release, sysinfo.machine);
    printf("Arch.:              %dbit\n", (sizeof(long) == 8 ? 64 : 32));
    printf("AVX:                ");
#if USE_AVX
    printf("yes\n");
#else
    printf("no\n");
#endif
    printf("Apple(R) Accelerate:   ");
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    printf("yes\n");
#else
    printf("no\n");
#endif
    printf("BLAS:               ");
#ifdef HAS_BLAS
    printf("yes\n");
#else
    printf("no\n");
#endif
    printf("CBLAS:              ");
#ifdef HAS_CBLAS
    printf("yes\n");
#else
    printf("no\n");
#endif
    printf("GCC:                %d.%d.%d\n",
#ifdef __GNUC__
            __GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);
#else
            0,0,0);
#endif
#ifdef __clang__
#ifdef __clang_version__
    printf("Clang:              %s\n", __clang_version__);
#else
    printf("Clang:              yes\n");
#endif
#endif
#ifdef PS_OPTIMIZATION
    printf("Optimization:       %s\n", PS_OPTIMIZATION);
#endif
    printf("Global Flags:       %d\n", PSGlobalFlags);
    printf("Unixtime:           %lu\n", time(NULL));
    if (last_debug_info.has_info) printLastDebugInfo();

    printf("\n\n------ STACK TRACE ------\n");
    logStackTrace(uc);

    if (sig == SIGFPE) dumpfloatinPointExecption();

    printf("\n\n---- SIZEOF STRUCTS ----\n");
    printf("PSTrainingOptions: %d\n", (int) sizeof(PSTrainingOptions));
    printf("PSTrainingInfo:    %d\n", (int) sizeof(PSTrainingInfo));
    printf("PSNeuron:          %d\n", (int) sizeof(PSNeuron));
    printf("PSLayer:           %d\n", (int) sizeof(PSLayer));
    printf("PSModel:   %d\n", (int) sizeof(PSLayer));
    printf("PSDict:            %d\n", (int) sizeof(PSDict));
    printf("PSDictItem:        %d\n", (int) sizeof(PSDictItem));
    printf("PSVocabulary:      %d\n", (int) sizeof(PSVocabulary));
    printf("\n\n---- SIZEOF TYPES ----\n");
    printf("PSFloat: %d\n", (int) sizeof(PSFloat));
#if USE_AVX
    printf("\n\n---- AVX ----\n");
    printf("AVX_VECTOR_SIZE:        %d\n", AVX_VECTOR_SIZE);
    printf("AVX128_VECTOR_SIZE:     %d\n", AVX128_VECTOR_SIZE);
    printf("AVX256_VECTOR_SIZE:     %d\n", AVX256_VECTOR_SIZE);
    printf("AVX_MIN_VECTOR_SIZE:    %d\n", AVX_MIN_VECTOR_SIZE);
    printf("AVX_MAX_VECTOR_SIZE:    %d\n", AVX_MAX_VECTOR_SIZE);
#endif

    if (eip != NULL) {
        Dl_info info;
        if (dladdr(eip, &info) != 0) {
            printf(
                "\n\n------ DUMPING CODE AROUND EIP ------\n"
                "Symbol: %s (base: %p)\n"
                "Module: %s (base %p)\n"
                "$ xxd -r -p /tmp/dump.hex /tmp/dump.bin\n"
                "$ objdump --adjust-vma=%p -D -b binary -m i386:x86-64 "
                "/tmp/dump.bin\n"
                "------\n",
                info.dli_sname, info.dli_saddr, info.dli_fname, info.dli_fbase,
                info.dli_saddr);
            size_t len = (long)eip - (long)info.dli_saddr;
            unsigned long sz = sysconf(_SC_PAGESIZE);
            if (len < 1<<13) {
                unsigned long next = ((unsigned long)eip + sz) & ~(sz-1);
                unsigned long end = (unsigned long)eip + 128;
                if (end > next) end = next;
                len = end - (unsigned long)info.dli_saddr;
                logHexDump("dump of function ", info.dli_saddr ,len);
                dumpX86Calls(info.dli_saddr,len);
            }
        }
    }

    sigemptyset (&act.sa_mask);
    act.sa_flags = SA_NODEFER | SA_ONSTACK | SA_RESETHAND;
    act.sa_handler = SIG_DFL;
    sigaction (sig, &act, NULL);
    kill(getpid(),sig);
}


#endif /* BACKTRACE_AVAILABLE */

#if defined(__APPLE__) && defined(__MACH__)

/*  Public domain polyfill for feenableexcept on OS X */
/*  http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c */

#if defined(__arm) || defined(__arm64) || defined(__aarch64__)
#define IS_ARM 1
#define FE_EXCEPT_SHIFT 8
#endif

int feenableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    /*  previous masks */
    unsigned int old_excepts;

    if (fegetenv(&fenv)) {
        return -1;
    }
#if (IS_ARM == 1)
    UNUSED(new_excepts);
    old_excepts = fenv.__fpcr;
    /*  unmask */
    fenv.__fpcr = fenv.__fpcr | (excepts << FE_EXCEPT_SHIFT);
#else
    old_excepts = fenv.__control & FE_ALL_EXCEPT;
    /*  unmask */
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr   &= ~(new_excepts << 7);
#endif

    return fesetenv(&fenv) ? -1 : old_excepts;
}

int fedisableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    /*  all previous masks */
    unsigned int old_excepts;

    if (fegetenv(&fenv)) {
        return -1;
    }
#if (IS_ARM == 1)
    UNUSED(new_excepts);
    old_excepts = fenv.__fpcr;
    fenv.__fpcr &= ~(excepts << FE_EXCEPT_SHIFT);
#else
    old_excepts = fenv.__control & FE_ALL_EXCEPT;
    /*  mask */
    fenv.__control |= new_excepts;
    fenv.__mxcsr   |= new_excepts << 7;
#endif

    return fesetenv(&fenv) ? -1 : old_excepts;
}

#endif

char *getLossFunctionName(PSLossFunction function);
char *getModelStatusLabel(PSModel *model);

int PSIsFunctionAvailable(const char *func) {
    return dlsym(RTLD_DEFAULT, func) != NULL;
}

int PSCatchFloatingPointExceptions(int except) {
    if (!PSIsFunctionAvailable("feenableexcept")) return 0;
    feenableexcept(except);
    return 1;
}

char *PSGetNeuronDebugID(PSNeuron *neuron, PSLayer *layer) {
    static char neuron_id[255];
    if (neuron == NULL || layer == NULL) return "null";
    int n_index = neuron->index;
    int l_lindex = layer->index;
    if (layer->output_depth > 1) {
        int fsize = layer->size / layer->output_depth;
        int f_index = n_index / fsize;
        snprintf(neuron_id, 255, "%d-%d-%d", l_lindex, f_index, n_index);
    } else snprintf(neuron_id, 255, "%d-%d", l_lindex, n_index);
    return neuron_id;
}

void PSTrainingDebugDump(PSModel *model, char *format, ...) {
    if (model->training == NULL) return;
    if (model->training->debug_dump_to == NULL) return;
    if (model->training->current_element > 0) return;
    if (model->training->current_batch > 0) return;
    if (model->training->current_epoch > 0) return;
    va_list ap;
    va_start(ap, format);
    vfprintf(model->training->debug_dump_to, format, ap);
    va_end(ap);
}

void PSTrainingDebugDumpStep(PSDebugStepInfo *info, char *format, ...) {
    if (info == NULL) return;
    PSModel *model = info->model;
    if (model->training == NULL) return;
    if (model->training->debug_dump_to == NULL) return;
    if (model->training->current_element > 0) return;
    if (model->training->current_batch > 0) return;
    if (model->training->current_epoch > 0) return;
    int training_phase = info->training_phase;
    char *phase_name = NULL;
    if (training_phase == PS_TRAINING_PHASE_FORWARD)
        phase_name = "forward";
    else if (training_phase == PS_TRAINING_PHASE_BACKPROP)
        phase_name = "backprop";
    else phase_name = "unknown";
    fprintf(
        model->training->debug_dump_to,
        "step:phase=%s,func=%s",
        phase_name, info->func
    );
    if (info->layer != NULL) {
        char *type_name = PSGetLayerTypeLabel(info->layer);
        fprintf(model->training->debug_dump_to,
            ",layer=%d,type=%s", info->layer->index, type_name);
        if (info->neuron != NULL) {
            char *neuron_id =
                PSGetNeuronDebugID(info->neuron, info->layer);
            fprintf(model->training->debug_dump_to,",neuron=%s",neuron_id);
        }
    }
    if (format != NULL) {
        fprintf(model->training->debug_dump_to, ",");
        va_list ap;
        va_start(ap, format);
        vfprintf(model->training->debug_dump_to, format, ap);
        va_end(ap);
    }
}

void PSTrainingDebugDumpGradient(PSModel *model,
                                 int phase,
                                 const char *func,
                                 PSLayer *layer,
                                 int gradient_idx,
                                 int weight_size,
                                 int weight_idx,
                                 int is_avx,
                                 int avx_len)
{
    if (model->training == NULL) return;
    if (model->training->debug_dump_to == NULL) return;
    int batch_size = model->training->batch_size;
    if (model->training->current_element != (batch_size - 1)) return;
    char *phase_name = NULL;
    switch (phase) {
    case PS_DEBUG_PHASE_UPDATE_GRADS: phase_name = "update_gradients"; break;
    case PS_DEBUG_PHASE_UPDATE_WEIGHTS: phase_name = "update_weights"; break;
    default: phase_name = "unknown";
    }
    fprintf(
        model->training->debug_dump_to, "gradient:phase=%s,func=%s",
        phase_name, func
    );
    if (layer != NULL) {
        char *type_name = PSGetLayerTypeLabel(layer);
        fprintf(model->training->debug_dump_to,
            ",layer=%d,type=%s",layer->index, type_name);
    }
    fprintf(model->training->debug_dump_to, ",gradient_idx=%d,weight_size=%d",
        gradient_idx, weight_size);
    int last_widx = -1;
    if (!is_avx) last_widx = weight_size - 1;
    else {
        int avx_steps = weight_size / avx_len;
        last_widx = (avx_steps * avx_len) - 1;
    }
    fprintf(model->training->debug_dump_to, ",weight_range=(%d,%d)",
        weight_idx, last_widx);
    if (is_avx) {
        fprintf(
            model->training->debug_dump_to, ",avx=1,avx_step_len=%d\n",
            avx_len
        );
    } else fprintf(model->training->debug_dump_to, "\n");
}

void PSTrainingDebugDumpHeader(PSModel *model,
                              int data_size,
                              int test_size,
                              int epochs,
                              PSFloat learning_rate,
                              int batch_size)
{
    if (model->training == NULL) return;
    if (model->training->debug_dump_to == NULL) return;
    PSTrainingDebugDump(model, "### HEADER\n");
    PSTrainingDebugDump(model, "psyc:version=%s\n", PSYC_VERSION);
    const char *name = model->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED MODEL";
    char *loss_name = getLossFunctionName(model->loss);
    int avx_enabled = !PSAVXEnabled(model->acceleration);
    PSTrainingDebugDump(model,
        "network:name=%s,size=%d,loss_function=%s,status=%s,avx=%d\n",
        name, model->size, loss_name, getModelStatusLabel(model),
        avx_enabled
    );
    int i;
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        PSLayerType ltype = layer->type;
        char *type_name = PSGetLayerTypeLabel(layer);
        PSTrainingDebugDump(model, "layer:index=%d,type=%s,size=%d",
            i, type_name, layer->size);
        if (i == 0 && layer->flags & PS_FLAG_ONEHOT) {
            PSTrainingDebugDump(model, ",vector_size=%d",
                                layer->onehot_vector_size);
        }
        if (ltype == Convolutional || ltype == Pooling) {
            PSConvolutionalSettings *settings =
                PSGetConvolutionalSettings(layer);
            int input_w = 0, input_h = 0, filter_w = 0, filter_h = 0;
            int stride = 0, padding = 0;
            if (settings != NULL) {
                input_w = settings->input_width;
                input_h = settings->input_height;
                stride = settings->stride;
                padding = settings->padding;
                filter_w = settings->filter_width;
                filter_h = settings->filter_height;
            }
            if (stride <= 0 && ltype == Pooling) stride = filter_w;
            PSTrainingDebugDump(
                model,
                ",input_size=%dx%d,output_size=%dx%d,features=%d"
                ",region=%dx%d,stride=%d",
                input_w, input_h, layer->output_columns, layer->output_rows,
                layer->output_depth, filter_w, filter_h, stride
            );
            if (ltype == Convolutional) {
                if (padding < 0) padding = 0;
                PSTrainingDebugDump(
                    model, ",padding=%d", padding
                );
            }
            const char *actvname = PSGetActivationName(layer->activate);
            if (actvname != NULL)
                PSTrainingDebugDump(model, ",activation=%s\n", actvname);
            else PSTrainingDebugDump(model, "\n");
        } else if (ltype == FullyConnected) {
            if (layer->output_depth > 1) {
                PSTrainingDebugDump(
                    model, ",features=%d\n", layer->output_depth
                );
            } else PSTrainingDebugDump(model, "\n");
        } else PSTrainingDebugDump(model, "\n");
    }
    PSTrainingDebugDump(model,
        "training:started_at=%ld,data_size=%d,test_size=%d,batch_size=%d,"
        "epochs=%d,learning_rate=%.3f\n",
        time(NULL), data_size, test_size, batch_size, epochs, learning_rate
    );
}

static int dumpModelGradients(PSModel *model,
                                PSGradient **gradients,
                                FILE *f, PSTrainingOptions *opts)
{
    int success = 1;
    PSFloat clip_h = 0.0, clip_l = 0.0;
    int i, j, apply_clip = 0;
    if (opts != NULL) {
        if ((apply_clip = (opts->clip != 0.0))) {
            clip_h = PSAbs(opts->clip);
            clip_l = clip_h * -1;
        }
    }
    for (i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        DumpLayerInfo(layer, f, 0);
        if (i == 0) {
            fprintf(f, ",weight_gradients=(),bias_gradients=()\n");
            continue;
        }
        PSGradient *lgradients = gradients[i - 1];
        if (lgradients == NULL) {
            fprintf(f, ",weight_gradients=(),bias_gradients=()\n");
            continue;
        }
        if (lgradients->weight_count > 0 && lgradients->weights == NULL) {
            PSErr(NULL, "gradients[%d]: missing weights", (i - 1));
            success = 0;
            goto final;
        }
        fprintf(f, ",weight_gradients=(");
        for(j = 0; (uint64_t) j < lgradients->weight_count; j++) {
            PSFloat wg = lgradients->weights[j];
            if (apply_clip) wg = PSClipValue(wg, clip_l, clip_h);
            if (j > 0) fprintf(f, ",");
            writeSerializedFloat(f, wg, 0);
        }
        if (lgradients->bias_count > 0 && lgradients->biases == NULL) {
            PSErr(NULL, "gradients[%d]: missing biases", (i - 1));
            success = 0;
            goto final;
        }
        fprintf(f, "),bias_gradients=(");
        for(j = 0; (uint64_t) j < lgradients->bias_count; j++) {
            if (j > 0) fprintf(f, ",");
            PSFloat bg = lgradients->biases[j];
            if (apply_clip) bg = PSClipValue(bg, clip_l, clip_h);
            writeSerializedFloat(f, bg, 0);
        }
        fprintf(f, ")\n");
    }
final:
    return success;
}

int PSDumpGradients(PSModel *model, PSGradient ***gradients,
                    const char* filename, PSTrainingOptions *opts)
{
    assert(model != NULL);
    if (model->size == 0) {
        PSErr(NULL, "Empty model!\n");
        return 0;
    }
    if (PSShouldDumpGradientsCallback != NULL) {
        if (!PSShouldDumpGradientsCallback(model)) return 0;
    }
    int success = 1;
    char default_filename[PATH_MAX];
    if (filename == NULL) {
        if (PSDumpGradientsPath == NULL) {
            PSErr(NULL, "Cannot dump gradients: missing filename or "
                  "PSDumpGradientsPath");
            return 0;
        }
        char *p = default_filename;
        int maxlen = PATH_MAX - 1;
        int len = snprintf(p, maxlen, "%s", PSDumpGradientsPath);
        if (len >= maxlen) {
            PSErr(NULL, "PSDumpGradientsPath too long!");
            return 0;
        }
        if (p[len - 1] != '/') {
            p[len] = '/';
            if (++len >= maxlen) {
                PSErr(NULL, "PSDumpGradientsPath too long!");
                return 0;
            }
        }
        p += len;
        maxlen -= len;
        char *name = (char *) model->name;
        if (name == NULL || strlen(name) == 0) name = "unnamed";
        name = strdup(name);
        int namelen = strlen(name);
        for (int i = 0; i < namelen; i++) {
            char c = name[i];
            int valid = (isalnum(c) || c == '_' || c == '-');
            if (!valid) name[i] = '-';
            else name[i] = tolower(name[i]);
        }
        if (model->training != NULL) {
            int epoch = model->training->current_epoch,
                batch = model->training->current_batch,
                elem  = model->training->current_element;
            len += snprintf(
                p, maxlen, "psyc-gradients-%s-%d-%d-%d.dump",
                name, epoch, batch, elem
            );
        } else len += snprintf(p, maxlen, "psyc-gradients-%s.dump", name);
        filename = default_filename;
        free(name);
    }
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    PSModel *current = model;
    int count = PSModelChainLength(model), nidx = 0;
    if (count > 1) current = PSModelChainHead(model);
    success = (current != NULL);
    if (!success) goto final;
    while (current != NULL) {
        success = dumpModelGradients(current, gradients[nidx++], f, opts);
        if (!success) goto final;
        current = current->next;
    }
final:
    fclose(f);
    return success;
}

void PSResetDebugInfo(void) {
    memset(&last_debug_info, 0, sizeof(last_debug_info));
    last_debug_info.timestep = -1;
    last_debug_info.layer_index = -1;
    last_debug_info.layer2_index = -1;
    last_debug_info.neuron_index = -1;
    last_debug_info.neuron2_index = -1;
    last_debug_info.weight = -99999;
}

void PSAddDebugInfo(PSModel *model, char *file, const char *func,
                    int line, PSLayer *layer, void *neuron1, void *neuron2,
                    char *prop, double val_d, ...)
{
    PSFloat val = (PSFloat) val_d;
    memset(&last_debug_info, 0, sizeof(last_debug_info));
    last_debug_info.has_info = 1;
    last_debug_info.time = time(NULL);
    last_debug_info.file = file;
    last_debug_info.func = func;
    last_debug_info.line = line;
    last_debug_info.timestep = -1;
    last_debug_info.layer_index = -1;
    last_debug_info.layer2_index = -1;
    last_debug_info.neuron_index = -1;
    last_debug_info.neuron2_index = -1;
    last_debug_info.weight = -99999;
    int is_recurrent = 0, t = 0;
    if (model != NULL) {
        PSModel *net = (PSModel *) model;
        last_debug_info.status = net->status;
        if (net->training != NULL) {
            last_debug_info.current_epoch = net->training->current_epoch;
            last_debug_info.current_batch = net->training->current_batch;
            last_debug_info.current_element =
                net->training->current_element;
        }
    }
    if (layer != NULL) {
        PSLayer *l = layer;
        last_debug_info.layer_index = l->index;
        last_debug_info.layer_type = l->type;
        if ((is_recurrent = PSIsRecurrent(layer))) {
            va_list args;
            va_start(args, val_d);
            t = va_arg(args, int);
            va_end(args);
            last_debug_info.timestep = t;
        }
    }
    if (neuron1 != NULL) {
        PSNeuron *n = neuron1;
        last_debug_info.neuron_index = n->index;
        PSLayer *l = (PSLayer *) n->layer;
        if (layer == NULL) {
            last_debug_info.layer_index = l->index;
            last_debug_info.layer_type = l->type;
        }
        last_debug_info.activation = PSGetNeuronState(n, t);
        last_debug_info.bias = (n->bias != NULL ? *(n->bias) : 0);
        if (l->delta != NULL) last_debug_info.delta = l->delta[n->index];
    }
    if (neuron2 != NULL) {
        /* TODO: refactor */
        /*PSNeuron *n2 = neuron2;
        last_debug_info.neuron2_index = n2->index;
        PSLayer *l2 = (PSLayer *) n2->layer;
        last_debug_info.layer2_index = l2->index;
        last_debug_info.activation2 = PSGetNeuronActivation(n2, t);
        if (l2->delta != NULL) last_debug_info.delta2 = l2->delta[n2->index];
        if (neuron1 != NULL) {
            PSNeuron *n = neuron1;
            int widx = 0;
            if (l2->index == (last_debug_info.layer_index - 1)) {
                if (last_debug_info.layer_type != Convolutional &&
                    last_debug_info.layer_type != Pooling)
                {
                    widx = n2->index;
                    if (widx < n->weights_size)
                        last_debug_info.weight = n->weights[widx];
                }
            } else if (l2->index == (last_debug_info.layer_index + 1)) {
                if (l2->type != Convolutional && l2->type != Pooling)
                {
                    widx = n->index;
                    if (widx < n2->weights_size)
                        last_debug_info.weight = n->weights[widx];
                }
            }
        }*/
    }
    last_debug_info.custom_prop = prop;
    last_debug_info.custom_val = val;
}

static void printLastDebugInfo(void) {
    if (!last_debug_info.has_info) return;
    time_t now = time(NULL);
    printf("\n\n------ DEBUG INFO ------\n");
    if (last_debug_info.time > 0) {
        time_t elapsed = now - last_debug_info.time;
        printf("Debug time: %s (%ld sec. before crash)\n",
            ctime(&last_debug_info.time), elapsed);
    }
    if (last_debug_info.file != NULL)
        printf("File: %s\n", last_debug_info.file);
    if (last_debug_info.func != NULL)
        printf("Func: %s\n", last_debug_info.func);
    if (last_debug_info.file != NULL || last_debug_info.func != NULL)
        printf("Line: %d\n", last_debug_info.line);
    if (last_debug_info.layer_index >= 0) {
        printf("Layer: %d\n", last_debug_info.layer_index);
        if (last_debug_info.layer_type >= 0) {
            printf(
                "Layer type: %s\n",
                PSGetLabelForType(last_debug_info.layer_type)
            );
        }
    }
    if (last_debug_info.neuron_index >= 0) {
        printf("Neuron: %d\n", last_debug_info.neuron_index);
        printf(" -> Activation: %g\n", last_debug_info.activation);
        printf(" -> Delta: %g\n", last_debug_info.delta);
        printf(" -> Bias: %g\n", last_debug_info.bias);
    }
    if (last_debug_info.layer2_index>=0 && last_debug_info.neuron2_index>=0) {
        int l2idx = last_debug_info.layer2_index;
        int lidx = last_debug_info.layer_index;
        char *rankstr = NULL;
        if (l2idx == (lidx - 1)) rankstr = "Previous";
        else if (l2idx == (lidx + 1)) rankstr = "Next";
        if (rankstr != NULL) {
            printf("%s Neuron: %d\n", rankstr, last_debug_info.neuron2_index);
            printf(" -> Activation: %g\n", last_debug_info.activation2);
            printf(" -> Delta: %g\n", last_debug_info.delta2);
            if (last_debug_info.weight > -99999)
                printf(" -> Weight: %g\n", last_debug_info.weight);
        }
    }
    if (last_debug_info.custom_prop != NULL) {
        printf(
            "%s: %g\n", last_debug_info.custom_prop, last_debug_info.custom_val
        );
    }
}

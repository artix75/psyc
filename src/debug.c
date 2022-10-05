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

#include <arpa/inet.h>
#include <signal.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "platform.h"
#include "psyc.h"
#include "debug.h"
#include "convolutional.h"

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
#include <xmmintrin.h>
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
static void printLastDebugInfo(void);

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
    #endif
#elif defined(__FreeBSD__)
    /* FreeBSD */
    #if defined(__i386__)
    return (void*) uc->uc_mcontext.mc_eip;
    #elif defined(__x86_64__)
    return (void*) uc->uc_mcontext.mc_rip;
    #endif
#elif defined(__OpenBSD__)
    /* OpenBSD */
    #if defined(__i386__)
    return (void*) uc->sc_eip;
    #elif defined(__x86_64__)
    return (void*) uc->sc_rip;
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

    if (last_debug_info.has_info) printLastDebugInfo();
    printf("\n\n------ STACK TRACE ------\n");
    logStackTrace(uc);

    if (sig == SIGFPE) dumpfloatinPointExecption();

    printf("\n\n---- SIZEOF STRUCTS ----\n");
    printf("PSLayerParameters: %d\n", (int) sizeof(PSLayerParameters));
    printf("PSTrainingOptions: %d\n", (int) sizeof(PSTrainingOptions));
    printf("PSTrainingInfo:    %d\n", (int) sizeof(PSTrainingInfo));
    printf("PSNeuron:          %d\n", (int) sizeof(PSNeuron));
    printf("PSLayer:           %d\n", (int) sizeof(PSLayer));
    printf("PSNeuralNetwork:   %d\n", (int) sizeof(PSLayer));
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
                "$ objdump --adjust-vma=%p -D -b binary -m i386:x86-64 /tmp/dump.bin\n"
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
    old_excepts = env.__fpcr;
    /*  unmask */
    env.__fpcr = env.__fpcr | (excepts << FE_EXCEPT_SHIFT);
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
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    /*  mask */
    fenv.__control |= new_excepts;
    fenv.__mxcsr   |= new_excepts << 7;

    return fesetenv(&fenv) ? -1 : old_excepts;
}

#endif

char *getLossFunctionName(PSLossFunction function);
char *getNetworkStatusLabel(PSNeuralNetwork *network);

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
    int fcount = 0, f_index;
    PSLayerParameters *lparams = layer->parameters;
    if (lparams != NULL) {
        PSFloat *params = lparams->parameters;
        fcount = (int) (params[PARAM_FEATURE_COUNT]);
    }
    if (fcount > 1) {
        int fsize = layer->size / fcount;
        f_index = n_index / fsize;
        snprintf(neuron_id, 255, "%d-%d-%d", l_lindex, f_index, n_index);
    } else snprintf(neuron_id, 255, "%d-%d", l_lindex, n_index);
    return neuron_id;
}

void PSTrainingDebugDump(PSNeuralNetwork *network, char *format, ...) {
    if (network->training == NULL) return;
    if (network->training->debug_dump_to == NULL) return;
    if (network->training->current_element > 0) return;
    if (network->training->current_batch > 0) return;
    if (network->training->current_epoch > 0) return;
    va_list ap;
    va_start(ap, format);
    vfprintf(network->training->debug_dump_to, format, ap);
    va_end(ap);
}

void PSTrainingDebugDumpStep(PSNeuralNetwork *network,
                             int training_phase,
                             char *func,
                             PSLayer *layer,
                             PSNeuron *neuron,
                             char *format, ...)
{
    if (network->training == NULL) return;
    if (network->training->debug_dump_to == NULL) return;
    if (network->training->current_element > 0) return;
    if (network->training->current_batch > 0) return;
    if (network->training->current_epoch > 0) return;
    char *phase_name = NULL;
    if (training_phase == TRAINING_PHASE_FEEDFORWARD)
        phase_name = "feedforward";
    else if (training_phase == TRAINING_PHASE_BACKPROP) phase_name = "backprop";
    else phase_name = "unknown";
    fprintf(
        network->training->debug_dump_to,
        "step:phase=%s,func=%s",
        phase_name, func
    );
    if (layer != NULL) {
        char *type_name = PSGetLayerTypeLabel(layer);
        fprintf(network->training->debug_dump_to,
            ",layer=%d,type=%s",layer->index, type_name);
        if (neuron != NULL) {
            char *neuron_id = PSGetNeuronDebugID(neuron, layer);
            fprintf(network->training->debug_dump_to,",neuron=%s",neuron_id);
        }
    }
    if (format != NULL) {
        fprintf(network->training->debug_dump_to, ",");
        va_list ap;
        va_start(ap, format);
        vfprintf(network->training->debug_dump_to, format, ap);
        va_end(ap);
    }
}

void PSTrainingDebugDumpGradient(PSNeuralNetwork *network,
                                 int phase,
                                 char *func,
                                 PSLayer *layer,
                                 int gradient_idx,
                                 int weight_size,
                                 int weight_idx,
                                 int is_avx,
                                 int avx_len)
{
    if (network->training == NULL) return;
    if (network->training->debug_dump_to == NULL) return;
    int batch_size = network->training->batch_size;
    if (network->training->current_element != (batch_size - 1)) return;
    char *phase_name = NULL;
    switch (phase) {
    case DEBUG_PHASE_UPDATE_GRADS: phase_name = "update_gradients"; break;
    case DEBUG_PHASE_UPDATE_WEIGHTS: phase_name = "update_weights"; break;
    default: phase_name = "unknown";
    }
    fprintf(
        network->training->debug_dump_to, "gradient:phase=%s,func=%s",
        phase_name, func
    );
    if (layer != NULL) {
        char *type_name = PSGetLayerTypeLabel(layer);
        fprintf(network->training->debug_dump_to,
            ",layer=%d,type=%s",layer->index, type_name);
    }
    fprintf(network->training->debug_dump_to, ",gradient_idx=%d,weight_size=%d",
        gradient_idx, weight_size);
    int last_widx = -1;
    if (!is_avx) last_widx = weight_size - 1;
    else {
        int avx_steps = weight_size / avx_len;
        last_widx = (avx_steps *avx_len) - 1;
    }
    fprintf(network->training->debug_dump_to, ",weight_range=(%d,%d)",
        weight_idx, last_widx);
    if (is_avx) {
        fprintf(
            network->training->debug_dump_to, ",avx=1,avx_step_len=%d\n",
            avx_len
        );
    } else fprintf(network->training->debug_dump_to, "\n");
}

void PSTrainingDebugDumpHeader(PSNeuralNetwork *network,
                              int data_size,
                              int test_size,
                              int epochs,
                              PSFloat learning_rate,
                              int batch_size)
{
    if (network->training == NULL) return;
    if (network->training->debug_dump_to == NULL) return;
    PSTrainingDebugDump(network, "### HEADER\n");
    PSTrainingDebugDump(network, "psyc:version=%s\n", PSYC_VERSION);
    const char *name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    char *loss_name = getLossFunctionName(network->loss);
    int avx_enabled = !PSIsAVXDisabled(network);
    PSTrainingDebugDump(network,
        "network:name=%s,size=%d,loss_function=%s,status=%s,avx=%d\n",
        name, network->size, loss_name, getNetworkStatusLabel(network),
        avx_enabled
    );
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        PSLayerType ltype = layer->type;
        char *type_name = PSGetLayerTypeLabel(layer);
        PSLayerParameters *lparams = layer->parameters;
        PSTrainingDebugDump(network, "layer:index=%d,type=%s,size=%d",
            i, type_name, layer->size);
        if (i == 0 && layer->flags & FLAG_ONEHOT) {
            PSLayerParameters *params = layer->parameters;
            int onehot_sz = (int) (params->parameters[0]);
            PSTrainingDebugDump(network, ",vector_size=%d", onehot_sz);
        }
        if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
            PSFloat *params = lparams->parameters;
            int fcount = (int) (params[PARAM_FEATURE_COUNT]);
            int rsize = (int) (params[PARAM_REGION_SIZE]);
            int input_w = (int) (params[PARAM_INPUT_WIDTH]);
            int input_h = (int) (params[PARAM_INPUT_HEIGHT]);
            int output_w = (int) (params[PARAM_OUTPUT_WIDTH]);
            int output_h = (int) (params[PARAM_OUTPUT_HEIGHT]);
            int stride = (int) (params[PARAM_STRIDE]);
            int use_relu = (int) (params[PARAM_USE_RELU]);
            if (stride <= 0 && ltype == Pooling) stride = rsize;
            PSTrainingDebugDump(
                network,
                ",input_size=%dx%d,output_size=%dx%d,features=%d"
                ",region=%dx%d,stride=%d",
                input_w, input_h, output_w, output_h, fcount,
                rsize, rsize, stride
            );
            if (ltype == Convolutional) {
                char *actv = (use_relu ? "relu" : "sigmoid");
                int padding = (int) (params[PARAM_PADDING]);
                if (padding < 0) padding = 0;
                PSTrainingDebugDump(
                    network, ",padding=%d,activation=%s\n", padding, actv
                );
            } else PSTrainingDebugDump(network, "\n");
        } else if (lparams != NULL && ltype == FullyConnected) {
            PSFloat *params = lparams->parameters;
            int fcount = (int) (params[PARAM_FEATURE_COUNT]);
            if (fcount > 1) {
                PSTrainingDebugDump(
                    network, ",features=%d\n", fcount
                );
            } else PSTrainingDebugDump(network, "\n");
        } else PSTrainingDebugDump(network, "\n");
    }
    PSTrainingDebugDump(network,
        "training:started_at=%ld,data_size=%d,test_size=%d,batch_size=%d,"
        "epochs=%d,learning_rate=%.3f\n",
        time(NULL), data_size, test_size, batch_size, epochs, learning_rate
    );
}

void PSResetDebugInfo(void) {
    memset(&last_debug_info, 0, sizeof(last_debug_info));
    last_debug_info.layer_index = -1;
    last_debug_info.layer2_index = -1;
    last_debug_info.neuron_index = -1;
    last_debug_info.neuron2_index = -1;
    last_debug_info.weight = -99999;
}

void PSAddDebugInfo(void *network, char *file, const char *func, int line,
                    void *layer, void *neuron1, void *neuron2,
                    char *prop, PSFloat val)
{
    memset(&last_debug_info, 0, sizeof(last_debug_info));
    last_debug_info.has_info = 1;
    last_debug_info.time = time(NULL);
    last_debug_info.file = file;
    last_debug_info.func = func;
    last_debug_info.line = line;
    last_debug_info.layer_index = -1;
    last_debug_info.layer2_index = -1;
    last_debug_info.neuron_index = -1;
    last_debug_info.neuron2_index = -1;
    last_debug_info.weight = -99999;
    if (network != NULL) {
        PSNeuralNetwork *net = (PSNeuralNetwork *) network;
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
    }
    if (neuron1 != NULL) {
        PSNeuron *n = neuron1;
        last_debug_info.neuron_index = n->index;
        PSLayer *l = (PSLayer *) n->layer;
        if (layer == NULL) {
            last_debug_info.layer_index = l->index;
            last_debug_info.layer_type = l->type;
        }
        last_debug_info.activation = n->activation;
        last_debug_info.z_value = n->z_value;
        last_debug_info.bias = n->bias;
        if (l->delta != NULL) last_debug_info.delta = l->delta[n->index];
    }
    if (neuron2 != NULL) {
        PSNeuron *n2 = neuron2;
        last_debug_info.neuron2_index = n2->index;
        PSLayer *l2 = (PSLayer *) n2->layer;
        last_debug_info.layer2_index = l2->index;
        last_debug_info.activation2 = n2->activation;
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
        }
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
        printf(" -> ZValue: %g\n", last_debug_info.z_value);
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

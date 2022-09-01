/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.

 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
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

#define _XOPEN_SOURCE 1

#ifdef BACKTRACE_AVAILABLE
#include <execinfo.h>
#ifndef __OpenBSD__
#include <ucontext.h>
#else
typedef ucontext_t sigcontext_t;
#endif
#include <fcntl.h>
#include <unistd.h>
#endif /* BACKTRACE_AVAILABLE */

#ifdef __CYGWIN__
#ifndef SA_ONSTACK
#define SA_ONSTACK 0x08000000
#endif
#endif

#if defined(__APPLE__) && defined(__arm64__)
#include <mach/mach.h>
#endif

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
    int trace_size = 0, fd = STDOUT_FILENO;
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

void segvHandler(int sig, siginfo_t *info, void *secret) {
    ucontext_t *uc = (ucontext_t*) secret;
    void *eip = getEip(uc);
    struct sigaction act;

    printf("=== BUG REPORT ===\n");
    printf("Psyc %s crashed by signal: %d\n", PSYC_VERSION, sig);
    if (eip != NULL)
        printf("Running instruction at: %p\n", eip);
    if (sig == SIGSEGV || sig == SIGBUS)
        printf("Accessing address: %p\n", (void*)info->si_addr);
    printf("\n\n------ STACK TRACE ------\n");
    logStackTrace(uc);

    printf("\n\n---- SIZEOF STRUCTS ----\n");
    printf("PSLayerParameters: %d\n", (int) sizeof(PSLayerParameters));
    printf("PSTrainingOptions: %d\n", (int) sizeof(PSTrainingOptions));
    printf("PSTrainingInfo:    %d\n", (int) sizeof(PSTrainingInfo));
    printf("PSNeuron:          %d\n", (int) sizeof(PSNeuron));
    printf("PSLayer:           %d\n", (int) sizeof(PSLayer));
    printf("PSNeuralNetwork:   %d\n", (int) sizeof(PSLayer));
#if USE_AVX
    printf("\n\n---- AVX ----\n");
    printf("AVX_VECTOR_SIZE:   %d\n", AVX_VECTOR_SIZE);
    printf("AVX_VECTOR2_SIZE:  %d\n", AVX_VECTOR2_SIZE);
    printf("AVX_VECTOR4_SIZE:  %d\n", AVX_VECTOR4_SIZE);
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

char *getLossFunctionName(PSLossFunction function);
char *getNetworkStatusLabel(PSNeuralNetwork *network);

char *PSGetNeuronDebugID(PSNeuron *neuron, PSLayer *layer) {
    static char neuron_id[255];
    if (neuron == NULL || layer == NULL) return "null";
    int n_index = neuron->index;
    int l_lindex = layer->index;
    int fcount = 0, f_index;
    PSLayerParameters *lparams = layer->parameters;
    if (lparams != NULL) {
        double *params = lparams->parameters;
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
        char * type_name = PSGetLayerTypeLabel(layer);
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
        char * type_name = PSGetLayerTypeLabel(layer);
        fprintf(network->training->debug_dump_to,
            ",layer=%d,type=%s",layer->index, type_name);
    }
    fprintf(network->training->debug_dump_to, ",gradient_idx=%d,weight_size=%d",
        gradient_idx, weight_size);
    int last_widx = -1;
    if (!is_avx) last_widx = weight_size - 1;
    else {
        int avx_steps = weight_size / avx_len;
        last_widx = (avx_steps * avx_len) - 1;
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
                              double learning_rate,
                              int batch_size)
{
    if (network->training == NULL) return;
    if (network->training->debug_dump_to == NULL) return;
    PSTrainingDebugDump(network, "### HEADER\n");
    PSTrainingDebugDump(network, "psyc:version=%s\n", PSYC_VERSION);
    const char * name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    char * loss_name = getLossFunctionName(network->loss);
    int avx_enabled = !PSIsAVXDisabled(network);
    PSTrainingDebugDump(network,
        "network:name=%s,size=%d,loss_function=%s,status=%s,avx=%d\n",
        name, network->size, loss_name, getNetworkStatusLabel(network),
        avx_enabled
    );
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        PSLayerType ltype = layer->type;
        char * type_name = PSGetLayerTypeLabel(layer);
        PSLayerParameters * lparams = layer->parameters;
        PSTrainingDebugDump(network, "layer:index=%d,type=%s,size=%d",
            i, type_name, layer->size);
        if (i == 0 && layer->flags & FLAG_ONEHOT) {
            PSLayerParameters * params = layer->parameters;
            int onehot_sz = (int) (params->parameters[0]);
            PSTrainingDebugDump(network, ",vector_size=%d", onehot_sz);
        }
        if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
            double * params = lparams->parameters;
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
                char * actv = (use_relu ? "relu" : "sigmoid");
                int padding = (int) (params[PARAM_PADDING]);
                if (padding < 0) padding = 0;
                PSTrainingDebugDump(
                    network, ",padding=%d,activation=%s\n", padding, actv
                );
            } else PSTrainingDebugDump(network, "\n");
        } else if (lparams != NULL && ltype == FullyConnected) {
            double * params = lparams->parameters;
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

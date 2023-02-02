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
#include <assert.h>
#include "test.h"
#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../utils.h"
#include "../lstm.h"
#include "../mnist.h"

#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define BOLD    "\x1b[1m"
#define DIM     "\x1b[2m"
#define HIDDEN  "\x1b[8m"
#define RESET   "\x1b[0m"


#define getRoundedFloat(d) d/* (round(d * 1000000.0) / 1000000.0) */
#define getRoundedFloatDec(d, dec) (PSRound(d * dec) / dec)

int compareNetworks(PSNeuralNetwork *network, PSNeuralNetwork *other)
{
    int ok = 1, i, k, w;
    char msg[2000];
    for (i = 0; i < network->size; i++) {
        PSLayer *orig_l = network->layers[i];
        PSLayer *other_l = other->layers[i];
        int o_size = orig_l->size;
        int c_size = other_l->size;
        PSLayerType otype = orig_l->type;
        PSLayerType ctype = other_l->type;

        if (i == 0) continue;
        if (otype == Pooling) continue;

        int conv_features_checked = 0;
        for (k = 0; k < o_size; k++) {
            PSNeuron *orig_n = orig_l->neurons[k];
            PSNeuron *other_n = other_l->neurons[k];
            if (otype == Convolutional) {
                PSSharedParams* oshared;
                PSSharedParams* cshared;
                oshared = (PSSharedParams*) orig_l->extra;
                cshared = (PSSharedParams*) other_l->extra;
                if (!conv_features_checked) {
                    conv_features_checked = 1;
                    ok = (oshared->feature_count == cshared->feature_count);
                    if (!ok) {
                        sprintf(msg, "Layer[%d]: Feature count %d != %d\n",
                                i, oshared->feature_count,
                                cshared->feature_count);
                        break;
                    }
                }
                int fsize = orig_l->size / oshared->feature_count;
                int fidx = k / fsize;
                PSFloat obias = getRoundedFloat(oshared->biases[fidx]);
                PSFloat cbias = getRoundedFloat(cshared->biases[fidx]);
                ok = (obias == cbias);
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: bias %.15e != %.15e\n",
                            i, fidx, obias, cbias);
                    break;
                }
            } else if (otype != Recurrent && otype != LSTM) {
                PSFloat obias = getRoundedFloat(orig_n->bias);
                PSFloat cbias = getRoundedFloat(other_n->bias);
                ok = (obias == cbias);
            } else if (otype == LSTM) {
                PSLSTMCell *ocell =  PSGetLSTMCell(orig_n);
                PSLSTMCell *ccell =  PSGetLSTMCell(other_n);
                ok = (ocell->candidate_bias == ccell->candidate_bias);
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: candidate_bias "
                            "%.15e != %.15e\n",
                            i, k, ocell->candidate_bias,
                            ccell->candidate_bias);
                    break;
                }
                ok = (ocell->input_bias == ccell->input_bias);
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: input_bias %.15e != %.15e\n",
                            i, k, ocell->input_bias, ccell->input_bias);
                    break;
                }
                ok = (ocell->output_bias == ccell->output_bias);
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: output_bias %.15e != %.15e\n",
                            i, k, ocell->output_bias, ccell->output_bias);
                    break;
                }
                ok = (ocell->forget_bias == ccell->forget_bias);
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: forget_bias %.15e != %.15e\n",
                            i, k, ocell->forget_bias, ccell->forget_bias);
                    break;
                }
            }
            if (!ok) {
                sprintf(msg, "Layer[%d][%d]: bias %.15e != %.15e\n",
                        i, k, orig_n->bias, other_n->bias);
                break;
            }
            for (w = 0; w < orig_n->weights_size; w++) {
                PSFloat ow = getRoundedFloat(orig_n->weights[w]);
                PSFloat cw = getRoundedFloat(other_n->weights[w]);
                ok = ow == cw;
                if (!ok) {
                    sprintf(msg, "Layer[%d][%d]: w[%d] %.15e != %.15e\n",
                            i, k, w, ow, cw);
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
    }
    return ok;
}

int main(int argc, char** argv) {
    PSHandleSignals(NULL);
    PSNeuralNetwork *std_network = PSCreateNetwork("STD Network");
    PSNeuralNetwork *avx_network = PSCreateNetwork("AVX Network");
    printf(DIM);
    int loaded = PSLoadNetwork(std_network, "/tmp/no_avx.nn.data");
    assert(loaded);
    loaded = PSLoadNetwork(avx_network, "/tmp/avx.nn.data");
    assert(loaded);
    printf(RESET CYAN "Fully Connected comparison: " RESET);
    int ok = compareNetworks(std_network, avx_network);
    if (!ok) printf(RED "FAILED" RESET);
    else printf(GREEN "OK\n" RESET);

    PSDeleteNetwork(std_network);
    PSDeleteNetwork(avx_network);

    printf(DIM);

    std_network = PSCreateNetwork("STD CNN Network");
    avx_network = PSCreateNetwork("AVX CNN Network");

    loaded = PSLoadNetwork(std_network, "/tmp/no_avx.cnn.data");
    assert(loaded);
    loaded = PSLoadNetwork(avx_network, "/tmp/avx.cnn.data");
    assert(loaded);

    printf(RESET CYAN "Convolutional comparison: " RESET);
    ok = compareNetworks(std_network, avx_network);
    if (!ok) printf(RED "FAILED" RESET);
    else printf(GREEN "OK\n" RESET);

    PSDeleteNetwork(std_network);
    PSDeleteNetwork(avx_network);

    std_network = PSCreateNetwork("STD CNN Network");
    avx_network = PSCreateNetwork("AVX CNN Network");

    loaded = PSLoadNetwork(std_network, "/tmp/no_avx.l2_nn.data");
    assert(loaded);
    loaded = PSLoadNetwork(avx_network, "/tmp/avx.l2_nn.data");
    assert(loaded);
    printf(RESET CYAN "Fully Connected L2 comparison: " RESET);
    ok = compareNetworks(std_network, avx_network);
    if (!ok) printf(RED "FAILED" RESET);
    else printf(GREEN "OK\n" RESET);

    PSDeleteNetwork(std_network);
    PSDeleteNetwork(avx_network);

    printf(DIM);

    std_network = PSCreateNetwork("STD CNN Network");
    avx_network = PSCreateNetwork("AVX CNN Network");

    loaded = PSLoadNetwork(std_network, "/tmp/no_avx.l2_cnn.data");
    assert(loaded);
    loaded = PSLoadNetwork(avx_network, "/tmp/avx.l2_cnn.data");
    assert(loaded);

    printf(RESET CYAN "Convolutional L2 comparison: " RESET);
    ok = compareNetworks(std_network, avx_network);
    if (!ok) printf(RED "FAILED" RESET);
    else printf(GREEN "OK\n" RESET);

    PSDeleteNetwork(std_network);
    PSDeleteNetwork(avx_network);

    return 0;
}

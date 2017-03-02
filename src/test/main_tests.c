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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test.h"
#include "../neural.h"
#include "../mnist.h"

#define PRETRAINED_FULL_NETWORK "../../resources/pretrained.mnist.data"
#define CONVOLUTIONAL_NETWORK "cnn.data"
#define TEST_IMAGE_FILE "../../resources/t10k-images-idx3-ubyte.gz"
#define TEST_LABEL_FILE "../../resources/t10k-labels-idx1-ubyte.gz"
#define TEST_IMAGE_SIZE 28
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE * TEST_IMAGE_SIZE
#define BP_DELTAS_CHECKS 8
#define BP_CONV_DELTAS_CHECKS 4
#define CONV_L1F0_BIAS 0.02630446809718423

#define getNetwork(tc) (NeuralNetwork*)(tc->data[0])
#define getTestData(tc) (double*)(tc->data[1])

TestCase * fullNetworkTests;
TestCase * convNetworkTests;

void genericSetup (void* test_case);
void genericTeardown (void* test_case);
int testFullLoad(void* test_case, void* test);
int testFullFeedforward(void* test_case, void* test);
int testFullAccuracy(void* tc, void* t);
int testFullBackprop(void* test_case, void* test);

int testConvLoad(void* test_case, void* test);
int testConvFeedforward(void* test_case, void* test);
//int testConvAccuracy(void* tc, void* t);
int testConvBackprop(void* test_case, void* test);

int testlen = 0;

double fullNetworkFeedForwardResults[] = {
    0.000000,
    0.000000,
    0.000003,
    0.135474,
    0.000000,
    0.000000,
    0.000000,
    0.999762,
    0.000000,
    0.000000
};

double convNetworkFeedForwardResults[] = {
    0.961797,
    0.000264,
    0.004449,
    0.054507,
    0.456677,
    0.989324,
    0.005734,
    0.251899,
    0.945555,
    0.003376
};


// Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2
double backpropDeltas[8][7] = {
    {1.0, 6.0, 0.00000302, 0.0, 202.0, 0.00000000, 0.00000099},
    {1.0, 10.0, 0.00001238, 0.0, 202.0, 0.00000000, 0.00000408},
    {1.0, 28.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {1.0, 29.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {2.0, 0.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 1.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 8.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 9.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000}
};

// Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2
double backpropConvDeltas[8][8] = {
    {1.0, 0.0, 0.74930726, 0.0, 1.0, 0.01887213, 0.00558867},
    {1.0, 1.0, -0.12929553, 0.0, 1.0, -0.00580211, -0.00211678},
    {3.0, 6.0, 0.00000055, 0.0, 25.0, 0.00000028, 0.00000035},
    {4.0, 0.0, 0.03533965, 0.0, 2.0, 0.00000000, 0.03533965}
};

int main(int argc, char** argv) {
    
    fullNetworkTests = createTest("Fully Connected Network");
    fullNetworkTests->setup = genericSetup;
    fullNetworkTests->teardown = genericTeardown;
    addTest(fullNetworkTests, "Load", NULL, testFullLoad);
    addTest(fullNetworkTests, "Feedforward", NULL, testFullFeedforward);
    addTest(fullNetworkTests, "Accuracy", NULL, testFullAccuracy);
    addTest(fullNetworkTests, "Backprop", NULL, testFullBackprop);
    performTests(fullNetworkTests);
    deleteTest(fullNetworkTests);
    
    convNetworkTests = createTest("Convolutional Network");
    convNetworkTests->setup = genericSetup;
    convNetworkTests->teardown = genericTeardown;
    addTest(convNetworkTests, "Load", NULL, testConvLoad);
    addTest(convNetworkTests, "Feedforward", NULL, testConvFeedforward);
    addTest(convNetworkTests, "Backprop", NULL, testConvBackprop);
    performTests(convNetworkTests);
    deleteTest(convNetworkTests);
    return 0;
    
}

void genericSetup (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    NeuralNetwork * network = createNetwork();
    test_case->data = malloc(2 * sizeof(void*));
    test_case->data[0] = network;
    double * test_data = NULL;
    testlen = loadMNISTData(TEST_DATA, TEST_IMAGE_FILE, TEST_LABEL_FILE,
                            &test_data);
    test_case->data[1] = test_data;
}

void genericTeardown (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    NeuralNetwork * network = getNetwork(test_case);
    if (network != NULL) deleteNetwork(network);
    double * test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
}

int testFullLoad(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    NeuralNetwork * network = getNetwork(test_case);
    return loadNetwork(network, PRETRAINED_FULL_NETWORK);
}

int testFullFeedforward(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    NeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    feedforward(network, test_data);

    Layer * output = network->layers[network->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        Neuron * n = output->neurons[i];
        double a = n->activation;
        double expected = fullNetworkFeedForwardResults[i];
        //printf("Layer[%d]->neuron[%d]: %.15e == exp: %.15e\n", network->size - 1, i, a, expected);
        a = round(a * 1000000.0) / 1000000.0;
        expected = round(expected * 1000000.0) / 1000000.0;
        if (a != expected) {
            res = 0;
            test->error_message = malloc(255 * sizeof(char));
            sprintf(test->error_message, "Output[%d]-> %lf != %lf", i, a,
                    expected);
            break;
        }
    }
    return res;
}

int testFullAccuracy(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * testobj = (Test*) t;
    NeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    double accuracy = test(network, test_data, testlen);
    accuracy = round(0.95 * 100.0);
    int ok = (accuracy == 95.0);
    if (!ok) {
        testobj->error_message = malloc(255 * sizeof(char));
        sprintf(testobj->error_message, "Accuracy %lf != from expected (%lf)",
                accuracy, 95.0);
    }
    return ok;
}

int testFullBackprop(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * testobj = (Test*) t;
    testobj->error_message = malloc(255 * sizeof(char));
    NeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    double * x = test_data;
    double * y = test_data + input_size;
    Delta ** deltas = backprop(network, x, y);
    int ok = 1, i;
    for (i = 0; i < BP_DELTAS_CHECKS; i++) {
        int lidx = (int) (backpropDeltas[i][0]);
        int nidx = (int) (backpropDeltas[i][1]);
        double bias = backpropDeltas[i][2];
        int widx1 = (int) (backpropDeltas[i][3]);
        int widx2 = (int) (backpropDeltas[i][4]);
        double w1 = backpropDeltas[i][5];
        double w2 = backpropDeltas[i][6];
        Delta * dl = deltas[lidx - 1];
        Delta * d = &(dl[nidx]);
        double val = round(d->bias * 100000000.0) / 100000000.0;
        ok = (val == bias);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] bias %lf != from expected (%lf)",
                    lidx - 1, nidx, val, bias);
            break;
        }
        val = round(d->weights[widx1] * 100000000.0) / 100000000.0;
        ok = (val == w1);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] weight[%d] %lf != from expected (%lf)",
                    lidx - 1, nidx, widx1, val, w1);
            break;
        }
        val = round(d->weights[widx2] * 100000000.0) / 100000000.0;
        ok = (val == w2);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] weight[%d] %lf != from expected (%lf)",
                    lidx - 1, nidx, widx2, val, w2);
            break;
        }
    }
    deleteDeltas(deltas, network);
    return ok;
}

int testConvLoad(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    NeuralNetwork * network = getNetwork(test_case);
    int loaded = loadNetwork(network, CONVOLUTIONAL_NETWORK);
    if (!loaded) {
        test->error_message = malloc(255 * sizeof(char));
        sprintf(test->error_message, "Failed to load %s\n",
                CONVOLUTIONAL_NETWORK);
        return 0;
    }
    Layer * layer = network->layers[1];
    ConvolutionalSharedParams * shared;
    shared = (ConvolutionalSharedParams *) layer->extra;
    double bias = shared->biases[0];
    bias = round(bias * 1000000.0) / 1000000.0;
    double expected = CONV_L1F0_BIAS;
    expected = round(expected * 1000000.0) / 1000000.0;
    int ok = (expected == bias);
    if (!ok) {
        test->error_message = malloc(255 * sizeof(char));
        sprintf(test->error_message,
                "Layer[1]->bias[0] %lf != from bias loaded from data %lf\n",
                bias, expected);
    }
    return ok;
}

int testConvFeedforward(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    NeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    feedforward(network, test_data);
    
    Layer * output = network->layers[network->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        Neuron * n = output->neurons[i];
        double a = n->activation;
        double expected = convNetworkFeedForwardResults[i];
        //printf("Layer[%d]->neuron[%d]: %.15e == exp: %.15e\n", network->size - 1, i, a, expected);
        a = round(a * 1000000.0) / 1000000.0;
        expected = round(expected * 1000000.0) / 1000000.0;
        if (a != expected) {
            res = 0;
            test->error_message = malloc(255 * sizeof(char));
            sprintf(test->error_message, "Output[%d]-> %lf != %lf", i, a,
                    expected);
            break;
        }
    }
    return res;
}

int testConvBackprop(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * testobj = (Test*) t;
    testobj->error_message = malloc(255 * sizeof(char));
    NeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    double * x = test_data;
    double * y = test_data + input_size;
    Delta ** deltas = backprop(network, x, y);
    int ok = 1, i;
    for (i = 0; i < BP_CONV_DELTAS_CHECKS; i++) {
        int lidx = (int) (backpropConvDeltas[i][0]);
        int nidx = (int) (backpropConvDeltas[i][1]);
        double bias = backpropConvDeltas[i][2];
        int widx1 = (int) (backpropConvDeltas[i][3]);
        int widx2 = (int) (backpropConvDeltas[i][4]);
        double w1 = backpropConvDeltas[i][5];
        double w2 = backpropConvDeltas[i][6];
        Delta * dl = deltas[lidx - 1];
        if (dl == NULL) continue;
        Delta * d = &(dl[nidx]);
        double val = round(d->bias * 100000000.0) / 100000000.0;
        ok = (val == bias);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] bias %.8lf != from expected (%.8lf)",
                    lidx - 1, nidx, val, bias);
            break;
        }
        val = round(d->weights[widx1] * 100000000.0) / 100000000.0;
        ok = (val == w1);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] weight[%d] %.8lf != from expected (%.8lf)",
                    lidx - 1, nidx, widx1, val, w1);
            break;
        }
        val = round(d->weights[widx2] * 100000000.0) / 100000000.0;
        ok = (val == w2);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Delta[%d][%d] weight[%d] %.8lf != from expected (%.8lf)",
                    lidx - 1, nidx, widx2, val, w2);
            break;
        }
    }
    deleteDeltas(deltas, network);
    return ok;
}

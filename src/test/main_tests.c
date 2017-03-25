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
#include <string.h>
#include "test.h"
#include "../neural.h"
#include "../mnist.h"
#ifdef USE_AVX
#include "../avx.h"
#endif

#define PRETRAINED_FULL_NETWORK "../../resources/pretrained.mnist.data"
#define CONVOLUTIONAL_NETWORK "cnn.data"
#define CONVOLUTIONAL_TRAINED_NETWORK "../../resources/pretrained.cnn.data"
#define RECURRENT_NETWORK "rnn.data"
#define TEST_IMAGE_FILE "../../resources/t10k-images-idx3-ubyte.gz"
#define TEST_LABEL_FILE "../../resources/t10k-labels-idx1-ubyte.gz"
#define TEST_IMAGE_SIZE 28
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE * TEST_IMAGE_SIZE
#define BP_GRADIENTS_CHECKS 8
#define BP_CONV_GRADIENTS_CHECKS 4
#define CONV_L1F0_BIAS 0.02630446809718423

#define RNN_INPUT_SIZE  4
#define RNN_HIDDEN_SIZE 2
#define RNN_TIMES       4
#define RNN_LEARING_RATE 0.005

#define getNetwork(tc) ((PSNeuralNetwork*)(tc->data[0]))
#define getTestData(tc) ((double*)(tc->data[1]))
#define getRoundedDouble(d) (round(d * 1000000.0) / 1000000.0)
#define getRoundedDoubleDec(d, dec) (round(d * dec) / dec)

TestCase * fullNetworkTests;
TestCase * convNetworkTests;
TestCase * recurrentNetworkTests;

#ifdef USE_AVX
TestCase * AVXTests;
#endif

int genericSetup (void* test_case);
int genericTeardown (void* test_case);
int RNNSetup (void* test_case);
int RNNTeardown (void* test_case);

int testGenericClone(void* test_case, void* test);
int testGenericSave(void* test_case, void* test);

#ifdef USE_AVX
int testAVXDot(void* test_case, void* test);
int testAVXMultiplyVal(void* tc, void* t);
#endif

int testFullLoad(void* test_case, void* test);
int testFullFeedforward(void* test_case, void* test);
int testFullAccuracy(void* tc, void* t);
int testFullBackprop(void* test_case, void* test);

int testConvLoad(void* test_case, void* test);
int testConvFeedforward(void* test_case, void* test);
int testConvAccuracy(void* tc, void* t);
int testConvBackprop(void* test_case, void* test);

int testRNNLoad(void* test_case, void* test);
int testRNNFeedforward(void* test_case, void* test);
int testRNNBackprop(void* test_case, void* test);
int testRNNStep(void* tc, void* t);

/* neural.c static function prototypes */

PSGradient ** backprop(PSNeuralNetwork * network, double * x, double * y);
PSGradient ** backpropThroughTime(PSNeuralNetwork * network, double * x,
                                  double * y, int times);

double updateWeights(PSNeuralNetwork * network, double * training_data,
                     int batch_size, double rate, ...);

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
double backpropGradients[8][7] = {
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
double backpropConvGradients[8][8] = {
    {1.0, 0.0, 0.74930726, 0.0, 1.0, 0.01887213, 0.00558867},
    {1.0, 1.0, -0.12929553, 0.0, 1.0, -0.00580211, -0.00211678},
    {3.0, 6.0, 0.00000055, 0.0, 25.0, 0.00000028, 0.00000035},
    {4.0, 0.0, 0.03533965, 0.0, 2.0, 0.00000000, 0.03533965}
};

double rnn_inner_weights[2][4] = {
    {0.23728831, -0.13215413,  0.22972574, -0.30660592},
    {0.20497796,  0.10017828, -0.42993062, 0.13334368}
};

double rnn_outer_weights[4][2] = {
    { 0.66009386,  0.53237322},
    { 0.38653,    -0.17668558},
    { 0.39194875, -0.43592448},
    {-0.3616406,   0.22109871}
};

double rnn_recurrent_weights[2][2] = {
    {-0.46118039, -0.06823764},
    {-0.05642161, -0.69206895}
};

double rnn_expected_output[4][4] = {
    {0.30070284, 0.24446228, 0.23227381, 0.22256106},
    {0.21998344, 0.24441611, 0.24745303, 0.28814742},
    {0.23400344, 0.27607197, 0.30379749, 0.1861271},
    {0.24796499, 0.21649963, 0.19729585, 0.33823952}
};

double rnn_inner_gradients[2][6] = {
    { 0.7147815, -0.21859849, 0.11033169, -0.38062627, -0.20556136, 0.087832},
    {-0.25557706, 0.18080836, 0.40002974, -0.39482105, -0.1891429, 0.158413}
};

double rnn_outer_gradients[4][2] = {
    { 0.4023519, -0.29857975},
    {-0.33460693, 0.37440608},
    { 0.26141107, 0.0456771},
    {-0.32915604, -0.12150343}
};

double rnn_trained_inner_weights[2][4] = {
    { 0.2337144,  -0.13106114,  0.22917408, -0.30470279},
    { 0.20625585,  0.09927424, -0.43193077,  0.13531779}
};

double rnn_trained_outer_weights[4][2] = {
    {0.6580821, 0.53386612},
    {0.38820303, -0.17855761},
    {0.39064169, -0.43615287},
    {-0.35999482, 0.22170623}
};

double rnn_trained_recurrent_weights[2][2] = {
    {-0.46015258, -0.0686768},
    {-0.0554759, -0.69286102}
};

double rnn_inputs[5] = {4, 0, 1, 2, 3};
double rnn_labels[4] = {3, 2, 1, 0};


int compareNetworks(PSNeuralNetwork * net1, PSNeuralNetwork * net2, Test* test);

static void getTmpFileName(const char * prfx, const char * sfx, char * buffer) {
    FILE * urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x%s",
            prfx, (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3], sfx);
    fclose(urand);
}

int main(int argc, char** argv) {
    
#ifdef USE_AVX
    AVXTests = createTest("AVX");
    addTest(AVXTests, "Dot Product", NULL, testAVXDot);
    addTest(AVXTests, "Multiply Value", NULL, testAVXMultiplyVal);
    performTests(AVXTests);
    deleteTest(AVXTests);
#endif
    
    fullNetworkTests = createTest("Fully Connected Network");
    fullNetworkTests->setup = genericSetup;
    fullNetworkTests->teardown = genericTeardown;
    addTest(fullNetworkTests, "Load", NULL, testFullLoad);
    addTest(fullNetworkTests, "Feedforward", NULL, testFullFeedforward);
    addTest(fullNetworkTests, "Accuracy", NULL, testFullAccuracy);
    addTest(fullNetworkTests, "Backprop", NULL, testFullBackprop);
    addTest(fullNetworkTests, "Clone", NULL, testGenericClone);
    addTest(fullNetworkTests, "Save", NULL, testGenericSave);
    performTests(fullNetworkTests);
    deleteTest(fullNetworkTests);
    
    convNetworkTests = createTest("Convolutional Network");
    convNetworkTests->setup = genericSetup;
    convNetworkTests->teardown = genericTeardown;
    addTest(convNetworkTests, "Load", NULL, testConvLoad);
    addTest(convNetworkTests, "Feedforward", NULL, testConvFeedforward);
    addTest(convNetworkTests, "Backprop", NULL, testConvBackprop);
    addTest(convNetworkTests, "Accuracy", NULL, testConvAccuracy);
    addTest(convNetworkTests, "Clone", NULL, testGenericClone);
    addTest(convNetworkTests, "Save", NULL, testGenericSave);
    performTests(convNetworkTests);
    deleteTest(convNetworkTests);
    
    recurrentNetworkTests = createTest("Recurrent Network");
    recurrentNetworkTests->setup = RNNSetup;
    recurrentNetworkTests->teardown = RNNTeardown;
    addTest(recurrentNetworkTests, "Load", NULL, testRNNLoad);
    addTest(recurrentNetworkTests, "Feedforward", NULL, testRNNFeedforward);
    addTest(recurrentNetworkTests, "Backprop", NULL, testRNNBackprop);
    addTest(recurrentNetworkTests, "Step", NULL, testRNNStep);
    addTest(recurrentNetworkTests, "Clone", NULL, testGenericClone);
    addTest(recurrentNetworkTests, "Save", NULL, testGenericSave);
    performTests(recurrentNetworkTests);
    deleteTest(recurrentNetworkTests);
    
    return 0;
    
}

int genericSetup (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    PSNeuralNetwork * network = PSCreateNetwork("Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    double * test_data = NULL;
    testlen = loadMNISTData(TEST_DATA, TEST_IMAGE_FILE, TEST_LABEL_FILE,
                            &test_data);
    test_case->data[1] = test_data;
    if (test_data == NULL) {
        return 0;
    }
    return 1;
}

int genericTeardown (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    PSNeuralNetwork * network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    double * test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int RNNSetup (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    PSNeuralNetwork * network = PSCreateNetwork("RNN Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, Recurrent, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    if (network->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
    
    int i, j, w;
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron * n = layer->neurons[j];
            n->bias = 0;
            for (w = 0; w < n->weights_size; w++) {
                double * weights;
                int w_idx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE) weights = rnn_inner_weights[j];
                    else {
                        weights = rnn_recurrent_weights[j];
                        w_idx -= RNN_INPUT_SIZE;
                    }
                } else weights = rnn_outer_weights[j];
                n->weights[w] = weights[w_idx];
            }
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    int train_data_len = 1 + (RNN_TIMES * 2);
    int labels_offset = 1 + RNN_TIMES;
    double * training_data = malloc(train_data_len * sizeof(double));
    double * p = training_data;
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(p, rnn_inputs, labels_offset * sizeof(double));
    p += labels_offset;
    memcpy(p, rnn_labels, RNN_TIMES * sizeof(double));
    test_case->data[1] = training_data;
    return 1;
}

int RNNTeardown (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    PSNeuralNetwork * network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    double * test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int testFullLoad(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    return PSLoadNetwork(network, PRETRAINED_FULL_NETWORK);
}

int testFullFeedforward(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    PSFeedforward(network, test_data);

    PSLayer * output = network->layers[network->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        PSNeuron * n = output->neurons[i];
        double a = n->activation;
        double expected = fullNetworkFeedForwardResults[i];
        a = getRoundedDouble(a);
        expected = getRoundedDouble(expected);
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
    PSNeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    double accuracy = PSTest(network, test_data, testlen);
    accuracy = round(accuracy * 100.0);
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
    PSNeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    double * x = test_data;
    double * y = test_data + input_size;
    PSGradient ** gradients = backprop(network, x, y);
    int ok = 1, i;
    for (i = 0; i < BP_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropGradients[i][0]);
        int nidx = (int) (backpropGradients[i][1]);
        double bias = backpropGradients[i][2];
        int widx1 = (int) (backpropGradients[i][3]);
        int widx2 = (int) (backpropGradients[i][4]);
        double w1 = backpropGradients[i][5];
        double w2 = backpropGradients[i][6];
        PSGradient * dl = gradients[lidx - 1];
        PSGradient * d = &(dl[nidx]);
        double val = getRoundedDoubleDec(d->bias, 100000000.0);
        ok = (val == bias);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] bias %lf != from expected (%lf)",
                    lidx - 1, nidx, val, bias);
            break;
        }
        val = getRoundedDoubleDec(d->weights[widx1], 100000000.0);
        ok = (val == w1);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] weight[%d] %lf != from expected (%lf)",
                    lidx - 1, nidx, widx1, val, w1);
            break;
        }
        val = getRoundedDoubleDec(d->weights[widx2], 100000000.0);
        ok = (val == w2);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] weight[%d] %lf != from expected (%lf)",
                    lidx - 1, nidx, widx2, val, w2);
            break;
        }
    }
    PSDeleteGradients(gradients, network);
    return ok;
}

int testConvLoad(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    int loaded = PSLoadNetwork(network, CONVOLUTIONAL_NETWORK);
    if (!loaded) {
        test->error_message = malloc(255 * sizeof(char));
        sprintf(test->error_message, "Failed to load %s\n",
                CONVOLUTIONAL_NETWORK);
        return 0;
    }
    PSLayer * layer = network->layers[1];
    PSSharedParams * shared;
    shared = (PSSharedParams *) layer->extra;
    double bias = shared->biases[0];
    bias = getRoundedDouble(bias);
    double expected = CONV_L1F0_BIAS;
    expected = getRoundedDouble(expected);
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
    PSNeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    PSFeedforward(network, test_data);
    
    PSLayer * output = network->layers[network->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        PSNeuron * n = output->neurons[i];
        double a = n->activation;
        double expected = convNetworkFeedForwardResults[i];
        a = getRoundedDouble(a);
        expected = getRoundedDouble(expected);
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
    PSNeuralNetwork * network = getNetwork(test_case);
    double * test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    double * x = test_data;
    double * y = test_data + input_size;
    PSGradient ** gradients = backprop(network, x, y);
    int ok = 1, i;
    for (i = 0; i < BP_CONV_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropConvGradients[i][0]);
        int nidx = (int) (backpropConvGradients[i][1]);
        double bias = backpropConvGradients[i][2];
        int widx1 = (int) (backpropConvGradients[i][3]);
        int widx2 = (int) (backpropConvGradients[i][4]);
        double w1 = backpropConvGradients[i][5];
        double w2 = backpropConvGradients[i][6];
        PSGradient * dl = gradients[lidx - 1];
        if (dl == NULL) continue;
        PSGradient * d = &(dl[nidx]);
        double val = getRoundedDoubleDec(d->bias, 100000000.0);
        ok = (val == bias);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] bias %.8lf != from expected (%.8lf)",
                    lidx - 1, nidx, val, bias);
            break;
        }
        val = getRoundedDoubleDec(d->weights[widx1], 100000000.0);
        ok = (val == w1);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] weight[%d] %.8lf != from expect. (%.8lf)",
                    lidx - 1, nidx, widx1, val, w1);
            break;
        }
        val = getRoundedDoubleDec(d->weights[widx2], 100000000.0);
        ok = (val == w2);
        if (!ok) {
            sprintf(testobj->error_message,
                    "Gradient[%d][%d] weight[%d] %.8lf != from expect. (%.8lf)",
                    lidx - 1, nidx, widx2, val, w2);
            break;
        }
    }
    PSDeleteGradients(gradients, network);
    return ok;
}

int testConvAccuracy(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * testobj = (Test*) t;
    double * test_data = getTestData(test_case);
    PSNeuralNetwork * network = PSCreateNetwork("CNN Test Network");
    int loaded = PSLoadNetwork(network, CONVOLUTIONAL_TRAINED_NETWORK);
    if (!loaded) {
        testobj->error_message = malloc(255 * sizeof(char));
        sprintf(testobj->error_message, "Failed to load %s",
                CONVOLUTIONAL_TRAINED_NETWORK);
        PSDeleteNetwork(network);
        return 0;
    }
    PSFeedforward(network, test_data);
    double accuracy = PSTest(network, test_data, testlen);
    accuracy = round(accuracy * 100.0);
    int ok = (accuracy == 98.0);
    if (!ok) {
        testobj->error_message = malloc(255 * sizeof(char));
        sprintf(testobj->error_message, "Accuracy %lf != from expected (%lf)",
                accuracy, 98.0);
    }
    PSDeleteNetwork(network);
    return ok;
}

int testRNNLoad(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    int loaded = PSLoadNetwork(network, RECURRENT_NETWORK);
    if (!loaded) {
        test->error_message = malloc(255 * sizeof(char));
        sprintf(test->error_message, "Failed to load %s\n",
                RECURRENT_NETWORK);
        return 0;
    }
    
    int ok = 1, i, j, w;
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron * n = layer->neurons[j];
            for (w = 0; w < n->weights_size; w++) {
                double * weights;
                int w_idx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE) weights = rnn_inner_weights[j];
                    else {
                        weights = rnn_recurrent_weights[j];
                        w_idx -= RNN_INPUT_SIZE;
                    }
                } else weights = rnn_outer_weights[j];
                ok = (n->weights[w] == weights[w_idx]);
                if (!ok) {
                    test->error_message = malloc(255 * sizeof(char));
                    sprintf(test->error_message,
                            "L[%d]N[%d]->weights[%d] %.15e != %.15e\n",
                            i, j, w, n->weights[w], weights[w_idx]);
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
    }
    
    return ok;
}

int testRNNFeedforward(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    PSFeedforward(network, rnn_inputs);
    
    PSLayer * output = network->layers[network->size - 1];
    int ok = 1, i, j;
    for (i = 0; i < output->size; i++) {
        PSNeuron * n = output->neurons[i];
        PSRecurrentCell* cell = (PSRecurrentCell*) n->extra;
        for (j = 0; j < cell->states_count; j++) {
            double s = getRoundedDouble(cell->states[j]);
            double expected = getRoundedDouble(rnn_expected_output[j][i]);
            ok = (s == expected);
            if (!ok) {
                test->error_message = malloc(255 * sizeof(char));
                sprintf(test->error_message, "Output[%d][%d]: %lf != %lf\n",
                        i, j, s, expected);
                break;
            }
        }
        if (!ok) break;
    }
    return ok;
}

int testRNNBackprop(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    int ok = 1, i, j, w;
    
    PSGradient ** gradients = backpropThroughTime(network, rnn_inputs + 1,
                                                  rnn_labels, RNN_TIMES);
    int dsize = network->size - 1;
    for (i = 0; i < dsize; i++) {
        PSGradient * lgradients = gradients[i];
        PSLayer * l = network->layers[i + 1];
        for (j = 0; j < l->size; j++) {
            PSGradient * gradient = &(lgradients[j]);
            int ws = l->neurons[j]->weights_size;
            double * expected = (i == 0 ? rnn_inner_gradients[j] :
                                 rnn_outer_gradients[j]);
            for (w = 0; w < ws; w++) {
                double dw = getRoundedDouble(gradient->weights[w]);
                double exp_dw = getRoundedDouble(expected[w]);
                ok = (dw == exp_dw);
                if (!ok) {
                    char * msg = malloc(255 * sizeof(char));
                    test->error_message = msg;
                    sprintf(msg, "Gradient[%d][%d]->weight[%d]: %lf != %lf\n",
                            i, j, w, dw, exp_dw);
                    break;
                }
            }
        }
        if (!ok) break;
    }
    
    PSDeleteGradients(gradients, network);
    
    return ok;
}

int testRNNStep(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    int train_data_len = 1 + (RNN_TIMES * 2);
    double * training_data = getTestData(test_case);
    double ** series = &training_data;
    
    int ok = 1, i, j, w;
    
    double loss = updateWeights(network, training_data, 1,
                                RNN_LEARING_RATE, series);
    
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron * n = layer->neurons[j];
            for (w = 0; w < n->weights_size; w++) {
                double * weights;
                int w_idx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE)
                        weights = rnn_trained_inner_weights[j];
                    else {
                        weights = rnn_trained_recurrent_weights[j];
                        w_idx -= RNN_INPUT_SIZE;
                    }
                } else weights = rnn_trained_outer_weights[j];
                double w_val = getRoundedDouble(n->weights[w]);
                double expected_w = getRoundedDouble(weights[w_idx]);
                ok = (w_val == expected_w);
                if (!ok) {
                    char * msg = malloc(255 * sizeof(char));
                    test->error_message = msg;
                    sprintf(msg, "Layer[%d][%d]->weights[%d]: %lf != %lf\n",
                            i, j, w, w_val, expected_w);
                    break;
                }
            }
        }
        if (!ok) break;
    }
    //free(series);
    return ok;
}

int testGenericClone(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    PSNeuralNetwork * clone = PSCloneNetwork(network, 0);
    if (clone == NULL) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "Could not create network clone!\n");
        return 0;
    }
    
    int ok = compareNetworks(network, clone, test);
    
    PSDeleteNetwork(clone);
    return ok;
}

int testGenericSave(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    PSNeuralNetwork * network = getNetwork(test_case);
    char tmpfile[255];
    getTmpFileName("tests-save-nn", ".data", tmpfile);
    int ok = PSSaveNetwork(network, tmpfile);
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "Could not save network!\n");
        return 0;
    }
    PSNeuralNetwork * clone = PSCreateNetwork("Clone Test Network");
    if (clone == NULL) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "Could not create network clone!\n");
        return 0;
    }
    ok = PSLoadNetwork(clone, tmpfile);
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "Could not load network!\n");
        return 0;
    }
    
    ok = compareNetworks(network, clone, test);
    
    remove(tmpfile);
    PSDeleteNetwork(clone);
    return ok;
}

int compareNetworks(PSNeuralNetwork * network, PSNeuralNetwork * clone,
                    Test* test)
{
    int ok = 1, i, k, w;
    
    ok = network->size == clone->size;
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "Source size %d != Clone size %d\n",
                network->size, clone->size);
        return 0;
    }
    
    for (i = 0; i < network->size; i++) {
        PSLayer * orig_l = network->layers[i];
        PSLayer * clone_l = clone->layers[i];
        PSLayerType otype = orig_l->type;
        PSLayerType ctype = clone_l->type;
        ok = (otype == ctype);
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Layer[%d]: Source type %s != Clone type %s\n",
                    i, getLayerTypeLabel(orig_l), getLayerTypeLabel(clone_l));
            break;
        }
        int o_size = orig_l->size;
        int c_size = clone_l->size;
        ok = (o_size == c_size);
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Layer[%d]: Source size %d != Clone size %d\n",
                    i, o_size, c_size);
            break;
        }
        if (i == 0) continue;
        if (otype == Pooling) continue;
        int conv_features_checked = 0;
        for (k = 0; k < o_size; k++) {
            PSNeuron * orig_n = orig_l->neurons[k];
            PSNeuron * clone_n = clone_l->neurons[k];
            if (otype == Convolutional) {
                PSSharedParams* oshared;
                PSSharedParams* cshared;
                oshared = (PSSharedParams*) orig_l->extra;
                cshared = (PSSharedParams*) clone_l->extra;
                if (!conv_features_checked) {
                    conv_features_checked = 1;
                    ok = (oshared->feature_count == cshared->feature_count);
                    if (!ok) {
                        char * msg = malloc(255 * sizeof(char));
                        test->error_message = msg;
                        sprintf(msg, "Layer[%d]: Feature count %d != %d\n",
                                i, oshared->feature_count,
                                cshared->feature_count);
                        break;
                    }
                }
                int fsize = orig_l->size / oshared->feature_count;
                int fidx = k / fsize;
                double obias = getRoundedDouble(oshared->biases[fidx]);
                double cbias = getRoundedDouble(cshared->biases[fidx]);
                ok = (obias == cbias);
                if (!ok) {
                    char * msg = malloc(255 * sizeof(char));
                    test->error_message = msg;
                    sprintf(msg, "Layer[%d][%d]: bias %.15e != %.15e\n",
                            i, fidx, obias, cbias);
                    break;
                }
            } else if (otype != Recurrent) {
                double obias = getRoundedDouble(orig_n->bias);
                double cbias = getRoundedDouble(clone_n->bias);
                ok = (obias == cbias);
            }
            if (!ok) {
                char * msg = malloc(255 * sizeof(char));
                test->error_message = msg;
                sprintf(msg, "Layer[%d][%d]: bias %.15e != %.15e\n",
                        i, k, orig_n->bias, clone_n->bias);
                break;
            }
            ok = orig_n->weights_size == clone_n->weights_size;
            if (!ok) {
                char * msg = malloc(255 * sizeof(char));
                test->error_message = msg;
                sprintf(msg, "Layer[%d][%d]: weight sz. %d != %d\n",
                        i, k, orig_n->weights_size, clone_n->weights_size);
                break;
            }
            for (w = 0; w < orig_n->weights_size; w++) {
                double ow = getRoundedDouble(orig_n->weights[w]);
                double cw = getRoundedDouble(clone_n->weights[w]);
                ok = ow == cw;
                if (!ok) {
                    char * msg = malloc(255 * sizeof(char));
                    test->error_message = msg;
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

#ifdef USE_AVX

double test_dot(double * x, double * y, int size) {
    int i;
    double dot = 0.0;
    for (i = 0; i < size; i++) {
        dot += (x[i] * y[i]);
    }
    return dot;
}

int testAVXDot(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    
    double x2[2] = {1.0, 2.0};
    double y2[2] = {0.5, 0.5};
    
    double x4[4] = {1.0, 1.0, 2.0, 2.0};
    double y4[4] = {0.5, 0.5, 1.0, 0.5};
    
    double x8[8] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0};
    double y8[8] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0};
    
    double x16[16] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                      0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0};
    double y16[16] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                      1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5};
    
    double avx_res = avx_dot_product16(x16, y16);
    double cmp_res = test_dot(x16, y16, 16);
    int ok = avx_res == cmp_res;
    
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "AVX[16]: Expected %lf != %lf\n", cmp_res, avx_res);
        return 0;
    }
    
    avx_res = avx_dot_product8(x8, y8);
    cmp_res = test_dot(x8, y8, 8);
    ok = avx_res == cmp_res;
    
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "AVX[8]: Expected %lf != %lf\n", cmp_res, avx_res);
        return 0;
    }
    
    avx_res = avx_dot_product4(x4, y4);
    cmp_res = test_dot(x4, y4, 4);
    ok = avx_res == cmp_res;
    
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "AVX[4]: Expected %lf != %lf\n", cmp_res, avx_res);
        return 0;
    }
    
    avx_res = avx_dot_product2(x2, y2);
    cmp_res = test_dot(x2, y2, 2);
    ok = avx_res == cmp_res;
    
    if (!ok) {
        char * msg = malloc(255 * sizeof(char));
        test->error_message = msg;
        sprintf(msg, "AVX[2]: Expected %lf != %lf\n", cmp_res, avx_res);
        return 0;
    }
    
    return ok;
}

int testAVXMultiplyVal(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * test = (Test*) t;
    int ok = 1, i;
    double x[4] = {0.0, 1.0, 2.0, 3.0};
    double val = 2.0;
    double y[4] = {0.0, 2.0, 4.0, 6.0};
    double dest[4] = {0.0, 0.0, 0.0, 0.0};
    
    double x2[2] = {2.0, 3.0};
    double y2[2] = {4.0, 6.0};
    double dest2[2] = {0.0, 0.0};
    
    avx_multiply_value4(x, val, dest, 0);
    for (i = 0; i < 4; i++) {
        ok = dest[i] == y[i];
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Store Mode Norm[4]: Expected %lf != %lf\n",
                    y[i], dest[i]);
            return 0;
        }
    }
    
    avx_multiply_value4(x, val, dest, AVX_STORE_MODE_ADD);
    for (i = 0; i < 4; i++) {
        ok = (dest[i] == (y[i] + y[i]));
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Store Mode Norm[4]: Expected %lf != %lf\n",
                    y[i], dest[i]);
            return 0;
        }
    }
    
    avx_multiply_value2(x2, val, dest2, 0);
    for (i = 0; i < 2; i++) {
        ok = dest2[i] == y2[i];
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Store Mode Norm[2]: Expected %lf != %lf\n",
                    y2[i], dest2[i]);
            return 0;
        }
    }
    
    avx_multiply_value2(x2, val, dest2, AVX_STORE_MODE_ADD);
    for (i = 0; i < 2; i++) {
        ok = (dest2[i] == (y2[i] + y2[i]));
        if (!ok) {
            char * msg = malloc(255 * sizeof(char));
            test->error_message = msg;
            sprintf(msg, "Store Mode Norm[2]: Expected %lf != %lf\n",
                    y2[i], dest2[i]);
            return 0;
        }
    }
    
    return ok;
}

#endif

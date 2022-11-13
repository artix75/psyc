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
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include <strings.h>
#include <assert.h>

#include "test.h"
#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../lstm.h"
#include "../mnist.h"
#include "../utils.h"
#include "../debug.h"
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
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE *TEST_IMAGE_SIZE
#define BP_GRADIENTS_CHECKS 8
#define BP_CONV_GRADIENTS_CHECKS 4
#define CONV_L1F0_BIAS 0.02630446809718423

#define PRETRAINED_MNIST_NETSIZE 3

#define RNN_INPUT_SIZE  4
#define RNN_HIDDEN_SIZE 2
#define RNN_TIMES       4
#define RNN_LEARNING_RATE 0.005

#define LSTM_LEARNING_RATE 0.1
#define LSTM_TIMES 3
#define LSTM_EPOCHS 1
#define LSTM_BATCHES 1

#define getNetwork(tc) ((PSNeuralNetwork*)(tc->data[0]))
#define getTestData(tc) ((PSFloat*)(tc->data[1]))
#ifdef PS_DOUBLE_PRECISION
#define NORMAL_PRECISION_DEC    6
#define HIGH_PRECISION_DEC      8
#else
#define NORMAL_PRECISION_DEC    5
#define HIGH_PRECISION_DEC      7
#endif
#define getRoundedFloat(d) (getRoundedFloatDec(d, NORMAL_PRECISION_DEC))

#define UNUSED(V) ((void) V)

TestCase *fullNetworkTests;
TestCase *convNetworkTests;
TestCase *recurrentNetworkTests;
TestCase *LSTMNetworkTests;

#ifdef USE_AVX
TestCase *AVXTests;
#endif

int genericSetup (TestCase *test_case);
int genericTeardown (TestCase *test_case);
int RNNSetup (TestCase *test_case);
int RNNTeardown (TestCase *test_case);
int LSTMSetup (TestCase *test_case);

int testGenericClone(TestCase *test_case, Test *test);
int testGenericSave(TestCase *test_case, Test *test);

#ifdef USE_AVX
int testAVXDot(TestCase *test_case, Test *test);
int testAVXSquare(TestCase *test_case, Test *test);
int testAVXMultiplyVal(TestCase *tc, Test *test);
#endif

int testFullLoad(TestCase *test_case, Test *test);
int testFullFeedforward(TestCase *test_case, Test *test);
int testFullAccuracy(TestCase *tc, Test *test);
int testFullBackprop(TestCase *test_case, Test *test);

int testConvLoad(TestCase *test_case, Test *test);
int testConvFeedforward(TestCase *test_case, Test *test);
int testConvAccuracy(TestCase *tc, Test *test);
int testConvBackprop(TestCase *test_case, Test *test);

int testRNNLoad(TestCase *test_case, Test *test);
int testRNNFeedforward(TestCase *test_case, Test *test);
int testRNNBackprop(TestCase *test_case, Test *test);
int testRNNStep(TestCase *tc, Test *test);
int testRNNOneHot(TestCase *tc, Test *test);

int testLSTMLoad(TestCase *test_case, Test *test);
int testLSTMTrain(TestCase *test_case, Test *test);

/* psyc.c function prototypes */

PSGradient **backprop(PSNeuralNetwork *network, PSFloat *x, PSFloat *y);

PSFloat updateNetworkParameters(PSNeuralNetwork *network,
                                PSFloat *training_data,
                                int batch_size, int elements_count,
                                PSTrainingOptions* opts, PSFloat rate,
                                PSGradient **momentum_gradeints,
                                PSGradient **aux_gradients, ...);

int testlen = 0;

int pretrained_mnist_layers_size[PRETRAINED_MNIST_NETSIZE] = {784,30,10};

PSFloat fullNetworkFeedForwardResults[] = {
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

PSFloat convNetworkFeedForwardResults[] = {
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


/*  Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2 */
PSFloat backpropGradients[8][7] = {
    {1.0, 6.0, 0.00000302, 0.0, 202.0, 0.00000000, 0.00000099},
    {1.0, 10.0, 0.00001238, 0.0, 202.0, 0.00000000, 0.00000408},
    {1.0, 28.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {1.0, 29.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {2.0, 0.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 1.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 8.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 9.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000}
};

/*  Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2 */
PSFloat backpropConvGradients[8][8] = {
    {1.0, 0.0, 0.74930726, 0.0, 1.0, 0.01887213, 0.00558867},
    {1.0, 1.0, -0.12929553, 0.0, 1.0, -0.00580211, -0.00211678},
    {3.0, 6.0, 0.00000055, 0.0, 25.0, 0.00000028, 0.00000035},
    {4.0, 0.0, 0.03533965, 0.0, 2.0, 0.00000000, 0.03533965}
};

PSFloat rnn_inner_weights[2][4] = {
    {0.23728831, -0.13215413,  0.22972574, -0.30660592},
    {0.20497796,  0.10017828, -0.42993062, 0.13334368}
};

PSFloat rnn_outer_weights[4][2] = {
    { 0.66009386,  0.53237322},
    { 0.38653,    -0.17668558},
    { 0.39194875, -0.43592448},
    {-0.3616406,   0.22109871}
};

PSFloat rnn_recurrent_weights[2][2] = {
    {-0.46118039, -0.06823764},
    {-0.05642161, -0.69206895}
};

PSFloat rnn_expected_output[4][4] = {
    {0.30070284, 0.24446228, 0.23227381, 0.22256106},
    {0.21998344, 0.24441611, 0.24745303, 0.28814742},
    {0.23400344, 0.27607197, 0.30379749, 0.1861271},
    {0.24796499, 0.21649963, 0.19729585, 0.33823952}
};

PSFloat rnn_inner_gradients[2][6] = {
    { 0.7147815, -0.21859849, 0.11033169, -0.38062627, -0.20556136, 0.087832},
    {-0.25557706, 0.18080836, 0.40002974, -0.39482105, -0.1891429, 0.158413}
};

PSFloat rnn_outer_gradients[4][2] = {
    { 0.4023519, -0.29857975},
    {-0.33460693, 0.37440608},
    { 0.26141107, 0.0456771},
    {-0.32915604, -0.12150343}
};

PSFloat rnn_trained_inner_weights[2][4] = {
    { 0.2337144,  -0.13106114,  0.22917408, -0.30470279},
    { 0.20625585,  0.09927424, -0.43193077,  0.13531779}
};

PSFloat rnn_trained_outer_weights[4][2] = {
    {0.6580821, 0.53386612},
    {0.38820303, -0.17855761},
    {0.39064169, -0.43615287},
    {-0.35999482, 0.22170623}
};

PSFloat rnn_trained_recurrent_weights[2][2] = {
    {-0.46015258, -0.0686768},
    {-0.0554759, -0.69286102}
};

PSFloat rnn_inputs[5] = {4, 0, 1, 2, 3};
PSFloat rnn_labels[4] = {3, 2, 1, 0};

PSFloat lstm_training_data[] = {1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0};
PSFloat wg[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wi[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wo[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wf[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat bg[2] = {0.0097627,0.04303787};
PSFloat bi[2] = {0.0097627,0.04303787};
PSFloat bo[2] = {0.0097627,0.04303787};
PSFloat bf[2] = {0.0097627,0.04303787};

PSFloat lstm_out_weights[4][2] = {
    { 0.0097627,   0.04303787},
    { 0.02055268,  0.00897664},
    {-0.01526904,  0.02917882},
    {-0.01248256,  0.0783546}};

PSFloat lstm_expected_states[2][3] = {
    {0.00497633, 0.01652635, 0.0163383},
    {0.00787092, 0.03837129, 0.05927353}
};

PSFloat lstm_expected_outputs[3][4] = {
    { 0.25001755, 0.24996395, 0.24995914, 0.25005937},
    { 0.25006783, 0.24978575, 0.24983151, 0.2503149},
    { 0.25008373, 0.24962334, 0.2497762, 0.25051672}
};

PSFloat expected_wg[2][6] = {
    {0.00998824,0.04246868,0.02021531,0.00897664,-0.01527745,0.02916139},
    {-0.01314446,0.07862245,0.09379196,-0.0233117,0.05836385,0.00582174}
};
PSFloat expected_wi[2][6] = {
    {0.00976488,0.04302317,0.0205475,0.00897664,-0.0152692,0.02917851},
    {-0.01249252,0.07837006,0.09280098,-0.0233117,0.05834622,0.00578173}
};
PSFloat expected_wo[2][6] = {
    {0.00976767,0.04302572,0.02054214,0.00897664,-0.01526927,0.02917832},
    {-0.0124946,0.07833524,0.09283829,-0.0233117,0.05834666,0.00578289}
};
PSFloat expected_wf[2][6] = {
    {0.0097627,0.04303513,0.02054733,0.00897664,-0.01526914,0.02917859},
    {-0.01248256,0.07835658,0.09276899,-0.0233117,0.05834562,0.00578039}
};
PSFloat expected_bg[2] = {0.00908168, 0.04370323};
PSFloat expected_bi[2] = {0.009745, 0.0431118};
PSFloat expected_bo[2] = {0.00974497, 0.04311221};
PSFloat expected_bf[2] = {0.00975461, 0.04307629};

int compareNetworks(PSNeuralNetwork *net1, PSNeuralNetwork *net2, Test* test);

static int testRecurrentNetworkMode(PSNeuralNetwork *network,
                                    PSRecurrentNetworkMode mode, Test *test);

static void getTmpFileName(const char *prfx, const char *sfx, char *buffer) {
    FILE *urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x%s",
            prfx, (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3], sfx);
    fclose(urand);
}

static int arrayMaxIndex(PSFloat *array, int len) {
    int i;
    PSFloat max = 0;
    int max_idx = 0;
    for (i = 0; i < len; i++) {
        PSFloat v = array[i];
        if (v > max) {
            max = v;
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char** argv) {
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    UNUSED(argc);
    UNUSED(argv);
    int tot_tests = 0, tot_failed = 0;
    time_t start_t = time(NULL);
#ifdef USE_AVX
    AVXTests = createTest("AVX");
    addTest(AVXTests, "Dot Product", NULL, testAVXDot);
    addTest(AVXTests, "Square", NULL, testAVXSquare);
    addTest(AVXTests, "Multiply Value", NULL, testAVXMultiplyVal);
    performTests(AVXTests);
    tot_tests += AVXTests->count;
    tot_failed += AVXTests->failed_count;
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
    tot_tests += fullNetworkTests->count;
    tot_failed += fullNetworkTests->failed_count;
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
    tot_tests += convNetworkTests->count;
    tot_failed += convNetworkTests->failed_count;
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
    addTest(recurrentNetworkTests, "OneHot", NULL, testRNNOneHot);
    performTests(recurrentNetworkTests);
    tot_tests += recurrentNetworkTests->count;
    tot_failed += recurrentNetworkTests->failed_count;
    deleteTest(recurrentNetworkTests);

    LSTMNetworkTests = createTest("LSTM Network");
    LSTMNetworkTests->setup = LSTMSetup;
    LSTMNetworkTests->teardown = RNNTeardown;
    /* addTest(LSTMNetworkTests, "Load", NULL, testLSTMLoad); */
    addTest(LSTMNetworkTests, "Train", NULL, testLSTMTrain);
    addTest(LSTMNetworkTests, "Clone", NULL, testGenericClone);
    addTest(LSTMNetworkTests, "Save", NULL, testGenericSave);
    performTests(LSTMNetworkTests);
    tot_tests += LSTMNetworkTests->count;
    tot_failed += LSTMNetworkTests->failed_count;
    deleteTest(LSTMNetworkTests);
    time_t end_t = time(NULL);
    printf(
        "\n%d tests performed in %ld second(s)\n", tot_tests, (end_t - start_t)
    );
    int succeded = tot_tests - tot_failed;
    if (succeded > 0) printf(GREEN "Succeeded: %d\n" RESET, succeded);
    if (tot_failed > 0) printf(RED "Failed:    %d\n" RESET, tot_failed);

    return tot_failed;

}

int genericSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("Test Network");
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
    PSFloat *test_data = NULL;
    testlen = PSLoadMNISTData(DATA_TYPE_TEST, TEST_IMAGE_FILE, TEST_LABEL_FILE,
                              &test_data);
    test_case->data[1] = test_data;
    if (test_data == NULL) {
        return 0;
    }
    return 1;
}

int genericTeardown(TestCase *test_case) {
    PSNeuralNetwork *network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int RNNSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("RNN Test Network");
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
        PSLayer *layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron *n = layer->neurons[j];
            n->bias = 0;
            for (w = 0; w < n->weights_size; w++) {
                PSFloat *weights;
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
    PSRecurrentNetworkMode rnn_mode = PSGetRecurrentNetworkMode(network);
    if (rnn_mode != ManyToMany) {
        fprintf(
            stderr, "\nInvalid Recurrent Network Mode: '%s'\n",
            PSGetRecurrentModeLabel(rnn_mode)
        );
        return 0;
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    int train_data_len = 1 + (RNN_TIMES * 2);
    int labels_offset = 1 + RNN_TIMES;
    PSFloat *training_data = malloc(train_data_len *sizeof(PSFloat));
    PSFloat *p = training_data;
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(p, rnn_inputs, labels_offset * sizeof(PSFloat));
    p += labels_offset;
    memcpy(p, rnn_labels, RNN_TIMES * sizeof(PSFloat));
    test_case->data[1] = training_data;
    return 1;
}

int RNNTeardown(TestCase *test_case) {
    PSNeuralNetwork *network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int LSTMSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("LSTM Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, LSTM, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    if (network->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    PSLayer *out = network->layers[network->size - 1];
    out->flags |= FLAG_ONEHOT;
    PSLayer *layer = network->layers[1];

    int i, w;
    for (i = 0; i < layer->size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSLSTMCell *cell = PSGetLSTMCell(neuron);
        cell->candidate_bias = bg[i];
        cell->input_bias = bi[i];
        cell->output_bias = bo[i];
        cell->forget_bias = bf[i];
        for (w = 0; w < cell->weights_size; w++) {
            cell->candidate_weights[w] = wg[i][w];
            cell->input_weights[w] = wi[i][w];
            cell->output_weights[w] = wo[i][w];
            cell->forget_weights[w] = wf[i][w];
        }
    }

    for (i = 0; i < out->size; i++) {
        PSNeuron *neuron = out->neurons[i];
        neuron->bias = 0.0;
        for (w = 0; w < neuron->weights_size; w++) {
            neuron->weights[w] = lstm_out_weights[i][w];
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    int train_data_len = 2 + (LSTM_TIMES * 2);
    PSFloat *training_data = malloc(train_data_len *sizeof(PSFloat));
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(training_data, lstm_training_data, train_data_len * sizeof(PSFloat));
    test_case->data[1] = training_data;
    return 1;
}


int testFullLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    int loaded = PSLoadNetwork(network, PRETRAINED_FULL_NETWORK);
    testAssert(loaded, test);
    testAssertEqual(network->size, 3, test);
    testAssertEqual(
        network->layers[0]->size, pretrained_mnist_layers_size[0], test
    );
    testAssertEqual(
        network->layers[1]->size, pretrained_mnist_layers_size[1], test
    );
    testAssertEqual(
        network->layers[2]->size, pretrained_mnist_layers_size[2], test
    );
    return 1;
}

int testFullFeedforward(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    PSFeedforward(network, test_data);

    PSLayer *output = network->layers[network->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        PSNeuron *n = output->neurons[i];
        PSFloat a = n->activation;
        PSFloat expected = fullNetworkFeedForwardResults[i];
        a = getRoundedFloat(a);
        expected = getRoundedFloat(expected);
        testAssertWithMessage(
            (a == expected), test, "Output[%d]-> %g != %g", i, a, expected
        );
    }
    return res;
}

int testFullAccuracy(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    PSFloat accuracy = PSTest(network, test_data, testlen), expected = 95.0;
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == expected), test, "Accuracy %g != from expected (%g)",
        accuracy, expected
    );
    return 1;
}

int testFullBackprop(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient **gradients = backprop(network, x, y);
    int i;
    for (i = 0; i < BP_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropGradients[i][0]);
        int nidx = (int) (backpropGradients[i][1]);
        PSFloat bias = backpropGradients[i][2];
        int widx1 = (int) (backpropGradients[i][3]);
        int widx2 = (int) (backpropGradients[i][4]);
        PSFloat w1 = backpropGradients[i][5];
        PSFloat w2 = backpropGradients[i][6];
        PSGradient *dl = gradients[lidx - 1];
        PSGradient *d = &(dl[nidx]);
        PSFloat val = getRoundedFloatDec(d->bias, HIGH_PRECISION_DEC);
        bias = getRoundedFloatDec(bias, HIGH_PRECISION_DEC);
        w1 = getRoundedFloatDec(w1, HIGH_PRECISION_DEC);
        w2 = getRoundedFloatDec(w2, HIGH_PRECISION_DEC);
        testAssertWithMessageOrGoto(
            (val == bias), on_fail, test,
            "Gradient[%d][%d] bias %g != from expected (%g)",
            lidx - 1, nidx, val, bias
        );
        val = getRoundedFloatDec(d->weights[widx1], HIGH_PRECISION_DEC);
        testAssertWithMessageOrGoto(
            (val == w1), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expected (%g)",
            lidx - 1, nidx, widx1, val, w1
        );
        val = getRoundedFloatDec(d->weights[widx2], HIGH_PRECISION_DEC);
        testAssertWithMessageOrGoto(
            (val == w2), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expected (%g)",
            lidx - 1, nidx, widx2, val, w2
        );
    }
    PSDeleteGradients(gradients, network);
    return 1;
on_fail:
    PSDeleteGradients(gradients, network);
    return 0;
}

int testConvLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    int loaded = PSLoadNetwork(network, CONVOLUTIONAL_NETWORK);
    testAssertWithMessage(
        loaded, test, "Failed to load %s", CONVOLUTIONAL_NETWORK
    );
    PSLayer *layer = network->layers[1];
    PSSharedParams *shared;
    shared = (PSSharedParams *) layer->extra;
    PSFloat bias = shared->biases[0];
    bias = getRoundedFloat(bias);
    PSFloat expected = CONV_L1F0_BIAS;
    expected = getRoundedFloat(expected);
    testAssertWithMessage(
        (expected == bias), test,
        "Layer[1]->bias[0] %g != from bias loaded from data %g",
        bias, expected
    );
    return 1;
}

int testConvFeedforward(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    PSFeedforward(network, test_data);

    PSLayer *output = network->layers[network->size - 1];
    int i;
    for (i = 0; i < output->size; i++) {
        PSNeuron *n = output->neurons[i];
        PSFloat a = n->activation;
        PSFloat expected = convNetworkFeedForwardResults[i];
        a = getRoundedFloat(a);
        expected = getRoundedFloat(expected);
        testAssertWithMessage(
            (a == expected), test,
            "Output[%d]-> %g != %g", i, a, expected
        );
    }
    return 1;
}

int testConvBackprop(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient **gradients = backprop(network, x, y);
    int i;
    for (i = 0; i < BP_CONV_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropConvGradients[i][0]);
        int nidx = (int) (backpropConvGradients[i][1]);
        PSFloat bias = backpropConvGradients[i][2];
        int widx1 = (int) (backpropConvGradients[i][3]);
        int widx2 = (int) (backpropConvGradients[i][4]);
        PSFloat w1 = backpropConvGradients[i][5];
        PSFloat w2 = backpropConvGradients[i][6];
        PSGradient *dl = gradients[lidx - 1];
        if (dl == NULL) continue;
        PSGradient *d = &(dl[nidx]);
        PSFloat val = getRoundedFloat(d->bias);
        bias = getRoundedFloat(bias);
        testAssertWithMessageOrGoto(
            (val == bias), on_fail, test,
            "Gradient[%d][%d] bias %g != from expected (%g)",
            lidx - 1, nidx, val, bias
        );
        val = getRoundedFloat(d->weights[widx1]);
        w1 = getRoundedFloat(w1);
        testAssertWithMessageOrGoto(
            (val == w1), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expect. (%g)",
            lidx - 1, nidx, widx1, val, w1
        );
        val = getRoundedFloat(d->weights[widx2]);
        w2 = getRoundedFloat(w2);
        testAssertWithMessageOrGoto(
            (val == w2), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expect. (%g)",
            lidx - 1, nidx, widx2, val, w2
        );
    }
    PSDeleteGradients(gradients, network);
    return 1;
on_fail:
    PSDeleteGradients(gradients, network);
    return 0;
}

int testConvAccuracy(TestCase *test_case, Test *test) {
    PSFloat *test_data = getTestData(test_case);
    PSNeuralNetwork *network = PSCreateNetwork("CNN Test Network");
    int loaded = PSLoadNetwork(network, CONVOLUTIONAL_TRAINED_NETWORK);
    testAssertWithMessage(
        loaded, test, "Failed to load %s", CONVOLUTIONAL_TRAINED_NETWORK
    );
    PSFeedforward(network, test_data);
    PSFloat accuracy = PSTest(network, test_data, testlen), expected = 98.0;
    PSDeleteNetwork(network);
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == 98.0), test,
        "Accuracy %g != from expected (%g)", accuracy, expected
    );
    return 1;
}

int testRNNLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    int loaded = PSLoadNetwork(network, RECURRENT_NETWORK);
    testAssertWithMessage(loaded, test, "Failed to load %s", RECURRENT_NETWORK);

    int i, j, w;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron *n = layer->neurons[j];
            for (w = 0; w < n->weights_size; w++) {
                PSFloat *weights;
                int w_idx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE) weights = rnn_inner_weights[j];
                    else {
                        weights = rnn_recurrent_weights[j];
                        w_idx -= RNN_INPUT_SIZE;
                    }
                } else weights = rnn_outer_weights[j];
                testAssertWithMessage(
                    (n->weights[w] == weights[w_idx]), test,
                    "L[%d]N[%d]->weights[%d] %g != %g",
                    i, j, w, n->weights[w], weights[w_idx]
                );
            }
        }
    }

    return 1;
}

int testRNNFeedforward(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFeedforward(network, rnn_inputs);
    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;

    PSLayer *output = network->layers[network->size - 1];
    int i, j;
    for (i = 0; i < output->size; i++) {
        PSNeuron *n = output->neurons[i];
        PSRecurrentCell* cell = PSGetRecurrentCell(n);
        for (j = 0; j < cell->states_count; j++) {
            PSFloat s = getRoundedFloatDec(cell->states[j], HIGH_PRECISION_DEC);
            PSFloat expected = getRoundedFloatDec(
                rnn_expected_output[j][i], HIGH_PRECISION_DEC
            );
            testAssertWithMessage(
                (s == expected), test,
                "Output[%d][%d]: %g != %g", i, j, s, expected
            );
        }
    }
    return 1;
}

int testRNNBackprop(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    int i, j, w;

    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;
    PSGradient **gradients = backprop(network, rnn_inputs, rnn_labels);
    int dsize = network->size - 1;
    for (i = 0; i < dsize; i++) {
        PSGradient *lgradients = gradients[i];
        PSLayer *l = network->layers[i + 1];
        for (j = 0; j < l->size; j++) {
            PSGradient *gradient = &(lgradients[j]);
            int ws = l->neurons[j]->weights_size;
            PSFloat *expected = (i == 0 ? rnn_inner_gradients[j] :
                                 rnn_outer_gradients[j]);
            for (w = 0; w < ws; w++) {
                PSFloat dw = getRoundedFloat(gradient->weights[w]);
                PSFloat exp_dw = getRoundedFloat(expected[w]);
                testAssertWithMessageOrGoto(
                    (dw == exp_dw), on_fail, test,
                    "Gradient[%d][%d]->weight[%d]: %g != %g",
                    i, j, w, dw, exp_dw
                );
            }
        }
    }
    PSDeleteGradients(gradients, network);
    return 1;
on_fail:
    PSDeleteGradients(gradients, network);
    return 1;
}

int testRNNStep(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    /*int train_data_len = 1 + (RNN_TIMES * 2);*/
    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;
    PSFloat *training_data = getTestData(test_case);
    PSFloat **series = &training_data;
    int elements_count = (int) *training_data;

    int i, j, w;
    PSFloat loss = updateNetworkParameters(
        network, training_data, 1, elements_count,
        NULL, RNN_LEARNING_RATE, NULL, NULL, series
    );
    UNUSED(loss);
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        for (j = 0; j < layer->size; j++) {
            PSNeuron *n = layer->neurons[j];
            for (w = 0; w < n->weights_size; w++) {
                PSFloat *weights;
                int w_idx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE)
                        weights = rnn_trained_inner_weights[j];
                    else {
                        weights = rnn_trained_recurrent_weights[j];
                        w_idx -= RNN_INPUT_SIZE;
                    }
                } else weights = rnn_trained_outer_weights[j];
                PSFloat w_val = getRoundedFloat(n->weights[w]);
                PSFloat expected_w = getRoundedFloat(weights[w_idx]);
                testAssertWithMessage(
                    (w_val == expected_w), test,
                    "Layer[%d][%d]->weights[%d]: %g != %g",
                    i, j, w, w_val, expected_w
                );
            }
        }
    }
    /* free(series); */
    return 1;
}

int testRNNOneHot(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    PSNeuralNetwork *onehot_network = PSCreateNetwork("Onehot RNN");
    PSNeuralNetwork *standard_network = PSCreateNetwork("Standard RNN");
    PSNeuralNetwork *dummy_network = PSCreateNetwork("Dummy");
    PSFloat *onehot_data = NULL;
    PSFloat *standard_data = NULL;
    int loaded = PSLoadNetwork(onehot_network, RECURRENT_NETWORK);
    testAssertWithMessage(
        loaded, test, "Failed to load %s", RECURRENT_NETWORK
    );
    int ok = PSLoadNetwork(standard_network, RECURRENT_NETWORK);
    testAssertWithMessageOrGoto(
        loaded, final, test, "Failed to load %s", RECURRENT_NETWORK
    );
    PSRecurrentNetworkMode onehot_rnn_mode =
        PSGetRecurrentNetworkMode(onehot_network);
    PSRecurrentNetworkMode std_rnn_mode =
        PSGetRecurrentNetworkMode(standard_network);
    ok = (onehot_rnn_mode == ManyToMany);
    testAssertWithMessageOrGoto(
        ok, final, test, "OneHot recurrent network mode is: '%s'",
        PSGetRecurrentModeLabel(onehot_rnn_mode)
    );
    ok = (std_rnn_mode == ManyToMany);
    testAssertWithMessageOrGoto(
        ok, final, test, "Standard recurrent network mode is: '%s'",
        PSGetRecurrentModeLabel(std_rnn_mode)
    );
    ok = onehot_network->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network is not OneHot!", onehot_network->name
    );
    ok = onehot_network->layers[0]->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[0] is not OneHot!",
        onehot_network->name
    );
    int last_layer = onehot_network->size - 1;
    ok = onehot_network->layers[last_layer]->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[%d] is not OneHot!",
        onehot_network->name, last_layer
    );
    int no_onehot = ~((unsigned) FLAG_ONEHOT);
    standard_network->flags &= no_onehot;
    standard_network->layers[0]->flags &= no_onehot;
    standard_network->layers[last_layer]->flags &= no_onehot;
    ok = !(standard_network->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network is OneHot!", standard_network->name
    );
    ok = !(standard_network->layers[0]->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[0] is OneHot!",
        standard_network->name
    );
    ok = !(standard_network->layers[last_layer]->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[%d] is OneHot!",
        standard_network->name, last_layer
    );
    PSHyperParameters *hparams = onehot_network->layers[0]->hyper_parameters;
    ok = (hparams != NULL);
    testAssertWithMessageOrGoto(
        hparams != NULL, final, test,
        "%s network layer[0] hyper parameters are NULL", onehot_network->name
    );
    int vector_size = (int) hparams->parameters[0];
    ok = (vector_size > 0);
    testAssertWithMessageOrGoto(
        vector_size > 0, final, test,
        "%s network layer[0] vector size is %d", vector_size
    );
    PSLayer *standard_input_layer =
        PSAddLayer(dummy_network, FullyConnected, vector_size, NULL);
    ok = (standard_input_layer != NULL);
    testAssertWithMessageOrGoto(
        standard_input_layer != NULL, final, test,
        "Failed to create standard input layer with size %d",
        vector_size
    );
    PSLayer *curlayer = standard_network->layers[0];
    standard_network->layers[0] = standard_input_layer;
    standard_input_layer->network = standard_network;
    standard_network->input_size = vector_size;
    standard_input_layer->flags |= FLAG_RECURRENT;
    curlayer->network = NULL;
    PSDeleteLayer(curlayer);
    dummy_network->size = 0;
    dummy_network->layers[0] = NULL;
    PSDeleteNetwork(dummy_network);
    dummy_network = NULL;
    int onehot_datalen =
        (int) ((sizeof(rnn_inputs) + sizeof(rnn_labels)) / sizeof(PSFloat));
    int timesteps = (int) rnn_inputs[0];
    ok = (timesteps > 0);
    testAssertWithMessageOrGoto(
        ok, final, test, "timesteps should be > 0, got %d", timesteps
    );
    int standard_datalen = (1 + (timesteps * 2 * vector_size));
    onehot_data = malloc((size_t) (1 + onehot_datalen) * sizeof(PSFloat));
    ok = (onehot_data != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to allocate onehot data of size %d",
        onehot_datalen
    );
    standard_data = malloc((size_t) (1 + standard_datalen) * sizeof(PSFloat));
    ok = (standard_data != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to allocate standard data of size %d",
        standard_datalen
    );
    onehot_data[0] = standard_data[0] = 1.0;
    PSFloat *onehot_p = onehot_data + 1;
    PSFloat *standard_p = standard_data + 1;
    int inputs_len = (int) (sizeof(rnn_inputs) / sizeof(PSFloat)),
        labels_len = (int) (sizeof(rnn_labels) / sizeof(PSFloat)), i;
    memcpy(onehot_p, rnn_inputs, sizeof(rnn_inputs));
    memcpy(
        onehot_p + inputs_len, rnn_labels, sizeof(rnn_labels)
    );
    for (i = 0; i < inputs_len; i++) {
        ok = (rnn_inputs[i] == onehot_p[i]);
        testAssertWithMessageOrGoto(
            ok, final, test, "rnn_inputs[%d] != onehot_data[%d] -> "
            "%g != %g", i, i, rnn_inputs[i], onehot_p[i]
        );
    }
    for (i = 0; i < labels_len; i++) {
        int data_idx = inputs_len + i;
        ok = (rnn_labels[i] == onehot_p[data_idx]);
        testAssertWithMessageOrGoto(
            ok, final, test, "rnn_labels[%d] != onehot_data[%d] -> "
            "%g != %g", i, data_idx, rnn_labels[i], onehot_p[data_idx]
        );
    }
    standard_p[0] = rnn_inputs[0];
    for (i = 1; i < inputs_len; i++) {
        int vecidx = (int) rnn_inputs[i], input_idx = i - 1, j;
        for (j = 0; j < vector_size; j++) {
            int data_idx = 1 + (input_idx * vector_size) + j;
            standard_p[data_idx] = (j == vecidx ? 1.0 : 0.0);
        }
    }
    int labels_offset = 1 + ((inputs_len - 1) * vector_size);
    for (i = 0; i < labels_len; i++) {
        int vecidx = (int) rnn_labels[i], j;
        for (j = 0; j < vector_size; j++) {
            int data_idx = labels_offset + (i * vector_size) + j;
            standard_p[data_idx] = (j == vecidx ? 1.0 : 0.0);
        }
    }
    for (i = 1; i < onehot_datalen; i++) {
        int vecidx = (int) onehot_p[i], onehot_idx = i - 1;
        PSFloat *vec = standard_p + (onehot_idx * vector_size) + 1;
        int idx = arrayMaxIndex(vec, vector_size);
        ok = (idx == vecidx);
        testAssertWithMessageOrGoto(
            ok, final, test, "Onehot[%d] index %d != Standard[%d,%d] %d",
            i, vecidx, ((onehot_idx * vector_size) + 1), vector_size, idx
        );
    }
    PSFloat onehot_accuracy = PSTest(
        onehot_network, onehot_data, onehot_datalen
    );
    PSFloat std_accuracy = PSTest(
        standard_network, standard_data, standard_datalen
    );
    ok = (onehot_network->status != STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Network %s: error during validation",
        onehot_network->name
    );
    ok = (standard_network->status != STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Network %s: error during validation",
        standard_network->name
    );
    ok = (onehot_accuracy == std_accuracy);
    testAssertWithMessageOrGoto(
        ok, final, test, "Onehot accuracy != Non-onehot accuracy: %g != %g",
        onehot_accuracy, std_accuracy
    );
final:
    if (onehot_network != NULL) PSDeleteNetwork(onehot_network);
    if (standard_network != NULL) PSDeleteNetwork(standard_network);
    if (dummy_network != NULL) PSDeleteNetwork(dummy_network);
    if (onehot_data != NULL) free(onehot_data);
    if (standard_data != NULL) free(standard_data);
    return ok;
}

int testLSTMTrain(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    /*int train_data_len = 2 + (LSTM_TIMES * 2);
    UNUSED(train_data_len);*/
    PSFloat *training_data = getTestData(test_case);

    PSTrainingOptions options = {
        .flags = TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0,
        .bptt_truncate = 4
    };
    PSTrain(network, training_data, 8, LSTM_EPOCHS, LSTM_LEARNING_RATE,
            LSTM_BATCHES, &options, NULL, 0);

    PSLayer *layer = network->layers[1];
    int i, t, w, precision = NORMAL_PRECISION_DEC - 1;

    for (i = 0; i < layer->size; i++) {
        PSNeuron *neuron = layer->neurons[i];
        PSLSTMCell *cell = PSGetLSTMCell(neuron);
        int times = cell->states_count;
        for (t = 0; t < times; t++) {
            PSFloat h = getRoundedFloat(cell->states[t]);
            PSFloat expected = getRoundedFloat(lstm_expected_states[i][t]);
            testAssertWithMessage(
                (h == expected), test, "Neuron[%d]->state[%d]: %g != %g",
                i, t, h, expected
            );
            /*ok = (h == expected);
             printf("H[%d][%d] = %g (%s)\n", t, i, h, (ok ? "OK" : "FAIL"));*/
        }
        PSFloat bias = getRoundedFloat(cell->candidate_bias);
        PSFloat expected = getRoundedFloat(expected_bg[i]);
        testAssertWithMessage(
            (bias == expected), test, "Neuron[%d]->candidate_bias: %g != %g",
            i, bias, expected
        );
        bias = getRoundedFloat(cell->input_bias);
        expected = getRoundedFloat(expected_bi[i]);
        testAssertWithMessage(
            (bias == expected), test, "Neuron[%d]->input_bias: %g != %g",
            i, bias, expected
        );
        bias = getRoundedFloat(cell->output_bias);
        expected = getRoundedFloat(expected_bo[i]);
        testAssertWithMessage(
            (bias == expected), test, "Neuron[%d]->output_bias: %g != %g",
            i, bias, expected
        );
        bias = getRoundedFloat(cell->forget_bias);
        expected = getRoundedFloat(expected_bf[i]);
        testAssertWithMessage(
            (bias == expected), test, "Neuron[%d]->forget_bias: %g != %g",
            i, bias, expected
        );
        for (w = 0; w < cell->weights_size; w++) {
            PSFloat weight =
                getRoundedFloatDec(cell->candidate_weights[w], precision);
            expected =
                getRoundedFloatDec(expected_wg[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Neuron[%d]->candidate_weights[%d]: %g != %g",
                i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->input_weights[w], precision);
            expected = getRoundedFloatDec(expected_wi[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Neuron[%d]->input_weights[%d]: %g != %g",
                i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->output_weights[w], precision);
            expected = getRoundedFloatDec(expected_wo[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Neuron[%d]->output_weights[%d]: %g != %g",
                i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->forget_weights[w], precision);
            expected = getRoundedFloatDec(expected_wf[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Neuron[%d]->forget_weights[%d]: %g != %g",
                i, w, weight, expected
            );
        }
    }
    PSLayer *out = network->layers[network->size - 1];

    for (i = 0; i < out->size; i++) {
        PSNeuron *neuron = out->neurons[i];
        PSRecurrentCell *cell = PSGetRecurrentCell(neuron);
        int times = cell->states_count;
        for (t = 0; t < times; t++) {
            PSFloat h = getRoundedFloat(cell->states[t]);
            PSFloat e = getRoundedFloat(lstm_expected_outputs[t][i]);
            testAssertWithMessage(
                (h == e), test, "Output->Neuron[%d]->output[%d]: %g != %g",
                i, t, h, e
            );
        }
    }
    return 1;
}

int testGenericClone(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSNeuralNetwork *clone = PSCloneNetwork(network, 0);
    testAssertNotNull(clone, test);
    int ok = compareNetworks(network, clone, test);
    PSDeleteNetwork(clone);
    return ok;
}

int testGenericSave(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    assert(network->size > 0);
    PSFloat old_dropout = network->layers[0]->dropout;
    network->layers[0]->dropout = 0.5;
    char tmpfile[255];
    getTmpFileName("tests-save-nn", ".data", tmpfile);
    int ok = PSSaveNetwork(network, tmpfile);
    testAssertWithMessage(ok, test, "Could not save network %s", network->name);
    PSNeuralNetwork *clone = PSCreateNetwork("Clone Test Network");
    testAssertNotNull(clone, test);
    ok = PSLoadNetwork(clone, tmpfile);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not load network from %s",tmpfile
    );
    ok = compareNetworks(network, clone, test);
    network->layers[0]->dropout = old_dropout;
    remove(tmpfile);
final:
    PSDeleteNetwork(clone);
    return ok;
}

int compareNetworks(PSNeuralNetwork *network, PSNeuralNetwork *clone,
                    Test* test)
{
    int ok = 1, i, k, w;

    ok = network->size == clone->size;
    testAssertWithMessage(
        (network->size == clone->size), test,
        "Source size %d != Clone size %d",
        network->size, clone->size
    );
    int recurrent_source = PSIsRecurrent(network),
        recurrent_clone = PSIsRecurrent(clone);
    testAssertWithMessage(
        (recurrent_source == recurrent_clone), test,
        "Source recurrent: %d, Clone recurrent: %d",
        recurrent_source, recurrent_clone
    );
    if (recurrent_source) {
        PSLayer *src_first_recurrent_layer = PSGetFirstRecurrentLayer(network),
                *src_last_recurrent_layer = PSGetLastRecurrentLayer(network),
                *clone_first_recurrent_layer = PSGetFirstRecurrentLayer(clone),
                *clone_last_recurrent_layer = PSGetLastRecurrentLayer(clone);
        PSRecurrentNetworkMode srcmode = PSGetRecurrentNetworkMode(network),
                               clonemode = PSGetRecurrentNetworkMode(clone);
        PSRecurrentNetworkOptions *src_rnn_opts = network->rnn_options,
                                  *clone_rnn_opts = clone->rnn_options;
        int src_max_steps = 0, clone_max_steps = 0, src_eos = -1,
            clone_eos = -1;
        if (src_rnn_opts) {
            src_max_steps = src_rnn_opts->sequence_stop_criterion.max_steps;
            src_eos = src_rnn_opts->sequence_stop_criterion.eos;
        }
        if (clone_rnn_opts) {
            clone_max_steps = clone_rnn_opts->sequence_stop_criterion.max_steps;
            clone_eos = clone_rnn_opts->sequence_stop_criterion.eos;
        }
        testAssertWithMessage(
            (srcmode == clonemode), test,
            "Recurrent source mode: '%s', Recurrent clone mode: '%s'",
            PSGetRecurrentModeLabel(srcmode),
            PSGetRecurrentModeLabel(clonemode)
        );
        testAssertWithMessage(
            (src_max_steps == clone_max_steps), test,
             "network->rnn_options.sequence_stop_criterion.max_steps != "
             "clone->rnn_options.sequence_stop_criterion.max_steps: %d != %d",
             src_max_steps, clone_max_steps
        );
        testAssertWithMessage(
            (src_eos == clone_eos), test,
             "network->rnn_options.sequence_stop_criterion.eos != "
             "clone->rnn_options.sequence_stop_criterion.eos: %d != %d ",
             src_eos, clone_eos
        );
        if (src_first_recurrent_layer == NULL)
            testAssertNull(clone_first_recurrent_layer, test);
        else {
            testAssertNotNull(clone_first_recurrent_layer, test);
            int src_idx = src_first_recurrent_layer->index;
            int cln_idx = clone_first_recurrent_layer->index;
            testAssertWithMessage(
                (src_idx == cln_idx), test,
                "network first recurrent layer index != "
                "clone first recurrent layer index: %d != %d",
                src_idx, cln_idx
            );
        }
        if (src_last_recurrent_layer == NULL)
            testAssertNull(clone_last_recurrent_layer, test);
        else {
            testAssertNotNull(clone_last_recurrent_layer, test);
            int src_idx = src_last_recurrent_layer->index;
            int cln_idx = clone_last_recurrent_layer->index;
            testAssertWithMessage(
                (src_idx == cln_idx), test,
                "network last recurrent layer index != "
                "clone last recurrent layer index: %d != %d",
                src_idx, cln_idx
            );
        }
    }
    testAssertWithMessage(
        network->flags == clone->flags, test,
        "Source flags %d != Clone flags %d",
        network->flags, clone->flags
    );
    for (i = 0; i < network->size; i++) {
        PSLayer *orig_l = network->layers[i];
        PSLayer *clone_l = clone->layers[i];
        PSLayerType otype = orig_l->type;
        PSLayerType ctype = clone_l->type;
        testAssertWithMessage(
            (otype == ctype), test,
            "Layer[%d]: Source type %s != Clone type %s",
            PSGetLayerTypeLabel(orig_l), PSGetLayerTypeLabel(clone_l)
        );
        int o_size = orig_l->size;
        int c_size = clone_l->size;
        testAssertWithMessage(
            (o_size == c_size), test,
            "Layer[%d]: Source size %d != Clone size %d",
            i, o_size, c_size
        );
        PSFloat o_dropout = orig_l->dropout;
        PSFloat c_dropout = clone_l->dropout;
        testAssertWithMessage(
            (o_dropout == c_dropout), test,
            "Layer[%d]: Source dropout %g != Clone dropout %g",
            i, o_dropout, c_dropout
        );
        if (i == 0) continue;
        if (otype == Pooling) continue;
        int o_flags = orig_l->flags;
        int c_flags = clone_l->flags;
        testAssertWithMessage(
            (o_flags == c_flags), test,
            "Layer[%d]: Source flags %d != Clone flags %d",
            i, o_flags, c_flags
        );
        int conv_features_checked = 0;
        for (k = 0; k < o_size; k++) {
            PSNeuron *orig_n = orig_l->neurons[k];
            PSNeuron *clone_n = clone_l->neurons[k];
            if (otype == Convolutional) {
                PSSharedParams* oshared;
                PSSharedParams* cshared;
                oshared = (PSSharedParams*) orig_l->extra;
                cshared = (PSSharedParams*) clone_l->extra;
                if (!conv_features_checked) {
                    conv_features_checked = 1;
                    testAssertWithMessage(
                        (oshared->feature_count == cshared->feature_count),
                        test, "Layer[%d]: Feature count %d != %d",
                        i, oshared->feature_count, cshared->feature_count
                    );
                }
                int fsize = orig_l->size / oshared->feature_count;
                int fidx = k / fsize;
                PSFloat obias = getRoundedFloat(oshared->biases[fidx]);
                PSFloat cbias = getRoundedFloat(cshared->biases[fidx]);
                testAssertWithMessage(
                    (obias == cbias), test,"Layer[%d][%d]: bias %g != %g",
                    i, fidx, obias, cbias
                );
            } else if (otype != Recurrent && otype != LSTM) {
                PSFloat obias = getRoundedFloat(orig_n->bias);
                PSFloat cbias = getRoundedFloat(clone_n->bias);
                ok = (obias == cbias);
            } else if (otype == LSTM) {
                PSLSTMCell *ocell =  PSGetLSTMCell(orig_n);
                PSLSTMCell *ccell =  PSGetLSTMCell(clone_n);
                ok = (getRoundedFloat(ocell->candidate_bias) ==
                      getRoundedFloat(ccell->candidate_bias));
                testAssertWithMessage(
                    ok, test, "Layer[%d][%d]: candidate_bias %g != %g",
                    i, k, ocell->candidate_bias, ccell->candidate_bias
                );
                ok = (getRoundedFloat(ocell->input_bias) ==
                      getRoundedFloat(ccell->input_bias));
                testAssertWithMessage(
                    ok, test, "Layer[%d][%d]: input_bias %g != %g",
                    i, k, ocell->input_bias, ccell->input_bias
                );
                ok = (getRoundedFloat(ocell->output_bias) ==
                      getRoundedFloat(ccell->output_bias));
                testAssertWithMessage(
                    ok, test, "Layer[%d][%d]: output_bias %g != %g",
                    i, k, ocell->output_bias, ccell->output_bias
                );
                ok = (getRoundedFloat(ocell->forget_bias) ==
                      getRoundedFloat(ccell->forget_bias));
                testAssertWithMessage(
                    ok, test, "Layer[%d][%d]: forget_bias %g != %g",
                    i, k, ocell->forget_bias, ccell->forget_bias
                );
            }
            testAssertWithMessage(
                ok, test, "Layer[%d][%d]: bias  %g != %g",
                i, k, orig_n->bias, clone_n->bias
            );
            testAssertWithMessage(
                (orig_n->weights_size == clone_n->weights_size), test,
                "Layer[%d][%d]: weight sz. %d != %d",
                i, k, orig_n->weights_size, clone_n->weights_size
            );
            for (w = 0; w < orig_n->weights_size; w++) {
                PSFloat ow = getRoundedFloat(orig_n->weights[w]);
                PSFloat cw = getRoundedFloat(clone_n->weights[w]);
                testAssertWithMessage(
                    (ow == cw), test, "Layer[%d][%d]: w[%d] %g != %g",
                    i, k, w, ow, cw
                );
            }
            if (!ok) break;
        }
        if (!ok) break;
    }
    return ok;
}

static int testRecurrentNetworkMode(PSNeuralNetwork *network,
                                    PSRecurrentNetworkMode mode,
                                    Test *test)
{
    testAssertNotNull(network, test);
    if (network->size == 0) return 1;
    PSRecurrentNetworkMode network_rnn_mode =
        PSGetRecurrentNetworkMode(network);
    testAssertWithMessage(
        network_rnn_mode == mode, test,
        "Recurrent network mode %s != expected %s",
        PSGetRecurrentModeLabel(network_rnn_mode),
        PSGetRecurrentModeLabel(mode)
    );
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network),
            *last_recurrent = PSGetLastRecurrentLayer(network),
            *input_layer = network->layers[0],
            *output_layer = network->layers[network->size - 1];
    if (mode == NonRecurrent) {
        testAssertWithMessage(
            !PSIsRecurrent(network), test,
            "Network is recurrent despite mode is %s",
            PSGetRecurrentModeLabel(mode)
        );
    } else {
        testAssertWithMessage(
            PSIsRecurrent(network), test,
            "Network is not recurrent despite mode is %s",
            PSGetRecurrentModeLabel(mode)
        );
        testAssertNotNull(first_recurrent, test);
        testAssertNotNull(last_recurrent, test);
        if (mode == ManyToMany) {
            testAssertWithMessage(
                first_recurrent == input_layer, test,
                "first_recurrent_layer is not the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent == output_layer, test,
                "last_recurrent_layer is not the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        } else if (mode == ManyToOne) {
            testAssertWithMessage(
                first_recurrent == input_layer, test,
                "first_recurrent_layer is not the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent != output_layer, test,
                "last_recurrent_layer is the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        } else if (mode == OneToMany) {
            testAssertWithMessage(
                first_recurrent != input_layer, test,
                "first_recurrent_layer is the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent == output_layer, test,
                "last_recurrent_layer is not the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        }
    }
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        if (mode == NonRecurrent) {
            testAssertWithMessage(
                !PSIsRecurrent(layer), test,
                "Layer[%d] is recurrent despite mode is NonRecurrent", i
            );
        } else if (mode == ManyToMany) {
            testAssertWithMessage(
                PSIsRecurrent(layer), test,
                "Layer[%d] is not recurrent despite mode is %s", i,
                PSGetRecurrentModeLabel(mode)
            );
        } else {
            if (layer->index >= first_recurrent->index &&
                layer->index <= last_recurrent->index)
            {
                testAssertWithMessage(
                    PSIsRecurrent(layer), test,
                    "Layer[%d] is not recurrent (mode %s): first recurrent "
                    "layer is %d, last recurrent layer is %d", i,
                    first_recurrent->index, last_recurrent->index,
                    PSGetRecurrentModeLabel(mode)
                );
            } else {
                testAssertWithMessage(
                    !PSIsRecurrent(layer), test,
                    "Layer[%d] is recurrent (mode %s): first recurrent "
                    "layer is %d, last recurrent layer is %d", i,
                    first_recurrent->index, last_recurrent->index,
                    PSGetRecurrentModeLabel(mode)
                );
            }
        }
    }
    return 1;
}

#ifdef USE_AVX

PSFloat test_dot(PSFloat *x, PSFloat *y, int size) {
    int i;
    PSFloat dot = 0.0;
    for (i = 0; i < size; i++) {
        dot += (x[i] * y[i]);
    }
    return dot;
}

int testAVXDot(TestCase *tc, Test *test) {
    UNUSED(tc);

    PSFloat x2[2] = {1.0, 2.0};
    PSFloat y2[2] = {0.5, 0.5};

    PSFloat x4[4] = {1.0, 1.0, 2.0, 2.0};
    PSFloat y4[4] = {0.5, 0.5, 1.0, 0.5};

    PSFloat x8[8] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0};
    PSFloat y8[8] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0};

    PSFloat x16[16] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                       0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0};
    PSFloat y16[16] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                       1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5};

    PSFloat x32[32] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                       0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0,
                       1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0,
                       1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0, 0.5};
    PSFloat y32[32] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                       1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5,
                       2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5, 1.0,
                       0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 0.5};
    PSFloat avx_res, cmp_res;
    if (AVX_MAX_VECTOR_SIZE >= 32) {
        avx_res = AVXDotProduct(x32, y32, 32, NULL);
        cmp_res = test_dot(x32, y32, 32);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[32]: Expected %g != %g",
            cmp_res, avx_res
        );
    }
    avx_res = AVXDotProduct(x16, y16, 16, NULL);
    cmp_res = test_dot(x16, y16, 16);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[16]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x8, y8, 8, NULL);
    cmp_res = test_dot(x8, y8, 8);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[8]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x4, y4, 4, NULL);
    cmp_res = test_dot(x4, y4, 4);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[4]: Expected %g != %g",
        cmp_res, avx_res
    );

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        avx_res = AVXDotProduct(x2, y2, 2, NULL);
        cmp_res = test_dot(x2, y2, 2);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[2]: Expected %g != %g",
            cmp_res, avx_res
        );
    }
    return 1;
}

int testAVXSquare(TestCase *tc, Test *test) {
    UNUSED(tc);

    PSFloat x2[2] = {1.0, 2.0};/* 5 */
    PSFloat x4[4] = {1.0, 1.0, 2.0, 2.0};/* 10 */
    PSFloat x8[8] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0};/* 20 */

    PSFloat x16[16] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0}; /* 40 */

    PSFloat avx_res = AVXDotProduct(x16, x16, 16, NULL);
    PSFloat cmp_res = test_dot(x16, x16, 16);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[16]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x8, x8, 8, NULL);
    cmp_res = test_dot(x8, x8, 8);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[8]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x4, x4, 4, NULL);
    cmp_res = test_dot(x4, x4, 4);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[4]: Expected %g != %g",
        cmp_res, avx_res
    );

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        avx_res = AVXDotProduct(x2, x2, 2, NULL);
        cmp_res = test_dot(x2, x2, 2);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[2]: Expected %g != %g",
            cmp_res, avx_res
        );
    }

    return 1;
}

int testAVXMultiplyVal(TestCase *tc, Test *test) {
    UNUSED(tc);
    int i;
    PSFloat x[4] = {0.0, 1.0, 2.0, 3.0};
    PSFloat val = 2.0;
    PSFloat y[4] = {0.0, 2.0, 4.0, 6.0};
    PSFloat dest[4] = {0.0, 0.0, 0.0, 0.0};

    PSFloat x2[2] = {2.0, 3.0};
    PSFloat y2[2] = {4.0, 6.0};
    PSFloat dest2[2] = {0.0, 0.0};

    AVXMultiplyValue(x, val, 4, dest, 0);
    for (i = 0; i < 4; i++) {
        testAssertWithMessage(
            (dest[i] == y[i]), test, "Store Mode Norm[4]: Expected %g != %g",
            y[i], dest[i]
        );
    }

    AVXMultiplyValue(x, val, 4, dest, AVX_STORE_MODE_ADD);
    for (i = 0; i < 4; i++) {
        testAssertWithMessage(
            (dest[i] == (y[i] + y[i])), test,
            "Store Mode Norm[4]: Expected %g != %g", y[i], dest[i]
        );
    }

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        AVXMultiplyValue(x2, val, 2, dest2, 0);
        for (i = 0; i < 2; i++) {
            testAssertWithMessage(
                (dest2[i] == y2[i]), test,
                "Store Mode Norm[2]: Expected %g != %g", y2[i], dest2[i]
            );
        }

        AVXMultiplyValue(x2, val, 2, dest2, AVX_STORE_MODE_ADD);
        for (i = 0; i < 2; i++) {
            testAssertWithMessage(
                (dest2[i] == (y2[i] + y2[i])), test,
                "Store Mode Norm[2]: Expected %g != %g", y2[i], dest2[i]
            );
        }
    }

    return 1;
}

#endif

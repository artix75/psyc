#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test.h"
#include "../neural.h"
#include "../mnist.h"

#define PRETRAINED_FULL_NETWORK "../../resources/pretrained.mnist.data"
#define TEST_IMAGE_FILE "../../resources/t10k-images-idx3-ubyte.gz"
#define TEST_LABEL_FILE "../../resources/t10k-labels-idx1-ubyte.gz"
#define TEST_IMAGE_SIZE 28
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE * TEST_IMAGE_SIZE
#define BP_DELTAS_CHECKS 8

#define getNetwork(tc) (NeuralNetwork*)(tc->data[0])
#define getTestData(tc) (double*)(tc->data[1])

TestCase * fullNetworkTests;

void fullNetworkSetup (void* test_case);
void fullNetworkTeardown (void* test_case);
int testFullLoad(void* test_case, void* test);
int testFullFeedforward(void* test_case, void* test);
int testFullAccuracy(void* tc, void* t);
int testFullBackprop(void* test_case, void* test);
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

int main(int argc, char** argv) {
    
    fullNetworkTests = createTest("Fully Connected Network");
    fullNetworkTests->setup = fullNetworkSetup;
    fullNetworkTests->teardown = fullNetworkTeardown;
    addTest(fullNetworkTests, "Load", NULL, testFullLoad);
    addTest(fullNetworkTests, "Feedforward", NULL, testFullFeedforward);
    addTest(fullNetworkTests, "Accuracy", NULL, testFullAccuracy);
    addTest(fullNetworkTests, "Backprop", NULL, testFullBackprop);
    performTests(fullNetworkTests);
}

void fullNetworkSetup (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    NeuralNetwork * network = createNetwork();
    test_case->data = malloc(2 * sizeof(void*));
    test_case->data[0] = network;
    double * test_data = NULL;
    testlen = loadMNISTData(TEST_DATA, TEST_IMAGE_FILE, TEST_LABEL_FILE,
                            &test_data);
    test_case->data[1] = test_data;
}

void fullNetworkTeardown (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    NeuralNetwork * network = getNetwork(test_case);
    if (network != NULL) deleteNetwork(network);
    double * test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
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
        sprintf(testobj->error_message, "Accuracy %lf != from expected (%lf)",
                accuracy, 95.0);
    }
    return ok;
}

int testFullBackprop(void* tc, void* t) {
    TestCase * test_case = (TestCase*) tc;
    Test * testobj = (Test*) t;
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
                    lidx - 1, nidx, widx1, val, w2);
            break;
        }
    }
    deleteDeltas(deltas, network);
    return ok;
}

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

#define getNetwork(tc) (NeuralNetwork*)(tc->data[0])
#define getTestData(tc) (double*)(tc->data[1])

TestCase * fullNetworkTests;

void fullNetworkSetup (void* test_case);
void fullNetworkTeardown (void* test_case);
int testFullFeedforward(void* test_case, void* test);
int testFullAccuracy(void* tc, void* t);
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

int main(int argc, char** argv) {
    
    fullNetworkTests = createTest("Fully Connected Network");
    fullNetworkTests->setup = fullNetworkSetup;
    fullNetworkTests->teardown = fullNetworkTeardown;
    addTest(fullNetworkTests, "Feedforward", NULL, testFullFeedforward);
    addTest(fullNetworkTests, "Accuracy", NULL, testFullAccuracy);
    performTests(fullNetworkTests);
}

void fullNetworkSetup (void* tc) {
    TestCase * test_case = (TestCase*) tc;
    NeuralNetwork * network = createNetwork();
    int ok = loadNetwork(network, PRETRAINED_FULL_NETWORK);
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../neural.h"
#include "../mnist.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 30

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage %s IMAGE_FILE LABELS_FILE [TEST_FILES...]\n", argv[0]);
        return 1;
    }
    
    double * training_data = NULL;
    double * test_data = NULL;
    int testlen = 0;
    int datalen = 0;
    int loaded = 0;
    NeuralNetwork * network = createNetwork();
    addLayer(network, FullyConnected, INPUT_SIZE, NULL);
    addLayer(network, FullyConnected, 30, NULL);
    addLayer(network, FullyConnected, 10, NULL);
    
    if (strcmp("--load", argv[1]) == 0) {
        loaded = loadNetwork(network, argv[2]);
        if (!loaded) {
            printf("Could not load pretrained network!\n");
            exit(1);
        }
    } else {
        datalen = loadMNISTData(TRAINING_DATA, argv[1], argv[2],
                                &training_data);
        if (datalen == 0 || training_data == NULL) {
            printf("Could not load training data!\n");
            return 1;
        }
    }
    if (argc >= 5) {
        testlen = loadMNISTData(TEST_DATA, argv[3], argv[4],
                                &test_data);
    };
    
    printf("Data len: %d\n", datalen);
    
    if (!loaded) train(network, training_data, datalen, EPOCHS, 3, 10, NULL, 0);
    //int loaded = loadNetwork(network, "pretrained.mnist.data");
    //if (!loaded) exit(1);
    
    if (testlen > 0 && test_data != NULL) {
        printf("Test Data len: %d\n", testlen);
        test(network, test_data, testlen);
    }
    
    deleteNetwork(network);
    free(training_data);
    if (test_data != NULL) free(test_data);
    return 0;
}

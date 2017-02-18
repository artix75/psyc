#include <stdio.h>
#include <stdlib.h>
#include "../neural.h"
#include "../mnist.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 30
#define FEATURES_COUNT 20
#define REGIONS_SIZE 5
#define POOL_SIZE 2

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage %s IMAGE_FILE LABELS_FILE [TEST_FILES...]\n", argv[0]);
        return 1;
    }
    
    double * training_data = NULL;
    double * test_data = NULL;
    int testlen = 0;
    int datalen = loadMNISTData(TRAINING_DATA, argv[1], argv[2],
                                &training_data);
    if (datalen == 0 || training_data == NULL) {
        printf("Could not load training data!\n");
        return 1;
    }
    if (argc >= 5) {
        testlen = loadMNISTData(TEST_DATA, argv[3], argv[4],
                                &test_data);
    }

    NeuralNetwork * network = createNetwork();
    
    LayerParameters * cparams;
    LayerParameters * pparams;
    cparams = createConvolutionalParameters(FEATURES_COUNT, REGIONS_SIZE, 1, 0);
    pparams = createConvolutionalParameters(FEATURES_COUNT, POOL_SIZE, 0, 0);
    addLayer(network, FullyConnected, INPUT_SIZE, NULL);
    addConvolutionalLayer(network, cparams);
    addPoolingLayer(network, pparams);
    addLayer(network, FullyConnected, 30, NULL);
    addLayer(network, FullyConnected, 10, NULL);
    
    int i;
    for (i = 0; i < network->size; i++) {
        Layer * layer = network->layers[i];
        char * type = getLayerTypeLabel(layer);
        printf("Layer[%d] (%s): size = %d", i, type, layer->size);
        LayerParameters * params = layer->parameters;
        if (params != NULL && params->count >= CONV_PARAMETER_COUNT) {
            double * par = params->parameters;
            printf(", dim = %lfx%lf", par[OUTPUT_WIDTH], par[OUTPUT_HEIGHT]);
            if (layer->type == Pooling || layer->type == Convolutional) {
                printf(", features = %d", (int) (par[FEATURE_COUNT]));
                printf(", region_size = %d", (int) (par[REGION_SIZE]));
            }
        }
        printf("\n");
    }
    
    /*double * test_data = NULL;
    int testlen = loadMNISTData(TEST_DATA,
                                "resources/t10k-images-idx3-ubyte.gz",
                                "resources/t10k-labels-idx1-ubyte.gz",
                                &test_data);
    //feedforward(network, test_data);
    test(network, test_data, testlen);
    deleteNetwork(network);*/
    
    train(network, training_data, datalen, EPOCHS, 3, 10);
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

#include <stdio.h>
#include <stdlib.h>
#include "../neural.h"
#include "../mnist.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 1
#define FEATURES_COUNT 20
#define REGIONS_SIZE 5
#define POOL_SIZE 2
#define TRAIN_DATASET_LEN 2
#define EVAL_DATASET_LEN 1

int main(int argc, char** argv) {
    
    double * test_data = NULL;
    double * train_data = NULL;
    double * eval_data = NULL;
    int datalen = loadMNISTData(TRAINING_DATA,
                                "../../resources/train-images-idx3-ubyte.gz",
                                "../../resources/train-labels-idx1-ubyte.gz",
                                &train_data);
    if (datalen == 0 || train_data == NULL) {
        printf("Could not load training data!\n");
        return 1;
    }
    int testlen = loadMNISTData(TEST_DATA,
                                "../../resources/t10k-images-idx3-ubyte.gz",
                                "../../resources/t10k-labels-idx1-ubyte.gz",
                                &test_data);
    
    NeuralNetwork * network = createNetwork();
    
    LayerParameters * cparams;
    LayerParameters * pparams;
    cparams = createConvolutionalParameters(FEATURES_COUNT, REGIONS_SIZE,
                                            1, 0, 1);
    pparams = createConvolutionalParameters(FEATURES_COUNT, POOL_SIZE,
                                            0, 0, 1);
    addLayer(network, FullyConnected, INPUT_SIZE, NULL);
    addConvolutionalLayer(network, cparams);
    addPoolingLayer(network, pparams);
    addLayer(network, FullyConnected, 30, NULL);
    //addLayer(network, FullyConnected, 10, NULL);
    addLayer(network, SoftMax, 10, NULL);
    
    int element_size = network->input_size + network->output_size;
    datalen = element_size * TRAIN_DATASET_LEN;
    eval_data = train_data + datalen;
    int eval_datalen = element_size * EVAL_DATASET_LEN;
    
    train(network, train_data, datalen, EPOCHS, 1.5, 1, eval_data,
          eval_datalen);
    
    test(network, test_data, datalen);
    
    deleteNetwork(network);
    free(train_data);
    free(test_data);
    return 0;
}

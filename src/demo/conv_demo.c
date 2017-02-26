#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../neural.h"
#include "../mnist.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 30
#define FEATURES_COUNT 20
#define REGIONS_SIZE 5
#define POOL_SIZE 2
#define TRAIN_DATASET_LEN 50000
#define EVAL_DATASET_LEN 10000

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage %s IMAGE_FILE LABELS_FILE [TEST_FILES...]\n", argv[0]);
        printf("      %s --load TRAINED_DT_FILE [TEST_FILES...]\n", argv[0]);
        return 1;
    }
    
    double * training_data = NULL;
    double * test_data = NULL;
    double * validation_data = NULL;
    const char * pretrained_file = NULL;
    int testlen = 0;
    int datalen = 0;
    int valdlen = 0;
    
    int train_dataset_len = TRAIN_DATASET_LEN;
    int eval_dataset_len = EVAL_DATASET_LEN;
    
    if (strcmp("--load", argv[1]) != 0) {
        datalen = loadMNISTData(TRAINING_DATA, argv[1], argv[2],
                                &training_data);
        if (datalen == 0 || training_data == NULL) {
            printf("Could not load training data!\n");
            return 1;
        }
    } else {
        pretrained_file = argv[2];
    }
    if (argc >= 5) {
        testlen = loadMNISTData(TEST_DATA, argv[3], argv[4],
                                &test_data);
    }

    NeuralNetwork * network = createNetwork();
    
    if (pretrained_file == NULL) {
        LayerParameters * cparams;
        LayerParameters * pparams;
        cparams = createConvolutionalParameters(FEATURES_COUNT, REGIONS_SIZE,
                                                1, 0);
        pparams = createConvolutionalParameters(FEATURES_COUNT, POOL_SIZE,
                                                0, 0);
        addLayer(network, FullyConnected, INPUT_SIZE, NULL);
        addConvolutionalLayer(network, cparams);
        addPoolingLayer(network, pparams);
        addLayer(network, FullyConnected, 30, NULL);
        addLayer(network, FullyConnected, 10, NULL);
        //addLayer(network, SoftMax, 10, NULL);
        
        int element_size = network->input_size + network->output_size;
        int element_count = datalen / element_size;
        if (element_count < train_dataset_len) {
            printf("Loaded dataset elements %d < %d\n", element_count,
                   TRAIN_DATASET_LEN);
            deleteNetwork(network);
            return 1;
        } else {
            int remaining = element_count - train_dataset_len;
            if (remaining < eval_dataset_len && eval_dataset_len > 0) {
                printf("WARNING: eval. dataset cannot be > %d!\n", remaining);
                eval_dataset_len = remaining;
            }
            if (remaining == 0) {
                printf("WARNING: no dataset remained for evaluation!\n");
                eval_dataset_len = remaining;
            }
            datalen = train_dataset_len * element_size;
            if (eval_dataset_len == 0) validation_data = NULL;
            else {
                validation_data = training_data + datalen;
                valdlen = eval_dataset_len * element_size;
            }
        }
        
    } else {
        int loaded = loadNetwork(network, pretrained_file);
        if (!loaded) {
            printf("Could not load pretrained data %s\n", pretrained_file);
            deleteNetwork(network);
            return 1;
        }
    }
    
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
    
    if (datalen > 0)
        train(network, training_data, datalen, EPOCHS, 1.5, 10, validation_data,
              valdlen);
    //int loaded = loadNetwork(network, "pretrained.mnist.data");
    //if (!loaded) exit(1);
    
    if (testlen > 0 && test_data != NULL) {
        printf("Test Data len: %d\n", testlen);
        test(network, test_data, testlen);
    }
    if (pretrained_file == NULL)
        saveNetwork(network, "/tmp/pretrained.cnn.data");
    deleteNetwork(network);
    free(training_data);
    if (test_data != NULL) free(test_data);
    return 0;
}

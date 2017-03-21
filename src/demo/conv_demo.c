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
#define RELU_ENABLED 0

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
    if (network == NULL) {
        fprintf(stderr, "Could not create network!\n");
        if (training_data != NULL) free(training_data);
        if (test_data != NULL) free(test_data);
        return 1;
    }
    
    if (pretrained_file == NULL) {
        LayerParameters * cparams;
        LayerParameters * pparams;
        cparams = createConvolutionalParameters(FEATURES_COUNT, REGIONS_SIZE,
                                                1, 0, RELU_ENABLED);
        pparams = createConvolutionalParameters(FEATURES_COUNT, POOL_SIZE,
                                                0, 0, RELU_ENABLED);
        
        if (cparams == NULL || pparams == NULL) {
            fprintf(stderr, "Could not create layer params!\n");
            deleteNetwork(network);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            return 1;
        }
        
        addLayer(network, FullyConnected, INPUT_SIZE, NULL);
        addConvolutionalLayer(network, cparams);
        addPoolingLayer(network, pparams);
        addLayer(network, FullyConnected, 30, NULL);
        //addLayer(network, FullyConnected, 10, NULL);
        addLayer(network, SoftMax, 10, NULL);
        
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            deleteNetwork(network);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
            return 1;
        }
        
        int element_size = network->input_size + network->output_size;
        int element_count = datalen / element_size;
        if (element_count < train_dataset_len) {
            printf("Loaded dataset elements %d < %d\n", element_count,
                   TRAIN_DATASET_LEN);
            if (training_data != NULL) free(training_data);
            if (test_data != NULL) free(test_data);
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
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            deleteNetwork(network);
            return 1;
        }
    }
    
    if (datalen > 0)
        train(network, training_data, datalen, EPOCHS, 1.5, 10, 0,
              validation_data, valdlen);
    //int loaded = loadNetwork(network, "pretrained.mnist.data");
    //if (!loaded) exit(1);
    
    if (network->status == STATUS_ERROR) {
        deleteNetwork(network);
        if (training_data != NULL) free(training_data);
        if (test_data != NULL) free(test_data);
        return 1;
    }
    
    if (testlen > 0 && test_data != NULL) {
        printf("Test Data len: %d\n", testlen);
        test(network, test_data, testlen);
    }
    if (pretrained_file == NULL)
        saveNetwork(network, "/tmp/pretrained.cnn.data");
    deleteNetwork(network);
    if (training_data != NULL) free(training_data);
    if (test_data != NULL) free(test_data);
    return 0;
}

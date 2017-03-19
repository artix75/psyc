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

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <fenv.h>
#include <xmmintrin.h>

#include "../neural.h"
#include "w2v_training_data.h"

#define BATCHES 1
#define EPOCHS 30
#define LEARNING_RATE 0.025

void handler(int sig) {
    void *array[10];
    size_t size;
    
    // get void*'s for all entries on the stack
    size = backtrace(array, 10);
    
    // print out all the frames to stderr
    fprintf(stdout, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDOUT_FILENO);
    exit(1);
}

int main(int argc, char** argv) {
    
    signal(SIGSEGV, handler);
    signal(8, handler);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    
    const char * pretrained_file = NULL;
    
    if (argc >= 3 && strcmp("--load", argv[1]) == 0) {
        pretrained_file = argv[2];
    }
    
    NeuralNetwork * network = createNetwork();
    network->flags |= FLAG_ONEHOT;
    
    if (pretrained_file == NULL) {
        addLayer(network, FullyConnected, VOCABULARY_SIZE, NULL);
        addLayer(network, Recurrent, 60, NULL);
        addLayer(network, SoftMax, VOCABULARY_SIZE, NULL);
        network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
    } else {
        int loaded = loadNetwork(network, pretrained_file);
        if (!loaded) {
            printf("Could not load pretrained data %s\n", pretrained_file);
            deleteNetwork(network);
            return 1;
        }
    }
    
    train(network, training_data, TRAIN_DATA_LEN, EPOCHS, LEARNING_RATE,BATCHES,
          TRAINING_NO_SHUFFLE | TRAINING_ADJUST_RATE,
          validation_data, EVAL_DATA_LEN);
    //int loaded = loadNetwork(network, "pretrained.mnist.data");
    //if (!loaded) exit(1);
    
    if (TEST_DATA_LEN > 0) {
        printf("Test Data len: %d\n", TEST_DATA_LEN);
        test(network, test_data, TEST_DATA_LEN);
    }
    if (pretrained_file == NULL)
        saveNetwork(network, "/tmp/pretrained.cnn.data");
    deleteNetwork(network);
    //free(training_data);
    //if (TEST_DATA_LEN) free(test_data);
    return 0;
}

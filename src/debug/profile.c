/*
 * Copyright (C) 2016-2023 Fabio Nicotra <artix2 at gmail dot com>.
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
#include "../psyc.h"
#include "../dataset.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 1
#define FEATURES_COUNT 20
#define REGIONS_SIZE 5
#define POOL_SIZE 2
#define TRAIN_DATASET_LEN 2
#define EVAL_DATASET_LEN 1

#define RNN_INPUT_SIZE  4
#define RNN_HIDDEN_SIZE 2
#define RNN_TIMES       4
#define RNN_LEARNING_RATE 0.005

#define UNUSED(V) ((void) V)

PSFloat rnn_train_data[10] = {1, 4, 0, 1, 2, 3, 3, 2, 1, 0};

int main(int argc, char** argv) {
    UNUSED(argc);
    UNUSED(argv);
    PSFloat *test_data = NULL;
    PSFloat *train_data = NULL;
    PSFloat *eval_data = NULL;
    int datalen = PSLoadMNISTData(DATA_TYPE_TRAINING,
                                  "../../resources/train-images-idx3-ubyte.gz",
                                  "../../resources/train-labels-idx1-ubyte.gz",
                                  &train_data);
    if (datalen == 0 || train_data == NULL) {
        printf("Could not load training data!\n");
        return 1;
    }
    int testlen = PSLoadMNISTData(DATA_TYPE_TEST,
                                  "../../resources/t10k-images-idx3-ubyte.gz",
                                  "../../resources/t10k-labels-idx1-ubyte.gz",
                                  &test_data);
    UNUSED(testlen);
    PSNeuralNetwork *network = PSCreateNetwork("Profiling Network");

    PSLayerDef cdef = {
        .output_depth = FEATURES_COUNT, .filter_width = REGIONS_SIZE,
        .filter_height = REGIONS_SIZE, .stride = 1, .activation = PSRelu
    };
    PSLayerDef pdef = {
        .filter_width = REGIONS_SIZE, .filter_height = REGIONS_SIZE
    };
    PSAddLayer(network, FullyConnected, INPUT_SIZE, NULL);
    PSAddConvolutionalLayer(network, &cdef);
    PSAddPoolingLayer(network, &pdef);
    PSAddLayer(network, FullyConnected, 30, NULL);
    /* PSAddLayer(network, FullyConnected, 10, NULL); */
    PSAddLayer(network, SoftMax, 10, NULL);

    int element_size = network->input_size + network->output_size;
    datalen = element_size *TRAIN_DATASET_LEN;
    eval_data = train_data + datalen;
    int eval_datalen = element_size *EVAL_DATASET_LEN;

    PSTrain(network, train_data, datalen, eval_data, eval_datalen, PSTRAINOPT(
        .epochs = EPOCHS,
        .learning_rate = 1.5,
        .batch_size = 1
    ));

    PSTest(network, test_data, datalen);

    PSDeleteNetwork(network);
    free(train_data);
    free(test_data);

    network = PSCreateNetwork("Profiling RNN");
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, Recurrent, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;

    PSTrain(network, rnn_train_data, 10, NULL, 0, PSTRAINOPT(
        .epochs = EPOCHS,
        .learning_rate = RNN_LEARNING_RATE,
        .batch_size = 1
    ));

    PSDeleteNetwork(network);

    network = PSCreateNetwork("Profiling LSTM");
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, LSTM, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;

    PSTrain(network, rnn_train_data, 10, NULL, 0, PSTRAINOPT(
        .epochs = EPOCHS,
        .learning_rate = RNN_LEARNING_RATE,
        .batch_size = 1
    ));

    PSDeleteNetwork(network);

    return 0;
}

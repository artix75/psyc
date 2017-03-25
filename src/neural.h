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

#ifndef __NEURAL_H
#define __NEURAL_H

#define NN_VERSION      "0.0.1"

#define LAYER_TYPES  6

#define FEATURE_COUNT   0
#define REGION_SIZE     1
#define STRIDE          2
#define INPUT_WIDTH     3
#define INPUT_HEIGHT    4
#define OUTPUT_WIDTH    5
#define OUTPUT_HEIGHT   6
#define PADDING         7
#define USE_RELU        8

#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2
#define STATUS_ERROR        3

#define CONV_PARAMETER_COUNT 9
#define NULL_VALUE -9999999.99

#define FLAG_NONE 0
#define FLAG_RECURRENT  (1 << 0)
#define FLAG_ONEHOT     (1 << 1)

#define TRAINING_NO_SHUFFLE     (1 << 0)
#define TRAINING_ADJUST_RATE    (1 << 1)

#define BPTT_TRUNCATE   4


typedef double (*ActivationFunction)(double);
typedef int (*FeedforwardFunction)(void * network, void * layer, ...);
typedef double (*LossFunction)(double* x, double* y, int size, int onehot_size);

typedef struct {
    double bias;
    double * weights;
} Gradient;

typedef enum {
    FullyConnected,
    Convolutional,
    Pooling,
    Recurrent,
    LSTM,
    SoftMax
} LayerType;

typedef struct {
    int count;
    double * parameters;
} LayerParameters;

typedef struct {
    int feature_count;
    int weights_size;
    double * biases;
    double ** weights;
} ConvolutionalSharedParams;

typedef struct {
    int states_count;
    double * states;
    int weights_size;
    double * weights;
} RecurrentCell;

typedef struct {
    int index;
    int weights_size;
    double bias;
    double * weights;
    double activation;
    double z_value;
    void * extra;
    void * layer;
} Neuron;

typedef struct {
    LayerType type;
    int index;
    int size;
    LayerParameters * parameters;
    ActivationFunction activate;
    ActivationFunction derivative;
    FeedforwardFunction feedforward;
    Neuron ** neurons;
    int flags;
    void * extra;
#ifdef USE_AVX
    double * avx_activation_cache;
#endif
    void * network;
} Layer;

typedef struct {
    int size;
    Layer ** layers;
    LossFunction loss;
    int flags;
    unsigned char status;
    int input_size;
    int output_size;
    int current_epoch;
    int current_batch;
} NeuralNetwork;

NeuralNetwork * createNetwork();
NeuralNetwork * cloneNetwork(NeuralNetwork * network, int layout_only);
int loadNetwork(NeuralNetwork * network, const char* filename);
int saveNetwork(NeuralNetwork * network, const char* filename);
Layer * addLayer(NeuralNetwork * network, LayerType type, int size,
                 LayerParameters* params);
Layer * addConvolutionalLayer(NeuralNetwork * network, LayerParameters* params);
Layer * addPoolingLayer(NeuralNetwork * network, LayerParameters* params);
LayerParameters * createLayerParamenters(int count, ...);
int setLayerParameter(LayerParameters * params, int param, double value);
int addLayerParameter(LayerParameters * params, double val);
LayerParameters * createConvolutionalParameters(double feature_count,
                                                double region_size,
                                                int stride,
                                                int padding,
                                                int use_relu);
void deleteLayerParamenters(LayerParameters * params);
int feedforward(NeuralNetwork * network, double * values);

void deleteNetwork(NeuralNetwork * network);
void deleteLayer(Layer * layer);
void deleteNeuron(Neuron * neuron, Layer * layer);
void deleteGradients(Gradient ** gradients, NeuralNetwork * network);
Gradient ** backprop(NeuralNetwork * network, double * x, double * y);
void train(NeuralNetwork * network,
           double * training_data,
           int data_size,
           int epochs,
           double learning_rate,
           int batch_size,
           int flags,
           double * test_data,
           int test_size);
float test(NeuralNetwork * network, double * test_data, int data_size);
int verifyNetwork(NeuralNetwork * network);
//int arrayMaxIndex(double * array, int len);
char * getLabelForType(LayerType type);
char * getLayerTypeLabel(Layer * layer);

// Loss functions

double quadraticLoss(double * x, double * y, int size, int onehot_size);
double crossEntropyLoss(double * x, double * y, int size, int onehot_size);

void testShuffle(double * array, int size, int element_size);

#endif



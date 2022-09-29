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

#ifndef __PSYC_H
#define __PSYC_H

#include <time.h>
#include "types.h"

#define PSYC_VERSION      "0.2.2"

#define LAYER_TYPES  6

#define DEFAULT_RHO     0.95
#define DEFAULT_EPS     1e-8
#define DEFAULT_BETA1   0.9
#define DEFAULT_BETA2   0.999


#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2
#define STATUS_ERROR        3
#define STATUS_PAUSED       4
#define STATUS_ABORTED      5

#define ACTION_NONE         0
#define ACTION_PAUSE        4
#define ACTION_ABORT        5

#define PARAM_TYPE_BIAS          1
#define PARAM_TYPE_WEIGHT        2

#define TRAINING_PHASE_FEEDFORWARD  1
#define TRAINING_PHASE_BACKPROP     2
#define TRAINING_PHASE_UPDATE_GRAD  3

#define NULL_VALUE -9999999.99

#define FLAG_NONE 0
#define FLAG_RECURRENT      (1 << 0)
#define FLAG_ONEHOT         (1 << 1)
#define FLAG_AVX_DISABLED   (1 << 2)

/* Global Flags*/

#define FLAG_LOG_COLORS (1 << 0)

#define TRAINING_NO_SHUFFLE     (1 << 0)
#define TRAINING_ADJUST_RATE    (1 << 1)
#define TRAINING_WEIGHT_DECAY   (1 << 2)

#define BPTT_TRUNCATE   4

#ifdef USE_AVX
#define PSIsAVXDisabled(network) (network->flags & FLAG_AVX_DISABLED)
#else
#define PSIsAVXDisabled(network) (1)
#endif

typedef PSFloat  (*PSActivationFunction) (PSFloat);
typedef int     (*PSFeedforwardFunction) (void * network, void * layer, ...);
typedef PSFloat  (*PSLossFunction) (PSFloat* x, PSFloat* y, int size,
                                   int onehot_size);
typedef void    (*PSTrainCallback) (void * network, int epoch, int epochs,
                                    PSFloat loss, PSFloat previous_loss,
                                    float accuracy, PSFloat *rate,
                                    PSFloat *training_data);
typedef void    (*PSSignalHandler) (int);

typedef struct {
    PSFloat bias;
    PSFloat *weights;
} PSGradient;

typedef enum {
    FullyConnected,
    Convolutional,
    Pooling,
    Recurrent,
    LSTM,
    SoftMax
} PSLayerType;

typedef enum {
    NoTrainingOptimization,
    Adam,
    AdaGrad,
    AdaDelta,
    WindowGrad,
    Nesterov
} PSTrainingOptimization;

typedef struct {
    int count;
    PSFloat *parameters;
} PSLayerParameters;

typedef struct {
    int feature_count;
    int weights_size;
    PSFloat *biases;
    PSFloat **weights;
} PSSharedParams;

typedef struct {
    int                     flags;
    PSFloat                 l1_decay;
    PSFloat                 l2_decay;
    PSFloat                 momentum;
    PSFloat                 rho;
    PSFloat                 eps;
    PSFloat                 beta1;
    PSFloat                 beta2;
    PSTrainingOptimization  optimization;
    int                     validate_every_batches;
    int                     max_validation_elements;
    FILE                    *debug_dump_to;
} PSTrainingOptions;

typedef struct {
    int current_epoch;
    int current_batch;
    int current_element;
    int batch_size;
    time_t started_at;
    time_t ended_at;
    int requested_action;
    FILE *debug_dump_to;
} PSTrainingInfo;

typedef struct {
    int index;
    int weights_size;
    PSFloat bias;
    PSFloat * weights;
    PSFloat activation;
    PSFloat z_value;
    void * extra;
    void * layer;
} PSNeuron;

typedef struct {
    PSLayerType type;
    int index;
    int size;
    PSLayerParameters * parameters;
    PSActivationFunction activate;
    PSActivationFunction derivative;
    PSFeedforwardFunction feedforward;
    PSNeuron ** neurons;
    PSFloat * delta;
    int flags;
    void * extra;
#ifdef USE_AVX
    PSFloat * avx_activation_cache;
#endif
    void * network;
} PSLayer;

typedef struct {
    const char * name;
    int size;
    PSLayer ** layers;
    PSLossFunction loss;
    int flags;
    unsigned char status;
    int input_size;
    int output_size;
    PSTrainingInfo * training;
    PSTrainCallback onEpochTrained;
    PSTrainCallback onBatchTrained;
} PSNeuralNetwork;

extern int PSGlobalFlags;

PSNeuralNetwork * PSCreateNetwork(const char* name);
PSNeuralNetwork * PSCloneNetwork(PSNeuralNetwork * network, int layout_only);
int PSLoadNetwork(PSNeuralNetwork * network, const char* filename);
int PSSaveNetwork(PSNeuralNetwork * network, const char* filename);
PSLayer * PSAddLayer(PSNeuralNetwork * network, PSLayerType type, int size,
                     PSLayerParameters* params);
PSLayer * PSAddConvolutionalLayer(PSNeuralNetwork * network,
                                  PSLayerParameters* params);
PSLayer * PSAddPoolingLayer(PSNeuralNetwork * network,
                            PSLayerParameters* params);
PSLayerParameters * PSCreateLayerParamenters(int count, ...);
int PSSetLayerParameter(PSLayerParameters * params, int param, PSFloat value);
int PSAddLayerParameter(PSLayerParameters * params, PSFloat val);
PSLayerParameters * PSCreateConvolutionalParameters(PSFloat feature_count,
                                                    PSFloat region_size,
                                                    int stride,
                                                    int padding,
                                                    int use_relu);
void PSDeleteLayerParamenters(PSLayerParameters * params);
int PSFeedforward(PSNeuralNetwork * network, PSFloat * values);
int PSClassify(PSNeuralNetwork * network, PSFloat * values);

void PSDeleteNetwork(PSNeuralNetwork * network);
void PSDeleteLayer(PSLayer * layer);
void PSDeleteNeuron(PSNeuron * neuron, PSLayer * layer);
void PSDeleteGradients(PSGradient ** gradients, PSNeuralNetwork * network);

void PSTrain(PSNeuralNetwork * network,
             PSFloat * training_data,
             int data_size,
             int epochs,
             PSFloat learning_rate,
             int batch_size,
             PSTrainingOptions * options,
             PSFloat * test_data,
             int test_size);
void PSPauseTraining(PSNeuralNetwork * network);
void PSAbortTraining(PSNeuralNetwork * network);
float PSTest(PSNeuralNetwork * network, PSFloat * test_data, int data_size);
int PSVerifyNetwork(PSNeuralNetwork * network);
//int arrayMaxIndex(PSFloat * array, int len);
char * PSGetLabelForType(PSLayerType type);
char * PSGetLayerTypeLabel(PSLayer * layer);
void PSPrintNetworkInfo(PSNeuralNetwork * network);
int PSDumpNetworkActivations(PSNeuralNetwork * network, const char* filename);
int PSDumpNetworkDeltas(PSNeuralNetwork * network, const char* filename);
void PSSetDefaultTrainingOptions(PSTrainingOptions *options);

// Loss functions

PSFloat PSQuadraticLoss(PSFloat * x, PSFloat * y, int size, int onehot_size);
PSFloat PSCrossEntropyLoss(PSFloat * x, PSFloat * y, int size, int onehot_size);

/* Miscellaneous functions */

void PSHandleSignals(PSSignalHandler shutdown_handler);

#endif // __PSYC_H



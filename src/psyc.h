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

#ifndef __PSYC_H
#define __PSYC_H

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "types.h"
#include "config.h"
#include "maths.h"
#include "activation.h"
#include "optimization.h"

#define PSYC_VERSION      "0.9.0"

#define LAYER_TYPES     9

#define DEFAULT_RHO     0.95
#define DEFAULT_BETA1   0.9
#define DEFAULT_BETA2   0.999
#ifdef PS_DOUBLE_PRECISION
#define DEFAULT_EPS     1e-8
#else
#define DEFAULT_EPS     1e-7
#endif

#define DEFAULT_RECURRENT_MODE ManyToMany
#define MAX_RECURRENT_OUTPUT_STEPS 10
#define DEFAULT_EOS_INDEX   -1

#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2
#define STATUS_ERROR        3
#define STATUS_PAUSED       4
#define STATUS_ABORTED      5
#define STATUS_VALIDATING   6
#define STATUS_PRETRAINING  7

#define ACTION_NONE         0
#define ACTION_PAUSE        4
#define ACTION_ABORT        5

#define PARAM_TYPE_BIAS          1
#define PARAM_TYPE_WEIGHT        2

#define INIT_MODE_AUTO      0
#define INIT_MODE_RAND      1
#define INIT_MODE_ZERO      2

#define TRAINING_PHASE_FEEDFORWARD  1
#define TRAINING_PHASE_BACKPROP     2
#define TRAINING_PHASE_UPDATE_GRAD  3

#define PS_NULL_VALUE -9999999.99

/* Layer/Network Flags */
#define FLAG_NONE 0
#define FLAG_RECURRENT      (1 << 0)
#define FLAG_ONEHOT         (1 << 1)
#define FLAG_ACCEL_DISABLED (1 << 2) /* Formerly FLAG_AVX_DISABLED (< v0.4):
                                      * not used anymore, but disables any
                                      * acceleration when loading models
                                      * generated with versiob < 0.4 */
#define FLAG_NO_BIAS        (1 << 3)
#define FLAG_PRETRAINER     (1 << 4)

/* Training Flags */
#define TRAINING_NO_SHUFFLE         (1 << 0)
#define TRAINING_ADJUST_RATE        (1 << 1)
#define TRAINING_WEIGHT_DECAY       (1 << 2)
#define TRAINING_EPOCH_AS_SEQUENCE  (1 << 3)

#define PSIsRecurrent(o) (o->flags & FLAG_RECURRENT)
#define PSSetRecurrent(o) (o->flags |= FLAG_RECURRENT)
#define PSLDEF(...) ((PSLayerDef *) &((PSLayerDef) {__VA_ARGS__}))
#define PSTRAINOPT(...) \
    ((PSTrainingOptions *) &((PSTrainingOptions) {__VA_ARGS__}))

struct PSNeuralNetwork;
struct PSLayer;
struct PSGradient;
struct PSTrainingOptions;

typedef int      (*PSFeedforwardFunction) (struct PSLayer *layer, ...);
typedef int      (*PSBackpropFunction) (struct PSLayer *layer,
                                        struct PSLayer *previousLayer,
                                        struct PSGradient *layer_gradients,
                                        ...);
typedef void     (*PSGenericLayerCallback) (struct PSLayer *layer);
typedef int      (*PSCopyLayerCallback) (struct PSLayer *, struct PSLayer *);
typedef int      (*PSPretrainLayerFunction) (struct PSLayer *,
                                             PSFloat *training_data,
                                             int data_size);
typedef uint64_t (*PSGetParamCountFunction) (struct PSLayer *layer, int type);

typedef PSFloat  (*PSLossFunction) (PSFloat* x, PSFloat* y, int size,
                                    int onehot_size);
typedef void     (*PSTrainCallback) (struct PSNeuralNetwork *network,
                                     int epoch, int epochs,
                                     PSFloat average_loss,
                                     PSFloat current_loss,
                                     float accuracy, PSFloat *rate,
                                     PSFloat *training_data);
typedef void     (*PSSignalHandler) (int);

typedef struct PSLayerDef {
    PSActivationFunction activation;
    int flags;
    PSFloat dropout;
    int weight_init_mode;
    int bias_init_mode;
    PSFloat init_range;
    PSFloat init_scale;
    int output_depth;
    int output_columns;
    int output_rows;
    int stride;         /* Used by Convolutional and Pooling layers */
    int padding;        /* Used by Convolutional layers */
    int filter_width;   /* Used by Convolutional layers */
    int filter_height;  /* Used by Convolutional layers */
    int pretrained;
    int embedding_type;
    const char *load_from;
    const char *save_pretrained_to;
    PSFloat *training_data;
    int training_data_size;
    struct PSTrainingOptions *pretraining_options;
} PSLayerDef;

typedef struct PSGradient {
    uint64_t bias_count;
    uint64_t weight_count;
    PSFloat *biases;
    PSFloat *weights;
    PSFloat *tmp;
} PSGradient;

typedef enum {
    FullyConnected,
    Convolutional,
    Pooling,
    Recurrent,
    LSTM,
    SoftMax,
    GRU,
    Dropout,
    Embedding
} PSLayerType;

typedef enum {
    NonRecurrent,
    ManyToMany,
    ManyToOne,
    OneToMany
} PSRecurrentNetworkMode;

typedef struct {
    int max_steps;
    int eos;
} PSSequenceStopCriterion;

typedef struct {
    PSRecurrentNetworkMode  mode;
    PSSequenceStopCriterion sequence_stop_criterion;
} PSRecurrentNetworkOptions;

typedef struct PSTrainingOptions {
    int                     epochs;
    PSFloat                 learning_rate;
    int                     batch_size;
    int                     flags;
    PSFloat                 l1_decay;
    PSFloat                 l2_decay;
    PSFloat                 momentum;
    PSFloat                 rho;
    PSFloat                 eps;
    PSFloat                 beta1;
    PSFloat                 beta2;
    PSFloat                 clip;
    PSOptimization          optimization;
    int                     bptt_truncate;
    int                     validate_every_batches;
    int                     max_validation_elements;
    FILE                    *debug_dump_to;
} PSTrainingOptions;

typedef struct {
    int     current_epoch;
    int     current_batch;
    int     current_element;
    int     batch_size;
    time_t  started_at;
    time_t  ended_at;
    int     requested_action;
    FILE    *debug_dump_to;
} PSTrainingInfo;

typedef struct PSNeuron {
    int             index;
    PSFloat         *bias;
    PSFloat         *weights;
    void            *extra;
    struct PSLayer  *layer;
} PSNeuron;

typedef struct PSLayer {
    PSLayerType             type;
    int                     index;
    int                     size;
    int                     weight_types_count;
    PSMatrix                *weights;
    PSFloat                 *biases;
    PSNeuron                **neurons;
    PSFloat                 *states;
    PSFloat                 *delta;
    PSFloat                 *initial_states;
    uint32_t                recurrent_states_count;
    uint32_t                flags;
    int                     onehot_vector_size;
    int                     output_depth;
    int                     output_columns;
    int                     output_rows;
    int                     pretrained;
    void                    *extra;
    void                    *private;
    PSActivationFunction    activate;
    PSActivationFunction    derivative;
    PSFeedforwardFunction   feedforward;
    PSBackpropFunction      backprop;
    PSGenericLayerCallback  on_delete;
    PSCopyLayerCallback     on_copy;
    PSGenericLayerCallback  before_batch_training;
    PSGetParamCountFunction get_param_count;
    PSPretrainLayerFunction pretrain;
    struct PSNeuralNetwork  *network;
    struct PSNeuralNetwork  *pretrainer;
} PSLayer;

typedef struct PSNeuralNetwork {
    const char                  *name;
    int                         size;
    PSLayer                     **layers;
    PSLossFunction              loss;
    uint32_t                    flags;
    uint8_t                     acceleration;
    uint8_t                     status;
    uint32_t                    input_size;
    uint32_t                    output_size;
    PSRecurrentNetworkOptions   *rnn_options;
    PSTrainingInfo              *training;
    PSTrainCallback             onEpochTrained;
    PSTrainCallback             onBatchTrained;
    void                        *context;
} PSNeuralNetwork;

PSNeuralNetwork *PSCreateNetwork(const char* name);
PSNeuralNetwork *PSCloneNetwork(PSNeuralNetwork *network, int layout_only);
int PSLoadNetwork(PSNeuralNetwork *network, const char* filename);
int PSSaveNetwork(PSNeuralNetwork *network, const char* filename);
int PSLoadLayer(PSLayer *layer, const char *filepath);
int PSSaveLayer(PSLayer *layer, const char *filepath, int save_definition);
PSLayer *PSAddLayer(PSNeuralNetwork *network, PSLayerType type, int size,
                    PSLayerDef *layer_def);
PSLayer *PSAddConvolutionalLayer(PSNeuralNetwork *network, PSLayerDef *ldef);
PSLayer *PSAddPoolingLayer(PSNeuralNetwork *network, PSLayerDef *ldef);
PSLayer *PSGetFirstRecurrentLayer(PSNeuralNetwork *network);
PSLayer *PSGetLastRecurrentLayer(PSNeuralNetwork *network);
int PSGetOneHotLayerVectorSize(PSLayer *layer);
uint64_t PSGetLayerParametersCount(PSLayer *layer, int param_type);
PSLayer *PSGetPreviousLayer(PSLayer *layer);
PSLayer *PSGetNextLayer(PSLayer *layer);
int PSGetLayerInputSize(PSLayer *layer);
uint64_t PSGetLayerInputWeightsCount(PSLayer *layer, int per_neuron);
PSFloat *PSGetNeuronInputWeights(PSNeuron *neuron);

int PSResetLayerRecurrentStates(PSLayer *layer, uint32_t steps,
                                int retain_previous);
int PSResetNetworkRecurrentStates(PSNeuralNetwork *network, uint32_t steps,
                                int retain_previous);
PSFloat PSGetState(PSLayer *layer, int index, ...);
PSFloat *PSGetStates(PSLayer *layer, ...);
PSFloat PSGetNeuronState(PSNeuron *neuron, ...);
int PSSetState(PSLayer *layer, PSFloat state, int index, ...);
int PSSetNeuronState(PSNeuron *neuron, double state, ...);
int PSCheckNetwork(PSNeuralNetwork *network);
int PSFeedforward(PSNeuralNetwork *network, PSFloat *values);
int PSClassify(PSNeuralNetwork *network, PSFloat *values);
int PSFindLayerMaxState(PSLayer *layer, PSFloat *max_p, int *index_p,...);

void PSResetTransposedWeights(PSNeuralNetwork *network);
void PSDeleteNetwork(PSNeuralNetwork *network);
void PSDeleteLayer(PSLayer *layer);
void PSDeleteNeuron(PSNeuron *neuron);
void PSDeleteGradients(PSGradient *gradients);
void PSDeleteNetworkGradients(PSGradient **gradients, PSNeuralNetwork *net);
void PSTrain(PSNeuralNetwork *network,
             PSFloat *training_data,
             int data_size,
             PSFloat *test_data,
             int test_size,
             PSTrainingOptions *options);
void PSPauseTraining(PSNeuralNetwork *network);
void PSAbortTraining(PSNeuralNetwork *network);
float PSTest(PSNeuralNetwork *network, PSFloat *test_data, int data_size);
int PSCheckNetwork(PSNeuralNetwork *network);
/* int arrayMaxIndex(PSFloat *array, int len); */
char *PSGetLabelForType(PSLayerType type);
char *PSGetLayerTypeLabel(PSLayer *layer);
int PSIsNetworkBuilt(PSNeuralNetwork *network);
int PSBuildNetwork(PSNeuralNetwork *network);
int PSRebuildNetwork(PSNeuralNetwork *network);
char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
void PSPrintNetworkInfo(PSNeuralNetwork *network);
int PSDumpNetworkStates(PSNeuralNetwork *network, const char* filename);
int PSDumpNetworkDeltas(PSNeuralNetwork *network, const char* filename);
void PSSetDefaultRNNOptions(PSRecurrentNetworkOptions *opts);
void PSSetDefaultTrainingOptions(PSTrainingOptions *options);
PSRecurrentNetworkMode PSGetRecurrentNetworkMode(PSNeuralNetwork *network);
int PSSetRecurrentNetworkMode(
    PSNeuralNetwork *network, PSRecurrentNetworkMode mode
);
PSLayer *PSGetFirstRecurrentLayer(PSNeuralNetwork *network);
PSLayer *PSGetLastRecurrentLayer(PSNeuralNetwork *network);

/*  Loss functions */

PSFloat PSQuadraticLoss(PSFloat *x, PSFloat *y, int size, int onehot_size);
PSFloat PSCrossEntropyLoss(PSFloat *x, PSFloat *y, int size, int onehot_size);

/* Miscellaneous functions */

void PSHandleSignals(PSSignalHandler shutdown_handler);
size_t PSIterateLossFunctions(
    void ( *callback) (const char *name, PSLossFunction func)
);

#endif /*  __PSYC_H */

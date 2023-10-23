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

#define PSYC_VERSION      "0.9.3"

#define LAYER_TYPES     14

#define DEFAULT_RHO     0.95
#define DEFAULT_BETA1   0.9
#define DEFAULT_BETA2   0.999
#ifdef PS_DOUBLE_PRECISION
#define DEFAULT_EPS     1e-8
#else
#define DEFAULT_EPS     1e-7
#endif
#define PS_MAX_MEMORY_GRADIENTS 2

#define DEFAULT_RECURRENT_MODE ManyToMany
#define MAX_SEQUENCE_LENGTH 10
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
#define INIT_MODE_VALUE     3

#define TRAINING_PHASE_FORWARD      1
#define TRAINING_PHASE_BACKPROP     2
#define TRAINING_PHASE_UPDATE_GRAD  3

#define PS_NULL_VALUE PSFLOAT_MIN

/* Layer/Model Flags */
#define FLAG_NONE 0
#define FLAG_RECURRENT      (1 << 0)
#define FLAG_ONEHOT         (1 << 1)
#define FLAG_ACCEL_DISABLED (1 << 2) /* Formerly FLAG_AVX_DISABLED (< v0.4):
                                      * not used anymore, but disables any
                                      * acceleration when loading models
                                      * generated with versiob < 0.4 */
#define FLAG_NO_BIAS        (1 << 3)
#define FLAG_PRETRAINER     (1 << 4)
#define FLAG_NON_TRAINABLE  (1 << 5)
#define FLAG_USE_SEQUENCES  (1 << 6)
#define FLAG_AUTOREGRESSION (1 << 7)
#define FLAG_RANDREGRESSION (1 << 8)
#define FLAG_SELF_ATTENTION (1 << 16)

/* Training Flags */
#define TRAINING_NO_SHUFFLE             (1 << 0)
#define TRAINING_ADJUST_RATE            (1 << 1)
#define TRAINING_WEIGHT_DECAY           (1 << 2)
#define TRAINING_EPOCH_AS_SEQUENCE      (1 << 3)
#define TRAINING_FLAG_SELFSUPERVISED    (1 << 4)
#define TRAINING_FLAG_AUTOREGRESSION    (1 << 7)
#define TRAINING_FLAG_TEACHER_FORCING   (1 << 8)
#define TRAINING_FLAG_SEQ2SEQ           (1 << 9)

/* I/O options */

#define PS_IO_SAVE_DEFINITION           (1 << 0)
#define PS_IO_BINARY_MODE               (1 << 1)

#define PSIsRecurrent(o) (o->flags & FLAG_RECURRENT)
#define PSSetRecurrent(o) (o->flags |= FLAG_RECURRENT)
#define PSUseSequences(o) \
    (o->flags & (FLAG_RECURRENT | FLAG_USE_SEQUENCES))
#define PSHandleSequenceAtOnce(o) (PSUseSequences(o) && !PSIsRecurrent(o))
#define PSIsModelChain(model) (model->previous != NULL || model->next != NULL)
#define PSLDEF(...) ((PSLayerDef *) &((PSLayerDef) {__VA_ARGS__}))
#define PSTRAINOPT(...) \
    ((PSTrainingOptions *) &((PSTrainingOptions) {__VA_ARGS__}))
#define PSDisablePretraining(layer) (layer->pretrain = NULL)
#define PSIsModelTraining(model) (PSModelGetStatus(model) == STATUS_TRAINING)

struct PSModel;
struct PSLayer;
struct PSGradient;
struct PSTrainingOptions;

typedef int      (*PSForwardFunction)  (struct PSLayer *layer, ...);
typedef int      (*PSBackpropFunction) (struct PSLayer *layer,
                                        struct PSLayer *previousLayer,
                                        struct PSGradient *layer_gradients,
                                        ...);
typedef void     (*PSGenericLayerCallback) (struct PSLayer *layer);
typedef int      (*PSBooleanLayerCallback) (struct PSLayer *layer);
typedef int      (*PSCopyLayerCallback) (struct PSLayer *, struct PSLayer *);
typedef int      (*PSPretrainLayerFunction) (struct PSLayer *,
                                             PSFloat *training_data,
                                             int data_size);
typedef uint64_t (*PSGetParamCountFunction) (struct PSLayer *layer, int type);
typedef int      (*PSInitStatesFunc) (struct PSLayer *layer, uint32_t steps,
                                      int retain_previous);
typedef int      (*PSResizeStatesFunc) (struct PSLayer *layer, uint32_t steps,
                                        uint32_t previous_steps);
typedef PSFloat  (*PSLossFunction) (PSFloat* x, PSFloat* y, int size,
                                    int onehot_size);
typedef void     (*PSTrainCallback) (struct PSModel *model,
                                     int epoch, int epochs,
                                     PSFloat average_loss,
                                     PSFloat current_loss,
                                     float accuracy, PSFloat *rate,
                                     PSFloat *training_data);
typedef int      (*PSLinkDataRetriever) (struct PSLayer *layer);
typedef int      (*PSBeforeForwardCallback) (struct PSModel *model,
                                             PSFloat *inputs,
                                             int seqlen, int backprop,
                                             void *opts);
typedef int      (*PSBeforeBackpropCallback) (struct PSModel *model,
                                              PSFloat *y,
                                              struct PSTrainingOptions *opts,
                                              struct PSGradient **gradients);
typedef void     (*PSTrainingProgressFunc) (struct PSModel *model,
                                            int status, int epochs,
                                            int batches, PSFloat *loss,
                                            PSFloat *accuracy,
                                            time_t *elapsed,
                                            int validating_current,
                                            int validating_tot);
typedef void     (*PSSignalHandler) (int);

typedef struct PSLayerDef {
    PSActivationFunction activation;
    int flags;
    int weight_init_mode;
    int bias_init_mode;
    PSFloat init_range;
    PSFloat init_scale;
    PSFloat init_value;
    int output_depth;
    /* Convolutional and Pooling layers hyperparamaters */
    int output_columns;
    int output_rows;
    int stride;         /* Used by Convolutional and Pooling layers */
    int padding;        /* Used by Convolutional layers */
    int filter_width;   /* Used by Convolutional layers */
    int filter_height;  /* Used by Convolutional layers */
    /* Embedding layers options */
    int embedding_type;
    /* Dropout layers hyperparamaters */
    PSFloat dropout;
    /* Normalization layers hyperparamaters */
    PSFloat epsilon;
    /* Pretrainable layers options */
    int pretrained;
    const char *load_from;
    const char *save_pretrained_to;
    PSFloat *training_data;
    int training_data_size;
    struct PSTrainingOptions *pretraining_options;
    /* Attention Layer */
    int attention_type;
    struct PSLayer *query_provider;
    struct PSLayer *keys_provider;
    struct PSLayer *values_provider;
    PSFloat attention_scale;
    int attention_heads;
    int causal_attention;
    int self_attention;
    int trainable_parameters;
    /* OperatorLayer */
    int operator;
    int providers_count;
    struct PSLayer **providers;
    /* PositionalEncoding */
    int positional_initial_capacity;
    int positional_base;
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
    Embedding,
    Normalization,
    Attention,
    OperatorLayer,
    Linear,
    PositionalEncoding
} PSLayerType;

typedef enum {
    NonRecurrent,
    ManyToMany,
    ManyToOne,
    OneToMany
} PSRecurrentNetworkMode;

typedef struct PSSequenceSettings {
    int         max_length;
    PSFloat     *start;
    int         end;
} PSSequenceSettings;

typedef struct PSForwardOptions {
    int                 flags;
    PSSequenceSettings  *sequence_settings;
} PSForwardOptions;

typedef struct PSTrainingOptions {
    int                         epochs;
    PSFloat                     learning_rate;
    int                         batch_size;
    int                         flags;
    PSFloat                     l1_decay;
    PSFloat                     l2_decay;
    PSFloat                     momentum;
    PSFloat                     rho;
    PSFloat                     eps;
    PSFloat                     beta1;
    PSFloat                     beta2;
    PSFloat                     clip;
    PSOptimization              optimization;
    int                         bptt_truncate;
    int                         validate_every_batches;
    int                         max_validation_elements;
    PSTrainingProgressFunc      printProgress;
    FILE                        *debug_dump_to;
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
    PSLayerType                 type;
    int                         index;
    int                         size;
    int                         weight_types;
    PSMatrix                    *weights;
    PSFloat                     *biases;
    PSMatrix                    states;
    PSMatrix                    delta;
    PSFloat                     *initial_states;
    uint32_t                    flags;
    int                         onehot_vector_size;
    int                         output_depth;
    int                         output_columns;
    int                         output_rows;
    int                         pretrained;
    void                        *extra;
    void                        *private;
    PSForwardFunction           forward;
    PSBackpropFunction          backprop;
    PSActivationFunction        activate;
    PSActivationFunction        derivative;
    PSBooleanLayerCallback      build;
    PSGenericLayerCallback      onDelete;
    PSCopyLayerCallback         onCopy;
    PSGenericLayerCallback      beforeBatchTraining;
    PSGetParamCountFunction     getParamCount;
    PSInitStatesFunc            onStatesInit;
    PSResizeStatesFunc          onStatesResize;
    PSPretrainLayerFunction     pretrain;
    PSLinkDataRetriever         getInputFromLink;
    struct PSModel      *model;
    struct PSModel      *pretrainer;
} PSLayer;

typedef struct PSModelLink {
    PSLayer *layer;
    PSLayer *previous_layer;
} PSModelLink;

typedef struct PSModel {
    const char                  *name;
    int                         size;
    int                         index;
    PSLayer                     **layers;
    PSLossFunction              loss;
    uint32_t                    flags;
    uint16_t                    acceleration;
    uint8_t                     status;
    uint32_t                    input_size;
    uint32_t                    output_size;
    struct PSModel              *previous;
    struct PSModel              *next;
    PSModelLink                 *previous_model_link;
    PSSequenceSettings          sequence_settings;
    PSRecurrentNetworkMode      rnn_mode;
    PSTrainingInfo              *training;
    PSBeforeForwardCallback     beforeForward;
    PSBeforeBackpropCallback    beforeBackprop;
    PSTrainCallback             onEpochTrained;
    PSTrainCallback             onBatchTrained;
    void                        *context;
} PSModel;

PSModel *PSModelCreate(const char* name);
int PSAddModel(PSModel *parent, PSModel *model, PSModelLink *link);
PSModel *PSModelClone(PSModel *model, int layout_only);
int PSModelLoad(PSModel *model, const char* filename);
int PSModelSave(PSModel *model, const char* filename);
int PSLoadLayer(PSLayer *layer, const char *filepath);
int PSSaveLayer(PSLayer *layer, const char *filepath, int opts);
void PSModelSetStatus(PSModel *model, int status, int *old);
int PSModelGetStatus(PSModel *model);
PSLayer *PSAddLayer(PSModel *model, PSLayerType type, int size,
                    PSLayerDef *layer_def);
PSLayer *PSAddConvolutionalLayer(PSModel *model, PSLayerDef *ldef);
PSLayer *PSAddPoolingLayer(PSModel *model, PSLayerDef *ldef);
int PSGetOneHotLayerVectorSize(PSLayer *layer);
uint64_t PSGetLayerParametersCount(PSLayer *layer, int param_type);
PSLayer *PSGetPreviousLayer(PSLayer *layer);
PSLayer *PSGetNextLayer(PSLayer *layer);
PSLayer *PSGetOutputLayer(PSModel *model);
PSLayer *PSGetLayerByIndex(PSModel *model, int layer_index, int model_index);
int PSGetLayerInputSize(PSLayer *layer);
uint64_t PSGetLayerInputWeightsCount(PSLayer *layer, int per_neuron);
PSFloat *PSGetNeuronInputWeights(PSNeuron *neuron);

int PSResetLayerStateSequence(PSLayer *layer, uint32_t steps,
                              int retain_previous);
int PSResetModelStateSequences(PSModel *model, uint32_t steps,
                               int retain_previous);
PSFloat PSGetState(PSLayer *layer, int index, ...);
PSFloat *PSGetStates(PSLayer *layer, ...);
PSFloat PSGetNeuronState(PSNeuron *neuron, ...);
PSFloat *PSGetOutputs(PSLayer *layer);
int PSSetState(PSLayer *layer, PSFloat state, int index, ...);
int PSSetNeuronState(PSNeuron *neuron, double state, ...);
PSNeuron *PSGetNeuron(PSLayer *layer, int index, PSNeuron *neuron);
int PSStateSequenceLength(PSLayer *layer);
int PSForward(PSModel *model, PSFloat *values);
int PSAutoregression(PSModel *model, PSFloat *inputs,
                     int randomized, PSSequenceSettings *sequence_settings);
int PSClassify(PSModel *model, PSFloat *values);
int PSFindLayerMaxState(PSLayer *layer, PSFloat *max_p, int *index_p,...);

void PSResetTransposedWeights(PSModel *model);
void PSModelFree(PSModel *model);
void PSDeleteLayer(PSLayer *layer);
void PSDeleteNeuron(PSNeuron *neuron);
void PSDeleteGradient(PSGradient *gradient);
void PSDeleteModelGradients(PSGradient **gradients, PSModel *net);
void PSDeleteGradientsChain(PSGradient ***gradients, PSModel *model);
void PSTrain(PSModel *model,
             PSFloat *training_data,
             int data_size,
             PSFloat *test_data,
             int test_size,
             PSTrainingOptions *options);
void PSPauseTraining(PSModel *model);
void PSAbortTraining(PSModel *model);
float PSTest(PSModel *model, PSFloat *test_data, int data_size,
             PSTrainingOptions *options);
int PSModelCheck(PSModel *model);
/* int arrayMaxIndex(PSFloat *array, int len); */
char *PSGetLabelForType(PSLayerType type);
char *PSGetLayerTypeLabel(PSLayer *layer);
int PSModelIsBuilt(PSModel *model);
int PSModelBuild(PSModel *model);
int PSModelRebuild(PSModel *model);
void PSModelPrintInfo(PSModel *model);
int PSModelDumpStates(PSModel *model, const char* filename);
int PSModelDumpDeltas(PSModel *model, const char* filename);
void PSSetDefaultTrainingOptions(PSTrainingOptions *options);
int PSSetRecurrentNetworkMode(PSModel *model, PSRecurrentNetworkMode mode);
PSLayer *PSGetFirstRecurrentLayer(PSModel *model);
PSLayer *PSGetLastRecurrentLayer(PSModel *model);
int PSModelChainLength(PSModel *model);
PSModel *PSGetModelAtIndex(PSModel *entrypoint, int index);
PSModel *PSModelChainHead(PSModel *model);
PSModel *PSModelChainTail(PSModel *model);
int PSModelChainContains(PSModel *chain, PSModel *model);

/*  Loss functions */

PSFloat PSQuadraticLoss(PSFloat *x, PSFloat *y, int size, int onehot_size);
PSFloat PSCrossEntropyLoss(PSFloat *x, PSFloat *y, int size, int onehot_size);

/* Training progress logging functions */
void PSTrainingProgressBar(PSModel *model, int status, int epochs,
                           int batches, PSFloat *loss, PSFloat *accuracy,
                           time_t *elapsed, int validating_current,
                           int validating_tot);

/* Miscellaneous functions */

void PSHandleSignals(PSSignalHandler shutdown_handler);
size_t PSIterateLossFunctions(
    void ( *callback) (const char *name, PSLossFunction func)
);

/* Kept type name used in older version since I still love it :)
 * (and it also sounds more 'Psyc-y') */
typedef PSModel PSNeuralNetwork;

#endif /*  __PSYC_H */

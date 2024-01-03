/*
 * Copyright (C) 2016-2024 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#ifndef __PS_LEGACY_H__
#define __PS_LEGACY_H__

/* Maths */
#define PSSumVectors            PSAddVectors
#define PSSumVectorElements     PSVectorReduceSum
#define PSSumVectorScalar       PSAddVectorScalar

/* Model/Neural Netwrok */

#define PSIsNetworkChain(model)             PSIsModelChain(model)
#define PSGetNetworkChainLength(model)      PSModelChainLength(model)
#define PSGetNetworkChainHead(model)        PSModelChainHead(model)
#define PSGetNetworkChainTail(model)        PSModelChainTail(model)
#define PSNetworkChainContains(chain, model) \
    PSModelChainContains(chain, model)
#define PSGetNetworkAtIndex(entrypoint,idx) PSGetModelAtIndex(entrypoint,idx)
#define PSIsNetworkTraining(model)          PSIsModelTraining(model)
#define PSCreateNetwork(name)               PSModelCreate(name)
#define PSAddNetwork(parent, model, link)   PSAddModel(parent, model, link)
#define PSCloneNetwork(model, layout_only)  PSModelClone(model, layout_only)
#define PSLoadNetwork(model, filename)      PSModelLoad(model, filename)
#define PSSaveNetwork(model, filename)      PSModelSave(model, filename)
#define PSSetNetworkStatus(model, status, old) \
    PSModelSetStatus(model, status, old)
#define PSGetNetworkStatus(model)           PSModelGetStatus(model)
#define PSResetNetworkStateSequences(model, steps, retain_previous) \
    PSResetModelStateSequences(model, steps, retain_previous)
#define PSDeleteNetwork(model)              PSModelFree(model)
#define PSDeleteNetworkGradients(gradients, model) \
    PSDeleteModelGradients(gradients, model)
#define PSCheckNetwork(model)               PSModelCheck(model)
#define PSIsNetworkBuilt(model)             PSModelIsBuilt(model)
#define PSBuildNetwork(model)               PSModelBuild(model)
#define PSRebuildNetwork(model)             PSModelRebuild(model)
#define PSPrintNetworkInfo(model)           PSModelPrintInfo(model)
#define PSDumpNetworkStates(model, fname)   PSModelDumpStates(model, fname)
#define PSDumpNetworkDeltas(model, fname)   PSModelDumpDeltas(model, fname)
#define PSMatrixDelete(matrix)              PSMatrixFree(matrix)
#define PSDictRelease(dict)                 PSDictFree(dict)
#define PSDictDelete(dict, key)             PSDictRemove(dict, key)
#define PSVocabularyRelease(vocab)          PSVocabularyFree(vocab)
#define PSDeleteLayer(layer)                PSLayerFree(layer)
#define PSLoadLayer(layer, path)            PSLayerLoad(layer, path)
#define PSSaveLayer(layer, path, opts)      PSLayerSave(layer, path, opts)
#define PSGetStates(layer, ...)             PSLayerStates(layer, __VA_ARGS__)
#define PSGetOutputs(layer)                 PSLayerOutputs(layer)
#define PSACFEnabled(acceleration)          PSACFEnabled(acceleration)

#define FLAG_LOG_COLORS                     PS_FLAG_LOG_COLORS
#define DATA_TYPE_TRAINING                  PS_DATA_TYPE_TRAINING
#define DATA_TYPE_TEST                      PS_DATA_TYPE_TEST
#define MNIST_INPUT_SIZE                    PS_MNIST_INPUT_SIZE
#define CIFAR_IMAGE_SIZE                    PS_CIFAR_IMAGE_SIZE
#define LAYER_TYPES                         PS_LAYER_TYPES
#define DEFAULT_RHO                         PS_DEFAULT_RHO
#define DEFAULT_BETA1                       PS_DEFAULT_BETA1
#define DEFAULT_BETA2                       PS_DEFAULT_BETA2
#define DEFAULT_EPS                         PS_DEFAULT_EPS
#define DEFAULT_RECURRENT_MODE              PS_DEFAULT_RECURRENT_MODE
#define MAX_SEQUENCE_LENGTH                 PS_MAX_SEQUENCE_LENGTH
#define DEFAULT_EOS_INDEX                   PS_DEFAULT_EOS_INDEX
#define STATUS_UNTRAINED                    PS_STATUS_UNTRAINED
#define STATUS_ERROR                        PS_STATUS_ERROR
#define STATUS_PAUSED                       PS_STATUS_PAUSED
#define STATUS_ABORTED                      PS_STATUS_ABORTED
#define STATUS_VALIDATING                   PS_STATUS_VALIDATING
#define STATUS_PRETRAINING                  PS_STATUS_PRETRAINING
#define ACTION_NONE                         PS_ACTION_NONE
#define ACTION_PAUSE                        PS_ACTION_PAUSE
#define ACTION_ABORT                        PS_ACTION_ABORT
#define PARAM_TYPE_BIAS                     PS_PARAM_BIAS
#define PARAM_TYPE_WEIGHT                   PS_PARAM_WEIGHT
#define INIT_MODE_AUTO                      PS_INIT_MODE_AUTO
#define INIT_MODE_RAND                      PS_INIT_MODE_RAND
#define INIT_MODE_ZERO                      PS_INIT_MODE_ZERO
#define INIT_MODE_VALUE                     PS_INIT_MODE_VALUE
#define TRAINING_PHASE_FORWARD              PS_TRAINING_PHASE_FORWARD
#define TRAINING_PHASE_BACKPROP             PS_TRAINING_PHASE_BACKPROP
#define TRAINING_PHASE_UPDATE_GRAD          PS_TRAINING_PHASE_UPDATE_GRAD
#define FLAG_NONE                           PS_FLAG_NONE
#define FLAG_RECURRENT                      PS_FLAG_RECURRENT
#define FLAG_ONEHOT                         PS_FLAG_ONEHOT
#define FLAG_ACCEL_DISABLED                 PS_FLAG_ACCEL_DISABLED
#define FLAG_NO_BIAS                        PS_FLAG_NO_BIAS
#define FLAG_PRETRAINER                     PS_FLAG_PRETRAINER
#define FLAG_NON_TRAINABLE                  PS_FLAG_NON_TRAINABLE
#define FLAG_USE_SEQUENCES                  PS_FLAG_USE_SEQUENCES
#define FLAG_AUTOREGRESSION                 PS_FLAG_AUTOREGRESSION
#define FLAG_RANDREGRESSION                 PS_FLAG_RANDREGRESSION
#define FLAG_SELF_ATTENTION                 PS_FLAG_SELF_ATTENTION
#define TRAINING_NO_SHUFFLE                 PS_TRAINING_NO_SHUFFLE
#define TRAINING_ADJUST_RATE                PS_TRAINING_ADJUST_RATE
#define TRAINING_WEIGHT_DECAY               PS_TRAINING_WEIGHT_DECAY
#define TRAINING_EPOCH_AS_SEQUENCE          PS_TRAINING_EPOCH_AS_SEQUENCE
#define TRAINING_FLAG_SELFSUPERVISED        PS_TRAINING_FLAG_SELFSUPERVISED
#define TRAINING_FLAG_AUTOREGRESSION        PS_TRAINING_FLAG_AUTOREGRESSION
#define TRAINING_FLAG_TEACHER_FORCING       PS_TRAINING_FLAG_TEACHER_FORCING
#define TRAINING_FLAG_SEQ2SEQ               PS_TRAINING_FLAG_SEQ2SEQ
#define OPT_TIME_LONG                       PS_OPT_TIME_LONG
#define OPT_TIME_FULL                       PS_OPT_TIME_FULL
#define OPT_TIME_HUMAN                      PS_OPT_TIME_HUMAN
#define PSAcceleration_ACF                  PSAcceleration_Accelerate

#define Recurrent                           RNNLayer

typedef PSModelLink PSNeuralNetworkLink;

#endif /* __PS_LEGACY_H__ */

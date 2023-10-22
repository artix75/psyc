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

typedef PSModelLink PSNeuralNetworkLink;

#endif /* __PS_LEGACY_H__ */

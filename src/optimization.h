/*
 * Copyright (C) 2016-present Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#ifndef __OPTIMIZATION_H
#define __OPTIMIZATION_H

#include "types.h"
struct PSTrainingOptions;

#define PSDefaultOptimization PSSGDOptimization

typedef int (*PSOptimization) (PSFloat *params, PSFloat *grads,
                               PSFloat *mgrads, PSFloat *xgrads,
                               PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp,
                               PSFloat rate, PSFloat momentum,
                               long len, int acceleration, long iteration,
                               struct PSTrainingOptions *options);

int PSSGDOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                      PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                      PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                      long len, int acceleration, long iteration,
                      struct PSTrainingOptions *options);

int PSNesterovOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                           PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                           PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                           long len, int acceleration, long iteration,
                           struct PSTrainingOptions *options);

int PSAdaDeltaOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                           PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                           PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                           long len, int acceleration, long iteration,
                           struct PSTrainingOptions *options);

int PSWindowGradOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                             PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                             PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                             long len, int acceleration, long iteration,
                             struct PSTrainingOptions *options);

int PSAdaGradOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                          PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                          PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                          long len, int acceleration, long iteration,
                          struct PSTrainingOptions *options);

int PSRMSPropOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                          PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                          PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                          long len, int acceleration, long iteration,
                          struct PSTrainingOptions *options);

int PSAdamOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                       PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                       PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                       long len, int acceleration, long iteration,
                       struct PSTrainingOptions *options);

int PSLRegularization(PSFloat l1, PSFloat l2, PSFloat *weights,
                      PSFloat *wgradients, PSFloat *tmp, long len,
                      PSFloat *l1_loss, PSFloat *l2_loss,
                      int batches, int weight_decay, int acceleration);

#endif /* __OPTIMIZATION_H */

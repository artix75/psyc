/*
 * Copyright (C) 2016-2022 Fabio Nicotra <artix2 at gmail dot com>.
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

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "optimization.h"
#include "config.h"
#include "maths.h"
#include "log.h"
#include "psyc.h"

#define UNUSED(V) ((void) V)

int PSDefaultOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                          PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                          PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                          uint64_t len, int acceleration, int iteration,
                          PSTrainingOptions *options)
{
    UNUSED(xgrads);
    UNUSED(tmp);
    UNUSED(mtmp);
    UNUSED(xtmp);
    UNUSED(iteration);
    UNUSED(options);
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    int has_momentum = (momentum != 0);
    if (has_momentum && mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory for momentum");
        return 0;
    }
    if (acceleration == PSAcceleration_None) {
        if (has_momentum) {
            for (uint64_t i = 0; i < len; i++) {
                mgrads[i] = momentum * mgrads[i] - rate * grads[i];
                params[i] += mgrads[i];
            }
        } else {
            for (uint64_t i = 0; i < len; i++) params[i] += -(rate * grads[i]);
        }
    } else {
        PSMathOpts mopts = {.acceleration = acceleration};
        if (has_momentum) {
            assert(mgrads != NULL);
            mopts.store_mode = MATHS_STORE_MODE_NORM;
            PSMultiplyVectorScalar(mgrads, momentum, mgrads, len, &mopts);
            mopts.store_mode = MATHS_STORE_MODE_SUB;
            PSMultiplyVectorScalar(grads, rate, mgrads, len, &mopts);
            mopts.store_mode = MATHS_STORE_MODE_NORM;
            PSSumVectors(params, mgrads, params, len, &mopts);
        } else {
            mopts.store_mode = MATHS_STORE_MODE_ADD;
            PSMultiplyVectorScalar(grads, -rate, params, len, &mopts);
        }
    }
    return 1;
}

int PSNesterovOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                           PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                           PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                           uint64_t len, int acceleration, int iteration,
                           PSTrainingOptions *options)
{
    UNUSED(xgrads);
    UNUSED(mtmp);
    UNUSED(xtmp);
    UNUSED(iteration);
    UNUSED(options);
    if (mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory");
        return 0;
    }
    PSFloat *tmpalloc = NULL;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            PSFloat dx = mgrads[i];
            mgrads[i] = mgrads[i] * momentum + rate * grads[i];
            dx = momentum * dx - (1.0 + momentum) * mgrads[i];
            params[i] += dx;
        }
    } else {
        if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        PSMathOpts mopts = {.acceleration = acceleration};
        memcpy(tmp, mgrads, len * sizeof(mgrads));
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectorScalar(mgrads, momentum, mgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(grads, rate, mgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectorScalar(tmp, momentum, tmp, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_SUB;
        PSMultiplyVectorScalar(mgrads, (1.0 + momentum), tmp, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSSumVectors(params, tmp, params, len, &mopts);
    }
    if (tmpalloc != NULL) free(tmpalloc);
    return 1;
}

int PSAdaDeltaOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                           PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                           PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                           uint64_t len, int acceleration, int iteration,
                           PSTrainingOptions *options)
{
    UNUSED(rate);
    UNUSED(momentum);
    UNUSED(iteration);
    if (mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory");
        return 0;
    }
    if (xgrads == NULL) {
        PSErr(__func__, "argument `xgrads` is mandatory");
        return 0;
    }
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    PSFloat rho = options->rho;
    PSFloat eps = options->eps;
    PSFloat *tmpalloc = NULL, *tmpalloc1 = NULL, *tmpalloc2 = NULL;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    int success = 1;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            mgrads[i] = rho * mgrads[i] + (1 - rho) * grads[i] * grads[i];
            PSFloat dx = - (
                PSSqrt((xgrads[i] + eps) / (mgrads[i] + eps)) * grads[i]
            );
            xgrads[i] = rho * xgrads[i] + (1 - rho) * dx * dx;
            params[i] += dx;
        }
    } else {
        if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        if (mtmp == NULL) {
            tmpalloc1 = malloc(len * sizeof(PSFloat));
            if (tmpalloc1 == NULL) {
                PSPrintMemoryErrorMsg();
                success = 0;
                goto final;
            }
            mtmp = tmpalloc1;
        }
        if (xtmp == NULL) {
            tmpalloc2 = malloc(len * sizeof(PSFloat));
            if (tmpalloc2 == NULL) {
                PSPrintMemoryErrorMsg();
                success = 0;
                goto final;
            }
            xtmp = tmpalloc2;
        }
        PSMathOpts mopts = {.acceleration = acceleration};
        PSFloat *dx = tmp, *dxsqr = malloc(len  * sizeof(PSFloat));
        if (dxsqr == NULL) {
            PSPrintMemoryErrorMsg();
            success = 0;
            goto final;
        }
        /**mgrads = rho * *mgrads + (1 - rho) * grad * grad;*/
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectors(grads, grads, tmp, len, &mopts);
        PSMultiplyVectorScalar(mgrads, rho, mgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(tmp, (1 - rho), mgrads, len, &mopts);
        /*dx = - PSSqrt((*xgrads + eps) / (*mgrads + eps)) * grad;*/
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSSumVectorScalar(xgrads, eps, xtmp, len, &mopts);
        PSSumVectorScalar(mgrads, eps, mtmp, len, &mopts);
        PSDivideVectors(xtmp, mtmp, dx, len, &mopts);
        PSVectorSqrt(tmp, tmp, len, &mopts);
        PSMultiplyVectors(tmp, grads, tmp, len, &mopts);
        PSVectorNeg(tmp, dx, len, &mopts);
        /* *xgrads = rho * *xgrads + (1 - rho) * dx *dx; */
        PSMultiplyVectors(dx, dx, dxsqr, len, &mopts);
        PSMultiplyVectorScalar(xgrads, rho, xgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(dxsqr, (1 - rho), xgrads, len, &mopts);
        /* param + dx */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSSumVectors(params, dx, params, len, &mopts);
        free(dxsqr);
    }
final:
    if (tmpalloc != NULL) free(tmpalloc);
    if (tmpalloc1 != NULL) free(tmpalloc1);
    if (tmpalloc2 != NULL) free(tmpalloc2);
    return success;
}

int PSWindowGradOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                             PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                             PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                             uint64_t len, int acceleration, int iteration,
                             PSTrainingOptions *options)
{
    UNUSED(xgrads);
    UNUSED(mtmp);
    UNUSED(xtmp);
    UNUSED(momentum);
    UNUSED(iteration);
    if (mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory");
        return 0;
    }
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    PSFloat rho = options->rho;
    PSFloat eps = options->eps;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    PSFloat *tmpalloc = NULL;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            mgrads[i] = rho * mgrads[i] + (1 - rho) * grads[i] * grads[i];
            PSFloat dx = - rate / PSSqrt(mgrads[i] + eps) * grads[i];
            params[i] += dx;
        }
    } else {
        if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        PSMathOpts mopts = {.acceleration = acceleration};
        /* m = rho * m + (1 - rho) * grad * grad */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectorScalar(mgrads, rho, mgrads, len, &mopts);
        PSMultiplyVectors(grads, grads, tmp, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(tmp, (1 - rho), mgrads, len, &mopts);
        /* dx = - rate / PSSqrt(mgrads[i] + eps) * grads[i] */
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSSumVectorScalar(mgrads, eps, tmp, len, &mopts);
        PSVectorSqrt(tmp, tmp, len, &mopts);
        PSDivideScalarVector(-rate, tmp, tmp, len, &mopts);
        PSMultiplyVectors(grads, tmp, tmp, len, &mopts);
        /* params[i] += dx */
        PSSumVectors(params, tmp, params, len, &mopts);
    }
    if (tmpalloc != NULL) free(tmpalloc);
    return 1;
}

int PSAdaGradOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                          PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                          PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                          uint64_t len, int acceleration, int iteration,
                          PSTrainingOptions *options)
{
    UNUSED(xgrads);
    UNUSED(mtmp);
    UNUSED(xtmp);
    UNUSED(momentum);
    UNUSED(iteration);
    if (mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory");
        return 0;
    }
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    PSFloat eps = options->eps;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    PSFloat *tmpalloc = NULL;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            mgrads[i] += grads[i] * grads[i];
            PSFloat dx = - rate / PSSqrt(mgrads[i] + eps) * grads[i];
            params[i] += dx;
        }
    } else {
        if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        PSMathOpts mopts = {.acceleration = acceleration};
        /* mgrads[i] += grads[i] * grads[i] */
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectors(grads, grads, mgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_NORM;
        /* dx = - rate / PSSqrt(mgrads[i] + eps) * grads[i] */
        PSSumVectorScalar(mgrads, eps, tmp, len, &mopts);
        PSVectorSqrt(tmp, tmp, len, &mopts);
        PSDivideScalarVector(rate, tmp, tmp, len, &mopts);
        PSVectorNeg(tmp, tmp, len, &mopts);
        PSMultiplyVectors(tmp, grads, tmp, len, &mopts);
        /* params[i] += dx */
        PSSumVectors(params, tmp, params, len, &mopts);
    }
    if (tmpalloc != NULL) free(tmpalloc);
    return 1;
}

int PSAdamOptimization(PSFloat *params, PSFloat *grads, PSFloat *mgrads,
                       PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp,
                       PSFloat *xtmp, PSFloat rate, PSFloat momentum,
                       uint64_t len, int acceleration, int iteration,
                       PSTrainingOptions *options)
{
    UNUSED(tmp);
    UNUSED(mtmp);
    UNUSED(xtmp);
    UNUSED(momentum);
    if (mgrads == NULL) {
        PSErr(__func__, "argument `mgrads` is mandatory");
        return 0;
    }
    if (xgrads == NULL) {
        PSErr(__func__, "argument `xgrads` is mandatory");
        return 0;
    }
    PSTrainingOptions default_opts = {0};
    if (options == NULL) {
        PSSetDefaultTrainingOptions(&default_opts);
        options = &default_opts;
    }
    PSFloat eps = options->eps;
    PSFloat beta1 = options->beta1;
    PSFloat beta2 = options->beta2;
    if (beta1 == 0) {
        PSErr(__func__, "options->beta1 cannot be zero");
        return 0;
    }
    if (beta2 == 0) {
        PSErr(__func__, "options->beta2 cannot be zero");
        return 0;
    }
    PSFloat *tmpalloc = NULL, *tmpalloc1 = NULL, *tmpalloc2 = NULL;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    acceleration = PSAcceleration_None; /* Acceleration disabled */
    int success = 1;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            PSFloat correct1, correct2, dx;
            mgrads[i] = mgrads[i] * beta1 + (1 - beta1) * grads[i];
            xgrads[i] = xgrads[i] * beta2 + (1 - beta2) * grads[i] * grads[i];
            correct1 = mgrads[i] * (1 - PSPow(beta1, iteration));
            correct2 = xgrads[i] * (1 - PSPow(beta2, iteration));
            dx =  - rate *correct1 / (PSSqrt(correct2) + eps);
            params[i] += dx;
        }
    } else {
        /* ACCELERATION DISABLED: MISSING ACCELERATED FUNCTION FOR POW */

        /*if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        PSFloat *correct1 = mtmp, *correct2 = xtmp;
        if (correct1 == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc1 == NULL) {
                PSPrintMemoryErrorMsg();
                success = 0;
                goto final;
            }
            correct1 = tmpalloc1;
        }
        if (correct2 == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc2 == NULL) {
                PSPrintMemoryErrorMsg();
                success = 0;
                goto final;
            }
            correct2 = tmpalloc2;
        }
        PSMathOpts mopts = {.acceleration = acceleration};*/

        /* mgrads[i] * beta1 + (1 - beta1) * grads[i] */
        /*mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectorScalar(mgrads, beta1, mgrads, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(grads, (1 - beta1), mgrads, len, &mopts);*/
        /* xgrads[i] = xgrads[i] * beta2 + (1 - beta2) * grads[i] * grads[i] */
        /*mopts.store_mode = MATHS_STORE_MODE_NORM;
        PSMultiplyVectorScalar(xgrads, beta2, xgrads, len, &mopts);
        PSMultiplyVectors(grads, grads, tmp, len, &mopts);
        mopts.store_mode = MATHS_STORE_MODE_ADD;
        PSMultiplyVectorScalar(tmp, (1 - beta2), xgrads, len, &mopts);*/
        /* correct1 = mgrads[i] * (1 - PSPow(beta1, iteration)) */

    }
    if (tmpalloc != NULL) free(tmpalloc);
    if (tmpalloc1 != NULL) free(tmpalloc1);
    if (tmpalloc2 != NULL) free(tmpalloc2);
    return success;
}

int PSLRegularization(PSFloat l1, PSFloat l2, PSFloat *weights,
                      PSFloat *wgradients, PSFloat *tmp, uint64_t len,
                      PSFloat *l1_loss, PSFloat *l2_loss,
                      int batches, int weight_decay, int acceleration)
{
    if (weights == NULL) return 0;
    if (wgradients == NULL) return 0;
    PSFloat *tmpalloc = NULL;
    if (!PSIsAccelerationAvailable(acceleration))
        acceleration = PSAcceleration_None;
    if (acceleration == PSAcceleration_None) {
        for (uint64_t i = 0; i < len; i++) {
            PSFloat l1_grad = 0.0, l2_grad = 0.0, w = weights[i];
            if (l1 != 0.0) {
                if (weight_decay) weights[i] *= l1;
                else {
                    l1_grad = l1 * (w > 0 ? 1 : -1);
                    if (batches > 1) l1_grad /= batches;
                    wgradients[i] += l1_grad;
                    if (l1_loss != NULL) *l1_loss += PSAbs(w);
                }
            }
            if (l2 != 0.0) {
                if (weight_decay) weights[i] *= l2;
                else {
                    l2_grad = l2 * w;
                    if (batches > 1) l2_grad /= batches;
                    wgradients[i] += l2_grad;
                    if (l2_loss != NULL) *l2_loss += (w * w);
                }
            }
        }
    } else {
        if (tmp == NULL) {
            tmpalloc = malloc(len * sizeof(PSFloat));
            if (tmpalloc == NULL) {
                PSPrintMemoryErrorMsg();
                return 0;
            }
            tmp = tmpalloc;
        }
        PSMathOpts mopts = {.acceleration = acceleration};
        if (l1 != 0.0) {
            if (weight_decay)
                PSMultiplyVectorScalar(weights, l1, weights, len, &mopts);
            else {
                PSFloat *l1_grads = tmp;
                PSVectorMapWithLimit(
                    weights, PSFLOAT_EPS, 1, l1_grads, len, &mopts
                );
                PSMultiplyVectorScalar(l1_grads, l1, l1_grads, len, &mopts);
                if (batches > 1)
                    PSDivideVectorScalar(l1_grads,batches,l1_grads,len,&mopts);
                PSSumVectors(wgradients, l1_grads, wgradients, len, &mopts);
                if (l1_loss != NULL) {
                    PSVectorAbs(weights, tmp, len, &mopts);
                    *l1_loss += PSSumVectorElements(tmp, len, &mopts);
                }
            }
        }
        if (l2 != 0.0) {
            if (weight_decay)
                PSMultiplyVectorScalar(weights, l2, weights, len, &mopts);
            else {
                PSFloat *l2_grads = tmp;
                PSMultiplyVectorScalar(weights, l2, l2_grads, len, &mopts);
                if (batches > 1)
                    PSDivideVectorScalar(l2_grads,batches,l2_grads,len,&mopts);
                PSSumVectors(wgradients, l2_grads, wgradients, len, &mopts);
                if (l2_loss != NULL) {
                    PSMultiplyVectors(weights, weights, tmp, len, &mopts);
                    PSFloat loss = PSSumVectorElements(tmp, len, &mopts);
                    *l2_loss += loss;
                }
            }
        }
    }
    if (tmpalloc != NULL) free(tmpalloc);
    return 1;
}

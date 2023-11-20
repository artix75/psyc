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
#include <math.h>
#include <string.h>
#include <strings.h>
#include <signal.h>
#include <strings.h>
#include <assert.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <libgen.h>
#include <sys/time.h>

#include "test.h"
#include "../psyc.h"
#include "../convolutional.h"
#include "../recurrent.h"
#include "../dropout.h"
#include "../lstm.h"
#include "../gru.h"
#include "../normalization.h"
#include "../operator-layer.h"
#include "../attention.h"
#include "../positional-encoding.h"
#include "../dataset.h"
#include "../maths.h"
#include "../activation.h"
#include "../optimization.h"
#include "../utils.h"
#include "../debug.h"
#include "../log.h"
#ifdef USE_AVX
#include "../avx.h"
#endif

#define MNIST_TEST_SAMPLE_PATH "resources/mnist-test-sample.psdata"
#define PRETRAINED_FULL_MODEL "resources/pretrained.mnist.psmodel"
#define CONVOLUTIONAL_MODEL "resources/cnn.data"
#define CONVOLUTIONAL_TRAINED_MODEL "resources/pretrained.cnn.psmodel"
#define CONVOLUTIONAL_CIFAR_MODEL "resources/cifar-cnn.psmodel"
#define CIFAR_IMAGE_PATH "resources/cifar-image.data"
#define CIFAR_LABEL_PATH "resources/cifar-label.data"
#define RECURRENT_MODEL "resources/rnn.data"
#define NORMALIZATION_MODEL "resources/normalization_nn.psmodel"
#define NORMALIZATION_MODEL_BP "resources/normalization_nn_bp.psmodel"
#define DROPOUT_MODEL "resources/dropout_nn.psmodel"
#define OP_CONCAT_MODEL "resources/concatenate-operator-layer.psmodel"
#define OP_ADD_MODEL "resources/add-operator-layer.psmodel"
#define OP_MUL_MODEL "resources/multiply-operator-layer.psmodel"
#define ENCDEC_BASIC_MODEL "resources/encoder-decoder.basic.psmodel"
#define POSITIONAL_MODEL  "resources/positional_embed.psmodel"
#define ADD_ATTENTION_MODEL "resources/pretrained.additive-attention.psmodel"
#define DOT_ATTENTION_MODEL \
    "resources/pretrained.dot-product-attention.psmodel"
#define MH_ATTENTION_MODEL \
    "resources/pretrained.mh-dot-product-attention.psmodel"
#define MH_CAUSAL_SELFATTENTION_MODEL \
    "resources/pretrained.mh-causal-attention.psmodel"
#define TEST_IMAGE_SIZE 28
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE *TEST_IMAGE_SIZE
#define BP_GRADIENTS_CHECKS 8
#define BP_CONV_GRADIENTS_CHECKS 4
#define CONV_L1F0_BIAS 0.02630446809718423

#define PRETRAINED_MNIST_NLAYERS 3

#define RNN_INPUT_SIZE  4
#define RNN_HIDDEN_SIZE 2
#define RNN_TIMES       4
#define RNN_LEARNING_RATE 0.005

#define LSTM_LEARNING_RATE 0.1
#define LSTM_TIMES 3
#define LSTM_EPOCHS 1
#define LSTM_BATCHES 1

#define getModel(tc) ((PSModel*)(tc->data[0]))
#define getTestData(tc) ((PSFloat*)(tc->data[1]))
#ifdef PS_DOUBLE_PRECISION
#define NORMAL_PRECISION_DEC    6
#define HIGH_PRECISION_DEC      8
#else
#define NORMAL_PRECISION_DEC    4
#define HIGH_PRECISION_DEC      5
#endif
#define getRoundedFloat(d) (getRoundedFloatDec(d, NORMAL_PRECISION_DEC))

#define UNUSED(V) ((void) V)

typedef int (*testMatrixOpFunc)(PSMatrix, PSMatrix, PSMatrix *, PSMathOpts *);

TestCase *FCTests;
TestCase *CNNTests;
TestCase *RNNTests;
TestCase *LSTMTests;
TestCase *GRUTests;
TestCase *NormalizationTests;
TestCase *DropoutTests;
TestCase *ConcatOperatorLayerTests;
TestCase *AddOperatorLayerTests;
TestCase *MulOperatorLayerTests;
TestCase *PositionalEmbedTests;
TestCase *EncoderDecoderTests;
TestCase *AdditiveAttentionTests;
TestCase *DotAttentionTests;
TestCase *MHAttentionTests;
TestCase *MHCausalSelfAttentionTests;

#ifdef USE_AVX
TestCase *AVXTests;
#endif

TestCase *mathsTests;
TestCase *activationTests;
TestCase *optimizationTests;
TestCase *datasetTests;

int genericSetup (TestCase *test_case);
int genericTeardown (TestCase *test_case);
int RNNSetup (TestCase *test_case);
int RNNTeardown (TestCase *test_case);
int LSTMSetup (TestCase *test_case);
int GRUSetup(TestCase *test_case);
int encoderDecoderSetup(TestCase *test_case);
int encoderDecoderTeardown(TestCase *test_case);
int attentionSetup(TestCase *test_case);
int attentionTeardown(TestCase *test_case);

int testGenericClone(TestCase *test_case, Test *test);
int testGenericSave(TestCase *test_case, Test *test);

#ifdef USE_AVX
int testAVXDot(TestCase *test_case, Test *test);
int testAVXSquare(TestCase *test_case, Test *test);
int testAVXMultiplyVal(TestCase *tc, Test *test);
#endif

int testMathsDotProduct(TestCase *tc, Test *test);
int testMathsDot(TestCase *tc, Test *test);
int testMathsVecProd(TestCase *tc, Test *test);
int testMathsSumV(TestCase *tc, Test *test);
int testMathsSubV(TestCase *tc, Test *test);
int testMathsMulV(TestCase *tc, Test *test);
int testMathsDivV(TestCase *tc, Test *test);
int testMathsSumVS(TestCase *tc, Test *test);
int testMathsSubSV(TestCase *tc, Test *test);
int testMathsMulVS(TestCase *tc, Test *test);
int testMathsDivVS(TestCase *tc, Test *test);
int testMathsDivSV(TestCase *tc, Test *test);
int testMathsClip(TestCase *tc, Test *test);
int testMathsThres(TestCase *tc, Test *test);
int testMathsMapLimit(TestCase *tc, Test *test);
int testMathsExp(TestCase *tc, Test *test);
int testMathsPow(TestCase *tc, Test *test);
int testMathsTanh(TestCase *tc, Test *test);
int testMathsSqrt(TestCase *tc, Test *test);
int testMathsNeg(TestCase *tc, Test *test);
int testMathsAbs(TestCase *tc, Test *test);
int testMathsMean(TestCase *tc, Test *test);
int testMathsVar(TestCase *tc, Test *test);
int testMathsStd(TestCase *tc, Test *test);
int testMathVectorFill(TestCase *tc, Test *test);
int testMathsMatMul(TestCase *tc, Test *test);
int testMathsVecToMatrix(TestCase *tc, Test *test);
int testMathsMatrixCopy(TestCase *tc, Test *test);
int testMathsMatrixDup(TestCase *tc, Test *test);
int testMathsMatrixTranspose(TestCase *tc, Test *test);
int testMathsMatrixSwap(TestCase *tc, Test *test);
int testMathsMatrixExpand(TestCase *tc, Test *test);
int testMathsMatrixProduct(TestCase *tc, Test *test);
int testMathsMatrixProductMV(TestCase *tc, Test *test);
int testMathsMatrixProductVM(TestCase *tc, Test *test);
int testMathsMatrixAdd(TestCase *tc, Test *test);
int testMathsMatrixMultiply(TestCase *tc, Test *test);
int testMathsMatrixSubtract(TestCase *tc, Test *test);
int testMathsMatrixDivide(TestCase *tc, Test *test);

int testActSigmoid(TestCase *tc, Test *test);
int testActSigmoidDeriv(TestCase *tc, Test *test);
int testActTanh(TestCase *tc, Test *test);
int testActTanhDeriv(TestCase *tc, Test *test);
int testActRelu(TestCase *tc, Test *test);
int testActReluDeriv(TestCase *tc, Test *test);
int testActGelu(TestCase *tc, Test *test);
int testActGeluDeriv(TestCase *tc, Test *test);
int testActSoftmax(TestCase *tc, Test *test);

int testDefaultOptimization(TestCase *tc, Test *test);
int testMomentumOptimization(TestCase *tc, Test *test);
int testNesterovOptimization(TestCase *tc, Test *test);
int testAdaDeltaOptimization(TestCase *tc, Test *test);
int testWindowGradOptimization(TestCase *tc, Test *test);
int testAdaGradOptimization(TestCase *tc, Test *test);
int testAdamOptimization(TestCase *tc, Test *test);
int testRMSPropOptimization(TestCase *tc, Test *test);
int testL1WeightDecay(TestCase *tc, Test *test);
int testL2WeightDecay(TestCase *tc, Test *test);
int testL1Regularization(TestCase *tc, Test *test);
int testL2Regularization(TestCase *tc, Test *test);

int testDatasetLoad(TestCase *tc, Test *test);
int testDatasetSave(TestCase *tc, Test *test);
int testDatasetSaveBinary(TestCase *tc, Test *test);

int testFullLoad(TestCase *test_case, Test *test);
int testFullForward(TestCase *test_case, Test *test);
int testFullAccuracy(TestCase *tc, Test *test);
int testFullBackprop(TestCase *test_case, Test *test);

int testConvLoad(TestCase *test_case, Test *test);
int testConvForward(TestCase *test_case, Test *test);
int testConvAccuracy(TestCase *tc, Test *test);
int testConvBackprop(TestCase *test_case, Test *test);
int testConvCIFAR(TestCase *tc, Test *test);

int testRNNLoad(TestCase *test_case, Test *test);
int testRNNForward(TestCase *test_case, Test *test);
int testRNNBackprop(TestCase *test_case, Test *test);
int testRNNStep(TestCase *tc, Test *test);
int testRNNOneHot(TestCase *tc, Test *test);

int testLSTMLoad(TestCase *test_case, Test *test);
int testLSTMTrain(TestCase *test_case, Test *test);
int testLSTMBackprop(TestCase *test_case, Test *test);
int testGRULoad(TestCase *test_case, Test *test);
int testGRUTrain(TestCase *test_case, Test *test);
int testGRUBackprop(TestCase *test_case, Test *test);


int testNormalizationLoad(TestCase *test_case, Test *test);
int testNormalizationForward(TestCase *test_case, Test *test);
int testNormalizationBackprop(TestCase *test_case, Test *test);

int testDropoutLoad(TestCase *test_case, Test *test);
int testDropoutForward(TestCase *test_case, Test *test);
int testDropoutBackprop(TestCase *test_case, Test *test);

int testEncodedDecoderLoad(TestCase *test_case, Test *test);
int testEncodedDecoderSave(TestCase *test_case, Test *test);
int testEncodedDecoderClone(TestCase *test_case, Test *test);
int testEncodedDecoderPredict(TestCase *test_case, Test *test);
int testEncodedDecoderBackprop(TestCase *test_case, Test *test);

int testConcatOperatorLoad(TestCase *test_case, Test *test);
int testConcatOperatorForward(TestCase *test_case, Test *test);
int testConcatOperatorBackprop(TestCase *test_case, Test *test);

int testAddOperatorLoad(TestCase *test_case, Test *test);
int testAddOperatorForward(TestCase *test_case, Test *test);
int testAddOperatorBackprop(TestCase *test_case, Test *test);

int testMulOperatorLoad(TestCase *test_case, Test *test);
int testMulOperatorForward(TestCase *test_case, Test *test);
int testMulOperatorBackprop(TestCase *test_case, Test *test);

int testPositionalEmbedLoad(TestCase *test_case, Test *test);
int testPositionalEmbedForward(TestCase *test_case, Test *test);

int testAdditiveAttentionLoad(TestCase *test_case, Test *test);
int testAdditiveAttentionBackprop(TestCase *test_case, Test *test);
int testDotAttentionLoad(TestCase *test_case, Test *test);
int testDotAttentionBackprop(TestCase *test_case, Test *test);
int testMHAttentionLoad(TestCase *test_case, Test *test);
int testMHAttentionBackprop(TestCase *test_case, Test *test);
int testMHCausalSelfAttentionLoad(TestCase *test_case, Test *test);
int testMHCausalSelfAttentionBackprop(TestCase *test_case, Test *test);


/* psyc.c function prototypes */

PSGradient ***backprop(PSModel *model, PSFloat *x, PSFloat *y,
                       PSTrainingOptions *opts, PSGradient **gradients);

PSFloat updateModelParameters(PSModel *model,
                              PSFloat *training_data,
                              int batch_size, int elements_count,
                              PSFloat rate, PSTrainingOptions* opts, ...);

PSFloat *PSGetDropoutMask(PSLayer *layer, int t);

static int compareFloats(PSFloat a, PSFloat b, int rounding, int precision);
static int compareArrays(PSFloat *arr, PSFloat *exp, int len, Test* test,
                         char *descr, int rounding, int precision);
char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
int PSUpdateDelta(PSMatrix destdelta, PSMatrix srcdelta, PSMatrix weights,
                  int seqlen, int acceleration);

uint64_t testlen = 0;

int pretrained_mnist_layers_size[PRETRAINED_MNIST_NLAYERS] = {784,30,10};

PSFloat fullNetworkForwardResults[] = {
    0.000000,
    0.000000,
    0.000003,
    0.135474,
    0.000000,
    0.000000,
    0.000000,
    0.999762,
    0.000000,
    0.000000
};

PSFloat convNetworkForwardResults[] = {
    0.961797,
    0.000264,
    0.004449,
    0.054507,
    0.456677,
    0.989324,
    0.005734,
    0.251899,
    0.945555,
    0.003376
};


/*  Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2 */
PSFloat backpropGradients[8][7] = {
    {1.0, 6.0, 0.00000302, 0.0, 202.0, 0.00000000, 0.00000099},
    {1.0, 10.0, 0.00001238, 0.0, 202.0, 0.00000000, 0.00000408},
    {1.0, 28.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {1.0, 29.0, 0.00000000, 0.0, 202.0, 0.00000000, 0.00000000},
    {2.0, 0.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 1.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 8.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000},
    {2.0, 9.0, 0.00000000, 0.0, 29.0, 0.00000000, 0.00000000}
};

/*  Layer, Neuron, Bias, Weight1 idx, Weight2 idx, Weight1, Weight2 */
PSFloat backpropConvGradients[8][8] = {
    {1.0, 0.0, 0.74930726, 0.0, 1.0, 0.01887213, 0.00558867},
    {1.0, 1.0, -0.12929553, 0.0, 1.0, -0.00580211, -0.00211678},
    {3.0, 6.0, 0.00000055, 0.0, 25.0, 0.00000028, 0.00000035},
    {4.0, 0.0, 0.03533965, 0.0, 2.0, 0.00000000, 0.03533965}
};

PSFloat rnn_inputs_weights[2][4] = {
    {0.23728831, -0.13215413,  0.22972574, -0.30660592},
    {0.20497796,  0.10017828, -0.42993062, 0.13334368}
};

PSFloat rnn_outer_weights[4][2] = {
    { 0.66009386,  0.53237322},
    { 0.38653,    -0.17668558},
    { 0.39194875, -0.43592448},
    {-0.3616406,   0.22109871}
};

PSFloat rnn_recurrent_weights[2][2] = {
    {-0.46118039, -0.06823764},
    {-0.05642161, -0.69206895}
};

PSFloat rnn_expected_output[4][4] = {
    {0.30070284, 0.24446228, 0.23227381, 0.22256106},
    {0.21998344, 0.24441611, 0.24745303, 0.28814742},
    {0.23400344, 0.27607197, 0.30379749, 0.1861271},
    {0.24796499, 0.21649963, 0.19729585, 0.33823952}
};

PSFloat rnn_inner_gradients[2][6] = {
    { 0.7147815, -0.21859849, 0.11033169, -0.38062627, -0.20556136, 0.087832},
    {-0.25557706, 0.18080836, 0.40002974, -0.39482105, -0.1891429, 0.158413}
};

PSFloat rnn_outer_gradients[4][2] = {
    { 0.4023519, -0.29857975},
    {-0.33460693, 0.37440608},
    { 0.26141107, 0.0456771},
    {-0.32915604, -0.12150343}
};

PSFloat rnn_trained_inner_weights[2][4] = {
    { 0.2337144,  -0.13106114,  0.22917408, -0.30470279},
    { 0.20625585,  0.09927424, -0.43193077,  0.13531779}
};

PSFloat rnn_trained_outer_weights[4][2] = {
    {0.6580821, 0.53386612},
    {0.38820303, -0.17855761},
    {0.39064169, -0.43615287},
    {-0.35999482, 0.22170623}
};

PSFloat rnn_trained_recurrent_weights[2][2] = {
    {-0.46015258, -0.0686768},
    {-0.0554759, -0.69286102}
};

PSFloat rnn_inputs[5] = {4, 0, 1, 2, 3};
PSFloat rnn_labels[4] = {3, 2, 1, 0};

PSFloat lstm_training_data[] = {1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0};
PSFloat wg[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wi[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wo[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat wf[2][6] = {
    { 0.0097627,0.04303787,0.02055268,0.00897664,-0.01526904,0.02917882},
    {-0.01248256,0.0783546,0.09273255,-0.0233117,0.05834501,0.00577898}
};
PSFloat bg[2] = {0.0097627,0.04303787};
PSFloat bi[2] = {0.0097627,0.04303787};
PSFloat bo[2] = {0.0097627,0.04303787};
PSFloat bf[2] = {0.0097627,0.04303787};

PSFloat lstm_out_weights[4][2] = {
    { 0.0097627,   0.04303787},
    { 0.02055268,  0.00897664},
    {-0.01526904,  0.02917882},
    {-0.01248256,  0.0783546}};

PSFloat lstm_expected_states[2][3] = {
    {0.00497633, 0.01652635, 0.0163383},
    {0.00787092, 0.03837129, 0.05927353}
};

PSFloat lstm_expected_outputs[3][4] = {
    { 0.25001755, 0.24996395, 0.24995914, 0.25005937},
    { 0.25006783, 0.24978575, 0.24983151, 0.2503149},
    { 0.25008373, 0.24962334, 0.2497762, 0.25051672}
};

PSFloat expected_wg[2][6] = {
    {0.00998824,0.04246868,0.02021531,0.00897664,-0.01527745,0.02916139},
    {-0.01314446,0.07862245,0.09379196,-0.0233117,0.05836385,0.00582174}
};
PSFloat expected_wi[2][6] = {
    {0.00976488,0.04302317,0.0205475,0.00897664,-0.0152692,0.02917851},
    {-0.01249252,0.07837006,0.09280098,-0.0233117,0.05834622,0.00578173}
};
PSFloat expected_wo[2][6] = {
    {0.00976767,0.04302572,0.02054214,0.00897664,-0.01526927,0.02917832},
    {-0.0124946,0.07833524,0.09283829,-0.0233117,0.05834666,0.00578289}
};
PSFloat expected_wf[2][6] = {
    {0.0097627,0.04303513,0.02054733,0.00897664,-0.01526914,0.02917859},
    {-0.01248256,0.07835658,0.09276899,-0.0233117,0.05834562,0.00578039}
};
PSFloat expected_bg[2] = {0.00908168, 0.04370323};
PSFloat expected_bi[2] = {0.009745, 0.0431118};
PSFloat expected_bo[2] = {0.00974497, 0.04311221};
PSFloat expected_bf[2] = {0.00975461, 0.04307629};

PSFloat gru_expected_states[2][3] = {
    {0.0097, 0.0307, 0.0309},
    {0.015, 0.0649, 0.098}
};

PSFloat gru_expected_outputs[3][4] = {
    { 0.25, 0.2499, 0.24995914, 0.25005937},
    { 0.2501, 0.24978575, 0.24983151, 0.2503149},
    { 0.25008373, 0.24962334, 0.2497762, 0.25051672}
};

PSFloat gru_expected_wg[2][6] = {
    {0.02, 0.03, 0.01, 0.01,-0.01527745,0.02916139},
    {-0.03, 0.09,0.12,-0.02,0.05836385,0.00582174}
};
PSFloat gru_expected_wu[2][6] = {
    {0.00976488,0.04302317,0.0205475,0.00897664,-0.0152692,0.02917851},
    {-0.01249252,0.07837006,0.09280098,-0.0233117,0.05834622,0.00578173}
};
PSFloat gru_expected_wr[2][6] = {
    {0.00976767,0.04302572,0.02054214,0.00897664,-0.01526927,0.02917832},
    {-0.0124946,0.07833524,0.09283829,-0.0233117,0.05834666,0.00578289}
};

PSFloat gru_expected_bg[2] = {-0.01, 0.06};
PSFloat gru_expected_bu[2] = {0.009745, 0.0431118};
PSFloat gru_expected_br[2] = {0.00974497, 0.04311221};

PSFloat expected_loaded_dataset[10] = {
    0.506985, 0.0964945, -0.235624, -1.73432, 0.0523617,
    -0.391815, 0.961963, -0.910949, 1.88932, 0.0525423
};

PSTrainingOptions optimization_train_opts = {0};

int compareModels(PSModel *model1, PSModel *model2, Test* test);
int compareModelChain(PSModel *model1, PSModel *model2, Test* test);

static int testRecurrentNetworkMode(PSModel *model,
                                    PSRecurrentNetworkMode mode, Test *test);
PSFloat *readSerializedFloatArray(FILE *in, char *sep, uint64_t *length,
                                  uint64_t maxlen, uint64_t capacity);

static char *getExecutablePath(char *executable) {
    static char path[PATH_MAX + 1] = {0};
    char _realpath[PATH_MAX + 1];
    if (path[0]) return path;
    _realpath[0] = 0;
    if (realpath(executable, _realpath) != NULL) {
        char *dir = dirname(_realpath);
        if (dir == NULL) return NULL;
        int len = strlen((const char*) dir);
        if (len >= PATH_MAX) {
            fprintf(stderr, "WARN: getPsycPath(): dirname length > %d",
                    PATH_MAX);
            return NULL;
        }
        memcpy(path, dir, len);
        path[len] = 0;
        return path;
    }
    return NULL;
}

static int joinPath(const char *dir, const char *fname, char *output) {
    assert(output != NULL);
    int maxlen = PATH_MAX, avail = maxlen;
    char *p = output;
    int len = snprintf(p, avail, "%s", dir);
    avail -= len;
    if (avail <= 0) goto exceeded;
    p += len;
    if (output[len - 1] != '/') {
        *(p++) = '/';
        len++;
        avail--;
        if (avail <= 0) goto exceeded;
    }
    len += snprintf(p, avail, "%s", fname);
    if (avail <= 0) goto exceeded;
    return 1;
exceeded:
    PSErr(NULL, "Path length exceeded");
    return 0;
}

static void getTmpFileName(const char *prfx, const char *sfx, char *buffer) {
    FILE *urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x%s",
            prfx, (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3], sfx);
    fclose(urand);
}

static int arrayMaxIndex(PSFloat *array, int len) {
    int i;
    PSFloat max = 0;
    int max_idx = 0;
    for (i = 0; i < len; i++) {
        PSFloat v = array[i];
        if (v > max) {
            max = v;
            max_idx = i;
        }
    }
    return max_idx;
}

/* Enabled tests */
static int avx_tests = 1, maths_tests = 1, activation_tests = 1,
           optimization_tests = 1, dataset_tests = 1, fullnet_tests = 1,
           convnet_tests = 1, rnn_tests = 1, lstm_tests = 1, gru_tests = 1,
           normalization_tests = 1, dropout_tests = 1, encdec_tests = 1,
           concat_op_tests = 1, add_op_tests = 1, mul_op_tests = 1,
           positional_embed_tests = 1, attention_tests = 1;

static int *test_ptrs[] = {
    &avx_tests, &maths_tests, &activation_tests, &optimization_tests,
    &dataset_tests, &fullnet_tests, &convnet_tests, &rnn_tests, &lstm_tests,
    &gru_tests, &normalization_tests, &dropout_tests, &concat_op_tests,
    &add_op_tests, &mul_op_tests, &positional_embed_tests, &encdec_tests,
    &attention_tests
};

static char*test_ids[] = {
    "avx", "maths", "activation", "optimization", "dataset", "fully-connected",
    "convolutional", "rnn", "lstm", "gru", "normalization", "dropout",
    "concatenate-layer", "add-layer", "multiply-layer", "positional-embedding",
    "encoder_decoder", "attention"
};

static void printTestList(void) {
    size_t i;
    for (i = 0; i < (sizeof(test_ids) / sizeof(char*)); i++) {
        char *test_id = test_ids[i];
        printf("%s\n", test_id);
    }
}

static int *testEnabledPointerByID(char *id) {
    int *ptr = NULL;
    size_t i;
    for (i = 0; i < (sizeof(test_ids) / sizeof(char*)); i++) {
        char *test_id = test_ids[i];
        if (strcasecmp(test_id, id) == 0) {
            ptr = test_ptrs[i];
            break;
        }
    }
    return ptr;
}

static int setTestEnabledStatus(char *test_id, int enabled) {
    int *ptr = testEnabledPointerByID(test_id);
    if (ptr == NULL) {
        fprintf(stderr, "ERROR: invalid test ID: '%s'\n", test_id);
        return 0;
    }
    *ptr = enabled;
    return 1;
}

static void disableAllTests(void) {
    size_t i;
    for (i = 0; i < (sizeof(test_ptrs) / sizeof(int*)); i++) {
        int *ptr = test_ptrs[i];
        *ptr = 0;
    }
}

void printHelp(char *executable) {
    fprintf(stderr, "Usage: %s [OPTIONS] [TEST_ID, ...]\n", executable);
    fprintf(stderr, "\nOPTIONS:\n\n");
    fprintf(stderr, "   --skip TEST_ID          Skip test (can be used "
        "multiple times\n");
    fprintf(stderr, "   --list-tests            List all test IDS\n");
    fprintf(stderr, "   -h, --help              Print this help\n");
}

int parseOptions(int argc, char **argv) {
    int i, is_last = 0, last_arg_idx = argc - 1;
    for (i = 1; i < argc; i++) {
        is_last = (last_arg_idx == i);
        char *arg = argv[i];
        if (strcmp("--skip", arg) == 0 && !is_last) {
            char *test_id = argv[++i];
            if (!setTestEnabledStatus(test_id, 0)) exit(1);
        } else if (strcmp("--list-tests", arg) == 0) {
            printTestList();
            exit(1);
        } else if ((strcmp("-h", arg) == 0) || (strcmp("--help", arg) == 0)) {
            printHelp(argv[0]);
            exit(1);
        } else if (arg[0] == '-') {
            fprintf(
                stderr, "ERROR: invalid option '%s'. Use '-h' to see all "
                "available options\n", arg
            );
            exit(1);
        } else break;
    }
    return i;
}

const char *executable_path = NULL;

int main(int argc, char** argv) {
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    executable_path = getExecutablePath(argv[0]);
    if (executable_path == NULL) executable_path = "./";
    PSSetDefaultTrainingOptions(&optimization_train_opts);
    /* Prevent differences between tests with float and tests with double */
    optimization_train_opts.eps = 1e-7;
    int argidx = parseOptions(argc, argv), all_disabled = 0;
    while (argidx < argc) {
        if (!all_disabled) disableAllTests();
        char *test_id = argv[argidx++];
        if (!setTestEnabledStatus(test_id, 1)) return 1;
    }
    int tot_tests = 0, tot_failed = 0;
    struct timeval start_t, end_t;
    gettimeofday(&start_t, NULL);
#ifdef USE_AVX
    if (avx_tests) {
        AVXTests = createTest("AVX");
        addTest(AVXTests, "Dot Product", NULL, testAVXDot);
        addTest(AVXTests, "Square", NULL, testAVXSquare);
        addTest(AVXTests, "Multiply Value", NULL, testAVXMultiplyVal);
        performTests(AVXTests);
        tot_tests += AVXTests->count;
        tot_failed += AVXTests->failed_count;
        deleteTest(AVXTests);
    }
#endif

    if (maths_tests) {
        mathsTests = createTest("Maths");
        addTest(mathsTests, "Dot Product", NULL, testMathsDotProduct);
        addTest(mathsTests, "Dot (Matrix-Vec.)", NULL, testMathsDot);
        addTest(mathsTests, "Outer Product", NULL, testMathsVecProd);
        addTest(mathsTests, "Sum vectors", NULL, testMathsSumV);
        addTest(mathsTests, "Sub vectors", NULL, testMathsSubV);
        addTest(mathsTests, "Mul. vectors", NULL, testMathsMulV);
        addTest(mathsTests, "Div. vectors", NULL, testMathsDivV);
        addTest(mathsTests, "Sum scalar to vec.", NULL, testMathsSumVS);
        addTest(mathsTests, "Sub. vec. from scalar", NULL, testMathsSubSV);
        addTest(mathsTests, "Mul. vec. by scalar", NULL, testMathsMulVS);
        addTest(mathsTests, "Div vec. by scalar", NULL, testMathsDivVS);
        addTest(mathsTests, "Div scalar by vec.", NULL, testMathsDivSV);
        addTest(mathsTests, "Clip", NULL, testMathsClip);
        addTest(mathsTests, "Threshold", NULL, testMathsThres);
        addTest(mathsTests, "Mapped Limit", NULL, testMathsMapLimit);
        addTest(mathsTests, "Exp", NULL, testMathsExp);
        addTest(mathsTests, "Power", NULL, testMathsPow);
        addTest(mathsTests, "Tanh", NULL, testMathsTanh);
        addTest(mathsTests, "Sqrt", NULL, testMathsSqrt);
        addTest(mathsTests, "Negate", NULL, testMathsNeg);
        addTest(mathsTests, "Abs.", NULL, testMathsAbs);
        addTest(mathsTests, "Mean", NULL, testMathsMean);
        addTest(mathsTests, "Variance", NULL, testMathsVar);
        addTest(mathsTests, "StdDev", NULL, testMathsStd);
        addTest(mathsTests, "Vector Fill", NULL, testMathVectorFill);
        addTest(mathsTests, "MatMul (vectors)", NULL, testMathsMatMul);
        addTest(mathsTests, "Vector To Matrix", NULL, testMathsVecToMatrix);
        addTest(mathsTests, "Matrix Copy", NULL, testMathsMatrixCopy);
        addTest(mathsTests, "Matrix Dup.", NULL, testMathsMatrixDup);
        addTest(mathsTests, "Matrix Expand", NULL, testMathsMatrixExpand);
        addTest(mathsTests, "Matrix Transp.", NULL, testMathsMatrixTranspose);
        addTest(mathsTests, "Matrix Swap", NULL, testMathsMatrixSwap);
        addTest(mathsTests, "Matrix Add", NULL, testMathsMatrixAdd);
        addTest(mathsTests, "Matrix Multiply", NULL, testMathsMatrixMultiply);
        addTest(mathsTests, "Matrix Subtract", NULL, testMathsMatrixSubtract);
        addTest(mathsTests, "Matrix Divide", NULL, testMathsMatrixDivide);
        addTest(mathsTests, "Matrix Product", NULL, testMathsMatrixProduct);
        addTest(mathsTests, "Matrix Product (MV)", NULL,
                testMathsMatrixProductMV);
        addTest(mathsTests, "Matrix Product (VM)", NULL,
                testMathsMatrixProductVM);
        performTests(mathsTests);
        tot_tests += mathsTests->count;
        tot_failed += mathsTests->failed_count;
        deleteTest(mathsTests);
    }

    if (activation_tests) {
        activationTests = createTest("Activation Functions");
        addTest(activationTests, "Sigmoid", NULL, testActSigmoid);
        addTest(activationTests, "Sigmoid der.", NULL, testActSigmoidDeriv);
        addTest(activationTests, "Tanh", NULL, testActTanh);
        addTest(activationTests, "Tanh der.", NULL, testActTanhDeriv);
        addTest(activationTests, "ReLU", NULL, testActRelu);
        addTest(activationTests, "ReLU der.", NULL, testActReluDeriv);
        addTest(activationTests, "GeLU", NULL, testActGelu);
        addTest(activationTests, "GeLU der.", NULL, testActGeluDeriv);
        addTest(activationTests, "Softmax", NULL, testActSoftmax);
        performTests(activationTests);
        tot_tests += activationTests->count;
        tot_failed += activationTests->failed_count;
        deleteTest(activationTests);
    }

    if (optimization_tests) {
        optimizationTests = createTest("Optimization");
        addTest(optimizationTests, "Default", NULL, testDefaultOptimization);
        addTest(optimizationTests, "Momentum", NULL, testMomentumOptimization);
        addTest(optimizationTests, "Nesterov", NULL, testNesterovOptimization);
        addTest(optimizationTests, "AdaDelta", NULL, testAdaDeltaOptimization);
        addTest(optimizationTests, "WindowGrad", NULL,
            testWindowGradOptimization);
        addTest(optimizationTests, "AdaGrad", NULL, testAdaGradOptimization);
        addTest(optimizationTests, "RMSProp", NULL, testRMSPropOptimization);
        addTest(optimizationTests, "Adam", NULL, testAdamOptimization);
        addTest(optimizationTests, "L1 W.Decay", NULL, testL1WeightDecay);
        addTest(optimizationTests, "L2 W.Decay", NULL, testL2WeightDecay);
        addTest(optimizationTests, "L1 Regul.", NULL, testL1Regularization);
        addTest(optimizationTests, "L2 Regul.", NULL, testL2Regularization);
        performTests(optimizationTests);
        tot_tests += optimizationTests->count;
        tot_failed += optimizationTests->failed_count;
        deleteTest(optimizationTests);
    }

    if (dataset_tests) {
        datasetTests = createTest("Dataset");
        addTest(datasetTests, "Load", NULL, testDatasetLoad);
        addTest(datasetTests, "Save", NULL, testDatasetSave);
        addTest(datasetTests, "Save (Binary)", NULL, testDatasetSaveBinary);
        performTests(datasetTests);
        tot_tests += datasetTests->count;
        tot_failed += datasetTests->failed_count;
        deleteTest(datasetTests);
    }

    if (fullnet_tests) {
        FCTests = createTest("Fully Connected Neural Network");
        FCTests->setup = genericSetup;
        FCTests->teardown = genericTeardown;
        addTest(FCTests, "Load", NULL, testFullLoad);
        addTest(FCTests, "Forward", NULL, testFullForward);
        addTest(FCTests, "Accuracy", NULL, testFullAccuracy);
        addTest(FCTests, "Backprop", NULL, testFullBackprop);
        addTest(FCTests, "Clone", NULL, testGenericClone);
        addTest(FCTests, "Save", NULL, testGenericSave);
        performTests(FCTests);
        tot_tests += FCTests->count;
        tot_failed += FCTests->failed_count;
        deleteTest(FCTests);
    }

    if (convnet_tests) {
        CNNTests = createTest("Convolutional Neural Network");
        CNNTests->setup = genericSetup;
        CNNTests->teardown = genericTeardown;
        addTest(CNNTests, "Load", NULL, testConvLoad);
        addTest(CNNTests, "Forward", NULL, testConvForward);
        addTest(CNNTests, "Backprop", NULL, testConvBackprop);
        /*addTest(CNNTests, "Accuracy", NULL, testConvAccuracy);*/
        addTest(CNNTests, "CIFAR Backprop", NULL, testConvCIFAR);
        addTest(CNNTests, "Clone", NULL, testGenericClone);
        addTest(CNNTests, "Save", NULL, testGenericSave);
        performTests(CNNTests);
        tot_tests += CNNTests->count;
        tot_failed += CNNTests->failed_count;
        deleteTest(CNNTests);
    }

    if (rnn_tests) {
        RNNTests = createTest("Recurrent Neural Network");
        RNNTests->setup = RNNSetup;
        RNNTests->teardown = RNNTeardown;
        addTest(RNNTests, "Load", NULL, testRNNLoad);
        addTest(RNNTests, "Forward", NULL, testRNNForward);
        addTest(RNNTests, "Backprop", NULL, testRNNBackprop);
        addTest(RNNTests, "Step", NULL, testRNNStep);
        addTest(RNNTests, "Clone", NULL, testGenericClone);
        addTest(RNNTests, "Save", NULL, testGenericSave);
        addTest(RNNTests, "OneHot", NULL, testRNNOneHot);
        performTests(RNNTests);
        tot_tests += RNNTests->count;
        tot_failed += RNNTests->failed_count;
        deleteTest(RNNTests);
    }

    if (lstm_tests) {
        LSTMTests = createTest("LSTM Neural Network");
        LSTMTests->setup = LSTMSetup;
        LSTMTests->teardown = RNNTeardown;
        /* addTest(LSTMTests, "Load", NULL, testLSTMLoad); */
        addTest(LSTMTests, "Train", NULL, testLSTMTrain);
        addTest(LSTMTests, "Backprop", NULL, testLSTMBackprop);
        addTest(LSTMTests, "Clone", NULL, testGenericClone);
        addTest(LSTMTests, "Save", NULL, testGenericSave);
        performTests(LSTMTests);
        tot_tests += LSTMTests->count;
        tot_failed += LSTMTests->failed_count;
        deleteTest(LSTMTests);
    }
    if (gru_tests) {
        GRUTests = createTest("GRU Neural Network");
        GRUTests->setup = GRUSetup;
        GRUTests->teardown = RNNTeardown;
        /* addTest(GRUTests, "Load", NULL, testGRULoad); */
        addTest(GRUTests, "Train", NULL, testGRUTrain);
        addTest(GRUTests, "Backprop", NULL, testGRUBackprop);
        addTest(GRUTests, "Clone", NULL, testGenericClone);
        addTest(GRUTests, "Save", NULL, testGenericSave);
        performTests(GRUTests);
        tot_tests += GRUTests->count;
        tot_failed += GRUTests->failed_count;
        deleteTest(GRUTests);
    }
    if (normalization_tests) {
        NormalizationTests = createTest("Normalization Layer");
        NormalizationTests->setup = genericSetup;
        NormalizationTests->teardown = genericTeardown;
        addTest(NormalizationTests, "Load", NULL, testNormalizationLoad);
        addTest(NormalizationTests, "Forward", NULL,
               testNormalizationForward);
        addTest(NormalizationTests, "Backprop", NULL,
               testNormalizationBackprop);
        addTest(NormalizationTests, "Save", NULL, testGenericSave);
        performTests(NormalizationTests);
        tot_tests += NormalizationTests->count;
        tot_failed += NormalizationTests->failed_count;
        deleteTest(NormalizationTests);
    }
    if (dropout_tests) {
        DropoutTests = createTest("Dropout Layer");
        DropoutTests->setup = genericSetup;
        DropoutTests->teardown = genericTeardown;
        addTest(DropoutTests, "Load", NULL, testDropoutLoad);
        addTest(DropoutTests, "Forward", NULL,
               testDropoutForward);
        addTest(DropoutTests, "Backprop", NULL,
               testDropoutBackprop);
        addTest(DropoutTests, "Save", NULL, testGenericSave);
        performTests(DropoutTests);
        tot_tests += DropoutTests->count;
        tot_failed += DropoutTests->failed_count;
        deleteTest(DropoutTests);
    }
    if (concat_op_tests) {
        ConcatOperatorLayerTests = createTest("Concatenate Operator Layer");
        ConcatOperatorLayerTests->setup = genericSetup;
        ConcatOperatorLayerTests->teardown = genericTeardown;
        addTest(ConcatOperatorLayerTests, "Load", NULL, testConcatOperatorLoad);
        addTest(ConcatOperatorLayerTests, "Forward", NULL,
                testConcatOperatorForward);
        addTest(ConcatOperatorLayerTests, "Backprop", NULL,
                testConcatOperatorBackprop);
        addTest(ConcatOperatorLayerTests, "Save", NULL, testGenericSave);
        performTests(ConcatOperatorLayerTests);
        tot_tests += ConcatOperatorLayerTests->count;
        tot_failed += ConcatOperatorLayerTests->failed_count;
        deleteTest(ConcatOperatorLayerTests);
    }
    if (add_op_tests) {
        AddOperatorLayerTests = createTest("Add Operator Layer");
        AddOperatorLayerTests->setup = genericSetup;
        AddOperatorLayerTests->teardown = genericTeardown;
        addTest(AddOperatorLayerTests, "Load", NULL, testAddOperatorLoad);
        addTest(AddOperatorLayerTests, "Forward", NULL,
                testAddOperatorForward);
        addTest(AddOperatorLayerTests, "Backprop", NULL,
                testAddOperatorBackprop);
        addTest(AddOperatorLayerTests, "Save", NULL, testGenericSave);
        performTests(AddOperatorLayerTests);
        tot_tests += AddOperatorLayerTests->count;
        tot_failed += AddOperatorLayerTests->failed_count;
        deleteTest(AddOperatorLayerTests);
    }
    if (mul_op_tests) {
        MulOperatorLayerTests = createTest("Multiply Operator Layer");
        MulOperatorLayerTests->setup = genericSetup;
        MulOperatorLayerTests->teardown = genericTeardown;
        addTest(MulOperatorLayerTests, "Load", NULL, testMulOperatorLoad);
        addTest(MulOperatorLayerTests, "Forward", NULL,
                testMulOperatorForward);
        addTest(MulOperatorLayerTests, "Backprop", NULL,
                testMulOperatorBackprop);
        addTest(MulOperatorLayerTests, "Save", NULL, testGenericSave);
        performTests(MulOperatorLayerTests);
        tot_tests += MulOperatorLayerTests->count;
        tot_failed += MulOperatorLayerTests->failed_count;
        deleteTest(MulOperatorLayerTests);
    }
    if (positional_embed_tests) {
        PositionalEmbedTests = createTest("Positional Encoding (Embed)");
        PositionalEmbedTests->setup = genericSetup;
        PositionalEmbedTests->teardown = genericTeardown;
        addTest(PositionalEmbedTests, "Load", NULL, testPositionalEmbedLoad);
        addTest(PositionalEmbedTests, "Forward", NULL,
                testPositionalEmbedForward);
        addTest(PositionalEmbedTests, "Clone", NULL, testGenericClone);
        addTest(PositionalEmbedTests, "Save", NULL, testGenericSave);
        performTests(PositionalEmbedTests);
        tot_tests += PositionalEmbedTests->count;
        tot_failed += PositionalEmbedTests->failed_count;
        deleteTest(PositionalEmbedTests);
    }
    if (encdec_tests) {
        EncoderDecoderTests = createTest("Encoder-Decoder");
        EncoderDecoderTests->setup = encoderDecoderSetup;
        EncoderDecoderTests->teardown = encoderDecoderTeardown;
        addTest(EncoderDecoderTests, "Load", NULL, testEncodedDecoderLoad);
        addTest(EncoderDecoderTests, "Save", NULL, testEncodedDecoderSave);
        addTest(EncoderDecoderTests, "Clone", NULL, testEncodedDecoderClone);
        addTest(EncoderDecoderTests, "Predict", NULL,
            testEncodedDecoderPredict);
        addTest(EncoderDecoderTests, "Backprop", NULL,
            testEncodedDecoderBackprop);
        performTests(EncoderDecoderTests);
        tot_tests += EncoderDecoderTests->count;
        tot_failed += EncoderDecoderTests->failed_count;
        deleteTest(EncoderDecoderTests);
    }
    if (attention_tests) {
        AdditiveAttentionTests = createTest("Additive Attention");
        AdditiveAttentionTests->setup = attentionSetup;
        AdditiveAttentionTests->teardown = attentionTeardown;
        addTest(AdditiveAttentionTests, "Load", NULL,
            testAdditiveAttentionLoad);
        addTest(AdditiveAttentionTests, "Save", NULL, testEncodedDecoderSave);
        addTest(AdditiveAttentionTests, "Clone", NULL, testEncodedDecoderClone);
        addTest(AdditiveAttentionTests, "Backprop", NULL,
            testAdditiveAttentionBackprop);
        performTests(AdditiveAttentionTests);
        tot_tests += AdditiveAttentionTests->count;
        tot_failed += AdditiveAttentionTests->failed_count;
        deleteTest(AdditiveAttentionTests);

        DotAttentionTests = createTest("Dot Product Attention");
        DotAttentionTests->setup = attentionSetup;
        DotAttentionTests->teardown = attentionTeardown;
        addTest(DotAttentionTests, "Load", NULL, testDotAttentionLoad);
        addTest(DotAttentionTests, "Save", NULL, testEncodedDecoderSave);
        addTest(DotAttentionTests, "Clone", NULL, testEncodedDecoderClone);
        addTest(DotAttentionTests, "Backprop", NULL, testDotAttentionBackprop);
        performTests(DotAttentionTests);
        tot_tests += DotAttentionTests->count;
        tot_failed += DotAttentionTests->failed_count;
        deleteTest(DotAttentionTests);

        MHAttentionTests = createTest("Multi-Head Attention");
        MHAttentionTests->setup = attentionSetup;
        MHAttentionTests->teardown = attentionTeardown;
        addTest(MHAttentionTests, "Load", NULL, testMHAttentionLoad);
        addTest(MHAttentionTests, "Save", NULL, testEncodedDecoderSave);
        addTest(MHAttentionTests, "Clone", NULL, testEncodedDecoderClone);
        addTest(MHAttentionTests, "Backprop", NULL, testMHAttentionBackprop);
        performTests(MHAttentionTests);
        tot_tests += MHAttentionTests->count;
        tot_failed += MHAttentionTests->failed_count;
        deleteTest(MHAttentionTests);

        MHCausalSelfAttentionTests =
            createTest("Multi-Head Causal Self-Attention");
        MHCausalSelfAttentionTests->setup = attentionSetup;
        MHCausalSelfAttentionTests->teardown = attentionTeardown;
        addTest(MHCausalSelfAttentionTests, "Load", NULL,
            testMHCausalSelfAttentionLoad);
        addTest(MHCausalSelfAttentionTests, "Save", NULL,
            testGenericSave);
        addTest(MHCausalSelfAttentionTests, "Clone", NULL,
            testGenericClone);
        addTest(MHCausalSelfAttentionTests, "Backprop", NULL,
            testMHCausalSelfAttentionBackprop);
        performTests(MHCausalSelfAttentionTests);
        tot_tests += MHCausalSelfAttentionTests->count;
        tot_failed += MHCausalSelfAttentionTests->failed_count;
        deleteTest(MHCausalSelfAttentionTests);
    }
    gettimeofday(&end_t, NULL);
    time_t elapsed = PSGetElapsedTimeUS(start_t, end_t);
    char *elapsed_str = PSGetElapsedTimeString(elapsed, PS_OPT_TIME_FULL);
    printf(
        "\n%d tests performed in %s\n", tot_tests, elapsed_str
    );
    int succeded = tot_tests - tot_failed;
    if (succeded > 0)
        printf(PSCOLOR_GREEN "Succeeded: %d\n" PSCOLOR_RESET, succeded);
    if (tot_failed > 0)
        printf(PSCOLOR_RED "Failed:    %d\n" PSCOLOR_RESET, tot_failed);

    return tot_failed;

}

int genericSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("Test Model");
    if (model == NULL) {
        fprintf(stderr, "\ncould not create model\n");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\ncould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    PSFloat *test_data = NULL;
    char dataset_path[PATH_MAX] = {0};
    if (!joinPath(executable_path, MNIST_TEST_SAMPLE_PATH, dataset_path))
        return 0;
    uint64_t expected_datalen = PS_MNIST_INPUT_SIZE + 10;
    testlen = 0;
    FILE *f = fopen(dataset_path, "r");
    if (f == NULL) {
        fprintf(stderr, "\nCould not open dataset at: %s\n", dataset_path);
        return 0;
    }
    test_data = readSerializedFloatArray(
        f, ",", &testlen, expected_datalen, expected_datalen
    );
    fclose(f);
    if (testlen != expected_datalen) {
        fprintf(
            stderr, "\nExpected datalen %" PRIu64" != datalen %" PRIu64 "\n",
            expected_datalen, testlen
        );
        free(test_data);
        return 0;
    }
    test_case->data[1] = test_data;
    if (test_data == NULL) return 0;
    return 1;
}

int genericTeardown(TestCase *test_case) {
    PSModel *model = getModel(test_case);
    if (model != NULL) PSModelFree(model);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int RNNSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("RNN Test Model");
    if (model == NULL) {
        PSErr(NULL, "\nCould not create model!");
        return 0;
    }
    model->flags |= PS_FLAG_ONEHOT;
    PSAddLayer(model, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(model, RNNLayer, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(model, SoftMax, RNN_INPUT_SIZE, NULL);
    if (model->size < 1) {
        PSErr(NULL, "\nCould not add all layers!");
        return 0;
    }
    model->layers[1]->flags |= PS_FLAG_NO_BIAS;
    model->layers[model->size - 1]->flags |= PS_FLAG_ONEHOT;

    int i, j, w;
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (layer->weights == NULL) {
            PSErr(NULL, "\nLayer[%d] weights is NULL", i);
            return 0;
        }
        if (layer->weights[0] == NULL) {
            PSErr(NULL, "\nLayer[%d] weights[0] is NULL", i);
            return 0;
        }
        if (i == 1) {
            if (layer->weights[1] == NULL) {
                PSErr(NULL, "\nLayer[%d] weights[1] is NULL", i);
                return 0;
            }
            uint64_t input_weights_count = PSMatrixLength(layer->weights[0]) /
                layer->size;
            /*uint64_t hidden_weights_count=PSMatrixLength(layer->weights[0]);*/
            /*uint64_t tot_weights_count = input_weights_count + layer->size;*/
            if ((uint64_t) RNN_INPUT_SIZE != input_weights_count) {
                PSErr(
                    NULL, "\nRNN Layer input_weights_count expected to be %d, "
                    "got %llu", RNN_INPUT_SIZE, input_weights_count
                );
                return 0;
            }
            for (j = 0; j < layer->size; j++) {
                PSFloat *input_weights = layer->weights[0] +
                                         (j * RNN_INPUT_SIZE);
                PSFloat *hidden_weights = layer->weights[0] +
                                          (j * layer->size);
                for (w = 0; w < RNN_INPUT_SIZE; w++)
                    input_weights[w] = rnn_inputs_weights[j][w];
                for (w = 0; w < layer->size; w++)
                    hidden_weights[w] = rnn_recurrent_weights[j][w];
            }
        } else {
            for (j = 0; j < layer->size; j++) {
                int prev_size = model->layers[i - 1]->size;
                PSFloat *input_weights = layer->weights[0] + (j * prev_size);
                for (w = 0; w < prev_size; w++)
                    input_weights[w] = rnn_inputs_weights[j][w];
            }
        }
    }
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSRecurrentNetworkMode rnn_mode = model->rnn_mode;
    if (rnn_mode != ManyToMany) {
        fprintf(
            stderr, "\nInvalid Recurrent Network Mode: '%s'\n",
            PSGetRecurrentModeLabel(rnn_mode)
        );
        return 0;
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    int train_data_len = 1 + (RNN_TIMES * 2);
    int labels_offset = 1 + RNN_TIMES;
    PSFloat *training_data = malloc(train_data_len * sizeof(PSFloat));
    PSFloat *p = training_data;
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(p, rnn_inputs, labels_offset * sizeof(PSFloat));
    p += labels_offset;
    memcpy(p, rnn_labels, RNN_TIMES * sizeof(PSFloat));
    test_case->data[1] = training_data;
    return 1;
}

int RNNTeardown(TestCase *test_case) {
    PSModel *model = getModel(test_case);
    if (model != NULL) PSModelFree(model);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int LSTMSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("LSTM Test Model");
    if (model == NULL) {
        fprintf(stderr, "\nCould not create model!\n");
        return 0;
    }
    model->flags |= PS_FLAG_ONEHOT;
    PSAddLayer(model, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(model, LSTM, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(model, SoftMax, RNN_INPUT_SIZE, NULL);
    if (model->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    PSLayer *out = model->layers[model->size - 1];
    out->flags |= PS_FLAG_ONEHOT;
    PSLayer *layer = model->layers[1];
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSLSTMCell *cell = PSGetLSTMCell(layer);
    if (cell == NULL) {
        PSErr(NULL, "\nLSTM Cell is NULL for layer %d", layer->index);
        return 0;
    }
    if (layer->weights == NULL || cell->candidate_weights == NULL) {
        PSErr(NULL, "\nLSTM layer has incomplete weights");
        return 0;
    }
    int i, w;
    int input_size = (int) PSMatrixLength(cell->candidate_weights) /
                           layer->size;
    for (i = 0; i < layer->size; i++) {
        cell->candidate_biases[i] = bg[i];
        cell->input_biases[i] = bi[i];
        cell->output_biases[i] = bo[i];
        cell->forget_biases[i] = bf[i];
        int woffs = (i * input_size), woffs_h = (i * layer->size);
        for (w = 0; w < input_size; w++) {
            cell->candidate_weights[woffs + w] = wg[i][w];
            cell->input_weights[woffs + w] = wi[i][w];
            cell->output_weights[woffs + w] = wo[i][w];
            cell->forget_weights[woffs + w] = wf[i][w];
        }
        for (w = 0; w < layer->size; w++) {
            int src_idx = input_size + w;
            cell->candidate_hidden_weights[woffs_h + w] = wg[i][src_idx];
            cell->input_hidden_weights[woffs_h + w] = wi[i][src_idx];
            cell->output_hidden_weights[woffs_h + w] = wo[i][src_idx];
            cell->forget_hidden_weights[woffs_h + w] = wf[i][src_idx];
        }
    }

    for (i = 0; i < out->size; i++) {
        PSNeuron neuron = {0};
        if (!PSGetNeuron(out, i, &neuron)) {
            fprintf(stderr, "\nCould not get layer[%d] neuron[%d]\n",
                    out->index, i);
            return 0;
        }
        *neuron.bias = 0.0;
        for (w = 0; w < layer->size; w++) {
            neuron.weights[w] = lstm_out_weights[i][w];
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    int train_data_len = 2 + (LSTM_TIMES * 2);
    PSFloat *training_data = malloc(train_data_len * sizeof(PSFloat));
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(training_data, lstm_training_data, train_data_len * sizeof(PSFloat));
    test_case->data[1] = training_data;
    return 1;
}

int GRUSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("GRU Test");
    if (model == NULL) {
        fprintf(stderr, "\nCould not create model!\n");
        return 0;
    }
    model->flags |= PS_FLAG_ONEHOT;
    PSAddLayer(model, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(model, GRU, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(model, SoftMax, RNN_INPUT_SIZE, NULL);
    if (model->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    PSLayer *out = model->layers[model->size - 1];
    out->flags |= PS_FLAG_ONEHOT;
    PSLayer *layer = model->layers[1];
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSGRUCell *cell = PSGetGRUCell(layer);
    if (cell == NULL) {
        PSErr(NULL, "\nGRU Cell is NULL for layer %d", layer->index);
        return 0;
    }
    if (layer->weights == NULL || cell->candidate_weights == NULL) {
        PSErr(NULL, "\nGRU layer has incomplete weights");
        return 0;
    }
    int i, w;
    int input_size = (int) PSMatrixLength(cell->candidate_weights) /
                           layer->size;
    for (i = 0; i < layer->size; i++) {
        cell->candidate_biases[i] = bg[i];
        cell->update_biases[i] = bi[i];
        cell->reset_biases[i] = bf[i];
        int woffs = (i * input_size), woffs_h = (i * layer->size);
        for (w = 0; w < input_size; w++) {
            cell->candidate_weights[woffs + w] = wg[i][w];
            cell->update_weights[woffs + w] = wi[i][w];
            cell->reset_weights[woffs + w] = wf[i][w];
        }
        for (w = 0; w < layer->size; w++) {
            int src_idx = input_size + w;
            cell->candidate_hidden_weights[woffs_h + w] = wg[i][src_idx];
            cell->update_hidden_weights[woffs_h + w] = wi[i][src_idx];
            cell->reset_hidden_weights[woffs_h + w] = wf[i][src_idx];
        }
    }

    for (i = 0; i < out->size; i++) {
        PSNeuron neuron = {0};
        if (!PSGetNeuron(out, i, &neuron)) {
            fprintf(stderr, "\nCould not get layer[%d] neuron[%d]\n",
                    out->index, i);
            return 0;
        }
        *neuron.bias = 0.0;
        for (w = 0; w < layer->size; w++) {
            neuron.weights[w] = lstm_out_weights[i][w];
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    int train_data_len = 2 + (LSTM_TIMES * 2);
    PSFloat *training_data = malloc(train_data_len * sizeof(PSFloat));
    if (training_data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    memcpy(training_data, lstm_training_data, train_data_len * sizeof(PSFloat));
    test_case->data[1] = training_data;
    return 1;
}

int ModelBackpropTest(Test *test, char *model_file, char *data_file_prefix,
                      char *training_data_file, char *labels_data_file,
                      uint64_t training_data_len, uint64_t label_data_len,
                      char *model_name, PSTrainingOptions *topts,
                      int rounding, int precision, int expected_size,
                      int acceleration)
{
    assert(test != NULL);
    assert(model_file != NULL);
    assert(data_file_prefix != NULL);
    assert(training_data_file != NULL);
    assert(labels_data_file != NULL);
    int ok = 1;
    const char *acceleration_name = NULL;
    if (acceleration == PSGlobalAcceleration) acceleration_name = "Default";
    else acceleration_name = PSGetAccelerationName(acceleration);
    PSFloat *x = NULL, *y = NULL, *states = NULL, *deltas = NULL, *grads = NULL;
    PSGradient ***gradients = NULL;
    FILE *f = NULL;
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, model_file, path), test
    );
    if (model_name != NULL) model_name = "Backprop Model";
    PSModel *model = PSModelCreate(model_name);
    testAssertNotNull(model, test);
    ok = PSModelLoad(model, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to load model from '%s'", path
    );
    if (!PSModelIsBuilt(model)) {
        ok = PSModelBuild(model);
        if (!ok) {
            fprintf(stderr, "\nFailed to build model!\n");
            goto final;
        }
    }
    ok = model->layers != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test,
        "model %s has no layers",model->name
    );
    if (expected_size > 0) {
        ok = model->size == expected_size;
        testAssertWithMessageOrGoto(
            ok, final, test,
            "model has %d layers, expected %d", model->size, expected_size
        );
    }
    ok = joinPath(executable_path, training_data_file, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not get path for %s", training_data_file
    );
    f = fopen(path, "r");
    ok = (f != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not open '%s' for reading", path
    );
    uint64_t image_len = 0, label_len = 0;
    x = readSerializedFloatArray(f, ",", &image_len, training_data_len,
                                 training_data_len);
    fclose(f);
    f = NULL;
    ok = x != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not read image data from '%s'",path);
    if (training_data_len > 0) {
        ok = image_len == training_data_len;
        testAssertWithMessageOrGoto(
            ok, final, test, "Image length should be %d, got %d",
            training_data_len, image_len
        );
    }
    ok = joinPath(executable_path, labels_data_file, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not get path for %s", labels_data_file
    );
    f = fopen(path, "r");
    ok = (f != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not open '%s' for reading", path
    );
    y = readSerializedFloatArray(f, ",", &label_len, label_len, label_len);
    fclose(f);
    f = NULL;
    ok = y != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not read label data from '%s'",path);
    if (label_data_len > 0) {
        ok = label_len == label_data_len;
        testAssertWithMessageOrGoto(
            ok, final, test, "Label length should be %d, got %d",
            label_data_len, label_len
        );
    }
    model->acceleration = acceleration;
    gradients = backprop(model, x, y, topts, NULL);
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for model %s (acceleration: %s)",
        model->name, acceleration_name
    );
    PSGradient **netgradients = gradients[0];
    ok = netgradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL",
        model->name, acceleration_name
    );
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        char testlabel[1024];
        uint64_t lsize = layer->size;
        char fname[PATH_MAX] = {0};
        if (layer->states != NULL) {
            snprintf(
                fname, PATH_MAX-1, "resources/%s-layer-%d-states.data",
                data_file_prefix, i
            );
            ok = joinPath(executable_path, fname, path);
            testAssertWithMessageOrGoto(
                ok,final,test,"Could not get path for %s",fname
            );
            f = fopen(path, "r");
            ok = f != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could not open '%s'", path
            );
            uint64_t len = 0;
            free(states);
            states = readSerializedFloatArray(f, ",", &len, lsize, lsize);
            fclose(f);
            f = NULL;
            ok = states != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could nor read data from '%s'", path
            );
            ok = len == lsize;
            testAssertWithMessageOrGoto(
                ok, final, test, "States length expected to be %d, got %d",
                lsize, len
            );
            snprintf(testlabel, 1023, "Layer[%d] states (accel. %s)", i,
                     acceleration_name);
            ok = compareArrays(
                layer->states, states, lsize, test, testlabel, rounding,
                precision
            );
            testAssertWithMessageOrGoto(ok,final,test,"%s mismatch",testlabel);
        }
        if (layer->delta != NULL) {
            snprintf(
                fname, PATH_MAX-1, "resources/%s-layer-%d-deltas.data",
                data_file_prefix, i
            );
            ok = joinPath(executable_path, fname, path);
            testAssertWithMessageOrGoto(
                ok,final,test,"Could not get path for %s",fname
            );
            f = fopen(path, "r");
            ok = f != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could not open '%s'", path
            );
            uint64_t len = 0, lsize = layer->size;
            free(deltas);
            deltas = readSerializedFloatArray(f, ",", &len, lsize, lsize);
            fclose(f);
            f = NULL;
            ok = deltas != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could nor read data from '%s'", path
            );
            ok = len == lsize;
            testAssertWithMessageOrGoto(
                ok, final, test, "deltas length expected to be %d, got %d",
                lsize, len
            );
            snprintf(testlabel, 1023, "Layer[%d] delta (accel. %s)", i,
                     acceleration_name);
            ok = compareArrays(layer->delta, deltas, lsize, test,testlabel,
                               rounding, precision);
            testAssertWithMessageOrGoto(ok,final,test,"%s mismatch",testlabel);
        }
        PSGradient *grad = NULL;
        if (i > 0 && layer->weights != NULL && Pooling != layer->type)
            grad = netgradients[i - 1];
        if (grad != NULL) {
            /* Bias gradients */
            uint64_t len = 0, grad_len = PSGetLayerParametersCount(
                layer, PS_PARAM_BIAS
            );
            if (grad_len <= 0 || !grad->biases) goto weight_gradients;
            snprintf(
                fname, PATH_MAX-1, "resources/%s-layer-%d-bgrads.data",
                data_file_prefix, i
            );
            ok = joinPath(executable_path, fname, path);
            testAssertWithMessageOrGoto(
                ok,final,test,"Could not get path for %s",fname
            );
            f = fopen(path, "r");
            ok = f != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could not open '%s'", path
            );
            free(grads);
            grads = readSerializedFloatArray(f, ",", &len, grad_len, grad_len);
            fclose(f);
            f = NULL;
            ok = grads != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could nor read data from '%s'", path
            );
            ok = len == grad_len;
            testAssertWithMessageOrGoto(
                ok, final, test, "bias gradient length expected to be %d, "
                "got %d", grad_len, len
            );
            snprintf(testlabel, 1023, "Layer[%d] bias gradients (accel. %s)", i,
                     acceleration_name);
            ok = compareArrays(grad->biases, grads, grad_len, test,
                               testlabel, rounding, precision);
            testAssertWithMessageOrGoto(ok,final,test,"%s mismatch",testlabel);
weight_gradients:
            /* Weight gradients */
            len = 0, grad_len = PSGetLayerParametersCount(
                layer, PS_PARAM_WEIGHT
            );
            if (grad_len <= 0 || !grad->weights) goto weight_gradients;
            snprintf(
                fname, PATH_MAX-1, "resources/%s-layer-%d-wgrads.data",
                data_file_prefix, i
            );
            ok = joinPath(executable_path, fname, path);
            testAssertWithMessageOrGoto(
                ok,final,test,"Could not get path for %s",fname
            );
            f = fopen(path, "r");
            ok = f != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could not open '%s'", path
            );
            free(grads);
            grads = readSerializedFloatArray(f, ",", &len, grad_len, grad_len);
            fclose(f);
            f = NULL;
            ok = grads != NULL;
            testAssertWithMessageOrGoto(
                ok, final, test, "Could nor read data from '%s'", path
            );
            ok = len == grad_len;
            testAssertWithMessageOrGoto(
                ok, final, test, "weight gradient length expected to be %d, "
                "got %d", grad_len, len
            );
            snprintf(testlabel, 1023, "Layer[%d] weight gradients (accel. %s)",
                     i, acceleration_name);
            ok = compareArrays(grad->weights, grads, grad_len, test,
                               testlabel, rounding, precision);
            testAssertWithMessageOrGoto(ok,final,test,"%s mismatch",testlabel);
        }
    }
final:
    if (gradients != NULL && model != NULL)
        PSDeleteGradientsChain(gradients, model);
    PSModelFree(model);
    free(x);
    free(y);
    free(states);
    free(deltas);
    free(grads);
    return ok;
}

int testFullLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, PRETRAINED_FULL_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssert(loaded, test);
    testAssertEqual(model->size, 3, test);
    testAssertEqual(
        model->layers[0]->size, pretrained_mnist_layers_size[0], test
    );
    testAssertEqual(
        model->layers[1]->size, pretrained_mnist_layers_size[1], test
    );
    testAssertEqual(
        model->layers[2]->size, pretrained_mnist_layers_size[2], test
    );
    testAssertNotNull(model->layers[1]->biases, test);
    testAssertNotNull(model->layers[1]->weights, test);
    testAssertNotNull(model->layers[1]->weights[0], test);
    model->acceleration = PSGlobalAcceleration;
    PSFloat expected_biases[2][2] = {
        {-1.1618, -2.3288},
        {-6.0822, 0.8330}
    };
    PSFloat expected_weights[2][2] = {
        {-1.8497, -0.5419},
        {-1.2359, -4.677}
    };
    for(int l = 1; l < model->size; l++) {
        for (int i = 0; i < 2; i++) {
            PSFloat expected_bias = expected_biases[l - 1][i];
            PSFloat bias = getRoundedFloatDec(model->layers[l]->biases[i], 4);
            testAssertWithMessage(
                bias == expected_bias, test,
                "Layer[%d] Bias[%d] expected to be %g, got %g",
                l, i, expected_bias, bias
            );
            PSNeuron n = {0};
            testAssertNotNull(PSGetNeuron(model->layers[l], i, &n), test);
            testAssertNotNull(n.weights, test);
            PSFloat expected_w = expected_weights[l - 1][i];
            PSFloat w = getRoundedFloatDec(n.weights[0], 4);
            testAssertWithMessage(
                w == expected_w, test,
                "Layer[%d] N[%d] Weight[0] expected to be %g, got %g",
                l, i, expected_w, w
            );
        }
    };
    return 1;
}

int testFullForward(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSFloat *test_data = getTestData(test_case);
    PSForward(model, test_data);

    PSLayer *output = model->layers[model->size - 1];
    int i, res = 1;
    for (i = 0; i < output->size; i++) {
        PSFloat a = PSGetState(output, i);
        PSFloat expected = fullNetworkForwardResults[i];
        a = getRoundedFloat(a);
        expected = getRoundedFloat(expected);
        testAssertWithMessage(
            (a == expected), test, "Output[%d]-> %g != %g", i, a, expected
        );
    }
    return res;
}

int testFullAccuracy(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    PSFloat *test_data = getTestData(test_case);
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSFloat accuracy = PSTest(model, test_data, testlen, NULL),
            expected = 100.0;
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == expected), test, "Accuracy %g != from expected (%g)",
        accuracy, expected
    );
    return 1;
}

int testFullBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    PSFloat *test_data = getTestData(test_case);
    int input_size = model->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient ***grads = backprop(model, x, y, NULL, NULL);
    testAssertNotNull(grads, test);
    PSGradient **gradients = grads[0];
    testAssertNotNull(gradients, test);
    int i;
    for (i = 0; i < BP_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropGradients[i][0]);
        int nidx = (int) (backpropGradients[i][1]);
        PSFloat bias = backpropGradients[i][2];
        int widx1 = (int) (backpropGradients[i][3]);
        int widx2 = (int) (backpropGradients[i][4]);
        PSFloat w1 = backpropGradients[i][5];
        PSFloat w2 = backpropGradients[i][6];

        PSGradient *dl = gradients[lidx - 1];
        testAssertNotNull(dl, test);
        PSLayer *layer = model->layers[lidx];
        testAssertNotNull(layer, test);
        testAssertNotNull(dl->biases, test);
        PSFloat val = getRoundedFloat(dl->biases[nidx]);
        bias = getRoundedFloat(bias);
        w1 = getRoundedFloat(w1);
        w2 = getRoundedFloat(w2);
        testAssertWithMessageOrGoto(
            (val == bias), on_fail, test,
            "Gradient[%d][%d] bias %g != from expected (%g)",
            lidx - 1, nidx, val, bias
        );
        uint64_t wsize = dl->weight_count / layer->size;
        uint64_t widx1_g = (nidx * wsize) + widx1;
        uint64_t widx2_g = (nidx * wsize) + widx2;
        testAssertNotNull(dl->weights, test);
        testAssert(widx1_g < dl->weight_count, test);
        testAssert(widx2_g < dl->weight_count, test);
        val = getRoundedFloat(dl->weights[widx1_g]);
        testAssertWithMessageOrGoto(
            (val == w1), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expected (%g)",
            lidx - 1, nidx, widx1, val, w1
        );
        val = getRoundedFloat(dl->weights[widx2_g]);
        testAssertWithMessageOrGoto(
            (val == w2), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expected (%g)",
            lidx - 1, nidx, widx2, val, w2
        );
    }
    PSDeleteGradientsChain(grads, model);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, model);
    return 0;
}

int testConvLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, CONVOLUTIONAL_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSLayer *layer = model->layers[1];
    PSFloat bias = layer->biases[0];
    bias = getRoundedFloat(bias);
    PSFloat expected = CONV_L1F0_BIAS;
    expected = getRoundedFloat(expected);
    testAssertWithMessage(
        (expected == bias), test,
        "Layer[1]->bias[0] %g != from bias loaded from data %g",
        bias, expected
    );
    return 1;
}

int testConvForward(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    PSFloat *test_data = getTestData(test_case);
    PSForward(model, test_data);
    PSLayer *output = model->layers[model->size - 1];
    int i;
    for (i = 0; i < output->size; i++) {
        PSFloat a = PSGetState(output, i);
        PSFloat expected = convNetworkForwardResults[i];
        a = getRoundedFloat(a);
        expected = getRoundedFloat(expected);
        testAssertWithMessage(
            (a == expected), test,
            "Output[%d]-> %g != %g", i, a, expected
        );
    }
    return 1;
}

int testConvBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssert(model->acceleration == PSGlobalAcceleration, test);
    PSFloat *test_data = getTestData(test_case);
    int input_size = model->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient ***grads = backprop(model, x, y, NULL, NULL);
    testAssertNotNull(grads, test);
    PSGradient **gradients = grads[0];
    testAssertNotNull(gradients, test);
    for (int i = 0; i < BP_CONV_GRADIENTS_CHECKS; i++) {
        int lidx = (int) (backpropConvGradients[i][0]);
        int nidx = (int) (backpropConvGradients[i][1]);
        PSFloat bias = backpropConvGradients[i][2];
        int widx1 = (int) (backpropConvGradients[i][3]);
        int widx2 = (int) (backpropConvGradients[i][4]);
        PSFloat w1 = backpropConvGradients[i][5];
        PSFloat w2 = backpropConvGradients[i][6];
        PSGradient *dl = gradients[lidx - 1];
        if (dl == NULL) continue;
        PSLayer *layer = model->layers[lidx];
        testAssertNotNull(layer->weights, test);
        testAssertNotNull(layer->weights[0], test);
        int wsize = (int) PSMatrixLength(layer->weights[0]);
        if (layer->type != Convolutional) wsize /= layer->size;

        PSFloat val = getRoundedFloatDec(dl->biases[nidx], 4);
        bias = getRoundedFloatDec(bias, 4);
        testAssertWithMessageOrGoto(
            (val == bias), on_fail, test,
            "Gradient[%d][%d] bias %g != from expected (%g)",
            lidx - 1, nidx, val, bias
        );
        val = getRoundedFloatDec(dl->weights[(nidx * wsize) + widx1], 4);
        w1 = getRoundedFloatDec(w1, 4);
        testAssertWithMessageOrGoto(
            (val == w1), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expect. (%g)",
            lidx - 1, nidx, widx1, val, w1
        );
        val = getRoundedFloatDec(dl->weights[(nidx * wsize) + widx2], 4);
        w2 = getRoundedFloatDec(w2, 4);
        testAssertWithMessageOrGoto(
            (val == w2), on_fail, test,
            "Gradient[%d][%d] weight[%d] %g != from expect. (%g)",
            lidx - 1, nidx, widx2, val, w2
        );
    }
    PSDeleteGradientsChain(grads, model);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, model);
    return 0;
}

int testConvAccuracy(TestCase *test_case, Test *test) {
    PSFloat *test_data = getTestData(test_case);
    PSModel *model = PSModelCreate("CNN Test");
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, CONVOLUTIONAL_TRAINED_MODEL, path), test
    );
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSForward(model, test_data);
    PSFloat accuracy = PSTest(model, test_data, testlen, NULL),
            expected = 98.0;
    PSModelFree(model);
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == 98.0), test,
        "Accuracy %g != from expected (%g)", accuracy, expected
    );
    return 1;
}

int testConvCIFAR(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    int ok = ModelBackpropTest(test, CONVOLUTIONAL_CIFAR_MODEL,
                                 "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                                 PS_CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                                 2, 0, 8, PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = ModelBackpropTest(test, CONVOLUTIONAL_CIFAR_MODEL,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              PS_CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = ModelBackpropTest(test, CONVOLUTIONAL_CIFAR_MODEL,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              PS_CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = ModelBackpropTest(test, CONVOLUTIONAL_CIFAR_MODEL,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              PS_CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = ModelBackpropTest(test, CONVOLUTIONAL_CIFAR_MODEL,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              PS_CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_None);
    return ok;
}

int testRNNLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path,RECURRENT_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    int i, j, w, rnn_size = 0;
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        int exp_weight_types = 1, is_rnn_layer = (i == 1);
        uint64_t input_weight_count = 0, hidden_weight_count = 0;
        if (is_rnn_layer) {
            /* Recurrent Layer */
            rnn_size = layer->size;
            exp_weight_types = 2;
        }
        testAssertWithMessage(
            layer->weight_types == exp_weight_types, test,
            "Layer[%d] should have %d weight matrices, got: %d",
            i, exp_weight_types, layer->weight_types
        );
        testAssertNotNull(layer->weights, test);
        if (is_rnn_layer) {
            testAssertNotNull(layer->weights[0], test);
            testAssertNotNull(layer->weights[1], test);
            input_weight_count = PSMatrixLength(layer->weights[0]);
            hidden_weight_count = PSMatrixLength(layer->weights[1]);
            testAssertWithMessage(
                (input_weight_count == (uint64_t)(RNN_INPUT_SIZE*layer->size)),
                test,
                "Expected RNN Layer input weight count is %llu, got %llu",
                (uint64_t) RNN_INPUT_SIZE, input_weight_count
            );
            testAssertWithMessage(
                (hidden_weight_count == (uint64_t)(layer->size * layer->size)),
                test,
                "Expected RNN Layer hidden weight count is %llu, got %llu",
                (uint64_t) (layer->size * layer->size), hidden_weight_count
            );
            for (j = 0; j < layer->size; j++) {
                PSFloat *input_weights = layer->weights[0] +
                                         (j * RNN_INPUT_SIZE);
                PSFloat *hidden_weights = layer->weights[1] +
                                         (j * layer->size);
                for (w = 0; w < RNN_INPUT_SIZE; w++) {
                    testAssertWithMessage(
                        (input_weights[w] == rnn_inputs_weights[j][w]), test,
                        "RNN Layer[%d]: input weights[%d][%d] %g != %g",
                        i, j, w, input_weights[w], rnn_inputs_weights[j][w]
                    );
                }
                for (w = 0; w < layer->size; w++) {
                    testAssertWithMessage(
                        (hidden_weights[w] == rnn_recurrent_weights[j][w]),
                        test, "RNN Layer[%d]: hidden weights[%d][%d] %g != %g",
                        i, j, w, hidden_weights[w], rnn_recurrent_weights[j][w]
                    );
                }
            }
        } else {
            testAssertNotNull(layer->weights[0], test);
            input_weight_count = PSMatrixLength(layer->weights[0]);
            testAssertWithMessage(
                (input_weight_count = (uint64_t) rnn_size), test,
                "Expected Output Layer input weight count is %llu, got %llu",
                (uint64_t) rnn_size, input_weight_count
            );
            for (j = 0; j < layer->size; j++) {
                PSFloat *weights = layer->weights[0] + (j * rnn_size);
                for (w = 0; w < rnn_size; w++) {
                    testAssertWithMessage(
                        (weights[w] = rnn_outer_weights[j][w]), test,
                        "Output Layer[%d] weights[%d][%d] %g != %g",
                        i, j, w, weights[w], rnn_outer_weights[j][w]
                    );
                }
            }
        }
    }

    return 1;
}

int testRNNForward(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    PSForward(model, rnn_inputs);
    if (!testRecurrentNetworkMode(model, ManyToMany, test)) return 0;

    PSLayer *output = model->layers[model->size - 1];
    int i, j, seqlen = PSStateSequenceLength(output);
    for (i = 0; i < output->size; i++) {
        for (j = 0; j < seqlen; j++) {
            PSFloat s = PSGetState(output, i, j);
            s = getRoundedFloat(s);
            PSFloat expected = getRoundedFloat(rnn_expected_output[j][i]);
            testAssertWithMessage(
                (s == expected), test,
                "Output[%d][%d]: %g != %g", i, j, s, expected
            );
        }
    }
    return 1;
}

int testRNNBackpropOld(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    if (!PSModelIsBuilt(model)) {
        if (!PSModelBuild(model)) {
            fprintf(stderr, "\nFailed to build model!\n");
            return 0;
        }
    }
    int i, j, w;

    if (!testRecurrentNetworkMode(model, ManyToMany, test)) return 0;
    PSTrainingOptions opts = {
        .bptt_truncate = 4
    };
    PSGradient ***grads =backprop(model, rnn_inputs, rnn_labels, &opts, NULL);
    testAssertNotNull(grads, test);
    PSGradient **gradients = grads[0];
    testAssertNotNull(gradients, test);
    int dsize = model->size - 1;
    for (i = 0; i < dsize; i++) {
        PSGradient *gradient = gradients[i];
        PSLayer *l = model->layers[i + 1];
        testAssertNotNull(l->weights, test);
        testAssertNotNull(l->weights[0], test);
        int input_size = gradient->weight_count;
        int input_ws = input_size / l->size;
        if (RNNLayer == l->type) {
            testAssertNotNull(l->weights[1], test);
            //input_size -= l->size;
            input_ws -= l->size;
        }
        for (j = 0; j < l->size; j++) {
            PSFloat *expected = (i == 0 ? rnn_inner_gradients[j] :
                                 rnn_outer_gradients[j]);
            for (w = 0; w < input_ws; w++) {
                int widx = (j * input_ws) + w;
                PSFloat dw = getRoundedFloatDec(gradient->weights[widx], 5);
                PSFloat exp_dw = getRoundedFloatDec(expected[w], 5);
                testAssertWithMessageOrGoto(
                    (dw == exp_dw), on_fail, test,
                    "Gradient[%d][%d]->weight[%d]: %g != %g",
                    i, j, w, dw, exp_dw
                );
            }
            if (RNNLayer == l->type) {
                for (w = 0; w < l->size; w++) {
                    int widx = (input_ws * l->size) + (j * l->size) + w;
                    int gwidx = input_ws + w;
                    PSFloat dw = getRoundedFloat(gradient->weights[widx]);
                    PSFloat exp_dw = getRoundedFloat(expected[gwidx]);
                    testAssertWithMessageOrGoto(
                        (dw == exp_dw), on_fail, test,
                        "Gradient[%d][%d]->weight[%d]: %g != %g",
                        i, j, w, dw, exp_dw
                    );
                }
            }
        }
    }
    PSDeleteGradientsChain(grads, model);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, model);
    return 0;
}

int testRNNBackprop(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    char *model_file = "resources/basic-rnn.psmodel";
    char *inputs_file = "resources/rnn-inputs.data";
    char *labels_file = "resources/rnn-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (PS_TRAINING_EPOCH_AS_SEQUENCE | PS_TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = ModelBackpropTest(test, model_file,
                               "rnn", inputs_file, labels_file,
                               input_len, label_len, "RNN", &opts, 3, 0, 3,
                               PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = ModelBackpropTest(test, model_file,
                           "rnn", inputs_file, labels_file,
                           input_len, label_len, "RNN", &opts, 3, 0, 3,
                           PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = ModelBackpropTest(test, model_file,
                           "rnn", inputs_file, labels_file,
                           input_len, label_len, "RNN", &opts, 3, 0, 3,
                           PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = ModelBackpropTest(test, model_file,
                             "rnn", inputs_file, labels_file,
                              input_len, label_len, "RNN", &opts, 3, 0, 3,
                              PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = ModelBackpropTest(test, model_file,
                           "rnn", inputs_file, labels_file,
                           input_len, label_len, "RNN", &opts, 3, 0, 3,
                           PSAcceleration_None);
    return ok;
}

int testRNNStep(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    /*int train_data_len = 1 + (RNN_TIMES * 2);*/
    if (!testRecurrentNetworkMode(model, ManyToMany, test)) return 0;
    PSResetModelStateSequences(model, 0, 0);
    PSFloat *training_data = getTestData(test_case);
    PSFloat **sequences = &training_data;
    int elements_count = (int) *training_data;

    int i, j, w;
    PSFloat loss = updateModelParameters(
        model, training_data, 1, elements_count, RNN_LEARNING_RATE,
        NULL, sequences
    );
    UNUSED(loss);
    for (i = 1; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        int wsize = (int) PSGetLayerInputWeightsCount(layer, 1);
        if (layer->type == RNNLayer) wsize += layer->size;
        for (j = 0; j < layer->size; j++) {
            PSNeuron n = {0};
            testAssertNotNull(PSGetNeuron(layer, j, &n), test);
            for (w = 0; w < wsize; w++) {
                PSFloat *eweights, *lweights;
                int widx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE) {
                        eweights = rnn_trained_inner_weights[j];
                        lweights = PSGetNeuronInputWeights(&n);
                    } else {
                        widx -= RNN_INPUT_SIZE;
                        eweights = rnn_trained_recurrent_weights[j];
                        lweights = PSGetRecurrentNeuronHiddenWeights(&n);
                    }
                } else {
                    eweights = rnn_trained_outer_weights[j];
                    lweights = PSGetNeuronInputWeights(&n);
                }
                testAssertNotNull(lweights, test);
                PSFloat w_val = getRoundedFloatDec(lweights[widx], 2);
                PSFloat expected_w = getRoundedFloatDec(eweights[widx], 2);
                testAssertWithMessage(
                    (w_val == expected_w), test,
                    "Layer[%d][%d]->weights[%d]: %g != %g",
                    i, j, w, w_val, expected_w
                );
            }
        }
    }
    /* free(sequences); */
    return 1;
}

int testRNNOneHot(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    PSModel *onehot_model = PSModelCreate("Onehot RNN");
    PSModel *standard_model = PSModelCreate("Standard RNN");
    PSModel *dummy_model = PSModelCreate("Dummy");
    PSFloat *onehot_data = NULL;
    PSFloat *standard_data = NULL;
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, RECURRENT_MODEL, path), test);
    int loaded = PSModelLoad(onehot_model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    onehot_model->acceleration = PSGlobalAcceleration;
    int ok = PSModelLoad(standard_model, path);
    testAssertWithMessageOrGoto(loaded, final, test, "Failed to load %s", path);
    if (!PSModelIsBuilt(onehot_model)) {
        if (!PSModelBuild(onehot_model)) {
            fprintf(stderr, "\nFailed to build onehot model!\n");
            return 0;
        }
    }
    if (!PSModelIsBuilt(standard_model)) {
        if (!PSModelBuild(standard_model)) {
            fprintf(stderr, "\nFailed to build standard model!\n");
            return 0;
        }
    }
    standard_model->acceleration = PSGlobalAcceleration;
    PSRecurrentNetworkMode onehot_rnn_mode = onehot_model->rnn_mode;
    PSRecurrentNetworkMode std_rnn_mode = standard_model->rnn_mode;
    ok = (onehot_rnn_mode == ManyToMany);
    testAssertWithMessageOrGoto(
        ok, final, test, "OneHot recurrent network mode is: '%s'",
        PSGetRecurrentModeLabel(onehot_rnn_mode)
    );
    ok = (std_rnn_mode == ManyToMany);
    testAssertWithMessageOrGoto(
        ok, final, test, "Standard recurrent network mode is: '%s'",
        PSGetRecurrentModeLabel(std_rnn_mode)
    );
    ok = onehot_model->flags & PS_FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model is not OneHot!", onehot_model->name
    );
    ok = onehot_model->layers[0]->flags & PS_FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model layer[0] is not OneHot!",
        onehot_model->name
    );
    int last_layer = onehot_model->size - 1;
    ok = onehot_model->layers[last_layer]->flags & PS_FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model layer[%d] is not OneHot!",
        onehot_model->name, last_layer
    );
    int no_onehot = ~((unsigned) PS_FLAG_ONEHOT);
    standard_model->flags &= no_onehot;
    standard_model->layers[0]->flags &= no_onehot;
    standard_model->layers[last_layer]->flags &= no_onehot;
    ok = !(standard_model->flags & PS_FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model is OneHot!", standard_model->name
    );
    ok = !(standard_model->layers[0]->flags & PS_FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model layer[0] is OneHot!",
        standard_model->name
    );
    ok = !(standard_model->layers[last_layer]->flags & PS_FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s model layer[%d] is OneHot!",
        standard_model->name, last_layer
    );
    int vector_size = onehot_model->layers[0]->onehot_vector_size;
    ok = (vector_size > 0);
    testAssertWithMessageOrGoto(
        vector_size > 0, final, test,
        "%s model layer[0] vector size is %d", vector_size
    );
    PSLayer *standard_input_layer =
        PSAddLayer(dummy_model, FullyConnected, vector_size, NULL);
    ok = (standard_input_layer != NULL);
    testAssertWithMessageOrGoto(
        standard_input_layer != NULL, final, test,
        "Failed to create standard input layer with size %d",
        vector_size
    );
    PSLayer *curlayer = standard_model->layers[0];
    standard_model->layers[0] = standard_input_layer;
    standard_input_layer->model = standard_model;
    standard_model->input_size = vector_size;
    standard_input_layer->flags |= PS_FLAG_RECURRENT;
    curlayer->model = NULL;
    PSLayerFree(curlayer);
    dummy_model->size = 0;
    dummy_model->layers[0] = NULL;
    PSModelFree(dummy_model);
    dummy_model = NULL;
    int onehot_datalen =
        (int) ((sizeof(rnn_inputs) + sizeof(rnn_labels)) / sizeof(PSFloat));
    int timesteps = (int) rnn_inputs[0];
    ok = (timesteps > 0);
    testAssertWithMessageOrGoto(
        ok, final, test, "timesteps should be > 0, got %d", timesteps
    );
    int standard_datalen = (1 + (timesteps * 2 * vector_size));
    onehot_data = malloc((size_t) (1 + onehot_datalen) * sizeof(PSFloat));
    ok = (onehot_data != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to allocate onehot data of size %d",
        onehot_datalen
    );
    standard_data = malloc((size_t) (1 + standard_datalen) * sizeof(PSFloat));
    ok = (standard_data != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to allocate standard data of size %d",
        standard_datalen
    );
    onehot_data[0] = standard_data[0] = 1.0;
    PSFloat *onehot_p = onehot_data + 1;
    PSFloat *standard_p = standard_data + 1;
    int inputs_len = (int) (sizeof(rnn_inputs) / sizeof(PSFloat)),
        labels_len = (int) (sizeof(rnn_labels) / sizeof(PSFloat)), i;
    memcpy(onehot_p, rnn_inputs, sizeof(rnn_inputs));
    memcpy(
        onehot_p + inputs_len, rnn_labels, sizeof(rnn_labels)
    );
    for (i = 0; i < inputs_len; i++) {
        ok = (rnn_inputs[i] == onehot_p[i]);
        testAssertWithMessageOrGoto(
            ok, final, test, "rnn_inputs[%d] != onehot_data[%d] -> "
            "%g != %g", i, i, rnn_inputs[i], onehot_p[i]
        );
    }
    for (i = 0; i < labels_len; i++) {
        int data_idx = inputs_len + i;
        ok = (rnn_labels[i] == onehot_p[data_idx]);
        testAssertWithMessageOrGoto(
            ok, final, test, "rnn_labels[%d] != onehot_data[%d] -> "
            "%g != %g", i, data_idx, rnn_labels[i], onehot_p[data_idx]
        );
    }
    standard_p[0] = rnn_inputs[0];
    for (i = 1; i < inputs_len; i++) {
        int vecidx = (int) rnn_inputs[i], input_idx = i - 1, j;
        for (j = 0; j < vector_size; j++) {
            int data_idx = 1 + (input_idx * vector_size) + j;
            standard_p[data_idx] = (j == vecidx ? 1.0 : 0.0);
        }
    }
    int labels_offset = 1 + ((inputs_len - 1) * vector_size);
    for (i = 0; i < labels_len; i++) {
        int vecidx = (int) rnn_labels[i], j;
        for (j = 0; j < vector_size; j++) {
            int data_idx = labels_offset + (i * vector_size) + j;
            standard_p[data_idx] = (j == vecidx ? 1.0 : 0.0);
        }
    }
    for (i = 1; i < onehot_datalen; i++) {
        int vecidx = (int) onehot_p[i], onehot_idx = i - 1;
        PSFloat *vec = standard_p + (onehot_idx * vector_size) + 1;
        int idx = arrayMaxIndex(vec, vector_size);
        ok = (idx == vecidx);
        testAssertWithMessageOrGoto(
            ok, final, test, "Onehot[%d] index %d != Standard[%d,%d] %d",
            i, vecidx, ((onehot_idx * vector_size) + 1), vector_size, idx
        );
    }
    if (!PSModelRebuild(standard_model)) {
        fprintf(stderr, "\nFailed to re-build standard model!\n");
        return 0;
    }
    PSFloat onehot_accuracy = PSTest(
        onehot_model, onehot_data, onehot_datalen, NULL
    );
    PSFloat std_accuracy = PSTest(
        standard_model, standard_data, standard_datalen, NULL
    );
    ok = (onehot_model->status != PS_STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Model %s: error during validation",
        onehot_model->name
    );
    ok = (standard_model->status != PS_STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Model %s: error during validation",
        standard_model->name
    );
    ok = (onehot_accuracy == std_accuracy);
    testAssertWithMessageOrGoto(
        ok, final, test, "Onehot accuracy != Non-onehot accuracy: %g != %g",
        onehot_accuracy, std_accuracy
    );
final:
    if (onehot_model != NULL) PSModelFree(onehot_model);
    if (standard_model != NULL) PSModelFree(standard_model);
    if (dummy_model != NULL) PSModelFree(dummy_model);
    if (onehot_data != NULL) free(onehot_data);
    if (standard_data != NULL) free(standard_data);
    return ok;
}

int testLSTMTrain(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    /*int train_data_len = 2 + (LSTM_TIMES * 2);
    UNUSED(train_data_len);*/
    PSFloat *training_data = getTestData(test_case);

    PSTrainingOptions options = {
        .epochs = LSTM_EPOCHS,
        .batch_size = LSTM_BATCHES,
        .learning_rate = LSTM_LEARNING_RATE,
        .flags = PS_TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0,
        .bptt_truncate = 4
    };
    PSTrain(model, training_data, 8, NULL, 0, &options);

    PSLayer *layer = model->layers[1];
    int i, t, w, precision = NORMAL_PRECISION_DEC - 2;

    PSLSTMCell *cell = PSGetLSTMCell(layer);
    testAssertNotNull(cell, test);
    testAssertNotNull(cell->candidate_biases, test);
    testAssertNotNull(cell->input_biases, test);
    testAssertNotNull(cell->output_biases, test);
    testAssertNotNull(cell->forget_biases, test);
    testAssertNotNull(cell->candidate_weights, test);
    testAssertNotNull(cell->input_weights, test);
    testAssertNotNull(cell->output_weights, test);
    testAssertNotNull(cell->forget_weights, test);
    testAssertNotNull(cell->candidate_hidden_weights, test);
    testAssertNotNull(cell->input_hidden_weights, test);
    testAssertNotNull(cell->output_hidden_weights, test);
    testAssertNotNull(cell->forget_hidden_weights, test);
    int input_size = (int) PSGetLayerInputWeightsCount(layer, 1);
    testAssert(input_size > 0, test);
    int times = PSStateSequenceLength(layer);
    for (i = 0; i < layer->size; i++) {
        for (t = 0; t < times; t++) {
            PSFloat h = PSGetState(layer, i, t);
            h = getRoundedFloat(h);
            PSFloat expected = getRoundedFloat(lstm_expected_states[i][t]);
            testAssertWithMessage(
                (h == expected), test,
                "Layer[%d] Neuron[%d]->state[%d]: %g != %g",
                layer->index, i, t, h, expected
            );
            /*int ok = (h == expected);
            printf("H[%d][%d] = %g == %g(%s)\n", t, i, h, expected,
                (ok ? "OK" : "FAIL"));*/
        }
        PSFloat bias = getRoundedFloatDec(cell->candidate_biases[i],precision);
        PSFloat expected = getRoundedFloatDec(expected_bg[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->candidate_bias: %g != %g",
            layer->index, i, bias, expected
        );
        bias = getRoundedFloatDec(cell->input_biases[i], precision);
        expected = getRoundedFloatDec(expected_bi[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->input_bias: %g != %g",
            layer->index, i, bias, expected
        );
        bias = getRoundedFloatDec(cell->output_biases[i], precision);
        expected = getRoundedFloatDec(expected_bo[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->output_bias: %g != %g",
            layer->index, i, bias, expected
        );
        bias = getRoundedFloatDec(cell->forget_biases[i], precision);
        expected = getRoundedFloatDec(expected_bf[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->forget_bias: %g != %g",
            layer->index, i, bias, expected
        );
        int woffs = (i * input_size);
        for (w = 0; w < input_size; w++) {
            int widx = woffs + w;
            PSFloat weight = getRoundedFloatDec(
                cell->candidate_weights[widx], precision
            );
            expected = getRoundedFloatDec(expected_wg[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->candidate_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->input_weights[widx], precision);
            expected = getRoundedFloatDec(expected_wi[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->input_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->output_weights[widx], precision);
            expected = getRoundedFloatDec(expected_wo[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->output_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->forget_weights[widx], precision);
            expected = getRoundedFloatDec(expected_wf[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->forget_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
        }
        woffs = (i * layer->size);
        for (w = 0; w < layer->size; w++) {
            int widx = woffs + w, ewidx = input_size + w;
            PSFloat weight = getRoundedFloatDec(
                cell->candidate_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(expected_wg[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->candidate_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(
                cell->input_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(expected_wi[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->input_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(
                cell->output_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(expected_wo[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->output_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(
                cell->forget_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(expected_wf[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->forget_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
        }
    }
    PSLayer *out = model->layers[model->size - 1];

    for (i = 0; i < out->size; i++) {
        int times = PSStateSequenceLength(out);
        for (t = 0; t < times; t++) {
            PSFloat h = getRoundedFloat(PSGetState(out, i, t));
            PSFloat e = getRoundedFloat(lstm_expected_outputs[t][i]);
            testAssertWithMessage(
                (h == e), test, "Output->Neuron[%d]->output[%d]: %g != %g",
                i, t, h, e
            );
        }
    }
    return 1;
}

int testLSTMBackprop(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    char *model_file = "resources/basic-lstm.psmodel";
    char *inputs_file = "resources/gru-inputs.data";
    char *labels_file = "resources/gru-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (PS_TRAINING_EPOCH_AS_SEQUENCE | PS_TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = ModelBackpropTest(test, model_file,
                               "lstm", inputs_file, labels_file,
                               input_len, label_len, "LSTM", &opts, 0, 3, 3,
                               PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = ModelBackpropTest(test, model_file,
                             "lstm", inputs_file, labels_file,
                              input_len, label_len, "LSTM", &opts, 0, 2, 3,
                              PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = ModelBackpropTest(test, model_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 3, 3,
                             PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = ModelBackpropTest(test, model_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 3, 3,
                             PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = ModelBackpropTest(test, model_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 2, 3,
                             PSAcceleration_None);
    return ok;
}

int testGRUTrain(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    PSFloat *training_data = getTestData(test_case);

    PSTrainingOptions options = {
        .epochs = LSTM_EPOCHS,
        .batch_size = LSTM_BATCHES,
        .learning_rate = 1.5,//LSTM_LEARNING_RATE,
        .flags = PS_TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0,
        .bptt_truncate = 4
    };
    PSTrain(model, training_data, 8, NULL, 0, &options);

    PSLayer *layer = model->layers[1];
    int i, t, w, precision = 2;

    PSGRUCell *cell = PSGetGRUCell(layer);
    testAssertNotNull(cell, test);
    testAssertNotNull(cell->candidate_biases, test);
    testAssertNotNull(cell->update_biases, test);
    testAssertNotNull(cell->reset_biases, test);
    testAssertNotNull(cell->candidate_weights, test);
    testAssertNotNull(cell->update_weights, test);
    testAssertNotNull(cell->reset_weights, test);
    testAssertNotNull(cell->candidate_hidden_weights, test);
    testAssertNotNull(cell->update_hidden_weights, test);
    testAssertNotNull(cell->reset_hidden_weights, test);
    int input_size = (int) PSGetLayerInputWeightsCount(layer, 1);
    testAssert(input_size > 0, test);
    for (i = 0; i < layer->size; i++) {
        int times = PSStateSequenceLength(layer);
        for (t = 0; t < times; t++) {
            PSFloat h = PSGetState(layer, i, t);
            h = getRoundedFloatDec(h, 4);
            PSFloat expected = getRoundedFloatDec(gru_expected_states[i][t],4);
            testAssertWithMessage(
                (h == expected), test,
                "Layer[%d] Neuron[%d]->state[%d]: %g != %g",
                layer->index, i, t, h, expected
            );
        }
        PSFloat bias = getRoundedFloatDec(cell->candidate_biases[i],precision);
        PSFloat expected = getRoundedFloatDec(gru_expected_bg[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->candidate_bias: %g != %g",
            layer->index, i, bias, expected
        );
        bias = getRoundedFloatDec(cell->update_biases[i], precision);
        expected = getRoundedFloatDec(gru_expected_bu[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->update_bias: %g != %g",
            layer->index, i, bias, expected
        );
        bias = getRoundedFloatDec(cell->reset_biases[i], precision);
        expected = getRoundedFloatDec(gru_expected_br[i], precision);
        testAssertWithMessage(
            (bias == expected), test,
            "Layer[%d] Neuron[%d]->reset_bias: %g != %g",
            layer->index, i, bias, expected
        );
        int woffs = (i * input_size);
        for (w = 0; w < input_size; w++) {
            int widx = woffs + w;
            PSFloat weight = getRoundedFloatDec(
                cell->candidate_weights[widx], precision
            );
            expected = getRoundedFloatDec(gru_expected_wg[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->candidate_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->update_weights[widx], precision);
            expected = getRoundedFloatDec(gru_expected_wu[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->update_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(cell->reset_weights[widx], precision);
            expected = getRoundedFloatDec(gru_expected_wr[i][w], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->reset_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
        }
        woffs = (i * layer->size);
        for (w = 0; w < layer->size; w++) {
            int widx = woffs + w, ewidx = input_size + w;
            PSFloat weight = getRoundedFloatDec(
                cell->candidate_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(gru_expected_wg[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->candidate_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(
                cell->update_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(gru_expected_wu[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->update_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
            weight = getRoundedFloatDec(
                cell->reset_hidden_weights[widx], precision
            );
            expected = getRoundedFloatDec(gru_expected_wr[i][ewidx], precision);
            testAssertWithMessage(
                (weight == expected), test,
                "Layer[%d] Neuron[%d]->reset_hidden_weights[%d]: %g != %g",
                layer->index, i, w, weight, expected
            );
        }
    }
    PSLayer *out = model->layers[model->size - 1];

    for (i = 0; i < out->size; i++) {
        int times = PSStateSequenceLength(out);
        for (t = 0; t < times; t++) {
            PSFloat h = getRoundedFloatDec(PSGetState(out, i, t), 2);
            PSFloat e = getRoundedFloatDec(gru_expected_outputs[t][i], 2);
            testAssertWithMessage(
                (h == e), test, "Output->Neuron[%d]->output[%d]: %g != %g",
                i, t, h, e
            );
        }
    }
    return 1;
}

int testGRUBackprop(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    char *model_file = "resources/basic-gru.psmodel";
    char *inputs_file = "resources/gru-inputs.data";
    char *labels_file = "resources/gru-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (PS_TRAINING_EPOCH_AS_SEQUENCE | PS_TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = ModelBackpropTest(test, model_file,
                                 "gru", inputs_file, labels_file,
                                 input_len, label_len, "GRU", &opts, 3, 0, 3,
                                 PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = ModelBackpropTest(test, model_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = ModelBackpropTest(test, model_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = ModelBackpropTest(test, model_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = ModelBackpropTest(test, model_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_None);
    return ok;
}

int testNormalizationLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, NORMALIZATION_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *normlayer = model->layers[1];
    PSLayer *softmax = model->layers[2];
    testAssertNotNull(normlayer, test);
    testAssert(normlayer->type == Normalization, test);
    testAssert(normlayer->size == 4, test);
    testAssertNotNull(softmax, test);
    testAssert(softmax->type == SoftMax, test);
    testAssert(softmax->size == 2, test);
    testAssert(!(normlayer->flags & PS_FLAG_NON_TRAINABLE), test);
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testNormalizationForward(TestCase *test_case, Test *test) {
    static PSFloat data[] = {90.3487, 198.1253, 18.3623, 162.5884, 0, 0};
    static PSFloat normalized[] = {-0.3909, 1.1689, -1.4326, 0.6546};
    PSModel *model = getModel(test_case);
    PSForward(model, data);
    PSLayer *normlayer = model->layers[1];
    testAssertNotNull(normlayer, test);
    PSFloat *states = PSGetStates(normlayer, 0);
    testAssertNotNull(states, test);
    for (int i = 0; i < normlayer->size; i++) {
        PSFloat s = getRoundedFloatDec(states[i], 4);
        PSFloat expected =  getRoundedFloatDec(normalized[i], 4);
        testAssertWithMessage(
            s == expected, test, "Normalized state[%d] != expected: %g != %g",
            i, s, expected
        );
    }
    return 1;
}

int testNormalizationBackprop(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    static PSFloat x[] = {90.3487, 198.1253, 18.3623, 162.5884, 0, 0};
    static PSFloat expected_softmax[] = {0.00236786,0.997632};
    static PSFloat y[] = {1.0, 0.0};
    static PSFloat expected_normdelta[] = {
        -2.04090834,0.0250784121,0.65737313,1.57224989
    };
    static PSFloat expected_prevdelta[] = {
        -2.4473462e-36,4.01267542e-23,6.9116731e-22,-0
    };
    static PSFloat expected_wgrads[] = {
        1.17828763,-0.0144786425,-0.37952444,2.72314405
    };
    int ok = 1;
    PSGradient ***grads = NULL;
    PSModel *model = PSModelCreate("Normalization Backprop");
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, NORMALIZATION_MODEL_BP, path),test);
    ok = PSModelLoad(model, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not load model from '%s'", path
    );
    model->acceleration = PSGlobalAcceleration;
    ok = PSModelBuild(model);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not build model '%s'", model->name
    );
    ok = PSForward(model, x);
    testAssertWithMessageOrGoto(
        ok, final, test, "Forward failed to model %s", model->name
    );
    PSLayer *outlayer = model->layers[model->size - 1];
    testAssertNotNull(outlayer, test);
    PSFloat *outstates = PSGetStates(outlayer, 0);
    testAssertNotNull(outstates, test);
    PSLayer *normlayer = model->layers[2];
    testAssertNotNull(normlayer, test);
    ok = compareArrays(outstates, expected_softmax, outlayer->size, test,
                       "output states", 4, 0);
    if (!ok) goto final;
    grads = backprop(model, x, y, NULL, NULL);
    ok = grads != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for model %s", model->name
    );
    PSGradient **gradients = grads[0];
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL", model->name
    );
    PSMatrix normdelta = normlayer->delta;
    testAssertWithMessageOrGoto(
        normdelta != NULL, final, test,
        "Normalization Layer[%d] has no delta", normlayer->index
    );
    ok = compareArrays(normdelta, expected_normdelta, normlayer->size, test,
                       "normalization delta", 4, 0);
    if (!ok) goto final;
    PSLayer *prev = PSGetPreviousLayer(normlayer);
    testAssertNotNull(prev, test);
    PSMatrix prevdelta = prev->delta;
    testAssertWithMessageOrGoto(
        prevdelta != NULL, final, test,
        " Layer[%d] has no delta", prev->index
    );
    ok = compareArrays(prevdelta, expected_prevdelta, prev->size, test,
                       "previous layer delta", 4, 0);
    if (!ok) goto final;
    testAssertNotNull(normlayer->weights, test);
    PSMatrix weights = normlayer->weights[0];
    PSFloat *biases = normlayer->biases;
    testAssertWithMessageOrGoto(
        weights != NULL, final, test,
        "Normalization Layer[%d] has no weights", normlayer->index
    );
    testAssertWithMessageOrGoto(
        biases != NULL, final, test,
        "Normalization Layer[%d] has no biases", normlayer->index
    );
    int wlen = PSMatrixLength(weights);
    testAssertWithMessageOrGoto(
        wlen == normlayer->size, final, test,
        "Normalization weights should match layer size: %d != %d",
        wlen, normlayer->size
    );
    PSGradient *normgrads = gradients[normlayer->index - 1];
    testAssertNotNull(normgrads, test);
    testAssertWithMessageOrGoto(
        normgrads->weight_count == (uint64_t)normlayer->size, final, test,
        "Normalization gradient weights should match layer size: %d != %d",
        normgrads->weight_count, normlayer->size
    );
    testAssertWithMessageOrGoto(
        normgrads->bias_count == (uint64_t)normlayer->size, final, test,
        "Normalization gradient biases should match layer size: %d != %d",
        normgrads->bias_count, normlayer->size
    );
    testAssertWithMessageOrGoto(
        normgrads->weights != NULL, final, test,
        "Normalization layer[%d] has no gradient weights", normlayer->index
    );
    testAssertWithMessageOrGoto(
        normgrads->biases != NULL, final, test,
        "Normalization layer[%d] has no gradient weights", normlayer->index
    );
    ok = compareArrays(normgrads->biases, normdelta, normlayer->size, test,
                       "normalization biases", 4, 0);
    testAssertWithMessageOrGoto(
        ok, final, test, "Normalization gradient biases should match "
        "normalization delta%s",""
    );
    ok = compareArrays(normgrads->weights, expected_wgrads, wlen, test,
                       "normalization weights", 4, 0);
    testAssertWithMessageOrGoto(
        ok, final, test, "Normalization gradient weights mismatch%s", ""
    );
final:
    if (model != NULL) {
        if (grads != NULL) PSDeleteGradientsChain(grads, model);
        PSModelFree(model);
    }
    return ok;
}

int testDropoutLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, DROPOUT_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *dropout_layer = model->layers[2];
    PSLayer *softmax = model->layers[model->size - 1];
    testAssertNotNull(dropout_layer, test);
    testAssert(dropout_layer->type == Dropout, test);
    testAssert(dropout_layer->size == 4, test);
    testAssertNotNull(softmax, test);
    testAssert(softmax->type == SoftMax, test);
    testAssert(softmax->size == 2, test);
    PSFloat dropout = PSGetDropout(dropout_layer);
    dropout = getRoundedFloatDec(dropout, 1);
    PSFloat expected = getRoundedFloatDec(0.9, 1);
    testAssertWithMessage(
        dropout == expected, test, "Dropout should be %g, got %g",
        expected, dropout
    );
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testDropoutForward(TestCase *test_case, Test *test) {
    static PSFloat data[] = {90.3487, 198.1253};
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    ok = PSForward(model, data);
    testAssert(ok, test);
    PSLayer *dropout_layer = model->layers[2];
    PSFloat dropout = PSGetDropout(dropout_layer);
    PSLayer *prev = PSGetPreviousLayer(dropout_layer);
    testAssertNotNull(prev, test);
    PSFloat *prev_states = PSGetStates(prev, 0);
    testAssertNotNull(prev_states, test);
    PSFloat *dropout_states = PSGetStates(dropout_layer, 0);
    testAssertNotNull(dropout_states, test);
    for (int i = 0; i < dropout_layer->size; i++) {
        PSFloat prev_state = prev_states[i];
        PSFloat dropped_out = dropout_states[i];
        testAssertWithMessage(
            dropped_out == (prev_state * dropout), test,
            "dropped out state[%d] expected to be %g (%g * %g), "
            "got %g", i, (prev_state * dropout), prev_state, dropout,
            dropped_out
        );
    }
    return ok;
}

int testDropoutBackprop(TestCase *test_case, Test *test) {
    static PSFloat x[] = {90.3487, 198.1253};
    static PSFloat y[] = {1.0, 0.0};
    PSGradient ***grads = NULL;
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSLayer *dropout_layer = model->layers[2];
    PSLayer *prev = PSGetPreviousLayer(dropout_layer);
    testAssertNotNull(prev, test);
    int old_status = model->status;
    model->status = PS_STATUS_TRAINING;
    grads = backprop(model, x, y, NULL, NULL);
    ok = grads != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for model %s", model->name
    );
    PSGradient **gradients = grads[0];
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL", model->name
    );
    PSMatrix prevdelta = prev->delta;
    testAssertWithMessageOrGoto(
        prevdelta != NULL, final, test,
        "Layer[%d] has no delta", prev->index
    );
    PSFloat *dropout_mask = PSGetDropoutMask(dropout_layer, 0);
    testAssertWithMessageOrGoto(
        dropout_mask != NULL, final, test,
        "Dropout Layer[%d] has no dropout mask", dropout_layer->index
    );
    for (int i = 0; i < dropout_layer->size; i++) {
        int dropped_out = dropout_mask[i]  == 0;
        if (!dropped_out) continue;
        testAssertWithMessageOrGoto(
            prevdelta[i] == 0.0, final, test, "Layer[%d] delta[%d] expected "
            "to be 0.0, got: %g", prev->index, i, prevdelta[i]
        );
    }
final:
    model->status = old_status;
    if (grads != NULL) PSDeleteGradientsChain(grads, model);
    return ok;
}

int testConcatOperatorLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_CONCAT_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = model->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSConcatenateOperator, test);
    int expected_size = model->layers[1]->size + model->layers[2]->size,
        providers_count;
    testAssert(expected_size == oplayer->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == model->layers[1], test);
    testAssert(providers[1] == model->layers[2], test);
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testConcatOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(model, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat *concatenated = PSGetStates(op_layer, 0);
    testAssertNotNull(concatenated, test);
    PSFloat *l1_out = PSGetStates(l1, 0);
    testAssertNotNull(l1_out, test);
    PSFloat *l2_out = PSGetStates(l2, 0);
    testAssertNotNull(l2_out, test);
    ok = compareArrays(concatenated, l1_out, l1->size, test,
                       NULL, 0, 0);
    if (!ok) return 0;
    ok = compareArrays(concatenated + l1->size, l2_out, l2->size, test,
                       NULL, 0, 0);
    return ok;
}

int testConcatOperatorBackprop(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[model->output_size];
    for (uint32_t i = 0; i < model->output_size; i++)
        y[i] = PSGaussianRandom(0, 1);
    PSGradient ***grads = backprop(model, x, y, NULL, NULL);
    testAssertNotNull(grads, test);
    testAssertWithMessageOrGoto(
        l1->delta != NULL, final, test, "layer[%d] delta is null", 1
    );
    testAssertWithMessageOrGoto(
        l2->delta != NULL, final, test, "layer[%d] delta is null", 2
    );
    testAssertWithMessageOrGoto(
        op_layer->delta != NULL, final, test, "layer[%d] delta is null", 3
    );
    l1_delta = PSMatrixDupShape(l1->delta);
    testAssertWithMessageOrGoto(
        l1_delta != NULL, final, test, "could not duplicate layer[%d] delta "
        "shape", 1
    );
    ok = PSUpdateDelta(l1_delta, l2->delta, l2->weights[0], 1,
                       model->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = model->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(op_layer->delta, l1_delta, l1->size, test,
                       NULL, 0, 4);
    if (!ok) return 0;
    ok = compareArrays(op_layer->delta + l1->size, l2->delta, l2->size, test,
                       NULL, 0, 4);
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, model);
    PSMatrixFree(l1_delta);
    return ok;
}

int testAddOperatorLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_ADD_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = model->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSAddOperator, test);
    int expected_size = model->layers[1]->size, providers_count;
    testAssert(expected_size == oplayer->size, test);
    testAssert(expected_size == model->layers[2]->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == model->layers[1], test);
    testAssert(providers[1] == model->layers[2], test);
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testAddOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(model, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat *opstates = PSGetStates(op_layer, 0);
    testAssertNotNull(opstates, test);
    PSFloat *l1_out = PSGetStates(l1, 0);
    testAssertNotNull(l1_out, test);
    PSFloat *l2_out = PSGetStates(l2, 0);
    testAssertNotNull(l2_out, test);
    for (int i = 0; i < op_layer->size; i++) {
        PSFloat l1state = l1_out[i];
        PSFloat l2state = l2_out[i];
        PSFloat opstate = opstates[i];
        PSFloat expected = l1state + l2state;
        ok = compareFloats(opstate, expected, 0, 4);
        testAssertWithMessage(
            ok, test, "Operator layer state[%d] != expected: %g != %g "
            "(%g + %g)", i, opstate, expected, l1state, l2state
        );
    }
    return ok;
}

int testAddOperatorBackprop(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[model->output_size];
    for (uint32_t i = 0; i < model->output_size; i++)
        y[i] = PSGaussianRandom(0, 1);
    PSGradient ***grads = backprop(model, x, y, NULL, NULL);
    testAssertNotNull(grads, test);
    testAssertWithMessageOrGoto(
        l1->delta != NULL, final, test, "layer[%d] delta is null", 1
    );
    testAssertWithMessageOrGoto(
        l2->delta != NULL, final, test, "layer[%d] delta is null", 2
    );
    testAssertWithMessageOrGoto(
        op_layer->delta != NULL, final, test, "layer[%d] delta is null", 3
    );
    l1_delta = PSMatrixDupShape(l1->delta);
    testAssertWithMessageOrGoto(
        l1_delta != NULL, final, test, "could not duplicate layer[%d] delta "
        "shape", 1
    );
    ok = PSUpdateDelta(l1_delta, l2->delta, l2->weights[0], 1,
                       model->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = model->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(l1_delta, l2->delta, l1->size, test,
                       NULL, 0, 4);
    if (!ok) return 0;
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, model);
    PSMatrixFree(l1_delta);
    return ok;
}

int testMulOperatorLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_MUL_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = model->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSMultiplyOperator, test);
    int expected_size = model->layers[1]->size, providers_count;
    testAssert(expected_size == oplayer->size, test);
    testAssert(expected_size == model->layers[2]->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == model->layers[1], test);
    testAssert(providers[1] == model->layers[2], test);
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testMulOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(model, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat *opstates = PSGetStates(op_layer, 0);
    testAssertNotNull(opstates, test);
    PSFloat *l1_out = PSGetStates(l1, 0);
    testAssertNotNull(l1_out, test);
    PSFloat *l2_out = PSGetStates(l2, 0);
    testAssertNotNull(l2_out, test);
    for (int i = 0; i < op_layer->size; i++) {
        PSFloat l1state = l1_out[i];
        PSFloat l2state = l2_out[i];
        PSFloat opstate = opstates[i];
        PSFloat expected = l1state * l2state;
        ok = compareFloats(opstate, expected, 0, 4);
        testAssertWithMessage(
            ok, test, "Operator layer state[%d] != expected: %g != %g "
            "(%g * %g)", i, opstate, expected, l1state, l2state
        );
    }
    return ok;
}

int testMulOperatorBackprop(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = model->layers[3], *l1 = model->layers[1],
            *l2 = model->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[] = {-1.19145763, -0.723464847, 0.460539848};
    PSFloat l1_expct[] = {-0.423889, 0.112024, 4.51742, -1.43311, -1.15413};
    PSFloat l2_expct[] = {-0.0644108, -0.0631921, 1.18236, -2.13912, -1.77627};
    PSGradient ***grads = backprop(model, x, y, NULL, NULL);
    testAssertNotNull(grads, test);
    testAssertWithMessageOrGoto(
        l1->delta != NULL, final, test, "layer[%d] delta is null", 1
    );
    testAssertWithMessageOrGoto(
        l2->delta != NULL, final, test, "layer[%d] delta is null", 2
    );
    testAssertWithMessageOrGoto(
        op_layer->delta != NULL, final, test, "layer[%d] delta is null", 3
    );
    l1_delta = PSMatrixDupShape(l1->delta);
    testAssertWithMessageOrGoto(
        l1_delta != NULL, final, test, "could not duplicate layer[%d] delta "
        "shape", 1
    );
    ok = PSUpdateDelta(l1_delta, l2->delta, l2->weights[0], 1,
                       model->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = model->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(l1_delta, l1_expct, l1->size, test, "L1 DELTA: ", 0, 4);
    if (!ok) goto final;
    ok = compareArrays(l2->delta, l2_expct, l2->size, test, "L2 DELTA", 0, 4);
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, model);
    PSMatrixFree(l1_delta);
    return ok;
}

int testPositionalEmbedLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, POSITIONAL_MODEL, path), test);
    int loaded = PSModelLoad(model, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    model->acceleration = PSGlobalAcceleration;
    PSLayer *poslayer = model->layers[model->size - 1];
    testAssertNotNull(poslayer, test);
    testAssert(poslayer->type == PositionalEncoding, test);
    int expected_size = 6;
    testAssert(expected_size == poslayer->size, test);
    if (!PSModelIsBuilt(model)) {
        int built = PSModelBuild(model);
        testAssert(built, test);
    }
    return 1;
}

int testPositionalEmbedForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    PSFloat inputs[] = {5, 1, 9, 7, 3, 5};
    PSFloat expected_y[] = {
        0.841471, 1.5403, 0.0463992, 1.99892, 0.00215443, 2, 1.25359,
        -0.370828, 0.452098, 1.91293, 0.0215431, 1.99981, 1.56628, 0.337755,
        0.411923, 1.94337, 0.0193893, 1.99988, 0.28224, -1.97998, 0.277596,
        1.98064, 0.0129265, 1.99996, -1.71573, -0.369981, 0.4146, 1.956,
        0.0193896, 1.9999
    };
    ok = PSForward(model, inputs);
    testAssert(ok, test);
    PSLayer *poslayer = model->layers[model->size - 1];
    PSFloat *states = PSGetStates(poslayer, 0);
    testAssertNotNull(states, test);
    uint64_t explen = (uint64_t) (sizeof(expected_y) / sizeof(PSFloat));
    testAssert(PSMatrixLength(states) == explen, test);
    ok = compareArrays(states, expected_y, explen, test,
                       "Positional Layer Outputs:", 0, 4);
    return ok;
}

int encoderDecoderSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("Encoder-Decoder");
    if (model == NULL) {
        PSErr(NULL, "\nCould not create model!");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    test_case->data[1] = NULL;
    return 1;
}

int encoderDecoderTeardown(TestCase *test_case) {
    PSModel *model = getModel(test_case);
    if (model) PSModelFree(model);
    return 1;
}

int testEncodedDecoderLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, ENCDEC_BASIC_MODEL, path), test
    );
    int ok = PSModelLoad(model, path);
    testAssert(ok, test);
    testAssert(PSModelChainLength(model) == 2, test);
    return ok;
}

int testEncodedDecoderSave(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    testAssert(model->size > 0, test);
    testAssert(PSModelChainLength(model) == 2, test);
    const char *fpath = "/tmp/psyc-enc-dec-test.psmodel";
    int ok = PSModelSave(model, fpath);
    testAssert(ok, test);
    PSModel *loaded = PSModelCreate(NULL);
    testAssertNotNull(loaded, test);
    ok = PSModelLoad(loaded, fpath);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to load model from '%s'", fpath
    );
    ok = compareModelChain(model, loaded, test);
final:
    if (loaded != NULL) PSModelFree(loaded);
    return ok;
}

int testEncodedDecoderClone(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    testAssert(model->size > 0, test);
    testAssert(PSModelChainLength(model) == 2, test);
    PSModel *clone = PSModelClone(model, 0);
    testAssertNotNull(clone, test);
    int ok = compareModelChain(model, clone, test);
final:
    if (clone != NULL) PSModelFree(clone);
    return ok;
}

int testEncodedDecoderPredict(TestCase *test_case, Test *test) {
    int ok = 1;
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    testAssert(model->size > 0, test);
    testAssert(PSModelChainLength(model) == 2, test);
    PSFloat x[] = {2, 2, 1};
    PSFloat expected[2][4] = {
          {0.00577807,0.0132289,0.978728,0.00226505},
          {0.998611,0.000101474,0.00128536,2.34558e-06}
    };
    PSModel *decoder = model->next;
    testAssertNotNull(decoder, test);
    decoder->sequence_settings.end = 0;
    ok = PSAutoregression(model, x, 0, NULL);
    testAssert(ok, test);
    PSLayer *out = PSGetOutputLayer(model);
    testAssertNotNull(out, test);
    testAssert(out->model == decoder, test);
    int seqlen = PSStateSequenceLength(out), t;
    testAssert(seqlen == 2, test);
    for (t = 0; t < seqlen; t++) {
        PSFloat *states = PSGetStates(out, t);
        testAssertWithMessage(states != NULL, test, "NULL states at t[%d]", t);
        ok = compareArrays(states, expected[t], out->size, test,
                           NULL, 0, 4);
    }
    return ok;
}

Test *encDecBackpropTest = NULL;
static int beforeDecoderForward(PSModel *decoder,
                                PSFloat *inputs, int seqlen, int backprop,
                                void *opts)
{
    UNUSED(inputs);
    UNUSED(seqlen);
    UNUSED(backprop);
    UNUSED(opts);
    assert(encDecBackpropTest != NULL);
    Test *test = encDecBackpropTest;
    int ok = 1;
    PSModelLink *link = decoder->previous_model_link;
    testAssertNotNull(link, test);
    testAssertNotNull(link->layer, test);
    testAssertNotNull(link->previous_layer, test);
    testAssertWithMessage(link->layer->initial_states != NULL, test,
                          "Decoder layer[%d] has no initial_states",
                          link->layer->index);
    PSFloat *encoder_output_states = PSGetOutputs(link->previous_layer);
    testAssertWithMessage(encoder_output_states != NULL, test,
                          "Missing Encoder layer[%d] output states",
                          link->layer->index);
    ok = compareArrays(link->layer->initial_states, encoder_output_states,
                       link->layer->size, test, "Decoder initial states",0,0);
    return ok;
}

int beforeEncoderBackprop(PSModel *encoder, PSFloat *y,
                          PSTrainingOptions *opts, PSGradient **gradients)
{
    UNUSED(y);
    UNUSED(opts);
    UNUSED(gradients);
    assert(encDecBackpropTest != NULL);
    Test *test = encDecBackpropTest;
    int ok = 1;
    PSModel *decoder = encoder->next;
    testAssertNotNull(decoder, test);
    PSModelLink *link = decoder->previous_model_link;
    testAssertNotNull(link, test);
    testAssertNotNull(link->layer, test);
    testAssertNotNull(link->previous_layer, test);
    testAssertWithMessage(link->layer->delta != NULL, test,
                          "Decoder layer[%d] has no delta",
                          link->layer->index);
    testAssertWithMessage(link->previous_layer->delta != NULL, test,
                          "Encoder layer[%d] has no delta",
                          link->previous_layer->index);
    ok = compareArrays(link->layer->delta, link->previous_layer->delta,
                       link->layer->size, test, "Decoder delta",0,0);
    return ok;
}

int testEncodedDecoderBackprop(TestCase *test_case, Test *test) {
    encDecBackpropTest = test;
    int ok = 1;
    int elements_count = 1;
    PSFloat training_data[] = {
        2, 2, 1, 1, 2,
    };
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    testAssert(model->size > 0, test);
    testAssert(PSModelChainLength(model) == 2, test);
    PSModel *decoder = model->next;
    testAssertNotNull(decoder, test);
    decoder->sequence_settings.end = -1;
    PSLayer *out = PSGetOutputLayer(model);
    testAssertNotNull(out, test);
    testAssert(out->flags & PS_FLAG_ONEHOT, test);
    PSTrainingOptions opts = {
        .flags = PS_TRAINING_FLAG_TEACHER_FORCING | PS_TRAINING_FLAG_SEQ2SEQ,
        .epochs = 1,
        .learning_rate = 0.3,
        .batch_size = 1,
        .bptt_truncate = 0,
    };
    model->beforeBackprop = beforeEncoderBackprop;
    decoder->beforeForward = beforeDecoderForward;
    PSFloat *seq[] = {NULL};
    seq[0] = training_data;
    PSFloat loss = updateModelParameters(
        model, training_data, 1, elements_count, 0.3, &opts, seq
    );
    UNUSED(loss);
    ok = (model->status != PS_STATUS_ERROR);
    if (ok) ok = (decoder->status != PS_STATUS_ERROR);
    encDecBackpropTest = NULL;
    return ok;
}

int attentionSetup(TestCase *test_case) {
    PSModel *model = PSModelCreate("Attention Test");
    if (model == NULL) {
        PSErr(NULL, "\nCould not create model!");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = model;
    test_case->data[1] = NULL;
    return 1;
}

int attentionTeardown(TestCase *test_case) {
    PSModel *model = getModel(test_case);
    if (model) PSModelFree(model);
    return 1;
}

int testGenericAttentionBackprop(PSModel *model, char *file_prefix,
                                 Test *test)
{
    int ok = 1, i;
    uint64_t xlen = 0, ylen = 0, olen = 0;
    int old_status = model->status;
    PSGradient ***grads = NULL;
    FILE *f = NULL;
    PSFloat *x = NULL, *y = NULL, *inputs = NULL, *targets = NULL,
            *outputs = NULL;
    PSFloat **expgrads_w = NULL, **expgrads_b = NULL;
    PSModel *tail = PSModelChainTail(model);
    testAssertNotNull(tail, test);
    PSLayer *attn_layer = NULL;
    for (i = 0; i < tail->size; i++) {
        PSLayer *l = tail->layers[i];
        if (l != NULL && l->type == Attention) {
            attn_layer = l;
            break;
        }
    }
    testAssertWithMessage(attn_layer != NULL, test,
                          "attention layer not found in model %s",
                          model->name);
    char fname[PATH_MAX] = {0};
    char path[PATH_MAX] = {0};
    /* Load inputs */
    sprintf(fname, "resources/%s%s.data", file_prefix, "-inputs");
    testAssert(
        joinPath(executable_path, fname, path), test
    );
    f = fopen(path, "r");
    testAssertWithMessage(f != NULL, test, "could not open %s", path);
    inputs = readSerializedFloatArray(f, ",", &xlen, 1024, 10);
    ok = (inputs != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not read inputs from %s", path
    );
    fclose(f);
    f = NULL;
    /* Load targets */
    sprintf(fname, "resources/%s%s.data", file_prefix, "-targets");
    testAssert(
        joinPath(executable_path, fname, path), test
    );
    f = fopen(path, "r");
    testAssertWithMessage(f != NULL, test, "could not open %s", path);
    targets = readSerializedFloatArray(f, ",", &ylen, 1024, 10);
    ok = (targets != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not read targets from %s", path
    );
    fclose(f);
    f = NULL;
    /* Load outputs */
    sprintf(fname, "resources/%s%s.data", file_prefix, "-outputs");
    testAssert(
        joinPath(executable_path, fname, path), test
    );
    f = fopen(path, "r");
    testAssertWithMessage(f != NULL, test, "could not open %s", path);
    outputs = readSerializedFloatArray(
        f, ",", &olen, attn_layer->size * 100, attn_layer->size
    );
    ok = (outputs != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not read outputs from %s", path
    );
    fclose(f);
    f = NULL;
    /* Load gradients */
    expgrads_w = calloc(attn_layer->weight_types, sizeof(PSFloat *));
    ok = (expgrads_w != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not allocate gradients%s",""
    );
    expgrads_b = calloc(attn_layer->weight_types, sizeof(PSFloat *));
    ok = (expgrads_b != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not allocate gradients%s",""
    );
    char suffix[25] = {0};
    for (i = 0; i < attn_layer->weight_types; i++) {
        if (attn_layer->weights[i] == NULL) continue;
        uint64_t wlen = PSMatrixLength(attn_layer->weights[i]),
                 blen = (i == PS_SCORES_PROJ_IDX ? 1 : attn_layer->size),
                 exp_wlen = 0, exp_blen = 0;
        /* Load saved weight gradients */
        sprintf(suffix, "-%d-wgrads", i);
        sprintf(fname, "resources/%s%s.data", file_prefix, suffix);
        testAssert(
            joinPath(executable_path, fname, path), test
        );
        f = fopen(path, "r");
        testAssertWithMessage(f != NULL, test, "could not open %s", path);
        expgrads_w[i] = readSerializedFloatArray(f, ",", &exp_wlen, wlen, wlen);
        ok = (expgrads_w[i] != NULL);
        testAssertWithMessageOrGoto(
            ok, final, test, "could not read weight gradients "
            " %d from %s", i, path
        );
        fclose(f);
        f = NULL;
        ok = (wlen == exp_wlen);
        testAssertWithMessageOrGoto(
            ok, final, test, "expected weights[%d] length is %d, got %d",
            i, exp_wlen, wlen
        );
        /* Load saved bias gradients */
        sprintf(suffix, "-%d-bgrads", i);
        sprintf(fname, "resources/%s%s.data", file_prefix, suffix);
        testAssert(
            joinPath(executable_path, fname, path), test
        );
        f = fopen(path, "r");
        testAssertWithMessage(f != NULL, test, "could not open %s", path);
        expgrads_b[i] = readSerializedFloatArray(f, ",", &exp_blen, blen, blen);
        ok = (expgrads_b[i] != NULL);
        testAssertWithMessageOrGoto(
            ok, final, test, "could not read bias gradients "
            " %d from %s", i, path
        );
        fclose(f);
        f = NULL;
        ok = (wlen == exp_wlen);
        testAssertWithMessageOrGoto(
            ok, final, test, "expected bias[%d] length is %d, got %d",
            i, exp_blen, blen
        );
    }
    x = malloc((xlen + ylen) * sizeof(PSFloat));
    testAssertWithMessageOrGoto(
        x != NULL, final, test, "could not allocate '%s'", "x"
    );
    PSVectorCopy(x, inputs, xlen);
    y = x + xlen;
    PSVectorCopy(y, targets, ylen);
    PSTrainingOptions topts = {0};
    PSSetDefaultTrainingOptions(&topts);
    PSModelSetStatus(model, PS_STATUS_TRAINING, NULL);
    grads = backprop(model, x, y, &topts, NULL);
    ok = (grads != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "backpropagation failed%s",""
    );
    PSMatrix attn_out = attn_layer->states;
    ok = (attn_out != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "attention layer has not outputs%s",""
    );
    ok = compareArrays(attn_out, outputs, PSMatrixLength(attn_out), test,
                       "Attention Layer Outputs:", 0, 4);
    if (!ok) goto final;
    PSGradient **n_grads = grads[attn_layer->model->index];
    ok = (n_grads != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "missing gradients for model %d",
        attn_layer->model->index
    );
    int gidx = attn_layer->index - 1;
    PSGradient *attn_grads = n_grads[gidx];
    ok = (attn_grads != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "missing gradients for attention layer at index %d",
        gidx
    );
    PSFloat *wgrads_p = attn_grads->weights,
            *bgrads_p = attn_grads->biases;
    ok = (wgrads_p  != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "missing attention gradient weights%s",""
    );
    ok = (bgrads_p  != NULL);
    testAssertWithMessageOrGoto(
        ok, final, test, "missing attention gradient biases%s",""
    );
    char cmpdescr[255] = {0};
    for (i = 0; i < attn_layer->weight_types; i++) {
        if (attn_layer->weights[i] == NULL) continue;
        int wlen = PSMatrixLength(attn_layer->weights[i]),
            blen = (i == PS_SCORES_PROJ_IDX ? 1 : attn_layer->size);
        PSFloat *exp_wg = expgrads_w[i];
        PSFloat *exp_bg = expgrads_b[i];
        ok = (exp_wg != NULL);
        testAssertWithMessageOrGoto(
            ok, final, test, "missing expected weight gradients %d", i
        );
        ok = (exp_bg != NULL);
        testAssertWithMessageOrGoto(
            ok, final, test, "missing expected bias gradients %d", i
        );
        snprintf(cmpdescr, 255, "Attention weight gradients[%d]", i);
        ok = compareArrays(wgrads_p, exp_wg, wlen, test, cmpdescr, 0, 4);
        if (!ok) goto final;
        snprintf(cmpdescr, 255, "Attention bias gradients[%d]", i);
        ok = compareArrays(bgrads_p, exp_bg, blen, test, cmpdescr, 0, 4);
        if (!ok) goto final;
        wgrads_p += wlen;
        bgrads_p += blen;
    }
final:
    if (f != NULL) fclose(f);
    if (expgrads_w != NULL) {
        for (i = 0; i < attn_layer->weight_types; i++) {
            PSFloat *g = expgrads_w[i];
            free(g);
        }
        free(expgrads_w);
    }
    if (expgrads_b != NULL) {
        for (i = 0; i < attn_layer->weight_types; i++) {
            PSFloat *g = expgrads_b[i];
            free(g);
        }
        free(expgrads_b);
    }
    free(x);
    free(inputs);
    free(targets);
    free(outputs);
    PSModelSetStatus(model, old_status, NULL);
    if (grads != NULL) PSDeleteGradientsChain(grads, model);
    return ok;
}

int testAdditiveAttentionLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, ADD_ATTENTION_MODEL, path), test
    );
    int ok = PSModelLoad(model, path);
    testAssert(ok, test);
    testAssert(PSModelChainLength(model) == 2, test);
    return ok;
}


int testAdditiveAttentionBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    int ok = testGenericAttentionBackprop(
        model, "pretrained.additive-attention", test
    );
    return ok;
}

int testDotAttentionLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, DOT_ATTENTION_MODEL, path), test
    );
    int ok = PSModelLoad(model, path);
    testAssert(ok, test);
    testAssert(PSModelChainLength(model) == 2, test);
    return ok;
}


int testDotAttentionBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    int ok = testGenericAttentionBackprop(
        model, "pretrained.dot-product-attention", test
    );
    return ok;
}

int testMHAttentionLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, MH_ATTENTION_MODEL, path), test
    );
    int ok = PSModelLoad(model, path);
    testAssert(ok, test);
    testAssert(PSModelChainLength(model) == 2, test);
    return ok;
}


int testMHAttentionBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    int ok = testGenericAttentionBackprop(
        model, "pretrained.mh-dot-product-attention", test
    );
    return ok;
}

int testMHCausalSelfAttentionLoad(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, MH_CAUSAL_SELFATTENTION_MODEL, path), test
    );
    int ok = PSModelLoad(model, path);
    testAssert(ok, test);
    testAssert(PSModelChainLength(model) == 1, test);
    return ok;
}


int testMHCausalSelfAttentionBackprop(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    testAssertNotNull(model, test);
    int ok = testGenericAttentionBackprop(
        model, "pretrained.mh-causal-attention", test
    );
    return ok;
}

int testGenericClone(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    PSModel *clone = PSModelClone(model, 0);
    testAssertNotNull(clone, test);
    int ok = compareModels(model, clone, test);
    PSModelFree(clone);
    return ok;
}

int testGenericSave(TestCase *test_case, Test *test) {
    PSModel *model = getModel(test_case);
    assert(model->size > 0);
    char tmpfile[PATH_MAX];
    getTmpFileName("tests-save-nn", ".psmodel", tmpfile);
    int ok = PSModelSave(model, tmpfile);
    testAssertWithMessage(ok, test, "Could not save model %s", model->name);
    PSModel *clone = PSModelCreate("Clone Test Model");
    testAssertNotNull(clone, test);
    ok = PSModelLoad(clone, tmpfile);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not load model from %s",tmpfile
    );
    ok = compareModels(model, clone, test);
    remove(tmpfile);
final:
    PSModelFree(clone);
    return ok;
}

int compareModels(PSModel *model, PSModel *clone, Test* test) {
    int ok = 1, i, k;
    ok = model->size == clone->size;
    testAssertWithMessage(
        (model->size == clone->size), test,
        "Source size %d != Clone size %d",
        model->size, clone->size
    );
    int recurrent_source = PSIsRecurrent(model),
        recurrent_clone = PSIsRecurrent(clone);
    testAssertWithMessage(
        (recurrent_source == recurrent_clone), test,
        "Source recurrent: %d, Clone recurrent: %d",
        recurrent_source, recurrent_clone
    );
    if (recurrent_source) {
        PSLayer *src_first_recurrent_layer = PSGetFirstRecurrentLayer(model),
                *src_last_recurrent_layer = PSGetLastRecurrentLayer(model),
                *clone_first_recurrent_layer = PSGetFirstRecurrentLayer(clone),
                *clone_last_recurrent_layer = PSGetLastRecurrentLayer(clone);
        PSRecurrentNetworkMode srcmode = model->rnn_mode,
                               clonemode = clone->rnn_mode;
        int src_max_steps = model->sequence_settings.max_length,
            clone_max_steps = clone->sequence_settings.max_length,
            src_eos = model->sequence_settings.end,
            clone_eos = clone->sequence_settings.end;
        testAssertWithMessage(
            (srcmode == clonemode), test,
            "Recurrent source mode: '%s', Recurrent clone mode: '%s'",
            PSGetRecurrentModeLabel(srcmode),
            PSGetRecurrentModeLabel(clonemode)
        );
        testAssertWithMessage(
            (src_max_steps == clone_max_steps), test,
             "model->sequence_settings.max_length != "
             "clone->sequence_settings.max_length: %d != %d",
             src_max_steps, clone_max_steps
        );
        testAssertWithMessage(
            (src_eos == clone_eos), test,
             "model->sequence_settings.end != "
             "clone->sequence_settings.end: %d != %d ",
             src_eos, clone_eos
        );
        if (src_first_recurrent_layer == NULL)
            testAssertNull(clone_first_recurrent_layer, test);
        else {
            testAssertNotNull(clone_first_recurrent_layer, test);
            int src_idx = src_first_recurrent_layer->index;
            int cln_idx = clone_first_recurrent_layer->index;
            testAssertWithMessage(
                (src_idx == cln_idx), test,
                "model first recurrent layer index != "
                "clone first recurrent layer index: %d != %d",
                src_idx, cln_idx
            );
        }
        if (src_last_recurrent_layer == NULL)
            testAssertNull(clone_last_recurrent_layer, test);
        else {
            testAssertNotNull(clone_last_recurrent_layer, test);
            int src_idx = src_last_recurrent_layer->index;
            int cln_idx = clone_last_recurrent_layer->index;
            testAssertWithMessage(
                (src_idx == cln_idx), test,
                "model last recurrent layer index != "
                "clone last recurrent layer index: %d != %d",
                src_idx, cln_idx
            );
        }
    }
    testAssertWithMessage(
        model->flags == clone->flags, test,
        "Source flags %d != Clone flags %d",
        model->flags, clone->flags
    );
    for (i = 0; i < model->size; i++) {
        PSLayer *orig_l = model->layers[i];
        PSLayer *clone_l = clone->layers[i];
        PSLayerType otype = orig_l->type;
        PSLayerType ctype = clone_l->type;
        testAssertWithMessage(
            (otype == ctype), test,
            "Layer[%d]: Source type %s != Clone type %s",
            PSGetLayerTypeLabel(orig_l), PSGetLayerTypeLabel(clone_l)
        );
        int o_size = orig_l->size;
        int c_size = clone_l->size;
        testAssertWithMessage(
            (o_size == c_size), test,
            "Layer[%d]: Source size %d != Clone size %d",
            i, o_size, c_size
        );
        int o_flags = orig_l->flags;
        int c_flags = clone_l->flags;
        testAssertWithMessage(
            (o_flags == c_flags), test,
            "Layer[%d]: Source flags %d != Clone flags %d",
            i, o_flags, c_flags
        );
        if (Dropout == orig_l->type) {
            PSFloat o_dropout = PSGetDropout(orig_l);
            PSFloat c_dropout = PSGetDropout(clone_l);
            testAssertWithMessage(
                (o_dropout == c_dropout), test,
                "Layer[%d]: Source dropout %g != Clone dropout %g",
                i, o_dropout, c_dropout
            );
            continue;
        } else if (OperatorLayer == orig_l->type) {
            PSOperatorType orig_op = PSGetOperatorLayerType(orig_l);
            PSOperatorType clone_op = PSGetOperatorLayerType(clone_l);
            testAssertWithMessage(
                orig_op == clone_op, test,
                "Layer[%d]: Source operator != Clone operator -> %d (%s) != "
                "%d (%s)", i, orig_op, PSGetOperatorLayerTypeLabel(orig_op),
                clone_op, PSGetOperatorLayerTypeLabel(clone_op)
            );
            int orig_providers_c = 0, clone_providers_c = 0;
            PSLayer **orig_providers = NULL, **clone_providers = NULL;
            orig_providers = PSGetOperatorLayerProviders(
                orig_l, &orig_providers_c
            );
            clone_providers = PSGetOperatorLayerProviders(
                clone_l, &clone_providers_c
            );
            testAssertWithMessage(
                orig_providers_c == clone_providers_c, test,
                "Layer[%d]: Source providers count != Clone providers count > "
                "%d != %d", i, orig_providers_c, clone_providers_c
            );
            testAssertNotNull(orig_providers, test);
            testAssertNotNull(clone_providers, test);
            for (int j = 0; j < orig_providers_c; j++) {
                PSLayer *orig_prv = orig_providers[j];
                PSLayer *clone_prv = clone_providers[j];
                testAssertNotNull(orig_prv, test);
                testAssertNotNull(clone_prv, test);
                testAssertNotNull(orig_prv->model, test);
                testAssertNotNull(clone_prv->model, test);
                testAssertWithMessage(
                    orig_prv->model->index == clone_prv->model->index,
                    test, "Layer[%d]: Source provider[%d] model[%d] != "
                    "Clone provider model[%d]", i, j,
                    orig_prv->model->index, clone_prv->model->index
                );
                testAssertWithMessage(
                    orig_prv->index == clone_prv->index,
                    test, "Layer[%d]: Source provider[%d] index[%d] != "
                    "Clone provider index[%d]", i, j, orig_prv->index,
                    clone_prv->index
                );
                testAssertWithMessage(
                    orig_prv->size == clone_prv->size, test,
                    "Layer[%d]: Source provider[%d] size != Clone provider[%d]"
                    " size: %d != %d", i, j, j, orig_prv->size, clone_prv->size
                );
            }
        } else if (PositionalEncoding == orig_l->type) {
            int orig_enclen = PSGetPositionalEncodingLength(orig_l),
                clone_enclen = PSGetPositionalEncodingLength(clone_l);
            testAssertWithMessage(
                orig_enclen == clone_enclen, test, "Layer[%d]: Source "
                "positional encoding size != Clone ones: %d != %d",
                i, orig_enclen, clone_enclen
            );
            int orig_base = PSGetPositionalEncodingBase(orig_l),
                clone_base = PSGetPositionalEncodingBase(clone_l);
            testAssertWithMessage(
                orig_base == clone_base, test, "Layer[%d]: Source "
                "base != Clone one: %d != %d",
                i, orig_base, clone_base
            );
        } else if (Attention == orig_l->type) {
            PSAttentionType oattn_type = PSGetAttentionType(orig_l),
                            cattn_type = PSGetAttentionType(clone_l);
            testAssertWithMessage(
                oattn_type == cattn_type, test, "Layer[%d] attention type %d "
                "(%s) != Clone one %d (%s)", i, oattn_type,
                PSGetAttentionTypeLabel(oattn_type), cattn_type,
                PSGetAttentionTypeLabel(cattn_type)
            );
            int orig_nheads = PSGetAttentionHeadCount(orig_l),
                clone_nheads = PSGetAttentionHeadCount(clone_l);
            testAssertWithMessage(
                orig_nheads == clone_nheads, test, "Layer[%d] attention heads "
                "%d != Clone attention heads %d", i, orig_nheads, clone_nheads
            );
            PSFloat orig_scale = PSGetAttentionScale(orig_l),
                    clone_scale = PSGetAttentionScale(clone_l);
            ok = compareFloats(orig_scale, clone_scale, 0, 4);
            testAssertWithMessage(
                ok, test, "Layer[%d] attention scale %g != Clone scale %g",
                i, orig_scale, clone_scale
            );
            int orig_is_causal = PSIsCausalAttention(orig_l),
                clone_is_causal = PSIsCausalAttention(clone_l);
            testAssertWithMessage(
                orig_is_causal == clone_is_causal, test, "Layer[%d] causal "
                "attention is %d, but Clone is %d", i, orig_is_causal,
                clone_is_causal
            );
            PSLayer *qprov_o = NULL, *qprov_c = NULL,
                    *kprov_o = NULL, *kprov_c = NULL,
                    *vprov_o = NULL, *vprov_c = NULL;
            ok = PSGetAttentionProviders(orig_l, &qprov_o, &kprov_o, &vprov_o);
            testAssert(ok, test);
            ok = PSGetAttentionProviders(clone_l, &qprov_c, &kprov_c, &vprov_c);
            testAssert(ok, test);
            if (qprov_o != NULL) {
                testAssertWithMessage(
                    qprov_c != NULL, test, "Layer[%d] has query provider, but "
                    "cloned doesn't", i
                );
                testAssertWithMessage(
                    qprov_o->model->index == qprov_c->model->index, test,
                    "Layer[%d] query provider's model index is %d, but "
                    "clone one is %d", i, qprov_o->model->index,
                    qprov_c->model->index
                );
                testAssertWithMessage(
                    qprov_o->index == qprov_c->index, test, "Layer[%d] query "
                    "provider's index is %d, but clone one is %d",
                    i, qprov_o->index, qprov_c->index
                );
                testAssertWithMessage(
                    qprov_o->type == qprov_c->type, test, "Layer[%d] query "
                    "provider's type is %d, but clone one is %d",
                    i, qprov_o->type, qprov_c->type
                );
                testAssertWithMessage(
                    qprov_o->size == qprov_c->size, test, "Layer[%d] query "
                    "provider's size is %d, but clone one is %d",
                    i, qprov_o->size, qprov_c->size
                );
            } else testAssertWithMessage(
                qprov_c == NULL, test, "Layer[%d] has no query provider, but "
                "cloned layer does", i
            );
            if (kprov_o != NULL) {
                testAssertWithMessage(
                    kprov_c != NULL, test, "Layer[%d] has key provider, but "
                    "cloned doesn't", i
                );
                testAssertWithMessage(
                    kprov_o->model->index == kprov_c->model->index, test,
                    "Layer[%d] key provider's model index is %d, but "
                    "clone one is %d", i, kprov_o->model->index,
                    kprov_c->model->index
                );
                testAssertWithMessage(
                    kprov_o->index == kprov_c->index, test, "Layer[%d] key "
                    "provider's index is %d, but clone one is %d",
                    i, kprov_o->index, kprov_c->index
                );
                testAssertWithMessage(
                    kprov_o->type == kprov_c->type, test, "Layer[%d] key "
                    "provider's type is %d, but clone one is %d",
                    i, kprov_o->type, kprov_c->type
                );
                testAssertWithMessage(
                    kprov_o->size == kprov_c->size, test, "Layer[%d] key "
                    "provider's size is %d, but clone one is %d",
                    i, kprov_o->size, kprov_c->size
                );
            } else testAssertWithMessage(
                kprov_c == NULL, test, "Layer[%d] has no key provider, but "
                "cloned layer does", i
            );
            if (vprov_o != NULL) {
                testAssertWithMessage(
                    vprov_c != NULL, test, "Layer[%d] has value provider, but "
                    "cloned doesn't", i
                );
                testAssertWithMessage(
                    vprov_o->model->index == vprov_c->model->index, test,
                    "Layer[%d] value provider's model index is %d, but "
                    "clone one is %d", i, vprov_o->model->index,
                    vprov_c->model->index
                );
                testAssertWithMessage(
                    vprov_o->index == vprov_c->index, test, "Layer[%d] value "
                    "provider's index is %d, but clone one is %d",
                    i, vprov_o->index, vprov_c->index
                );
                testAssertWithMessage(
                    vprov_o->type == vprov_c->type, test, "Layer[%d] value "
                    "provider's type is %d, but clone one is %d",
                    i, vprov_o->type, vprov_c->type
                );
                testAssertWithMessage(
                    vprov_o->size == vprov_c->size, test, "Layer[%d] value "
                    "provider's size is %d, but clone one is %d",
                    i, vprov_o->size, vprov_c->size
                );
            } else testAssertWithMessage(
                vprov_c == NULL, test, "Layer[%d] has no value provider, but "
                "cloned layer does", i
            );
        }
        if (i == 0) continue;
        if (otype == Pooling) continue;
        testAssertWithMessage(
            (orig_l->weight_types == clone_l->weight_types), test,
            "Layer[%d]: Source weight_types %d != Clone %d",
            orig_l->weight_types, clone_l->weight_types
        );
        if (orig_l->biases != NULL) {
            testAssertWithMessage(
                clone_l->biases != NULL, test,
                "Layer[%d]: Source biases not null, but clone biases is null",i
            );
        }
        if (orig_l->biases == NULL) {
            testAssertWithMessage(
                clone_l->biases == NULL, test,
                "Layer[%d]: Source biases null, but clone biases not null",i
            );
        }
        if (orig_l->biases != NULL) {
            int bias_count = PSGetLayerParametersCount(orig_l, PS_PARAM_BIAS);
            for (k = 0; k < bias_count; k++) {
                PSFloat obias = getRoundedFloat(orig_l->biases[k]);
                PSFloat cbias = getRoundedFloat(clone_l->biases[k]);
                ok = (obias == cbias);
                testAssertWithMessage(
                    (obias == cbias), test, "Layer[%d]: bias[%d]  %g != %g",
                    orig_l->index, k, orig_l->biases[k], clone_l->biases[k]
                );
            }
        }
        if (orig_l->weights != NULL) {
            testAssertWithMessage(
                clone_l->weights != NULL, test,
                "Layer[%d]: Source weights not null, but clone weights null",i
            );
        }
        if (orig_l->weights == NULL) {
            testAssertWithMessage(
                clone_l->weights == NULL, test,
                "Layer[%d]: Source weights null, but clone weights not null",i
            );
        }
        if (orig_l->weights != NULL) {
            for (k = 0; k < orig_l->weight_types; k++) {
                PSMatrix o_weights = orig_l->weights[k];
                PSMatrix c_weights = clone_l->weights[k];
                if (Attention != orig_l->type) {
                    testAssertNotNull(o_weights, test);
                    testAssertNotNull(c_weights, test);
                } else {
                    int orig_wnull = (o_weights == NULL),
                        clone_wnull = (c_weights == NULL);
                    if (!orig_wnull) {
                        testAssertWithMessage(
                            !clone_wnull, test, "Layer[%d] has weights[%d], "
                            "but cloned layer[%d] is missing them", i, k, i
                        );
                    } else {
                        testAssertWithMessage(
                            clone_wnull, test, "Layer[%d] has no weights[%d], "
                            "but cloned layer[%d] has them", i, k, i
                        );
                    }
                }
                uint64_t o_wsize = PSMatrixLength(o_weights),
                         c_wsize = PSMatrixLength(c_weights);
                testAssertWithMessage(
                    o_wsize == c_wsize, test,
                    "Layer[%d]: source weights[%d] size %llu != %llu",
                    i, k, o_wsize, c_weights
                );
                for (uint64_t w = 0; w < o_wsize; w++) {
                    PSFloat ow = o_weights[w];
                    PSFloat cw = c_weights[w];
                    int equal_weights = compareFloats(ow, cw, 0, 5);
                    testAssertWithMessage(
                        equal_weights, test,
                        "Layer[%d]: source weights[%d] %g != %g", i, ow, cw
                    );
                }
            }
        }
        if (!ok) break;
    }
    return ok;
}

int compareModelChain(PSModel *model, PSModel *clone, Test* test) {
    int ok = 1;
    PSModel *n1 = model, *n2 = clone;
    while (n1 != NULL && n2 != NULL) {
        ok = compareModels(n1, n2, test);
        if (!ok) break;
        n1 = n1->next;
        n2 = n2->next;
    }
    testAssertNull(n1, test);
    testAssertNull(n2, test);
    return ok;
}

static int testRecurrentNetworkMode(PSModel *model,
                                    PSRecurrentNetworkMode mode,
                                    Test *test)
{
    testAssertNotNull(model, test);
    if (model->size == 0) return 1;
    PSRecurrentNetworkMode model_rnn_mode = model->rnn_mode;
    testAssertWithMessage(
        model_rnn_mode == mode, test,
        "Recurrent network mode %s != expected %s",
        PSGetRecurrentModeLabel(model_rnn_mode),
        PSGetRecurrentModeLabel(mode)
    );
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(model),
            *last_recurrent = PSGetLastRecurrentLayer(model),
            *input_layer = model->layers[0],
            *output_layer = model->layers[model->size - 1];
    if (mode == NonRecurrent) {
        testAssertWithMessage(
            !PSIsRecurrent(model), test,
            "Model is recurrent despite mode is %s",
            PSGetRecurrentModeLabel(mode)
        );
    } else {
        testAssertWithMessage(
            PSIsRecurrent(model), test,
            "Model is not recurrent despite mode is %s",
            PSGetRecurrentModeLabel(mode)
        );
        testAssertNotNull(first_recurrent, test);
        testAssertNotNull(last_recurrent, test);
        if (mode == ManyToMany) {
            testAssertWithMessage(
                first_recurrent == input_layer, test,
                "first_recurrent_layer is not the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent == output_layer, test,
                "last_recurrent_layer is not the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        } else if (mode == ManyToOne) {
            testAssertWithMessage(
                first_recurrent == input_layer, test,
                "first_recurrent_layer is not the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent != output_layer, test,
                "last_recurrent_layer is the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        } else if (mode == OneToMany) {
            testAssertWithMessage(
                first_recurrent != input_layer, test,
                "first_recurrent_layer is the input layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
            testAssertWithMessage(
                last_recurrent == output_layer, test,
                "last_recurrent_layer is not the output layer for mode %s",
                PSGetRecurrentModeLabel(mode)
            );
        }
    }
    for (int i = 0; i < model->size; i++) {
        PSLayer *layer = model->layers[i];
        if (mode == NonRecurrent) {
            testAssertWithMessage(
                !PSIsRecurrent(layer), test,
                "Layer[%d] is recurrent despite mode is NonRecurrent", i
            );
        } else if (mode == ManyToMany) {
            testAssertWithMessage(
                PSIsRecurrent(layer), test,
                "Layer[%d] is not recurrent despite mode is %s", i,
                PSGetRecurrentModeLabel(mode)
            );
        } else {
            if (layer->index >= first_recurrent->index &&
                layer->index <= last_recurrent->index)
            {
                testAssertWithMessage(
                    PSIsRecurrent(layer), test,
                    "Layer[%d] is not recurrent (mode %s): first recurrent "
                    "layer is %d, last recurrent layer is %d", i,
                    first_recurrent->index, last_recurrent->index,
                    PSGetRecurrentModeLabel(mode)
                );
            } else {
                testAssertWithMessage(
                    !PSIsRecurrent(layer), test,
                    "Layer[%d] is recurrent (mode %s): first recurrent "
                    "layer is %d, last recurrent layer is %d", i,
                    first_recurrent->index, last_recurrent->index,
                    PSGetRecurrentModeLabel(mode)
                );
            }
        }
    }
    return 1;
}

PSFloat test_dot(PSFloat *x, PSFloat *y, int size) {
    int i;
    PSFloat dot = 0.0;
    for (i = 0; i < size; i++) {
        dot += (x[i] * y[i]);
    }
    return dot;
}

static int compareFloats(PSFloat a, PSFloat b, int rounding, int precision) {
    if (a == b) return 1;
    if (precision > 0) {
        PSFloat diff = fabs(a - b);
        PSFloat maxdiff = PSPow(10, precision * -1);
        if (diff <= maxdiff) return 1;
    }
    if (rounding) {
        a = getRoundedFloatDec(a, rounding);
        b = getRoundedFloatDec(b, rounding);
    }
    return a == b;
}

static int compareArrays(PSFloat *arr, PSFloat *exp, int len, Test* test,
                         char *descr, int rounding, int precision)
{
    for (int i = 0; i < len; i++) {
        PSFloat value = arr[i];
        PSFloat expected = exp[i];
        if (precision > 0) {
            int ok = compareFloats(value, expected, 0, precision);
            testAssertWithMessage(
                ok, test, "%s: value[%d] != expected[%d] -> %.*g != %.*g",
                descr, i, i, PSFLOAT_DIG, value, PSFLOAT_DIG, expected
            );
            continue;
        }
        if (rounding > 0) {
            value = getRoundedFloatDec(value, rounding);
            expected = getRoundedFloatDec(expected, rounding);
        }
        testAssertWithMessage(
            (value == expected), test,
            "%s: value[%d] != expected[%d] -> %.*g != %.*g",
            descr, i, i, PSFLOAT_DIG, value, PSFLOAT_DIG, expected
        );
    }
    return 1;
}

static int checkMatrixTransposition(PSMatrix m, const PSFloat *tdata,
                                    int acceleration, char *descr, Test *test)
{
    PSMathOpts opts = {.acceleration = acceleration};
    PSMatrix tm = PSMatrixTranspose(m, 1, &opts);
    testAssertWithMessage(
        (tm != NULL), test, "%s: Transposed is NULL", descr
    );
    testAssertWithMessage(
        (tm != m), test, "%s: Transposed = Matrix", descr
    );
    int tmlen = PSMatrixLength(tm), mlen = PSMatrixLength(m), i;
    testAssertWithMessage(
        (tmlen == mlen), test,
        "%s: Transposed 2D len %d != %d", descr, tmlen, mlen
    );
    int ndims, tndims;
    int dims[3] = {0};
    int tdims[3] = {0};
    ndims = PSMatrixDimensions(m, dims);
    tndims = PSMatrixDimensions(tm, tdims);
    testAssertWithMessage(
        (tndims == ndims), test,
        "%s: Transposed dimension count %d != %d", descr, tndims, ndims
    );
    for (i = 0; i < ndims; i++) {
        int tidx = ndims - 1 - i;
        testAssertWithMessage(
            tdims[tidx] == dims[i], test,
            "%s: Transposed dim[%d] -> Matrix dim[%d]: %d != %d", descr,
            tidx, i, tdims[tidx], dims[i]
        );
    }
    for (i = 0; i < mlen; i++) {
        testAssertWithMessage(
            (tdata[i] == tm[i]), test,
            "%s: Transposed[%d] should be %g, got %g", descr, i,
            tdata[i], tm[i]
        );
    }
    return 1;
}

int testMathsDotProduct(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[16] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                     0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0};
    PSFloat y[16] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                     1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5};
    PSFloat cmp_res = test_dot(x, y, 16), res;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    res = PSDotProduct(x, y, 16, &opts);
    testAssertWithMessage(
        (res == cmp_res), test, "Accelerate Framework: Expected %g != %g",
        cmp_res, res
    );
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    res = PSDotProduct(x, y, 16, &opts);
    testAssertWithMessage(
        (res == cmp_res), test, "AVX: Expected %g != %g",
        cmp_res, res
    );
#endif
    opts.acceleration = PSAcceleration_None;
    res = PSDotProduct(x, y, 16, &opts);
    testAssertWithMessage(
        (res == cmp_res), test, "No Acceleration: Expected %g != %g",
        cmp_res, res
    );
    return 1;
}

int testMathsDot(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[32] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                     0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0,
                     -2.0, -0.2, 3.0, 0.0, 0.5, 1.0, 1.0, 0.1,
                     -0.05, 0.0, 2.5, 1.25, 0.0, -2.0, 3.2, -0.25};
    PSFloat y[16] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                     1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5};
    PSFloat res[2] = {0, 0};
    PSFloat cmp_res[2] = {0, 0};
    PSMatrix matrix = PSMatrixZeros(2, 2, 16);
    testAssertNotNull(matrix, test);
    memcpy(matrix, x, 32 * sizeof(PSFloat));
    PSMathOpts opts = {0};
    opts.argtype[1] = 'V';
    int r, c, ok = 1, failed = 0;
    ok = compareArrays(matrix, x, 32, test, "Matrix data", 0, 0);
    if (!ok) {
        PSMatrixFree(matrix);
        return 0;
    }
    for (r = 0; r < 2; r++) {
        PSFloat *row = x + (r * 16);
        PSFloat sum = 0.0;
        for (c = 0; c < 16; c++) sum += row[c] * y[c];
        cmp_res[r] = sum;
    }
    int decrnd = NORMAL_PRECISION_DEC;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    ok = PSDot(matrix, y, res, &opts);
    testAssert(ok, test);
    ok = compareArrays(res, cmp_res, 2, test, "Accelerate Framework",decrnd,0);
    if (!ok) {
        failed++;
        appendTestErrorMessage(test, "\n%*s", 4, "");
    }
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    ok = PSDot(matrix, y, res, &opts);
    testAssert(ok, test);
    ok = compareArrays(res, cmp_res, 2, test, "AVX", decrnd, 0);
    if (!ok) {
        failed++;
        appendTestErrorMessage(test, "\n%*s", 4, "");
    }
#endif
    opts.acceleration = PSAcceleration_BLAS;
    ok = PSDot(matrix, y, res, &opts);
    testAssert(ok, test);
    ok = compareArrays(res, cmp_res, 2, test, "BLAS", decrnd, 0);
    if (!ok) {
        failed++;
        appendTestErrorMessage(test, "\n%*s", 4, "");
    }
    opts.acceleration = PSAcceleration_None;
    ok = PSDot(matrix, y, res, &opts);
    testAssert(ok, test);
    ok = compareArrays(res, cmp_res, 2, test, "No Acceleration", 0, 0);
    if (!ok) {
        failed++;
        appendTestErrorMessage(test, "\n%*s", 4, "");
    }
final:
    if (matrix) PSMatrixFree(matrix);
    return (failed == 0);
}

int testMathsVecProd(TestCase *tc, Test *test) {
    UNUSED(tc);
    int ok = 1, i;
    PSFloat a[] = {2, 3};
    PSFloat b[] = {4, 5, 6};
    PSFloat res[6] = {0};
    PSFloat expected[] = {8, 10, 12, 12, 15, 18};
    PSMathOpts opts = {0};
#ifdef HAS_BLAS
    opts.acceleration = PSAcceleration_BLAS;
    ok = PSOuterProduct(a, b, res, 2, 3, &opts);
    testAssertWithMessage(ok, test, "PSOuterProduct (BLAS) failed%s", "");
    for (i = 0; i < 6; i++) {
        testAssertWithMessage(
            (res[i] == expected[i]), test,
            "PSOuterProduct (BLAS): res[%d] != expected[%d] -> %g != %g",
            i, i, res[i], expected[i]
        );
        res[i] = 0;
    }
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorClear(res, 6);
    ok = PSOuterProduct(a, b, res, 2, 3, &opts);
    testAssertWithMessage(
        ok, test, "PSOuterProduct (Accelerate Framework) failed%s", ""
    );
    for (i = 0; i < 6; i++) {
        testAssertWithMessage(
            (res[i] == expected[i]), test,
            "PSOuterProduct (Accelerate Framework): res[%d] != expected[%d] "
            "-> %g != %g",
            i, i, res[i], expected[i]
        );
        res[i] = 0;
    }
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorClear(res, 6);
    ok = PSOuterProduct(a, b, res, 2, 3, &opts);
    testAssertWithMessage(ok, test, "PSOuterProduct (no accel.) failed%s", "");
    for (i = 0; i < 6; i++) {
        testAssertWithMessage(
            (res[i] == expected[i]), test,
            "PSOuterProduct (no accel.): res[%d] != expected[%d] -> %g != %g",
            i, i, res[i], expected[i]
        );
    }
    return ok;
}

int testMathsSumV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 3.0, 0.2};
    PSFloat y[6] = {0.5, -0.35, 1.0, -0.1, 0.0, 4.0};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] + y[i];
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSAddVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSAddVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSAddVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}
int testMathsSubV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 3.0, 0.2};
    PSFloat y[6] = {0.5, -0.35, 1.0, -0.1, 0.0, 4.0};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] - y[i];
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSSubtractVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSubtractVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSubtractVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsMulV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 3.0, 0.2};
    PSFloat y[6] = {0.5, -0.35, 1.0, -0.1, 0.0, 4.0};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] * y[i];
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSMultiplyVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSMultiplyVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSMultiplyVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsDivV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 0.0, 18.5};
    PSFloat y[6] = {0.5, 2.0, 1.0, -0.5, 3.0, 4.0};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] / y[i];
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSDivideVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSDivideVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSDivideVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsSumVS(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 0.0, 18.5};
    PSFloat y = -1.5;
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] + y;
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSAddVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSAddVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSAddVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsSubSV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x = -1.5;
    PSFloat y[6] = {1.0, 8.3, 2.0, -1.5, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x - y[i];
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSSubtractScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSubtractScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSubtractScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsMulVS(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 0.0, 18.5};
    PSFloat y = -1.5;
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] * y;
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSMultiplyVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSMultiplyVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSMultiplyVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsDivVS(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, 2.0, -1.5, 0.0, 18.5};
    PSFloat y = 2.0;
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x[i] / y;
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSDivideVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSDivideVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSDivideVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsDivSV(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x = 8.0;
    PSFloat y[6] = {0.5, 2.0, 1.0, -0.5, 3.0, 4.0};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = x / y[i];
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSDivideScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSDivideScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSDivideScalarVector(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsClip(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat min = -1.0, max = 2.0;
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSClipValue(x[i], min, max);
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorClip(x, min, max, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorClip(x, min, max, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorClip(x, min, max, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsThres(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat min = 0.0;
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++)
        cmp_res[i] = PSClipValue(x[i], min, PSFLOAT_MAX);
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorThreshold(x, min, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorThreshold(x, min, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorThreshold(x, min, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsMapLimit(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat expected[6] = {1, 1, -1, -1, 1, 1};
    PSFloat limit = 0.0;
    PSFloat res[6] = {0};
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorMapWithLimit(x, limit, 1, res, 6,&opts);
    ok = compareArrays(res, expected, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorMapWithLimit(x, limit, 1, res, 6, &opts);
    ok = compareArrays(res, expected, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsExp(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSExp(x[i]);
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorExp(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorExp(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#else
    UNUSED(decrnd);
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorExp(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsPow(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    PSFloat exp = 3;
    for (int i = 0; i < 6; i++) cmp_res[i] = PSPow(x[i], exp);
    int ok = 1;
    int precision = 3;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorPower(x, exp, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0,
                       precision);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorPower(x, exp, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, precision);
    if (!ok) return 0;
#else
    UNUSED(precision);
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorPower(x, exp, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsTanh(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSTanh(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorTanh(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorTanh(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorTanh(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsSqrt(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {0.1, 8.3, 1.2, 1.0, 0.5, 28.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSSqrt(x[i]);
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorSqrt(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorSqrt(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorSqrt(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsNeg(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = -(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorNeg(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorNeg(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorNeg(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsAbs(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {1.0, 8.3, 2.0, 1.0, 0.0, 18.5};
    PSFloat res[6] = {0};
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorAbs(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSVectorAbs(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorAbs(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testMathsMean(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {34.8428, 76.1856, 21.2119, 137.8675, 40.4213, 67.1189};
    PSFloat expected = 62.9413, mean;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    mean = PSMean(x, 6, &opts);
    mean = getRoundedFloatDec(mean, 4);
    testAssertWithMessage(
        (mean == expected), test, "Mean (ACF) != Expected: %g != %g",
        mean, expected
    );
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    mean = PSMean(x, 6, &opts);
    mean = getRoundedFloatDec(mean, 4);
    testAssertWithMessage(
        (mean == expected), test, "Mean (AVX) != Expected: %g != %g",
        mean, expected
    );
#endif
    opts.acceleration = 0;
    mean = PSMean(x, 6, &opts);
    mean = getRoundedFloatDec(mean, 4);
    testAssertWithMessage(
        (mean == expected), test, "Mean (No accel.) != Expected: %g != %g",
        mean, expected
    );
    return 1;
}

int testMathsVar(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {34.8428, 76.1856, 21.2119, 137.8675, 40.4213, 67.1189};
    PSFloat expected = 1474.14, var;
    expected = getRoundedFloatDec(expected, 2);
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    var = PSVariance(x, 6, &opts);
    var = getRoundedFloatDec(var, 2);
    testAssertWithMessage(
        (var == expected), test, "Variance (ACF) != Expected: %g != %g",
        var, expected
    );
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    var = PSVariance(x, 6, &opts);
    var = getRoundedFloatDec(var, 2);
    testAssertWithMessage(
        (var == expected), test, "Variance (AVX) != Expected: %g != %g",
        var, expected
    );
#endif
    opts.acceleration = 0;
    var = PSVariance(x, 6, &opts);
    var = getRoundedFloatDec(var, 2);
    testAssertWithMessage(
        (var == expected), test, "Variance (No accel.) != Expected: %g != %g",
        var, expected
    );
    return 1;
}

int testMathsStd(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {34.8428, 76.1856, 21.2119, 137.8675, 40.4213, 67.1189};
    PSFloat expected = 38.3945, stddev;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    stddev = PSStdDev(x, 6, &opts);
    stddev = getRoundedFloatDec(stddev, 4);
    testAssertWithMessage(
        (stddev == expected), test, "StdDev (ACF) != Expected: %g != %g",
        stddev, expected
    );
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    stddev = PSStdDev(x, 6, &opts);
    stddev = getRoundedFloatDec(stddev, 4);
    testAssertWithMessage(
        (stddev == expected), test, "StdDev (AVX) != Expected: %g != %g",
        stddev, expected
    );
#endif
    opts.acceleration = 0;
    stddev = PSStdDev(x, 6, &opts);
    stddev = getRoundedFloatDec(stddev, 4);
    testAssertWithMessage(
        (stddev == expected), test, "StdDev (No accel.) != Expected: %g != %g",
        stddev, expected
    );
    return 1;
}

int testMathVectorFill(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1, i;
    PSFloat vec[10] = {0};
    PSFloat filler = 3.1234;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSVectorFill(vec, filler, 10, &opts);
    for (i = 0; i < 10; i++) {
        testAssertWithMessage(
            vec[i] == filler, test, "(ACF) vec[%d] != filler: %g != %g",
            i, vec[i], filler
        );
    }
    memset(vec, 0, 10 * sizeof(PSFloat));
#endif
    opts.acceleration = PSAcceleration_None;
    PSVectorFill(vec, filler, 10, &opts);
    for (i = 0; i < 10; i++) {
        testAssertWithMessage(
            vec[i] == filler, test, "(No accel.) vec[%d] != filler: %g != %g",
            i, vec[i], filler
        );
    }
    memset(vec, 0, 10 * sizeof(PSFloat));
    return res;
}

int testMathsMatrixCopy(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1;
    PSMatrix src = PSMatrixRandom(2, 2, 3);
    testAssertNotNull(src, test);
    PSMatrix dst = PSMatrixRandom(2, 2, 3);
    if (dst == NULL) {
        PSMatrixFree(src);
        testAssertNotNull(dst, test);
    }
    testAssertNotNull(dst, test);
    uint64_t len = PSMatrixLength(src), i;
    testAssertWithMessage(
        (len == 6), test,
        "Expected len is 6, got: %d", (int) len
    );
    res = PSMatrixCopy(src, dst);
    testAssertWithMessageOrGoto(
        res, fail, test, "Failed to copy matrix%s", ""
    );
    for (i = 0; i < len; i++) {
        PSFloat srci = src[i], dsti = dst[i];
        testAssertWithMessageOrGoto(
            (srci == dsti), fail, test, "Source[%d] != Dest[%d] -> %g != %g",
            (int) i, (int) i, srci, dsti
        );
    }
    goto final;
fail:
    res = 0;
final:
    if (src != NULL) PSMatrixFree(src);
    if (dst != NULL) PSMatrixFree(dst);
    return res;
}

int testMathsMatrixDup(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1;
    PSMatrix src = PSMatrixRandom(2, 2, 3);
    testAssertNotNull(src, test);
    PSMatrix dst = PSMatrixDup(src);
    if (dst == NULL) {
        PSMatrixLength(src);
        testAssertNotNull(dst, test);
    }
    uint64_t len = PSMatrixLength(src), i;
    testAssertWithMessageOrGoto(
        (len == 6), fail, test,
        "Expected len is 6, got: %d", (int) len
    );
    for (i = 0; i < len; i++) {
        PSFloat srci = src[i], dsti = dst[i];
        testAssertWithMessageOrGoto(
            (srci == dsti), fail, test, "Source[%d] != Dest[%d] -> %g != %g",
            (int) i, (int) i, srci, dsti
        );
    }
    goto final;
fail:
    res = 0;
final:
    if (src != NULL) PSMatrixFree(src);
    if (dst != NULL) PSMatrixFree(dst);
    return res;
}

int testMathsMatrixTranspose(TestCase *tc, Test *test) {
    UNUSED(tc);
    const PSFloat data2D[] = {1, 2, 3, 4, 5, 6};
    const PSFloat data2DT[] = {1, 4, 2, 5, 3, 6};
    const PSFloat data3D[] = {1,  2,  3,  4, 5,  6,  7,  8, 9, 10, 11, 12, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const PSFloat data3DT[] = {1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22,
                               3, 11, 19, 7, 15, 23, 4, 12, 20, 8, 16, 24};
    int len2d = (int) (sizeof(data2D) / sizeof(PSFloat));
    int len3d = (int) (sizeof(data3D) / sizeof(PSFloat));
    int res = 1, i;
    PSMatrix m2d = PSMatrixZeros(2, 2, 3);
    testAssertNotNull(m2d, test);
    PSMatrix m3d = PSMatrixZeros(3, 3, 2, 4);
    if (m3d == NULL) {
        PSMatrixFree(m3d);
        testAssertNotNull(m3d, test);
    }
    int m2dlen = (int) PSMatrixLength(m2d),
        m3dlen = (int) PSMatrixLength(m3d);
    testAssertWithMessageOrGoto(
        (m2dlen == len2d), fail, test,
        "2D Matrix length should be %d, got: %d", len2d, m2dlen
    );
    testAssertWithMessageOrGoto(
        (m3dlen == len3d), fail, test,
        "3D Matrix length should be %d, got: %d", len3d, m3dlen
    );
    memcpy(m2d, data2D, sizeof(data2D));
    memcpy(m3d, data3D, sizeof(data3D));
    for (i = 0; i < m2dlen; i++) testAssertWithMessageOrGoto(
        (m2d[i] == data2D[i]), fail, test,
        "2DMatrix[%d] != data2D[%d]: %g != %g", i, i, m2d[i], data2D[i]
    );
    for (i = 0; i < m3dlen; i++) testAssertWithMessageOrGoto(
        (m3d[i] == data3D[i]), fail, test,
        "3DMatrix[%d] != data3D[%d]: %g != %g", i, i, m3d[i], data3D[i]
    );
    int acceleration = PSAcceleration_None;
    char descr[55] = {0};
    snprintf(descr, 54, "Matrix 2D (No Acceleration)");
    res = checkMatrixTransposition(m2d, data2DT, acceleration, descr, test);
    if (!res) goto final;
    snprintf(descr, 54, "Matrix 3D (No Acceleration)");
    res = checkMatrixTransposition(m3d, data3DT, acceleration, descr, test);
    if (!res) goto final;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    snprintf(descr, 54, "Matrix 2D (%s)", PSGetAccelerationName(acceleration));
    res = checkMatrixTransposition(m2d, data2DT, acceleration, descr, test);
    if (!res) goto final;
#endif
    goto final;
fail:
    res = 0;
final:
    PSMatrixFree(m2d);
    PSMatrixFree(m3d);
    return res;
}

int testMathsMatrixSwap(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat data[] = {
        0.464504,0.913899,0.907141,0.325932,0.944054,0.721402,0.598695,
        0.273243,0.398003,0.242346,0.107517,0.0299835,0.932873,0.794529,
        0.651348,0.209432,0.926456,0.944398
    };
    PSFloat exp1data[] = {
        0.464504,0.325932,0.913899,0.944054,0.907141,0.721402,0.598695,
        0.242346,0.273243,0.107517,0.398003,0.0299835,0.932873,0.209432,
        0.794529,0.926456,0.651348,0.944398
    };
    PSFloat exp2data[] = {
        0.464504,0.913899,0.907141,0.598695,0.273243,0.398003,0.932873,
        0.794529,0.651348,0.325932,0.944054,0.721402,0.242346,0.107517,
        0.0299835,0.209432,0.926456,0.944398
    };
    int ok = 1;
    PSMatrix matrix = NULL, swap1 = NULL, swap2 = NULL;
    matrix = PSMatrixFromArray(data, 3, 3, 2, 3);
    testAssertNotNull(matrix, test);
    swap1 = PSMatrixSwapAxes(matrix, -1, -2);
    ok = swap1 != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "failed to swap axes %d, %d", -1, -2
    );
    ok = compareArrays(
        swap1, exp1data, PSMatrixLength(swap1), test, "Swap -1, -2", 0, 4
    );
    testAssertWithMessageOrGoto(
        ok, final, test, "swap %d, %d comparison failed", -1, -2
    );
    swap2 = PSMatrixSwapAxes(matrix, 0, 1);
    ok = swap2 != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "failed to swap axes %d, %d", 0, 1
    );
    ok = compareArrays(
        swap2, exp2data, PSMatrixLength(swap2), test, "Swap 0, 1", 0, 4
    );
    testAssertWithMessageOrGoto(
        ok, final, test, "swap %d, %d comparison failed", 0, 1
    );
final:
    PSMatrixFree(matrix);
    PSMatrixFree(swap1);
    PSMatrixFree(swap2);
    return ok;
}

int testMathsMatrixExpand(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1;
    PSMatrix src = PSMatrixRandom(2, 2, 4), expanded = NULL;
    PSFloat *srcvalues = NULL;
    testAssertNotNull(src, test);
    uint64_t curlen = PSMatrixLength(src);
    srcvalues = malloc(curlen * sizeof(PSFloat));
    testAssertWithMessageOrGoto(srcvalues != NULL, fail, test, "%s", "");
    memcpy(srcvalues, src, curlen * sizeof(PSFloat));
    testAssertWithMessageOrGoto(
        curlen == (2 * 4), fail, test,
        "Current length is %lld, expected: %d",
        curlen, (2 * 4)
    );
    expanded = PSMatrixExpand(src, 1, 0);
    testAssertWithMessageOrGoto(
        expanded != NULL, fail, test, "PSMatrixExpand returned NULL%s",""
    );
    src = NULL;
    uint64_t newlen = PSMatrixLength(expanded), expectedlen = (3 * 4), i;
    testAssertWithMessageOrGoto(
        (newlen == expectedlen), fail, test,
        "Expected len is %lld, got: %lld", (int) expectedlen, (int) newlen
    );
    for (i = 0; i < curlen; i++) {
        PSFloat srci = srcvalues[i], dsti = expanded[i];
        testAssertWithMessageOrGoto(
            (srci == dsti), fail, test,
            "Source[%d] != Expanded[%d] -> %g != %g",
            (int) i, (int) i, srci, dsti
        );
    }
    for (; i < newlen; i++) {
        PSFloat dsti = expanded[i];
        testAssertWithMessageOrGoto(
            (dsti == 0.0), fail, test, "Expanded[%d] != 0.0 -> %g != 0.0",
            (int) i, dsti
        );
    }
    goto final;
fail:
    res = 0;
final:
    if (src != NULL) PSMatrixFree(src);
    if (expanded != NULL) PSMatrixFree(expanded);
    free(srcvalues);
    return res;
}

int testMatrixProduct(Test *test, int acceleration) {
    int res = 1;
    PSMatrix a = NULL, b = NULL, c = NULL, avec = NULL, result = NULL;
    PSFloat avalues[] = {1, 2, 3, 4, 5, 6};
    PSFloat bvalues[] = {1, 2, 3, 4, 5, 6};
    PSFloat cvalues[] = {1, 2, 3, 4, 5, 6, 3, 2, 1, 6, 5, 4};
    PSFloat avec_values[] = {7, 8, 9};
    PSFloat ab_expected[] = {22, 28, 49, 64}; /* a * b */
    PSFloat act_expected[] = {14, 32, 10, 28, 32, 77, 28, 73}; // a * c(t)
    PSFloat btb_expected[] = {35, 44, 44, 56};/* b(t) * b */
    PSFloat atbt_expected[] = {9, 19, 29, 12, 26,
                              40, 15, 33, 51}; /* a(t) @ b(t) */
    PSFloat avat_expected[] = {50, 122};
    int ab_l = sizeof(ab_expected) / sizeof(PSFloat);
    int btb_l = sizeof(btb_expected) / sizeof(PSFloat);
    int act_l = sizeof(act_expected) / sizeof(PSFloat);
    int atbt_l = sizeof(atbt_expected) / sizeof(PSFloat);
    int avat_l = sizeof(avat_expected) / sizeof(PSFloat);
    int rlen = 0;
    a = PSMatrixFromArray(avalues, 2, 2, 3);
    b = PSMatrixFromArray(bvalues, 2, 3, 2);
    c = PSMatrixFromArray(cvalues, 2, 4, 3);
    avec = PSMatrixFromArray(avec_values, 1, 3);
    testAssertWithMessageOrGoto(
        a != NULL, final, test, "matrix `%s` is NULL%s", "a"
    );
    testAssertWithMessageOrGoto(
        b != NULL, final, test, "matrix `%s` is NULL%s", "b"
    );
    testAssertWithMessageOrGoto(
        c != NULL, final, test, "matrix `%s` is NULL%s", "c"
    );
    testAssertWithMessageOrGoto(
        avec != NULL, final, test, "matrix `%s` is NULL%s", "avec"
    );
    const char *acceleration_name = PSGetAccelerationName(acceleration);
    if (acceleration_name == NULL) acceleration_name = "";
    PSMathOpts opts = {.acceleration = acceleration};
    opts.transpose = 0;
    res = PSMatrixProduct(a, b, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test,
        "Failed PSMatrixProduct(%s,%s) (transp = 0, accel = '%s')", "a","b",
        acceleration_name
    );
    testAssertWithMessageOrGoto(
        result != NULL, final, test, "No result for PSMatrixProduct(%s,%s) "
        "(transp = 0, accel = '%s')", "a","b", acceleration_name
    );
    rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == ab_l, final, test,
        "Length of result for PSMatrixProduct(%s,%s,accel = '%s') is %d, "
        "should be %d", "a", "b", acceleration_name, rlen, ab_l
    );
    char comparison_label[255] = {0};
    snprintf(comparison_label, 254, "a * b (accel = '%s')", acceleration_name);
    res = compareArrays(result, ab_expected, ab_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProduct(%s,%s,"
        "accel='%s')", "a", "b", acceleration_name
    );

    PSMatrixFree(result);
    result = NULL;
    opts.transpose = 1;
    res = PSMatrixProduct(b, b, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProduct(%s,%s) (transp = 1, "
        "accel = '%s')", "b","b", acceleration_name
    );
    testAssertWithMessageOrGoto(
        result != NULL, final, test, "No result for PSMatrixProduct(%s,%s) "
        "(transp = 1, accel = '%s')", "b","b", acceleration_name
    );
    rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == btb_l, final, test,
        "Length of result for PSMatrixProduct(%s,%s,accel='%s') is %d, "
        "should be %d", "a", "b", acceleration_name, rlen, btb_l
    );
    snprintf(comparison_label, 254, "b(T) * b (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, btb_expected, btb_l, test,comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProduct(%s,%s,"
        "accel='%s')", "b(T)", "b", acceleration_name
    );

    PSMatrixFree(result);
    result = NULL;
    opts.transpose = 2;
    res = PSMatrixProduct(a, c, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProduct(%s,%s) "
        "(transp = 2, accel = '%s')", "a","c", acceleration_name
    );
    testAssertWithMessageOrGoto(
        result != NULL, final, test, "No result for PSMatrixProduct(%s,%s) "
        "(transp = 2, accel = '%s')", "a","c", acceleration_name
    );
    rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == act_l, final, test,
        "Length of result for PSMatrixProduct(%s,%s,accel='%s') is %d, "
        "should be %d", "a", "c", acceleration_name, rlen, act_l
    );
    snprintf(comparison_label, 254, "a * c(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, act_expected, act_l, test,comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProduct(%s,%s,"
        "accel='%s')", "a", "c(T)", acceleration_name
    );

    PSMatrixFree(result);
    result = NULL;
    opts.transpose = 1 | 2;
    res = PSMatrixProduct(a, b, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProduct(%s,%s) (transp = 1|2, "
        "accel = '%s')", "a","b", acceleration_name
    );
    testAssertWithMessageOrGoto(
        result != NULL, final, test, "No result for PSMatrixProduct(%s,%s) "
        "(transp = 1|2, accel = '%s')", "a","b", acceleration_name
    );
    rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == atbt_l, final, test,
        "Length of result for PSMatrixProduct(%s,%s,accel='%s') is %d, "
        "should be %d", "a", "b", acceleration_name, rlen, atbt_l
    );
    snprintf(comparison_label, 254, "a(T) * b(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, atbt_expected, atbt_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProduct(%s,%s,"
        "accel='%s')", "a(T)", "b(T)", acceleration_name
    );

    PSMatrixFree(result);
    result = NULL;
    opts.transpose = 2;
    res = PSMatrixProduct(avec, a, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProduct(%s,%s) (transp = 2, "
        "accel = '%s')", "avec","a(T)", acceleration_name
    );
    testAssertWithMessageOrGoto(
        result != NULL, final, test, "No result for PSMatrixProduct(%s,%s) "
        "(transp = 2, accel = '%s')", "avec","a(T)", acceleration_name
    );
    rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == avat_l, final, test,
        "Length of result for PSMatrixProduct(%s,%s,accel='%s') is %d, "
        "should be %d", "avec", "a(T)", acceleration_name, rlen, avat_l
    );
    snprintf(comparison_label, 254, "avec * a(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, avat_expected, avat_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProduct(%s,%s,"
        "accel='%s')", "avec", "a(T)", acceleration_name
    );
final:
    PSMatrixFree(a);
    PSMatrixFree(b);
    PSMatrixFree(c);
    PSMatrixFree(avec);
    PSMatrixFree(result);
    return res;
}

int testMatrixProductMV(Test *test, int acceleration) {
    int res = 1;
    PSFloat avalues[] = {1, 2, 3, 4, 5, 6};
    PSFloat bvalues1[] = {7, 8, 9};
    PSFloat bvalues2[] = {7, 8};
    PSFloat ab1_expected[] = {50, 122};
    PSFloat ab2_expected[] = {39, 54, 69};
    int ab1_l = sizeof(ab1_expected) / sizeof(PSFloat);
    int ab2_l = sizeof(ab2_expected) / sizeof(PSFloat);
    int b1len = sizeof(bvalues1) / sizeof(PSFloat);
    int b2len = sizeof(bvalues2) / sizeof(PSFloat);
    PSMatrix a = NULL;
    PSFloat results[50] = {0};
    PSFloat *res_p = results;
    a = PSMatrixFromArray(avalues, 2, 2, 3);
    testAssertWithMessageOrGoto(
        a != NULL, final, test, "matrix `%s` is NULL%s", "a"
    );
    const char *acceleration_name = PSGetAccelerationName(acceleration);
    if (acceleration_name == NULL) acceleration_name = "";
    PSMathOpts opts = {.acceleration = acceleration};

    opts.transpose = 0;
    res = PSMatrixProductMV(a, bvalues1, b1len, &res_p, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProductMV(%s,%s) (transp = 0, "
        "accel = '%s')", "a","b1", acceleration_name
    );
    char comparison_label[255] = {0};
    snprintf(comparison_label, 254, "a * b1 (accel = '%s')", acceleration_name);
    res = compareArrays(res_p, ab1_expected, ab1_l, test, comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProductMV(%s,%s,"
        "accel='%s')", "a", "b1", acceleration_name
    );

    memset(res_p, 0, 50 * sizeof(PSFloat));
    opts.transpose = 1;
    res = PSMatrixProductMV(a, bvalues2, b2len, &res_p, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProductMV(%s,%s) (transp = 1, "
        "accel = '%s')", "a","b2"
    );
    snprintf(comparison_label, 254, "a(T) * b2 (accel = '%s')",
             acceleration_name);
    res = compareArrays(res_p, ab2_expected, ab2_l, test, "a(T) * b2", 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProductMV(%s,%s,"
        "accel='%s')", "a(T)", "b2", acceleration_name
    );
final:
    PSMatrixFree(a);
    return res;
}

int testMatrixProductVM(Test *test, int acceleration) {
    int res = 1;
    PSFloat avalues[] = {7, 8, 9};
    PSFloat bvalues[] = {1, 2, 3, 4, 5, 6};
    PSFloat ab1_expected[] = {50, 122};
    int ab1_l = sizeof(ab1_expected) / sizeof(PSFloat);
    int alen = sizeof(avalues) / sizeof(PSFloat);
    PSMatrix b = NULL, result = NULL;
    b = PSMatrixFromArray(bvalues, 2, 2, 3);
    testAssertWithMessageOrGoto(
        b != NULL, final, test, "matrix `%s` is NULL%s", "b"
    );
    const char *acceleration_name = PSGetAccelerationName(acceleration);
    if (acceleration_name == NULL) acceleration_name = "";
    PSMathOpts opts = {.acceleration = acceleration};

    opts.transpose = 2;
    res = PSMatrixProductVM(avalues, b, alen, &result, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatrixProductVM(%s,%s) (transp = 2, "
        "accel = '%s')", "a","b", acceleration_name
    );
    int rlen = PSMatrixLength(result);
    testAssertWithMessageOrGoto(
        rlen == ab1_l, final, test,
        "Length of result for PSMatrixProductVM(%s,%s,accel='%s') is %d, "
        "should be %d", "a", "b(T)", acceleration_name, rlen, ab1_l
    );
    char comparison_label[255] = {0};
    snprintf(comparison_label, 254, "a * b(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, ab1_expected, ab1_l, test,comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatrixProductVM(%s,%s,"
        "accel='%s')", "a", "b(T)", acceleration_name
    );
final:
    PSMatrixFree(b);
    PSMatrixFree(result);
    return res;
}

int testMathsMatrixProduct(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1, numtests = 0, acceleration;
#ifdef HAS_BLAS
    acceleration = PSAcceleration_BLAS;
    res = testMatrixProduct(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    res = testMatrixProduct(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
    acceleration = PSAcceleration_None;
    res = testMatrixProduct(test, acceleration);
    if (!res) return 0;
    numtests++;
    testAssertWithMessage(numtests > 0, test,
                          "BLAS disabled and no acceleration method suitable "
                          "for PSMatrixProduct%s", "");
    return res;
}

int testMathsMatrixProductMV(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1, numtests = 0, acceleration;
#ifdef HAS_BLAS
    acceleration = PSAcceleration_BLAS;
    res = testMatrixProductMV(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    res = testMatrixProductMV(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
    acceleration = PSAcceleration_None;
    res = testMatrixProductMV(test, acceleration);
    if (!res) return 0;
    numtests++;
    return res;
}

int testMathsMatrixProductVM(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1, numtests = 0, acceleration;
#ifdef HAS_BLAS
    acceleration = PSAcceleration_BLAS;
    res = testMatrixProductVM(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    res = testMatrixProductVM(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
    acceleration = PSAcceleration_None;
    res = testMatrixProductVM(test, acceleration);
    if (!res) return 0;
    numtests++;
    return res;
}

int testMatrixOp(PSMatrix a, PSMatrix b, testMatrixOpFunc func,
                 int expected_nd, int *expected_shape, PSFloat *expected,
                 char *funcname, Test *test)
{
    int ok = 1, nd, i;
    PSMathOpts opts = {0};
    PSMatrix res = NULL;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    char testdescr[255] = {0};
#ifdef HAS_BLAS
    snprintf(testdescr, 255, "[BLAS] %s", funcname);
    opts.acceleration = PSAcceleration_BLAS;
    ok = func(a, b, &res, &opts);
    testAssertWithMessageOrGoto(
        (ok && res != NULL), final, test, "%s: operation failed", testdescr
    );
    nd = PSMatrixDimensions(res, shape);
    testAssertWithMessageOrGoto(
        (ok = expected_nd == nd), final, test, "%s: expected shape size "
        "was %d, got %d", testdescr, expected_nd, nd
    );
    for (i = 0; i < nd; i++) {
        testAssertWithMessageOrGoto(
            (ok = shape[i] == expected_shape[i]), final, test, "%s: invalid "
            "result shape[%d] %d != %d", testdescr, shape[i], expected_shape[i]
        );
    }
    ok = compareArrays(res, expected, PSMatrixLength(res), test,
                       testdescr, 0, 5);
    if (!ok) goto final;
    PSMatrixFree(res);
    res = NULL;
    memset(shape, 0, sizeof(shape));
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    snprintf(testdescr, 255, "[ACF] %s", funcname);
    opts.acceleration = PSAcceleration_ACF;
    ok = func(a, b, &res, &opts);
    testAssertWithMessageOrGoto(
        (ok && res != NULL), final, test, "%s: operation failed", testdescr
    );
    nd = PSMatrixDimensions(res, shape);
    testAssertWithMessageOrGoto(
        (ok = expected_nd == nd), final, test, "%s: expected shape size "
        "was %d, got %d", testdescr, expected_nd, nd
    );
    for (i = 0; i < nd; i++) {
        testAssertWithMessageOrGoto(
            (ok = shape[i] == expected_shape[i]), final, test, "%s: invalid "
            "result shape[%d] %d != %d", testdescr, shape[i], expected_shape[i]
        );
    }
    ok = compareArrays(res, expected, PSMatrixLength(res), test,
                       testdescr, 0, 5);
    if (!ok) goto final;
    PSMatrixFree(res);
    res = NULL;
    memset(shape, 0, sizeof(shape));
#endif
#ifdef USE_AVX
    snprintf(testdescr, 255, "[AVX] %s", funcname);
    opts.acceleration = PSAcceleration_AVX;
    ok = func(a, b, &res, &opts);
    testAssertWithMessageOrGoto(
        (ok && res != NULL), final, test, "%s: operation failed", testdescr
    );
    nd = PSMatrixDimensions(res, shape);
    testAssertWithMessageOrGoto(
        (ok = expected_nd == nd), final, test, "%s: expected shape size "
        "was %d, got %d", testdescr, expected_nd, nd
    );
    for (i = 0; i < nd; i++) {
        testAssertWithMessageOrGoto(
            (ok = shape[i] == expected_shape[i]), final, test, "%s: invalid "
            "result shape[%d] %d != %d", testdescr, shape[i], expected_shape[i]
        );
    }
    ok = compareArrays(res, expected, PSMatrixLength(res), test,
                       testdescr, 0, 5);
    if (!ok) goto final;
    PSMatrixFree(res);
    res = NULL;
    memset(shape, 0, sizeof(shape));
#endif
    snprintf(testdescr, 255, "[NO ACCEL] %s", funcname);
    opts.acceleration = PSAcceleration_None;
    ok = func(a, b, &res, &opts);
    testAssertWithMessageOrGoto(
        (ok && res != NULL), final, test, "%s: operation failed", testdescr
    );
    nd = PSMatrixDimensions(res, shape);
    testAssertWithMessageOrGoto(
        (ok = expected_nd == nd), final, test, "%s: expected shape size "
        "was %d, got %d", testdescr, expected_nd, nd
    );
    for (i = 0; i < nd; i++) {
        testAssertWithMessageOrGoto(
            (ok = shape[i] == expected_shape[i]), final, test, "%s: invalid "
            "result shape[%d] %d != %d", testdescr, shape[i], expected_shape[i]
        );
    }
    ok = compareArrays(res, expected, PSMatrixLength(res), test,
                       testdescr, 0, 5);
    if (!ok) goto final;
    PSMatrixFree(res);
    res = NULL;
    memset(shape, 0, sizeof(shape));
final:
    PSMatrixFree(res);
    return ok;
}

int testMathsMatrixAdd(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat am_values[] = {
        -2.1282408 ,  1.09726265,  2.1837216 ,
        1.13824648, -0.39490999, -0.45876261
    };
    PSFloat bm_values[] = {
        0.57556455,  0.3889593 , -1.17216897,
        0.6723251 ,  0.34371944, -0.02343899
    };
    PSFloat av_values[] = {-1.56267393, -1.00281759,  0.09569721};
    PSFloat bv_values[] = {0.27088287,  0.93006405, -0.35962201};
    PSFloat scalar_value = 2.0;
    PSFloat exp_am_bm[] = {
        -1.55267625,  1.48622195,  1.01155262,
        1.81057158, -0.05119055, -0.4822016
    };
    PSFloat exp_av_bv[] = {-1.29179107, -0.07275354, -0.2639248};
    PSFloat exp_am_bv[] = {
        -1.85735794,  2.0273267 ,  1.82409958,
        1.40912935,  0.53515406, -0.81838462
    };
    PSFloat exp_am_scalar[2 * 3];
    PSVectorCopy(exp_am_scalar, am_values, (2 * 3));
    int ok = 1, nd = 0, i;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrix am = NULL, bm = NULL, av = NULL, bv = NULL, bv_t = NULL,
             scalar = NULL;
    for (i = 0; i < (int)(sizeof(exp_am_scalar)/sizeof(PSFloat)); i++)
        exp_am_scalar[i] += scalar_value;
    am = PSMatrixCreate(0, NULL, 2, 2, 3);
    bm = PSMatrixCreate(0, NULL, 2, 2, 3);
    testAssertWithMessageOrGoto(
        (ok = am != NULL), final, test,
        "could not create matrix %s with shape 2,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bm != NULL), final, test,
        "could not create matrix %s with shape 2,3", "B"
    );
    PSVectorCopy(am, am_values, PSMatrixLength(am));
    PSVectorCopy(bm, bm_values, PSMatrixLength(bm));
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bm, PSMatrixAdd, nd, shape, exp_am_bm,
                      "PSMatrixAdd: shape(2,3) + shape(2,3)", test);
    if (!ok) goto final;

    av = PSMatrixCreate(0, NULL, 2, 1, 3);
    bv = PSMatrixCreate(0, NULL, 2, 1, 3);
    testAssertWithMessageOrGoto(
        (ok = av != NULL), final, test,
        "could not create matrix %s with shape 1,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bv != NULL), final, test,
        "could not create matrix %s with shape 1,3", "B"
    );
    PSVectorCopy(av, av_values, PSMatrixLength(av));
    PSVectorCopy(bv, bv_values, PSMatrixLength(bv));
    nd = PSMatrixDimensions(av, shape);
    ok = testMatrixOp(av, bv, PSMatrixAdd, nd, shape, exp_av_bv,
                      "PSMatrixAdd: shape(1,3) + shape(1,3)", test);
    if (!ok) goto final;

    bv_t = PSMatrixReshape(bv, 2, 3, 1);
    testAssertWithMessageOrGoto(
        (ok = bv_t != NULL), final, test, "could not reshape matrix %s","B"
    );
    ok = testMatrixOp(av, bv_t, PSMatrixAdd, nd, shape, exp_av_bv,
                      "PSMatrixAdd: shape(1,3) + shape(3,1)", test);
    if (!ok) goto final;

    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bv, PSMatrixAdd, nd, shape, exp_am_bv,
                      "PSMatrixAdd: shape(2,3) + shape(1,3)", test);
    if (!ok) goto final;

    scalar = PSMatrixCreate(scalar_value, NULL, 1, 1);
    testAssertWithMessageOrGoto(
        (ok = scalar != NULL), final, test,
        "could not create %s matrix with shape 1", "scalar"
    );
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, scalar, PSMatrixAdd, nd, shape, exp_am_scalar,
                      "PSMatrixAdd: shape(2,3) + shape(1)", test);
    if (!ok) goto final;
    ok = testMatrixOp(scalar, am, PSMatrixAdd, nd, shape, exp_am_scalar,
                      "PSMatrixAdd: shape(1) + shape(2,3)", test);
    if (!ok) goto final;
final:
    PSMatrixFree(am);
    PSMatrixFree(bm);
    PSMatrixFree(av);
    PSMatrixFree(bv);
    PSMatrixFree(bv_t);
    PSMatrixFree(scalar);
    return ok;
}

int testMathsMatrixMultiply(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat am_values[] = {
        -2.1282408 ,  1.09726265,  2.1837216 ,
        1.13824648, -0.39490999, -0.45876261
    };
    PSFloat bm_values[] = {
        0.57556455,  0.3889593 , -1.17216897,
        0.6723251 ,  0.34371944, -0.02343899
    };
    PSFloat av_values[] = {-1.56267393, -1.00281759,  0.09569721};
    PSFloat bv_values[] = {0.27088287,  0.93006405, -0.35962201};
    PSFloat scalar_value = 2.0;
    PSFloat exp_am_bm[] = {
        -1.22493996,  0.42679051, -2.5596907 ,
        0.76527168, -0.13573824, 0.01075293
    };
    PSFloat exp_av_bv[] = {-0.42330159, -0.93268459, -0.03441482};
    PSFloat exp_am_bv[] = {
        -0.57650397,  1.02052454, -0.78531435,
        0.30833147, -0.36729158, 0.16498113
    };
    PSFloat exp_am_scalar[2 * 3];
    PSVectorCopy(exp_am_scalar, am_values, (2 * 3));
    int ok = 1, nd = 0, i;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrix am = NULL, bm = NULL, av = NULL, bv = NULL, bv_t = NULL,
             scalar = NULL;
    for (i = 0; i < (int)(sizeof(exp_am_scalar)/sizeof(PSFloat)); i++)
        exp_am_scalar[i] *= scalar_value;
    am = PSMatrixCreate(0, NULL, 2, 2, 3);
    bm = PSMatrixCreate(0, NULL, 2, 2, 3);
    testAssertWithMessageOrGoto(
        (ok = am != NULL), final, test,
        "could not create matrix %s with shape 2,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bm != NULL), final, test,
        "could not create matrix %s with shape 2,3", "B"
    );
    PSVectorCopy(am, am_values, PSMatrixLength(am));
    PSVectorCopy(bm, bm_values, PSMatrixLength(bm));
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bm, PSMatrixMultiply, nd, shape, exp_am_bm,
                      "PSMatrixMultiply: shape(2,3) + shape(2,3)", test);
    if (!ok) goto final;

    av = PSMatrixCreate(0, NULL, 2, 1, 3);
    bv = PSMatrixCreate(0, NULL, 2, 1, 3);
    testAssertWithMessageOrGoto(
        (ok = av != NULL), final, test,
        "could not create matrix %s with shape 1,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bv != NULL), final, test,
        "could not create matrix %s with shape 1,3", "B"
    );
    PSVectorCopy(av, av_values, PSMatrixLength(av));
    PSVectorCopy(bv, bv_values, PSMatrixLength(bv));
    nd = PSMatrixDimensions(av, shape);
    ok = testMatrixOp(av, bv, PSMatrixMultiply, nd, shape, exp_av_bv,
                      "PSMatrixMultiply: shape(1,3) + shape(1,3)", test);
    if (!ok) goto final;

    bv_t = PSMatrixReshape(bv, 2, 3, 1);
    testAssertWithMessageOrGoto(
        (ok = bv_t != NULL), final, test, "could not reshape matrix %s","B"
    );
    ok = testMatrixOp(av, bv_t, PSMatrixMultiply, nd, shape, exp_av_bv,
                      "PSMatrixMultiply: shape(1,3) + shape(3,1)", test);
    if (!ok) goto final;

    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bv, PSMatrixMultiply, nd, shape, exp_am_bv,
                      "PSMatrixMultiply: shape(2,3) + shape(1,3)", test);
    if (!ok) goto final;

    scalar = PSMatrixCreate(scalar_value, NULL, 1, 1);
    testAssertWithMessageOrGoto(
        (ok = scalar != NULL), final, test,
        "could not create %s matrix with shape 1", "scalar"
    );
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, scalar, PSMatrixMultiply, nd, shape, exp_am_scalar,
                      "PSMatrixMultiply: shape(2,3) + shape(1)", test);
    if (!ok) goto final;
    ok = testMatrixOp(scalar, am, PSMatrixMultiply, nd, shape, exp_am_scalar,
                      "PSMatrixMultiply: shape(1) + shape(2,3)", test);
    if (!ok) goto final;
final:
    PSMatrixFree(am);
    PSMatrixFree(bm);
    PSMatrixFree(av);
    PSMatrixFree(bv);
    PSMatrixFree(bv_t);
    PSMatrixFree(scalar);
    return ok;
}

int testMathsMatrixSubtract(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat am_values[] = {
        -2.1282408 ,  1.09726265,  2.1837216 ,
        1.13824648, -0.39490999, -0.45876261
    };
    PSFloat bm_values[] = {
        0.57556455,  0.3889593 , -1.17216897,
        0.6723251 ,  0.34371944, -0.02343899
    };
    PSFloat av_values[] = {-1.56267393, -1.00281759,  0.09569721};
    PSFloat bv_values[] = {0.27088287,  0.93006405, -0.35962201};
    PSFloat scalar_value = 2.0;
    PSFloat exp_am_bm[] = {
        -2.70380535,  0.70830334,  3.35589057,
        0.46592139, -0.73862943, -0.43532362
    };
    PSFloat exp_av_bv[] = {-1.8335568 , -1.93288164,  0.45531922};
    PSFloat exp_am_bv[] = {
        -2.39912367,  0.1671986 ,  2.54334361,
        0.86736362, -1.32497403, -0.0991406
    };
    PSFloat exp_am_scalar[2 * 3];
    PSFloat exp_scalar_am[2 * 3];
    PSVectorCopy(exp_am_scalar, am_values, (2 * 3));
    PSVectorCopy(exp_scalar_am, am_values, (2 * 3));
    int ok = 1, nd = 0, i;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrix am = NULL, bm = NULL, av = NULL, bv = NULL, bv_t = NULL,
             scalar = NULL;
    for (i = 0; i < (int)(sizeof(exp_am_scalar)/sizeof(PSFloat)); i++) {
        exp_am_scalar[i] -= scalar_value;
        exp_scalar_am[i] = scalar_value - exp_scalar_am[i];
    }
    am = PSMatrixCreate(0, NULL, 2, 2, 3);
    bm = PSMatrixCreate(0, NULL, 2, 2, 3);
    testAssertWithMessageOrGoto(
        (ok = am != NULL), final, test,
        "could not create matrix %s with shape 2,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bm != NULL), final, test,
        "could not create matrix %s with shape 2,3", "B"
    );
    PSVectorCopy(am, am_values, PSMatrixLength(am));
    PSVectorCopy(bm, bm_values, PSMatrixLength(bm));
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bm, PSMatrixSubtract, nd, shape, exp_am_bm,
                      "PSMatrixSubtract: shape(2,3) + shape(2,3)", test);
    if (!ok) goto final;

    av = PSMatrixCreate(0, NULL, 2, 1, 3);
    bv = PSMatrixCreate(0, NULL, 2, 1, 3);
    testAssertWithMessageOrGoto(
        (ok = av != NULL), final, test,
        "could not create matrix %s with shape 1,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bv != NULL), final, test,
        "could not create matrix %s with shape 1,3", "B"
    );
    PSVectorCopy(av, av_values, PSMatrixLength(av));
    PSVectorCopy(bv, bv_values, PSMatrixLength(bv));
    nd = PSMatrixDimensions(av, shape);
    ok = testMatrixOp(av, bv, PSMatrixSubtract, nd, shape, exp_av_bv,
                      "PSMatrixSubtract: shape(1,3) + shape(1,3)", test);
    if (!ok) goto final;

    bv_t = PSMatrixReshape(bv, 2, 3, 1);
    testAssertWithMessageOrGoto(
        (ok = bv_t != NULL), final, test, "could not reshape matrix %s","B"
    );
    ok = testMatrixOp(av, bv_t, PSMatrixSubtract, nd, shape, exp_av_bv,
                      "PSMatrixSubtract: shape(1,3) + shape(3,1)", test);
    if (!ok) goto final;

    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bv, PSMatrixSubtract, nd, shape, exp_am_bv,
                      "PSMatrixSubtract: shape(2,3) + shape(1,3)", test);
    if (!ok) goto final;

    scalar = PSMatrixCreate(scalar_value, NULL, 1, 1);
    testAssertWithMessageOrGoto(
        (ok = scalar != NULL), final, test,
        "could not create %s matrix with shape 1", "scalar"
    );
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, scalar, PSMatrixSubtract, nd, shape, exp_am_scalar,
                      "PSMatrixSubtract: shape(2,3) + shape(1)", test);
    if (!ok) goto final;
    ok = testMatrixOp(scalar, am, PSMatrixSubtract, nd, shape, exp_scalar_am,
                      "PSMatrixSubtract: shape(1) + shape(2,3)", test);
    if (!ok) goto final;
final:
    PSMatrixFree(am);
    PSMatrixFree(bm);
    PSMatrixFree(av);
    PSMatrixFree(bv);
    PSMatrixFree(bv_t);
    PSMatrixFree(scalar);
    return ok;
}

int testMathsMatrixDivide(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat am_values[] = {
        -2.1282408 ,  1.09726265,  2.1837216 ,
        1.13824648, -0.39490999, -0.45876261
    };
    PSFloat bm_values[] = {
        0.57556455,  0.3889593 , -1.17216897,
        0.6723251 ,  0.34371944, -0.02343899
    };
    PSFloat av_values[] = {-1.56267393, -1.00281759,  0.09569721};
    PSFloat bv_values[] = {0.27088287,  0.93006405, -0.35962201};
    PSFloat scalar_value = 2.0;
    PSFloat exp_am_bm[] = {
        -3.69765789,  2.82102173, -1.86297509,
        1.69300014, -1.14893119, 19.57262971
    };
    PSFloat exp_av_bv[] = {-5.7688179 , -1.07822423, -0.26610498};
    PSFloat exp_am_bv[] = {
        -7.85668295,  1.17977106, -6.07226903,
        4.20198773, -0.42460515, 1.27568002
    };
    PSFloat exp_am_scalar[2 * 3];
    PSFloat exp_scalar_am[2 * 3];
    PSVectorCopy(exp_am_scalar, am_values, (2 * 3));
    PSVectorCopy(exp_scalar_am, am_values, (2 * 3));
    int ok = 1, nd = 0, i;
    int shape[PS_MATRIX_MAX_DIMENSIONS] = {0};
    PSMatrix am = NULL, bm = NULL, av = NULL, bv = NULL, bv_t = NULL,
             scalar = NULL;
    for (i = 0; i < (int)(sizeof(exp_am_scalar)/sizeof(PSFloat)); i++) {
        exp_am_scalar[i] /= scalar_value;
        exp_scalar_am[i] = scalar_value / exp_scalar_am[i];
    }
    am = PSMatrixCreate(0, NULL, 2, 2, 3);
    bm = PSMatrixCreate(0, NULL, 2, 2, 3);
    testAssertWithMessageOrGoto(
        (ok = am != NULL), final, test,
        "could not create matrix %s with shape 2,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bm != NULL), final, test,
        "could not create matrix %s with shape 2,3", "B"
    );
    PSVectorCopy(am, am_values, PSMatrixLength(am));
    PSVectorCopy(bm, bm_values, PSMatrixLength(bm));
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bm, PSMatrixDivide, nd, shape, exp_am_bm,
                      "PSMatrixDivide: shape(2,3) + shape(2,3)", test);
    if (!ok) goto final;

    av = PSMatrixCreate(0, NULL, 2, 1, 3);
    bv = PSMatrixCreate(0, NULL, 2, 1, 3);
    testAssertWithMessageOrGoto(
        (ok = av != NULL), final, test,
        "could not create matrix %s with shape 1,3", "A"
    );
    testAssertWithMessageOrGoto(
        (ok = bv != NULL), final, test,
        "could not create matrix %s with shape 1,3", "B"
    );
    PSVectorCopy(av, av_values, PSMatrixLength(av));
    PSVectorCopy(bv, bv_values, PSMatrixLength(bv));
    nd = PSMatrixDimensions(av, shape);
    ok = testMatrixOp(av, bv, PSMatrixDivide, nd, shape, exp_av_bv,
                      "PSMatrixDivide: shape(1,3) + shape(1,3)", test);
    if (!ok) goto final;

    bv_t = PSMatrixReshape(bv, 2, 3, 1);
    testAssertWithMessageOrGoto(
        (ok = bv_t != NULL), final, test, "could not reshape matrix %s","B"
    );
    ok = testMatrixOp(av, bv_t, PSMatrixDivide, nd, shape, exp_av_bv,
                      "PSMatrixDivide: shape(1,3) + shape(3,1)", test);
    if (!ok) goto final;

    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, bv, PSMatrixDivide, nd, shape, exp_am_bv,
                      "PSMatrixDivide: shape(2,3) + shape(1,3)", test);
    if (!ok) goto final;

    scalar = PSMatrixCreate(scalar_value, NULL, 1, 1);
    testAssertWithMessageOrGoto(
        (ok = scalar != NULL), final, test,
        "could not create %s matrix with shape 1", "scalar"
    );
    nd = PSMatrixDimensions(am, shape);
    ok = testMatrixOp(am, scalar, PSMatrixDivide, nd, shape, exp_am_scalar,
                      "PSMatrixDivide: shape(2,3) + shape(1)", test);
    if (!ok) goto final;
    ok = testMatrixOp(scalar, am, PSMatrixDivide, nd, shape, exp_scalar_am,
                      "PSMatrixDivide: shape(1) + shape(2,3)", test);
    if (!ok) goto final;
final:
    PSMatrixFree(am);
    PSMatrixFree(bm);
    PSMatrixFree(av);
    PSMatrixFree(bv);
    PSMatrixFree(bv_t);
    PSMatrixFree(scalar);
    return ok;
}

int testMatMul(Test *test, int acceleration) {
    int res = 1;
    PSFloat *result = calloc(100, sizeof(PSFloat));
    if (result == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    PSFloat avalues[] = {1, 2, 3, 4, 5, 6}; /* shape: 2, 3 */
    PSFloat bvalues[] = {1, 2, 3, 4, 5, 6}; /* shape: 3, 2 */
    PSFloat cvalues[] = {1, 2, 3, 4, 5, 6, 3, 2, 1, 6, 5, 4}; /* shape:4,3 */
    /*PSFloat avec_values[] = {7, 8, 9};*/ /* shape: 1, 3 */
    PSFloat ab_expected[] = {22, 28, 49, 64}; /* a * b */
    PSFloat act_expected[] = {14, 32, 10, 28, 32, 77, 28, 73}; // a * c(t)
    PSFloat btb_expected[] = {35, 44, 44, 56};/* b(t) * b */
    PSFloat atbt_expected[] = {9, 19, 29, 12, 26,
                              40, 15, 33, 51}; /* a(t) @ b(t) */
    /*PSFloat avat_expected[] = {50, 122};*/
    int ab_l = sizeof(ab_expected) / sizeof(PSFloat);
    int btb_l = sizeof(btb_expected) / sizeof(PSFloat);
    int act_l = sizeof(act_expected) / sizeof(PSFloat);
    int atbt_l = sizeof(atbt_expected) / sizeof(PSFloat);
    /*int avat_l = sizeof(avat_expected) / sizeof(PSFloat);*/
    const char *acceleration_name = PSGetAccelerationName(acceleration);
    if (acceleration_name == NULL) acceleration_name = "";
    PSMathOpts opts = {.acceleration = acceleration};
    opts.transpose = 0;
    int m, n, k;
    int a_shape[] = {2,3};
    int b_shape[] = {3,2};
    int c_shape[] = {4,3};
    /*int avec_shape[] = {3};*/
    m = a_shape[0], n = b_shape[1], k = a_shape[1];
    res = PSMatMul(avalues, bvalues, result, m, n, k, &opts);
    testAssertWithMessageOrGoto(
        res, final, test,
        "Failed PSMatMul(%s,%s) (transp = 0, accel = '%s')", "a","b",
        acceleration_name
    );
    char comparison_label[255] = {0};
    snprintf(comparison_label, 254, "a * b (accel = '%s')", acceleration_name);
    res = compareArrays(result, ab_expected, ab_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatMul(%s,%s,"
        "accel='%s')", "a", "b", acceleration_name
    );

    PSVectorClear(result, 100);
    opts.transpose = 1;
    m = b_shape[1], n = b_shape[1], k = b_shape[0];
    res = PSMatMul(bvalues, bvalues, result, m, n, k, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatMul(%s,%s) (transp = 1, "
        "accel = '%s')", "b","b", acceleration_name
    );
    snprintf(comparison_label, 254, "b(T) * b (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, btb_expected, btb_l, test,comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatMul(%s,%s,"
        "accel='%s')", "b(T)", "b", acceleration_name
    );

    PSVectorClear(result, 100);
    opts.transpose = 2;
    m = a_shape[0], n = c_shape[0], k = a_shape[1];
    res = PSMatMul(avalues, cvalues, result, m, n, k, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatMul(%s,%s) "
        "(transp = 2, accel = '%s')", "a","c", acceleration_name
    );
    snprintf(comparison_label, 254, "a * c(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, act_expected, act_l, test,comparison_label,0,0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatMul(%s,%s,"
        "accel='%s')", "a", "c(T)", acceleration_name
    );

    PSVectorClear(result, 100);
    opts.transpose = 1 | 2;
    m = a_shape[1], n = b_shape[0], k = a_shape[0];
    res = PSMatMul(avalues, bvalues, result, m, n, k, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatMul(%s,%s) (transp = 1|2, "
        "accel = '%s')", "a","b", acceleration_name
    );
    snprintf(comparison_label, 254, "a(T) * b(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, atbt_expected, atbt_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatMul(%s,%s,"
        "accel='%s')", "a(T)", "b(T)", acceleration_name
    );

    /*PSVectorClear(result, 100);
    opts.transpose = 2;
    m = 1, n = a_shape[0], k = 3;
    res = PSMatMul(avec_values, avalues, result, m, n, k, &opts);
    testAssertWithMessageOrGoto(
        res, final, test, "Failed PSMatMul(%s,%s) (transp = 2, "
        "accel = '%s')", "avec","a(T)", acceleration_name
    );
    snprintf(comparison_label, 254, "avec * a(T) (accel = '%s')",
             acceleration_name);
    res = compareArrays(result, avat_expected, avat_l, test,
                        comparison_label, 0, 0);
    testAssertWithMessageOrGoto(
        res, final, test, "Result != expected for PSMatMul(%s,%s,"
        "accel='%s')", "avec", "a(T)", acceleration_name
    );*/
final:
    free(result);
    return res;
}

int testMathsMatMul(TestCase *tc, Test *test) {
    UNUSED(tc);
    int res = 1, numtests = 0, acceleration;
#ifdef HAS_BLAS
    acceleration = PSAcceleration_BLAS;
    res = testMatMul(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    res = testMatMul(test, acceleration);
    if (!res) return 0;
    numtests++;
#endif
    acceleration = PSAcceleration_None;
    res = testMatMul(test, acceleration);
    if (!res) return 0;
    numtests++;
    testAssertWithMessage(numtests > 0, test,
                          "BLAS disabled and no acceleration method suitable "
                          "for PSMatMul%s", "");
    return res;
}

int testMathsVecToMatrix(TestCase *tc, Test *test) {
    UNUSED(tc);
    uint64_t veclen = 6, i;
    PSFloat orig_vec[veclen];
    PSFloat *vec = PSVectorCreate(veclen);
    testAssertNotNull(vec, test);
    for (i = 0; i < veclen; i++) vec[i] = orig_vec[i] = PSGaussianRandom(0, 1);
    int shape[PS_MAX_SEQUENCE_LENGTH] = {2, 3, 0};
    int success = 1;
    PSMatrix matrix = PSVectorConvertToMatrix(vec, veclen, 2, shape);
    testAssertNotNull(matrix, test);
    vec = NULL;
    success = compareArrays(matrix, orig_vec, veclen, test, NULL, 0, 0);
    if (!success) goto final;
    int mshape[PS_MAX_SEQUENCE_LENGTH] = {0};
    int ndims = 0, d;
    ndims = PSMatrixDimensions(matrix, mshape);
    for (d = 0; d < ndims; d++) {
        success = (shape[d] == mshape[d]);
        testAssertWithMessageOrGoto(
            success, final, test, "matrix shape[%d] != expected: %d != %d",
            d, shape[d], mshape[d]
        );
    }
final:
    free(vec);
    PSMatrixFree(matrix);
    return success;
}

int testActSigmoid(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSSigmoidS(x[i]);
    int ok = 1;
    int decrnd = NORMAL_PRECISION_DEC;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSSigmoid(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", decrnd,0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSigmoid(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", decrnd, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSigmoid(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", decrnd, 0);
    if (!ok) return 0;
    return ok;
}

int testActSigmoidDeriv(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSSigmoidDerivativeS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSSigmoidDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSigmoidDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSigmoidDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testActTanh(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSTanh(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSTanhActivation(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSTanhActivation(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSTanhActivation(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testActTanhDeriv(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSTanhDerivativeS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSTanhDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSTanhDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSTanhDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testActRelu(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSReluS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSRelu(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSRelu(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSRelu(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testActReluDeriv(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[6] = {1.0, 8.3, -2.0, -1.0, 0.0, 18.5};
    PSFloat cmp_res[6] = {0};
    PSFloat res[6] = {0};
    for (int i = 0; i < 6; i++) cmp_res[i] = PSReluDerivativeS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSReluDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSReluDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSReluDerivative(x, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "No Acceleration:", 0, 0);
    if (!ok) return 0;
    return ok;
}

int testActGelu(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[4] = {-1.79316703, -1.38185011, 0.98695828, -0.43326748};
    PSFloat y[4] = {-0.06547327, -0.11563015, 0.82708914, -0.1440327};
    PSFloat cmp_res[4] = {0};
    PSFloat res[4] = {0};
    for (int i = 0; i < 4; i++) cmp_res[i] = PSGeluS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSGelu(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "Accelerate Framework", 0, 4);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluS: Accelerate Framework",
                       0, 4);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSGelu(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "AVX", 0, 4);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluS: AVX", 0, 4);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSGelu(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "No Acceleration:", 0, 4);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluS: No Acceleration:",0,4);
    if (!ok) return 0;
    return ok;
}

int testActGeluDeriv(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[4] = {-0.06547327, -0.11563015, 0.82708914, -0.1440327};
    PSFloat y[4] = {0.44783455, 0.40815094, 1.03017281, 0.3858705};
    PSFloat cmp_res[4] = {0};
    PSFloat res[4] = {0};
    for (int i = 0; i < 4; i++) cmp_res[i] = PSGeluDerivativeS(x[i]);
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSGeluDerivative(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "Accelerate Framework", 0, 3);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluDerivativeS: "
                      "Accelerate Framework", 0, 3);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSGeluDerivative(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "AVX", 0, 4);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluDerivativeS: AVX", 0, 3);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSGeluDerivative(x, res, 4, &opts);
    ok = compareArrays(res, y, 4, test, "No Acceleration:", 0, 3);
    if (!ok) return 0;
    ok = compareArrays(res, cmp_res, 4, test, "PSGeluDerivativeS: "
                       "No Acceleration:",0,3);
    if (!ok) return 0;
    return ok;
}

int testActSoftmax(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat x[3] = {0.3, 0.5, 0.1};
    PSFloat expected[3] = {
        0.3289329222889067, 0.4017595785333554, 0.2693074991777379
    };
    PSFloat res[3] = {0};
    int ok = 1;
    PSMathOpts opts = {0};
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    opts.acceleration = PSAcceleration_ACF;
    PSSoftmax(x, res, 3, &opts);
    ok = compareArrays(res, expected, 3, test, "Accelerate Framework", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSoftmax(x, res, 3, &opts);
    ok = compareArrays(res, expected, 3, test, "AVX", 5, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSoftmax(x, res, 3, &opts);
    ok = compareArrays(res, expected, 3, test, "No Acceleration:", 5, 0);
    if (!ok) return 0;
    return ok;
}

int testDefaultOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.237288, -0.132154, -19.9703, -0.306606, 0.100178, -0.529931,
        0.000580211, 1.00021
    };
    uint64_t len = 8;
    PSFloat params[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    PSFloat rate = 0.1;
    int acceleration = PSAcceleration_None;
    int ok = PSDefaultOptimization(
        params, gradients, NULL, NULL, NULL, NULL, NULL, rate, 0.0, len,
        acceleration, 0, NULL
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSDefaultOptimization(
        params, gradients, NULL, NULL, NULL, NULL, NULL, rate, 0.0, len,
        acceleration, 0, NULL
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSDefaultOptimization(
        params, gradients, NULL, NULL, NULL, NULL, NULL, rate, 0.0, len,
        acceleration, 0, NULL
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testMomentumOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.237287, -0.132154, -58.3503, -0.306606, 0.100178,
        -0.719931, 0.00168261, 1.00061
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSDefaultOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSDefaultOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF.", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSDefaultOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testNesterovOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.237287, -0.132154, -92.8923, -0.306606, 0.100178, -0.890931,
        0.00267477, 1.00098
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSNesterovOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSNesterovOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSNesterovOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, NULL
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testAdaDeltaOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.237282, -0.132154, 0.226879, -0.306606, 0.100176, -0.432777,
        0.00276497, 1.00236
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem1[len];
    PSFloat mem2[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSAdaDeltaOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdaDeltaOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdaDeltaOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testWindowGradOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.235378, -0.132154, -0.537744, -0.306606, 0.0995521,
        -1.1974, 0.74998, 1.66075
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSWindowGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSWindowGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSWindowGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testAdaGradOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.235378, -0.132154, 0.059015, -0.306606, 0.0995521,
        -0.600641, 0.17051, 1.16922
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSAdaGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdaGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdaGradOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testRMSPropOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.23709731, -0.13215413,  0.17516139, -0.30660592,  0.10011567,
        -0.48449495, 0.05392763,  1.05029508
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.01, momentum = 0.0;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSRMSPropOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSRMSPropOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSRMSPropOptimization(
            params, gradients, mem, NULL, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testAdamOptimization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.182258, -0.132154, -1.11413, -0.306606, 0.0816279,
        -1.77369, 1.32767, 2.30041
    };
    uint64_t len = 8;
    PSFloat params[len];
    PSFloat mem1[len];
    PSFloat mem2[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    PSFloat rate = 0.1, momentum = 0.9;
    int iterations = 2, ok, i;
    int acceleration = PSAcceleration_None;
    for (i = 0; i < iterations; i++) {
        ok = PSAdamOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdamOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    memset(mem1, 0, len * sizeof(PSFloat));
    memset(mem2, 0, len * sizeof(PSFloat));
    for (i = 0; i < iterations; i++) {
        ok = PSAdamOptimization(
            params, gradients, mem1, mem2, NULL, NULL, NULL, rate, momentum,
            len, acceleration, i, &optimization_train_opts
        );
        testAssert(ok, test);
    }
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testL1WeightDecay(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.0237288, -0.0132154, 0.0229726, -0.0306606, 0.0100178,
        -0.0429931, 0, 0.1
    };
    uint64_t len = 8;
    PSFloat params[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    PSFloat l1 = 0.1, l1_loss = 0.0;
    int batches = 1;
    int acceleration = PSAcceleration_None;
    int ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testL2WeightDecay(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat orig_params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.0237288, -0.0132154, 0.0229726, -0.0306606, 0.0100178,
        -0.0429931, 0, 0.1
    };
    uint64_t len = 8;
    PSFloat params[len];
    memcpy(params, orig_params, len * sizeof(PSFloat));
    PSFloat l2 = 0.1, l2_loss = 0.0;
    int batches = 1;
    int acceleration = PSAcceleration_None;
    int ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(params, orig_params, len * sizeof(PSFloat));
    ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        1, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(params, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
#endif
    return 1;
}

int testL1Regularization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat orig_gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.100003, -0.1, 202.1, -0.1, 0.100001, 0.9, -0.105802, 0.0978832
    };
    PSFloat expected_l1_loss = getRoundedFloatDec(2.43588, 4);
    uint64_t len = 8;
    PSFloat gradients[len];
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    PSFloat l1 = 0.1, l1_loss = 0.0;
    int batches = 1;
    int acceleration = PSAcceleration_None;
    int ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
    l1_loss = getRoundedFloatDec(l1_loss, 4);
    testAssertWithMessage(
        l1_loss == expected_l1_loss, test, "L1 Loss %g != expected %g",
        l1_loss, expected_l1_loss
    );
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    l1_loss = 0.0;
    ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
    l1_loss = getRoundedFloatDec(l1_loss, 4);
    testAssertWithMessage(
        l1_loss == expected_l1_loss, test, "(ACF) L1 Loss %g != expected %g",
        l1_loss, expected_l1_loss
    );
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    l1_loss = 0.0;
    ok = PSLRegularization(
        l1, 0.0, params, gradients, NULL, len, &l1_loss, NULL, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
    l1_loss = getRoundedFloatDec(l1_loss, 4);
    testAssertWithMessage(
        l1_loss == expected_l1_loss, test, "(AVX) L1 Loss %g != expected %g",
        l1_loss, expected_l1_loss
    );
#endif
    return 1;
}

int testL2Regularization(TestCase *tc, Test *test) {
    UNUSED(tc);
    PSFloat orig_gradients[] = {
        0.00000302, 0.0, 202.0, 0.0, 0.00000099, 1.0, -0.00580211, -0.00211678
    };
    PSFloat params[] = {
        0.23728831, -0.13215413,  0.22972574, -0.30660592,
        0.10017828, -0.42993062, 0.0, 1.0
    };
    PSFloat expected[] = {
        0.0237319, -0.0132154, 202.023, -0.0306606, 0.0100188, 0.957007,
        -0.00580211, 0.0978832
    };
    PSFloat expected_l2_loss = getRoundedFloatDec(1.41543, 4);
    uint64_t len = 8;
    PSFloat gradients[len];
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    PSFloat l2 = 0.1, l2_loss = 0.0;
    int batches = 1;
    int acceleration = PSAcceleration_None;
    int ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "No Accel.", 4, 0);
    if (!ok) return 0;
    l2_loss = getRoundedFloatDec(l2_loss, 4);
    testAssertWithMessage(
        l2_loss == expected_l2_loss, test, "L2 Loss %g != expected %g",
        l2_loss, expected_l2_loss
    );
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    acceleration = PSAcceleration_ACF;
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    l2_loss = 0.0;
    ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "ACF", 4, 0);
    if (!ok) return 0;
    l2_loss = getRoundedFloatDec(l2_loss, 4);
    l2_loss = getRoundedFloatDec(l2_loss, 4);
    testAssertWithMessage(
        l2_loss == expected_l2_loss, test, "(ACF) L2 Loss %g != expected %g",
        l2_loss, expected_l2_loss
    );
#endif
#ifdef USE_AVX
    acceleration = PSAcceleration_AVX;
    memcpy(gradients, orig_gradients, len * sizeof(PSFloat));
    l2_loss = 0.0;
    ok = PSLRegularization(
        0.0, l2, params, gradients, NULL, len, NULL, &l2_loss, batches,
        0, acceleration
    );
    testAssert(ok, test);
    ok = compareArrays(gradients, expected, len, test, "AVX", 4, 0);
    if (!ok) return 0;
    l2_loss = getRoundedFloatDec(l2_loss, 4);
    l2_loss = getRoundedFloatDec(l2_loss, 4);
    testAssertWithMessage(
        l2_loss == expected_l2_loss, test, "(AVX) L2 Loss %g != expected %g",
        l2_loss, expected_l2_loss
    );
#endif
    return 1;
}

int testDatasetLoad(TestCase *tc, Test *test) {
    UNUSED(tc);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, "resources/test-dataset.psdata", path), test
    );
    testAssert(PSFileExists(path), test);
    uint64_t datalen = 0,
             expected_len = (sizeof(expected_loaded_dataset)/sizeof(PSFloat));
    PSFloat *data = PSLoadDataFromFile(path, &datalen);
    int success = (data != NULL);
    testAssertWithMessageOrGoto(
        success, final, test, "failed to load dataset", ""
    );
    success = (datalen == expected_len);
    testAssertWithMessageOrGoto(
        success, final, test,
        "dataset length differs from expected: %" PRIu64 " != %" PRIu64,
        datalen, expected_len
    );
    success = compareArrays(data, expected_loaded_dataset, datalen,
                            test, NULL, 4, 0);
final:
    free(data);
    return success;
}

int testDatasetSave(TestCase *tc, Test *test) {
    UNUSED(tc);
    char tmpfile[PATH_MAX];
    getTmpFileName("tests-save-dataset", ".psdata", tmpfile);
    uint64_t len = (sizeof(expected_loaded_dataset)/sizeof(PSFloat));
    int success = PSSaveDataToFile(tmpfile, expected_loaded_dataset, len, 0);
    testAssertWithMessage(success, test, "could not save dataset", "");
    testAssertWithMessage(
        PSFileExists(tmpfile), test, "saved file not found", ""
    );
    uint64_t datalen = 0;
    PSFloat *data = PSLoadDataFromFile(tmpfile, &datalen);
    success = (data != NULL);
    testAssertWithMessageOrGoto(
        success, final, test, "failed to load dataset", ""
    );
    success = (datalen == len);
    testAssertWithMessageOrGoto(
        success, final, test,
        "dataset length differs from expected: %" PRIu64 " != %" PRIu64,
        datalen, len
    );
    success = compareArrays(data, expected_loaded_dataset, datalen,
                            test, NULL, 4, 0);
final:
    free(data);
    return success;
}

int testDatasetSaveBinary(TestCase *tc, Test *test) {
    UNUSED(tc);
    char tmpfile[PATH_MAX];
    getTmpFileName("tests-save-dataset-bin", ".psdata", tmpfile);
    uint64_t len = (sizeof(expected_loaded_dataset)/sizeof(PSFloat));
    int success = PSSaveDataToFile(
        tmpfile, expected_loaded_dataset, len, PS_IO_BINARY_MODE
    );
    testAssertWithMessage(success, test, "could not save binary dataset", "");
    testAssertWithMessage(
        PSFileExists(tmpfile), test, "saved file not found", ""
    );
    uint64_t datalen = 0;
    PSFloat *data = PSLoadDataFromFile(tmpfile, &datalen);
    success = (data != NULL);
    testAssertWithMessageOrGoto(
        success, final, test, "failed to load binary dataset", ""
    );
    success = (datalen == len);
    testAssertWithMessageOrGoto(
        success, final, test,
        "dataset length differs from expected: %" PRIu64 " != %" PRIu64,
        datalen, len
    );
    success = compareArrays(data, expected_loaded_dataset, datalen,
                            test, NULL, 4, 0);
final:
    free(data);
    return success;
}

#ifdef USE_AVX

int testAVXDot(TestCase *tc, Test *test) {
    UNUSED(tc);

    PSFloat x2[2] = {1.0, 2.0};
    PSFloat y2[2] = {0.5, 0.5};

    PSFloat x4[4] = {1.0, 1.0, 2.0, 2.0};
    PSFloat y4[4] = {0.5, 0.5, 1.0, 0.5};

    PSFloat x8[8] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0};
    PSFloat y8[8] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0};

    PSFloat x16[16] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                       0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0};
    PSFloat y16[16] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                       1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5};

    PSFloat x32[32] = {1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0,
                       0.5, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0,
                       1.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0,
                       1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 1.0, 0.5};
    PSFloat y32[32] = {0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
                       1.0, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5,
                       2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5, 1.0,
                       0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 0.5};
    PSFloat avx_res, cmp_res;
    if (AVX_MAX_VECTOR_SIZE >= 32) {
        avx_res = AVXDotProduct(x32, y32, 32, NULL);
        cmp_res = test_dot(x32, y32, 32);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[32]: Expected %g != %g",
            cmp_res, avx_res
        );
    }
    avx_res = AVXDotProduct(x16, y16, 16, NULL);
    cmp_res = test_dot(x16, y16, 16);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[16]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x8, y8, 8, NULL);
    cmp_res = test_dot(x8, y8, 8);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[8]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x4, y4, 4, NULL);
    cmp_res = test_dot(x4, y4, 4);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[4]: Expected %g != %g",
        cmp_res, avx_res
    );

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        avx_res = AVXDotProduct(x2, y2, 2, NULL);
        cmp_res = test_dot(x2, y2, 2);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[2]: Expected %g != %g",
            cmp_res, avx_res
        );
    }
    return 1;
}

int testAVXSquare(TestCase *tc, Test *test) {
    UNUSED(tc);

    PSFloat x2[2] = {1.0, 2.0};/* 5 */
    PSFloat x4[4] = {1.0, 1.0, 2.0, 2.0};/* 10 */
    PSFloat x8[8] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0};/* 20 */

    PSFloat x16[16] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0}; /* 40 */

    PSFloat avx_res = AVXDotProduct(x16, x16, 16, NULL);
    PSFloat cmp_res = test_dot(x16, x16, 16);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[16]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x8, x8, 8, NULL);
    cmp_res = test_dot(x8, x8, 8);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[8]: Expected %g != %g",
        cmp_res, avx_res
    );

    avx_res = AVXDotProduct(x4, x4, 4, NULL);
    cmp_res = test_dot(x4, x4, 4);
    testAssertWithMessage(
        (avx_res == cmp_res), test, "AVX[4]: Expected %g != %g",
        cmp_res, avx_res
    );

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        avx_res = AVXDotProduct(x2, x2, 2, NULL);
        cmp_res = test_dot(x2, x2, 2);
        testAssertWithMessage(
            (avx_res == cmp_res), test, "AVX[2]: Expected %g != %g",
            cmp_res, avx_res
        );
    }

    return 1;
}

int testAVXMultiplyVal(TestCase *tc, Test *test) {
    UNUSED(tc);
    int i;
    PSFloat x[4] = {0.0, 1.0, 2.0, 3.0};
    PSFloat val = 2.0;
    PSFloat y[4] = {0.0, 2.0, 4.0, 6.0};
    PSFloat dest[4] = {0.0, 0.0, 0.0, 0.0};

    PSFloat x2[2] = {2.0, 3.0};
    PSFloat y2[2] = {4.0, 6.0};
    PSFloat dest2[2] = {0.0, 0.0};

    AVXMultiplyValue(x, val, 4, dest, 0);
    for (i = 0; i < 4; i++) {
        testAssertWithMessage(
            (dest[i] == y[i]), test, "Store Mode Norm[4]: Expected %g != %g",
            y[i], dest[i]
        );
    }

    AVXMultiplyValue(x, val, 4, dest, AVX_STORE_MODE_ADD);
    for (i = 0; i < 4; i++) {
        testAssertWithMessage(
            (dest[i] == (y[i] + y[i])), test,
            "Store Mode Norm[4]: Expected %g != %g", y[i], dest[i]
        );
    }

    if (AVX_MIN_VECTOR_SIZE <= 2) {
        AVXMultiplyValue(x2, val, 2, dest2, 0);
        for (i = 0; i < 2; i++) {
            testAssertWithMessage(
                (dest2[i] == y2[i]), test,
                "Store Mode Norm[2]: Expected %g != %g", y2[i], dest2[i]
            );
        }

        AVXMultiplyValue(x2, val, 2, dest2, AVX_STORE_MODE_ADD);
        for (i = 0; i < 2; i++) {
            testAssertWithMessage(
                (dest2[i] == (y2[i] + y2[i])), test,
                "Store Mode Norm[2]: Expected %g != %g", y2[i], dest2[i]
            );
        }
    }

    return 1;
}

#endif

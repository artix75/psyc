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
#include "../operator_layer.h"
#include "../mnist.h"
#include "../maths.h"
#include "../activation.h"
#include "../optimization.h"
#include "../utils.h"
#include "../debug.h"
#include "../log.h"
#include "../cifar.h"
#ifdef USE_AVX
#include "../avx.h"
#endif

#define PRETRAINED_FULL_NETWORK "resources/pretrained.mnist.psmodel"
#define CONVOLUTIONAL_NETWORK "resources/cnn.data"
#define CONVOLUTIONAL_TRAINED_NETWORK "resources/pretrained.cnn.psmodel"
#define CONVOLUTIONAL_CIFAR_NETWORK "resources/cifar-cnn.psmodel"
#define CIFAR_IMAGE_PATH "resources/cifar-image.data"
#define CIFAR_LABEL_PATH "resources/cifar-label.data"
#define RECURRENT_NETWORK "resources/rnn.data"
#define NORMALIZATION_NETWORK "resources/normalization_nn.psmodel"
#define NORMALIZATION_NETWORK_BP "resources/normalization_nn_bp.psmodel"
#define DROPOUT_NETWORK "resources/dropout_nn.psmodel"
#define OP_CONCAT_NETWORK "resources/concatenate-operator-layer.psmodel"
#define OP_ADD_NETWORK "resources/add-operator-layer.psmodel"
#define OP_MUL_NETWORK "resources/multiply-operator-layer.psmodel"
#define ENCDEC_BASIC_NETWORK "resources/encoder-decoder.basic.psmodel"
#define TEST_IMAGE_FILE "resources/t10k-images-idx3-ubyte.gz"
#define TEST_LABEL_FILE "resources/t10k-labels-idx1-ubyte.gz"
#define TEST_IMAGE_SIZE 28
#define TEST_INPUT_SIZE TEST_IMAGE_SIZE *TEST_IMAGE_SIZE
#define BP_GRADIENTS_CHECKS 8
#define BP_CONV_GRADIENTS_CHECKS 4
#define CONV_L1F0_BIAS 0.02630446809718423

#define PRETRAINED_MNIST_NETSIZE 3

#define RNN_INPUT_SIZE  4
#define RNN_HIDDEN_SIZE 2
#define RNN_TIMES       4
#define RNN_LEARNING_RATE 0.005

#define LSTM_LEARNING_RATE 0.1
#define LSTM_TIMES 3
#define LSTM_EPOCHS 1
#define LSTM_BATCHES 1

#define getNetwork(tc) ((PSNeuralNetwork*)(tc->data[0]))
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

TestCase *fullNetworkTests;
TestCase *convNetworkTests;
TestCase *recurrentNetworkTests;
TestCase *LSTMNetworkTests;
TestCase *GRUNetworkTests;
TestCase *NormalizationNetworkTests;
TestCase *DropoutNetworkTests;
TestCase *ConcatOperatorLayerTests;
TestCase *AddOperatorLayerTests;
TestCase *MulOperatorLayerTests;
TestCase *EncoderDecoderTests;

#ifdef USE_AVX
TestCase *AVXTests;
#endif

TestCase *mathsTests;
TestCase *activationTests;
TestCase *optimizationTests;

int genericSetup (TestCase *test_case);
int genericTeardown (TestCase *test_case);
int RNNSetup (TestCase *test_case);
int RNNTeardown (TestCase *test_case);
int LSTMSetup (TestCase *test_case);
int GRUSetup(TestCase *test_case);
int encoderDecoderSetup(TestCase *test_case);
int encoderDecoderTeardown(TestCase *test_case);

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
int testMathsTanh(TestCase *tc, Test *test);
int testMathsSqrt(TestCase *tc, Test *test);
int testMathsNeg(TestCase *tc, Test *test);
int testMathsAbs(TestCase *tc, Test *test);
int testMathsMean(TestCase *tc, Test *test);
int testMathsVar(TestCase *tc, Test *test);
int testMathsStd(TestCase *tc, Test *test);
int testMathVectorFill(TestCase *tc, Test *test);
int testMathsMatMul(TestCase *tc, Test *test);
int testMathsMatrixCopy(TestCase *tc, Test *test);
int testMathsMatrixDup(TestCase *tc, Test *test);
int testMathsMatrixTranspose(TestCase *tc, Test *test);
int testMathsMatrixExpand(TestCase *tc, Test *test);
int testMathsMatrixProduct(TestCase *tc, Test *test);
int testMathsMatrixProductMV(TestCase *tc, Test *test);
int testMathsMatrixProductVM(TestCase *tc, Test *test);

int testActSigmoid(TestCase *tc, Test *test);
int testActSigmoidDeriv(TestCase *tc, Test *test);
int testActTanh(TestCase *tc, Test *test);
int testActTanhDeriv(TestCase *tc, Test *test);
int testActRelu(TestCase *tc, Test *test);
int testActReluDeriv(TestCase *tc, Test *test);
int testActSoftmax(TestCase *tc, Test *test);

int testDefaultOptimization(TestCase *tc, Test *test);
int testMomentumOptimization(TestCase *tc, Test *test);
int testNesterovOptimization(TestCase *tc, Test *test);
int testAdaDeltaOptimization(TestCase *tc, Test *test);
int testWindowGradOptimization(TestCase *tc, Test *test);
int testAdaGradOptimization(TestCase *tc, Test *test);
int testAdamOptimization(TestCase *tc, Test *test);
int testL1WeightDecay(TestCase *tc, Test *test);
int testL2WeightDecay(TestCase *tc, Test *test);
int testL1Regularization(TestCase *tc, Test *test);
int testL2Regularization(TestCase *tc, Test *test);

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


/* psyc.c function prototypes */

PSGradient ***backprop(PSNeuralNetwork *network, PSFloat *x, PSFloat *y,
                       PSTrainingOptions *opts, PSGradient **gradients);

PSFloat updateNetworkParameters(PSNeuralNetwork *network,
                                PSFloat *training_data,
                                int batch_size, int elements_count,
                                PSTrainingOptions* opts, PSFloat rate,
                                PSGradient **memory_gradients1,
                                PSGradient **memory_gradients2, ...);

PSFloat *PSGetDropoutMask(PSLayer *layer, int t);

static int compareFloats(PSFloat a, PSFloat b, int rounding, int precision);
static int compareArrays(PSFloat *arr, PSFloat *exp, int len, Test* test,
                         char *descr, int rounding, int precision);
char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
int PSUpdateDelta(PSMatrix destdelta, PSMatrix srcdelta, PSMatrix weights,
                  int seqlen, int acceleration);

int testlen = 0;

int pretrained_mnist_layers_size[PRETRAINED_MNIST_NETSIZE] = {784,30,10};

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

PSTrainingOptions optimization_train_opts = {0};

int compareNetworks(PSNeuralNetwork *net1, PSNeuralNetwork *net2, Test* test);
int compareNetworkChain(PSNeuralNetwork *net1, PSNeuralNetwork *net2, Test* test);

static int testRecurrentNetworkMode(PSNeuralNetwork *network,
                                    PSRecurrentNetworkMode mode, Test *test);
PSFloat *readSerializedFloatArray(FILE *in, char *sep, int *length,
                                  int maxlen, int capacity);

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
           optimization_tests = 1, fullnet_tests = 1, convnet_tests = 1,
           rnn_tests = 1, lstm_tests = 1, gru_tests = 1,
           normalization_tests = 1, dropout_tests = 1, encdec_tests = 1,
           concat_op_tests = 1, add_op_tests = 1, mul_op_tests = 1;

static int *test_ptrs[] = {
    &avx_tests, &maths_tests, &activation_tests, &optimization_tests,
    &fullnet_tests, &convnet_tests, &rnn_tests, &lstm_tests, &gru_tests,
    &normalization_tests, &dropout_tests, &concat_op_tests, &add_op_tests,
    &mul_op_tests, &encdec_tests
};

static char*test_ids[] = {
    "avx", "maths", "activation", "optimization", "fully-connected",
    "convolutional", "rnn", "lstm", "gru", "normalization", "dropout",
    "concatenate-layer", "add-layer", "multiply-layer", "encoder_decoder",
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
        addTest(mathsTests, "Tanh", NULL, testMathsTanh);
        addTest(mathsTests, "Sqrt", NULL, testMathsSqrt);
        addTest(mathsTests, "Negate", NULL, testMathsNeg);
        addTest(mathsTests, "Abs.", NULL, testMathsAbs);
        addTest(mathsTests, "Mean", NULL, testMathsMean);
        addTest(mathsTests, "Variance", NULL, testMathsVar);
        addTest(mathsTests, "StdDev", NULL, testMathsStd);
        addTest(mathsTests, "Vector Fill", NULL, testMathVectorFill);
        addTest(mathsTests, "MatMul (vectors)", NULL, testMathsMatMul);
        addTest(mathsTests, "Matrix Copy", NULL, testMathsMatrixCopy);
        addTest(mathsTests, "Matrix Dup.", NULL, testMathsMatrixDup);
        addTest(mathsTests, "Matrix Expand", NULL, testMathsMatrixExpand);
        addTest(mathsTests, "Matrix Transp.", NULL, testMathsMatrixTranspose);
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

    if (fullnet_tests) {
        fullNetworkTests = createTest("Fully Connected Network");
        fullNetworkTests->setup = genericSetup;
        fullNetworkTests->teardown = genericTeardown;
        addTest(fullNetworkTests, "Load", NULL, testFullLoad);
        addTest(fullNetworkTests, "Forward", NULL, testFullForward);
        addTest(fullNetworkTests, "Accuracy", NULL, testFullAccuracy);
        addTest(fullNetworkTests, "Backprop", NULL, testFullBackprop);
        addTest(fullNetworkTests, "Clone", NULL, testGenericClone);
        addTest(fullNetworkTests, "Save", NULL, testGenericSave);
        performTests(fullNetworkTests);
        tot_tests += fullNetworkTests->count;
        tot_failed += fullNetworkTests->failed_count;
        deleteTest(fullNetworkTests);
    }

    if (convnet_tests) {
        convNetworkTests = createTest("Convolutional Network");
        convNetworkTests->setup = genericSetup;
        convNetworkTests->teardown = genericTeardown;
        addTest(convNetworkTests, "Load", NULL, testConvLoad);
        addTest(convNetworkTests, "Forward", NULL, testConvForward);
        addTest(convNetworkTests, "Backprop", NULL, testConvBackprop);
        /*addTest(convNetworkTests, "Accuracy", NULL, testConvAccuracy);*/
        addTest(convNetworkTests, "CIFAR Backprop", NULL, testConvCIFAR);
        addTest(convNetworkTests, "Clone", NULL, testGenericClone);
        addTest(convNetworkTests, "Save", NULL, testGenericSave);
        performTests(convNetworkTests);
        tot_tests += convNetworkTests->count;
        tot_failed += convNetworkTests->failed_count;
        deleteTest(convNetworkTests);
    }

    if (rnn_tests) {
        recurrentNetworkTests = createTest("Recurrent Network");
        recurrentNetworkTests->setup = RNNSetup;
        recurrentNetworkTests->teardown = RNNTeardown;
        addTest(recurrentNetworkTests, "Load", NULL, testRNNLoad);
        addTest(recurrentNetworkTests, "Forward", NULL, testRNNForward);
        addTest(recurrentNetworkTests, "Backprop", NULL, testRNNBackprop);
        addTest(recurrentNetworkTests, "Step", NULL, testRNNStep);
        addTest(recurrentNetworkTests, "Clone", NULL, testGenericClone);
        addTest(recurrentNetworkTests, "Save", NULL, testGenericSave);
        addTest(recurrentNetworkTests, "OneHot", NULL, testRNNOneHot);
        performTests(recurrentNetworkTests);
        tot_tests += recurrentNetworkTests->count;
        tot_failed += recurrentNetworkTests->failed_count;
        deleteTest(recurrentNetworkTests);
    }

    if (lstm_tests) {
        LSTMNetworkTests = createTest("LSTM Network");
        LSTMNetworkTests->setup = LSTMSetup;
        LSTMNetworkTests->teardown = RNNTeardown;
        /* addTest(LSTMNetworkTests, "Load", NULL, testLSTMLoad); */
        addTest(LSTMNetworkTests, "Train", NULL, testLSTMTrain);
        addTest(LSTMNetworkTests, "Backprop", NULL, testLSTMBackprop);
        addTest(LSTMNetworkTests, "Clone", NULL, testGenericClone);
        addTest(LSTMNetworkTests, "Save", NULL, testGenericSave);
        performTests(LSTMNetworkTests);
        tot_tests += LSTMNetworkTests->count;
        tot_failed += LSTMNetworkTests->failed_count;
        deleteTest(LSTMNetworkTests);
    }
    if (gru_tests) {
        GRUNetworkTests = createTest("GRU Network");
        GRUNetworkTests->setup = GRUSetup;
        GRUNetworkTests->teardown = RNNTeardown;
        /* addTest(GRUNetworkTests, "Load", NULL, testGRULoad); */
        addTest(GRUNetworkTests, "Train", NULL, testGRUTrain);
        addTest(GRUNetworkTests, "Backprop", NULL, testGRUBackprop);
        addTest(GRUNetworkTests, "Clone", NULL, testGenericClone);
        addTest(GRUNetworkTests, "Save", NULL, testGenericSave);
        performTests(GRUNetworkTests);
        tot_tests += GRUNetworkTests->count;
        tot_failed += GRUNetworkTests->failed_count;
        deleteTest(GRUNetworkTests);
    }
    if (normalization_tests) {
        NormalizationNetworkTests = createTest("Normalization Network");
        NormalizationNetworkTests->setup = genericSetup;
        NormalizationNetworkTests->teardown = genericTeardown;
        addTest(NormalizationNetworkTests, "Load", NULL, testNormalizationLoad);
        addTest(NormalizationNetworkTests, "Forward", NULL,
               testNormalizationForward);
        addTest(NormalizationNetworkTests, "Backprop", NULL,
               testNormalizationBackprop);
        addTest(NormalizationNetworkTests, "Save", NULL, testGenericSave);
        performTests(NormalizationNetworkTests);
        tot_tests += NormalizationNetworkTests->count;
        tot_failed += NormalizationNetworkTests->failed_count;
        deleteTest(NormalizationNetworkTests);
    }
    if (dropout_tests) {
        DropoutNetworkTests = createTest("Dropout Network");
        DropoutNetworkTests->setup = genericSetup;
        DropoutNetworkTests->teardown = genericTeardown;
        addTest(DropoutNetworkTests, "Load", NULL, testDropoutLoad);
        addTest(DropoutNetworkTests, "Forward", NULL,
               testDropoutForward);
        addTest(DropoutNetworkTests, "Backprop", NULL,
               testDropoutBackprop);
        addTest(DropoutNetworkTests, "Save", NULL, testGenericSave);
        performTests(DropoutNetworkTests);
        tot_tests += DropoutNetworkTests->count;
        tot_failed += DropoutNetworkTests->failed_count;
        deleteTest(DropoutNetworkTests);
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
    if (add_op_tests) {
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
    gettimeofday(&end_t, NULL);
    time_t elapsed = PSGetElapsedTimeUS(start_t, end_t);
    char *elapsed_str = PSGetElapsedTimeString(elapsed, OPT_TIME_FULL);
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
    PSNeuralNetwork *network = PSCreateNetwork("Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    PSFloat *test_data = NULL;
    char test_img_path[PATH_MAX] = {0};
    char test_lbl_path[PATH_MAX] = {0};
    char *path_dup = strdup(executable_path);
    if (path_dup == NULL) {
        PSPrintMemoryErrorMsg();
        return 0;
    }
    char *root = dirname(path_dup);
    if (root == NULL) {
        PSErr(__func__, "Could not determine PsyC path from '%s'",
              executable_path);
        free(path_dup);
        return 0;
    }
    root = dirname(root);
    if (root == NULL) {
        PSErr(__func__, "Could not determine PsyC path from '%s'",
              executable_path);
        free(path_dup);
        return 0;
    }
    if (!joinPath(root, TEST_IMAGE_FILE, test_img_path)) {
        free(path_dup);
        return 0;
    }
    if (!joinPath(root, TEST_LABEL_FILE, test_lbl_path)) {
        free(path_dup);
        return 0;
    }
    free(path_dup);
    testlen = PSLoadMNISTData(DATA_TYPE_TEST, test_img_path, test_lbl_path,
                              &test_data);
    test_case->data[1] = test_data;
    if (test_data == NULL) {
        return 0;
    }
    return 1;
}

int genericTeardown(TestCase *test_case) {
    PSNeuralNetwork *network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int RNNSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("RNN Test Network");
    if (network == NULL) {
        PSErr(NULL, "\nCould not create network!");
        return 0;
    }
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, Recurrent, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    if (network->size < 1) {
        PSErr(NULL, "\nCould not add all layers!");
        return 0;
    }
    network->layers[1]->flags |= FLAG_NO_BIAS;
    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;

    int i, j, w;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
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
                int prev_size = network->layers[i - 1]->size;
                PSFloat *input_weights = layer->weights[0] + (j * prev_size);
                for (w = 0; w < prev_size; w++)
                    input_weights[w] = rnn_inputs_weights[j][w];
            }
        }
    }
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSRecurrentNetworkMode rnn_mode = network->rnn_mode;
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
    test_case->data[0] = network;
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
    PSNeuralNetwork *network = getNetwork(test_case);
    if (network != NULL) PSDeleteNetwork(network);
    PSFloat *test_data = getTestData(test_case);
    if (test_data != NULL) free(test_data);
    free(test_case->data);
    test_case->data = NULL;
    return 1;
}

int LSTMSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("LSTM Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, LSTM, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    if (network->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    PSLayer *out = network->layers[network->size - 1];
    out->flags |= FLAG_ONEHOT;
    PSLayer *layer = network->layers[1];
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
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
        PSNeuron *neuron = out->neurons[i];
        *neuron->bias = 0.0;
        for (w = 0; w < layer->size; w++) {
            neuron->weights[w] = lstm_out_weights[i][w];
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
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
    PSNeuralNetwork *network = PSCreateNetwork("GRU Test Network");
    if (network == NULL) {
        fprintf(stderr, "\nCould not create network!\n");
        return 0;
    }
    network->flags |= FLAG_ONEHOT;
    PSAddLayer(network, FullyConnected, RNN_INPUT_SIZE, NULL);
    PSAddLayer(network, GRU, RNN_HIDDEN_SIZE, NULL);
    PSAddLayer(network, SoftMax, RNN_INPUT_SIZE, NULL);
    if (network->size < 1) {
        fprintf(stderr, "\nCould not add all layers!\n");
        return 0;
    }
    PSLayer *out = network->layers[network->size - 1];
    out->flags |= FLAG_ONEHOT;
    PSLayer *layer = network->layers[1];
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
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
        PSNeuron *neuron = out->neurons[i];
        *neuron->bias = 0.0;
        for (w = 0; w < layer->size; w++) {
            neuron->weights[w] = lstm_out_weights[i][w];
        }
    }

    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
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

int NetworkBackpropTest(Test *test, char *model_file, char *data_file_prefix,
                        char *training_data_file, char *labels_data_file,
                        int training_data_len, int label_data_len,
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
    PSNeuralNetwork *network = PSCreateNetwork(model_name);
    testAssertNotNull(network, test);
    ok = PSLoadNetwork(network, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to load network from '%s'", path
    );
    if (!PSIsNetworkBuilt(network)) {
        ok = PSBuildNetwork(network);
        if (!ok) {
            fprintf(stderr, "\nFailed to build network!\n");
            goto final;
        }
    }
    ok = network->layers != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test,
        "network %s has no layers",network->name
    );
    if (expected_size > 0) {
        ok = network->size == expected_size;
        testAssertWithMessageOrGoto(
            ok, final, test,
            "network has %d layers, expected %d", network->size, expected_size
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
    int image_len = 0, label_len = 0;
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
    network->acceleration = acceleration;
    gradients = backprop(network, x, y, topts, NULL);
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for network %s (acceleration: %s)",
        network->name, acceleration_name
    );
    PSGradient **netgradients = gradients[0];
    ok = netgradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL",
        network->name, acceleration_name
    );
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        char testlabel[1024];
        int lsize = layer->size;
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
            int len = 0;
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
            int len = 0, lsize = layer->size;
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
            int len = 0, grad_len = PSGetLayerParametersCount(
                layer, PARAM_TYPE_BIAS
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
                layer, PARAM_TYPE_WEIGHT
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
    if (gradients != NULL && network != NULL)
        PSDeleteGradientsChain(gradients, network);
    PSDeleteNetwork(network);
    free(x);
    free(y);
    free(states);
    free(deltas);
    free(grads);
    return ok;
}

int testFullLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, PRETRAINED_FULL_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssert(loaded, test);
    testAssertEqual(network->size, 3, test);
    testAssertEqual(
        network->layers[0]->size, pretrained_mnist_layers_size[0], test
    );
    testAssertEqual(
        network->layers[1]->size, pretrained_mnist_layers_size[1], test
    );
    testAssertEqual(
        network->layers[2]->size, pretrained_mnist_layers_size[2], test
    );
    testAssertNotNull(network->layers[1]->biases, test);
    testAssertNotNull(network->layers[1]->weights, test);
    testAssertNotNull(network->layers[1]->weights[0], test);
    network->acceleration = PSGlobalAcceleration;
    PSFloat expected_biases[2][2] = {
        {-1.1618, -2.3288},
        {-6.0822, 0.8330}
    };
    PSFloat expected_weights[2][2] = {
        {-1.8497, -0.5419},
        {-1.2359, -4.677}
    };
    for(int l = 1; l < network->size; l++) {
        for (int i = 0; i < 2; i++) {
            PSFloat expected_bias = expected_biases[l - 1][i];
            PSFloat bias = getRoundedFloatDec(network->layers[l]->biases[i], 4);
            testAssertWithMessage(
                bias == expected_bias, test,
                "Layer[%d] Bias[%d] expected to be %g, got %g",
                l, i, expected_bias, bias
            );
            PSNeuron *n = network->layers[l]->neurons[i];
            testAssertNotNull(n->weights, test);
            PSFloat expected_w = expected_weights[l - 1][i];
            PSFloat w = getRoundedFloatDec(n->weights[0], 4);
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
    PSNeuralNetwork *network = getNetwork(test_case);
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSFloat *test_data = getTestData(test_case);
    PSForward(network, test_data);

    PSLayer *output = network->layers[network->size - 1];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSFloat accuracy = PSTest(network, test_data, testlen, NULL),
            expected = 95.0;
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == expected), test, "Accuracy %g != from expected (%g)",
        accuracy, expected
    );
    return 1;
}

int testFullBackprop(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient ***grads = backprop(network, x, y, NULL, NULL);
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
        PSLayer *layer = network->layers[lidx];
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
    PSDeleteGradientsChain(grads, network);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, network);
    return 0;
}

int testConvLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, CONVOLUTIONAL_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSLayer *layer = network->layers[1];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *test_data = getTestData(test_case);
    PSForward(network, test_data);
    PSLayer *output = network->layers[network->size - 1];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssert(network->acceleration == PSGlobalAcceleration, test);
    PSFloat *test_data = getTestData(test_case);
    int input_size = network->layers[0]->size;
    PSFloat *x = test_data;
    PSFloat *y = test_data + input_size;
    PSGradient ***grads = backprop(network, x, y, NULL, NULL);
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
        PSLayer *layer = network->layers[lidx];
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
    PSDeleteGradientsChain(grads, network);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, network);
    return 0;
}

int testConvAccuracy(TestCase *test_case, Test *test) {
    PSFloat *test_data = getTestData(test_case);
    PSNeuralNetwork *network = PSCreateNetwork("CNN Test Network");
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, CONVOLUTIONAL_TRAINED_NETWORK, path), test
    );
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSForward(network, test_data);
    PSFloat accuracy = PSTest(network, test_data, testlen, NULL),
            expected = 98.0;
    PSDeleteNetwork(network);
    accuracy = PSRound(accuracy * 100.0);
    testAssertWithMessage(
        (accuracy == 98.0), test,
        "Accuracy %g != from expected (%g)", accuracy, expected
    );
    return 1;
}

int testConvCIFAR(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    int ok = NetworkBackpropTest(test, CONVOLUTIONAL_CIFAR_NETWORK,
                                 "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                                 CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                                 2, 0, 8, PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = NetworkBackpropTest(test, CONVOLUTIONAL_CIFAR_NETWORK,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = NetworkBackpropTest(test, CONVOLUTIONAL_CIFAR_NETWORK,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = NetworkBackpropTest(test, CONVOLUTIONAL_CIFAR_NETWORK,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = NetworkBackpropTest(test, CONVOLUTIONAL_CIFAR_NETWORK,
                             "cifar", CIFAR_IMAGE_PATH, CIFAR_LABEL_PATH,
                              CIFAR_IMAGE_SIZE, 10, "CIFAR CNN", NULL,
                              2, 0, 8, PSAcceleration_None);
    return ok;
}

int testRNNLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path,RECURRENT_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    int i, j, w, rnn_size = 0;
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int exp_weight_types_count = 1, is_rnn_layer = (i == 1);
        uint64_t input_weight_count = 0, hidden_weight_count = 0;
        if (is_rnn_layer) {
            /* Recurrent Layer */
            rnn_size = layer->size;
            exp_weight_types_count = 2;
        }
        testAssertWithMessage(
            layer->weight_types_count == exp_weight_types_count, test,
            "Layer[%d] should have %d weight matrices, got: %d",
            i, exp_weight_types_count, layer->weight_types_count
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
    PSNeuralNetwork *network = getNetwork(test_case);
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    PSForward(network, rnn_inputs);
    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;

    PSLayer *output = network->layers[network->size - 1];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    if (!PSIsNetworkBuilt(network)) {
        if (!PSBuildNetwork(network)) {
            fprintf(stderr, "\nFailed to build network!\n");
            return 0;
        }
    }
    int i, j, w;

    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;
    PSTrainingOptions opts = {
        .bptt_truncate = 4
    };
    PSGradient ***grads =backprop(network, rnn_inputs, rnn_labels, &opts, NULL);
    testAssertNotNull(grads, test);
    PSGradient **gradients = grads[0];
    testAssertNotNull(gradients, test);
    int dsize = network->size - 1;
    for (i = 0; i < dsize; i++) {
        PSGradient *gradient = gradients[i];
        PSLayer *l = network->layers[i + 1];
        testAssertNotNull(l->weights, test);
        testAssertNotNull(l->weights[0], test);
        int input_size = gradient->weight_count;
        int input_ws = input_size / l->size;
        if (Recurrent == l->type) {
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
            if (Recurrent == l->type) {
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
    PSDeleteGradientsChain(grads, network);
    return 1;
on_fail:
    PSDeleteGradientsChain(grads, network);
    return 0;
}

int testRNNBackprop(TestCase *test_case, Test *test) {
    UNUSED(test_case);
    char *network_file = "resources/basic-rnn.psmodel";
    char *inputs_file = "resources/rnn-inputs.data";
    char *labels_file = "resources/rnn-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (TRAINING_EPOCH_AS_SEQUENCE | TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = NetworkBackpropTest(test, network_file,
                                 "rnn", inputs_file, labels_file,
                                 input_len, label_len, "RNN", &opts, 3, 0, 3,
                                 PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = NetworkBackpropTest(test, network_file,
                             "rnn", inputs_file, labels_file,
                              input_len, label_len, "RNN", &opts, 3, 0, 3,
                              PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = NetworkBackpropTest(test, network_file,
                             "rnn", inputs_file, labels_file,
                              input_len, label_len, "RNN", &opts, 3, 0, 3,
                              PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = NetworkBackpropTest(test, network_file,
                             "rnn", inputs_file, labels_file,
                              input_len, label_len, "RNN", &opts, 3, 0, 3,
                              PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = NetworkBackpropTest(test, network_file,
                             "rnn", inputs_file, labels_file,
                              input_len, label_len, "RNN", &opts, 3, 0, 3,
                              PSAcceleration_None);
    return ok;
}

int testRNNStep(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    /*int train_data_len = 1 + (RNN_TIMES * 2);*/
    if (!testRecurrentNetworkMode(network, ManyToMany, test)) return 0;
    PSResetNetworkStateSequences(network, 0, 0);
    PSFloat *training_data = getTestData(test_case);
    PSFloat **sequences = &training_data;
    int elements_count = (int) *training_data;

    int i, j, w;
    PSFloat loss = updateNetworkParameters(
        network, training_data, 1, elements_count,
        NULL, RNN_LEARNING_RATE, NULL, NULL, sequences
    );
    UNUSED(loss);
    for (i = 1; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
        int wsize = (int) PSGetLayerInputWeightsCount(layer, 1);
        if (layer->type == Recurrent) wsize += layer->size;
        for (j = 0; j < layer->size; j++) {
            PSNeuron *n = layer->neurons[j];
            for (w = 0; w < wsize; w++) {
                PSFloat *eweights, *lweights;
                int widx = w;
                if (i == 1) {
                    if (w < RNN_INPUT_SIZE) {
                        eweights = rnn_trained_inner_weights[j];
                        lweights = PSGetNeuronInputWeights(n);
                    } else {
                        widx -= RNN_INPUT_SIZE;
                        eweights = rnn_trained_recurrent_weights[j];
                        lweights = PSGetRecurrentNeuronHiddenWeights(n);
                    }
                } else {
                    eweights = rnn_trained_outer_weights[j];
                    lweights = PSGetNeuronInputWeights(n);
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
    PSNeuralNetwork *onehot_network = PSCreateNetwork("Onehot RNN");
    PSNeuralNetwork *standard_network = PSCreateNetwork("Standard RNN");
    PSNeuralNetwork *dummy_network = PSCreateNetwork("Dummy");
    PSFloat *onehot_data = NULL;
    PSFloat *standard_data = NULL;
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, RECURRENT_NETWORK, path), test);
    int loaded = PSLoadNetwork(onehot_network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    onehot_network->acceleration = PSGlobalAcceleration;
    int ok = PSLoadNetwork(standard_network, path);
    testAssertWithMessageOrGoto(loaded, final, test, "Failed to load %s", path);
    if (!PSIsNetworkBuilt(onehot_network)) {
        if (!PSBuildNetwork(onehot_network)) {
            fprintf(stderr, "\nFailed to build onehot network!\n");
            return 0;
        }
    }
    if (!PSIsNetworkBuilt(standard_network)) {
        if (!PSBuildNetwork(standard_network)) {
            fprintf(stderr, "\nFailed to build standard network!\n");
            return 0;
        }
    }
    standard_network->acceleration = PSGlobalAcceleration;
    PSRecurrentNetworkMode onehot_rnn_mode = onehot_network->rnn_mode;
    PSRecurrentNetworkMode std_rnn_mode = standard_network->rnn_mode;
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
    ok = onehot_network->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network is not OneHot!", onehot_network->name
    );
    ok = onehot_network->layers[0]->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[0] is not OneHot!",
        onehot_network->name
    );
    int last_layer = onehot_network->size - 1;
    ok = onehot_network->layers[last_layer]->flags & FLAG_ONEHOT;
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[%d] is not OneHot!",
        onehot_network->name, last_layer
    );
    int no_onehot = ~((unsigned) FLAG_ONEHOT);
    standard_network->flags &= no_onehot;
    standard_network->layers[0]->flags &= no_onehot;
    standard_network->layers[last_layer]->flags &= no_onehot;
    ok = !(standard_network->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network is OneHot!", standard_network->name
    );
    ok = !(standard_network->layers[0]->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[0] is OneHot!",
        standard_network->name
    );
    ok = !(standard_network->layers[last_layer]->flags & FLAG_ONEHOT);
    testAssertWithMessageOrGoto(
        ok, final, test, "%s network layer[%d] is OneHot!",
        standard_network->name, last_layer
    );
    int vector_size = onehot_network->layers[0]->onehot_vector_size;
    ok = (vector_size > 0);
    testAssertWithMessageOrGoto(
        vector_size > 0, final, test,
        "%s network layer[0] vector size is %d", vector_size
    );
    PSLayer *standard_input_layer =
        PSAddLayer(dummy_network, FullyConnected, vector_size, NULL);
    ok = (standard_input_layer != NULL);
    testAssertWithMessageOrGoto(
        standard_input_layer != NULL, final, test,
        "Failed to create standard input layer with size %d",
        vector_size
    );
    PSLayer *curlayer = standard_network->layers[0];
    standard_network->layers[0] = standard_input_layer;
    standard_input_layer->network = standard_network;
    standard_network->input_size = vector_size;
    standard_input_layer->flags |= FLAG_RECURRENT;
    curlayer->network = NULL;
    PSDeleteLayer(curlayer);
    dummy_network->size = 0;
    dummy_network->layers[0] = NULL;
    PSDeleteNetwork(dummy_network);
    dummy_network = NULL;
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
    if (!PSRebuildNetwork(standard_network)) {
        fprintf(stderr, "\nFailed to re-build standard network!\n");
        return 0;
    }
    PSFloat onehot_accuracy = PSTest(
        onehot_network, onehot_data, onehot_datalen, NULL
    );
    PSFloat std_accuracy = PSTest(
        standard_network, standard_data, standard_datalen, NULL
    );
    ok = (onehot_network->status != STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Network %s: error during validation",
        onehot_network->name
    );
    ok = (standard_network->status != STATUS_ERROR);
    testAssertWithMessageOrGoto(
        ok, final, test, "Network %s: error during validation",
        standard_network->name
    );
    ok = (onehot_accuracy == std_accuracy);
    testAssertWithMessageOrGoto(
        ok, final, test, "Onehot accuracy != Non-onehot accuracy: %g != %g",
        onehot_accuracy, std_accuracy
    );
final:
    if (onehot_network != NULL) PSDeleteNetwork(onehot_network);
    if (standard_network != NULL) PSDeleteNetwork(standard_network);
    if (dummy_network != NULL) PSDeleteNetwork(dummy_network);
    if (onehot_data != NULL) free(onehot_data);
    if (standard_data != NULL) free(standard_data);
    return ok;
}

int testLSTMTrain(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    /*int train_data_len = 2 + (LSTM_TIMES * 2);
    UNUSED(train_data_len);*/
    PSFloat *training_data = getTestData(test_case);

    PSTrainingOptions options = {
        .epochs = LSTM_EPOCHS,
        .batch_size = LSTM_BATCHES,
        .learning_rate = LSTM_LEARNING_RATE,
        .flags = TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0,
        .bptt_truncate = 4
    };
    PSTrain(network, training_data, 8, NULL, 0, &options);

    PSLayer *layer = network->layers[1];
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
    PSLayer *out = network->layers[network->size - 1];

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
    char *network_file = "resources/basic-lstm.psmodel";
    char *inputs_file = "resources/gru-inputs.data";
    char *labels_file = "resources/gru-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (TRAINING_EPOCH_AS_SEQUENCE | TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = NetworkBackpropTest(test, network_file,
                                 "lstm", inputs_file, labels_file,
                                 input_len, label_len, "LSTM", &opts, 0, 3, 3,
                                 PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = NetworkBackpropTest(test, network_file,
                             "lstm", inputs_file, labels_file,
                              input_len, label_len, "LSTM", &opts, 0, 2, 3,
                              PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = NetworkBackpropTest(test, network_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 3, 3,
                             PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = NetworkBackpropTest(test, network_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 3, 3,
                             PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = NetworkBackpropTest(test, network_file,
                             "lstm", inputs_file, labels_file,
                             input_len, label_len, "LSTM", &opts, 0, 2, 3,
                             PSAcceleration_None);
    return ok;
}

int testGRUTrain(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSFloat *training_data = getTestData(test_case);

    PSTrainingOptions options = {
        .epochs = LSTM_EPOCHS,
        .batch_size = LSTM_BATCHES,
        .learning_rate = 1.5,//LSTM_LEARNING_RATE,
        .flags = TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0,
        .bptt_truncate = 4
    };
    PSTrain(network, training_data, 8, NULL, 0, &options);

    PSLayer *layer = network->layers[1];
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
    PSLayer *out = network->layers[network->size - 1];

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
    char *network_file = "resources/basic-gru.psmodel";
    char *inputs_file = "resources/gru-inputs.data";
    char *labels_file = "resources/gru-labels.data";
    int input_len = 26;
    int label_len = 25;
    int train_flags = (TRAINING_EPOCH_AS_SEQUENCE | TRAINING_NO_SHUFFLE);
    PSTrainingOptions opts = {
        .bptt_truncate = 0,
        .flags = train_flags,
        .clip = 5
    };
    int ok = NetworkBackpropTest(test, network_file,
                                 "gru", inputs_file, labels_file,
                                 input_len, label_len, "GRU", &opts, 3, 0, 3,
                                 PSGlobalAcceleration);
    if (!ok) return 0;
#ifdef HAS_BLAS
    ok = NetworkBackpropTest(test, network_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_BLAS);
    if (!ok) return 0;
#endif
#if defined(__APPLE__) && defined(HAS_ACCELERATE_FRAMEWORK)
    ok = NetworkBackpropTest(test, network_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_ACF);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    ok = NetworkBackpropTest(test, network_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_AVX);
    if (!ok) return 0;
#endif
    ok = NetworkBackpropTest(test, network_file,
                             "gru", inputs_file, labels_file,
                              input_len, label_len, "GRU", &opts, 3, 0, 3,
                              PSAcceleration_None);
    return ok;
}

int testNormalizationLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, NORMALIZATION_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    PSLayer *normlayer = network->layers[1];
    PSLayer *softmax = network->layers[2];
    testAssertNotNull(normlayer, test);
    testAssert(normlayer->type == Normalization, test);
    testAssert(normlayer->size == 4, test);
    testAssertNotNull(softmax, test);
    testAssert(softmax->type == SoftMax, test);
    testAssert(softmax->size == 2, test);
    testAssert(!(normlayer->flags & FLAG_NON_TRAINABLE), test);
    if (!PSIsNetworkBuilt(network)) {
        int built = PSBuildNetwork(network);
        testAssert(built, test);
    }
    return 1;
}

int testNormalizationForward(TestCase *test_case, Test *test) {
    static PSFloat data[] = {90.3487, 198.1253, 18.3623, 162.5884, 0, 0};
    static PSFloat normalized[] = {-0.3909, 1.1689, -1.4326, 0.6546};
    PSNeuralNetwork *network = getNetwork(test_case);
    PSForward(network, data);
    PSLayer *normlayer = network->layers[1];
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
    PSNeuralNetwork *network = PSCreateNetwork("Normalization Backprop");
    testAssertNotNull(network, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, NORMALIZATION_NETWORK_BP, path),test);
    ok = PSLoadNetwork(network, path);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not load network from '%s'", path
    );
    network->acceleration = PSGlobalAcceleration;
    ok = PSBuildNetwork(network);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not build network '%s'", network->name
    );
    ok = PSForward(network, x);
    testAssertWithMessageOrGoto(
        ok, final, test, "Forward failed to network %s", network->name
    );
    PSLayer *outlayer = network->layers[network->size - 1];
    testAssertNotNull(outlayer, test);
    PSFloat *outstates = PSGetStates(outlayer, 0);
    testAssertNotNull(outstates, test);
    PSLayer *normlayer = network->layers[2];
    testAssertNotNull(normlayer, test);
    ok = compareArrays(outstates, expected_softmax, outlayer->size, test,
                       "output states", 4, 0);
    if (!ok) goto final;
    grads = backprop(network, x, y, NULL, NULL);
    ok = grads != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for network %s", network->name
    );
    PSGradient **gradients = grads[0];
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL", network->name
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
    if (network != NULL) {
        if (grads != NULL) PSDeleteGradientsChain(grads, network);
        PSDeleteNetwork(network);
    }
    return ok;
}

int testDropoutLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, DROPOUT_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    PSLayer *dropout_layer = network->layers[2];
    PSLayer *softmax = network->layers[network->size - 1];
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
    if (!PSIsNetworkBuilt(network)) {
        int built = PSBuildNetwork(network);
        testAssert(built, test);
    }
    return 1;
}

int testDropoutForward(TestCase *test_case, Test *test) {
    static PSFloat data[] = {90.3487, 198.1253};
    int ok = 1;
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    ok = PSForward(network, data);
    testAssert(ok, test);
    PSLayer *dropout_layer = network->layers[2];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSLayer *dropout_layer = network->layers[2];
    PSLayer *prev = PSGetPreviousLayer(dropout_layer);
    testAssertNotNull(prev, test);
    int old_status = network->status;
    network->status = STATUS_TRAINING;
    grads = backprop(network, x, y, NULL, NULL);
    ok = grads != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Backprop failed for network %s", network->name
    );
    PSGradient **gradients = grads[0];
    ok = gradients != NULL;
    testAssertWithMessageOrGoto(
        ok, final, test, "Gradients[0] is NULL", network->name
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
    network->status = old_status;
    if (grads != NULL) PSDeleteGradientsChain(grads, network);
    return ok;
}

int testConcatOperatorLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_CONCAT_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = network->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSConcatenateOperator, test);
    int expected_size = network->layers[1]->size + network->layers[2]->size,
        providers_count;
    testAssert(expected_size == oplayer->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == network->layers[1], test);
    testAssert(providers[1] == network->layers[2], test);
    if (!PSIsNetworkBuilt(network)) {
        int built = PSBuildNetwork(network);
        testAssert(built, test);
    }
    return 1;
}

int testConcatOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(network, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[network->output_size];
    for (uint32_t i = 0; i < network->output_size; i++)
        y[i] = PSGaussianRandom(0, 1);
    PSGradient ***grads = backprop(network, x, y, NULL, NULL);
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
                       network->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = network->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(op_layer->delta, l1_delta, l1->size, test,
                       NULL, 0, 4);
    if (!ok) return 0;
    ok = compareArrays(op_layer->delta + l1->size, l2->delta, l2->size, test,
                       NULL, 0, 4);
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, network);
    PSMatrixDelete(l1_delta);
    return ok;
}

int testAddOperatorLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_ADD_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = network->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSAddOperator, test);
    int expected_size = network->layers[1]->size, providers_count;
    testAssert(expected_size == oplayer->size, test);
    testAssert(expected_size == network->layers[2]->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == network->layers[1], test);
    testAssert(providers[1] == network->layers[2], test);
    if (!PSIsNetworkBuilt(network)) {
        int built = PSBuildNetwork(network);
        testAssert(built, test);
    }
    return 1;
}

int testAddOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(network, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[network->output_size];
    for (uint32_t i = 0; i < network->output_size; i++)
        y[i] = PSGaussianRandom(0, 1);
    PSGradient ***grads = backprop(network, x, y, NULL, NULL);
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
                       network->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = network->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(l1_delta, l2->delta, l1->size, test,
                       NULL, 0, 4);
    if (!ok) return 0;
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, network);
    PSMatrixDelete(l1_delta);
    return ok;
}

int testMulOperatorLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    char path[PATH_MAX] = {0};
    testAssert(joinPath(executable_path, OP_MUL_NETWORK, path), test);
    int loaded = PSLoadNetwork(network, path);
    testAssertWithMessage(loaded, test, "Failed to load %s", path);
    network->acceleration = PSGlobalAcceleration;
    PSLayer *oplayer = network->layers[3];
    testAssertNotNull(oplayer, test);
    testAssert(oplayer->type == OperatorLayer, test);
    PSOperatorType op = PSGetOperatorLayerType(oplayer);
    testAssert(op == PSMultiplyOperator, test);
    int expected_size = network->layers[1]->size, providers_count;
    testAssert(expected_size == oplayer->size, test);
    testAssert(expected_size == network->layers[2]->size, test);
    PSLayer **providers = PSGetOperatorLayerProviders(oplayer,&providers_count);
    testAssertNotNull(providers, test);
    testAssert(providers_count == 2, test);
    testAssert(providers[0] == network->layers[1], test);
    testAssert(providers[1] == network->layers[2], test);
    if (!PSIsNetworkBuilt(network)) {
        int built = PSBuildNetwork(network);
        testAssert(built, test);
    }
    return 1;
}

int testMulOperatorForward(TestCase *test_case, Test *test) {
    int ok = 1;
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSFloat inputs[] = {0.932902753, 1.48698103, -0.75523293};
    ok = PSForward(network, inputs);
    testAssert(ok, test);
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    PSMatrix l1_delta = NULL;
    PSLayer *op_layer = network->layers[3], *l1 = network->layers[1],
            *l2 = network->layers[2];
    PSFloat x[] = {0.932902753, 1.48698103, -0.75523293};
    PSFloat y[] = {-1.19145763, -0.723464847, 0.460539848};
    PSFloat l1_expct[] = {-0.423889, 0.112024, 4.51742, -1.43311, -1.15413};
    PSFloat l2_expct[] = {-0.0644108, -0.0631921, 1.18236, -2.13912, -1.77627};
    PSGradient ***grads = backprop(network, x, y, NULL, NULL);
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
                       network->acceleration);
    testAssertWithMessageOrGoto(
        ok, final, test, "could not compute delta propagated from layer[%d] "
        "to layer[%d]", 2, 1
    );
    PSMathOpts opts = {.acceleration = network->acceleration};
    PSSubtractVectors(l1->delta, l1_delta, l1_delta, l1->size, &opts);
    ok = compareArrays(l1_delta, l1_expct, l1->size, test, "L1 DELTA: ", 0, 4);
    if (!ok) goto final;
    ok = compareArrays(l2->delta, l2_expct, l2->size, test, "L2 DELTA", 0, 4);
final:
    if (grads != NULL) PSDeleteGradientsChain(grads, network);
    PSMatrixDelete(l1_delta);
    return ok;
}

int encoderDecoderSetup(TestCase *test_case) {
    PSNeuralNetwork *network = PSCreateNetwork("Encoder-Decoder");
    if (network == NULL) {
        PSErr(NULL, "\nCould not create network!");
        return 0;
    }
    test_case->data = malloc(2 * sizeof(void*));
    if (test_case->data == NULL) {
        fprintf(stderr, "\nCould not allocate memory!\n");
        return 0;
    }
    test_case->data[0] = network;
    test_case->data[1] = NULL;
    return 1;
}

int encoderDecoderTeardown(TestCase *test_case) {
    PSNeuralNetwork *network = getNetwork(test_case);
    if (network) PSDeleteNetwork(network);
    return 1;
}

int testEncodedDecoderLoad(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    char path[PATH_MAX] = {0};
    testAssert(
        joinPath(executable_path, ENCDEC_BASIC_NETWORK, path), test
    );
    int ok = PSLoadNetwork(network, path);
    testAssert(ok, test);
    testAssert(PSGetNetworkChainLength(network) == 2, test);
    return ok;
}
int testEncodedDecoderSave(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    testAssert(network->size > 0, test);
    testAssert(PSGetNetworkChainLength(network) == 2, test);
    const char *fpath = "/tmp/psyc-enc-dec-test.psmodel";
    int ok = PSSaveNetwork(network, fpath);
    testAssert(ok, test);
    PSNeuralNetwork *loaded = PSCreateNetwork(NULL);
    testAssertNotNull(loaded, test);
    ok = PSLoadNetwork(loaded, fpath);
    testAssertWithMessageOrGoto(
        ok, final, test, "Failed to load network from '%s'", fpath
    );
    ok = compareNetworkChain(network, loaded, test);
final:
    if (loaded != NULL) PSDeleteNetwork(loaded);
    return ok;
}

int testEncodedDecoderClone(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    testAssert(network->size > 0, test);
    testAssert(PSGetNetworkChainLength(network) == 2, test);
    PSNeuralNetwork *clone = PSCloneNetwork(network, 0);
    testAssertNotNull(clone, test);
    int ok = compareNetworkChain(network, clone, test);
final:
    if (clone != NULL) PSDeleteNetwork(clone);
    return ok;
}

int testEncodedDecoderPredict(TestCase *test_case, Test *test) {
    int ok = 1;
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    testAssert(network->size > 0, test);
    testAssert(PSGetNetworkChainLength(network) == 2, test);
    PSFloat x[] = {2, 2, 1};
    PSFloat expected[2][4] = {
          {0.00577807,0.0132289,0.978728,0.00226505},
          {0.998611,0.000101474,0.00128536,2.34558e-06}
    };
    PSNeuralNetwork *decoder = network->next;
    testAssertNotNull(decoder, test);
    decoder->sequence_settings.end = 0;
    ok = PSAutoregression(network, x, 0, NULL);
    testAssert(ok, test);
    PSLayer *out = PSGetOutputLayer(network);
    testAssertNotNull(out, test);
    testAssert(out->network == decoder, test);
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
static int beforeDecoderForward(PSNeuralNetwork *decoder,
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
    PSNeuralNetworkLink *link = decoder->previous_network_link;
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

int beforeEncoderBackprop(PSNeuralNetwork *encoder, PSFloat *y,
                          PSTrainingOptions *opts, PSGradient **gradients)
{
    UNUSED(y);
    UNUSED(opts);
    UNUSED(gradients);
    assert(encDecBackpropTest != NULL);
    Test *test = encDecBackpropTest;
    int ok = 1;
    PSNeuralNetwork *decoder = encoder->next;
    testAssertNotNull(decoder, test);
    PSNeuralNetworkLink *link = decoder->previous_network_link;
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
    PSNeuralNetwork *network = getNetwork(test_case);
    testAssertNotNull(network, test);
    testAssert(network->size > 0, test);
    testAssert(PSGetNetworkChainLength(network) == 2, test);
    PSNeuralNetwork *decoder = network->next;
    testAssertNotNull(decoder, test);
    decoder->sequence_settings.end = -1;
    PSLayer *out = PSGetOutputLayer(network);
    testAssertNotNull(out, test);
    testAssert(out->flags & FLAG_ONEHOT, test);
    PSTrainingOptions opts = {
        .flags = TRAINING_FLAG_TEACHER_FORCING | TRAINING_FLAG_SEQ2SEQ,
        .epochs = 1,
        .learning_rate = 0.3,
        .batch_size = 1,
        .bptt_truncate = 0,
    };
    network->beforeBackprop = beforeEncoderBackprop;
    decoder->beforeForward = beforeDecoderForward;
    PSFloat *seq[] = {NULL};
    seq[0] = training_data;
    PSFloat loss = updateNetworkParameters(
        network, training_data, 1, elements_count,
        &opts, 0.3, NULL, NULL, seq
    );
    UNUSED(loss);
    ok = (network->status != STATUS_ERROR);
    if (ok) ok = (decoder->status != STATUS_ERROR);
    encDecBackpropTest = NULL;
    return ok;
}

int testGenericClone(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    PSNeuralNetwork *clone = PSCloneNetwork(network, 0);
    testAssertNotNull(clone, test);
    int ok = compareNetworks(network, clone, test);
    PSDeleteNetwork(clone);
    return ok;
}

int testGenericSave(TestCase *test_case, Test *test) {
    PSNeuralNetwork *network = getNetwork(test_case);
    assert(network->size > 0);
    char tmpfile[255];
    getTmpFileName("tests-save-nn", ".psmodel", tmpfile);
    int ok = PSSaveNetwork(network, tmpfile);
    testAssertWithMessage(ok, test, "Could not save network %s", network->name);
    PSNeuralNetwork *clone = PSCreateNetwork("Clone Test Network");
    testAssertNotNull(clone, test);
    ok = PSLoadNetwork(clone, tmpfile);
    testAssertWithMessageOrGoto(
        ok, final, test, "Could not load network from %s",tmpfile
    );
    ok = compareNetworks(network, clone, test);
    remove(tmpfile);
final:
    PSDeleteNetwork(clone);
    return ok;
}

int compareNetworks(PSNeuralNetwork *network, PSNeuralNetwork *clone,
                    Test* test)
{
    int ok = 1, i, k;
    ok = network->size == clone->size;
    testAssertWithMessage(
        (network->size == clone->size), test,
        "Source size %d != Clone size %d",
        network->size, clone->size
    );
    int recurrent_source = PSIsRecurrent(network),
        recurrent_clone = PSIsRecurrent(clone);
    testAssertWithMessage(
        (recurrent_source == recurrent_clone), test,
        "Source recurrent: %d, Clone recurrent: %d",
        recurrent_source, recurrent_clone
    );
    if (recurrent_source) {
        PSLayer *src_first_recurrent_layer = PSGetFirstRecurrentLayer(network),
                *src_last_recurrent_layer = PSGetLastRecurrentLayer(network),
                *clone_first_recurrent_layer = PSGetFirstRecurrentLayer(clone),
                *clone_last_recurrent_layer = PSGetLastRecurrentLayer(clone);
        PSRecurrentNetworkMode srcmode = network->rnn_mode,
                               clonemode = clone->rnn_mode;
        int src_max_steps = network->sequence_settings.max_length,
            clone_max_steps = clone->sequence_settings.max_length,
            src_eos = network->sequence_settings.end,
            clone_eos = clone->sequence_settings.end;
        testAssertWithMessage(
            (srcmode == clonemode), test,
            "Recurrent source mode: '%s', Recurrent clone mode: '%s'",
            PSGetRecurrentModeLabel(srcmode),
            PSGetRecurrentModeLabel(clonemode)
        );
        testAssertWithMessage(
            (src_max_steps == clone_max_steps), test,
             "network->sequence_settings.max_length != "
             "clone->sequence_settings.max_length: %d != %d",
             src_max_steps, clone_max_steps
        );
        testAssertWithMessage(
            (src_eos == clone_eos), test,
             "network->sequence_settings.end != "
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
                "network first recurrent layer index != "
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
                "network last recurrent layer index != "
                "clone last recurrent layer index: %d != %d",
                src_idx, cln_idx
            );
        }
    }
    testAssertWithMessage(
        network->flags == clone->flags, test,
        "Source flags %d != Clone flags %d",
        network->flags, clone->flags
    );
    for (i = 0; i < network->size; i++) {
        PSLayer *orig_l = network->layers[i];
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
        }
        if (OperatorLayer == orig_l->type) {
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
                testAssertNotNull(orig_prv->network, test);
                testAssertNotNull(clone_prv->network, test);
                testAssertWithMessage(
                    orig_prv->network->index == clone_prv->network->index,
                    test, "Layer[%d]: Source provider[%d] network[%d] != "
                    "Clone provider network[%d]", i, j,
                    orig_prv->network->index, clone_prv->network->index
                );
                testAssertWithMessage(
                    orig_prv->index == clone_prv->index,
                    test, "Layer[%d]: Source provider[%d] index[%d] != "
                    "Clone provider index[%d]", i, j, orig_prv->index,
                    clone_prv->index
                );
            }
        }
        if (i == 0) continue;
        if (otype == Pooling) continue;
        testAssertWithMessage(
            (orig_l->weight_types_count == clone_l->weight_types_count), test,
            "Layer[%d]: Source weight_types_count %d != Clone %d",
            orig_l->weight_types_count, clone_l->weight_types_count
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
            int bias_count = PSGetLayerParametersCount(orig_l, PARAM_TYPE_BIAS);
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
            for (k = 0; k < orig_l->weight_types_count; k++) {
                PSMatrix o_weights = orig_l->weights[k];
                PSMatrix c_weights = clone_l->weights[k];
                testAssertNotNull(o_weights, test);
                testAssertNotNull(c_weights, test);
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

int compareNetworkChain(PSNeuralNetwork *network, PSNeuralNetwork *clone,
                        Test* test)
{
    int ok = 1;
    PSNeuralNetwork *n1 = network, *n2 = clone;
    while (n1 != NULL && n2 != NULL) {
        ok = compareNetworks(n1, n2, test);
        if (!ok) break;
        n1 = n1->next;
        n2 = n2->next;
    }
    testAssertNull(n1, test);
    testAssertNull(n2, test);
    return ok;
}

static int testRecurrentNetworkMode(PSNeuralNetwork *network,
                                    PSRecurrentNetworkMode mode,
                                    Test *test)
{
    testAssertNotNull(network, test);
    if (network->size == 0) return 1;
    PSRecurrentNetworkMode network_rnn_mode = network->rnn_mode;
    testAssertWithMessage(
        network_rnn_mode == mode, test,
        "Recurrent network mode %s != expected %s",
        PSGetRecurrentModeLabel(network_rnn_mode),
        PSGetRecurrentModeLabel(mode)
    );
    PSLayer *first_recurrent = PSGetFirstRecurrentLayer(network),
            *last_recurrent = PSGetLastRecurrentLayer(network),
            *input_layer = network->layers[0],
            *output_layer = network->layers[network->size - 1];
    if (mode == NonRecurrent) {
        testAssertWithMessage(
            !PSIsRecurrent(network), test,
            "Network is recurrent despite mode is %s",
            PSGetRecurrentModeLabel(mode)
        );
    } else {
        testAssertWithMessage(
            PSIsRecurrent(network), test,
            "Network is not recurrent despite mode is %s",
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
    for (int i = 0; i < network->size; i++) {
        PSLayer *layer = network->layers[i];
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
        PSMatrixDelete(matrix);
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
    if (matrix) PSMatrixDelete(matrix);
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
    PSSumVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSumVectors(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSumVectors(x, y, res, 6, &opts);
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
    PSSumVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "Accelerate Framework", 0, 0);
    if (!ok) return 0;
#endif
#ifdef USE_AVX
    opts.acceleration = PSAcceleration_AVX;
    PSSumVectorScalar(x, y, res, 6, &opts);
    ok = compareArrays(res, cmp_res, 6, test, "AVX", 0, 0);
    if (!ok) return 0;
#endif
    opts.acceleration = PSAcceleration_None;
    PSSumVectorScalar(x, y, res, 6, &opts);
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
        PSMatrixDelete(src);
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
    if (src != NULL) PSMatrixDelete(src);
    if (dst != NULL) PSMatrixDelete(dst);
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
    if (src != NULL) PSMatrixDelete(src);
    if (dst != NULL) PSMatrixDelete(dst);
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
        PSMatrixDelete(m3d);
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
    PSMatrixDelete(m2d);
    PSMatrixDelete(m3d);
    return res;
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
    if (src != NULL) PSMatrixDelete(src);
    if (expanded != NULL) PSMatrixDelete(expanded);
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

    PSMatrixDelete(result);
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

    PSMatrixDelete(result);
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

    PSMatrixDelete(result);
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

    PSMatrixDelete(result);
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
    PSMatrixDelete(a);
    PSMatrixDelete(b);
    PSMatrixDelete(c);
    PSMatrixDelete(avec);
    PSMatrixDelete(result);
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
    PSMatrixDelete(a);
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
    PSMatrixDelete(b);
    PSMatrixDelete(result);
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

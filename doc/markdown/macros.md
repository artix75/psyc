# PsyC Documentation - 0.9.5
## Macros

### DBL_DECIMAL_DIG

In: types.h, line: 29

```c
#define DBL_DECIMAL_DIG (DBL_DIG + 2)
```




### FLT_DECIMAL_DIG

In: types.h, line: 25

```c
#define FLT_DECIMAL_DIG (FLT_DIG + 2)
```




### PS_ACCURACY_DATASIZE_AUTO

In: psyc.h, line: 78

```c
#define PS_ACCURACY_DATASIZE_AUTO -1
```




### PS_ACTION_ABORT

In: psyc.h, line: 64

```c
#define PS_ACTION_ABORT 5
```




### PS_ACTION_NONE

In: psyc.h, line: 62

```c
#define PS_ACTION_NONE 0
```




### PS_ACTION_PAUSE

In: psyc.h, line: 63

```c
#define PS_ACTION_PAUSE 4
```




### PS_BITMAP_OP_AND

In: utils.h, line: 56

```c
#define PS_BITMAP_OP_AND 1
```




### PS_BITMAP_OP_OR

In: utils.h, line: 57

```c
#define PS_BITMAP_OP_OR 2
```




### PS_BITMAP_OP_XOR

In: utils.h, line: 58

```c
#define PS_BITMAP_OP_XOR 3
```




### PS_CIFAR_IMAGE_SIZE

In: dataset.h, line: 253

```c
#define PS_CIFAR_IMAGE_SIZE (32 * 32 * 3)
```




### PS_DATA_ALWAYS_ALLOC

In: dataset.h, line: 34

```c
#define PS_DATA_ALWAYS_ALLOC (1 << 16)
```




### PS_DATA_EVENLY_SPREAD

In: dataset.h, line: 31

```c
#define PS_DATA_EVENLY_SPREAD (1 << 1)
```




### PS_DATA_SEQ2SEQ

In: dataset.h, line: 33

```c
#define PS_DATA_SEQ2SEQ PS_TRAINING_FLAG_SEQ2SEQ
```




### PS_DATA_SEQUENCES

In: dataset.h, line: 32

```c
#define PS_DATA_SEQUENCES PS_FLAG_USE_SEQUENCES
```




### PS_DATA_SHUFFLE

In: dataset.h, line: 30

```c
#define PS_DATA_SHUFFLE (1 << 0)
```




### PS_DATA_TYPE_TEST

In: dataset.h, line: 27

```c
#define PS_DATA_TYPE_TEST 1
```




### PS_DATA_TYPE_TRAINING

In: dataset.h, line: 26

```c
#define PS_DATA_TYPE_TRAINING 0
```




### PS_DEBUG_PHASE_UPDATE_GRADS

In: debug.h, line: 25

```c
#define PS_DEBUG_PHASE_UPDATE_GRADS 1
```




### PS_DEBUG_PHASE_UPDATE_WEIGHTS

In: debug.h, line: 26

```c
#define PS_DEBUG_PHASE_UPDATE_WEIGHTS 2
```




### PS_DEFAULT_BETA1

In: psyc.h, line: 38

```c
#define PS_DEFAULT_BETA1 0.9
```




### PS_DEFAULT_BETA2

In: psyc.h, line: 39

```c
#define PS_DEFAULT_BETA2 0.999
```




### PS_DEFAULT_END_TOKEN

In: dataset.h, line: 103

```c
#define PS_DEFAULT_END_TOKEN "<end>"
```




### PS_DEFAULT_EOS_INDEX

In: psyc.h, line: 51

```c
#define PS_DEFAULT_EOS_INDEX -1
```




### PS_DEFAULT_EPS

In: psyc.h, line: 42

```c
#ifdef PS_DOUBLE_PRECISION
#define PS_DEFAULT_EPS 1e-8
#else
#define PS_DEFAULT_EPS 1e-7
#endif
```




### PS_DEFAULT_MAX_VOCAB_SIZE

In: dataset.h, line: 100

```c
#define PS_DEFAULT_MAX_VOCAB_SIZE 15000
```




### PS_DEFAULT_PARSER_CAPACITY

In: dataset.h, line: 99

```c
#define PS_DEFAULT_PARSER_CAPACITY 50
```




### PS_DEFAULT_RECURRENT_MODE

In: psyc.h, line: 49

```c
#define PS_DEFAULT_RECURRENT_MODE ManyToMany
```




### PS_DEFAULT_RHO

In: psyc.h, line: 37

```c
#define PS_DEFAULT_RHO 0.95
```




### PS_DEFAULT_START_TOKEN

In: dataset.h, line: 102

```c
#define PS_DEFAULT_START_TOKEN "<start>"
```




### PS_DEFAULT_TOKEN_SEPARATOR

In: dataset.h, line: 38

```c
#define PS_DEFAULT_TOKEN_SEPARATOR " ,.:
```




### PS_DEFAULT_UNKNOWN_TOKEN

In: dataset.h, line: 101

```c
#define PS_DEFAULT_UNKNOWN_TOKEN "<unknown>"
```




### PS_EMBED_PRETRAIN_DEFAULT_EPOCHS

In: embedding.h, line: 23

```c
#define PS_EMBED_PRETRAIN_DEFAULT_EPOCHS 50
```




### PS_EMBED_PRETRAIN_DEFAULT_LEARN_RATE

In: embedding.h, line: 24

```c
#define PS_EMBED_PRETRAIN_DEFAULT_LEARN_RATE 0.1
```




### PS_FLAG_ACCEL_DISABLED

In: psyc.h, line: 86

```c
#define PS_FLAG_ACCEL_DISABLED (1 << 2) /* Formerly FLAG_AVX_DISABLED (< v0.4):
```




### PS_FLAG_AUTOREGRESSION

In: psyc.h, line: 94

```c
#define PS_FLAG_AUTOREGRESSION (1 << 7)
```




### PS_FLAG_LOG_COLORS

In: config.h, line: 25

```c
#ifndef FLAG_LOG_COLORS
#define PS_FLAG_LOG_COLORS (1 << 0)
#ifndef PS_FLAG_LOG_COLORS
#define PS_FLAG_LOG_COLORS (1 << 0)
#endif
```




### PS_FLAG_NO_BIAS

In: psyc.h, line: 90

```c
#define PS_FLAG_NO_BIAS (1 << 3)
```




### PS_FLAG_NON_TRAINABLE

In: psyc.h, line: 92

```c
#define PS_FLAG_NON_TRAINABLE (1 << 5)
```




### PS_FLAG_NONE

In: psyc.h, line: 83

```c
#define PS_FLAG_NONE 0
```




### PS_FLAG_ONEHOT

In: psyc.h, line: 85

```c
#define PS_FLAG_ONEHOT (1 << 1)
```




### PS_FLAG_PRETRAINER

In: psyc.h, line: 91

```c
#define PS_FLAG_PRETRAINER (1 << 4)
```




### PS_FLAG_RANDREGRESSION

In: psyc.h, line: 95

```c
#define PS_FLAG_RANDREGRESSION (1 << 8)
```




### PS_FLAG_RECURRENT

In: psyc.h, line: 84

```c
#define PS_FLAG_RECURRENT (1 << 0)
```




### PS_FLAG_SELF_ATTENTION

In: psyc.h, line: 96

```c
#define PS_FLAG_SELF_ATTENTION (1 << 16)
```




### PS_FLAG_USE_SEQUENCES

In: psyc.h, line: 93

```c
#define PS_FLAG_USE_SEQUENCES (1 << 6)
```




### PS_GRU_CANDIDATE_IDX

In: gru.h, line: 23

```c
#define PS_GRU_CANDIDATE_IDX 0
```




### PS_GRU_RESET_IDX

In: gru.h, line: 25

```c
#define PS_GRU_RESET_IDX 2
```




### PS_GRU_UPDATE_IDX

In: gru.h, line: 24

```c
#define PS_GRU_UPDATE_IDX 1
```




### PS_INIT_MODE_AUTO

In: psyc.h, line: 69

```c
#define PS_INIT_MODE_AUTO 0
```




### PS_INIT_MODE_RAND

In: psyc.h, line: 70

```c
#define PS_INIT_MODE_RAND 1
```




### PS_INIT_MODE_VALUE

In: psyc.h, line: 72

```c
#define PS_INIT_MODE_VALUE 3
```




### PS_INIT_MODE_ZERO

In: psyc.h, line: 71

```c
#define PS_INIT_MODE_ZERO 2
```




### PS_INVALID_TOKEN_ID

In: dataset.h, line: 40

```c
#define PS_INVALID_TOKEN_ID -1
```




### PS_IO_BINARY_MODE

In: psyc.h, line: 114

```c
#define PS_IO_BINARY_MODE (1 << 1)
```




### PS_IO_MAX_TOKEN_SIZE

In: dataset.h, line: 43

```c
#define PS_IO_MAX_TOKEN_SIZE 1024
```




### PS_IO_SAVE_DEFINITION

In: psyc.h, line: 113

```c
#define PS_IO_SAVE_DEFINITION (1 << 0)
```




### PS_KEYS_PROJ_IDX

In: attention.h, line: 30

```c
#define PS_KEYS_PROJ_IDX 1
```




### PS_KEYS_PROJECTION

In: attention.h, line: 24

```c
#define PS_KEYS_PROJECTION (1 << 1)
```




### PS_LAYER_TYPES

In: psyc.h, line: 35

```c
#define PS_LAYER_TYPES 14
```




### PS_LINE_CLEAR

In: log.h, line: 111

```c
#define PS_LINE_CLEAR (1 << 4)
```




### PS_LINE_CLEAR_ALL

In: log.h, line: 115

```c
#define PS_LINE_CLEAR_ALL 2
```




### PS_LINE_CLEAR_FROM_CURSOR

In: log.h, line: 113

```c
#define PS_LINE_CLEAR_FROM_CURSOR 0
```




### PS_LINE_CLEAR_TO_CURSOR

In: log.h, line: 114

```c
#define PS_LINE_CLEAR_TO_CURSOR 1
```




### PS_LINE_FILL

In: log.h, line: 108

```c
#define PS_LINE_FILL (1 << 1)
```




### PS_LINE_OVERWRITE

In: log.h, line: 109

```c
#define PS_LINE_OVERWRITE (1 << 2)
```




### PS_LINE_PLAIN_ASCII

In: log.h, line: 110

```c
#define PS_LINE_PLAIN_ASCII (1 << 3)
```




### PS_LSTM_CANDIDATE_IDX

In: lstm.h, line: 23

```c
#define PS_LSTM_CANDIDATE_IDX 0
```




### PS_LSTM_FORGET_IDX

In: lstm.h, line: 26

```c
#define PS_LSTM_FORGET_IDX 3
```




### PS_LSTM_INPUT_IDX

In: lstm.h, line: 24

```c
#define PS_LSTM_INPUT_IDX 1
```




### PS_LSTM_OUTPUT_IDX

In: lstm.h, line: 25

```c
#define PS_LSTM_OUTPUT_IDX 2
```




### PS_LSTM_RAWSTATE_IDX

In: lstm.h, line: 27

```c
#define PS_LSTM_RAWSTATE_IDX 4
```




### PS_MATRIX_MAX_DIMENSIONS

In: maths.h, line: 71

```c
#define PS_MATRIX_MAX_DIMENSIONS 3
```




### PS_MAX_MEMORY_GRADIENTS

In: psyc.h, line: 47

```c
#define PS_MAX_MEMORY_GRADIENTS 2
```




### PS_MAX_PROVIDERS

In: operator-layer.h, line: 24

```c
#define PS_MAX_PROVIDERS 1024
```




### PS_MAX_SEQUENCE_LENGTH

In: psyc.h, line: 50

```c
#define PS_MAX_SEQUENCE_LENGTH 10
```




### PS_MNIST_INPUT_SIZE

In: dataset.h, line: 247

```c
#define PS_MNIST_INPUT_SIZE (28 * 28)
```




### PS_NULL_VALUE

In: psyc.h, line: 80

```c
#define PS_NULL_VALUE PSFLOAT_MIN
```




### PS_OPT_TIME_FULL

In: utils.h, line: 52

```c
#define PS_OPT_TIME_FULL (1 << 1)
```




### PS_OPT_TIME_HUMAN

In: utils.h, line: 53

```c
#define PS_OPT_TIME_HUMAN (1 << 2)
```




### PS_OPT_TIME_LONG

In: utils.h, line: 51

```c
#define PS_OPT_TIME_LONG (1 << 0)
```




### PS_OPT_TIME_ROUND_SEC

In: utils.h, line: 54

```c
#define PS_OPT_TIME_ROUND_SEC (1 << 3)
```




### PS_OPTIMIZATION

In: buildinfo.h, line: 7

```c
#define PS_OPTIMIZATION "2"
```




### PS_OUTPUT_PROJ_IDX

In: attention.h, line: 32

```c
#define PS_OUTPUT_PROJ_IDX 3
```




### PS_OUTPUT_PROJECTION

In: attention.h, line: 26

```c
#define PS_OUTPUT_PROJECTION (1 << 3)
```




### PS_PADDING_FULL

In: convolutional.h, line: 32

```c
#define PS_PADDING_FULL -2
```

Automatically determine padding so that the output size is always bigger than the input size (padding = filter_width - 1). Stride must be 1 and filter_height must be 1 or the same of filer_width.


### PS_PADDING_SAME

In: convolutional.h, line: 28

```c
#define PS_PADDING_SAME -1
```

Automatically determine padding in order to make the output size the same as the input size (padding = filter_width / 2). Stride must be 1, filter_height must be 1 or the same of filter_width and filter_width must be odd.


### PS_PARAM_BIAS

In: psyc.h, line: 66

```c
#define PS_PARAM_BIAS 1
```




### PS_PARAM_WEIGHT

In: psyc.h, line: 67

```c
#define PS_PARAM_WEIGHT 2
```




### PS_PARSER_FLAG_ENCODE_ONLY

In: dataset.h, line: 59

```c
#define PS_PARSER_FLAG_ENCODE_ONLY (1 << 3)
```

Just generate a dataset that only consist of parsed tokens, with no metadata, no sequence splitting and not targets.


### PS_PARSER_FLAG_END_TOKEN

In: dataset.h, line: 89

```c
#define PS_PARSER_FLAG_END_TOKEN (1 << 5)
```

Add an 'ending' token to the dataset:  

 - If the string is being parsed as collection of fixed length sequences, the ending token will be appended to the last sequence only.
 - If the string is being split into variable-length sequences (ie. by using a sequence separator), but the target sequence has the same length of the input sequence, the ending token will be appended to every target sequence.
 - If the dataset has target sequences whose length can differ from the related input sequences (ie. targets come from another dataset), the ending token is appended to every target sequence and, unless the [PS_PARSER_FLAG_EXACT_INPUTS](macros.md#ps-parser-flag-exact-inputs) is set, to every input sequence.
 - If no string is being provided as the starting token with [end_token](types.md#pstextparseroptions) member of [PSTextParserOptions](types.md#pstextparseroptions), by default [PS_DEFAULT_END_TOKEN](macros.md#ps-default-end-token) is used.


### PS_PARSER_FLAG_EXACT_INPUTS

In: dataset.h, line: 97

```c
#define PS_PARSER_FLAG_EXACT_INPUTS (1 << 7)
```

When the dataset has target sequences whose length can differ from the  related input sequences (ie. targets come from another dataset), this  flag prevents start/end tokens (see [PS_PARSER_FLAG_START_TOKEN](macros.md#ps-parser-flag-start-token) and  [PS_PARSER_FLAG_START_TOKEN](macros.md#ps-parser-flag-start-token) to be added to the input sequences.


### PS_PARSER_FLAG_MAKE_TARGETS

In: dataset.h, line: 92

```c
#define PS_PARSER_FLAG_MAKE_TARGETS (1 << 6)
```

Let text parsing functions (ie. PSDataFromText) also generate the target sequence for every input sequence.


### PS_PARSER_FLAG_NO_NORMALIZATION

In: dataset.h, line: 51

```c
#define PS_PARSER_FLAG_NO_NORMALIZATION (1 << 0)
```




### PS_PARSER_FLAG_PRESERVE_STRING

In: dataset.h, line: 53

```c
#define PS_PARSER_FLAG_PRESERVE_STRING (1 << 1)
```




### PS_PARSER_FLAG_READONLY_VOCAB

In: dataset.h, line: 56

```c
#define PS_PARSER_FLAG_READONLY_VOCAB (1 << 2)
```

Prevent adding new tokens to vocabulary used for generating a dataset from a parsed string.


### PS_PARSER_FLAG_START_TOKEN

In: dataset.h, line: 74

```c
#define PS_PARSER_FLAG_START_TOKEN (1 << 4)
```

Add a 'starting' token to the dataset:  

 - If the string is being parsed as collection of fixed length sequences, the starting token will be prepended to the first sequence only.
 - If the string is being split into variable-length sequences (ie. by using a sequence separator), but the target sequence has the same length of the input sequence, the starting token will be prepended to every input sequence.
 - If the dataset has target sequences whose length can differ from the related input sequences (ie. targets come from another dataset), the starting token is prepended to every target sequence and, unless the [PS_PARSER_FLAG_EXACT_INPUTS](macros.md#ps-parser-flag-exact-inputs) is set, to every input sequence.
 - If no string is being provided as the starting token with [start_token](types.md#pstextparseroptions) member of [PSTextParserOptions](types.md#pstextparseroptions), by default [PS_DEFAULT_START_TOKEN](macros.md#ps-default-start-token) is used.


### PS_PARSER_MODE_CHARS

In: dataset.h, line: 48

```c
#define PS_PARSER_MODE_CHARS 1
```




### PS_PARSER_MODE_TOKENS

In: dataset.h, line: 46

```c
#define PS_PARSER_MODE_TOKENS 0
```




### PS_PREFIX

In: buildinfo.h, line: 8

```c
#define PS_PREFIX "/usr/local"
```




### PS_PROGRESS_FLAG_JUST_BAR

In: log.h, line: 100

```c
#define PS_PROGRESS_FLAG_JUST_BAR (1 << 2)
```




### PS_PROGRESS_FLAG_JUST_PERCENT

In: log.h, line: 99

```c
#define PS_PROGRESS_FLAG_JUST_PERCENT (1 << 1)
```




### PS_PROGRESS_FLAG_NO_GRADIENT

In: log.h, line: 102

```c
#define PS_PROGRESS_FLAG_NO_GRADIENT (1 << 4)
```




### PS_PROGRESS_FLAG_NO_TOTAL

In: log.h, line: 101

```c
#define PS_PROGRESS_FLAG_NO_TOTAL (1 << 3)
```




### PS_PROGRESS_FLAG_NO_XTERM256

In: log.h, line: 103

```c
#define PS_PROGRESS_FLAG_NO_XTERM256 (1 << 5)
```




### PS_PROGRESS_FLAG_PERCENT

In: log.h, line: 98

```c
#define PS_PROGRESS_FLAG_PERCENT (1 << 0)
```




### PS_PROGRESS_FLAG_PERCENT_RIGHT

In: log.h, line: 105

```c
#define PS_PROGRESS_FLAG_PERCENT_RIGHT (1 << 7)
```




### PS_PROGRESS_FLAG_XTERM256_CODE

In: log.h, line: 104

```c
#define PS_PROGRESS_FLAG_XTERM256_CODE (1 << 6)
```




### PS_PROGRESS_STYLE_BAR

In: log.h, line: 95

```c
#define PS_PROGRESS_STYLE_BAR 2
```




### PS_PROGRESS_STYLE_DOUBLE_DASH

In: log.h, line: 93

```c
#define PS_PROGRESS_STYLE_DOUBLE_DASH 0
```




### PS_PROGRESS_STYLE_LINE

In: log.h, line: 96

```c
#define PS_PROGRESS_STYLE_LINE 3
```




### PS_PROGRESS_STYLE_SINGLE_DASH

In: log.h, line: 94

```c
#define PS_PROGRESS_STYLE_SINGLE_DASH 1
```




### PS_QUERY_PROJ_IDX

In: attention.h, line: 29

```c
#define PS_QUERY_PROJ_IDX 0
```




### PS_QUERY_PROJECTION

In: attention.h, line: 23

```c
#define PS_QUERY_PROJECTION (1 << 0)
```




### PS_SCORES_PROJ_IDX

In: attention.h, line: 33

```c
#define PS_SCORES_PROJ_IDX 4
```




### PS_SCORES_PROJECTION

In: attention.h, line: 27

```c
#define PS_SCORES_PROJECTION (1 << 4)
```




### PS_SHAPE_TYPE_COL

In: maths.h, line: 68

```c
#define PS_SHAPE_TYPE_COL 3
```




### PS_SHAPE_TYPE_MATRIX

In: maths.h, line: 69

```c
#define PS_SHAPE_TYPE_MATRIX 4
```




### PS_SHAPE_TYPE_NONE

In: maths.h, line: 65

```c
#define PS_SHAPE_TYPE_NONE 0
```




### PS_SHAPE_TYPE_ROW

In: maths.h, line: 67

```c
#define PS_SHAPE_TYPE_ROW 2
```




### PS_SHAPE_TYPE_SCALAR

In: maths.h, line: 66

```c
#define PS_SHAPE_TYPE_SCALAR 1
```




### PS_STATUS_ABORTED

In: psyc.h, line: 58

```c
#define PS_STATUS_ABORTED 5
```




### PS_STATUS_ERROR

In: psyc.h, line: 56

```c
#define PS_STATUS_ERROR 3
```




### PS_STATUS_PAUSED

In: psyc.h, line: 57

```c
#define PS_STATUS_PAUSED 4
```




### PS_STATUS_PRETRAINING

In: psyc.h, line: 60

```c
#define PS_STATUS_PRETRAINING 7
```




### PS_STATUS_TRAINED

In: psyc.h, line: 54

```c
#define PS_STATUS_TRAINED 1
```




### PS_STATUS_TRAINING

In: psyc.h, line: 55

```c
#define PS_STATUS_TRAINING 2
```




### PS_STATUS_UNTRAINED

In: psyc.h, line: 53

```c
#define PS_STATUS_UNTRAINED 0
```




### PS_STATUS_VALIDATING

In: psyc.h, line: 59

```c
#define PS_STATUS_VALIDATING 6
```




### PS_STORE_MODE_ADD

In: maths.h, line: 62

```c
#define PS_STORE_MODE_ADD 1
```




### PS_STORE_MODE_SET

In: maths.h, line: 61

```c
#define PS_STORE_MODE_SET 0
```




### PS_STORE_MODE_SUB

In: maths.h, line: 63

```c
#define PS_STORE_MODE_SUB 2
```




### PS_TOKEN_NOT_FOUND

In: dataset.h, line: 41

```c
#define PS_TOKEN_NOT_FOUND -2
```




### PS_TRAINING_ADJUST_RATE

In: psyc.h, line: 100

```c
#define PS_TRAINING_ADJUST_RATE (1 << 1)
```




### PS_TRAINING_EPOCH_AS_SEQUENCE

In: psyc.h, line: 102

```c
#define PS_TRAINING_EPOCH_AS_SEQUENCE (1 << 3)
```




### PS_TRAINING_FLAG_AUTOREGRESSION

In: psyc.h, line: 104

```c
#define PS_TRAINING_FLAG_AUTOREGRESSION (1 << 7)
```




### PS_TRAINING_FLAG_SELFSUPERVISED

In: psyc.h, line: 103

```c
#define PS_TRAINING_FLAG_SELFSUPERVISED (1 << 4)
```




### PS_TRAINING_FLAG_SEQ2SEQ

In: psyc.h, line: 106

```c
#define PS_TRAINING_FLAG_SEQ2SEQ (1 << 9)
```




### PS_TRAINING_FLAG_TEACHER_FORCING

In: psyc.h, line: 105

```c
#define PS_TRAINING_FLAG_TEACHER_FORCING (1 << 8)
```




### PS_TRAINING_METRICS_ACCURACY

In: psyc.h, line: 109

```c
#define PS_TRAINING_METRICS_ACCURACY (1 << 0)
```




### PS_TRAINING_NO_SHUFFLE

In: psyc.h, line: 99

```c
#define PS_TRAINING_NO_SHUFFLE (1 << 0)
```




### PS_TRAINING_PHASE_BACKPROP

In: psyc.h, line: 75

```c
#define PS_TRAINING_PHASE_BACKPROP 2
```




### PS_TRAINING_PHASE_FORWARD

In: psyc.h, line: 74

```c
#define PS_TRAINING_PHASE_FORWARD 1
```




### PS_TRAINING_PHASE_UPDATE_GRAD

In: psyc.h, line: 76

```c
#define PS_TRAINING_PHASE_UPDATE_GRAD 3
```




### PS_TRAINING_WEIGHT_DECAY

In: psyc.h, line: 101

```c
#define PS_TRAINING_WEIGHT_DECAY (1 << 2)
```




### PS_VALUES_PROJ_IDX

In: attention.h, line: 31

```c
#define PS_VALUES_PROJ_IDX 2
```




### PS_VALUES_PROJECTION

In: attention.h, line: 25

```c
#define PS_VALUES_PROJECTION (1 << 2)
```




### PSAbs

In: maths.h, line: 35

```c
#ifdef PS_DOUBLE_PRECISION
#define PSAbs(v) fabs(v)
#else
#define PSAbs(v) fabsf(v)
#endif
```




### PSAccelerateEnabled

In: config.h, line: 36

```c
#define PSAccelerateEnabled(acceleration) (PSIsAccelerationEnabled(acceleration, PSAcceleration_Accelerate))
```




### PSAddContextualDebug

In: debug.h, line: 41

```c
#define PSAddContextualDebug(model,l,n1,n2,prop,v,...) PSAddDebugInfo( model, __FILE__, __func__, __LINE__, l, n1, n2, prop, v, __VA_ARGS__)
```




### PSAssertWithMessage

In: debug.h, line: 33

```c
#define PSAssertWithMessage(expr, fmt, ...) do { if (!(expr)) { printf("\n\n== ASSERTION FAILURE ==\n"); fprintf(stderr, fmt, __VA_ARGS__); assert(expr); }} while (0);
```




### PSAutoAccelerationEnabled

In: config.h, line: 40

```c
#define PSAutoAccelerationEnabled(acceleration) (PSIsAccelerationEnabled(acceleration, PSAcceleration_Auto))
```




### PSAVXEnabled

In: config.h, line: 34

```c
#define PSAVXEnabled(acceleration) (PSIsAccelerationEnabled(acceleration, PSAcceleration_AVX))
```




### PSBitmapAnd

In: utils.h, line: 60

```c
#define PSBitmapAnd(a, b, dest) PSBitmapOp(a, b, dest, PS_BITMAP_OP_AND)
```




### PSBitmapOr

In: utils.h, line: 61

```c
#define PSBitmapOr(a, b, dest) PSBitmapOp(a, b, dest, PS_BITMAP_OP_OR)
```




### PSBLAS_MAX

In: blas.h, line: 27

```c
#if PSBLAS_INT_SIZE == 8 && !defined(PS_LAPACK_I32)
#define PSBLAS_MAX INT64_MAX
#else
#define PSBLAS_MAX (long) INT32_MAX
#else
#define PSBLAS_MAX (long) INT32_MAX
#endif
```




### PSBLASEnabled

In: config.h, line: 38

```c
#define PSBLASEnabled(acceleration) (PSIsAccelerationEnabled(acceleration, PSAcceleration_BLAS))
```




### PSClearScreen

In: log.h, line: 117

```c
#define PSClearScreen() (printf("\x1b[1;1H\x1b[2J"))
```




### PSClipValue

In: maths.h, line: 53

```c
#define PSClipValue(v, min, max) (v > max ? max : (v < min ? min : v))
```




### PSCOLOR_BLACK

In: log.h, line: 32

```c
#define PSCOLOR_BLACK "\x1b[30m"
```




### PSCOLOR_BLUE

In: log.h, line: 36

```c
#define PSCOLOR_BLUE "\x1b[34m"
```




### PSCOLOR_BOLD

In: log.h, line: 56

```c
#define PSCOLOR_BOLD PSSTYLE_BOLD
```




### PSCOLOR_CYAN

In: log.h, line: 38

```c
#define PSCOLOR_CYAN "\x1b[36m"
```




### PSCOLOR_DARK

In: log.h, line: 58

```c
#define PSCOLOR_DARK PSSTYLE_DIM
```




### PSCOLOR_DIM

In: log.h, line: 57

```c
#define PSCOLOR_DIM PSSTYLE_DIM
```




### PSCOLOR_GRAY

In: log.h, line: 40

```c
#define PSCOLOR_GRAY "\x1b[90m"
```




### PSCOLOR_GREEN

In: log.h, line: 34

```c
#define PSCOLOR_GREEN "\x1b[32m"
```




### PSCOLOR_LIGHT_BLUE

In: log.h, line: 44

```c
#define PSCOLOR_LIGHT_BLUE "\x1b[94m"
```




### PSCOLOR_LIGHT_CYAN

In: log.h, line: 46

```c
#define PSCOLOR_LIGHT_CYAN "\x1b[96m"
```




### PSCOLOR_LIGHT_GREEN

In: log.h, line: 42

```c
#define PSCOLOR_LIGHT_GREEN "\x1b[92m"
```




### PSCOLOR_LIGHT_MAGENTA

In: log.h, line: 45

```c
#define PSCOLOR_LIGHT_MAGENTA "\x1b[95m"
```




### PSCOLOR_LIGHT_RED

In: log.h, line: 41

```c
#define PSCOLOR_LIGHT_RED "\x1b[91m"
```




### PSCOLOR_LIGHT_WHITE

In: log.h, line: 47

```c
#define PSCOLOR_LIGHT_WHITE "\x1b[97m"
```




### PSCOLOR_LIGHT_YELLOW

In: log.h, line: 43

```c
#define PSCOLOR_LIGHT_YELLOW "\x1b[93m"
```




### PSCOLOR_MAGENTA

In: log.h, line: 37

```c
#define PSCOLOR_MAGENTA "\x1b[35m"
```




### PSCOLOR_RED

In: log.h, line: 33

```c
#define PSCOLOR_RED "\x1b[31m"
```




### PSCOLOR_RESET

In: log.h, line: 48

```c
#define PSCOLOR_RESET "\x1b[0m"
```




### PSCOLOR_WHITE

In: log.h, line: 39

```c
#define PSCOLOR_WHITE "\x1b[37m"
```




### PSCOLOR_YELLOW

In: log.h, line: 35

```c
#define PSCOLOR_YELLOW "\x1b[33m"
```




### PSCos

In: maths.h, line: 38

```c
#ifdef PS_DOUBLE_PRECISION
#define PSCos(a) cos(a)
#else
#define PSCos(a) cosf(a)
#endif
```




### PSDEFAULT_LOGLEVEL

In: log.h, line: 87

```c
#define PSDEFAULT_LOGLEVEL PSLOGLEVEL_INFO
```




### PSDEFAULT_NORM_EPSILON

In: normalization.h, line: 23

```c
#define PSDEFAULT_NORM_EPSILON 1e-5
```




### PSDefaultOptimization

In: optimization.h, line: 24

```c
#define PSDefaultOptimization PSSGDOptimization
```




### PSDICT_HT_SIZE

In: utils.h, line: 33

```c
#define PSDICT_HT_SIZE 4096
```




### PSDICT_UPDATE_DISABLED

In: utils.h, line: 34

```c
#define PSDICT_UPDATE_DISABLED (1 << 0)
```




### PSDictGetStr

In: utils.h, line: 41

```c
#define PSDictGetStr(dict, key) ((const char *) PSDictGetPointer(dict, key))
```




### PSDictHash

In: utils.h, line: 35

```c
#define PSDictHash(key) (djb33_hash(key, 4096))
```




### PSDictItemFromFloat

In: utils.h, line: 37

```c
#define PSDictItemFromFloat(n) ((PSDictValue) {.as_float = n})
```




### PSDictItemFromInt

In: utils.h, line: 36

```c
#define PSDictItemFromInt(n) ((PSDictValue) {.as_int = n})
```




### PSDictItemFromPointer

In: utils.h, line: 38

```c
#define PSDictItemFromPointer(ptr) ((PSDictValue) {.as_ptr = ptr})
```




### PSDictItemFromString

In: utils.h, line: 39

```c
#define PSDictItemFromString(str) PSDictItemFromPointer(str)
```




### PSDictSlotForKey

In: utils.h, line: 40

```c
#define PSDictSlotForKey(key) (PSDictHash(key) % PSDICT_HT_SIZE)
```




### PSDisablePretraining

In: psyc.h, line: 125

```c
#define PSDisablePretraining(layer) (layer->pretrain = NULL)
```




### PSExp

In: maths.h, line: 31

```c
#ifdef PS_DOUBLE_PRECISION
#define PSExp(v) exp(v)
#else
#define PSExp(v) expf(v)
#endif
```




### PSFileExists

In: utils.h, line: 43

```c
#define PSFileExists(path) (access(path, F_OK) == 0)
```




### PSFLOAT_DIG

In: types.h, line: 34

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_DIG DBL_DECIMAL_DIG
#else
#define PSFLOAT_DIG FLT_DECIMAL_DIG
#endif
```




### PSFLOAT_EPS

In: types.h, line: 37

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_EPS DBL_EPSILON
#else
#define PSFLOAT_EPS FLT_EPSILON
#endif
```




### PSFLOAT_FORMAT

In: types.h, line: 33

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_FORMAT "%lg"
#else
#define PSFLOAT_FORMAT "%g"
#endif
```




### PSFLOAT_MAX

In: types.h, line: 36

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_MAX DBL_MAX
#else
#define PSFLOAT_MAX FLT_MAX
#endif
```




### PSFLOAT_MIN

In: types.h, line: 35

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFLOAT_MIN DBL_MIN
#else
#define PSFLOAT_MIN FLT_MIN
#endif
```




### PSFloor

In: maths.h, line: 30

```c
#ifdef PS_DOUBLE_PRECISION
#define PSFloor(v) floor(v)
#else
#define PSFloor(v) floorf(v)
#endif
```




### PSGetColumn

In: convolutional.h, line: 36

```c
#define PSGetColumn(index, width) (index % width)
```




### PSGetConvolutionalSettings

In: convolutional.h, line: 34

```c
#define PSGetConvolutionalSettings(layer) ((PSConvolutionalSettings *) layer->extra)
```




### PSGetElapsedTimeMS

In: utils.h, line: 46

```c
#define PSGetElapsedTimeMS(st, et) ((((et.tv_sec - st.tv_sec) * 1000000) /* Get elapsed time in microseconds */
```




### PSGetElapsedTimeUS

In: utils.h, line: 48

```c
#define PSGetElapsedTimeUS(st, et) (((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec))
```




### PSGetNormalizationSettings

In: normalization.h, line: 24

```c
#define PSGetNormalizationSettings(layer) ((PSNormalizationLayerSettings*) layer->extra)
```




### PSGetRow

In: convolutional.h, line: 37

```c
#define PSGetRow(index, width) ((int) ((int) index / (int) width))
```




### PSGlobalDisableAcceleration

In: config.h, line: 45

```c
#define PSGlobalDisableAcceleration(acceleration) PSDisableAcceleration( &PSGlobalAcceleration, acceleration)
```




### PSGlobalEnableAcceleration

In: config.h, line: 43

```c
#define PSGlobalEnableAcceleration(acceleration) PSEnableAcceleration( &PSGlobalAcceleration, acceleration)
```




### PSHandleSequenceAtOnce

In: psyc.h, line: 120

```c
#define PSHandleSequenceAtOnce(o) (PSUseSequences(o) && !PSIsRecurrent(o))
```




### PSHasFlag

In: config.h, line: 30

```c
#define PSHasFlag(flags, flag) (flags & flag)
```




### PSIsModelChain

In: psyc.h, line: 121

```c
#define PSIsModelChain(model) (model->previous != NULL || model->next != NULL)
```




### PSIsModelTraining

In: psyc.h, line: 126

```c
#define PSIsModelTraining(model) (PSModelGetStatus(model) == PS_STATUS_TRAINING)
```




### PSIsMultiHeadAttention

In: attention.h, line: 35

```c
#define PSIsMultiHeadAttention(layer) (PSGetAttentionHeadCount(layer) > 1)
```




### PSIsRecurrent

In: psyc.h, line: 116

```c
#define PSIsRecurrent(o) (o->flags & PS_FLAG_RECURRENT)
```




### PSLDEF

In: psyc.h, line: 122

```c
#define PSLDEF(...) ((PSLayerDef *) &((PSLayerDef) {__VA_ARGS__}))
```




### PSLog

In: maths.h, line: 33

```c
#ifdef PS_DOUBLE_PRECISION
#define PSLog(v) log(v)
#else
#define PSLog(v) logf(v)
#endif
```




### PSLog10

In: maths.h, line: 34

```c
#ifdef PS_DOUBLE_PRECISION
#define PSLog10(v) log10(v)
#else
#define PSLog10(v) log10f(v)
#endif
```




### PSLogColorEnabled

In: log.h, line: 118

```c
#define PSLogColorEnabled() (PSGlobalFlags & PS_FLAG_LOG_COLORS)
```




### PSLogDisableColor

In: log.h, line: 120

```c
#define PSLogDisableColor() (PSGlobalFlags &= ~((unsigned) PS_FLAG_LOG_COLORS))
```




### PSLogEnableColor

In: log.h, line: 119

```c
#define PSLogEnableColor() (PSGlobalFlags |= PS_FLAG_LOG_COLORS)
```




### PSLOGLEVEL_DEBUG

In: log.h, line: 24

```c
#define PSLOGLEVEL_DEBUG 0
```




### PSLOGLEVEL_ERROR

In: log.h, line: 29

```c
#define PSLOGLEVEL_ERROR 5
```




### PSLOGLEVEL_FATAL

In: log.h, line: 30

```c
#define PSLOGLEVEL_FATAL 6
```




### PSLOGLEVEL_INFO

In: log.h, line: 25

```c
#define PSLOGLEVEL_INFO 1
```




### PSLOGLEVEL_NOTICE

In: log.h, line: 26

```c
#define PSLOGLEVEL_NOTICE 2
```




### PSLOGLEVEL_SUCCESS

In: log.h, line: 27

```c
#define PSLOGLEVEL_SUCCESS 3
```




### PSLOGLEVEL_WARN

In: log.h, line: 28

```c
#define PSLOGLEVEL_WARN 4
```




### PSMatrixDataSize

In: maths.h, line: 130

```c
#define PSMatrixDataSize(matrix) (PSMatrixLength(matrix) * sizeof(PSFloat))
```




### PSMatrixDimensions

In: maths.h, line: 59

```c
#define PSMatrixDimensions(matrix, shape) PSMatrixShape(matrix, shape)
```




### PSMatrixStrideBytes

In: maths.h, line: 131

```c
#define PSMatrixStrideBytes(matrix,i) (PSMatrixStride(matrix,i) * sizeof(PSFloat))
```




### PSPow

In: maths.h, line: 36

```c
#ifdef PS_DOUBLE_PRECISION
#define PSPow(a,b) pow(a, b)
#else
#define PSPow(a,b) powf(a, b)
#endif
```




### PSPrintMemoryErrorMsg

In: log.h, line: 121

```c
#define PSPrintMemoryErrorMsg() PSErr(NULL, "Could not allocate memory!")
```




### PSRemoveFlag

In: config.h, line: 32

```c
#define PSRemoveFlag(flags, flag) (flags &= ~((unsigned) flags))
```




### PSRound

In: maths.h, line: 32

```c
#ifdef PS_DOUBLE_PRECISION
#define PSRound(v) round(v)
#else
#define PSRound(v) roundf(v)
#endif
```




### PSSetFlag

In: config.h, line: 31

```c
#define PSSetFlag(flags, flag) (flags != flag)
```




### PSSetRecurrent

In: psyc.h, line: 117

```c
#define PSSetRecurrent(o) (o->flags |= PS_FLAG_RECURRENT)
```




### PSShouldDebugDump

In: debug.h, line: 28

```c
#define PSShouldDebugDump(model) (model->training != NULL && model->training->debug_dump_to != NULL && model->training->current_example == 0 && model->training->current_epoch == 0)
```




### PSSin

In: maths.h, line: 37

```c
#ifdef PS_DOUBLE_PRECISION
#define PSSin(a) sin(a)
#else
#define PSSin(a) sinf(a)
#endif
```




### PSSqrt

In: maths.h, line: 29

```c
#ifdef PS_DOUBLE_PRECISION
#define PSSqrt(v) sqrt(v)
#else
#define PSSqrt(v) sqrtf(v)
#endif
```




### PSSTYLE_BOLD

In: log.h, line: 49

```c
#define PSSTYLE_BOLD "\x1b[1m"
```




### PSSTYLE_DIM

In: log.h, line: 50

```c
#define PSSTYLE_DIM "\x1b[2m"
```




### PSSTYLE_HIDDEN

In: log.h, line: 51

```c
#define PSSTYLE_HIDDEN "\x1b[8m"
```




### PSSTYLE_ITALICS

In: log.h, line: 52

```c
#define PSSTYLE_ITALICS "\x1b[3m"
```




### PSSTYLE_STRIKETHROUGH

In: log.h, line: 54

```c
#define PSSTYLE_STRIKETHROUGH "\x1b[9m"
```




### PSSTYLE_UNDERLINE

In: log.h, line: 53

```c
#define PSSTYLE_UNDERLINE "\x1b[4m"
```




### PSTanh

In: maths.h, line: 28

```c
#ifdef PS_DOUBLE_PRECISION
#define PSTanh(v) tanh(v)
#else
#define PSTanh(v) tanhf(v)
#endif
```




### PSTanhS

In: activation.h, line: 25

```c
#ifdef PS_DOUBLE_PRECISION
#define PSTanhS tanh
#else
#define PSTanhS tanhf
#endif
```




### PSTRAINOPT

In: psyc.h, line: 123

```c
#define PSTRAINOPT(...) ((PSTrainingOptions *) &((PSTrainingOptions) {__VA_ARGS__}))
```




### PSUseSequences

In: psyc.h, line: 118

```c
#define PSUseSequences(o) (o->flags & (PS_FLAG_RECURRENT | PS_FLAG_USE_SEQUENCES))
```




### PSUTF8CharSize

In: utf8.h, line: 24

```c
#define PSUTF8CharSize(s) utf8_length[(((uint8_t *)(s))[0] & 0xFF) >> 4]
```




### PSVectorClear

In: maths.h, line: 55

```c
#define PSVectorClear(vec, len) memset(vec, 0, len * sizeof(PSFloat))
```




### PSVectorCopy

In: maths.h, line: 54

```c
#define PSVectorCopy(dest, src, len) memcpy(dest, src, len * sizeof(PSFloat))
```




### PSVectorCreate

In: maths.h, line: 57

```c
#define PSVectorCreate(len) malloc(len * sizeof(PSFloat))
```




### PSVectorZero

In: maths.h, line: 56

```c
#define PSVectorZero(len) calloc(len, sizeof(PSFloat))
```




### PSXTERM256_GRADIENT_BLACK_BLUE

In: log.h, line: 60

```c
#define PSXTERM256_GRADIENT_BLACK_BLUE 16
```




### PSXTERM256_GRADIENT_GRAYSCALE

In: log.h, line: 85

```c
#define PSXTERM256_GRADIENT_GRAYSCALE 232
```




### PSXTERM256_GRADIENT_GREEN_AZURE

In: log.h, line: 62

```c
#define PSXTERM256_GRADIENT_GREEN_AZURE 28
```




### PSXTERM256_GRADIENT_GREEN_BLUE

In: log.h, line: 61

```c
#define PSXTERM256_GRADIENT_GREEN_BLUE 22
```




### PSXTERM256_GRADIENT_GREEN_CYAN1

In: log.h, line: 63

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN1 34
```




### PSXTERM256_GRADIENT_GREEN_CYAN2

In: log.h, line: 64

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN2 40
```




### PSXTERM256_GRADIENT_GREEN_CYAN3

In: log.h, line: 65

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN3 46
```




### PSXTERM256_GRADIENT_GREEN_CYAN4

In: log.h, line: 66

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN4 112
```




### PSXTERM256_GRADIENT_GREEN_CYAN5

In: log.h, line: 67

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN5 118
```




### PSXTERM256_GRADIENT_GREEN_CYAN6

In: log.h, line: 68

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN6 148
```




### PSXTERM256_GRADIENT_GREEN_CYAN7

In: log.h, line: 69

```c
#define PSXTERM256_GRADIENT_GREEN_CYAN7 154
```




### PSXTERM256_GRADIENT_MAGENTA1

In: log.h, line: 70

```c
#define PSXTERM256_GRADIENT_MAGENTA1 52
```




### PSXTERM256_GRADIENT_MAGENTA2

In: log.h, line: 71

```c
#define PSXTERM256_GRADIENT_MAGENTA2 88
```




### PSXTERM256_GRADIENT_MAGENTA3

In: log.h, line: 72

```c
#define PSXTERM256_GRADIENT_MAGENTA3 124
```




### PSXTERM256_GRADIENT_MAGENTA4

In: log.h, line: 73

```c
#define PSXTERM256_GRADIENT_MAGENTA4 160
```




### PSXTERM256_GRADIENT_MAGENTA5

In: log.h, line: 74

```c
#define PSXTERM256_GRADIENT_MAGENTA5 196
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET1

In: log.h, line: 75

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET1 130
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET2

In: log.h, line: 76

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET2 136
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET3

In: log.h, line: 77

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET3 166
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET4

In: log.h, line: 78

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET4 172
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET5

In: log.h, line: 79

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET5 178
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET6

In: log.h, line: 80

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET6 202
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET7

In: log.h, line: 81

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET7 208
```




### PSXTERM256_GRADIENT_ORANGE_VIOLET8

In: log.h, line: 82

```c
#define PSXTERM256_GRADIENT_ORANGE_VIOLET8 214
```




### PSXTERM256_GRADIENT_YELLOW_WHITE1

In: log.h, line: 83

```c
#define PSXTERM256_GRADIENT_YELLOW_WHITE1 190
```




### PSXTERM256_GRADIENT_YELLOW_WHITE2

In: log.h, line: 84

```c
#define PSXTERM256_GRADIENT_YELLOW_WHITE2 26
```




### PSYC_CONTACT

In: psyc.h, line: 33

```c
#define PSYC_CONTACT PSYC_SITE "/issues"
```




### PSYC_NAME

In: psyc.h, line: 31

```c
#define PSYC_NAME "PsyC"
```




### PSYC_SITE

In: psyc.h, line: 32

```c
#define PSYC_SITE "https://github.com/artix75/psyc"
```




### PSYC_VERSION

In: psyc.h, line: 30

```c
#define PSYC_VERSION "0.9.5"
```




### PSYCH_MAGICK_VERSION

In: magick-conf.h, line: 1

```c
#define PSYCH_MAGICK_VERSION 7
```





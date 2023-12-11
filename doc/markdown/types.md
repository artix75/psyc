# PsyC Documentation - 0.9.3
## Types

### PSAcceleration

In: config.h, line: 48

```c
typedef enum {  
    PSAcceleration_None  = 0  
    PSAcceleration_AVX  = 1 /* (1 << 0) */  
    PSAcceleration_ACF  = 2 /* (1 << 1) */  
    PSAcceleration_BLAS  = 4 /* (1 << 2) */  
    PSAcceleration_Auto  = 32768 /* (1 << 15) */  
    PSAcceleration_All  = 0xFFFF  
} PSAcceleration  
```




### PSActivationFunction

In: activation.h, line: 30

```c
typedef void (* PSActivationFunction) (PSFloat *vec, PSFloat *dest, uint64_t len, PSMathOpts *opts)
```




### PSAttentionType

In: attention.h, line: 37

```c
typedef enum {  
    PSDotAttention  = 0  
    PSAdditiveAttention  = 1  
    PSInvalidAttention  = 9999  
} PSAttentionType  
```




### PSBackpropFunction

In: psyc.h, line: 129

```c
typedef int (* PSBackpropFunction) (struct PSLayer *layer, struct PSLayer *previousLayer, struct PSGradient *layer_gradients, ...)
```




### PSBeforeBackpropCallback

In: psyc.h, line: 157

```c
typedef int (* PSBeforeBackpropCallback) (struct PSModel *model, PSFloat *y, struct PSTrainingOptions *opts, struct PSGradient ** gradients)
```




### PSBeforeForwardCallback

In: psyc.h, line: 153

```c
typedef int (* PSBeforeForwardCallback) (struct PSModel *model, PSFloat *inputs, int seqlen, int backprop, void *opts)
```




### PSBitmap

In: utils.h, line: 112

```c
typedef uint64_t * PSBitmap
```




### PSBlasErr

In: blas.h, line: 28

```c
typedef struct PSBlasErr
```




### PSBLASErr

In: blas.h, line: 28

```c
typedef struct {  
    const char * func;  
    const char * param;  
    int param_pos;  
    int param_value;  
} PSBLASErr  
```




### PSBLASOrder

In: blas.h, line: 23

```c
typedef enum {  
    PSBLASRowMajor  = 0  
    PSBLASColMajor  = 1  
} PSBLASOrder  
```




### PSBooleanLayerCallback

In: psyc.h, line: 134

```c
typedef int (* PSBooleanLayerCallback) (struct PSLayer *layer)
```




### PSConvolutionalSettings

In: convolutional.h, line: 29

```c
typedef struct {  
    int stride;  
    int padding;  
    int filter_width;  
    int filter_height;  
    int filter_depth;  
    int input_width;  
    int input_height;  
    int input_depth;  
} PSConvolutionalSettings  
```




### PSCopyLayerCallback

In: psyc.h, line: 135

```c
typedef int (* PSCopyLayerCallback) (struct PSLayer *, struct PSLayer *)
```




### PSDebugInfo

In: debug.h, line: 44

```c
typedef struct {  
    char * file;  
    const char * func;  
    int line;  
    int status;  
    int current_epoch;  
    int current_batch;  
    int current_element;  
    int layer_index;  
    int layer_type;  
    int neuron_index;  
    int neuron2_index;  
    int layer2_index;  
    int convolutional_feature;  
    int timestep;  
    PSFloat activation;  
    PSFloat activation2;  
    PSFloat z_value;  
    PSFloat bias;  
    PSFloat weight;  
    PSFloat delta;  
    PSFloat delta2;  
    char * custom_prop;  
    PSFloat custom_val;  
    time_t time;  
    int has_info;  
} PSDebugInfo  
```




### PSDebugStepInfo

In: debug.h, line: 72

```c
typedef struct {  
    PSModel * model;  
    int training_phase;  
    const char * func;  
    PSLayer * layer;  
    PSNeuron * neuron;  
} PSDebugStepInfo  
```




### PSDict

In: utils.h, line: 84

```c
typedef struct {  
    int64_t length;  
    int flags;  
    PSDictItem * table[PSDICT_HT_SIZE];  
    PSOnDictItemRelease onItemRelease;  
} PSDict  
```




### PSDictItem

In: utils.h, line: 75

```c
typedef struct {  
    const char * key;  
    PSDictValue value;  
    struct PSDictItem * prev;  
    struct PSDictItem * next;  
    struct PSDict * dict;  
    int slot;  
} PSDictItem  
```




### PSDictIterator

In: utils.h, line: 91

```c
typedef struct {  
    PSDict * dict;  
    PSDictItem * current;  
} PSDictIterator  
```




### PSDictValue

In: utils.h, line: 69

```c
typedef union {  
    int64_t as_int;  
    PSFloat as_float;  
    void * as_ptr;  
} PSDictValue  
```




### PSDotProductDebug

In: maths.h, line: 74

```c
typedef void (* PSDotProductDebug) (void)
```




### PSEmbeddingType

In: embedding.h, line: 27

```c
typedef enum {  
    PSWord2Vec  = 0  
} PSEmbeddingType  
```

PSWord2Vec embedding type is based on Word2Vec algorithm created by Tomas Mikolov: [https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) Reference: [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)


### PSFloat

In: types.h, line: 38

```c
#ifdef PS_DOUBLE_PRECISION
typedef double PSFloat
#else
typedef float PSFloat
#endif
```

PsyC's type for floating-point numbers. By default, it's an alias for the **float** type. However, if PsyC has been built with the **PS_DOUBLE_PRECISION** macro defined (usually by building PsyC with **DOUBLE_PRECISION** make variable, ie. `make DOUBLE_PRECISION=on`), PSFloat will be an alias for **double**.


### PSFloatFunc

In: maths.h, line: 77

```c
typedef PSFloat (* PSFloatFunc) (PSFloat n)
```




### PSForwardFunction

In: psyc.h, line: 128

```c
typedef int (* PSForwardFunction) (struct PSLayer *layer, ...)
```




### PSForwardOptions

In: psyc.h, line: 256

```c
typedef struct {  
    int flags;  
    PSSequenceSettings * sequence_settings;  
} PSForwardOptions  
```




### PSGenericLayerCallback

In: psyc.h, line: 133

```c
typedef void (* PSGenericLayerCallback) (struct PSLayer *layer)
```




### PSGetParamCountFunction

In: psyc.h, line: 139

```c
typedef uint64_t (* PSGetParamCountFunction) (struct PSLayer *layer, int type)
```




### PSGradient

In: psyc.h, line: 218

```c
typedef struct {  
    uint64_t bias_count;  
    uint64_t weight_count;  
    PSFloat * biases;  
    PSFloat * weights;  
    PSFloat * tmp;  
} PSGradient  
```




### PSGRUCell

In: gru.h, line: 27

```c
typedef struct {  
    PSFloat * candidate_biases;  
    PSFloat * update_biases;  
    PSFloat * reset_biases;  
    PSMatrix candidate_weights;  
    PSMatrix update_weights;  
    PSMatrix reset_weights;  
    PSMatrix candidate_hidden_weights;  
    PSMatrix update_hidden_weights;  
    PSMatrix reset_hidden_weights;  
    PSMatrix candidates;  
    PSMatrix update_gates;  
    PSMatrix reset_gates;  
} PSGRUCell  
```




### PSInitStatesFunc

In: psyc.h, line: 140

```c
typedef int (* PSInitStatesFunc) (struct PSLayer *layer, uint32_t steps, int retain_previous)
```




### PSLayer

In: psyc.h, line: 301

```c
typedef struct {  
    PSLayerType type;  
    int index;  
    int size;  
    int weight_types;  
    PSMatrix * weights;  
    PSFloat * biases;  
    PSMatrix states;  
    PSMatrix delta;  
    PSFloat * initial_states;  
    uint32_t flags;  
    int onehot_vector_size;  
    int output_depth;  
    int output_columns;  
    int output_rows;  
    int pretrained;  
    void * extra;  
    void * private;  
    PSForwardFunction forward;  
    PSBackpropFunction backprop;  
    PSActivationFunction activate;  
    PSActivationFunction derivative;  
    PSBooleanLayerCallback build;  
    PSGenericLayerCallback onDelete;  
    PSCopyLayerCallback onCopy;  
    PSGenericLayerCallback beforeBatchTraining;  
    PSGetParamCountFunction getParamCount;  
    PSInitStatesFunc onStatesInit;  
    PSResizeStatesFunc onStatesResize;  
    PSPretrainLayerFunction pretrain;  
    PSLinkDataRetriever getInputFromLink;  
    struct PSModel * model;  
    struct PSModel * pretrainer;  
} PSLayer  
```




### PSLayerDef

In: psyc.h, line: 170

```c
typedef struct {  
    PSActivationFunction activation;  
    int flags;  
    int weight_init_mode;  
    int bias_init_mode;  
    PSFloat init_range;  
    PSFloat init_scale;  
    PSFloat init_value;  
    int output_depth;  
    int output_columns;  
    int output_rows;  
    int stride;  
    int padding;  
    int filter_width;  
    int filter_height;  
    int embedding_type;  
    PSFloat dropout;  
    PSFloat epsilon;  
    int pretrained;  
    const char * load_from;  
    const char * save_pretrained_to;  
    PSFloat * training_data;  
    int training_data_size;  
    struct PSTrainingOptions * pretraining_options;  
    int attention_type;  
    struct PSLayer * query_provider;  
    struct PSLayer * keys_provider;  
    struct PSLayer * values_provider;  
    PSFloat attention_scale;  
    int attention_heads;  
    int causal_attention;  
    int self_attention;  
    int enabled_projections;  
    int operator;  
    int providers_count;  
    struct PSLayer ** providers;  
    int positional_initial_capacity;  
    int positional_base;  
} PSLayerDef  
```




### PSLayerType

In: psyc.h, line: 226

```c
typedef enum {  
    FullyConnected  = 0  
    Convolutional  = 1  
    Pooling  = 2  
    RNNLayer  = 3  
    LSTM  = 4  
    SoftMax  = 5  
    GRU  = 6  
    Dropout  = 7  
    Embedding  = 8  
    Normalization  = 9  
    Attention  = 10  
    OperatorLayer  = 11  
    Linear  = 12  
    PositionalEncoding  = 13  
} PSLayerType  
```




### PSLinkDataRetriever

In: psyc.h, line: 152

```c
typedef int (* PSLinkDataRetriever) (struct PSLayer *layer)
```




### PSLossFunction

In: psyc.h, line: 144

```c
typedef PSFloat (* PSLossFunction) (PSFloat* x, PSFloat* y, int size, int onehot_size)
```




### PSLSTMCell

In: lstm.h, line: 29

```c
typedef struct {  
    PSFloat * candidate_biases;  
    PSFloat * input_biases;  
    PSFloat * output_biases;  
    PSFloat * forget_biases;  
    PSMatrix candidate_weights;  
    PSMatrix input_weights;  
    PSMatrix output_weights;  
    PSMatrix forget_weights;  
    PSMatrix candidate_hidden_weights;  
    PSMatrix input_hidden_weights;  
    PSMatrix output_hidden_weights;  
    PSMatrix forget_hidden_weights;  
    PSMatrix raw_states;  
    PSMatrix candidates;  
    PSMatrix input_gates;  
    PSMatrix output_gates;  
    PSMatrix forget_gates;  
    PSFloat * initial_raw_states;  
} PSLSTMCell  
```




### PSMathOpts

In: maths.h, line: 111

```c
typedef struct {  
    int acceleration;  
    int store_mode;  
    int transpose;  
    char argtype[3];  
    int vector_len;  
    PSFloat * tmpdest;  
    PSDotProductDebug debugStep;  
    void * data;  
} PSMathOpts  
```

This structure can be passed to various operations. Not all of its properties are used by all operations.  
Properties:  

 - [acceleration](types.md#psmathopts): see [PSAcceleration](types.md#psacceleration)
 - [store_mode](types.md#psmathopts): specifies how results will be stored into destination:
     - [PS_STORE_MODE_SET](macros.md#ps-store-mode-set): results will overwrite dest.
     - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): results will be added to dest.
     - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): results will be subtracted from dest.
 - [transpose](types.md#psmathopts):  some operations involving PSMatrix could use this in order to transpose one or more matrices. The integer value indicates the (1-based) matrix argument position, ie. 1 for first matrix arg, 2 for second matrix arg, etc. More than one matrix can be set (ie. 1 | 2).
 - **argtype**:    specifies if arguments are [PSMatrix](types.md#psmatrix) or [PSFloat](types.md#psfloat) (vector). Functions using this property (such as [PSDot](functions.md#psdot)), must have PSMatrix arguments and the property can be used to tell the function that one or more arguments must be treated as vectors.
     - 'M' or 'm': [PSMatrix](types.md#psmatrix) matrix
     - 'V' or 'v': [PSFloat](types.md#psfloat) vector                 The index indicated argument position (zero-based), ie: argtype[1] means that second PSMatrix argument has to be treated as vector.
 - [vector_len](types.md#psmathopts): optionally pass vector length to functions that cannot retrieve this info from matrix arguments, ie. when [PSDot](functions.md#psdot) is called with both vectors (**argtype** = {'V', 'V'})
 - [tmpdest](types.md#psmathopts):    some operations may use this vector as a cache in order to avoid allocating extra memory, for intermediate computations.
 - [debugStep](types.md#psmathopts):  used for debugging by some operations.


### PSMatrix

In: maths.h, line: 148

```c
typedef PSFloat * PSMatrix
```

Basically, [PSMatrix](types.md#psmatrix) can be used as a normal array of [PSFloat](types.md#psfloat) numbers. However, PSMatrix objects created by using the specific functions ([PSMatrixCreate](functions.md#psmatrixcreate), [PSMatrixZeros](functions.md#psmatrixzeros), and so on) will also contain several private informations about the matrix itself that allow them to be used as multidimensional matrices.  
Therefore, by using the PSMatrix-related functions provided by PsyC's API, it's possible to get info about the matrix (length, shape, ...) or to perform several operations (ie. transposition, matrix multiplication, etc.) on them.  


**WARN**:  matrix's private data are actually allocated just before the memory address pointed by [PSMatrix](types.md#psmatrix), so the matrix object should **NEVER** be freed by calling the usual **free** function or similar functions: the dedicated [PSMatrixFree](functions.md#psmatrixfree) function should be called instead.  


### PSMatrixInitializer

In: maths.h, line: 149

```c
typedef PSFloat (* PSMatrixInitializer) (void)
```




### PSModel

In: psyc.h, line: 341

```c
typedef struct {  
    const char * name;  
    int size;  
    int index;  
    PSLayer ** layers;  
    PSLossFunction loss;  
    uint32_t flags;  
    uint16_t acceleration;  
    uint8_t status;  
    uint32_t input_size;  
    uint32_t output_size;  
    struct PSModel * previous;  
    struct PSModel * next;  
    PSModelLink * previous_model_link;  
    PSSequenceSettings sequence_settings;  
    PSRecurrentNetworkMode rnn_mode;  
    PSTrainingInfo * training;  
    PSBeforeForwardCallback beforeForward;  
    PSBeforeBackpropCallback beforeBackprop;  
    PSTrainCallback onEpochTrained;  
    PSTrainCallback onBatchTrained;  
    void * context;  
} PSModel  
```




### PSModelLink

In: psyc.h, line: 336

```c
typedef struct {  
    PSLayer * layer;  
    PSLayer * previous_layer;  
} PSModelLink  
```




### PSNeuralNetwork

In: psyc.h, line: 477

```c
typedef PSModel PSNeuralNetwork
```

Kept type name used in older version since I still love it :) (and it also sounds more 'Psyc-y')


### PSNeuron

In: psyc.h, line: 293

```c
typedef struct {  
    int index;  
    PSFloat * bias;  
    PSFloat * weights;  
    void * extra;  
    struct PSLayer * layer;  
} PSNeuron  
```




### PSNormalizationLayerSettings

In: normalization.h, line: 27

```c
typedef struct {  
    PSFloat epsilon;  
} PSNormalizationLayerSettings  
```




### PSOnDictItemRelease

In: utils.h, line: 67

```c
typedef void (* PSOnDictItemRelease) (struct PSDictItem *)
```




### PSOperatorType

In: operator-layer.h, line: 26

```c
typedef enum {  
    PSConcatenateOperator  = 0  
    PSAddOperator  = 1  
    PSMultiplyOperator  = 2  
    PSInvalidOperator  = 999  
} PSOperatorType  
```




### PSOptimization

In: optimization.h, line: 24

```c
typedef int (* PSOptimization) (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, uint64_t len, int acceleration, int iteration, struct PSTrainingOptions *options)
```




### PSPretrainLayerFunction

In: psyc.h, line: 136

```c
typedef int (* PSPretrainLayerFunction) (struct PSLayer *, PSFloat *training_data, int data_size)
```




### PSRecurrentNetworkMode

In: psyc.h, line: 243

```c
typedef enum {  
    NonRecurrent  = 0  
    ManyToMany  = 1  
    ManyToOne  = 2  
    OneToMany  = 3  
} PSRecurrentNetworkMode  
```




### PSResizeStatesFunc

In: psyc.h, line: 142

```c
typedef int (* PSResizeStatesFunc) (struct PSLayer *layer, uint32_t steps, uint32_t previous_steps)
```




### PSScalarActivationFunction

In: activation.h, line: 32

```c
typedef PSFloat (* PSScalarActivationFunction) (PSFloat)
```




### PSSequenceSettings

In: psyc.h, line: 250

```c
typedef struct {  
    int max_length;  
    PSFloat * start;  
    int end;  
} PSSequenceSettings  
```




### PSSignalHandler

In: psyc.h, line: 168

```c
typedef void (* PSSignalHandler) (int)
```




### PSTextParserOptions

In: dataset.h, line: 100

```c
typedef struct {  
    int mode;  
    int flags;  
    int64_t max_vocabulary_size;  
    const char * separator;  
    const char * unknown_token;  
    int capacity;  
    int buffer_size;  
    PSTokenNormalizer normalizer;  
    PSTokenMatch match_token;  
} PSTextParserOptions  
```

Options for text parsing:  

 - [mode](types.md#pstextparseroptions): text parsing mode:
     - [PS_PARSER_MODE_TOKENS](macros.md#ps-parser-mode-tokens): parse text as tokens (each token will be converted to a number).
     - [PS_PARSER_MODE_CHARS](macros.md#ps-parser-mode-chars): parse text as characters (each individual character will be converted to a number).
 - [flags](types.md#pstextparseroptions): text parsing flags:
     - [PS_PARSER_FLAG_NO_NORMALIZATION](macros.md#ps-parser-flag-no-normalization): do not perform token normalization on parsed text.
     - [PS_PARSER_FLAG_PRESERVE_STRING](macros.md#ps-parser-flag-preserve-string): prevent string from being modified during parsing.
     - [PS_PARSER_FLAG_READONLY_VOCAB](macros.md#ps-parser-flag-readonly-vocab): by enabling this flag, the vocabulary will be treated as read-only. Any parsed token that is not present in the vocabulary will not be added and will be considered <unknown> (see the [unknown_token](types.md#pstextparseroptions) option).
 - [max_vocabulary_size](types.md#pstextparseroptions): maximum number of tokens that can be added to the vocabulary, except for the <unknown> token. Every new parsed token will be automatically converted to the <unknown> token (see the [unknown_token](types.md#pstextparseroptions) option). If the value of this option is zero, the default value will be [PS_DEFAULT_MAX_VOCAB_SIZE](macros.md#ps-default-max-vocab-size).
 - [separator](types.md#pstextparseroptions): a set of characters that should be used as separators to split string into individual tokens (ie: ".," would split by using both '.' and ',' as separators).
 - [unknown_token](types.md#pstextparseroptions): string to be used for unmatched tokens.
 - [capacity](types.md#psvocabulary): initial capacity of vocabularies allocated by parsing functions (ie. [PSLoadDataFromString](functions.md#psloaddatafromstring)).
 - [buffer_size](types.md#pstextparseroptions): parsing buffer size.
 - [normalizer](types.md#pstextparseroptions): pointer to function to be used to normalize tokens (see **PSTokenNormalizer**)
 - [match_token](types.md#pstextparseroptions): pointer to function to be used to match individual tokens (it usually overrides the usage of [separator](types.md#pstextparseroptions) to split string).


### PSTokenMatch

In: dataset.h, line: 63

```c
typedef int (* PSTokenMatch) (char *str, int *len)
```




### PSTrainCallback

In: psyc.h, line: 146

```c
typedef void (* PSTrainCallback) (struct PSModel *model, int epoch, int epochs, PSFloat average_loss, PSFloat current_loss, float accuracy, PSFloat *rate, PSFloat *training_data)
```




### PSTrainingInfo

In: psyc.h, line: 282

```c
typedef struct {  
    int current_epoch;  
    int current_batch;  
    int current_element;  
    int batch_size;  
    time_t started_at;  
    time_t ended_at;  
    int requested_action;  
    FILE * debug_dump_to;  
} PSTrainingInfo  
```




### PSTrainingOptions

In: psyc.h, line: 261

```c
typedef struct {  
    int epochs;  
    PSFloat learning_rate;  
    int batch_size;  
    int flags;  
    PSFloat l1_decay;  
    PSFloat l2_decay;  
    PSFloat momentum;  
    PSFloat rho;  
    PSFloat eps;  
    PSFloat beta1;  
    PSFloat beta2;  
    PSFloat clip;  
    PSOptimization optimization;  
    int bptt_truncate;  
    int validate_every_batches;  
    int max_validation_elements;  
    PSTrainingProgressFunc printProgress;  
    FILE * debug_dump_to;  
} PSTrainingOptions  
```




### PSTrainingProgressFunc

In: psyc.h, line: 161

```c
typedef void (* PSTrainingProgressFunc) (struct PSModel *model, int status, int epochs, int batches, PSFloat *loss, PSFloat *accuracy, time_t *elapsed, int validating_current, int validating_tot)
```




### PSUTF8Char

In: utf8.h, line: 23

```c
typedef uint32_t PSUTF8Char
```




### PSVocabulary

In: dataset.h, line: 112

```c
typedef struct {  
    int64_t size;  
    int64_t capacity;  
    PSDict * token_map;  
    const char ** tokens;  
} PSVocabulary  
```





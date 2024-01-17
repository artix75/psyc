## psycl

### SYNOPSIS

```shell
Usage: psycl [OPTIONS]
```

### DESCRIPTION

psycl is a command-line interface to PsyC, an open-source C library that allows building neural networks ( [https://github.com/artix75/psyc](https://github.com/artix75/psyc) ).  
The utility can be used to build, load, save and train models without the need to write a C program, albeit with limitations.  


### OPTIONS



**--available-accelerations**  
List available accelerations.  


**--batch-script-every**     NUM  
Interval of NUM batches after which to run the
script defined by the `--on-batch-trained`
option (if set).  


**--batch-size**     SIZE  
Training batch size (default: 10)

**--classify-image**     FILE [OPTIONS...]  
Classify the image located at path FILE with the
current model (see [IMAGE OPTIONS](#section-image-options) section).  


**-c, --config**     FILE  
Load options from FILE (see the [CONFIG FILES](#section-config-files)
section).  


**--disable-accelerate**  
Disable Accelerate Framework.  


**--disable-avx**  
Disable AVX.  


**--disable-blas**  
Disable BLAS.  


**--download-cifar**     [CLASSES] [DEST_DIR]  
Download CIFAR dataset and exit. If no DEST_DIR
is provided, the dataset will be saved into
PsyC working directory.  

Optional CLASSES can be 10 or 100.  

(default: 10).  


**--download-mnist**     [DEST_DIR]  
Download MNIST dataset and exit. If no DEST_DIR
is provided, the dataset will be saved into
PsyC working directory.  


**--enable-colors**  
Colorized output.  


**--epochs**     EPOCHS  
Training epochs (default: 30).  


**--info**  
Print PsyC info.  


**--l1-decay**     SIZE  
L1 Decay (default: 0).  


**--l2-decay**     SIZE  
L2 Decay (default: 0).  


**-l, --layer**     TYPE (SIZE | OPTIONS...)  
Add layer (see [LAYER TYPES](#section-layer-types) and [LAYER OPTIONS](#section-layer-options)
sections).  


**--learning-rate**     SIZE  
Training learning rate (default: 1.50).  


**--load**     PRETRAINED  
Load a pretrained model.  


**--loglevel**     LEVEL  
Set log level (see [LOG LEVELS](#section-log-levels) section).  


**--loss-function**     FUNC  
Loss Function (see [LOSS FUNCTIONS](#section-loss-functions) section).  


**--metrics**  
Training metrics (see [METRICS](#section-metrics) section).  


**--model**  
Start new model (it can be used
multiple times to create chained models
composed of multiple neural networks).  


**--momentum**     MOMENTUM  
Momentum (default: 0).  


**--name**     NAME  
Model name.  


**--on-batch-trained**     SCRIPT  
Execute script after every batch is trained.  

(See [SCRIPTS](#section-scripts) section for more info).  


**--on-epoch-trained**     SCRIPT  
Execute script after every epoch is trained.  

(See [SCRIPTS](#section-scripts) section for more info).  


**--onehot**  
Sets one-hot-vector flag for input layer (if
before 1st layer) or target dataset (if after
output layer).  


**--optimization**     NAME  
Training optimization, NAME can be:  

(adagrad | adadelta | adam | nesterov |
sgd | windowgrad).  


**--pidfile**     PATH  
Save process PID to PATH.  


**--quiet**  
Quiet output (loglevel ERROR).  


**--save**     FILE  
Save model to FILE.  


**--test**     [OPTIONS] TEST_DATASET  
Test model against TEST_DATASET.  

(see [TRAIN|TEST OPTIONS](#section-train-test-options) section).  


**--train**     [OPTIONS] TRAIN_DATASET  
Train model with TRAIN_DATASET.  

(see [TRAIN|TEST OPTIONS](#section-train-test-options) section).  


**--training-adjust-rate**  
Auto-adjust learn rate.  


**--training-datalen**     LEN  
Training data length.  


**--training-no-shuffle**  
Prevent dataset shuffle.  


**--validation-datalen**     LEN  
Validation data length.  


**--verbose**  
Verbose output (loglevel DEBUG).  


**-v, --version**  
Print version.  


**--weight-decay**  
Enable L1/L2 weight decay
instead of L1/L2 regularization.  


**-h, --help**  
Print this help.  






### LAYER TYPES



**fully-connected, fc**  
    Fully Connected (dense) layer   
**convolutional**  
    Convolutional Layer   
**pooling**  
    Pooling Layer   
**rnn**  
    Basic Recurrent Layer   
**lstm**  
    LSTM Layer   
**softmax**  
    Softmax Layer   
**gru**  
    GRU Layer   
**dropout**  
    Dropout Layer   
**embedding**  
    Embedding Layer   
**normalization**  
    Normalization Layer   
**attention**  
    Attention Layer   
**operator, op**  
    Operator Layer (add,concatenate,mul)   
**linear**  
    Linear Layer   
**positional-encoding**  
    Position Encoding Layer   




### LOSS FUNCTIONS



quadratic   
cross-entropy   




### LAYER OPTIONS



**--activation**     FUNC  
Activation Function: (sigmoid,tanh,relu,gelu).  


**--attention-heads**     NUM  
Multi-Head Attention layer heads.  


**--attention-scale**     SCALE  
Attention layer scale.  


**--attention-type**     TYPE  
Attention layer type: (dot|additive).  


**--bias-init-mode**     MODE  
Bias initialization mode:  

(auto|random|zero) (default: auto).  


**--causal**  
Causal Attention.  


**--disable-biases**  
Disable biases.  


**--dropout**     DROPOUT  
Layer Dropout (float).  


**--filter-width**     WIDTH  
Convolutional filter width (default: 5).  


**--filter-height**     HEIGHT  
Convolutional filter height (default: 5).  


**--init-range**     RANGE  
Weight|Bias initialization range.  

(for 'random' init mode)

**--init-scale**     SCALE  
Weight|Bias initialization scale.  

(for 'random' init mode)

**--key-provider**     COORDS  
Attention keys provider.  

(See [LAYER COORDINATES](#section-layer-coordinates) section for details
about COORDS).  


**--link**     COORDS  
Link layer to previous model.  

(See [LAYER COORDINATES](#section-layer-coordinates) section for details
about COORDS).  


**--load-layer**     PATH  
Load layer parameters from PATH.  


**--operator**     OP  
Operator layer operator: (add|mul|concatenate).  


**--output-width**     WIDTH  
Output Width.  


**--output-height**     HEIGHT  
Output Height.  


**--output-depth**     DEPTH  
Output Depth.  


**--padding**     PADDING  
Convolutional padding (default: 0).  


**--pretrained**  
Pretrained layer.  


**--provider**     COORDS  
Operator layer provider.  

(See [LAYER COORDINATES](#section-layer-coordinates) section for details
about COORDS).  


**--query-provider**     COORDS  
Attention query provider. If current layer is
not an attention layer, current layer
will be set as provider of the layer defined by
COORDS. (See [LAYER COORDINATES](#section-layer-coordinates) section for
details about COORDS).  


**--recurrent-layer**  
Recurrent layer mode.  


**--stride**     STRIDE  
Convolutional stride (default: 1).  


**--value-provider**     COORDS  
Attention values provider.  

(See [LAYER COORDINATES](#section-layer-coordinates) section
for details about COORDS).  


**--weight-init-mode**     MODE  
Weight initialization mode:  

auto,random,zero (default: auto).  


**--whole-sequence**  
Whole sequence mode.  






### LAYER COORDINATES

Format: [model_index:]layer_index   
Examples:  


**1:2**  
    - Third layer (2) of second model(1)   
**3**  
    - Fourth layer (3) of current model   




### LOG LEVELS

debug, info, notice, success, warn, error 

### METRICS

- accuracy 

### TRAIN|TEST OPTIONS



**--cifar**     [CLASSES]  
Dataset format is CIFAR.  

(classes: 10 or 100, default: 10).  


**--max-images**  
Max images to load (CIFAR).  


**--max-files**  
Max files to load (CIFAR).  


**--mnist**  
Dataset format is MNIST.  






### IMAGE OPTIONS



**--background-color**     COLOR  
Padding background color (ie. none, white, ...),
default: white

**--dump-image**     FILE  
Save image to file.  


**--grayscale**  
Convert image to grayscale.  


**--invert**  
Invert image pixels.  






### CONFIG FILES

Configuration files can be loaded via the `-c` option (see above). Every option that can be passed to the command line can also be used inside configuration files by removing the dash prefix ('-' or '--'), for example:  
`layer` instead of `--layer` or `learning-rate` instead of `--learning-rate`.  
Option arguments can follow the option name by separating them with spaces and every option should be written in a separate line.  
The special `include` directive has the same effect of the `-c` option, and it loads another configuration file (ie. `include /path/to/config`).  
**Examples**

```ini
# This is an example of psycl configuration file
# Comment lines are prefixed with '#'

# Include other configuration files
include /path/to/other/config-file

# Define model
layer input 784
layer fully-connected 710
layer softmax 10

# Train
train /path/to/my/dataset

# Save it
save /path/to/save-model


```

```ini
# A convolutiona neural network

layer input 3072
layer convolutional
  region-size 5
  feature-count 16
  padding 2
  stride 1

layer pooling
  region-size 2
layer softmax 10

train /path/to/my/dataset


```




### SCRIPTS

By using options like `--on-batch-trained` or `--on-epoch-trained` it's possible to execute an arbitrary external script when such events happen.  
The script will eventually receive the following options:  
--event TYPE --name MODEL_NAME --epoch CURRENT_EPOCH --epochs TOT_EPOCHS --average-loss AVERAGE_LOSS --current-loss BATCH_LOSS --accuracy TRAINING_ACCURACY --learning-rate RATE.  
Optional options:  
--validation-loss VALIDATION_LOSS --validation-accuracy VALIDATION_ACCURACY --batch BATCH_NUM --example EXAMPLE_INDEX.  
The scripts can use special exit codes to force psycl aborting the training process:  


**- 3**  
    (PS_STATUS_ERROR)   
**- 5**  
    (PS_STATUS_ABORTED)   




### EXAMPLES

```shell
psycl --layer input 784 --layer fully-connected 30 --layer softmax 10 --train /path/to/my/dataset
```

Build a model with input size of 784, a fully connected layer of size 30, an output layer of type softmax and size 10 and train it with the provided dataset




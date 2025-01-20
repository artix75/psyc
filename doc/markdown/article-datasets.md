# Datasets

Datasets are collections of data and are generally used to train models. This
article talks about how datasets are used by PsyC.

## Uses

Datasets are usually used to train models (see [PSTrain](functions.md#pstrain)). They can also be
used as testing datasets or for validation purpose.
Tests are generally performed after training a model in order to compute the
model's accuracy (that is a measure of the percentage of successful predictions
over the total number of examples tested).
Usually, developers may want to test the model over a different dataset than the
one used for training.
Validation is similar to testing, but it's performed during training (usually
after every **epoch**).
Finally, data used as input for neural networks predictions has a very similar
structure to datasets used for training (see below).

## Structure

Basically, PsyC datasets are arrays of [PSFloat](types.md#psfloat) numbers. When they're used as
training datasets, they contain both inputs and expected predictions (usally
called *targets* or *labels* in classification tasks).
They can come in two flavours: standard datasets and sequence datasets.

### Standard Datasets

Standard datasets are quite simple since they represent the input data to be
forwarded to the model and (in training datasets) the expected outputs.
So every input sample (and, eventually, its related target), is independent.
This kind of datasets is made by a collection of inputs where each input array
is immediately followed by the related target array.
So, for instance, the popular [MNIST]([https://en.wikipedia.org/wiki/MNIST_database](https://en.wikipedia.org/wiki/MNIST_database)) dataset would be described by an array of inputs of size 784 (each MNIST image is made by 28x28 pixels) immediately followed by the target values of size 10 (MNIST predictions are composed of 10 classes), and, since the MNIST training dataset consists of 60,000 images, the final array would be composed of 47,640,000 [PSFloat](types.md#psfloat) elements.
While training models with standard datasets, PsyC automatically determines the
number of samples (individual inputs+targets) of the dataset itself by dividing
the number of dataset elements by the model's input size + model's output size.
So, with the MNISt dataset, the input layer would be of size 784 and the output
layer would be of size 10 and 47640000 / (784 + 10) actually results in 60,000
samples.
When working with classification models, the targets' size can be compressed by
using **onehot** targets: since the expected result is the index of the expected
"class" (in MNIST dataset is an index from 0 to 9 representing the predicted
digit while in image-classificatio models each index can represent the class, ie. dog, train, cat, ...), it's possible to directly use the index itself and reduce the target size to just one element.
In this cases, the [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) must be set into [flags](types.md#pstextparseroptions) of the output
layer (see [PSLayer](types.md#pslayer)).

Example:

```c
/* Let's build a very simple model with input size of 4 and and output size
 * of 2. */
PSModel *model = PSModelCreate("My symple model");
PSAddLayer(model, FullyConnected, 4, NULL);
PSAddLayer(model, FullyConnected, 10, NULL);
PSAddLayer(model, SoftMax, 2, NULL);

/* Let's make a dataset of only two inputs. */

PSFloat data[] = {
    -0.82, 1.2, 1.9, 0.2,    /* First input data of size 4 */
    0, 1,                    /* Expected predictions (targets) of size 2 for first input */,
    -0.8, -1.8, -0.45, -0.7, /* Second input data of size 4 */
    1, 0                     /* Expected predictions (targets) of size 2 for second input */
};

/* We have an array of 12 elements (2 inputs of 4 elements and 2 targets of
   2 elements, so 4+4+2+2 = 12).*/

```

### Sequence Datasets

Sequence datasets are composed of sequences of inputs, so in this case every
input sample is a sequence of inputs instead of a single input.
Sequence datasets are tipically used with recurrent neural networks or with
models that take sequences as inputs (ie. text generative models).
Dataset created from text are a typical example of this kind of dataset, since
texts are sequences of words where each word is dependent on the context of 
the whole sequence.
Targets (expected predictions) structure can vary depending on the model type.
In a *many-to-many* model, both inputs and targets are sequences: a typical
example of this kind of models are models used to predict the next word while
giving the model the input text.
In this case, the length of the input squence is equal to the length of the
target sequence.
By contrast, a *many-to-one* model only uses sequences for inputs, but not for
targets. An example of this kind of model is a model used for sentiment analysis
on texts.
Finally, a *sequence-to-sequence* model uses sequences for both inputs and
targets, but unline many-to-many models, the length of the input sequence may
differ from the length of the target sequence. An example of this kind of models
is a model used to translate text from one language to another.
PsyC usually automatically detects the model type by checking which of the
model's layer takes sequences as inputs. This is defined by the
[PS_FLAG_USE_SEQUENCES](macros.md#ps-flag-use-sequences) flag or by the [PS_FLAG_RECURRENT](macros.md#ps-flag-recurrent) flag (some layers
like **RNN**, [LSTM](types.md#pslayertype) or [GRU](types.md#pslayertype) layers automatically set the [PS_FLAG_RECURRENT](macros.md#ps-flag-recurrent)).
So, if a model has one of this flags in the input layer and in the output layer,
the model will be considered of type many-to-many ([ManyToMany](types.md#psrecurrentnetworkmode)). If the model
has one of those flag in the input layer, but not in the output layer, it will
be considered of type many-to-one ([ManyToOne](types.md#psrecurrentnetworkmode)).
Sequence-to-Sequence models needs to set the [PS_TRAINING_FLAG_SEQ2SEQ](macros.md#ps-training-flag-seq2seq) in the
training options ([PSTrainingOptions](types.md#pstrainingoptions)) passed to [PSTrain](functions.md#pstrain) function.

In order for PsyC to correctly handle sequences' data, the sequence length must precede the data itself.
So if the input size is 4 and the sequence is made of 2 inputs, they  would be
represented by 9 [PSFloat](types.md#psfloat) elements: the first element would be the sequence
length (2 in this case) and the following 8 elements would contain the two
inputs of size 4.

Finally, also the number of **total samples in the dataset must be explicit**
and must be declared as the first element of the dataset array.

For **many-to-many** models, the sequence length must be declared only once,
just before the input sequence, since the length of the target sequence is the
same. So, with the example before, the whole sample composed by an input
sequence of 2 inputs of size 4 and 2 expected results (targets) of size 2
would be composed by 13 elements: the first element for the sequence length,
the next 8 elements for the 2 inputs of size 4, and then, the next 4 elements
for the 2 target sequences of size 2.

This structure differs for **sequence-to-sequence** models, since the input
sequence length could differ from the output sequence length.
In this case the sequence length must be explicit for both the input and the
target. So, if we have an input sequence of 2 inputs of size 4 and 3 expected
results (targets) of size 2, the whole sample would be composed by 16 elements:
the first element would contain 2 (the input sequence length), followed by 8
elements (sequence of 2 inputs of size 4), followed by 3 (the target sequence
length), followed by 6 elements (sequence of 3 tragets of size 2).

Datasets made from texts are a typical example of **onehot** dataset, since
words are usually represented as indices of a vector that contains all the
possible words of the dataset's vocabulary.
In this case, the input size can be reduced to one and every input will just
contain the index of the word.
In order to work with **onehot** data, the [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) must be set into
the input layer.
Since in many cases the expected targets also represent words, the
[PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) flag can also be set on the output layer in order to reduce
the size of each target to one.
For example, if we have a vocabulary of 1000 words, the dataset for an input
sequence of 4 words in a **many-to-many** model would be composed by 9 elements:
the first element would contain 4 (the sequence length of both inputs and
targets) followed by 4 elements for the inputs and 4 elements for the targets (
where each element would be the onehot index of the word).
In a **sequence-to-sequence** model, with expected targets of 5 words, the
dataset would be composed by 11 elements: the first element would contain 4 (
the input sequence length), followed by the 4 elements of the input sequence,
followed by 5 (le target sequence length), followed by the 5 elements of the
target sequence.

Examples:

```c
/* A many-to-many recurrent model with onehot inputs. */

int vocab_size = 1000;
PSLayerDef onehot_ldef = {.flags = PS_FLAG_RECURRENT | PS_FLAG_ONEHOT};
PSModel *gru_model = PSModelCreate("GRU model (many-to-many)");
PSAddLayer(gru_model, FullyConnected, vocab_size, &onehot_ldef);
PSAddLayer(gru_model, GRU, 10, NULL); 
PSAddLayer(gru_model, SoftMax, vocab_size, &onehot_ldef); 

PSFloat gru_data[] = {
    2,              /* Total number of samples contained in the dataset */

    /* First sample */
    4,              /* Sequence length for both input sequence and targets sequence */
    4, 123, 982, 6, /* The input sequence */
    21, 62, 2, 340  /* The target sequence */
    
    /* Second sample */
    2,              /* Sequence length for both input sequence and targets sequence */
    427, 92,        /* The input sequence */
    2, 600          /* The target sequence */
};

PSTrainingOptions *train_opts = {
    .learning_rate = 0.01,
    .epochs = 100
};
int datalen = sizeof(gru_data) / sizeof(PSFloat);
PSTrain(gru_model, gru_data, datalen, NULL, 0, &train_opts);

```


```c
/* A sequence-to-sequence model with onehot inputs. */

int input_vocab_size = 1000, output_vocab_size = 500;
PSLayerDef onehot_ldef = {.flags = PS_FLAG_RECURRENT | PS_FLAG_ONEHOT};
PSModel *seq2seq_model = PSModelCreate("Sequence-to-Sequence Model (Encoder)"),
         decoder = NULL;
PSAddLayer(seq2seq_model, FullyConnected, vocab_size, &onehot_ldef);
PSAddLayer(seq2seq_model, GRU, 10, NULL); 

decoder = PSModelCreate("Decoder");
PSAddLayer(decoder, FullyConnected, output_vocab_size, &onehot_ldef);
PSAddLayer(decoder, GRU, 10, NULL); 
PSAddLayer(decoder, SoftMax, output_vocab_size, &onehot_ldef); 

PSAddModel(seq2seq_model, decoder, NULL);

PSFloat seq2seq_data[] = {
    2,                /* Total number of samples contained in the dataset */

    /* First sample */
    4,                /* Sequence length of the input sequence */
    4, 123, 982, 6,   /* The input sequence */
    3, /* Sequence length of the target sequence */
    21, 62, 2,        /* The target sequence */

    /* Second sample */
    2,                /* Sequence length of the input sequence */
    427, 92,          /* The input sequence */
    4,                /* Sequence length of the target sequence */
    2, 600, 921, 101  /* The target sequence */
};

PSTrainingOptions *train_opts = {
    .learning_rate = 0.01,
    .epochs = 100,
    .flags = PS_TRAINING_FLAG_SEQ2SEQ
};
int datalen = sizeof(seq2seq_data) / sizeof(PSFloat);
PSTrain(seq2seq_model, seq2seq_data, datalen, NULL, 0, &train_opts);

```

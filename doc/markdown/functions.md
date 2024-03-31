# PsyC Documentation - 0.9.5
## Functions

### PSAbortTraining

In: psyc.h, line: 462

```c
void PSAbortTraining (PSModel *model)
```




### PSAdaDeltaOptimization

In: optimization.h, line: 45

```c
int PSAdaDeltaOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSAdaGradOptimization

In: optimization.h, line: 57

```c
int PSAdaGradOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSAdamOptimization

In: optimization.h, line: 69

```c
int PSAdamOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSAddCIFARInputLayer

In: dataset.h, line: 257

```c
PSLayer  * PSAddCIFARInputLayer (PSModel *model)
```




### PSAddConvolutionalLayer

In: psyc.h, line: 412

```c
PSLayer  * PSAddConvolutionalLayer (PSModel *model, PSLayerDef *ldef)
```

Add a convolutional layer to [model](types.md#pslayer). This function basically calls [PSAddLayer](functions.md#psaddlayer):  

```c
PSAddLayer(model, Convolutional, 0, ldef);

```


**RETURN VALUES**

See [PSAddLayer](functions.md#psaddlayer).


### PSAddDebugInfo

In: debug.h, line: 105

```c
void PSAddDebugInfo (PSModel *model, char *file, const char *func, int line, PSLayer *layer, void *neuron1, void *neuron2, char *prop, double val, ...)
```




### PSAddInputLayer

In: psyc.h, line: 411

```c
PSLayer  * PSAddInputLayer (PSModel *model, long size, PSLayerDef *ldef)
```

Add input layer of size [size](types.md#psvocabulary) to [model](types.md#pslayer). The layer type will be set to the default [FullyConnected](types.md#pslayertype) type.  
Special layer properties can be defined by the optional argument **layer_def**.  
If **layer_def** is **NULL**, the function will use the default layer configuration.  
See [PSAddLayer](functions.md#psaddlayer) for further details about adding layers.  


**RETURN VALUES**

The added layer or **NULL** if:  

 - [model](types.md#pslayer) is **NULL**.
 - [model](types.md#pslayer) is not empty.
 - See [PSAddLayer](functions.md#psaddlayer) for more failure reasons.


### PSAddLayer

In: psyc.h, line: 409

```c
PSLayer  * PSAddLayer (PSModel *model, PSLayerType type, long size, PSLayerDef *layer_def)
```

Add a new layer (instance of [PSLayer](types.md#pslayer)) of type [type](types.md#pslayer) and size [size](types.md#psvocabulary) to [model](types.md#pslayer). Special layer properties can be defined by the optional argument **layer_def**.  
If **layer_def** is **NULL**, the function will use the default layer configuration.  
The member [load_from](types.md#pslayerdef) of **layer_def** can be used to load the new layer's parameters from a file.  
The new model will be automatically allocated and added to model layers.  


**NOTE**:  the new layer should never be freed directly. By freeing [model](types.md#pslayer) ([PSModelFree](functions.md#psmodelfree)), all model's layers will be automatically freed.  

**RETURN VALUES**

Pointer to the added layer or **NULL** if something goes wrong.  
Possible failure reasons:  

 - [model](types.md#pslayer) is **NULL**
 - [model](types.md#pslayer) is empty and [type](types.md#pslayer) is not [FullyConnected](types.md#pslayertype) (the first layer must be always of type FullyConnected).
 - The new layer cannot be allocated into memory or the model's [layers](types.md#psmodel) array cannot be resized.
 - The model's last layer is **NULL**.
 - The new layer cannot be initialized. The reason for the initialization failure can vary depending on the layer type.
 - The new layer is recurrent or the model is recurrent but the recurrent mode of all layers is not consistent. In order to build consistent recurrent models, one of the following feature must be satisfied:
   - All layers must be recurrent, or
   - first N layers are recurrent and the remaining layers are not      recurrent, or
   - first N layers are not recurrent the remaining layers are recurrent.

**SEE ALSO**

[PSModelFree](functions.md#psmodelfree)  



### PSAddModel

In: psyc.h, line: 404

```c
int PSAddModel (PSModel *parent, PSModel *model, PSModelLink *link)
```

Add [model](types.md#pslayer) to another model (**parent**), creating a chained, multi-model model.  
If **parent** is already member of a multi-model chain but it's not the chain head, the function will automatically find the actual chain head and it will append [model](types.md#pslayer) to the chain tail.  
The argument **link** allows setting the rules for data propagation (both forward propagation and bacpropagation) between [model](types.md#pslayer) and the model preceding it in the model chain:  

 - The [layer](types.md#psmodellink) member of **link** can be used to set the layer in [model](types.md#pslayer) that will receive inputs from previous model (in forward propagation) or that will back-propagate the error (delta) to previous model.
 - The [previous_layer](types.md#psmodellink) member of **link** can be used to set the layer in the previous model (the model in the chain that precedes [model](types.md#pslayer)) that will forward its outputs to [model](types.md#pslayer) (in forward propagation) or that will receive the error (deltas) from [model](types.md#pslayer) in backpropagation.

If **link** is **NULL**, the function will try to automatically determine it by searching for the first layer in [model](types.md#pslayer) whose size matches a layer in the previous model.  


**RETURN VALUES**

1 if [model](types.md#pslayer) is successfully added, 0 if something goes wrong.  
Possible failure reasons:  

 - [model](types.md#pslayer) is **NULL** or **parent** is **NULL** or both are **NULL**.
 - [model](types.md#pslayer) is already part of a multi-model chain.
 - **link** is **NULL** and it's not possible to automatically determine it.
 - **link** is not **NULL** but it's not valid, because:
    - `link->layer` is **NULL** or `link->previous_layer` is **NULL**.
    - size of `link->layer` differs from size of `link->previous_layer`.
 - Memory issues.

**SEE ALSO**

[PSGetModelAtIndex](functions.md#psgetmodelatindex), [PSModelChainLength](functions.md#psmodelchainlength), [PSModelChainHead](functions.md#psmodelchainhead), [PSModelChainTail](functions.md#psmodelchaintail), [PSModelChainContains](functions.md#psmodelchaincontains)  



### PSAddPoolingLayer

In: psyc.h, line: 413

```c
PSLayer  * PSAddPoolingLayer (PSModel *model, PSLayerDef *ldef)
```

Add a pooling layer to [model](types.md#pslayer). This function basically calls [PSAddLayer](functions.md#psaddlayer):  

```c
PSAddLayer(model, Pooling, 0, ldef);

```


**RETURN VALUES**

See [PSAddLayer](functions.md#psaddlayer).


### PSAddVectors

In: maths.h, line: 196

```c
PSFloat  * PSAddVectors (PSFloat *a, PSFloat *b, PSFloat *dest, long length, PSMathOpts *opts)
```

Add vector **b** to vector **a**. The argument [length](types.md#psdict) defines the length of **a** and **b**, so both **a** and **b** must contain at least [length](types.md#psdict) elements.  
The resulting vector will have the same length of **a** and **b** and each of its elements will be the sum of the corresponding element of **a** and **b** at the same index (`dest[i] = a[i] + b[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSAddVectorScalar

In: maths.h, line: 206

```c
PSFloat  * PSAddVectorScalar (PSFloat *a, PSFloat b, PSFloat *dest, long length, PSMathOpts *opts)
```

Add scalar **b** to vector **a**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the sum of the corresponding element of **a** at the same index and scalar value b (`dest[i] = a[i] + b`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSAutoregression

In: psyc.h, line: 445

```c
int PSAutoregression (PSModel *model, PSFloat *inputs, int randomized, PSSequenceSettings *sequence_settings)
```

Forward **inputs** to [model](types.md#pslayer) with autoregression mode. The model must take sequences as inputs and produce sequences as outputs.  
If [model](types.md#pslayer) is part of a multi-model chain, **inputs** are forwarded to the first layer of the first model of the chain and outputs are produced by the last layer of the last model of the chain.  
When the full input sequence as been forwarded, all subsequent outputs produced by the model (including the last outputs produced by the input sequence) are forwarded as the next inputs to the model itself.  
The size of the output layer must match the size of the input layer or, if the input layer has the [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) set, its [onehot_vector_size](types.md#pslayer).  
If the input layer has the [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) set, the index of the highest element of the produced outputs is forwarded to the input layer. In this case, if the **randomized** argument is true, a random index is generated by using the values of the outputs as a probability distribution.  
The iteration keeps forwarding outputs as the next inputs until one of the following events happens:  

 - The length of the whole output sequence produced (including the outputs produced by the original input sequence) reaches the maximum length defined into the (optional) argument [sequence_settings](types.md#psmodel) or by the default value of [PS_MAX_SEQUENCE_LENGTH](macros.md#ps-max-sequence-length).
 - The index of the highest value of the produced outputs matches the value of [end](types.md#pssequencesettings) in the optional argument [sequence_settings](types.md#psmodel). If the **randomized** argument is true, the random index generated by using outputs as a probability distribution is compared with [end](types.md#pssequencesettings). If [sequence_settings](types.md#psmodel) is **NULL** or the value of the [end](types.md#pssequencesettings) member is negative, the end-matching event is ignored.

**RETURN VALUES**

1 if the process succeeds or 0 if:  

 - [model](types.md#pslayer) is **NULL**
 - [model](types.md#pslayer) is not built.
 - The input layer doesn't take sequences as inputs and the output layer doesn't produce sequences as outputs.
 - The size of the output layer doesn't match the size of the input layer (or input layer's [onehot_vector_size](types.md#pslayer) if the input layer has the flag [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) set).
 - Something else in the forward process fails.

**SEE ALSO**

[PSForward](functions.md#psforward), [PSClassify](functions.md#psclassify), [PSClassifyImage](functions.md#psclassifyimage)  



### PSAxpy

In: blas.h, line: 53

```c
void PSAxpy (PSBLAS_int n, PSFloat alpha, PSFloat *x, PSBLAS_int incx, PSFloat *y, PSBLAS_int incy)
```




### PSBitmapClear

In: utils.h, line: 121

```c
void PSBitmapClear (PSBitmap bitmap)
```




### PSBitmapCopy

In: utils.h, line: 115

```c
int PSBitmapCopy (PSBitmap dst, PSBitmap src)
```




### PSBitmapCreate

In: utils.h, line: 114

```c
PSBitmap PSBitmapCreate (size_t size)
```




### PSBitmapDup

In: utils.h, line: 116

```c
PSBitmap PSBitmapDup (PSBitmap src)
```




### PSBitmapGetBit

In: utils.h, line: 119

```c
int PSBitmapGetBit (PSBitmap bitmap, uint64_t index)
```




### PSBitmapOp

In: utils.h, line: 122

```c
PSBitmap PSBitmapOp (PSBitmap a, PSBitmap b, PSBitmap dest, int op)
```




### PSBitmapRelease

In: utils.h, line: 117

```c
void PSBitmapRelease (PSBitmap bitmap)
```




### PSBitmapSetBit

In: utils.h, line: 120

```c
int PSBitmapSetBit (PSBitmap bitmap, uint64_t index, int val)
```




### PSBitmapSize

In: utils.h, line: 118

```c
size_t PSBitmapSize (PSBitmap bitmap)
```




### PSBLASCheckLimits

In: blas.h, line: 52

```c
int PSBLASCheckLimits (PSBLASErr *err, const char *argformat, ...)
```

Check variadic arguments integer values against ensuring that they're not greater than PSBLAS_MAX.  
Checked values can be either **long** scalars or **long** arrays.  
The mandatory **argformat** argument is a string used to specify how variadic arguments must be intepreted.  
The format string for every variadic argument is:  

 - zero or more of the following flag characters:
   - '@': indicates that the next variadic argument is an array of **long**
   - '$': indicates that the next variadic argument(s) are precedeed          by their names. Names are strings passed as variadic arguments just before the scalar and/or vector argument.
 - The number of the next variadic argument (scalar) or the length of the next array argument (if '@' was set):
   - As a fixed number string representation (between 1 and 9).
   - '*': indicates that argument count/array length must be read from         the next variadic argument (casted to **int**).

The optional **err** argument can be used to retrieve more info about the value that exceeds PSBLAS_MAX:  

  - [param](types.md#psblaserr) string is automatically set if argument names is provided via the '$' flag.
  - [param_pos](types.md#psblaserr) indicates the (1-based) position of the invalid argument.
  - [param_value](types.md#psblaserr) contains the value of the invalid argument that exceeded     PSBLAS_MAX.


**WARN**:  the count/length argument (if '*' is used in the format string) must always preceed the argument name (if '$' flag has also used in the format string).  

  
Examples:  

```c
PSBLASErr err = {0};
long a = 10, b = 1000;
long nums[3] = {10, 100, 1000};
int valid = PSBLASCheckLimits(&err, "2", a, b);
valid = PSBLASCheckLimits(&err, "@3", nums);
valid = PSBLASCheckLimits(&err, "2@3", a, b, nums);
valid = PSBLASCheckLimits(&err, "*@*", 2, a, b, 3, nums);
valid = PSBLASCheckLimits(&err, "$*$@*", 2, "a", a, "b", b, 3,
                          "numbers", nums);

```


  
**RETURN VALUES**

1 if all arguments are less or equal than PSBLAS_MAX, zero if one of the arguments exceeded PSBLAS_MAX.


### PSBLASErrorStr

In: blas.h, line: 51

```c
const char  * PSBLASErrorStr (PSBLASErr *err, const char ** param_names)
```

Get a string representing the BLAS error **err**. If `err->param` is **NULL**, the optional **param_names** argument can be used to pass the names of the parameters (determined by 1-based err->param_pos value).  


**NOTE**:  the returned string is a **static** string: it should never be freed and it's always overwritten by subsequent calls.  

**RETURN VALUES**

The error string of **NULL** if **err** is **NULL**.


### PSCalcIntStringLength

In: utils.h, line: 135

```c
size_t PSCalcIntStringLength (long long num)
```




### PSCatchFloatingPointExceptions

In: debug.h, line: 81

```c
int PSCatchFloatingPointExceptions (int except)
```




### PSClassify

In: psyc.h, line: 447

```c
long PSClassify (PSModel *model, PSFloat *inputs)
```

Forward **inputs** to [model](types.md#pslayer) and get the index of the maximum state from the output layer.  


**RETURN VALUES**

1 if the process succeeds or 0 if:  

 - [model](types.md#pslayer) is **NULL**
 - [model](types.md#pslayer) is not built.
 - The input layer doesn't take sequences as inputs and the output layer doesn't produce sequences as outputs.
 - The index of the maximum state could not be determined.
 - Something else in the forward process fails.

**SEE ALSO**

[PSForward](functions.md#psforward), [PSAutoregression](functions.md#psautoregression), [PSClassifyImage](functions.md#psclassifyimage)  



### PSClassifyImage

In: image-data.h, line: 23

```c
int PSClassifyImage (PSModel *model, char *filename, int grayscale, int invert, char* bgcolor, char* dump_file)
```



**SEE ALSO**

[PSForward](functions.md#psforward), [PSAutoregression](functions.md#psautoregression), [PSClassify](functions.md#psclassify)  



### PSCreateWord2VecTrainingData

In: embedding.h, line: 32

```c
PSFloat  * PSCreateWord2VecTrainingData (PSFloat *tokens, long token_count, long window_size, long vocabulary_size, int onehot, long *num_examples_ptr)
```




### PSCrossEntropyLoss

In: psyc.h, line: 476

```c
PSFloat PSCrossEntropyLoss (PSFloat *x, PSFloat *y, long size, long onehot_size)
```




### PSCumulativeSum

In: maths.h, line: 231

```c
long PSCumulativeSum (PSFloat *a, PSFloat *dest, long length)
```

Compute the cumulative sum on elements of vector **a** having length defined by [length](types.md#psdict). The results will be stored into the vector **dest** that must have at least the same length of **a**. The value of each element of the resulting vector will be the sum of the values of **a** up to the index of the current resulting vector element (ie. `dest[2] = a[0] + a[1] + a[2]`).  


**RETURN VALUES**

1 if the function is successfully executed or 0 if:  

 - **a** is **NULL** or **dest** is **NULL**.
 - [length](types.md#psdict) is zero or negative.


### PSDataFromText

In: dataset.h, line: 230

```c
PSFloat  * PSDataFromText (char *str, PSTextParserOptions *opts, long *datalen, PSVocabulary ** vocabulary)
```

Load a dataset (an array of [PSFloat](types.md#psfloat) numbers) from a string. Depending on the parsing mode, each token or character found in the string will be converted to a numeric representation of itself. The dataset can be used to train a model ([PSModel](types.md#psmodel)) or it can provide inputs to the model.  
Numeric representation of tokens/characters is defined by key-value pairs contained into **vocabulary** and indices (ids) related to token are cast to [PSFloat](types.md#psfloat).  
The function can use an existing vocabulary or create a new one from scratch.  
By default, the string will be parsed as a sequence of tokens ([PS_PARSER_MODE_TOKENS](macros.md#ps-parser-mode-tokens)) and each token will be converted to a number.  
The above behavior can be changed by using [PS_PARSER_MODE_CHARS](macros.md#ps-parser-mode-chars) in the (optional) **opts** argument (see [PSTextParserOptions](types.md#pstextparseroptions)).  
  
When [PS_PARSER_MODE_TOKENS](macros.md#ps-parser-mode-tokens) mode is used, tokens can be matched in two ways:  

 - By using a separator: in this case the string will be split by using the separators defined into [separator](types.md#pstextparseroptions) member of **opts**. If [separator](types.md#pstextparseroptions) is **NULL** or **opts** is **NULL**, the default separators will be those defined by the macro [PS_DEFAULT_TOKEN_SEPARATOR](macros.md#ps-default-token-separator).
 - By using a callback that let the developers to define their own logic for identifying and extracting tokens from the input string. The callback function can be set into the [match_token](types.md#pstextparseroptions) member of **opts**, and it's a function of type [PSTokenMatch](types.md#pstokenmatch). The callback receives a token to match and a pointer to an integer to store the length of the matched token. If the callback returns a non-zero value (true), it indicates a successful match and a token of the matched length will be extracted from the input string. In this case, the parsing position is moved forward by the matched length. If the callback returns 0, it signals that the token was not matched, and the parsing position will be advanced to the next byte.

  
If the flag [PS_PARSER_FLAG_ENCODE_ONLY](macros.md#ps-parser-flag-encode-only) is set, the resulting dataset will only consist of converted token values, with no further info or metadata about the dataset itself. Otherwise, other data is added to the dataset, such as sequence length, number of sequences, and so on.  
  
By default, the dataset is generated as a whole single sequence, and the sequence length is prepended as the first element of the dataset.  
In order to split text into multiple sequences, it's possible to use the properties of **opts**:  

 - If [sequence_length](types.md#pstextparseroptions) is greater than zero, the text will be split into into multiple fixed-length sequence having [sequence_length](types.md#pstextparseroptions) tokens.
 - If the [match_sequence_end](types.md#pstextparseroptions) callback is not **NULL**, it will be called with the current token and, if its return value is true, the token will be the ending token of the current sequence. It produces variable-length sequences.
 - If the [sequence_separator](types.md#pstextparseroptions) string is not **NULL** and the current token is equal to it, the current token will be the ending token of the current sequence. It produces variable-length sequences.

When the dataset is split into multiple sequences, the total number of sequence is set into the first element of the dataset itself.  
  
By default, no target data will be generated, unless the flag [PS_PARSER_FLAG_MAKE_TARGETS](macros.md#ps-parser-flag-make-targets) is set into [flags](types.md#pstextparseroptions) member of **opts** or a target dataset is specified into **opts** by using the [target_dataset](types.md#pstextparseroptions) and the [target_datalen](types.md#pstextparseroptions) members of **opts**.  
If no [target_dataset](types.md#pstextparseroptions) is provided, by default the target sequences will be generated by shifting the related input sequence by one. For example, if we have an the following sequence "hello world have a nice day" and a [sequence_length](types.md#pstextparseroptions) of five tokens, the input sequence will be "hello world have a nice" and the target sequence will be "world have a nice day".  
Incomplete sequences are dropped, but it's possible to "pad" them by using 'start' and 'end' tokens that can be set with the [PS_PARSER_FLAG_START_TOKEN](macros.md#ps-parser-flag-start-token) and [PS_PARSER_FLAG_END_TOKEN](macros.md#ps-parser-flag-end-token) flags into [flags](types.md#pstextparseroptions) member of **opts**.  
If not specified by [start_token](types.md#pstextparseroptions) and [end_token](types.md#pstextparseroptions) members of **opts**, the function will use the default string defined by [PS_DEFAULT_START_TOKEN](macros.md#ps-default-start-token) and [PS_DEFAULT_END_TOKEN](macros.md#ps-default-end-token).  
If a [target_dataset](types.md#pstextparseroptions) is provided, target sequences will be taken from it (the target dataset must contain at least the same number of sequences of the current dataset being generated).  


**WARN**:  The parsed string **str** may be modified during text parsing. By setting the [PS_PARSER_FLAG_PRESERVE_STRING](macros.md#ps-parser-flag-preserve-string) flag into `opts->flags`, the function will work on a copy of the string, preventing the original string from being altered.  

  
**ARGUMENTS**  

 - **str**: the (null-terminated) string to be parsed (mandatory).
 - **opts**: parsing options, it can be **NULL**.
 - **existing_data**: optional argument that can be used to append    parsed data to an existing dataset. WARN: Since data can be reallocated, always use the returned dataset after calling the function.
 - **datalen**: pointer to **int64_t** where the final length of the dataset    will be stored. If **existing_data** is not **NULL**, the address pointed by **datalen** must contain the current length of the existing dataset. If **NULL** is returned by the function, the pointed address will contain zero.
 - **vocabulary**: pointer to pointer to a [PSVocabulary](types.md#psvocabulary) struct. The    argument is mandatory and cannot be **NULL**. If the pointer pointed by **vocabulary** is **NULL**, a new [PSVocabulary](types.md#psvocabulary) will be allocated and it will be filled with parsed tokens|characters. If the pointer pointed by **vocabulary** points to an already existing vocabulary, its numeric values will be used for parsed tokens. If a token or character is not found in the existing vocabulary, it will be automatically added, unless [PS_PARSER_FLAG_READONLY_VOCAB](macros.md#ps-parser-flag-readonly-vocab) flag is set into **opts**.

**RETURN VALUES**

The dataset ([PSFloat](types.md#psfloat) array) or **NULL** is something goes wrong.


### PSDataFromTextFile

In: dataset.h, line: 232

```c
PSFloat  * PSDataFromTextFile (const char *filepath, PSTextParserOptions *opts, long *datalen, PSVocabulary ** vocabulary)
```

Load a dataset (an array of PSFloat numbers) from the text file found at **filepath**.  
For parsing options and other arguments, see [PSDataFromText](functions.md#psdatafromtext).  


**RETURN VALUES**

The dataset ([PSFloat](types.md#psfloat) array) or **NULL** is something goes wrong.


### PSDataLoad

In: dataset.h, line: 238

```c
PSFloat  * PSDataLoad (const char *filepath, long *datalen)
```

Load dataset from file located at **filepath**. Dataset is returned as an array of PSFloat elements whose length (number of elements) is stored into mandatory argument **datalen**.  
The file must be an ASCII file where every number of the dataset is written as a string representation of floating point numbers and separated by a comma character.  
Optionally, the whole dataset can be prefixed with its length written as a string representation of a decimal number followed by a colon separator caharcter (':').  
Example: 3:1.25,2,-0.15 (dataset of three elements 1.25, 2.0 and -0.15) Return value: the loaded dataset or **NULL** is somethign goes wrong.  
Possible failure reasons:  

 - Mandatory arguments **filepath** or **datalen** are **NULL**.
 - File is not found at **filepath**.
 - File at **filepath** cannot be opened or read.
 - Dataset cannot be allocated into memory.


### PSDataSave

In: dataset.h, line: 239

```c
int PSDataSave (const char *path, PSFloat *data, long len, int opts)
```

Save dataset [data](types.md#psmathopts) to the file located at **path**. The dataset must be an array of PSFloat elements whose length (number of elements) defined by argument **len**.  
By default, the dataset is saved as a comma-separated list of its values written as string representations of floating point numbers.  
The datasets itself is prefixed with its length written as a string representation of a decimal number followed by a colon separator caharcter (':').  
Example: 3:1.25,2,-0.15 (dataset of three elements 1.25, 2.0 and -0.15) If flag [PS_IO_BINARY_MODE](macros.md#ps-io-binary-mode) is set into **opts**, the dataset will be saved in binary format.  


**RETURN VALUES**

1 if datasets is successfully saved, 0 in case of failure.  
Possible failure reasons:  

 - Mandatory arguments **path** or [data](types.md#psmathopts) are **NULL**.
 - File at **path** cannot be opened for writing.
 - Some error occurs while writing to the file.


### PSDataSplit

In: dataset.h, line: 240

```c
int PSDataSplit (PSFloat *data, long datalen, float percentage, long input_size, long target_size, PSFloat ** left, PSFloat ** right, long *left_length, long *right_length, int opts)
```

Split [data](types.md#psmathopts) into two separated datasets. This function can useful to separate validation data used for testing models from data used for training them.  
The **datalen** argument must contain the length (number of elements) of the [data](types.md#psmathopts) array, while [input_size](types.md#psmodel) and **target_size** must contain the number of elements of each single input ([input_size](types.md#psmodel)) and each single target (**target_size**). For example, if the dataset consists of 28x28 images the value for [input_size](types.md#psmodel) must be 784 and if the targets are composed of ten classes (such in popular MNIST dataset), the value of **target_size** should be 10. If [data](types.md#psmathopts) contains no targets, the value of **target_size** must be zero.  
The size of the resulting datasets (called left and right dataset), are defined by the value of **percentage** that is the percentage of [data](types.md#psmathopts) that goes to the "left" datasets and must be expressed with a value between 0.0 and 1.0, so, for example, a percentage of 0.8 means that the left dataset will receive the 80% of the examples  from [data](types.md#psmathopts) and, consequently, the right dataset will receive the 20% of the examples from [data](types.md#psmathopts).  
The way the data is distributed between the two datasets can vary depending on the value of **opts**. By default, the examples at the beginning of [data](types.md#psmathopts) go to the left dataset until the selected percentage is reached and then the remaining examples go to the right dataset.  
If **opts** has the [PS_DATA_EVENLY_SPREAD](macros.md#ps-data-evenly-spread) flag enabled, data will be evenly distributed to both datasets in an uniform way.  
If **opts** has the [PS_DATA_SHUFFLE](macros.md#ps-data-shuffle) flag enabled, data will be randomly assigned to both datasets.  
If [data](types.md#psmathopts) is made up of sequences, the [PS_DATA_SEQUENCES](macros.md#ps-data-sequences) flag must be enabled into the **opts** argument. If input sequences and target sequences may have different lengths, the flag [PS_DATA_SEQ2SEQ](macros.md#ps-data-seq2seq) must be enabled into the **opts** argument.  
The addresses of the resulting datasets will be stored into the **left** and **right** arguments and their lengths (number of their respective elements) will be stored into the **left_length** and **right_length** arguments.  


**NOTE**:  if [data](types.md#psmathopts) has no sequences and none of [PS_DATA_ALWAYS_ALLOC](macros.md#ps-data-always-alloc), [PS_DATA_SHUFFLE](macros.md#ps-data-shuffle) and [PS_DATA_EVENLY_SPREAD](macros.md#ps-data-evenly-spread) flags are set, the function won't allocate the resulting datasets, unless [PS_DATA_ALWAYS_ALLOC](macros.md#ps-data-always-alloc). In this case, **left** will contain the pointer to the original address of [data](types.md#psmathopts) and **right** will contain the pointer to the first element of [data](types.md#psmathopts) that will belong to the right dataset. This means that the right dataset should **never be freed** by its own. In all the other cases, memory for both the left and the right dataset will be allocated and must be freed when not used anymore.  

**RETURN VALUES**

1 in case of success, 0 in case of failure.  
Possible failure reasons:  

 - at least one of [data](types.md#psmathopts), **left**, **right**, **left_length** or **right_length** is **NULL**.
 - **datalen** is zero.
 - [input_size](types.md#psmodel) is zero.
 - **opts** has the flags [PS_DATA_SEQUENCES](macros.md#ps-data-sequences) or [PS_DATA_SEQ2SEQ](macros.md#ps-data-seq2seq) enabled but **target_size** is zero.
 - **percentage** is less than or equal to 0 or greater than or equal to 1.
 - **opts** has the flags [PS_DATA_SEQUENCES](macros.md#ps-data-sequences) or [PS_DATA_SEQ2SEQ](macros.md#ps-data-seq2seq) enabled but the dataset contains no sequences (the value of the first element of [data](types.md#psmathopts) is less than or equal to zero.
 - There's no enough memory to be allocated for left and right datasets.


### PSDebug

In: log.h, line: 128

```c
void PSDebug (const char *format, ...)
```




### PSDeleteGradient

In: psyc.h, line: 450

```c
void PSDeleteGradient (PSGradient *gradient)
```




### PSDeleteGradientsChain

In: psyc.h, line: 452

```c
void PSDeleteGradientsChain (PSGradient *** gradients, PSModel *model)
```




### PSDeleteModelGradients

In: psyc.h, line: 451

```c
void PSDeleteModelGradients (PSGradient ** gradients, PSModel *net)
```




### PSDeleteNeuron

In: psyc.h, line: 441

```c
void PSDeleteNeuron (PSNeuron *neuron)
```




### PSDiagonalFlatten

In: maths.h, line: 250

```c
PSMatrix PSDiagonalFlatten (PSMatrix matrix)
```

Create a new squared matrix from source **matrix** having a shape with rows and columns equal to **matrix** length (matrix length x matrix length).  
The original values of **matrix** are distributed into the new matrix over a diagonal line starting from top-left side and ending to bottom-right side.  
Example:  

```c
PSFLoat vec[4] = {1, 2, 3, 4};
PSMatrix src = PSMatrixFromArray(vec, 2, 2, 2); // 2x2 matrix, total len = 4
PSMatrix new = PSDiagonalFlatten(src); // ->
// {1, 0, 0, 0,
//  0, 2, 0, 0,
//  0, 0, 3, 0,
//  0, 0, 0 ,4}

```


**RETURN VALUES**

The matrix or **NULL** if:  

 - **matrix** is **NULL** or **matrix** is empty.
 - The matrix cannot be allocated in memory.


### PSDiagonalFlattenVector

In: maths.h, line: 251

```c
PSMatrix PSDiagonalFlattenVector (PSFloat *vec, long len)
```

Create a matrix of shape **len**, **len** where values of vector **vec** having length defined by **len** are distributed over a diagonal line starting from top-left side and ending to bottom-right side.  
Example:  

```c
PSFLoat vec[4] = {1, 2, 3, 4};
PSDiagonalFlattenVector(vec, 4); // ->
// {1, 0, 0, 0,
//  0, 2, 0, 0,
//  0, 0, 3, 0,
//  0, 0, 0 ,4}

```


**RETURN VALUES**

The matrix or **NULL** if:  

 - **vec** is **NULL** or **len** is zero.
 - The matrix cannot be allocated in memory.


### PSDiagonalMask

In: maths.h, line: 249

```c
PSMatrix PSDiagonalMask (long size)
```

Create a matrix with shape [size](types.md#psvocabulary), [size](types.md#psvocabulary) diagonally filled with 1.0 from the top-left side to the bottom-right side, example:  

```c
PSDiagonalMask(4); // ->
// {1, 0, 0, 0,
//  1, 1, 0, 0,
//  1, 1, 1, 0,
//  1, 1, 1, 1}

```


**RETURN VALUES**

The matrix of **NULL** if:  

 - [size](types.md#psvocabulary) is less than 1
 - The matrix cannot be allocated in memory.


### PSDictClear

In: utils.h, line: 98

```c
void PSDictClear (PSDict *dict)
```

Delete all items from dictionary [dict](types.md#psdictiterator).


### PSDictCreate

In: utils.h, line: 97

```c
PSDict  * PSDictCreate (int flags)
```

Create a new PSDict dictionary.


### PSDictFree

In: utils.h, line: 109

```c
void PSDictFree (PSDict *dict)
```

Delete the dictionary and free it's allocated memory.


### PSDictGet

In: utils.h, line: 99

```c
PSDictItem  * PSDictGet (PSDict *dict, const char *key)
```

Get the item associated to [key](types.md#psdictitem) from dictionary [dict](types.md#psdictiterator), if any.  


**RETURN VALUES**

The item (PSDictItem) or **NULL**.


### PSDictGetItems

In: utils.h, line: 106

```c
PSDictItem  ** PSDictGetItems (PSDict *dict)
```

Return an array containing all items owned by dictionary [dict](types.md#psdictiterator).  
The size of the array is given by `dict->length`.  


**RETURN VALUES**

An array of [PSDictItem](types.md#psdictitem) containing all the values or **NULL** if                 something goes wrong.


### PSDictGetKeys

In: utils.h, line: 105

```c
const char  ** PSDictGetKeys (PSDict *dict)
```

Return an array containing all keys owned by dictionary [dict](types.md#psdictiterator).  
The size of the array is given by `dict->length`.  


**RETURN VALUES**

An array of strings containing all the keys or **NULL** if                 something goes wrong.


### PSDictGetOrSet

In: utils.h, line: 103

```c
PSDictItem  * PSDictGetOrSet (PSDict *dict, const char *key, PSDictValue val)
```

Return value **val** if it's already set for [key](types.md#psdictitem), elseway set it and return it.  


**RETURN VALUES**

The value associated with [key](types.md#psdictitem).


### PSDictGetPointer

In: utils.h, line: 100

```c
void  * PSDictGetPointer (PSDict *dict, const char *key)
```

Get the item associated to [key](types.md#psdictitem) in dictionary [dict](types.md#psdictiterator) as a pointer.  


**RETURN VALUES**

The item as a pointer or **NULL**.


### PSDictHasKey

In: utils.h, line: 101

```c
int PSDictHasKey (PSDict *dict, const char *key)
```

Check whether [dict](types.md#psdictiterator) has the key [key](types.md#psdictitem).  


**RETURN VALUES**

1 if [dict](types.md#psdictiterator) has [key](types.md#psdictitem), elseway 0.


### PSDictIteratorCreate

In: utils.h, line: 107

```c
struct PSDictIterator  * PSDictIteratorCreate (PSDict *dict)
```

Create a new iterator for dictionary [dict](types.md#psdictiterator). The dictionary will be allocated in memory, so it's up to the developer to free it as soon as it is no longer needed.  


**RETURN VALUES**

The iterator or **NULL** if something goes wrong.


### PSDictNext

In: utils.h, line: 108

```c
PSDictItem  * PSDictNext (PSDictIterator *iterator)
```

Iterate over the next item using **iterator**.  


**RETURN VALUES**

The next item ([PSDictItem](types.md#psdictitem)) or **NULL** if there are no more               items to iterate.


### PSDictRemove

In: utils.h, line: 104

```c
void PSDictRemove (PSDict *dict, const char *key)
```

Remove item associated to [key](types.md#psdictitem) from dictionary [dict](types.md#psdictiterator), if any.


### PSDictSet

In: utils.h, line: 102

```c
PSDictItem  * PSDictSet (PSDict *dict, const char *key, PSDictValue val)
```

Set value **val** for key [key](types.md#psdictitem) in dictionary [dict](types.md#psdictiterator). Unless flag [PSDICT_UPDATE_DISABLED](macros.md#psdict-update-disabled) is enabled in dictionary flags, value will be set even If [key](types.md#psdictitem) is already associated to another value.  


**RETURN VALUES**

The item ([PSDictItem](types.md#psdictitem)) associated to the [key](types.md#psdictitem) or **NULL**.


### PSDisableAcceleration

In: config.h, line: 77

```c
void PSDisableAcceleration (uint16_t *config, PSAcceleration acceleration)
```




### PSDivideScalarVector

In: maths.h, line: 214

```c
PSFloat  * PSDivideScalarVector (PSFloat b, PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Divide scalar **b** by vector **a**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the result of the division of the scalar value of **b** by the corresponding element of **a** ant the same index (`dest[i] = b / a[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSDivideVectors

In: maths.h, line: 202

```c
PSFloat  * PSDivideVectors (PSFloat *a, PSFloat *b, PSFloat *dest, long length, PSMathOpts *opts)
```

Divide vector **a** by vector **b**. The argument [length](types.md#psdict) defines the length of **a** and **b**, so both **a** and **b** must contain at least [length](types.md#psdict) elements.  
The resulting vector will have the same length of **a** and **b** and each of its elements will be the division of the corresponding element of **a** and **b** at the same index (`dest[i] = a[i] / b[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSDivideVectorScalar

In: maths.h, line: 212

```c
PSFloat  * PSDivideVectorScalar (PSFloat *a, PSFloat b, PSFloat *dest, long length, PSMathOpts *opts)
```

Divide vector **a** by scalar **b**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the result of the corresponding element of **a** at the same by the scalar value of **b** (`dest[i] = a[i] / b`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSDot

In: maths.h, line: 244

```c
int PSDot (PSMatrix a, PSMatrix b, PSFloat *dest, PSMathOpts *opts)
```

Performs matrix-matrix multiplication, matrix-vector multiplication, vector-matrix multiplication or vector-vector multiplication, depending on the value of **argtype** field in opts (default is matrix-matrix).  
The resulting vector is stored into **dest**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**NOTE**:  if **argtype** for both **a** and **b** is 'V', the function will compute the dot product of the two vectors, assuming that they have the same size.  

If you need to perform matrix multiplication on two PSFloat arrays, use [PSMatMul](functions.md#psmatmul) instead.  
**RETURN VALUES**

1 in case of success, 0 in case of failure.  
Possible failure reasons:  

 - **a** is **NULL** or **b** is **NULL** or **dest** is **NULL**.
 - Memory allocation failure.
 - **a** is vector or **b** is vector and the resulting length would be zero.
 - Both **a** and **b** are vectors but [vector_len](types.md#psmathopts) member of optional **opts** argument is zero or **opts** is **NULL**.


### PSDotMV

In: maths.h, line: 245

```c
int PSDotMV (PSMatrix a, PSFloat *b, PSFloat *dest, PSMathOpts *opts)
```

Perform matrix-vector multiplication by calling [PSDot](functions.md#psdot) and setting **argtype** member of **opts** to `argtype[0] = 'M', argtype[1] = 'V'`.  
See [PSDot](functions.md#psdot) for a more detailed description.  


**NOTE**:  **opts** argument is optional, and if given, it's never overwritten by the function since its values are copied to a local structure.  

**RETURN VALUES**

See [PSDot](functions.md#psdot).


### PSDotProduct

In: maths.h, line: 235

```c
PSFloat PSDotProduct (PSFloat *a, PSFloat *b, long length, PSMathOpts *opts)
```

Compute the dot product of vector **a** and vector **b**, both having length defined by [length](types.md#psdict).  
The dot product is the sum of the product of each element of **a** by the corresponding element of **b** at the same index (`a[0] * b[0] + a[1] * b[1] + ... + a[n] * b[n]`).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The resulting dot product (scalar) or zero if **a** is **NULL** or **b** is **NULL**.


### PSDotSquare

In: maths.h, line: 236

```c
PSFloat PSDotSquare (PSFloat *a, long length, PSMathOpts *opts)
```




### PSDotVM

In: maths.h, line: 246

```c
int PSDotVM (PSFloat *a, PSMatrix b, PSMatrix dest, PSMathOpts *opts)
```

Perform vector-matrix multiplication by calling [PSDot](functions.md#psdot) and setting **argtype** member of **opts** to `argtype[0] = 'V', argtype[1] = 'M'`.  
See [PSDot](functions.md#psdot) for a more detailed description.  


**NOTE**:  **opts** argument is optional, and if given, it's never overwritten by the function since its values are copied to a local structure.  

**RETURN VALUES**

See [PSDot](functions.md#psdot).


### PSDownloadFile

In: utils.h, line: 138

```c
int PSDownloadFile (const char *url, const char *dest_dir)
```

Try to download content from **url**. The file will be genrated into **dest_dir**.  
The functions tries to download the file by using **wget** or **curl command line utilities.  
If those utilities are not found, download will fail.  


**RETURN VALUES**

1 in case of success, elseway 0.


### PSEnableAcceleration

In: config.h, line: 76

```c
int PSEnableAcceleration (uint16_t *config, PSAcceleration acceleration)
```




### PSErr

In: log.h, line: 132

```c
void PSErr (const char *tag, const char *format, ...)
```




### PSErrNN

In: log.h, line: 133

```c
void PSErrNN (const char *tag, PSModel *model, PSLayer *layer, const char *format, ...)
```




### PSFillWithBlank

In: utils.h, line: 143

```c
void PSFillWithBlank (int line_length)
```




### PSFindLayerMaxState

In: psyc.h, line: 433

```c
int PSFindLayerMaxState (PSLayer *layer, PSFloat *max_p, long *index_p, ...)
```

Find max state value and the relative neuron index for layer [layer](types.md#psmodellink), and store them into **max_p** pointer (max state) and **index_p** pointer (index of neuron having maximum state value).  
At least **max_p** or **index_p** must be provided.  
If layer is recurrent, an extra argument for timestep must be provided as a variadic argument (as int).  
If timestep is negative, it will be used to read states in a reverse order (ie. -1 is last timestep, -2 is last timestep - 1, etc.).  
Timestep must be always in range of processed timesteps (hidden states), otherwise the function will fail.  


**RETURN VALUES**

1 in case of success, 0 in case of error.


### PSFloatEquals

In: maths.h, line: 255

```c
int PSFloatEquals (PSFloat a, PSFloat b, int precision)
```

Compare two floats **a** and **b**. Use **precision** to set precision tolerance.  
Lower precision leads to higher tolerance.  
By setting **precision** to zero, the two numbers must be perfectly equal (no precision tolerance at all).  


**RETURN VALUES**

1 if **a** and **b** equal, 0 if they differ.


### PSForward

In: psyc.h, line: 444

```c
int PSForward (PSModel *model, PSFloat *inputs)
```

Forward **inputs** to [model](types.md#pslayer). If [model](types.md#pslayer) is part of a multi-model chain, **inputs** are forwarded to the first layer of the first model of the chain.  
If the input layer doen't accept sequences as inputs, the **inputs** array's length must match the [size](types.md#psvocabulary) of the first layer.  
When the first layer takes sequences (if it has the flags [PS_FLAG_RECURRENT](macros.md#ps-flag-recurrent) or PS_FLAG_USE_SEQUENCES` set), the length of **inputs** should be the (input layer size * sequence length) + 1, and the first element of **inputs** should contain the length of the sequence.  


**RETURN VALUES**

1 if the process succeeds or 0 if:  

 - [model](types.md#pslayer) is **NULL**
 - [model](types.md#pslayer) is not built.
 - The input layer doesn't take sequences as inputs and the output layer doesn't produce sequences as outputs.
 - Something else in the forward process fails.

**SEE ALSO**

[PSClassify](functions.md#psclassify), [PSAutoregression](functions.md#psautoregression), [PSClassifyImage](functions.md#psclassifyimage)  



### PSGaussianRandom

In: maths.h, line: 125

```c
PSFloat PSGaussianRandom (PSFloat mean, PSFloat stddev)
```

Generate a random floating number from a gaussian distribution having mean defined by **mean** and standard deviation defined by **stddev**.  


**RETURN VALUES**

The random float number.


### PSGelu

In: activation.h, line: 45

```c
void PSGelu (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

GELU (Gaussian Error Linear Units) activation function for vectors.  
GELU is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
For info about GeLU:  
  [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)  
The equivalent function to be used with scalars is **PSGeLUS**.  
The derivative of this function is **PSGeLUDerivative**.


### PSGeluDerivative

In: activation.h, line: 50

```c
void PSGeluDerivative (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Computes the derivative of GELU activation function ([PSGelu](functions.md#psgelu)) for vectors.  
The derivative is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The equivalent function to be used with scalars is [PSGeluDerivativeS](functions.md#psgeluderivatives).


### PSGeluDerivativeS

In: activation.h, line: 39

```c
PSFloat PSGeluDerivativeS (PSFloat val)
```

Computes derivative for GELU activation function ([PSGeluS](functions.md#psgelus)). This function applies to scalar values, so it takes the scalar **val** as argument and returns a **PFloat** scalar.  
The equivalent function to be used with vectors/matrices is [PSGeluDerivative](functions.md#psgeluderivative).  


**RETURN VALUES**

GELU derivative scalar result.


### PSGeluS

In: activation.h, line: 36

```c
PSFloat PSGeluS (PSFloat val)
```

GELU (Gaussian Error Linear Units) activation function for scalars.  
It takes the scalar **val** as argument and returns a **PFloat** scalar.  
For info about GELU:  
  [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)  
The equivalent function to be used with vectors/matrices is [PSGelu](functions.md#psgelu).  
The derivative of this function is [PSGeluDerivativeS](functions.md#psgeluderivatives).  


**RETURN VALUES**

GELU scalar result.


### PSGemm

In: blas.h, line: 58

```c
void PSGemm (PSBLASOrder order, char trans_a, char trans_b, PSBLAS_int m, PSBLAS_int n, PSBLAS_int k, PSFloat alpha, PSFloat *a, PSBLAS_int lda, PSFloat *b, PSBLAS_int ldb, PSFloat beta, PSFloat *c, PSBLAS_int ldc)
```




### PSGemv

In: blas.h, line: 55

```c
void PSGemv (PSBLASOrder order, char trans, PSBLAS_int m, PSBLAS_int n, PSFloat alpha, PSFloat *a, PSBLAS_int lda, PSFloat *x, PSFloat incx, PSFloat beta, PSFloat *y, PSBLAS_int incy)
```




### PSGetAccelerationName

In: config.h, line: 78

```c
const char  * PSGetAccelerationName (PSAcceleration acceleration)
```




### PSGetAttentionEnabledProjections

In: attention.h, line: 55

```c
int PSGetAttentionEnabledProjections (PSLayer *layer)
```




### PSGetAttentionHeadCount

In: attention.h, line: 52

```c
int PSGetAttentionHeadCount (PSLayer *layer)
```




### PSGetAttentionProviders

In: attention.h, line: 53

```c
int PSGetAttentionProviders (PSLayer *layer, PSLayer ** query_provider, PSLayer ** keys_provider, PSLayer ** values_provider)
```




### PSGetAttentionScale

In: attention.h, line: 51

```c
PSFloat PSGetAttentionScale (PSLayer *layer)
```




### PSGetAttentionType

In: attention.h, line: 50

```c
PSAttentionType PSGetAttentionType (PSLayer *layer)
```




### PSGetAttentionTypeLabel

In: attention.h, line: 49

```c
const char  * PSGetAttentionTypeLabel (PSAttentionType type)
```




### PSGetCodeOptimizationLevel

In: config.h, line: 79

```c
int PSGetCodeOptimizationLevel (void)
```

Returns code optimization level (given by -O gcc option) as an integer.  
Returns -1 if optimization level is unknown.


### PSGetDropout

In: dropout.h, line: 24

```c
PSFloat PSGetDropout (PSLayer *dropout_layer)
```




### PSGetDropoutLayer

In: dropout.h, line: 23

```c
PSLayer  * PSGetDropoutLayer (PSLayer *parent_layer)
```




### PSGetElapsedTimeString

In: utils.h, line: 144

```c
char  * PSGetElapsedTimeString (time_t elapsed_us, int long_format)
```




### PSGetEmbeddingVocabularySize

In: embedding.h, line: 31

```c
long PSGetEmbeddingVocabularySize (PSLayer *layer)
```




### PSGetFirstRecurrentLayer

In: psyc.h, line: 470

```c
PSLayer  * PSGetFirstRecurrentLayer (PSModel *model)
```




### PSGetGRUCell

In: gru.h, line: 45

```c
PSGRUCell  * PSGetGRUCell (PSLayer *layer)
```




### PSGetHomeDirectory

In: utils.h, line: 126

```c
const char  * PSGetHomeDirectory (void)
```

Returns user HOME directory.


### PSGetLabelForType

In: psyc.h, line: 466

```c
char  * PSGetLabelForType (PSLayerType type)
```




### PSGetLastRecurrentLayer

In: psyc.h, line: 471

```c
PSLayer  * PSGetLastRecurrentLayer (PSModel *model)
```




### PSGetLayerByIndex

In: psyc.h, line: 419

```c
PSLayer  * PSGetLayerByIndex (PSModel *model, int layer_index, int model_index)
```

Get the layer at index [layer_index](types.md#psdebuginfo) in [model](types.md#pslayer). If [model](types.md#pslayer) is part of a multi-model chain, the argument **model_index** can be used to specify the index of the sub-model of the model chain.  
Both [layer_index](types.md#psdebuginfo) and **model_index** accept negative values: in this case, the index is calculated from the last layer/model, for example:  
if [layer_index](types.md#psdebuginfo) is -1 and [model](types.md#pslayer) has 5 layers, the actual index will be the last layer's index (4).  


**RETURN VALUES**

The layer at specified index/indices or **NULL** if:  

 - [model](types.md#pslayer) is **NULL**.
 - [model](types.md#pslayer) is a multi-model chain but it's broken/invalid.
 - [layer_index](types.md#psdebuginfo) is out-of-bounds.
 - **model_index** is out-of-bounds.


### PSGetLayerInputSize

In: psyc.h, line: 420

```c
long PSGetLayerInputSize (PSLayer *layer)
```

Determine the input size of [layer](types.md#psmodellink), depending on the size of its previous layer, if any.  
If previous layer has set the flag [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot), the function will determine the input size by previous layer's onehot vector size ([PSGetOneHotLayerVectorSize](functions.md#psgetonehotlayervectorsize)).  


**RETURN VALUES**

The input size of [layer](types.md#psmodellink) or zero if:  

 - [layer](types.md#psmodellink) is **NULL**.
 - [layer](types.md#psmodellink) has no previous layer.


### PSGetLayerInputWeightsCount

In: psyc.h, line: 421

```c
long PSGetLayerInputWeightsCount (PSLayer *layer, int per_neuron)
```




### PSGetLayerParametersCount

In: psyc.h, line: 415

```c
long PSGetLayerParametersCount (PSLayer *layer, int param_type)
```




### PSGetLayerTypeLabel

In: psyc.h, line: 467

```c
char  * PSGetLayerTypeLabel (PSLayer *layer)
```




### PSGetLSTMCell

In: lstm.h, line: 53

```c
PSLSTMCell  * PSGetLSTMCell (PSLayer *layer)
```




### PSGetMaxLogLevel

In: log.h, line: 137

```c
int PSGetMaxLogLevel (void)
```




### PSGetModelAtIndex

In: psyc.h, line: 395

```c
PSModel  * PSGetModelAtIndex (PSModel *entrypoint, int index)
```

Get the model at [index](types.md#psmodel) in the multi-model chain that contains the model **entrypoint**. If [index](types.md#psmodel) is negative, it will be counted from the end of the model chain (ie. -1 is the last model, or tail,  of the chain).  


**RETURN VALUES**

The model or **NULL** if:  

 - **entrypoint** is **NULL**
 - the model chain is broken
 - [index](types.md#psmodel) is out of bounds.

**SEE ALSO**

[PSModelChainLength](functions.md#psmodelchainlength), [PSModelChainHead](functions.md#psmodelchainhead), [PSModelChainTail](functions.md#psmodelchaintail), [PSModelChainContains](functions.md#psmodelchaincontains), [PSAddModel](functions.md#psaddmodel)  



### PSGetNeuron

In: psyc.h, line: 437

```c
PSNeuron  * PSGetNeuron (PSLayer *layer, long index, PSNeuron *neuron)
```




### PSGetNeuronDebugID

In: debug.h, line: 83

```c
char  * PSGetNeuronDebugID (PSNeuron *neuron, PSLayer *layer)
```




### PSGetNeuronInputWeights

In: psyc.h, line: 438

```c
PSFloat  * PSGetNeuronInputWeights (PSNeuron *neuron)
```




### PSGetNeuronState

In: psyc.h, line: 439

```c
PSFloat PSGetNeuronState (PSNeuron *neuron, ...)
```




### PSGetNextLayer

In: psyc.h, line: 417

```c
PSLayer  * PSGetNextLayer (PSLayer *layer)
```




### PSGetOneHotLayerVectorSize

In: psyc.h, line: 414

```c
long PSGetOneHotLayerVectorSize (PSLayer *layer)
```

Get the onehot vector size of [layer](types.md#psmodellink). If the flag [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) is not set into layer's flags, just return the layer size.


### PSGetOperatorLayerProviders

In: operator-layer.h, line: 35

```c
PSLayer  ** PSGetOperatorLayerProviders (PSLayer *layer, int *count)
```




### PSGetOperatorLayerType

In: operator-layer.h, line: 33

```c
PSOperatorType PSGetOperatorLayerType (PSLayer *layer)
```




### PSGetOperatorLayerTypeLabel

In: operator-layer.h, line: 34

```c
const char  * PSGetOperatorLayerTypeLabel (PSOperatorType operator)
```




### PSGetOutputLayer

In: psyc.h, line: 418

```c
PSLayer  * PSGetOutputLayer (PSModel *model)
```

Return the output layer (basically, the last layer) of [model](types.md#pslayer). If [model](types.md#pslayer) is part of a multi-model chain, the function will return the output layer of the output model (the last model) of the chain.  


**RETURN VALUES**

The output layer or **NULL** if:  

 - [model](types.md#pslayer) is **NULL** or [model](types.md#pslayer) has no layers.
 - [model](types.md#pslayer) is part of a multi-model chain, but the chain is broken.


### PSGetPositionalEncoding

In: positional-encoding.h, line: 24

```c
PSMatrix PSGetPositionalEncoding (long seqlen, long size, int base)
```




### PSGetPositionalEncodingBase

In: positional-encoding.h, line: 26

```c
int PSGetPositionalEncodingBase (PSLayer *layer)
```




### PSGetPositionalEncodingLength

In: positional-encoding.h, line: 25

```c
long PSGetPositionalEncodingLength (PSLayer *layer)
```




### PSGetPreviousLayer

In: psyc.h, line: 416

```c
PSLayer  * PSGetPreviousLayer (PSLayer *layer)
```




### PSGetRecurrentHiddenWeights

In: recurrent.h, line: 23

```c
PSMatrix PSGetRecurrentHiddenWeights (PSLayer *layer)
```




### PSGetRecurrentNeuronHiddenWeights

In: recurrent.h, line: 24

```c
PSFloat  * PSGetRecurrentNeuronHiddenWeights (PSNeuron *neuron)
```




### PSGetState

In: psyc.h, line: 427

```c
PSFloat PSGetState (PSLayer *layer, long index, ...)
```




### PSGetTerminalColumns

In: utils.h, line: 142

```c
unsigned int PSGetTerminalColumns (void)
```




### PSHandleSignals

In: psyc.h, line: 486

```c
void PSHandleSignals (PSSignalHandler shutdown_handler)
```




### PSInfo

In: log.h, line: 129

```c
void PSInfo (const char *format, ...)
```




### PSIsAccelerationAvailable

In: config.h, line: 74

```c
int PSIsAccelerationAvailable (PSAcceleration acceleration)
```




### PSIsAccelerationEnabled

In: config.h, line: 75

```c
int PSIsAccelerationEnabled (uint16_t config, PSAcceleration acceleration)
```




### PSIsCausalAttention

In: attention.h, line: 56

```c
int PSIsCausalAttention (PSLayer *layer)
```




### PSIsDirectory

In: utils.h, line: 125

```c
int PSIsDirectory (const char *path)
```

Checks whether **path** is a valid directory.


### PSIsFunctionAvailable

In: debug.h, line: 80

```c
int PSIsFunctionAvailable (const char *func)
```




### PSIsXTermColor256

In: log.h, line: 139

```c
int PSIsXTermColor256 (int always_check)
```




### PSIterateLossFunctions

In: psyc.h, line: 487

```c
size_t PSIterateLossFunctions ( *callback)
```




### PSLayerFree

In: psyc.h, line: 434

```c
void PSLayerFree (PSLayer *layer)
```

Free memory allocated for [layer](types.md#psmodellink) and all of its objects (ie. weights, states).  


**WARN**:  this function should be called only for layers not being part of any model, since by freeing models ([PSModelFree](functions.md#psmodelfree)), all their layers will be automatically freed.  


### PSLayerLoad

In: psyc.h, line: 407

```c
int PSLayerLoad (PSLayer *layer, const char *filepath)
```




### PSLayerOutputs

In: psyc.h, line: 429

```c
PSFloat  * PSLayerOutputs (PSLayer *layer)
```

Get the output values of [layer](types.md#psmodellink). If layer uses sequences (ie. it's recurrent or if has flag [PS_FLAG_USE_SEQUENCES](macros.md#ps-flag-use-sequences)), the function will pick the last states of the sequence, otherwise it will just return the layer [states](types.md#pslayer).  


**RETURN VALUES**

The output values of [layer](types.md#psmodellink) or **NULL** if:  

 - [layer](types.md#psmodellink) is **NULL**.
 - `layer->states` is **NULL**.


### PSLayerSave

In: psyc.h, line: 408

```c
int PSLayerSave (PSLayer *layer, const char *filepath, int opts)
```

Save [layer](types.md#psmodellink) to file located at **filepath**. By default, only the layer's trainable parameters (ie. weights, biases) are saved and the layer is saved in ASCII format.  
However, this behavior can be changed by setting the following flags into the **opts** argument:  

 - [PS_IO_BINARY_MODE](macros.md#ps-io-binary-mode): save the layer data in binary format.
 - [PS_IO_SAVE_DEFINITION](macros.md#ps-io-save-definition): also save layer's properties (ie. type, size, ...). This option cannot be used along with [PS_IO_BINARY_MODE](macros.md#ps-io-binary-mode).

**RETURN VALUES**

1 if the layer is saved, 0 if somethign goes wrong.  
Possible failure reasons:  

 - The [layer](types.md#psmodellink) argument is **NULL**.
 - Both [PS_IO_BINARY_MODE](macros.md#ps-io-binary-mode) and [PS_IO_SAVE_DEFINITION](macros.md#ps-io-save-definition) are set.
 - The file at **filepath** cannot be opened for writing.
 - Some error occurs qhile writing data.


### PSLayerStates

In: psyc.h, line: 428

```c
PSFloat  * PSLayerStates (PSLayer *layer, ...)
```

Grt the states (unit activation values) of [layer](types.md#psmodellink). If [layer](types.md#psmodellink) uses sequences (ie. it's recurrent or if has flag [PS_FLAG_USE_SEQUENCES](macros.md#ps-flag-use-sequences)), the function will also read the first variadic argument after [layer](types.md#psmodellink) that indicates the index of the states to retrieve inside the sequence.  
If the sequence index is negative, the function will retrieve the [initial_states](types.md#pslayer) of the layer, that are the initial values of the layer states before it has received the input sequence (the can be **NULL**).  


**RETURN VALUES**

The states of [layer](types.md#psmodellink) or **NULL** if:  

 - [layer](types.md#psmodellink) is **NULL**.
 - `layer->states` is **NULL**.
 - the index provided with the variadic argument is out-of-bounds (ie. it's equal or greater than the current sequence length).
 - the index provided with the variadic argument is negative but the layer has no [initial_states](types.md#pslayer).


### PSLineAppend

In: log.h, line: 146

```c
int PSLineAppend (int opts, char *format, ...)
```




### PSLineClear

In: log.h, line: 149

```c
void PSLineClear (int mode)
```




### PSLineEnd

In: log.h, line: 150

```c
void PSLineEnd (void)
```




### PSLineFill

In: log.h, line: 148

```c
int PSLineFill (void)
```




### PSLineStart

In: log.h, line: 145

```c
int PSLineStart (int opts, char *format, ...)
```




### PSLoadCIFARData

In: dataset.h, line: 255

```c
int PSLoadCIFARData (int type, int classes, const char *dataset_path, PSFloat ** data, int max_files, int max_examples)
```

Load the CIFAR dataset ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) from files.  
The dataset files must be in gzip format (.gz): dataset files must be in binary version and located into **dataset_path** directory.  
The dataset is allocated by the function itself and its pointer is stored into [data](types.md#psmathopts) pointer-to-pointer (it cannot be **NULL**). The length of the resulting dataset is returned by the function.  
The CIFAR dataset comes in two fashions:  

 - CIFAR-10:  each image can be classified with 10 classes.
 - CIFAR-100: each image can be classified with 100 classes.

The **classes** argument can be used to tell the function which kind of dataset is going to be loaded.  
The [type](types.md#pslayer) argument can be used to tell if the dataset is a training dataset ([PS_DATA_TYPE_TRAINING](macros.md#ps-data-type-training)) or a test dataset ([PS_DATA_TYPE_TEST](macros.md#ps-data-type-test)).  


**RETURN VALUES**

The length of the dataset (number of [PSFloat](types.md#psfloat) elements) or zero if some error occurs.  
Possibile errors:  

 - The data argument is **NULL**
 - The value for **class** is neither 10 not 100.
 - Dataset directory is **NULL**, or dataset files cannot be opened cannot be opened.
 - Dataset cannot be allocated into memory
 - Dataset file format is not valid


### PSLoadMNISTData

In: dataset.h, line: 249

```c
int PSLoadMNISTData (int type, const char *images_file, const char *labels_file, PSFloat ** data)
```

Load the MNIST dataset ([https://en.wikipedia.org/wiki/MNIST_database](https://en.wikipedia.org/wiki/MNIST_database)) from files.  
The dataset files must be in gzip format (.gz): images data must be loaded from **images_file** path and labels data must be loaded from **labels_file** path.  
The dataset is allocated by the function itself and its pointer is stored into [data](types.md#psmathopts) pointer-to-pointer (it cannot be **NULL**). The length of the resulting dataset is returned by the function.  
The [type](types.md#pslayer) argument can be used to tell if the dataset is a training dataset ([PS_DATA_TYPE_TRAINING](macros.md#ps-data-type-training)) or a test dataset ([PS_DATA_TYPE_TEST](macros.md#ps-data-type-test)).  


**RETURN VALUES**

The length of the dataset (number of [PSFloat](types.md#psfloat) elements) or zero if some error occurs.  
Possibile errors:  

 - The data argument is **NULL**
 - Dataset files are **NULL**, they don't exist or they cannot be opened.
 - Dataset cannot be allocated into memory
 - Dataset files are not in gzip format or some error occurs whil unzipping them.
 - The internal format of the dataset files format is not valid.


### PSLoadModel

In: psyc.h, line: 403

```c
PSModel  * PSLoadModel (const char* filename)
```

Load a new model from the file located at **filepath** into [model](types.md#pslayer).  
In order to load model's data into an already existing model, [PSModelLoad](functions.md#psmodelload) should be used instead.  
If the file defines a multi-model chain, the whole chain will be loaded.  


**RETURN VALUES**

1 if model is successfully loaded, 0 if:  

 - **filepath** is **NULL**.
 - The file at **filepath** could not be opened for reading.
 - The file at **filepath** is not a valid PsyC model file.
 - PsyC version is lower than version declared in the file.
 - Memory allocation issues.
 - Some error occurred while reading data from file.

**SEE ALSO**

[PSModelLoad](functions.md#psmodelload), [PSModelSave](functions.md#psmodelsave)  



### PSLogLevelByName

In: log.h, line: 136

```c
int PSLogLevelByName (const char *name)
```




### PSLogLevelName

In: log.h, line: 135

```c
const char  * PSLogLevelName (int level)
```




### PSLRegularization

In: optimization.h, line: 75

```c
int PSLRegularization (PSFloat l1, PSFloat l2, PSFloat *weights, PSFloat *wgradients, PSFloat *tmp, long len, PSFloat *l1_loss, PSFloat *l2_loss, int batches, int weight_decay, int acceleration)
```




### PSMakeDir

In: utils.h, line: 127

```c
int PSMakeDir (const char *path, int recursive)
```

Creates directory **path** is it does not exists. If **recursive** is 1, the function will try to also create intermediate paths if they don't exists, in a similar fashion to "mkdir -p".  
Return vale: 1 in case of success, elseway 0.


### PSMatMul

In: maths.h, line: 242

```c
int PSMatMul (PSFloat *a, PSFloat *b, PSFloat *dest, long m, long n, long k, PSMathOpts *opts)
```

Perform matrix multiplication between vectors (PSFloat arrays) **a** and **b**.  
If you need to perform matrix multiplication with involve at least one [PSMatrix](types.md#psmatrix), then use [PSMatrixProduct](functions.md#psmatrixproduct) (matrix-matrix), [PSMatrixProductMV](functions.md#psmatrixproductmv) (matrix-vector) or [PSMatrixProductVM](functions.md#psmatrixproductvm) (vector-matrix) instead.  
You can set matrix transposition using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the vector arguments you want to be transposed:  

 - opt->transpose = 1 (transpose **a**)

Results will be stored in **dest**, that must be at least **m** * **n** long.  
By default, data in result vector will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result vector.  
Other arguments:  

 - **m**: number of rows in **a** and result **dest**
 - **n**: number of columns in **b** and **dest**.
 - **k**: number of columns in **a** and rows in **n**.


**NOTE**:  if you set transposition for **a** or **b**, **m**,**n** and **k** will refer to rows and columns of the transposed matrix.  

**RETURN VALUES**

1 in case of success, 0 in case of failure.


### PSMatrixAdd

In: maths.h, line: 176

```c
int PSMatrixAdd (PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
```

Add matrix **a** to matrix **b**. Results are stored into matrix pointed by pointer **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opt**.  
Both matrices can be transposed using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)
 - opt->transpose = 2 (transpose matrix **b**)
 - opt->transpose = (1 | 2) (transpose both matrix **a** and **b**)

By default, data in result matrix will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result matrix.  
The function will take in account the shape of both matrices so the operation is performed in different ways depending on the shapes and the shapes' type (see [PSMatrixShape](functions.md#psmatrixshape) for more details about the shape type):  

 - If both **a** and **b** have the same shape or if both their shape types are vector-like shapes ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and both **a** and **b** have the same length (total number of values), every element of the resulting matrix will be the sum of every element of **a** and the corresponding element of **b**.
 - If only one of **a** or **b** has a vector-like shape ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and the other matrix has a matrix-like shape and the length of the vector-like matrix is the same of the last dimension of the other matrix, the resulting matrix will have the shape of the matrix with a matrix-like shape and the values from the vector-like matrix will be added to the values of the "rows" of the matrix-like matrix. For example: if **a** has a shape of 2,3 and **b** has a shape of 1,3, the result will be computed as a[0] + b and a[1] + b.
 - If **a** or **b** have a scalar-like shape ([PS_SHAPE_TYPE_SCALAR](macros.md#ps-shape-type-scalar)), the resulting matrix will have the shape of the non-scalar matrix with the scalar value of the scalar-like matrix (basically, its first and only element) added to the all the values of the non-scalar matrix.

**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **a** is **NULL** or **b** is **NULL** or **result** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Invalid shapes:
   - Shapes differ, and
   - neither **a** nor **b** have scalar-like shape, and
   - both **a** and **b** have vector-like shape but their total length differ
   - one of **a** or **b** has vector-like shape whose size differs from the matrix-like matrix last dimension.
 - Memory allocation failure.


### PSMatrixClear

In: maths.h, line: 191

```c
void PSMatrixClear (PSMatrix matrix)
```

Set all values of **matrix** to zero. If **matrix** is **NULL**, the function does nothing at all.


### PSMatrixCopy

In: maths.h, line: 189

```c
int PSMatrixCopy (PSMatrix src, PSMatrix dst)
```

Copy values of matrix **src** to matrix **dst**. Both **src** and **dst** must have the same shape.  


**NOTE**:  if **dst** owns a cached transposed version of itself, the cached version will be cleared. At the same time, if **dst** is the cached transposed version of another matrix, the cached version of the owner matrix will be cleared.  

**RETURN VALUES**

1 in case of success, 0 if:  

 - **src** is **NULL** or **dst** is **NULL**.
 - **src** and **dst** have different shapes.


### PSMatrixCreate

In: maths.h, line: 149

```c
PSMatrix PSMatrixCreate (PSFloat init_value, PSMatrixInitializer initializer, int ndims, ...)
```

Create a new matrix having number of dimensions defined by **ndims**. The shape of the matrix is given by variadic arguments that follow **ndims**.  
The argument [init_value](types.md#pslayerdef) can be used to define the initial value of the matrix numbers or, optionally, the **initializer** callback can be used to initialize the matrix values.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixCreateWithShape

In: maths.h, line: 151

```c
PSMatrix PSMatrixCreateWithShape (PSFloat init_value, PSMatrixInitializer initializer, int ndims, long *shape)
```

Create a new matrix having number of dimensions defined by **ndims** and shape defined by **shape**. The argument [init_value](types.md#pslayerdef) can be used to define the initial value of the matrix numbers or, optionally, the **initializer** callback can be used to initialize the matrix values.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixDim

In: maths.h, line: 160

```c
long PSMatrixDim (PSMatrix matrix, int dim)
```

Return the size of the dimension **dim** of **matrix**. If **dim** is out of bounds or if **matrix** is **NULL**, the function will return zero.


### PSMatrixDivide

In: maths.h, line: 179

```c
int PSMatrixDivide (PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
```

Divide matrix **a** from matrix **b**. Results are stored into matrix pointed by pointer **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opt**.  
Both matrices can be transposed using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)
 - opt->transpose = 2 (transpose matrix **b**)
 - opt->transpose = (1 | 2) (transpose both matrix **a** and **b**)

By default, data in result matrix will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result matrix.  
The function will take in account the shape of both matrices so the operation is performed in different ways depending on the shapes and the shapes' type (see [PSMatrixShape](functions.md#psmatrixshape) for more details about the shape type):  

 - If both **a** and **b** have the same shape or if both their shape types are vector-like shapes ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and both **a** and **b** have the same length (total number of values), every element of the resulting matrix will be the division of every element of **a** by the corresponding element of **b**.
 - If only one of **a** or **b** has a vector-like shape ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and the other matrix has a matrix-like shape and the length of the vector-like matrix is the same of the last dimension of the other matrix, the resulting matrix will have the shape of the matrix with a matrix-like shape and the values of the "rows" of the matrix-like matrix will be divided by the values of the vector-like matrix. For example: if **a** has a shape of 2,3 and **b** has a shape of 1,3, the result will be computed as a[0] / b and a[1] / b.
 - If **a** or **b** have a scalar-like shape ([PS_SHAPE_TYPE_SCALAR](macros.md#ps-shape-type-scalar)), the resulting matrix will have the shape of the non-scalar matrix with all the values of the non-scalar matrix divded by the scalar value of the scalar-like matrix (basically, its first and only element).

**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **a** is **NULL** or **b** is **NULL** or **result** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Invalid shapes:
   - Shapes differ, and
   - neither **a** nor **b** have scalar-like shape, and
   - both **a** and **b** have vector-like shape but their total length differ
   - one of **a** or **b** has vector-like shape whose size differs from the matrix-like matrix last dimension.
 - Memory allocation failure.


### PSMatrixDup

In: maths.h, line: 187

```c
PSMatrix PSMatrixDup (PSMatrix matrix)
```

Duplicate **matrix** by creating a new matrix having the same shape as **matrix** and by copying all values of **matrix** to the new matrix.  


**RETURN VALUES**

The new matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixDupShape

In: maths.h, line: 188

```c
PSMatrix PSMatrixDupShape (PSMatrix matrix)
```

Create a new (zero-filled) matrix having the same shape as **matrix**.  


**RETURN VALUES**

The new matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixEquals

In: maths.h, line: 190

```c
int PSMatrixEquals (PSMatrix a, PSMatrix b, int precision, int ignore_shape)
```

Compare two martrices **a** and **b** having [length](types.md#psdict) length. Use **precision** to set precision tolerance. Lower precision leads to higher tolerance.  
By setting **precision** to zero, the two vectors must be perfectly equal (no precision tolerance at all).  


**RETURN VALUES**

1 if **a** and **b** equal, 0 if they differ at some point.


### PSMatrixExpand

In: maths.h, line: 158

```c
PSMatrix PSMatrixExpand (PSMatrix src, long add, int keep_src)
```

Create a new matrix having the shape of **src** but with the first dimension increased by the value of **add**. The original values **src** will be copied to the new matrix, and all the new values belonging to thecexpanded dimension will be initialized to zero.  
If **keep_src** is zero, the original matrix **src** will be freed.  


**RETURN VALUES**

The new expanded matrix or:  

 - **src** itself if **add** is less that one.
 - **NULL** if **src** is **NULL**.
 - **NULL** if memory cannot be allocated.


### PSMatrixFlatten

In: maths.h, line: 181

```c
PSMatrix PSMatrixFlatten (PSMatrix matrix)
```

Create a new matrix that is the single-dimensioned, flatten version of **matrix**.  
For example, if **matrix** has a shape of (2,3), the resulting matrix will have a shape of (6).  


**RETURN VALUES**

The new flatten matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - Memory connot be allocated.


### PSMatrixFree

In: maths.h, line: 192

```c
void PSMatrixFree (PSMatrix matrix)
```

Free **matrix** by also deleting all its private data (including the cached transposed versiob of **matrix** if any).  
If **matrix** is **NULL**, the function will directly return.  


**WARN**:  this function should not be directly called on **matrix** if it's  the cached transposed version of another matrix (see [PSMatrixTranspose](functions.md#psmatrixtranspose)): in this case the function [PSMatrixResetTransposed](functions.md#psmatrixresettransposed) should be used instead.  

**SEE ALSO**

[PSMatrixResetTransposed](functions.md#psmatrixresettransposed)  



### PSMatrixFromArray

In: maths.h, line: 157

```c
PSMatrix PSMatrixFromArray (PSFloat *array, int ndims, ...)
```

Create a new matrix having number of dimensions defined by **ndims**. The shape of the matrix is given by variadic arguments that follow **ndims**.  
The values of the matrix will be initialized with values of **array**.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**WARN**:  the length of **array** must be at least the same of the length of the matrix, so if the matrix has two dimensions of shape [2, 3] (two rows with three columns), the provided array's length cannot be less than six.  

**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - **array** is **NULL**.
 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixGet

In: maths.h, line: 170

```c
PSFloat  * PSMatrixGet (PSMatrix matrix, int ndims, long *len, ...)
```




### PSMatrixLength

In: maths.h, line: 162

```c
long PSMatrixLength (PSMatrix matrix)
```

Get the total number of values belonging to **matrix** (ie. a matrix with shape (2,3) will return 6).  


**RETURN VALUES**

The total number of values belonging to **matrix** or zero if **matrix** is **NULL**.


### PSMatrixMultiply

In: maths.h, line: 177

```c
int PSMatrixMultiply (PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
```

Multiply matrix **a** by matrix **b**. Results are stored into matrix pointed by pointer **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opt**.  
Both matrices can be transposed using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)
 - opt->transpose = 2 (transpose matrix **b**)
 - opt->transpose = (1 | 2) (transpose both matrix **a** and **b**)

By default, data in result matrix will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result matrix.  
The function will take in account the shape of both matrices so the operation is performed in different ways depending on the shapes and the shapes' type (see [PSMatrixShape](functions.md#psmatrixshape) for more details about the shape type):  

 - If both **a** and **b** have the same shape or if both their shape types are vector-like shapes ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and both **a** and **b** have the same length (total number of values), every element of the resulting matrix will be the multiplication of every element of **a** by the corresponding element of **b**.
 - If only one of **a** or **b** has a vector-like shape ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and the other matrix has a matrix-like shape and the length of the vector-like matrix is the same of the last dimension of the other matrix, the resulting matrix will have the shape of the matrix with a matrix-like shape and the values from the vector-like matrix will be multiplied by the values of the "rows" of the matrix-like matrix. For example: if **a** has a shape of 2,3 and **b** has a shape of 1,3, the result will be computed as a[0] * b and a[1] * b.
 - If **a** or **b** have a scalar-like shape ([PS_SHAPE_TYPE_SCALAR](macros.md#ps-shape-type-scalar)), the resulting matrix will have the shape of the non-scalar matrix with all the values of the non-scalar matrix multiplied by the scalar value of the scalar-like matrix (basically, its first and only element).

**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **a** is **NULL** or **b** is **NULL** or **result** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Invalid shapes:
   - Shapes differ, and
   - neither **a** nor **b** have scalar-like shape, and
   - both **a** and **b** have vector-like shape but their total length differ
   - one of **a** or **b** has vector-like shape whose size differs from the matrix-like matrix last dimension.
 - Memory allocation failure.


### PSMatrixNumDims

In: maths.h, line: 159

```c
int PSMatrixNumDims (PSMatrix matrix)
```

Return the number of dimensions of **matrix**. If **matrix** is **NULL**, the function will return zero.


### PSMatrixPrint

In: maths.h, line: 169

```c
void PSMatrixPrint (PSMatrix matrix, const char *sep, int print_shape)
```

Print a string representation of **matrix** to the standard output.  
The optional argument **sep** can be used to specify the separator string for matrix values.  
If **sep** is **NULL**, the default separator is a comma (',').  
If **print_shape** is true, the matrix representation will be preceded by a header describing the shape of **matrix**.  
If **matrix** is **NULL**, the function will immediately return.


### PSMatrixPrintInfo

In: maths.h, line: 165

```c
void PSMatrixPrintInfo (PSMatrix matrix, const char *name, int newline)
```




### PSMatrixPrintShape

In: maths.h, line: 166

```c
void PSMatrixPrintShape (PSMatrix matrix, int newline)
```




### PSMatrixProduct

In: maths.h, line: 171

```c
int PSMatrixProduct (PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
```

Performs matrix-matrix multiplication between matrix **a** and matrix **b**.  
Results are stored into matrix pointed by **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The **opt** argument can be **NULL**.  
By default, function uses BLAS to compute the result. Anyway, if BLAS support is missing in PsyC build and **b** only has one dimension, function will try compute results by using [PSDotProduct](functions.md#psdotproduct).  
The acceleration method can be changed via the [acceleration](types.md#psmathopts) member of the optional **opt** argument.  
Matrices can be transpose by using the [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)
 - opt->transpose = 2 (transpose matrix **b**)
 - opt->transpose = (1 | 2) (transpose both matrix **a** and **b**)

By default, data in result vector will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opt**, result will be added to data already present in the result vector.  


**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **result** is **NULL** or **a** is **NULL** or **b** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Matrix aligment error.
 - Invalid result shape.
 - Matrix pointed by **result** is not **NULL** and its shape differs from resulting output shape.
 - Memory allocation failure.

**SEE ALSO**

[PSMatrixProductMV](functions.md#psmatrixproductmv), [PSMatrixProductVM](functions.md#psmatrixproductvm)  



### PSMatrixProductMV

In: maths.h, line: 172

```c
int PSMatrixProductMV (PSMatrix a, PSFloat *b, long len, PSFloat ** result, PSMathOpts *opts)
```

Performs matrix-vector multiplication between matrix **a** and vector **b**.  
Argument **len** must be the length of the vector **b**.  
Results are stored into vector pointed by pointer **result**. If pointer pointed by **result** is **NULL**, a new vector is automatically allocated by the function itself and its pointer will be stored into **result**.  
Length of **b** vector must equal matrix **a** second dimension.  
Length of result vector must equal matrix **a** first dimension.  
By default, function uses BLAS to compute the result. Anyway, if BLAS support is missing in PsyC build, function will compute results by using [PSDotProduct](functions.md#psdotproduct) as fallback.  
The acceleration method can be changed via the [acceleration](types.md#psmathopts) member of the optional **opt** argument.  
The matrix **a** can be transpose by using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)

By default, data in result vector will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result vector.  


**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **result** is **NULL** or **a** is **NULL** or **b** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Matrix aligment error.
 - Invalid result shape.
 - Matrix pointed by **result** is not **NULL** and its shape differs from resulting output shape.
 - Memory allocation failure.

**SEE ALSO**

[PSMatrixProduct](functions.md#psmatrixproduct), [PSMatrixProductVM](functions.md#psmatrixproductvm)  



### PSMatrixProductVM

In: maths.h, line: 174

```c
int PSMatrixProductVM (PSFloat *a, PSMatrix b, long len, PSMatrix *result, PSMathOpts *opts)
```

Performs vector-matrix multiplication between vector **a** and matrix **b**.  
Argument **len** must be the length of the vector **a**.  
Results are stored into matrix pointed by **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The function uses BLAS to compute the result. Anyway, if BLAS support is missing in PsyC build, function will compute results by using [PSDotProduct](functions.md#psdotproduct) as fallback.  
The acceleration method can be changed via the [acceleration](types.md#psmathopts) member of the optional **opt** argument.  
Matrix **b** can be transposed by using the  [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the operand arguments you want to be transposed:  

 - opt->transpose = 2 (transpose matrix **b**)

By default, data in result vector will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result vector.  


**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **result** is **NULL** or **a** is **NULL** or **b** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Matrix aligment error.
 - Invalid result shape.
 - Matrix pointed by **result** is not **NULL** and its shape differs from resulting output shape.
 - Memory allocation failure.

**SEE ALSO**

[PSMatrixProduct](functions.md#psmatrixproduct), [PSMatrixProductMV](functions.md#psmatrixproductmv)  



### PSMatrixRandom

In: maths.h, line: 155

```c
PSMatrix PSMatrixRandom (int ndims, ...)
```

Create a new matrix having number of dimensions defined by **ndims**. The shape of the matrix is given by variadic arguments that follow **ndims**.  
The values of the matrix will be initialized with random numbers from 0.0 to 1.0.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixResetTransposed

In: maths.h, line: 186

```c
void PSMatrixResetTransposed (PSMatrix matrix)
```

Invalidate and free the cached transposed version of **matrix**, if any (see [PSMatrixTranspose](functions.md#psmatrixtranspose)).


### PSMatrixReshape

In: maths.h, line: 180

```c
PSMatrix PSMatrixReshape (PSMatrix matrix, int num_dims, ...)
```

Create a new matrix that is the reshaped version of **matrix**. The new matrix will have the same values of **matrix** but a different shape having number of dimensions defined by **num_dims**. The new shape can be declared by using the variadic arguments after **num_dims**.  
The total number of elements given by the new shape must be equal to the total number of element of **matrix**, so, for example, reshaping a matrix with shape 2,3 to a matrix with shape 1,6 is valid and reshaping a matrix with shape 2,3,3 to a matrix of 1,18 or a matrix of 2,9 is also valid, but reshaping a matrix of 2,3 to a matrix of 1,3 is not valid.  
Result value: the new reshaped matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - **num_dims** is zero or negative.
 - **num_dims** is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions).
 - The total number of elements of the new matrix would differ from the total number of elements of **matrix**.
 - Memory cannot be allocated.


### PSMatrixShape

In: maths.h, line: 161

```c
int PSMatrixShape (PSMatrix matrix, long *shape)
```

Get the shape of **matrix** and store it into **shape** array. The **shape** array must be big enough to hold at least [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) elements.  
If **shape** is **NULL**, the function will just return the number of dimensions (so, the length of the shape array of **matrix**).  


**RETURN VALUES**

The number of dimensions of **matrix** or zero if **matrix** is **NULL**.


### PSMatrixShapeType

In: maths.h, line: 164

```c
int PSMatrixShapeType (PSMatrix matrix)
```

Get the shape type of **matrix**.  


**RETURN VALUES**

The shape type:  

 - [PS_SHAPE_TYPE_NONE](macros.md#ps-shape-type-none) if:
   - **matrix** is **NULL**.
   - Matrix's shape has zero dimensions.
 - [PS_SHAPE_TYPE_SCALAR](macros.md#ps-shape-type-scalar) if:
   - Matrix's shape has one dimension of size 1.
   - Matrix's shape has two dimensions, both of size 1.
 - [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col) if:
   - Matrix's shape has one dimension of size greater than 1.
   - Matrix's shape has two dimensions and the first dimension is greater than 1 but the second dimension is 1.
 - [PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) if:
   - Matrix's shape has two dimensions and the first dimension is 1 but the second dimension is greater than 1.
 - [PS_SHAPE_TYPE_MATRIX](macros.md#ps-shape-type-matrix)if:
   - All other cases.


### PSMatrixSplit

In: maths.h, line: 182

```c
PSMatrix  * PSMatrixSplit (PSMatrix matrix, long num_slices, int axis, PSMathOpts *opts)
```

Split **matrix** into smaller matrices whose number is defined by **num_slices**. The matrix will be split on the axis (dimension) defined by the **axis** argument.  
If the **axis** argument is negative, it will be counted from the last dimension of the shape of **matrix**: for example, an axis of -1 means the last dimension of the shape.  


**NOTE**:  this function currenlty works only if **matrix** has up-to two dimensions or if **matrix** has more than two dimensions but **axis** is the first dimension or the last dimensions (so it cannot be used to split a matrix with more than two dimensions by an intermediate axis).  

The optional **opts** argument can be used to change the default acceleration methods (by default, **PSGlobalAcceleration** is used).  
**RETURN VALUES**

An array of **num_slices** sub-matrices whose length is or **NULL** if:  

 - **matrix** is **NULL**.
 - **matrix** is empty.
 - **axis** is out of bounds.
 - **matrix** has more than two dimensions but **axis** is neither the first nor the last axis.
 - The value of **num_slices** would not lead to an equal division (`shape[axis] % num_slices != 0`).
 - Memory cannot be allocate.


**NOTE**:  it's up to the developer using this function to free both the sub-matrices (by using [PSMatrixFlatten](functions.md#psmatrixflatten)) and the returned array containing them.  


### PSMatrixStride

In: maths.h, line: 163

```c
long PSMatrixStride (PSMatrix matrix, int dim)
```

Get the stride of the dimension **dim** of **matrix**. For example, a matrix with shape (2,3) has a stride of 3 for dimension 0 while a matrix with shape (2,3,3) has a stride of 9 for dimension 0, 3 for dimension 1 and 1 for dimension 2.  


**RETURN VALUES**

The stride of dimension **dim** or zero if **matrix** is **NULL**.


### PSMatrixSubtract

In: maths.h, line: 178

```c
int PSMatrixSubtract (PSMatrix a, PSMatrix b, PSMatrix *result, PSMathOpts *opt)
```

Subtract matrix **b** from matrix **a**. Results are stored into matrix pointed by pointer **result**. If pointer pointed by **result** is **NULL**, a new matrix is automatically allocated by the function itself and its pointer will be stored into **result**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opt**.  
Both matrices can be transposed using [transpose](types.md#psmathopts) field in the **opt** argument. In that case, [transpose](types.md#psmathopts) will contain the (1-based) indices of the matrix arguments you want to be transposed:  

 - opt->transpose = 1 (transpose matrix **a**)
 - opt->transpose = 2 (transpose matrix **b**)
 - opt->transpose = (1 | 2) (transpose both matrix **a** and **b**)

By default, data in result matrix will be overwritten. Anyway, if [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add) is set as [store_mode](types.md#psmathopts) into **opts**, result will be added to data already present in the result matrix.  
The function will take in account the shape of both matrices so the operation is performed in different ways depending on the shapes and the shapes' type (see [PSMatrixShape](functions.md#psmatrixshape) for more details about the shape type):  

 - If both **a** and **b** have the same shape or if both their shape types are vector-like shapes ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and both **a** and **b** have the same length (total number of values), every element of the resulting matrix will be the subtraction of every element of **b** from the corresponding element of **a**.
 - If only one of **a** or **b** has a vector-like shape ([PS_SHAPE_TYPE_ROW](macros.md#ps-shape-type-row) or [PS_SHAPE_TYPE_COL](macros.md#ps-shape-type-col)) and the other matrix has a matrix-like shape and the length of the vector-like matrix is the same of the last dimension of the other matrix, the resulting matrix will have the shape of the matrix with a matrix-like shape and the values of the vector-like matrix will be subtracted from the values of the "rows" of the matrix-like matrix. For example: if **a** has a shape of 2,3 and **b** has a shape of 1,3, the result will be computed as a[0] - b and a[1] - b.
 - If **a** or **b** have a scalar-like shape ([PS_SHAPE_TYPE_SCALAR](macros.md#ps-shape-type-scalar)), the resulting matrix will have the shape of the non-scalar matrix with the scalar value of the scalar-like matrix (basically, its first and only element) subtracted from all the values of the non-scalar matrix.

**RETURN VALUES**

1 if operation succeeds, 0 if it fails.  
Possible failure reasons:  

 - **a** is **NULL** or **b** is **NULL** or **result** is **NULL**.
 - **a** has zero dimensions or **b** has zero dimensions.
 - Invalid shapes:
   - Shapes differ, and
   - neither **a** nor **b** have scalar-like shape, and
   - both **a** and **b** have vector-like shape but their total length differ
   - one of **a** or **b** has vector-like shape whose size differs from the matrix-like matrix last dimension.
 - Memory allocation failure.


### PSMatrixSwapAxes

In: maths.h, line: 185

```c
PSMatrix PSMatrixSwapAxes (PSMatrix matrix, int axis1, int axis2)
```

Create a new matrix by swapping axes of **matrix**. For example, swapping axes 0 and 1 of a matrix with shape of 2,3 would create a matrix with shape of 3,2 and swaping axes 1 and 2 of a matrix with shape 2,3,4 would create a matrix with a shape of 2,4,3.  
The axes to be swapped are defined by **axis1** and **axis2** arguments: by using a negative value for an axis, it will be counted from the last dimension of the shape, so, for example, swapping the axes -1 and -2 of a matrix with shape 2,3,4 would create a matrix with shape of 2,4,3.  
If both **axis1** and **axis2** refer to the same axis, the function will return a duplicated versiob of **matrix**.  


**NOTE**:  despite calling this function with the first and the last axis would have the same result of [PSMatrixTranspose](functions.md#psmatrixtranspose) in terms of matrix data and shape, the swapped matrix created by [PSMatrixSwapAxes](functions.md#psmatrixswapaxes) always is an independent matrix and not the cached tranposed matrix of **matrix**.  

**RETURN VALUES**

The new swapped matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - **axis1** is out of bounds or **axis2** is out of bounds.
 - Memory cannot be allocated.

**SEE ALSO**

[PSMatrixTranspose](functions.md#psmatrixtranspose)  



### PSMatrixTranspose

In: maths.h, line: 184

```c
PSMatrix PSMatrixTranspose (PSMatrix matrix, int rebuild, PSMathOpts *opts)
```

Transpose **matrix** by swapping its first dimension with its last dimension. For example, a matrix with a shape of 2,3 will be transposed to a matrix with shape of 3,2.  
The function won't modify **matrix** but it will create a new matrix that is the transposed version of **matrix**.  
The trasponsed matrix is cached in the private data of **matrix** so that subsequent calls of this function with the same **matrix** and with **rebuild** argument set to zero will directly return the cached transposed matrix without recomputing the transposition.  
The **rebuild** argument can be used to invalidate the cached transposed matrix forcing the function to rebuild it.  
If **matrix** already is the cached transposed matrix of another matrix, the function will directly return the source matrix of **matrix**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**WARN**:  The cached transposed matrix is automatically freed by freeing its owner (**matrix**) with [PSMatrixFree](functions.md#psmatrixfree), so it should not be freed directly. In order to free and reset the transposed cached matrix, the function [PSMatrixResetTransposed](functions.md#psmatrixresettransposed) should be called on **matrix**.  



**WARN**:  most of the functions who alter the original matrix (**matrix**) will also invalidate the cached transposed matrix if any. However, manually changing matrix's values would lead to inconsistency between the matrix and its transposed version so the transposed matrix should be invalidated with [PSMatrixResetTransposed](functions.md#psmatrixresettransposed) or rebuilt by calling [PSMatrixTranspose](functions.md#psmatrixtranspose) with **rebuild** argument set to true.  

**RETURN VALUES**

The transposed matrix or **NULL** if:  

 - **matrix** is **NULL**.
 - Memory canmot be allocated.

**SEE ALSO**

[PSMatrixSwapAxes](functions.md#psmatrixswapaxes), [PSMatrixResetTransposed](functions.md#psmatrixresettransposed), [PSMatrixFree](functions.md#psmatrixfree)  



### PSMatrixWithGaussianRandom

In: maths.h, line: 156

```c
PSMatrix PSMatrixWithGaussianRandom (PSFloat stddev, int ndims, ...)
```

Create a new matrix having number of dimensions defined by **ndims**. The shape of the matrix is given by variadic arguments that follow **ndims**.  
The values of the matrix will be initialized with random numbers from a gaussian distribution having zero mean and the standard deviation defined by **stddev**.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMatrixWrite

In: maths.h, line: 167

```c
size_t PSMatrixWrite (PSMatrix matrix, const char *sep, char bracket, int indent, FILE *out)
```

Write the string representation of **matrix** to file file stream **out**.  
If **out** is **NULL**, the string will be printed to the standard output by default.  
The optional **sep** argument can be used to specify the separator string for matrix's values: if **NULL**, the default separator is a comma (,).  
The optional **bracket** argument can be used to specify the type of brackets enclosing matrix's values, and only the opening bracket is accepted as a valid value:  

 - '[' to use '[' as opening bracket and ']' as closing bracket.
 - '(' to use '(' as opening bracket and ')' as closing bracket.
 - '{' to use '{' as opening bracket and '}' as closing bracket.

If **bracket** is set to zero, the default bracket is '['. Other values for **bracket** won't be accepted.  
The **indent** argument can be used to set indentation size (expressed in number of white spaces). If set to zero, no indentation will be used and the matrix string will be written in a single line.  


**RETURN VALUES**

The total number of bytes written or 0 if:  

 - **matrix** is **NULL**.
 - Invalid value for **bracket** (see above).


### PSMatrixZeros

In: maths.h, line: 154

```c
PSMatrix PSMatrixZeros (int ndims, ...)
```

Create a new, zero-filled, matrix having number of dimensions defined by **ndims**. The shape of the matrix is given by variadic arguments that follows **ndims**.  
If the matrix cannot be allocated, **errno** will be set to **ENOMEM**.  


**RETURN VALUES**

The allocated matrix or **NULL** if:  

 - the number of dimensions (**ndims**) is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) or less than one.
 - it's not possible to allocate the matrix in memory.


**WARN**:  the address pointed by the returned pointer should never be freed directly. The specific function [PSMatrixFree](functions.md#psmatrixfree) should be used instead.  


### PSMean

In: maths.h, line: 232

```c
PSFloat PSMean (PSFloat *a, long length, PSMathOpts *opts)
```

Compute the mean value of the elements of vector **a** having length defined by [length](types.md#psdict).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The mean value of vector **a** values or zero if **a** is **NULL**.


### PSModelBuild

In: psyc.h, line: 390

```c
int PSModelBuild (PSModel *model)
```

Build [model](types.md#pslayer) so that it can be used for training of for predictions.  
If the model is already built, the function will just return 1. In order to force rebuilding an already built model, [PSModelRebuild](functions.md#psmodelrebuild) should be used.  
The function will check the model's architecture and it will perfrom various actions on it:  

 - It will allocate and initialize all needed internal data.
 - It will set proper flags both on the model and the layers.
 - It will determine and set the eventual recurrent mode ([PSRecurrentNetworkMode](types.md#psrecurrentnetworkmode)) depening on model's architecture.
 - It will resolve eventual layer placeholders making them real layers.
 - If the loss function (member [loss](types.md#psmodel) of [model](types.md#pslayer)) is **NULL**, it will automatically determine it:
   - [PSCrossEntropyLoss](functions.md#pscrossentropyloss) will be used if the output layer is a SoftMax layer.
   - PSQuadraticLoss in all the other cases.
 - If [model](types.md#pslayer) is part of a multi-model chain, it will check and update all the chain properties.

**RETURN VALUES**

1 is [model](types.md#pslayer) is successfully built, 0 if:  

 - [model](types.md#pslayer) is **NULL**.
 - [model](types.md#pslayer) is empty (it contains no layers).
 - There was some memory allocation issue.
 - The structure of the [model](types.md#pslayer) is not valid (ie. some layer is **NULL**)
 - [model](types.md#pslayer) contains one or more layer placeholders and the function failed to resolve one of them.
 - [model](types.md#pslayer) (or one of its layers) handles sequences-at-once but some of its layers is recurrent.
 - [model](types.md#pslayer) (or one of its layers) is recurrent but some of its layers uses sequences-at-once.
 - [model](types.md#pslayer) is recurrent but both the input layer and the output layer are not.
 - [model](types.md#pslayer) is part of a multi-model chain but the chain is broken or not valid.

**SEE ALSO**

[PSModelIsBuilt](functions.md#psmodelisbuilt), [PSModelRebuild](functions.md#psmodelrebuild)  



### PSModelChainContains

In: psyc.h, line: 398

```c
int PSModelChainContains (PSModel *chain, PSModel *model)
```

Check whether [model](types.md#pslayer) is contained by the multi-model chain **chain**.  


**RETURN VALUES**

- 1 if [model](types.md#pslayer) is contained by **chain** or [model](types.md#pslayer) == **chain**
 - 0 if [model](types.md#pslayer) is not contained by **chain** or the chain is broken.

**SEE ALSO**

[PSGetModelAtIndex](functions.md#psgetmodelatindex), [PSModelChainLength](functions.md#psmodelchainlength), [PSModelChainHead](functions.md#psmodelchainhead), [PSModelChainTail](functions.md#psmodelchaintail), [PSAddModel](functions.md#psaddmodel)  



### PSModelChainHead

In: psyc.h, line: 396

```c
PSModel  * PSModelChainHead (PSModel *model)
```

Get the first model (head) of the multi-model chain that contains [model](types.md#pslayer).  
If [model](types.md#pslayer) is not a multi-model chain, the function will return the [model](types.md#pslayer) itself.  


**RETURN VALUES**

The first model of the chain or **NULL** if:  

 - [model](types.md#pslayer) is **NULL**
 - the chain is broken.

**SEE ALSO**

[PSGetModelAtIndex](functions.md#psgetmodelatindex), [PSModelChainLength](functions.md#psmodelchainlength), [PSModelChainTail](functions.md#psmodelchaintail), [PSModelChainContains](functions.md#psmodelchaincontains), [PSAddModel](functions.md#psaddmodel)  



### PSModelChainLength

In: psyc.h, line: 394

```c
int PSModelChainLength (PSModel *model)
```

Get the number of models in multi-model [model](types.md#pslayer).  


**RETURN VALUES**

The number of models or:  

   - 0 if [model](types.md#pslayer) is **NULL** or if the model chain is broken
   - 1 if [model](types.md#pslayer) is not a multi-model chain.

**SEE ALSO**

[PSGetModelAtIndex](functions.md#psgetmodelatindex), [PSModelChainHead](functions.md#psmodelchainhead), [PSModelChainTail](functions.md#psmodelchaintail), [PSModelChainContains](functions.md#psmodelchaincontains), [PSAddModel](functions.md#psaddmodel)  



### PSModelChainTail

In: psyc.h, line: 397

```c
PSModel  * PSModelChainTail (PSModel *model)
```

Get the last model (tail) of the multi-model chain that contains [model](types.md#pslayer).  
If [model](types.md#pslayer) is not a multi-model chain, the function will return the [model](types.md#pslayer) itself.  


**RETURN VALUES**

The last model of the chain or **NULL** if:  

 - [model](types.md#pslayer) is **NULL**
 - the chain is broken.

**SEE ALSO**

[PSGetModelAtIndex](functions.md#psgetmodelatindex), [PSModelChainLength](functions.md#psmodelchainlength), [PSModelChainHead](functions.md#psmodelchainhead), [PSModelChainContains](functions.md#psmodelchaincontains), [PSAddModel](functions.md#psaddmodel)  



### PSModelCheck

In: psyc.h, line: 392

```c
int PSModelCheck (PSModel *model)
```




### PSModelClone

In: psyc.h, line: 383

```c
PSModel  * PSModelClone (PSModel *model, int layout_only)
```




### PSModelCreate

In: psyc.h, line: 382

```c
PSModel  * PSModelCreate (const char* name)
```

Create a new, empty model. The optional argument [name](types.md#psmodel) can be used to give a name to the model.  


**NOTE**:  the model will duplicate the eventually provided [name](types.md#psmodel) and it will keep it inside its internal data. The duplicated string will be automatically freed by freeing the whole model ([PSModelFree](functions.md#psmodelfree)).  

**RETURN VALUES**

Pointer to the created model or **NULL** if memory could not be allocated for it.


### PSModelDumpDeltas

In: psyc.h, line: 400

```c
int PSModelDumpDeltas (PSModel *model, const char* filename)
```




### PSModelDumpStates

In: psyc.h, line: 399

```c
int PSModelDumpStates (PSModel *model, const char* filename)
```




### PSModelFree

In: psyc.h, line: 401

```c
void PSModelFree (PSModel *model)
```

Free [model](types.md#pslayer) and all its related objects (layers, data, ...). If the model is part of a multi-model chain, all models following [model](types.md#pslayer) will also be freed.  
The functions safely checks whether [model](types.md#pslayer) is **NULL** and it does nothing in this case.


### PSModelGetStatus

In: psyc.h, line: 387

```c
int PSModelGetStatus (PSModel *model)
```

Get the value of [status](types.md#psmodel) of [model](types.md#pslayer). If [model](types.md#pslayer) is part of a multi-model chain, the function will retrieve the status of the first model of the chain.  
Common status values are:  

 - [PS_STATUS_UNTRAINED](macros.md#ps-status-untrained)
 - [PS_STATUS_TRAINED](macros.md#ps-status-trained)
 - [PS_STATUS_TRAINING](macros.md#ps-status-training)
 - [PS_STATUS_VALIDATING](macros.md#ps-status-validating)
 - [PS_STATUS_PAUSED](macros.md#ps-status-paused)
 - [PS_STATUS_ABORTED](macros.md#ps-status-aborted)
 - [PS_STATUS_ERROR](macros.md#ps-status-error)

**RETURN VALUES**

The status of [model](types.md#pslayer) or 0 if [model](types.md#pslayer) is **NULL**.

**SEE ALSO**

[PSModelSetStatus](functions.md#psmodelsetstatus)  



### PSModelIsBuilt

In: psyc.h, line: 389

```c
int PSModelIsBuilt (PSModel *model)
```

Check whether [model](types.md#pslayer) is built (see: [PSModelBuild](functions.md#psmodelbuild)).  


**RETURN VALUES**

1 if the model is built, 0 if it's not built.

**SEE ALSO**

[PSModelBuild](functions.md#psmodelbuild), [PSModelRebuild](functions.md#psmodelrebuild)  



### PSModelLoad

In: psyc.h, line: 384

```c
int PSModelLoad (PSModel *model, const char* filepath)
```

Load model data (including layers and their parameters) from file located at **filepath** into [model](types.md#pslayer).  
This function requires an already existing model. In order to load a new model from scratch from, [PSLoadModel](functions.md#psloadmodel) should be used instead.  
If [model](types.md#pslayer) is empty (it has no layers), both the model structure and data such as layer parameters will be loaded into the model itself).  
If [model](types.md#pslayer) is not empty (it already has layers), only data such as layer parameters will be loaded into model. In this case, the structure of [model](types.md#pslayer) must match the structure declared by the file.  
If the file defines a multi-model chain, the whole chain will be loaded ( in this case, if [model](types.md#pslayer) is not empty, the [model](types.md#pslayer) chain structure must match the structure that has to be loaded from the file).  


**RETURN VALUES**

1 if [model](types.md#pslayer) is successfully loaded, 0 if:  

 - [model](types.md#pslayer) is **NULL** or **filepath** is **NULL**.
 - The file at **filepath** could not be opened for reading.
 - The file at **filepath** is not a valid PsyC model file.
 - PsyC version is lower than version declared in the file.
 - [model](types.md#pslayer) is not empty and its structure differs from the one declared by the file (ie. different number of layers or models, different layer types, and so on).
 - Some error occurred while reading data from file.

**SEE ALSO**

[PSModelSave](functions.md#psmodelsave), [PSLoadModel](functions.md#psloadmodel)  



### PSModelOutputs

In: psyc.h, line: 430

```c
PSFloat  * PSModelOutputs (PSModel *model)
```

Get the output values of the output (last) layer of [model](types.md#pslayer). The function basically calls [PSLayerOutputs](functions.md#pslayeroutputs) on the layer returned by [PSGetOutputLayer](functions.md#psgetoutputlayer).


### PSModelPrintInfo

In: psyc.h, line: 393

```c
void PSModelPrintInfo (PSModel *model)
```




### PSModelRebuild

In: psyc.h, line: 391

```c
int PSModelRebuild (PSModel *model)
```

Rebuild an already built [model](types.md#pslayer) by resetting its **built** state and calling [PSModelBuild](functions.md#psmodelbuild). If [model](types.md#pslayer) is not built, calling this function is the same as directly calling [PSModelBuild](functions.md#psmodelbuild).  


**RETURN VALUES**

See [PSModelBuild](functions.md#psmodelbuild).

**SEE ALSO**

[PSModelBuild](functions.md#psmodelbuild), [PSModelIsBuilt](functions.md#psmodelisbuilt)  



### PSModelSave

In: psyc.h, line: 385

```c
int PSModelSave (PSModel *model, const char* filepath)
```

Save [model](types.md#pslayer) to file located at **filepath**. The function will save both model's structure (ie layer propeties) and data (ie. parameters).  
If [model](types.md#pslayer) is part of a multi-model chain, the whole chain will be saved.  


**RETURN VALUES**

1 if [model](types.md#pslayer) is successfully saved, 0 if:  

 - [model](types.md#pslayer) is **NULL** or **filepath** is **NULL**.
 - [model](types.md#pslayer) is empty (it has no layers).
 - The file at **filepath** could not be opened for writing.
 - Some error occurred while writing data to file.

**SEE ALSO**

[PSModelLoad](functions.md#psmodelload), [PSLoadModel](functions.md#psloadmodel)  



### PSModelSetName

In: psyc.h, line: 386

```c
int PSModelSetName (PSModel *model, char *name)
```

Set the name of [model](types.md#pslayer) with the string provided with the argument [name](types.md#psmodel).  
If [name](types.md#psmodel) is **NULL** and [model](types.md#pslayer) already has a name, model's [name](types.md#psmodel) will be cleared.  


**NOTE**:  the model will duplicate the provided [name](types.md#psmodel) and it will keep it inside its internal data. The duplicated string will be automatically freed by freeing the whole model ([PSModelFree](functions.md#psmodelfree)). If [model](types.md#pslayer) already has a name, the original name will be automatically freed.  

**RETURN VALUES**

1 is name is successfully set or 0 if:  

 - [model](types.md#pslayer) is **NULL**.
 - memory cannot be allocated.


### PSModelSetStatus

In: psyc.h, line: 388

```c
void PSModelSetStatus (PSModel *model, int status, int *old)
```

Set [status](types.md#psmodel) as the status of [model](types.md#pslayer). The optional argument **old** can be used to retrieve the old status of [model](types.md#pslayer) before updating it with the value of [status](types.md#psmodel).  
If [model](types.md#pslayer) is part of a multi-model chain, hte new status will be set on all the models that are part of the model chain.  
Common used status values are:  

 - [PS_STATUS_UNTRAINED](macros.md#ps-status-untrained)
 - [PS_STATUS_TRAINED](macros.md#ps-status-trained)
 - [PS_STATUS_TRAINING](macros.md#ps-status-training)
 - [PS_STATUS_VALIDATING](macros.md#ps-status-validating)
 - [PS_STATUS_PAUSED](macros.md#ps-status-paused)
 - [PS_STATUS_ABORTED](macros.md#ps-status-aborted)
 - [PS_STATUS_ERROR](macros.md#ps-status-error)

**SEE ALSO**

[PSModelGetStatus](functions.md#psmodelgetstatus)  



### PSMultiplyVectors

In: maths.h, line: 200

```c
PSFloat  * PSMultiplyVectors (PSFloat *a, PSFloat *b, PSFloat *dest, long length, PSMathOpts *opts)
```

Multiply vector **a** by vector **b**. The argument [length](types.md#psdict) defines the length of **a** and **b**, so both **a** and **b** must contain at least [length](types.md#psdict) elements.  
The resulting vector will have the same length of **a** and **b** and each of its elements will be the multiplication of the corresponding element of **a** and **b** at the same index (`dest[i] = a[i] * b[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX, Accelerate Framework) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSMultiplyVectorScalar

In: maths.h, line: 204

```c
PSFloat  * PSMultiplyVectorScalar (PSFloat *a, PSFloat b, PSFloat *dest, long length, PSMathOpts *opts)
```

Multiply vector **a** by scalar **b**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the multiplication of the corresponding element of **a** at the same index by scalar value b (`dest[i] = a[i] * b`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSNesterovOptimization

In: optimization.h, line: 39

```c
int PSNesterovOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSNormalizedRandom

In: maths.h, line: 124

```c
PSFloat PSNormalizedRandom (void)
```

Generate a random floating number within a range of 0.0 and 1.0.  


**RETURN VALUES**

The random float number.


### PSNormalizeToken

In: dataset.h, line: 229

```c
char  * PSNormalizeToken (char *token, size_t len)
```




### PSNotice

In: log.h, line: 130

```c
void PSNotice (const char *format, ...)
```




### PSOneHotVector

In: utils.h, line: 145

```c
PSFloat  * PSOneHotVector (long index, long len)
```

Create a vector of length **len** where value at [index](types.md#psmodel) is one while all the other values contain zero.


### PSOuterProduct

In: maths.h, line: 247

```c
int PSOuterProduct (PSFloat *a, PSFloat *b, PSFloat *dest, long alen, long blen, PSMathOpts *opts)
```

Multiply every element of vector **a** (having **alen** length) by every element of vector **b** (having **blen** length) and store results into vector **dest** (whose length must be the product of **alen** by **blen**).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

**RETURN VALUES**

1 is the function succeeds or zero if:  

 - **a** is **NULL** or **b** is **NULL** or **dest** is **NULL**.
 - BLAS computation error if BLAS acceleration is used.


### PSPathJoin

In: utils.h, line: 129

```c
char  * PSPathJoin (int count, ...)
```

Joins multiple file path components into a single path string.  
Argument **count** is used to specify how many components will be consumed.  
Path components must be passed as variadic arguments.  


**RETURN VALUES**

String containing the joined path or **NULL** if something goes               wrong. Returned string is allocated into heap, so it's up               to the developer to free it as soon as it is no longer needed.


### PSPauseTraining

In: psyc.h, line: 461

```c
void PSPauseTraining (PSModel *model)
```




### PSPrint

In: log.h, line: 126

```c
void PSPrint (int level, const char *format, ...)
```




### PSPrintableLength

In: utils.h, line: 134

```c
int PSPrintableLength (const char *s)
```




### PSPrintSameLine

In: log.h, line: 142

```c
void PSPrintSameLine (char *format, ...)
```




### PSProgressBar

In: log.h, line: 143

```c
int PSProgressBar (long num, long tot, int style, int color, int flags, int maxlen, char *label)
```




### PSQuadraticLoss

In: psyc.h, line: 475

```c
PSFloat PSQuadraticLoss (PSFloat *x, PSFloat *y, long size, long onehot_size)
```




### PSRandomInt

In: maths.h, line: 126

```c
long PSRandomInt (long range, PSFloat *weights, PSMathOpts *opts)
```

Generate a random unsigned integer number within a range defined by argument **range** (between zero and **range** - 1).  
If the optional [weights](types.md#pslayer) argument is not **NULL**, it can be used as a probability distribution the affects the randomness of the result.  
In this case, [weights](types.md#pslayer) must be an array of [PSFloat](types.md#psfloat) whose length must be equal to **range**: each element of [weights](types.md#pslayer) represents the probability (weight) of its index to be generated (for example, the weights `{0.1, 0.7, 0.2}` with a range of 3 give a probability of 70% to number 1 to be generated).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The random integer number or -1 in case of error.


### PSRelu

In: activation.h, line: 44

```c
void PSRelu (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

ReLU (Rectified Linear Unit) activation function for vectors.  
ReLU is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
For info about ReLU:  
  [https://en.wikipedia.org/wiki/Rectifier_(neural_networks](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))  
  
The equivalent function to be used with scalars is [PSReluS](functions.md#psrelus).  
The derivative of this function is [PSReluDerivative](functions.md#psreluderivative).


### PSReluDerivative

In: activation.h, line: 49

```c
void PSReluDerivative (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Computes the derivative of ReLU activation function ([PSRelu](functions.md#psrelu)) for vectors.  
The derivative is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The equivalent function to be used with scalars is [PSReluDerivativeS](functions.md#psreluderivatives).


### PSReluDerivativeS

In: activation.h, line: 38

```c
PSFloat PSReluDerivativeS (PSFloat val)
```

Computes derivative for ReLU activation function ([PSReluS](functions.md#psrelus)). This function applies to scalar values, so it takes the scalar **val** as argument and returns a **PFloat** scalar.  
The equivalent function to be used with vectors/matrices is [PSReluDerivative](functions.md#psreluderivative).  


**RETURN VALUES**

ReLU derivative scalar result.


### PSReluS

In: activation.h, line: 35

```c
PSFloat PSReluS (PSFloat val)
```

ReLU (Rectified Linear Unit) activation function for scalars.  
It takes the scalar **val** as argument and returns a **PFloat** scalar.  
For info about ReLU:  
  [https://en.wikipedia.org/wiki/Rectifier_(neural_networks](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))  
The equivalent function to be used with vectors/matrices is [PSRelu](functions.md#psrelu).  
The derivative of this function is [PSReluDerivativeS](functions.md#psreluderivatives).  


**RETURN VALUES**

ReLU scalar result.


### PSResetDebugInfo

In: debug.h, line: 104

```c
void PSResetDebugInfo (void)
```




### PSResetLayerStateSequence

In: psyc.h, line: 423

```c
int PSResetLayerStateSequence (PSLayer *layer, long steps, int retain_previous)
```




### PSResetModelStateSequences

In: psyc.h, line: 425

```c
int PSResetModelStateSequences (PSModel *model, long steps, int retain_previous)
```




### PSResetTransposedWeights

In: psyc.h, line: 402

```c
void PSResetTransposedWeights (PSModel *model)
```




### PSRMSPropOptimization

In: optimization.h, line: 63

```c
int PSRMSPropOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSSetAttentionQueryProvider

In: attention.h, line: 57

```c
int PSSetAttentionQueryProvider (PSLayer *layer, PSLayer *provider)
```




### PSSetDefaultTrainingOptions

In: psyc.h, line: 468

```c
void PSSetDefaultTrainingOptions (PSTrainingOptions *options)
```




### PSSetDropout

In: dropout.h, line: 25

```c
void PSSetDropout (PSLayer *dropout_layer, PSFloat dropout)
```




### PSSetNeuronState

In: psyc.h, line: 440

```c
int PSSetNeuronState (PSNeuron *neuron, double state, ...)
```




### PSSetRecurrentNetworkMode

In: psyc.h, line: 469

```c
int PSSetRecurrentNetworkMode (PSModel *model, PSRecurrentNetworkMode mode)
```




### PSSetState

In: psyc.h, line: 431

```c
int PSSetState (PSLayer *layer, PSFloat state, long index, ...)
```




### PSSGDOptimization

In: optimization.h, line: 33

```c
int PSSGDOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSSigmoid

In: activation.h, line: 42

```c
void PSSigmoid (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Sigmoid activation function for vectors. Sigmoid is computed on vector **vec** of length **len** and stored into vector **dest**. If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
For info about sigmoid:  
  [https://en.wikipedia.org/wiki/Sigmoid_function](https://en.wikipedia.org/wiki/Sigmoid_function)  
The equivalent function to be used with scalars is [PSSigmoidS](functions.md#pssigmoids).  
The derivative of this function is [PSSigmoidDerivative](functions.md#pssigmoidderivative).


### PSSigmoidDerivative

In: activation.h, line: 46

```c
void PSSigmoidDerivative (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Computes the derivative of sigmoid activation function ([PSSigmoid](functions.md#pssigmoid)) for vectors. The sigmoid derivative is computed on vector **vec** of length **len** and stored into vector **dest**. If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The equivalent function to be used with scalars is [PSSigmoidDerivativeS](functions.md#pssigmoidderivatives).


### PSSigmoidDerivativeS

In: activation.h, line: 37

```c
PSFloat PSSigmoidDerivativeS (PSFloat val)
```

Computes derivative for sigmoid activation function ([PSSigmoidS](functions.md#pssigmoids)).  
This function applies to scalar values, so it takes the scalar **val** as argument and returns a **PFloat** scalar.  
The equivalent function to be used with vectors/matrices is [PSSigmoidDerivative](functions.md#pssigmoidderivative).  


**RETURN VALUES**

ReLU derivative scalar result.


### PSSigmoidS

In: activation.h, line: 34

```c
PSFloat PSSigmoidS (PSFloat val)
```

Sigmoid activation function for scalars. It takes the scalar **val** as argument and returns a **PFloat** scalar.  
For info about sigmoid:  
  [https://en.wikipedia.org/wiki/Sigmoid_function](https://en.wikipedia.org/wiki/Sigmoid_function)  
The equivalent function to be used with vectors/matrices is [PSSigmoid](functions.md#pssigmoid).  
The derivative of this function is [PSSigmoidDerivativeS](functions.md#pssigmoidderivatives).  


**RETURN VALUES**

Sigmoid scalar result.


### PSSoftmax

In: activation.h, line: 51

```c
void PSSoftmax (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Computes Softmax function on vector **vec** of length **len**. Result is stored into vector **dest**. If **dest** is **NULL**, result will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The Softmax function can be used to get the probability distribution from a series of numbers.  
For more info about Softmax:  
    [https://en.wikipedia.org/wiki/Softmax_function](https://en.wikipedia.org/wiki/Softmax_function)


### PSStateSequenceLength

In: psyc.h, line: 432

```c
long PSStateSequenceLength (PSLayer *layer)
```




### PSStdDev

In: maths.h, line: 234

```c
PSFloat PSStdDev (PSFloat *a, long len, PSMathOpts *opts)
```

Compute the standard deviation of the elements of vector **a** having length defined by [length](types.md#psdict).  
The standard deviation is the square root of the statistical variance (see [PSVariance](functions.md#psvariance)).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The variance of vector **a** values or zero if **a** is **NULL**.


### PSStringJoin

In: utils.h, line: 133

```c
char  * PSStringJoin (char ** strings, char *sep, int len)
```




### PSSubtractScalarVector

In: maths.h, line: 210

```c
PSFloat  * PSSubtractScalarVector (PSFloat b, PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Subtract vector **a** from scalar **b**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the result of the subtraction of the corresponding element of **a** ant the same index from value of **b** (`dest[i] = b - a[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSSubtractVectors

In: maths.h, line: 198

```c
PSFloat  * PSSubtractVectors (PSFloat *a, PSFloat *b, PSFloat *dest, long length, PSMathOpts *opts)
```

Subtract vector **b** from vector **a**. The argument [length](types.md#psdict) defines the length of **a** and **b**, so both **a** and **b** must contain at least [length](types.md#psdict) elements.  
The resulting vector will have the same length of **a** and **b** and each of its elements will be the subtraction of the corresponding element of **a** and **b** at the same index (`dest[i] = a[i] - b[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSSubtractVectorScalar

In: maths.h, line: 208

```c
PSFloat  * PSSubtractVectorScalar (PSFloat *a, PSFloat b, PSFloat *dest, long length, PSMathOpts *opts)
```

Subtract scalar **b** from vector **a**. The argument [length](types.md#psdict) defines the length of **a**.  
The resulting vector will have the same length of **a** and each of its elements will be the result of the subtraction of **b** from the corresponding element of **a** at the same index (`dest[i] = a[i] - b`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.

Some acceleration systems (ie. AVX) can use some storage modes to speed-up computation.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSTanhActivation

In: activation.h, line: 43

```c
void PSTanhActivation (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Tanh (hyperbolic tangent) activation function for vectors. Hyperbolic tangent is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The derivative of this function is [PSTanhDerivative](functions.md#pstanhderivative).


### PSTanhDerivative

In: activation.h, line: 48

```c
void PSTanhDerivative (PSFloat *vec, PSFloat *dest, long len, int acceleration)
```

Computes the derivative of tanh (hyperbolic tangent) activation function ([PSTanhActivation](functions.md#pstanhactivation)) for vectors. The derivative is computed on vector **vec** of length **len** and stored into vector **dest**.  
If **dest** is **NULL**, results will be stored into **vec** itself.  
For the [acceleration](types.md#psmathopts) argument, take a look at [PSAcceleration](types.md#psacceleration).  
The equivalent function to be used with scalars is [PSTanhDerivativeS](functions.md#pstanhderivatives).


### PSTanhDerivativeS

In: activation.h, line: 40

```c
PSFloat PSTanhDerivativeS (PSFloat val)
```




### PSTest

In: psyc.h, line: 463

```c
float PSTest (PSModel *model, PSFloat *test_data, long data_size, PSFloat *loss, PSTrainingOptions *options)
```

Test [model](types.md#pslayer) the against **test_data** dataset having length defined by the [data_size](types.md#pstraininginfo) argument.  
Tests are usualy performed on a different dataset than the one used for training in order to measure how the model performs on different data.  
This can be useful to determine undefitting (the model is not sufficiently trained) or overfitting (the model has been trained to much on the training dataset and it cannot generalize its predictions to different examples).  
Underfitting generally leads to lower performances in the training data, while overfitting generally leads to better performances on the training dataset than on the one used for testing.  
The function computes the accuracy of the predictions (the number of correct prediction with respect to the expected targets given by the dataset itself).  
In addition, the function can also compute the overall loss of the predictions made by using the pointer [loss](types.md#psmodel).  
The argument **opts** can be used to set the same training options used for training (ie. the [flags](types.md#pstextparseroptions)).  


**RETURN VALUES**

The accuracy of the predictions, where 1.0 means that all predictions were correct while 0.0 means that no prediction was correct.


### PSTrain

In: psyc.h, line: 455

```c
void PSTrain (PSModel *model, PSFloat *training_data, long data_size, PSFloat *test_data, long test_size, PSTrainingOptions *options)
```

Train [model](types.md#pslayer) over [training_data](types.md#pslayerdef). Training epochs, batch size, optimization, and other optimizer settings are defined into optional **options** argument.  
**ARGUMENTS**  

 - [model](types.md#pslayer): The neural model to be trained (mandatory)
 - [training_data](types.md#pslayerdef): an array of [PSFloat](types.md#psfloat) containing the tarining dataset    (ie. inputs, expected predictions)
 - [data_size](types.md#pstraininginfo): length of [training_data](types.md#pslayerdef) array.
 - **test_data**: optional dataset that can be used for testing (validation).
 - [test_size](types.md#pstraininginfo): length of **test_data** array.
 - **options**: optional training options (see [PSTrainingOptions](types.md#pstrainingoptions)). If **NULL**, the training process will use default options.

Training/test data layout:  

 - For normal feedforward models, the array must contain alternating inputs/predictions pairs, one pair for each example in the dataset. So, each training/test example pair must contain:     - Input values, having the same length of the model's input layer
     - Target values, having the same length of the model's output layer. If output layer has the [PS_FLAG_ONEHOT](macros.md#ps-flag-onehot) flag, predictions length muse be 1, and it must contain the index of the expected maximum state.   The total number of training examples is given by:     array size / (input_size + output_size)
 - For recurrent model or models using sequences, the layout of the dataset can have different forms. Regardless of that, the first element of the array must contain the total number of training/test sequences. For each training/test sequence, the sequence length must be specified. Different forms can be:
   - Many-to-many: the default mode for recurrent models that produce sequences having the same length of the input sequence. In this case, the first element of the sequence segment is the sequence length, followed by inputs/predictions pair.

If some error occurs, [PS_STATUS_ERROR](macros.md#ps-status-error) will be set on [model](types.md#pslayer) and the function will immediately exit.  
If [model](types.md#pslayer) is not built, the function will automatically try to build it by calling [PSModelBuild](functions.md#psmodelbuild).  
Possible failure reasons:  

 - [model](types.md#pslayer) is **NULL**
 - [model](types.md#pslayer) is not built and it cannot be build.
 - The learning rate is negative.
 - [model](types.md#pslayer) is part of a multi-model chain but the chain is broken or invalid.
 - [PS_TRAINING_FLAG_SEQ2SEQ](macros.md#ps-training-flag-seq2seq) is set into flags of **options** but the model's architecture is not valid for sequence-to-sequence mode (ie. the model does not use sequences at all).


### PSTrainingDebugDump

In: debug.h, line: 84

```c
void PSTrainingDebugDump (PSModel *model, char *fmt, ...)
```




### PSTrainingDebugDumpGradient

In: debug.h, line: 94

```c
void PSTrainingDebugDumpGradient (PSModel *model, int phase, const char *func, PSLayer *layer, int gradient_idx, int weight_size, int weight_idx, int is_avx, int avx_len)
```




### PSTrainingDebugDumpHeader

In: debug.h, line: 87

```c
void PSTrainingDebugDumpHeader (PSModel *model, long data_size, long test_size, int epochs, PSFloat learning_rate, long batch_size)
```




### PSTrainingDebugDumpStep

In: debug.h, line: 85

```c
void PSTrainingDebugDumpStep (PSDebugStepInfo *info, char *format, ...)
```




### PSTrainingProgressBar

In: psyc.h, line: 479

```c
void PSTrainingProgressBar (PSModel *model, int status, int epochs, long batches, PSFloat *loss, float *accuracy, PSFloat *test_loss, float *test_accuracy, time_t *elapsed)
```




### PSUTF8CodepointSize

In: utf8.h, line: 32

```c
int PSUTF8CodepointSize (uint32_t cp)
```




### PSUTF8Decode

In: utf8.h, line: 36

```c
uint32_t PSUTF8Decode (PSUTF8Char c)
```

from UTF-8 encoding to Unicode Codepoint


### PSUTF8Encode

In: utf8.h, line: 37

```c
PSUTF8Char PSUTF8Encode (uint32_t codepoint)
```

From Unicode Codepoint to UTF-8 encoding


### PSUTF8IsAlpha

In: utf8.h, line: 41

```c
int PSUTF8IsAlpha (PSUTF8Char uc)
```




### PSUTF8IsAlphaNum

In: utf8.h, line: 44

```c
int PSUTF8IsAlphaNum (PSUTF8Char uc)
```




### PSUTF8IsDigit

In: utf8.h, line: 40

```c
int PSUTF8IsDigit (PSUTF8Char uc)
```




### PSUTF8IsLower

In: utf8.h, line: 43

```c
int PSUTF8IsLower (PSUTF8Char uc)
```




### PSUTF8IsPunct

In: utf8.h, line: 39

```c
int PSUTF8IsPunct (PSUTF8Char uc)
```




### PSUTF8IsSpace

In: utf8.h, line: 38

```c
int PSUTF8IsSpace (PSUTF8Char uc)
```




### PSUTF8IsUpper

In: utf8.h, line: 42

```c
int PSUTF8IsUpper (PSUTF8Char uc)
```




### PSUTF8IsValidChar

In: utf8.h, line: 34

```c
int PSUTF8IsValidChar (PSUTF8Char c)
```




### PSUTF8Next

In: utf8.h, line: 35

```c
int PSUTF8Next (char *txt, PSUTF8Char *ch)
```




### PSUTF8StrLen

In: utf8.h, line: 31

```c
int PSUTF8StrLen (const char *s)
```




### PSUTF8StrNCpy

In: utf8.h, line: 33

```c
char  * PSUTF8StrNCpy (char *dest, const char *src, size_t n)
```




### PSUTF8ToLower

In: utf8.h, line: 46

```c
PSUTF8Char PSUTF8ToLower (PSUTF8Char uc)
```




### PSUTF8ToUpper

In: utf8.h, line: 45

```c
PSUTF8Char PSUTF8ToUpper (PSUTF8Char uc)
```




### PSVariance

In: maths.h, line: 233

```c
PSFloat PSVariance (PSFloat *a, long len, PSMathOpts *opts)
```

Compute the statistical variance of the elements of vector **a** having length defined by [length](types.md#psdict).  
The variance is the sum of the squared difference of the difference between each value of **a** and the mean value of **a**.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The variance of vector **a** values or zero if **a** is **NULL**.


### PSVectorAbs

In: maths.h, line: 220

```c
PSFloat  * PSVectorAbs (PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Compute the absolute value of every element of vector **a** having length defined by [length](types.md#psdict) (`dest[i] = abs(a[i])`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorClip

In: maths.h, line: 221

```c
PSFloat  * PSVectorClip (PSFloat *a, PSFloat min, PSFloat max, PSFloat *dest, long length, PSMathOpts *opts)
```

Clip values of vector **a** having length defined by [length](types.md#psdict) to minimum value defined by **min** and maximum value defined by **max** (`dest[i] = (a[i] < min ? min : (a[i] > max ? max : a[i]))`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorConvertToMatrix

In: maths.h, line: 258

```c
PSMatrix PSVectorConvertToMatrix (PSFloat *vec, long len, int ndims, long *shape)
```

Convert the vector **vec** of length **len** to a [PSMatrix](types.md#psmatrix). This function differs from [PSMatrixFromArray](functions.md#psmatrixfromarray) since it reallocates the vector in order to make room for the matrix header that will contain matrix's properties.  
So the vector is reallocated and its memory is moved by the size of the matrix header.  
It's possible to specify the matrix's shape by using the **ndims** argument and the **shape** argument:  

 - **ndims**: number of dimensions (axes) of the matrix shape.
 - **shape**: the shape itself.

If **shape** is **NULL** or **ndims** is zero, the function will use a default shape of {**len**} (if **ndims** is 0 or 1) or {1, **len**} (if **ndims** is 2).  
The function will fail if **shape** is **NULL** and **ndims** is greater than 2.  


**WARN**:  if the function succeeds, it's not possible to use the source vector **vec** anymore, since its data have been moved in memory and the original address could have been reallocated.  



**WARN**:  the vector **vec** must be an array of [PSFloat](types.md#psfloat) that was previously allocated (ie. by using [PSVectorCreate](macros.md#psvectorcreate), [PSVectorDup](functions.md#psvectordup), **malloc**, **calloc** or **realloc**). Using global/static arrays or arrays from the stack frame will lead to memory corruption.  

**RETURN VALUES**

The matrix or **NULL** if the function fails.  
Possible failure reasons:  

 - **vec** is **NULL**.
 - **len** is zero.
 - **ndims** is greater than [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions).
 - **shape** is **NULL** but **ndims** is greater than 2.
 - **len** mismatches **shape** (**len** must equals the product of shape axes).
 - Memory allocation failure

**SEE ALSO**

[PSMatrixFromArray](functions.md#psmatrixfromarray)  



### PSVectorDup

In: maths.h, line: 253

```c
PSFloat  * PSVectorDup (PSFloat *src, long length)
```

Duplicate vector **vec** having length defined by [length](types.md#psdict).  


**RETURN VALUES**

The duplicated vector or **NULL** is memory cannot be allocated.


### PSVectorEquals

In: maths.h, line: 256

```c
int PSVectorEquals (PSFloat *a, PSFloat *b, long length, int precision, long *index)
```

Compare two vectors **a** and **b** having [length](types.md#psdict) length. Use **precision** to set precision tolerance. Lower precision leads to higher tolerance.  
By setting **precision** to zero, the two vectors must be perfectly equal (no precision tolerance at all).  
Use [index](types.md#psmodel) pointer if you need to know the index of the first non-equal elements.  


**RETURN VALUES**

1 if **a** and **b** equal, 0 if they differ at some point.


### PSVectorExp

In: maths.h, line: 218

```c
PSFloat  * PSVectorExp (PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Compute base-e (Euler's number) exponential  on every element of vector **a** having length defined by [length](types.md#psdict).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorFill

In: maths.h, line: 237

```c
void PSVectorFill (PSFloat *vec, PSFloat val, long len, PSMathOpts *opts)
```

Fill vector **vec** having length defined by **len** with [value](types.md#psdictitem). The function will immediately return if **vec** is **NULL** or **len** is zero.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.


### PSVectorMapWithLimit

In: maths.h, line: 225

```c
PSFloat  * PSVectorMapWithLimit (PSFloat *a, PSFloat limit, PSFloat mapper, PSFloat *dest, long length, PSMathOpts *opts)
```

Map values of vector **a** having length defined by [length](types.md#psdict) with the value defined by **mapper**: values greater than **limit** will be represented with the value of **mapper**, while values equal or less than **limit** will be represented with negative value of **mapper** (`-(mapper)`); Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorMax

In: maths.h, line: 229

```c
PSFloat PSVectorMax (PSFloat *a, long *index, long length, PSMathOpts *opts)
```

Compute the maximum value among values of vector **a** having length defined by [length](types.md#psdict).  
The optional pointer [index](types.md#psmodel) can be used, if not **NULL**, to retrieve the index of the maximum value.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The maxium value in the vector **a**.


### PSVectorNeg

In: maths.h, line: 219

```c
PSFloat  * PSVectorNeg (PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Compute the negative value of every element of vector **a** having length defined by [length](types.md#psdict) (`dest[i] = -a[i]`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorPower

In: maths.h, line: 227

```c
PSFloat  * PSVectorPower (PSFloat *a, PSFloat exp, PSFloat *dest, long length, PSMathOpts *opts)
```

Raise each value of vector **a** having length [length](types.md#psdict) to power of **exp**.  
If the value of **exp** is 2, the function will just call [PSMultiplyVectors](functions.md#psmultiplyvectors) function, mutiplying **a** by itself.  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorPrint

In: maths.h, line: 239

```c
void PSVectorPrint (PSFloat *vec, long len, char* sep)
```

Print a string representation of vector **vec** having length of **len** to the standard output.  
The optional **sep** argument can be used to specify a separator string for vector's values (if **sep** is null, by default "," is used as separator).  
If **vec** is **NULL** the function will immediately return.


### PSVectorRandom

In: maths.h, line: 254

```c
PSFloat  * PSVectorRandom (long len)
```

Allocate a new vector having length defined by **len** and fill it with random values within a range of 0.0 and 1.0.  


**RETURN VALUES**

The allocated vector or **NULL** if memory cannot be allocated.


### PSVectorReduceSum

In: maths.h, line: 230

```c
PSFloat PSVectorReduceSum (PSFloat *a, long length, PSMathOpts *opts)
```

Compute the sum of all the elements of vector **a** having length defined by [length](types.md#psdict).  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**RETURN VALUES**

The sum of all the elements in the vector **a** or zero if **a** is **NULL**.


### PSVectorSplit

In: maths.h, line: 252

```c
PSFloat  ** PSVectorSplit (PSFloat *vec, long len, long num_slices)
```

Split vector **vec** having length defined by **len** into **num_slices** vectors.  
For example, a vector of 10 elements split into two slices will create two vectors of size 5.  


**RETURN VALUES**

An array of **num_slices** vectors (PSFloat *) or **NULL** if:  

 - **vec** is **NULL**.
 - **len** is zero or negative.
 - **num_slices** is zero or negative.
 - **len** / **num_slices** does not result in equal division.
 - Memory allocation issues.


### PSVectorSqrt

In: maths.h, line: 217

```c
PSFloat  * PSVectorSqrt (PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Compute square root on every element of vector **a** having length defined by [length](types.md#psdict).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorTanh

In: maths.h, line: 216

```c
PSFloat  * PSVectorTanh (PSFloat *a, PSFloat *dest, long length, PSMathOpts *opts)
```

Compute hyperbolic tangent (tanh) on every element of vector **a** having length defined by [length](types.md#psdict).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorThreshold

In: maths.h, line: 223

```c
PSFloat  * PSVectorThreshold (PSFloat *a, PSFloat min, PSFloat *dest, long length, PSMathOpts *opts)
```

Clip values of vector **a** having length defined by [length](types.md#psdict) to minimum value defined by **min** and maximum value of PSFloat ([PSFLOAT_MAX](macros.md#psfloat-max)).  
(`dest[i] = (a[i] < min ? min : (a[i] > PSFLOAT_MAX ? PSFLOAT_MAX : a[i]))`).  
Results are stored into the optional **dest** arguments. If **dest** is **NULL**, a new vector will be allocated and its address will be  returned by the function itself.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  
Aside from acceleration, **opts** can also be used to set the result storage mode (by using the [store_mode](types.md#psmathopts) member):  

 - [PS_STORE_MODE_ADD](macros.md#ps-store-mode-add): the result is added to the existing values of **dest**.
 - [PS_STORE_MODE_SUB](macros.md#ps-store-mode-sub): the result is subtracted from the existing values    of **dest**.


**WARN**:  by using [store_mode](types.md#psmathopts), acceleration will be currently disabled.  

**RETURN VALUES**

The pointer to the address of the vector containing results.  
If **dest** is not **NULL**, the return value is **dest** itself, but if **dest** is **NULL**, the return value is the address of the newly allocated vector.  
The function returns **NULL** if **dest** is **NULL** but the destination vector cannot be allocated in memory.


### PSVectorTranspose

In: maths.h, line: 240

```c
PSFloat  * PSVectorTranspose (PSFloat *vec, PSFloat *dest, int acceleration, int ndims, ...)
```

Create a transposed version of **vec**, considering it a matrix with a shape of **ndims** dimensions.  
Variadic arguments can be used to define the shape of the matrix (that must have max. [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions) dimensions).  
If **dest** is not **NULL**, the transposed vector will be stored into memory pointed by **dest** itself.  
If **dest** is **NULL**, the resulting vector will be allocated by the function.  
The function can take advantage of the available accelerations (both hardwware and software). By default, accelerations set in **PSGlobalAcceleration** are used, if any. However, the used accelerations methods can be changed via the [acceleration](types.md#psmathopts) member of the optional argument **opts**.  


**NOTE**:   

 - The shape defined by the variadic dimensions refer to original shape of the vector (seen as a matrix) and not to the resulting transposed vector. So, if the defined shape is 2,3, the resulting shape will be 3,2.
 - If you need to transpose a [PSMatrix](types.md#psmatrix), directly use [PSMatrixTranspose](functions.md#psmatrixtranspose) instead.  

**RETURN VALUES**

A pointer to the transposed vector, whose size will be the product of all dimensions of the defined shape or **NULL** if something goes wrong.  
If **dest** is not **NULL**, return value will be **dest** or **NULL** if something goes wrong.  
If **ndims** is 1, the function will immediately return **vec** itself.  
Possible failure reasons:  

 - **ndims** is greater that [PS_MATRIX_MAX_DIMENSIONS](macros.md#ps-matrix-max-dimensions).
 - **ndims** is zero or less than zero.
 - One of the shape's dimension in the variadic arguments is zero or less than zero.
 - Memory allocation failure.


### PSVectorWrite

In: maths.h, line: 238

```c
size_t PSVectorWrite (PSFloat *vec, long len, char* sep, FILE *f)
```

Write a string representation of vector **vec** having length of **len** to the file stream **f**.  
The optional **sep** argument can be used to specify a separator string for vector's values (if **sep** is null, by default "," is used as separator).  
If **vec** is null or **f** is null, the function will immediately return.


### PSVLineAppend

In: log.h, line: 147

```c
int PSVLineAppend (int opts, char *format, va_list args)
```




### PSVocabularyAdd

In: dataset.h, line: 221

```c
long PSVocabularyAdd (PSVocabulary *vocabulary, char *token)
```

Add token **token** to **vocabulary**. The token is added to the internal dictionary of **vocabulary** and a numeric index (ID) is assigned to it.  
The index is a progressive number. If the token already exists, the internal dictionary won't be updated and the token's numeric value (id) is immediately returned.  


**RETURN VALUES**

The numeric index (ID) of the token. If token could not be added to the dictionary or **token** is **NULL**, the function will return [PS_INVALID_TOKEN_ID](macros.md#ps-invalid-token-id).


### PSVocabularyCreate

In: dataset.h, line: 220

```c
PSVocabulary  * PSVocabularyCreate (long initial_capacity)
```

Create a [PSVocabulary](types.md#psvocabulary) with initial capacity of **initial_capacity**.  


**RETURN VALUES**

The vocabulary or **NULL** if memory cannot be allocated.


### PSVocabularyErrorString

In: dataset.h, line: 226

```c
const char  * PSVocabularyErrorString (long err)
```




### PSVocabularyFree

In: dataset.h, line: 227

```c
void PSVocabularyFree (PSVocabulary *vocabulary)
```




### PSVocabularyGetTokenByID

In: dataset.h, line: 223

```c
const char  * PSVocabularyGetTokenByID (PSVocabulary *vocabulary, long id)
```

Get the token associated with **id** from **vocabulary**.  


**RETURN VALUES**

The token associated with **id** or **NULL** if no **token** is found with **id**. Also return **NULL** if **vocabulary** is **NULL**.


### PSVocabularyGetTokenID

In: dataset.h, line: 222

```c
long PSVocabularyGetTokenID (PSVocabulary *vocabulary, char *token)
```

Get the ID of the token **token** from vocabulary **vocabulary**.  


**RETURN VALUES**

The numeric index (ID) of the token. If token is not found into **vocabulary**, the function will return [PS_TOKEN_NOT_FOUND](macros.md#ps-token-not-found).  
If **vocabulary** is **NULL** or **token** is **NULL**, the function will return [PS_INVALID_TOKEN_ID](macros.md#ps-invalid-token-id).


### PSVocabularyLoad

In: dataset.h, line: 224

```c
PSVocabulary  * PSVocabularyLoad (const char *path)
```

Load vocabulary from file located at **path**.  


**RETURN VALUES**

The pointer to vocabulary or **NULL** if:  

 - **path** is **NULL**.
 - **path** does not exists.
 - **path** cannot be opened for reading.
 - File at **path** is not a valid PsyC vocabulary file.
 - Vocabulary would have zero tokens.
 - Memory cannot be allocated.
 - Size of some token exceeds max. size ([PS_IO_MAX_TOKEN_SIZE](macros.md#ps-io-max-token-size)).
 - Some token cannot be added to vocabulary.


### PSVocabularySave

In: dataset.h, line: 225

```c
int PSVocabularySave (PSVocabulary *vocabulary, const char *path)
```

Save **vocabulary** to file located at **path**. Vocabulary tokens are written sequentially to the file as an ordered sequence of **NULL**-terminated strings.  


**RETURN VALUES**

1 if the vocabulary has been successfully saved, 0 if:  

 - **vocabulary** is **NULL** or **path** is **NULL**.
 - **path** cannot be opened for writing.
 - Size of some token exceeds max. size ([PS_IO_MAX_TOKEN_SIZE](macros.md#ps-io-max-token-size)).
 - Some writing error occurs.


### PSVPrint

In: log.h, line: 127

```c
void PSVPrint (int level, const char *format, va_list args)
```




### PSVPrintSameLine

In: log.h, line: 141

```c
void PSVPrintSameLine (char *format, va_list args)
```




### PSWarn

In: log.h, line: 131

```c
void PSWarn (const char *format, ...)
```




### PSWindowGradOptimization

In: optimization.h, line: 51

```c
int PSWindowGradOptimization (PSFloat *params, PSFloat *grads, PSFloat *mgrads, PSFloat *xgrads, PSFloat *tmp, PSFloat *mtmp, PSFloat *xtmp, PSFloat rate, PSFloat momentum, long len, int acceleration, long iteration, struct PSTrainingOptions *options)
```




### PSWorkingDirectory

In: utils.h, line: 128

```c
const char  * PSWorkingDirectory (void)
```

Get PsyC working directory, that is, by default `$HOME/.psyc`.  
A custom working directory can be specified at compile-time using **PS_WORKING_DIR** macro or by setting **PS_WORKING_DIR** environment variable.  
The function will try to automatically create the working directory if it doesn't exist.  


**RETURN VALUES**

Path to the working directory or **NULL** in case something goes wrong.  


**NOTE**:  it returns a static string, so it cannot be freed.  


### PSXTermColor256ToANSI

In: log.h, line: 140

```c
int PSXTermColor256ToANSI (uint8_t color, int bgcolor)
```





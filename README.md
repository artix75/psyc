# PsyC

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

PsyC is an open-source C library that allows building neural networks on POSIX systems such as Linux or macOS. It provides a collection of layer types and functionalities to facilitate the construction and training of neural networks.

## Idea and Project Name

PsyC is the result of the fascination with artificial neural networks and their similarities to biological neural networks and the human brain. The project's name, PsyC, is derived from a combination of "psyche" (mind) and "C" (programming language), representing the intersection of the developer's background in psychology and their subsequent career as a programmer.

The project started in 2016 as a self-directed journey to explore artificial neural networks, mainly for autodidactic purposes.
Recognizing the need for improved performance in neural network training, the developer opted to implement the library using the C programming language.

After a five-year pause, the project restarted in the summer of 2022.

## Features

- Fully Connected (Dense) layers.
- Linear layers.
- Convolutional/Pooling layers.
- Normalization layers.
- Dropout layers.
- Embedding layers.
- Recurrent, LSTM, and GRU layers.
- Positional Encoding.
- Attention mechanism.

PsyC also provides additional features:

- Building models composed of multiple neural networks, such as encoder-decoder models
- Building transformers by utilizing Attention layers and enabling a flag that allows feeding the whole sequence as inputs

## System Requirements

- POSIX-compliant operating system (Linux, macOS)
- zlib must be installed on your system

## Supported hardware/libraries

PsyC can leverage certain libraries and hardware for accelerated computation.

- [Apple® Accelerate Framework](https://developer.apple.com/documentation/accelerate) for optimal performance on Apple® Silicon CPUs
- [AVX](https://www.intel.com/content/www/us/en/support/articles/000005779/processors.html) instruction set support for improved performance on x86 CPUs
- [BLAS](https://www.netlib.org/blas/) routines support can be provided by either Apple Accelerate Framework or [GNU GSL Library](https://www.gnu.org/software/gsl/) if they're found. If none of the previous options are found, PsyC still provides its own partial implementation of some BLAS routines (but it's actually quite slow in this latter case).

The above libraries and hardware are automatically detected by PsyC and they're not dependecies, as PsyC can run without them.

Currently, there's no support for [NVIDIA® CUDA®](https://developer.nvidia.com/cuda-toolkit), but its support is planner for future releases.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/artix75/psyc
   ```

2. Build the library:

   ```shell
   cd psyc
   make
   ```

   Note: Make sure you have the necessary dependencies installed.

3. Install:

   ```shell
   make install
   ```

   Note: by default, PsyC will be installed inside /usr/local. If you want to install PsyC to a different location, provide the PREFIX variable to the make command, like this.

   ```shell
   make PREFIX=/opt install
   ```

## Testing

   ```shell
   make test
   ```

## Usage

After installing it, PsyC will provide:

- A dynamic library: **libpsyc.so** (**libpsyc.dylib** on macOS)
- A static library: **libpsyc.a**
- A command line tool: **psycl**
- An utility to build your own program with the PsyC lib: **psyc-cc**

### Example:

1. Include the necessary headers in your source file:

   ```c
   #include <psyc/all.h>
   ```

2. Create a neural network:

   ```c
   PSModel *model = PSModelCreate("My Awesome Model");
   ```

3. Add some layers:

   ```c
   PSAddLayer(model, FullyConnected, 784, NULL); /* Input layer, size: 784) */
   PSAddLayer(model, FullyConnected, 30, NULL); /* Hidden dense (FullyConnected) layer of size 30) */
   PSAddLayer(model, SoftMax, 10, NULL); /* Output softmax layer of size 10) */
   ```

   The code above creates a basic neural network composed of only Fully-Connected (Dense) layers and a Softmax layer used for the output layer. However, PsyC offers various types of layers (see the [Features](#features) section). Here are some example codes:

   ```c
   /* Add a convolutional layer */
   PSLayerDef conv_def = {.filter_width = 3, .stride = 1, .padding = 2};
   PSAddLayer(model, Convolutional, 0, &conv_def);

   /* Add LSTM or GRU layers */
   PSAddLayer(model, LSTM, 10, NULL);
   PSAddLayer(model, GRU, 10, NULL);

   /* Add an Attention layer */
   PSLayerDef attn_def = {.attention_heads = 4, .causal_attention = 1};
   PSAddLayer(model, Attention, 10, &attn_def);
   ```

4. Train your model:

   ```c
   PSFloat training_data[] = {...};
   PSTraningOptions opts = {.learning_rate = 0.01};
   int datalen = (int) sizeof(training_data) / sizeof(PSFloat);
   PSTrain(model, training_data, datalen, NULL, 0, &opts);
   ```

5. Get predictions:

   Get raw predictions by forwarding inputs to the model and by reading
   the output layer's state:

   ```c
   PSFloat inputs[] = {...};
   PSForward(model, inputs);
   PSLayer *output_layer = PSGetOutputLayer(model);
   PSFloat *outputs = PSGetOutputs(output_layer);
   ```

   Or get classification predictions:

   ```c
   PSFloat inputs[] = {...};
   int predicted = PSClassify(model, inputs)
   ```

6. Finally, release the model:

   ```c
   PSModelFree(model);
   ```

7. Compile your program by linking against the PsyC library, like this:

   ```shell
   PREFIX=/usr/local # Or whatever you used as PREFIX during install
   gcc -I$PREFIX/include -L $PREFIX/lib -lpsyc -o myprogram myprogram.c
   ```

   or, alternatively, you can use the provided `psyc-cc` utility:

   ```shell
   psyc-cc -o myprogram myprogram.c
   ```

## Command Line Tool and Demos

PsyC can also be built as a command line tool (`psycl`).
To see its usage:

   ```shell
   psycl --help
   ```

### Command Line Tool example:

   ```shell
   psycl --layer fully_connected 784 --layer fully_connected 30 --layer fully_connected 10 --train --mnist --test --mnist
   ```

PsyC also provides several demos.

- MNIST demo
- CIFAR demo
- Recurrent networks demo
- Encoder-decoder demo (both with and without attention mechanism)
- GPT2 emulation demo

After building PsyC, you can find them inside the `bin/` subdirectory that is
found inside the PsyC source directory.

## Future Roadmap

- GPU support (e.g., [CUDA®](https://developer.nvidia.com/cuda-toolkit)) for enhanced performance on compatible systems
- [OpenBLAS](https://www.openblas.net/) support
- Diffusion models, VAE, GAN, and so on.
- Bi-directional recurreent networks

## License

PsyC is licensed under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
```

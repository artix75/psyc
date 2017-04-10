PsyC
===

PsyC is a C implementation of some of the most common Artificial Neural Network models.
It provides a linkable dynamic library and a command line tool.
It has been written mainly for autodidactic purpose, so it's not guaranteed to 
be safe in production contexts, but you can play with it if you want.

Surely it's not the state-of-the-art neural network implementation, because its 
code has been structured with the aim of being easly readable and 
understandable rather than being peformant and efficient.
Anyway, it's quite fast and it also supports [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) extensions when built on 
CPUs that have support it.
It doesn't currenlty support [CUDA](http://www.nvidia.com/object/cuda_home_new.html), but i have plans for it in the future.

PsyC currently has support for the following neural network models:

- Fully Condensed Neural Networks
- Convolutional
- Neural Networks 
- Recurrent Neural Networks.
- LSTM

Supported Platforms
===

PsyC should build with no problems on Linux and OS X/macOS platforms.
I've never tested it on other POSIX platforms.
Support for non-POSIX platforms is not in my plans.

Build and Install
===

Building PsyC should be quite simple, just jump inside the PsyC source directory and 
type:

    make

The build process should automatically detect if your CPU supports AVX2, and 
consequently enable AVX2 extensions.
Anyway if you want to turn AVX2 support off, or if it's not automatically 
detected during build phase, you can always enable/disable it by adding 
the AVX variable after the make command, ie:

    make AVX=off  #disables AVX2 extensions

    make AVX=on   #explicitly enables AVX2 extensions

PsyC provides some convenience utility functions that make easy 
to feed image files directly into the network (useful with convolutional networks).
These functions require [ImageMagick](https://www.imagemagick.org/script/index.php) to be installed on your system.
Again, the build process should automatically detect if ImageMagick and 
its development libraries are installed on your system, so that it can automatically 
disable image utility functions in case you haven't it.

But if you encounter some problem with ImageMagick compatibility you can 
manually disable support for it by adding the variable MAGICK=off after make:

    make MAGICK=off

In order to install the library, headers and command line tool on your system,
just use the canonical 

    make install

By default, the installation prefix will be /usr/local/, but if you want to 
change it, just add the PREFIX variable:

    make install PREFIX=/usr/opt/local

Running some example
===

You can use the command line tool `psycl` in order to try PsyC.

Building and training a Fully Connected Network with MNIST data
---

    psycl --layer fully_connected 784 --layer fully_connected 30 --layer fully_connected 10 --train --mnist --test --mnist

The trained network will be saved in the /tmp/ directory, but you choose a different 
output file with via the --save option, ie:

    psycl --layer fully_connected 784 --layer fully_connected 30 --layer fully_connected 10 --train --mnist --test --mnist --save /home/myhome/pretrained.data

Loading a pretrained convolutional network
---

    psycl --load /usr/local/share/psyc/resources/pretrained.cnn.data --test --mnist

Trying to classify an image using a network pretrained on MNIST dataset
---

    psycl --load /usr/local/share/psyc/resources/pretrained.mnist.data --classify-image /usr/local/share/psyc/resources/digit.2.png --grayscale

Using the library
===

Here is an example of a simple Fully Connected Network

    #include <psyc/psyc.h>
    
    #define EPOCHS 30
    #define LEARNING_RATE 0.5
    #define BATCH_SIZE 10
    #define TRAIN_FLAGS 0
    
    double * training_data;
    double * evaluation_data;
    int datalen;
    int evaluation_datalen;

    ...
    ...
    
    PSNeuralNetwork * network = PSCreateNetwork("Simple Neural Network");
    
    PSAddLayer(network, FullyConnected, INPUT_SIZE, NULL);
    PSAddLayer(network, FullyConnected, 30, NULL);
    PSAddLayer(network, FullyConnected, 10, NULL);
    
    PSTrain(network, training_data, datalen, EPOCHS, LEARNING_RATE, 
            BATCH_SIZE, TRAIN_FLAGS, evaluation_data, evaluation_datalen);
    
    
    ...
    
    PSDeleteNetwork(netowrk);

And now an example of a Convolutioanl Neural Network:

    #define FEATURES_COUNT 20
    #define REGION_SIZE 5
    #define POOL_SIZE 2
    #define RELU_ENABLED 0
    
    PSNeuralNetwork * network = PSCreateNetwork("Convolutional Neural Network");
    
    PSLayerParameters * cparams;
    PSLayerParameters * pparams;
    cparams = PSCreateConvolutionalParameters(FEATURES_COUNT, REGIONS_SIZE,
    1, 0, RELU_ENABLED);
    pparams = PSCreateConvolutionalParameters(FEATURES_COUNT, POOL_SIZE,
    0, 0, RELU_ENABLED);


    PSAddLayer(network, FullyConnected, 784, NULL);
    PSAddConvolutionalLayer(network, cparams);
    PSAddPoolingLayer(network, pparams);
    PSAddLayer(network, FullyConnected, 30, NULL);
    PSAddLayer(network, SoftMax, 10, NULL);
    
    ...
    
    PSDeleteNetwork(netowrk);









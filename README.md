PsyC
===

PsyC is a C implementation of some of the most common Artificial Neural Networks.
It provides a linkable dynamic library and a command line tool.
It has been written mainly for autodidactic purpose, so it's not guaranteed to 
be safely used in production contexts, but you can play with it if you want.

Surely it's not the state-of-the-art neural network implementation, because its 
code has been structured with the aim of being easly readable and 
understandable rather than being peformant and efficient.
Anyway, it's quite fast and it also supports AVX2 extensions when built on 
CPUs that have support it.
It doesn't currenlty support CUDA, but i have plans for it in the future.

PsyC currently has support for the following neural network models:
Fully Condensed Neural Networks, Convolutional Neural Networks and basic 
Recurrent Neural Networks.
LSTM networks aren't currenlty implemented, but they should be available in 
the next future.

Supported Platforms
===

PsyC should build with no problems on Linux and OS X platforms.
I've never tested it on other POSIX platforms.
Support for non-POSIX platforms is not in my plans.

Build and Install
===

Building PsyC should be quite simple, just jump inside the PsyC directory and 
type:

    make

The build process should automatically detect if your CPU supports AVX2, and 
consequently enable AVX2 extension.
Anyway if you want to turn AVX2 support off, or if it's not automatically 
detected during build pahse, you can always enable/disable it by adding 
the AVX variable after the make command, ie:

    make AVX=off  #disables AVX2 extensions

    make AVX=on   #explicitly enables AVX2 extensions

PsyC does ship with some convenience utility functions that make easy 
to feed image files directly to the network (usefult with convolutional networks).
These functions require [ImageMagick](https://www.imagemagick.org/script/index.php) to be installed on your system.
Again, the build process should automatically detect if you have ImageMagick 
and its development libraries installed on your system, and automatically 
disable image utility functions in case you haven't it.

But if you encounter some problem with ImageMagick compatibility you can 
manually disable support for it by adding the variable MAGICK=off after make:

    make MAGICK=off

In order to install the library, headers and command line tool on your system,
just use the canonical 

    make install

By default the installation prefix will be /usr/local/, but if you want to 
change it, just add the PREFIX variable:

    make install PREFIX=/usr/opt/local

Running some example
===

You can use the command line tool psyc\_cli in order to try PsyC.

Building and training a Fully Connected Network with MNIST data
---

    psyc_cli --layer fully_connected 784 --layer fully_connected 30 --layer fully_connected 10 --train --mnist --test --mnist

The trained network will be saved in the /tmp/ directory, but you choose a different 
output file with via the --save option, ie:

    psyc_cli --layer fully_connected 784 --layer fully_connected 30 --layer fully_connected 10 --train --mnist --test --mnist --save /home/myhome/pretrained.data

Loading a pretrained convolutional network
---

    psyc_cli --load /usr/local/share/psyc/resources/pretrained.cnn.data --test --mnist

Trying to classify an image using a network pretrained on MNIST dataset
---

    psyc_cli --load /usr/local/share/psyc/resources/pretrained.mnist.data --classify-image /usr/local/share/psyc/resources/digit.2.png --grayscale











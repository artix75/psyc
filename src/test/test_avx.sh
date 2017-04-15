#!/bin/bash

cwd=$(pwd)
cwd=$(basename "$cwd")

if [ "$cwd" = "test" ]; then
    cd ../../
fi

if [ -e /tmp/avx.nn.data ]; then
    rm /tmp/avx.nn.data
fi

if [ -e /tmp/avx.cnn.data ]; then
    rm /tmp/avx.cnn.data
fi

if [ -e /tmp/no_avx.nn.data ]; then
    rm /tmp/no_avx.nn.data
fi

if [ -e /tmp/no_avx.cnn.data ]; then
    rm /tmp/no_avx.cnn.data
fi

make clean && make

bin/psycl --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 1 --save /tmp/avx.nn.data

bin/psycl --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 1 --save /tmp/avx.cnn.data

make clean && make AVX=off

bin/psycl --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 1 --save /tmp/no_avx.nn.data

bin/psycl --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 1 --save /tmp/no_avx.cnn.data

OBJS=(psyc utils convolutional recurrent lstm)
COBJS=""
for OBJ in ${OBJS[@]}; do
    echo "gcc -o /tmp/$OBJ.o -c src/$OBJ.c"
    gcc -o /tmp/$OBJ.o -c src/$OBJ.c
    COBJS="$COBJS /tmp/$OBJ.o"
done
gcc -o /tmp/compare_avx.o -c "src/test/compare_avx.c"
gcc -o /tmp/compare_avx $COBJS /tmp/compare_avx.o -lz -lm

/tmp/compare_avx

make clean > /dev/null

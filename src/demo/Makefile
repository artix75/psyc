SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99
LDFLAGS=-lz -lm
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=c99
endif

REQUIRED_OBJS=../neural.o ../mnist.o

mnist_demo: $(REQUIRED_OBJS) mnist_demo.o
	$(CC) -o ../../bin/mnist_demo $(LDFLAGS) $(REQUIRED_OBJS) mnist_demo.o
conv_demo: $(REQUIRED_OBJS) conv_demo.o
	$(CC) -o ../../bin/conv_demo $(LDFLAGS) $(REQUIRED_OBJS) conv_demo.o
all: mnist_demo conv_demo
	
        

SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99 -g -ggdb
LDFLAGS=-lz -lm
REQUIRED_OBJS=../psyc.o ../mnist.o

include ../avx.mk

ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=c99 -g -ggdb
        REQUIRED_OBJS+=../avx.o
endif

mnist_demo: $(REQUIRED_OBJS) mnist_demo.o
	$(CC) -o ../../bin/mnist_demo $(REQUIRED_OBJS) mnist_demo.o $(LDFLAGS)
conv_demo: $(REQUIRED_OBJS) conv_demo.o
	$(CC) -o ../../bin/conv_demo $(REQUIRED_OBJS) conv_demo.o $(LDFLAGS)
rnn_demo: $(REQUIRED_OBJS) rnn_demo.o
	$(CC) -o ../../bin/rnn_demo $(REQUIRED_OBJS) rnn_demo.o $(LDFLAGS)
all: mnist_demo conv_demo rnn_demo
	
        

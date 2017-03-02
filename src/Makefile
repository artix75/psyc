SHELL=/bin/bash
CC=gcc
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma
endif

neural_cli: neural.o mnist.o neural_cli.o
	gcc -o ../bin/neural_cli -lz neural.o mnist.o neural_cli.o
all: neural_cli
	
        

SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99
LDFLAGS=-lz -lm
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=c99
endif
NEURAL_CLI_OBJS=neural.o mnist.o neural_cli.o

neural_cli: $(NEURAL_CLI_OBJS)
	$(CC) -o ../bin/neural_cli $(LDFLAGS) $(NEURAL_CLI_OBJS)
all: neural_cli
	
        

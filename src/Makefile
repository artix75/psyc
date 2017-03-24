SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99
LDFLAGS=-lz -lm
NEURAL_CLI_OBJS=neural.o mnist.o neural_cli.o
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=gnu99
        NEURAL_CLI_OBJS+=avx.o
endif

neural_cli: $(NEURAL_CLI_OBJS)
	$(CC) -o ../bin/neural_cli $(NEURAL_CLI_OBJS) $(LDFLAGS)
all: neural_cli
	
        

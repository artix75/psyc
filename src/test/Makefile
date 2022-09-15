SHELL=/bin/bash
CC=gcc
OPTIMIZATION?=-O2
OPT=$(OPTIMIZATION)
CFLAGS=-std=gnu99 -Wall -W -Wno-missing-field-initializers $(OPT)
LDFLAGS=-lz -lm -ldl
OBJS=../psyc.o ../utils.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o test.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=../avx.o
endif

default: all

main_tests: $(OBJS) main_tests.o
	$(CC) -o main_tests $(OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        

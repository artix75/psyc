SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99
LDFLAGS=-lz -lm
REQUIRED_OBJS=../neural.o ../mnist.o test.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=gnu99
        REQUIRED_OBJS+=../avx.o
endif


main_tests: $(REQUIRED_OBJS) main_tests.o
	$(CC) -o main_tests $(REQUIRED_OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        

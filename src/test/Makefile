SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99
LDFLAGS=-lz -lm
REQUIRED_OBJS=../neural.o ../mnist.o test.o
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=c99
        REQUIRED_OBJS+=../avx.o
endif


main_tests: $(REQUIRED_OBJS) main_tests.o
	$(CC) -o main_tests $(REQUIRED_OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        

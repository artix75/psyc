SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -g -ggdb
LDFLAGS=-lz -lm
REQUIRED_OBJS=../psyc.o ../mnist.o test.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        REQUIRED_OBJS+=../avx.o
endif


main_tests: $(REQUIRED_OBJS) main_tests.o
	$(CC) -o main_tests $(REQUIRED_OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        

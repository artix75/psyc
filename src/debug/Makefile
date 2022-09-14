SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -g -ggdb -O0
LDFLAGS=-lz -lm -ldl
OBJS=../psyc.o ../utils.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=../avx.o
endif

default: all

profile: $(OBJS) profile.o
	$(CC) -o profile $(OBJS) profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        

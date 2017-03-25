SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -g -ggdb
LDFLAGS=-lz -lm
REQUIRED_OBJS=../psyc.o ../mnist.o

include ../avx.mk
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        REQUIRED_OBJS+=../avx.o
endif

profile: $(REQUIRED_OBJS) profile.o
	$(CC) -o profile $(REQUIRED_OBJS) profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        

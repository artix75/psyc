SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99 -g -ggdb
LDFLAGS=-lz -lm
REQUIRED_OBJS=../neural.o ../mnist.o
ifeq ($(AVX),on)
	CFLAGS=-DUSE_AVX -mavx2 -mfma -std=c99 -g -ggdb
        REQUIRED_OBJS+=../avx.o
endif

profile: $(REQUIRED_OBJS) profile.o
	$(CC) -o profile $(REQUIRED_OBJS) profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        

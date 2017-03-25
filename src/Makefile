SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99
LDFLAGS=-lz -lm
psyc_CLI_OBJS=psyc.o mnist.o psyc_cli.o

include avx.mk

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        psyc_CLI_OBJS+=avx.o
endif

psyc_cli: $(psyc_CLI_OBJS)
	$(CC) -o ../bin/psyc_cli $(psyc_CLI_OBJS) $(LDFLAGS)
all: psyc_cli
	
        

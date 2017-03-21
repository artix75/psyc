SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99 -g -ggdb
LDFLAGS=-lz -lm

profile: ../neural.o ../mnist.o profile.o
	$(CC) -o profile ../neural.o ../mnist.o profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        

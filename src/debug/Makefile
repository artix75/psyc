SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99
LDFLAGS=-lz -lm

profile: ../neural.o ../mnist.o profile.o
	$(CC) -o profile $(LDFLAGS) ../neural.o ../mnist.o profile.o
	valgrind --leak-check=yes ./profile
all: profile
        

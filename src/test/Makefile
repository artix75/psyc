SHELL=/bin/bash
CC=gcc
CFLAGS=-std=c99
LDFLAGS=-lz -lm

REQUIRED_OBJS=../neural.o ../mnist.o test.o

main_tests: $(REQUIRED_OBJS) main_tests.o
	$(CC) -o main_tests $(LDFLAGS) $(REQUIRED_OBJS) main_tests.o
all: main_tests
	./main_tests
        

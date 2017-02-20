SHELL=/bin/bash
CC=gcc

main_tests: ../neural.o ../mnist.o test.o main_tests.o
	gcc -o main_tests -lz ../neural.o ../mnist.o test.o main_tests.o
all: main_tests
	./main_tests
        

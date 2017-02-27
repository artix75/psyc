SHELL=/bin/bash
CC=gcc

profile: ../neural.o ../mnist.o profile.o
	gcc -o profile -lz ../neural.o ../mnist.o profile.o
	valgrind --leak-check=yes ./profile
all: profile
        

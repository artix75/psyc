SHELL=/bin/bash
CC=gcc
CFLAGS=-mavx

mnist_demo: ../neural.o ../mnist.o mnist_demo.o
	gcc -o ../../bin/mnist_demo -lz ../neural.o ../mnist.o mnist_demo.o
conv_demo: ../neural.o ../mnist.o conv_demo.o
	gcc -o ../../bin/conv_demo -lz ../neural.o ../mnist.o conv_demo.o
all: mnist_demo conv_demo
	
        

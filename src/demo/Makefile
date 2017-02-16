SHELL=/bin/bash
CC=gcc

mnist_demo: ../neural.o ../mnist.o mnist_demo.o
	gcc -o ../../bin/mnist_demo -lz ../neural.o ../mnist.o mnist_demo.o
clean:
	rm *.o && rm ../*.o && rm ../../bin/mnist_demo
all: mnist_demo
	
        

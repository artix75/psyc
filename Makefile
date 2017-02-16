SHELL=/bin/bash
CC=gcc

mnist_demo:
	cd src/demo/ && make
clean:
	rm -R src/*.o && rm bin/*
all: mnist_demo
	
        

SHELL=/bin/bash
CC=gcc

demo:
	cd src/demo/ && make all
test:
	cd src/test && make all
clean:
	if [ -e bin/README ]; then cp bin/README tmp/; fi
	rm -f src/*.o
	rm -f src/demo/*.o
	rm -f src/test/*.o
	rm -f bin/*
	if [ -e tmp/README ]; then cp tmp/README bin/; fi
all: demo
	
        

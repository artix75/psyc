include ../Makefile.common.mk
#OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o
valgrind := $(shell which valgrind)
memleak_cmd = valgrind --leak-check=yes ./profile
ifeq (,$(valgrind))
ifeq ($(PLATFORM), Darwin)
        memleak_cmd = leaks --atExit -- ./profile
else
        $(error "Could not find valgrind or other memory profiling tools")
endif
endif

default: all

profile: $(OBJS) profile.o
	$(CC) -o profile $(OBJS) profile.o $(LDFLAGS)
	$(memleak_cmd)
all: profile
        

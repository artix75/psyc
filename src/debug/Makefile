include ../Makefile.common.mk
#OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o
valgrind := $(shell which valgrind)
memleak_cmd = valgrind --leak-check=yes ./profile-test
ifeq (,$(valgrind))
ifeq ($(PLATFORM), Darwin)
        memleak_cmd = leaks --atExit -- ./profile-test
else
        $(error "Could not find valgrind or other memory profiling tools")
endif
endif

default: all

Makefile.dep:
	-$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

profile-test: $(OBJS) profile-test.o
	$(CC) -o $@ $(OBJS) profile-test.o $(LDFLAGS)
	$(memleak_cmd)
all: profile-test

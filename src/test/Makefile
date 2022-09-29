include ../Makefile.common.mk
include ../avx.mk

OBJS=../psyc.o ../utils.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o test.o
ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=../avx.o
endif
ifeq ($(VERBOSE_TESTS),on)
	CFLAGS+=-DPS_VERBOSE_TESTS
endif
ifeq ($(CATCH_FPE),on)
	CFLAGS+=-DCATCH_FPE
endif

default: all

main_tests: $(OBJS) main_tests.o
	$(CC) -o main_tests $(OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests
        

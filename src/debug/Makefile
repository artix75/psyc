include ../Makefile.common.mk
include ../avx.mk
OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../debug.o

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=../avx.o
endif


default: all

profile: $(OBJS) profile.o
	$(CC) -o profile $(OBJS) profile.o $(LDFLAGS)
	valgrind --leak-check=yes ./profile
all: profile
        

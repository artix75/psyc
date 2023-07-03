include ../Makefile.common.mk

#OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../debug.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../cifar.o
ifeq ($(CATCH_FPE),on)
	CFLAGS+=-DCATCH_FPE
endif
GPT_CFLAGS=''
GPT_LDFLAGS=''
ifeq ($(HAS_READLINE),yes)
	GPT_CFLAGS=-DHAS_READLINE
	GPT_LDFLAGS=-lreadline
endif

default: all

gpt_demo.o:
	$(CC) --std=gnu99 $(CFLAGS) $(GPT_CFLAGS) -c -o $@ gpt_demo.c

mnist_demo: $(OBJS) mnist_demo.o
	$(CC) -o ../../bin/mnist_demo $(OBJS) mnist_demo.o $(LDFLAGS)
conv_demo: $(OBJS) conv_demo.o
	$(CC) -o ../../bin/conv_demo $(OBJS) conv_demo.o $(LDFLAGS)
rnn_demo: $(OBJS) rnn_demo.o
	$(CC) -o ../../bin/rnn_demo $(OBJS) rnn_demo.o $(LDFLAGS)
lstm_demo: $(OBJS) lstm_demo.o
	$(CC) -o ../../bin/lstm_demo $(OBJS) lstm_demo.o $(LDFLAGS)
char_rnn: $(OBJS) char_rnn.o
	$(CC) -o ../../bin/char_rnn $(OBJS) char_rnn.o $(LDFLAGS)
cifar_demo: $(OBJS) cifar_demo.o
	$(CC) -o ../../bin/cifar_demo $(OBJS) cifar_demo.o $(LDFLAGS)
gpt_demo: $(OBJS) gpt_demo.o
	$(CC) -o ../../bin/gpt_demo $(OBJS) gpt_demo.o $(LDFLAGS) $(GPT_LDFLAGS)
all: mnist_demo conv_demo rnn_demo lstm_demo char_rnn cifar_demo gpt_demo
	
        

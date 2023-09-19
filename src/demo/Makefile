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

LOCALBINDIR = ../../bin

default: all

Makefile.dep:
	-$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

gpt_demo.o:
	$(call print-action,$(COMPILE_ACTION_NAME),$<)
	$(CC) --std=gnu99 $(CFLAGS) $(GPT_CFLAGS) -c -o $@ gpt_demo.c

$(LOCALBINDIR)/mnist_demo: $(OBJS) mnist_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) mnist_demo.o $(LDFLAGS)
$(LOCALBINDIR)/conv_demo: $(OBJS) conv_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) conv_demo.o $(LDFLAGS)
$(LOCALBINDIR)/rnn_demo: $(OBJS) rnn_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) rnn_demo.o $(LDFLAGS)
$(LOCALBINDIR)/lstm_demo: $(OBJS) lstm_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) lstm_demo.o $(LDFLAGS)
$(LOCALBINDIR)/char_rnn: $(OBJS) char_rnn.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) char_rnn.o $(LDFLAGS)
$(LOCALBINDIR)/cifar_demo: $(OBJS) cifar_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) cifar_demo.o $(LDFLAGS)
$(LOCALBINDIR)/seq2seq_demo: $(OBJS) seq2seq_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) seq2seq_demo.o $(LDFLAGS)
$(LOCALBINDIR)/gpt_demo: $(OBJS) gpt_demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(OBJS) gpt_demo.o $(LDFLAGS) $(GPT_LDFLAGS)
all: $(addprefix $(LOCALBINDIR)/, mnist_demo conv_demo rnn_demo lstm_demo char_rnn cifar_demo gpt_demo seq2seq_demo)

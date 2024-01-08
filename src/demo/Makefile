include ../Makefile.common.mk
.PHONY: default all

#OBJS=../psyc.o ../io.o ../utils.o ../maths.o ../log.o ../debug.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o ../cifar.o
ifeq ($(CATCH_FPE),on)
	CFLAGS+=-DCATCH_FPE
endif
GPT_CFLAGS=
GPT_LDFLAGS=
ifeq ($(HAS_READLINE),yes)
	GPT_CFLAGS=-DHAS_READLINE
	GPT_LDFLAGS=-lreadline
endif

LOCALBINDIR = ../../bin

default: all
	@:

Makefile.dep:
	-@$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

gpt-demo.o:
	$(call print-action,$(COMPILE_ACTION_NAME),$<)
	$(CC) $(CFLAGS) $(GPT_CFLAGS) -c -o $@ gpt-demo.c

$(LOCALBINDIR)/mnist-demo: $(OBJS) mnist-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/rnn-demo: $(OBJS) rnn-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/lstm-demo: $(OBJS) lstm-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/char-rnn: $(OBJS) char-rnn.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/cifar-demo: $(OBJS) cifar-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/seq2seq-demo: $(OBJS) seq2seq-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS)
$(LOCALBINDIR)/gpt-demo: $(OBJS) gpt-demo.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $^ $(LDFLAGS) $(GPT_LDFLAGS)
all: $(addprefix $(LOCALBINDIR)/, mnist-demo rnn-demo lstm-demo char-rnn cifar-demo gpt-demo seq2seq-demo)

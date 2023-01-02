include ../Makefile.common.mk

OBJS+=test.o

ifeq ($(VERBOSE_TESTS),on)
	CFLAGS+=-DPS_VERBOSE_TESTS
endif
ifeq ($(CATCH_FPE),on)
	CFLAGS+=-DCATCH_FPE
endif
TEST_ARGS=
ifneq ($(SKIP_TEST),)
	TEST_ARGS+=--skip $(SKIP_TEST)
endif
ifneq ($(TESTS),)
	TEST_ARGS+=$(TESTS)
endif

default: all

main_tests: $(OBJS) main_tests.o
	$(CC) -o main_tests $(OBJS) main_tests.o $(LDFLAGS)
all: main_tests
	./main_tests$(TEST_ARGS)
        

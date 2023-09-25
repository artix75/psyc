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

Makefile.dep:
	-$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

benchmark: $(OBJS) benchmark.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o benchmark $(OBJS) benchmark.o $(LDFLAGS)
	@cp benchmark ../../bin/
	@[ -z "$(BUILD_ONLY)" ] && ./benchmark$(TEST_ARGS) || echo '' > /dev/null

main_tests: $(OBJS) main_tests.o
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o main_tests $(OBJS) main_tests.o $(LDFLAGS)

all: main_tests
	@[ -z "$(BUILD_ONLY)" ] && ./main_tests$(TEST_ARGS) || echo '' > /dev/null
        

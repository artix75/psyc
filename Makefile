SHELL=/bin/bash
CC=gcc
gen_conf_mk := $(shell sh -c './conf.sh')

DEFAULT_BUILD_TARGETS=neural_cli demo
ifeq ($(DEMO),off)
	DEFAULT_BUILD_TARGETS=neural_cli
endif

default: all

.PHONY: clean
.PHONY: clean-full
.PHONY: show-build-info

demo:
	@cd src/demo/ && $(MAKE)
neural_cli:
	@cd src && $(MAKE)
test:
	cd src/test && $(MAKE)
profile:
	cd src/debug && $(MAKE)
clean:
	if ! [ -e tmp/ ]; then mkdir tmp/; fi
	if [ -e bin/README ]; then cp bin/README tmp/; fi
	rm -f src/*.o
	rm -f src/demo/*.o
	rm -f src/test/*.o
	rm -f src/test/main_tests
	rm -f bin/*
	rm -f lib/*
	if [ -e tmp/README ]; then cp tmp/README bin/; fi
	if [ -e tmp/README ]; then cp tmp/README lib/; fi

clean-full: clean
	rm -f .c_headers
	rm -f src/config.mk

install:
	@cd src && $(MAKE) install
uninstall:
	@cd src && $(MAKE) uninstall

all: $(DEFAULT_BUILD_TARGETS)

valgrind:
	$(MAKE) OPTIMIZATION="-O0"

helgrind:
	$(MAKE) OPTIMIZATION="-O0" CFLAGS="-D__ATOMIC_VAR_FORCE_SYNC_MACROS"
	
show-build-info:
	@cd src && $(MAKE) show-build-info

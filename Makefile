SHELL=/bin/bash
CC=gcc
gen_conf_mk := $(shell sh -c './conf.sh')

DEFAULT_BUILD_TARGETS=neural_cli demo
ifeq ($(DEMO),off)
	DEFAULT_BUILD_TARGETS=neural_cli
endif

default: all

.PHONY: clean
.PHONY: distclean
.PHONY: show-build-conf
.PHONY: help

demo:
	@cd src/demo/ && $(MAKE) --no-print-directory
neural_cli:
	@cd src && $(MAKE) --no-print-directory
test:
	cd src/test && $(MAKE) --no-print-directory
benchmark:
	cd src/test && $(MAKE) --no-print-directory benchmark
profile:
	cd src/debug && $(MAKE) --no-print-directory
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

distclean: clean
	rm -f .c_headers
	rm -f src/config.mk
	rm -f src/buildinfo.h

install:
	@cd src && $(MAKE) install
uninstall:
	@cd src && $(MAKE) uninstall

all: $(DEFAULT_BUILD_TARGETS)

valgrind:
	$(MAKE) OPTIMIZATION="-O0"

helgrind:
	$(MAKE) OPTIMIZATION="-O0" CFLAGS="-D__ATOMIC_VAR_FORCE_SYNC_MACROS"
	
show-build-conf:
	@cd src && $(MAKE) show-build-conf

list-available-options:
	@cd src && $(MAKE) list-available-options

help:
	@echo ''
	@echo "AVAILABLE OPTIONS:"
	@echo ''
	@cd src && $(MAKE) list-available-options
	@echo ''
	@echo "AVAILABLE RULES:"
	@echo ''
	@$(MAKE) -qp 2>/dev/null | awk -F':' '/^[a-zA-Z0-9][^$$#\/\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}' | grep -v '\.o' | grep -v '\.c' | grep -v '\.dylib' | grep -v '\.so' | grep -v '\.mk' | grep -v Makefile | sort -u
	@echo ''

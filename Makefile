SHELL=/bin/bash
CC=gcc

DEFAULT_BUILD_TARGETS=psyc-main demo
ifeq ($(DEMO),off)
	DEFAULT_BUILD_TARGETS=psyc-main
endif

default: all

.PHONY: clean
.PHONY: rebuildclean
.PHONY: distclean
.PHONY: show-build-conf
.PHONY: list-available-options
.PHONY: psyc-main
.PHONY: demo
.PHONY: test
.PHONY: profile
.PHONY: benchmark
.PHONY: install
.PHONY: uninstall
.PHONY: valgrinf
.PHONY: helgrind
.PHONY: all
.PHONY: help

demo:
	@cd src/demo/ && $(MAKE) --no-print-directory
psyc-main:
	@cd src && $(MAKE) --no-print-directory
test:
	@cd src/test && $(MAKE) --no-print-directory
benchmark:
	@cd src/test && $(MAKE) --no-print-directory benchmark
profile:
	@cd src/debug && $(MAKE) --no-print-directory
clean:
	if ! [ -e tmp/ ]; then mkdir tmp/; fi
	if [ -e bin/README ]; then cp bin/README tmp/; fi
	rm -f src/*.o
	rm -f src/Makefile.dep
	rm -f src/test/Makefile.dep
	rm -f src/demo/Makefile.dep
	rm -f src/demo/*.o
	rm -f src/test/*.o
	rm -f src/test/main_tests
	rm -f bin/*
	rm -f lib/*
	if [ -e tmp/README ]; then cp tmp/README bin/; fi
	if [ -e tmp/README ]; then cp tmp/README lib/; fi

rebuildclean: clean
	rm -f src/.make-*

distclean: rebuildclean
	rm -f .c_headers
	rm -f src/buildinfo.h
	rm -f src/config.mk

install:
	@cd src && $(MAKE) install
uninstall:
	@cd src && $(MAKE) uninstall
static:
	@cd src && $(MAKE) static

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
	@$(MAKE) -qp 2>/dev/null | awk -F':' '/^[a-zA-Z0-9][^$$#\/\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}' | grep -v '\.o' | grep -v '\.c' | grep -v '\.dylib' | grep -v '\.so' | grep -v '\.mk' | grep -v '\.h' | grep -v Makefile | sort -u
	@echo ''

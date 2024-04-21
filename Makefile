SHELL=/bin/bash
CC=gcc

DEFAULT_BUILD_TARGETS=psyc-main demo
ifeq ($(DEMO),off)
	DEFAULT_BUILD_TARGETS=psyc-main
endif

ifneq ($(COLORS),off)
	BOLD_STYLE="\033[1m"
	RED_COLOR="\033[31m"
	GREEN_COLOR="\033[32m"
	YELLOW_COLOR="\033[33m"
	BLUE_COLOR="\033[34m"
	MAGENTA_COLOR="\033[35m"
	CYAN_COLOR="\033[36m"
	END_COLOR="\033[0m"
else
	BOLD_STYLE=""
	RED_COLOR=""
	GREEN_COLOR=""
	YELLOW_COLOR=""
	BLUE_COLOR=""
	MAGENTA_COLOR=""
	CYAN_COLOR=""
	END_COLOR=""
endif

default: all
	@:

.PHONY: clean
.PHONY: rebuildclean
.PHONY: distclean
.PHONY: show-build-conf
.PHONY: list-available-options
.PHONY: psyc-main
.PHONY: demo
.PHONY: test
.PHONY: benchmark
.PHONY: install
.PHONY: uninstall
.PHONY: valgrind
.PHONY: helgrind
.PHONY: all
.PHONY: default
.PHONY: help

demo:
	@cd src/demo/ && $(MAKE) --no-print-directory
psyc-main:
	@cd src && $(MAKE) --no-print-directory
test:
	@cd src/test && $(MAKE) --no-print-directory
benchmark:
	@cd src/test && $(MAKE) --no-print-directory benchmark
clean:
	if ! [ -e tmp/ ]; then mkdir tmp/; fi
	if [ -e bin/README ]; then cp bin/README tmp/; fi
	rm -f src/*.o
	rm -f src/Makefile.dep
	rm -f src/test/Makefile.dep
	rm -f src/demo/Makefile.dep
	rm -f src/demo/*.o
	rm -f src/test/*.o
	rm -f src/test/main-tests
	rm -f src/test/benchmark
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
	rm -f src/all.h
	rm -f src/magick_conf.h
	rm -f src/magick-conf.h
	rm -f resources/testzlib

install:
	@cd src && $(MAKE) install
uninstall:
	@cd src && $(MAKE) uninstall
static:
	@cd src && $(MAKE) static

all: $(DEFAULT_BUILD_TARGETS)
	@printf '%b %b\n' $(GREEN_COLOR)Build complete.$(END_COLOR)
	@printf '%b\n' "Type \`make test\` to perform tests."
	@printf '%b\n' "Type \`make install\` to install PsyC."

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

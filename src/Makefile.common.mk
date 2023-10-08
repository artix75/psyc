SHELL=/bin/bash
CC=gcc
AR=ar
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not_found')
HARDWARE := $(shell sh -c 'uname -m 2>/dev/null || echo not_found')
SRCPATH := $(strip $(dir $(lastword $(MAKEFILE_LIST))))
PSYCPATH := $(SRCPATH)../
IS_CLANG := $(shell sh -c '$(CC) --version | grep clang > /dev/null && echo yes')
HAS_ZLIB := $(shell sh -c '$(CC) -o $(PSYCPATH)resources/testzlib $(PSYCPATH)resources/testzlib.c > /dev/null 2>&1 && echo yes')
ifneq (yes, $(HAS_ZLIB))
	HAS_APT := $(shell sh -c 'apt --version > /dev/null 2>&1 && echo yes')
	HAS_YUM := $(shell sh -c 'yum --version > /dev/null 2>&1 && echo yes')
	ERR := $(shell echo -e '\033[31mFATAL: Could not find zlib.h on your system\033[0m' >&2)
ifeq (yes, $(HAS_APT))
	TIP := $(shell echo -e '\033[33mTry to install it by typing:\nsudo apt install libz-dev\033[0m' >&2)
else

ifeq (yes, $(HAS_YUM))
	TIP := $(shell echo -e '\033[33mTry to install it by typing:\nsudo yum install zlib-devel\033[0m' >&2)
else
	TIP := $(shell echo -e '\033[33mPlase, install zlib developer files.\nFor more info see: https://www.zlib.net\033[0m' >&2)
endif

endif
        $(error 'aborting...')
endif
NOBUILD_GOALS := clean rebuildclean distclean uninstall

-include $(SRCPATH).make-build-info

OPTIMIZATION?=-O2
OPT=$(OPTIMIZATION)
CSTD=gnu99 -pedantic
CFLAGS=-std=$(CSTD) -Wall -W -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-label -Wno-unused-but-set-variable
ifeq (yes, $(IS_CLANG))
        CFLAGS+=-Wno-string-compare
        CFLAGS+=-Wno-unused-command-line-argument
endif
CFLAGS+=$(OPT)
LDFLAGS=-lz -lm -ldl
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
WAND_CONFIG=Wand-config
HAS_MAGICK=false
MAGICK_VERSION=none
MAGICK_VERSION_MAJOR=none
CONFIGMK := $(SRCPATH)config.mk

TMPDIR ?= $(shell test -e /tmp && echo /tmp || (mkdir -p $(PSYCPATH)tmp/ &>/dev/null && test -e $(PSYCPATH)tmp && echo $(PSYCPATH)tmp))
gen_conf_mk := $(shell sh -c 'cd $(PSYCPATH) && ./conf.sh > $(TMPDIR)/psyc_conf.log 2>&1')
conf_mk_exists := $(shell test -e $(CONFIGMK) && echo true)

ifneq (true, $(conf_mk_exists))
        $(error FATAL: Could not find $(CONFIGMK): take a look at logs in '$(TMPDIR)/psyc_conf.log' and try to manually run ./conf.sh)
endif

include $(CONFIGMK)
include $(SRCPATH)avx.mk

ifeq (off,$(ACCELERATE))
        HAS_ACCELERATE_FRAMEWORK=false
endif

ifeq (true,$(HAS_ACCELERATE_FRAMEWORK))
        CFLAGS+=$(ACCELERATE_CFLAGS)
        LDFLAGS+=$(ACCELERATE_LDFLAGS)
else
ifeq (true, $(BLAS_NEEDS_ACCELERATE))
        BLAS_CFLAGS=-DHAS_BLAS
        USE_PSYC_BLAS=on
else
ifneq (off,$(BLAS))
ifneq (true,$(HAS_CBLAS))
        BLAS_CFLAGS=-DHAS_BLAS
        USE_PSYC_BLAS=on
endif
endif
endif
endif

ifneq (off,$(BLAS))
        CFLAGS+=$(BLAS_CFLAGS)
        LDFLAGS+=$(BLAS_LDFLAGS)

ifeq (on, $(USE_PSYC_BLAS))
	CFLAGS+=-DUSE_PSYC_BLAS
endif

endif

OBJS=$(SRCPATH)psyc.o $(SRCPATH)config.o $(SRCPATH)io.o $(SRCPATH)utils.o $(SRCPATH)log.o $(SRCPATH)maths.o $(SRCPATH)activation.o $(SRCPATH)blas.o $(SRCPATH)optimization.o $(SRCPATH)convolutional.o $(SRCPATH)recurrent.o $(SRCPATH)lstm.o $(SRCPATH)gru.o $(SRCPATH)dropout.o $(SRCPATH)embedding.o $(SRCPATH)normalization.o $(SRCPATH)attention.o $(SRCPATH)operator_layer.o $(SRCPATH)positional_encoding.o $(SRCPATH)debug.o $(SRCPATH)dataset.o $(SRCPATH)utf8.o

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=$(SRCPATH)avx.o
ifeq ($(HAS_AVX512),true)
	CFLAGS+=-DHAS_AVX512 -mavx512f
endif
endif

ifeq ($(PLATFORM), Linux)
        CFLAGS+=-fdiagnostics-color -Wno-unused-result -Wno-maybe-uninitialized
        CFLAGS+=-fPIC -Wno-unused-but-set-variable
endif

HAS_READLINE=no
ifneq ($(READLINE), off)
        HAS_READLINE := $(shell sh -c '$(PSYCPATH)utils/has_lib.sh "readline/readline" $(CC) && echo yes')
endif

ifeq ($(MAGICK), off)
        HAS_MAGICK=false
else
        MAGICK_VERSION := $(shell sh -c '$(WAND_CONFIG) --version 2>/dev/null || echo none')
ifeq ($(MAGICK_VERSION), none)
        WAND_CONFIG=MagickWand-config
        MAGICK_VERSION := $(shell sh -c '$(WAND_CONFIG) --version 2>/dev/null || echo none')
endif
ifneq ($(MAGICK_VERSION), none)
        HAS_MAGICK=true
endif
endif

ifeq ($(DOUBLE_PRECISION),on)
	CFLAGS+=-DPS_DOUBLE_PRECISION
endif

ifeq ($(DEBUG_MODE),on)
	CFLAGS+=-DPS_DEBUG_MODE
endif

ifneq ($(COLORS),off)
	ACTION_COLOR="\033[1m"
	NAME_COLOR="\033[36m"
	LINK_ACTION_COLOR="\033[32m"
	UNINSTALL_COLOR="\033[35m"
	END_COLOR="\033[0m"
else
	ACTION_COLOR=""
	NAME_COLOR=""
	END_COLOR=""
	LINK_ACTION_COLOR=""
	UNINSTALL_COLOR=""
endif

COMPILE_ACTION_NAME = Compiling
LINK_ACTION_NAME    = Linking
OUTPUT_OPTION ?= -o $@

print-action = @printf '%b %b\n' $(ACTION_COLOR)$(3)$(1) $(NAME_COLOR)$(notdir $(2))$(END_COLOR)

.PHONY: print-action

.save-build-info:
	@cd $(PSYCPATH) && $(MAKE) rebuildclean
	@cd $(SRCPATH) && ./genbuildinfo.sh --optimization "$(OPT)" --prefix "$(PREFIX)"
	@echo PREV_CFLAGS='$(CFLAGS)' > $(SRCPATH).make-build-info
	@echo PREV_LDFLAGS='$(LDFLAGS)' >> $(SRCPATH).make-build-info

.PHONY: .save-build-info

.make-prerequisites:
	@touch $@

ifneq ($(strip $(PREV_CFLAGS)), $(strip $(CFLAGS)))
.make-prerequisites: .save-build-info
endif

ifneq ($(strip $(PREV_LDFLAGS)), $(strip $(LDFLAGS)))
.make-prerequisites: .save-build-info
endif

%.o: %.c .make-prerequisites
	$(call print-action,$(COMPILE_ACTION_NAME),$<)
	$(COMPILE.c) $(OUTPUT_OPTION) $<

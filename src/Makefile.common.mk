SHELL=/bin/bash
CC=gcc
OPTIMIZATION?=-O2
OPT=$(OPTIMIZATION)
CSTD=gnu99 -pedantic
CFLAGS=-std=$(CSTD) -Wall -W -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-label -Wno-string-compare -Wno-unused-command-line-argument $(OPT)
LDFLAGS=-lz -lm -ldl
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not_found')
HARDWARE := $(shell sh -c 'uname -m 2>/dev/null || echo not_found')
SRCPATH := $(strip $(dir $(lastword $(MAKEFILE_LIST))))
PSYCPATH := $(SRCPATH)../
build_info_h := $(shell sh -c '$(SRCPATH)genbuildinfo.sh')
WAND_CONFIG=Wand-config
HAS_MAGICK=false
MAGICK_VERSION=none
MAGICK_VERSION_MAJOR=none
CONFIGMK=$(SRCPATH)config.mk
ifeq (,$(wildcard $(CONFIGMK)))
        $(error "Could not find $(CONFIGMK): try to manually run ./conf.sh")
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

OBJS=$(SRCPATH)psyc.o $(SRCPATH)config.o $(SRCPATH)io.o $(SRCPATH)utils.o $(SRCPATH)log.o $(SRCPATH)maths.o $(SRCPATH)activation.o $(SRCPATH)blas.o $(SRCPATH)optimization.o $(SRCPATH)convolutional.o $(SRCPATH)recurrent.o $(SRCPATH)lstm.o $(SRCPATH)gru.o $(SRCPATH)dropout.o $(SRCPATH)mnist.o $(SRCPATH)debug.o $(SRCPATH)cifar.o

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=$(SRCPATH)avx.o
ifeq ($(HAS_AVX512),true)
	CFLAGS+=-DHAS_AVX512 -mavx512f
endif
endif

ifeq ($(PLATFORM), Linux)
        CFLAGS+=-fdiagnostics-color -Wno-unused-result -Wno-maybe-uninitialized
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

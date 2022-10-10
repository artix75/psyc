SHELL=/bin/bash
CC=gcc
OPTIMIZATION?=-O2
OPT=$(OPTIMIZATION)
CSTD=gnu99 -pedantic
CFLAGS=-std=$(CSTD) -Wall -W -Wno-missing-field-initializers -Wno-unknown-pragmas $(OPT)
LDFLAGS=-lz -lm -ldl
OBJS=psyc.o utils.o convolutional.o recurrent.o lstm.o mnist.o debug.o cifar.o
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not_found')
HARDWARE := $(shell sh -c 'uname -m 2>/dev/null || echo not_found')
WAND_CONFIG=Wand-config
HAS_MAGICK=false
MAGICK_VERSION=none
MAGICK_VERSION_MAJOR=none

ifeq ($(PLATFORM), Linux)
        CFLAGS+=-fdiagnostics-color -Wno-unused-result
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

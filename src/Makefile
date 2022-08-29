SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -Wall -W -Wno-missing-field-initializers
LDFLAGS=-lz -lm
OBJS=psyc.o utils.o convolutional.o recurrent.o lstm.o mnist.o debug.o
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not')
LIBNAME := libpsyc
WAND_CONFIG=Wand-config
HAS_MAGICK=false
MAGICK_VERSION=none
MAGICK_VERSION_MAJOR=none

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

ifeq ($(PLATFORM), Linux)
        LIBNAME := $(LIBNAME).so
        LIBOPTS := -shared -fPIC -Wl,-soname,$(LIBNAME)
        CFLAGS+=-fPIC -Wno-unused-but-set-variable
endif

ifeq ($(PLATFORM), Darwin)
        LIBNAME := $(LIBNAME).dylib
        LIBOPTS := -dynamiclib -install_name $(LIBDIR)/$(LIBNAME)
endif

include avx.mk

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        OBJS+=avx.o
endif

BIN_CFLAGS = $(CFLAGS)
BIN_LDFLAGS = $(LDFLAGS)
CLI_OBJS=$(OBJS) psycl.o

ifneq ($(HAS_MAGICK), false)
        BIN_CFLAGS+=-DHAS_MAGICK
        BIN_LDFLAGS+=$(shell sh -c '$(WAND_CONFIG) --ldflags')
        MAGICK_CFLAGS := $(shell sh -c '$(WAND_CONFIG) --cflags')
        CLI_OBJS+=image_data.o
        MAGICK_VERSION := $(shell echo $(MAGICK_VERSION) | cut -d ' ' -f 1)
        MAGICK_VERSION_MAJOR := $(shell echo $(MAGICK_VERSION) | cut -d '.' -f 1)
endif

default: all

image_data.o:
	@echo '#define PSYCH_MAGICK_VERSION $(MAGICK_VERSION_MAJOR)' > magick_conf.h
	$(CC) $(MAGICK_CFLAGS) $(CFLAGS)  -c -o $@ image_data.c

psycl.o: 
	$(CC) $(BIN_CFLAGS) -c -o $@ psycl.c

$(LIBNAME): $(OBJS)
	$(CC) $(LIBOPTS) -o ../lib/$@ $(OBJS) $(LDFLAGS)

psycl: $(CLI_OBJS)
	$(CC) -o ../bin/psycl $(CLI_OBJS) $(BIN_LDFLAGS)

all: psycl $(LIBNAME)

install: all
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)/psyc
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(SHAREDIR)/bin
	@mkdir -p $(SHAREDIR)/utils
	cp ../bin/psycl $(SHAREDIR)/bin/psycl
	ln -s -f $(SHAREDIR)/bin/psycl $(BINDIR)/psycl
	cp ../lib/$(LIBNAME) $(LIBDIR)/$(LIBNAME)
	cp psyc.h $(INCLUDEDIR)/psyc/
	cp mnist.h $(INCLUDEDIR)/psyc/
	cp image_data.h $(INCLUDEDIR)/psyc/
	cp -r ../resources $(SHAREDIR)/resources
	cp -r ../utils/*.rb $(SHAREDIR)/utils/
	#cp ../bin/*_demo $(SHAREDIR)/bin/
	@if [ "$(PLATFORM)" = "Linux" ]; then /sbin/ldconfig; fi

uninstall:
	rm -rvf $(SHAREDIR)
	rm -f $(BINDIR)/psycl
	rm -f $(LIBDIR)/$(LIBNAME)
	rm -rvf $(INCLUDEDIR)/psyc

show-build-info:
	@echo "PREFIX=$(PREFIX)"
	@echo "AVX=$(AVX)"
	@echo "WAND_CONFIG=$(WAND_CONFIG)"
	@echo "HAS_MAGICK=$(HAS_MAGICK)"
	@echo "MAGICK_VERSION=$(MAGICK_VERSION)"
	@echo "MAGICK_VERSION_MAJOR=$(MAGICK_VERSION_MAJOR)"

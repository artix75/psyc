SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99 -Wall -W -Wno-missing-field-initializers
LDFLAGS=-lz -lm
OBJS=psyc.o utils.o convolutional.o recurrent.o mnist.o
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not')
LIBNAME := libpsyc
HAS_MAGICK := $(shell sh -c 'Wand-config --version 2>/dev/null || echo false')

ifeq ($(MAGICK), off)
        HAS_MAGICK := false
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
CLI_OBJS=$(OBJS) psyc_cli.o

ifneq ($(HAS_MAGICK), false)
        BIN_CFLAGS+=-DHAS_MAGICK
        BIN_LDFLAGS+=$(shell sh -c 'Wand-config --ldflags')
        MAGICK_CFLAGS := $(shell sh -c 'Wand-config --cflags')
        CLI_OBJS+=image_data.o
endif

default: all

image_data.o:
	$(CC) $(MAGICK_CFLAGS) $(CFLAGS)  -c -o $@ image_data.c

psyc_cli.o: 
	$(CC) $(BIN_CFLAGS) -c -o $@ psyc_cli.c

$(LIBNAME): $(OBJS)
	$(CC) $(LIBOPTS) -o ../lib/$@ $(OBJS) $(LDFLAGS)

psyc_cli: $(CLI_OBJS)
	$(CC) -o ../bin/psyc_cli $(CLI_OBJS) $(BIN_LDFLAGS)

all: psyc_cli $(LIBNAME)

install: all
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(SHAREDIR)/bin
	@mkdir -p $(SHAREDIR)/utils
	cp ../bin/psyc_cli $(SHAREDIR)/bin/psyc_cli
	ln -s -f $(SHAREDIR)/bin/psyc_cli $(BINDIR)/psyc_cli
	cp ../lib/$(LIBNAME) $(LIBDIR)/$(LIBNAME)
	cp psyc.h $(INCLUDEDIR)/psyc.h
	cp mnist.h $(INCLUDEDIR)/psyc_mnist.h
	cp -r ../resources $(SHAREDIR)/resources
	cp -r ../utils/*.rb $(SHAREDIR)/utils/
	#cp ../bin/*_demo $(SHAREDIR)/bin/
	@if [ "$(PLATFORM)" = "Linux" ]; then /sbin/ldconfig; fi

uninstall:
	rm -rvf $(SHAREDIR)
	rm -f $(BINDIR)/psyc_cli
	rm -f $(LIBDIR)/$(LIBNAME)
	rm -f $(INCLUDEDIR)/psyc.h
	rm -f $(INCLUDEDIR)/psyc_mnist.h

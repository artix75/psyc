SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu99
LDFLAGS=-lz -lm
psyc_OBJS=psyc.o mnist.o psyc_cli.o
PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include
SHAREDIR=$(PREFIX)/share/psyc
PLATFORM := $(shell sh -c 'uname -s 2>/dev/null || echo not')
LIBNAME := libpsyc

ifeq ($(PLATFORM), Linux)
        LIBNAME := $(LIBNAME).so
        LIBOPTS := -shared -fPIC -Wl,-soname,$(LIBNAME)
        CFLAGS+=-fPIC
endif

ifeq ($(PLATFORM), Darwin)
        LIBNAME := $(LIBNAME).dylib
        LIBOPTS := -dynamiclib -install_name $(LIBDIR)/$(LIBNAME)
endif

include avx.mk

ifeq ($(AVX),on)
	CFLAGS+=-DUSE_AVX -mavx2 -mfma
        psyc_OBJS+=avx.o
endif

$(LIBNAME): $(psyc_OBJS)
	$(CC) $(LIBOPTS) -o ../lib/$@ $(psyc_OBJS) $(LDFLAGS)

psyc_cli: $(psyc_OBJS)
	$(CC) -o ../bin/psyc_cli $(psyc_OBJS) $(LDFLAGS)
all: psyc_cli $(LIBNAME)

install: all
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(SHAREDIR)/bin
	cp ../bin/psyc_cli $(BINDIR)/psyc_cli
	cp ../lib/$(LIBNAME) $(LIBDIR)/$(LIBNAME)
	cp psyc.h $(INCLUDEDIR)/psyc.h
	cp mnist.h $(INCLUDEDIR)/psyc_mnist.h
	cp -r ../resources $(SHAREDIR)/resources
	[ $(PLATFORM) = "Linux" ] && /sbin/ldconfig -v

uninstall:
	rm -rvf $(SHAREDIR)
	rm -f $(BINDIR)/psyc_cli
	rm -f $(LIBDIR)/$(LIBNAME)
	rm -f $(INCLUDEDIR)/psyc.h
	rm -f $(INCLUDEDIR)/psyc_mnist.h

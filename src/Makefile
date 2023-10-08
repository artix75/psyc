include Makefile.common.mk

LIBNAME=libpsyc
STATIC_LIBNAME := $(LIBNAME).a
ifeq ($(PLATFORM), Linux)
        LIBNAME := $(LIBNAME).so
        LIBOPTS := -shared -fPIC -Wl,-soname,$(LIBNAME)
endif

ifeq ($(PLATFORM), Darwin)
        LIBNAME := $(LIBNAME).dylib
        LIBOPTS := -dynamiclib -install_name $(LIBDIR)/$(LIBNAME)
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

HEADER_FILE_LIST = .make-header-files
HEADER_FILES := $(strip $(subst all.h,,$(sort $(notdir $(wildcard *.h)))))

$(HEADER_FILE_LIST):
	@echo 'PREV_HEADER_FILES = $(HEADER_FILES)' > $@

-include $(HEADER_FILE_LIST)

ifneq ($(strip $(PREV_HEADER_FILES)), $(strip $(HEADER_FILES)))
.PHONY: $(HEADER_FILE_LIST)
endif

HASH := \#
FINAL_HEADER_FILES := psyc.h $(subst psyc.h,,$(HEADER_FILES))
ALL_H_INCLUDE_DIRECTIVES := $(foreach hfile,$(FINAL_HEADER_FILES),echo '$(HASH)include "$(hfile)"' >> all.h;)

PSYCLIB := ../lib/$(LIBNAME)
PSYC_STATIC_LIB := ../lib/$(STATIC_LIBNAME)
PSYCL := ../bin/psycl
PSYC_CC_UTIL = ../bin/psyc-cc.sh

default: all

Makefile.dep:
	-$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

all.h: $(HEADER_FILE_LIST) $(PSYCPATH)resources/license_header.h
	$(call print-action,Creating,$@,$(LINK_ACTION_COLOR))
	@cat $(PSYCPATH)resources/license_header.h > $@
	@echo '' >> $@
	@echo "#ifndef __PS_ALL_H__" >> $@
	@echo "#define __PS_ALL_H__" >> $@
	@echo '' >> $@
	@$(ALL_H_INCLUDE_DIRECTIVES)
	@echo '' >> $@
	@echo "#endif /* __PS_ALL_H__ */" >> $@

image_data.o:
	@echo '#define PSYCH_MAGICK_VERSION $(MAGICK_VERSION_MAJOR)' > magick_conf.h
	$(call print-action,$(COMPILE_ACTION_NAME),image_data.c)
	$(CC) $(MAGICK_CFLAGS) $(CFLAGS)  -c -o $@ image_data.c

psycl.o: psycl.c
	$(call print-action,$(COMPILE_ACTION_NAME),psycl.c)
	$(CC) $(BIN_CFLAGS) -c -o $@ psycl.c

$(PSYCLIB): $(OBJS)
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) $(LIBOPTS) -o $@ $(OBJS) $(LDFLAGS)

$(PSYCL): $(CLI_OBJS)
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(CC) -o $@ $(CLI_OBJS) $(BIN_LDFLAGS)

$(PSYC_STATIC_LIB): $(OBJS)
	$(call print-action,$(LINK_ACTION_NAME),$@,$(LINK_ACTION_COLOR))
	$(AR) crs $@ $^

$(PSYC_CC_UTIL): ../resources/psyc-cc.sh ../resources/license-header.h
	$(call print-action,Creating,$@,$(LINK_ACTION_COLOR))
	@echo '#!/bin/bash' > $@
	@cat ../resources/license-header.sh >> $@
	@echo '' >> $@
	@echo "CC='$(CC)'" >> $@
	@echo "MAGICK_CFLAGS='$(MAGICK_CFLAGS)'" >> $@
	@echo "BLAS_CFLAGS='$(BLAS_CFLAGS)'" >> $@
	@echo "ACCELERATE_CFLAGS='$(ACCELERATE_CFLAGS)'" >> $@
	@echo "CFLAGS='$(CFLAGS)'" >> $@
	@echo "LDFLAGS='$(LDFLAGS)'" >> $@
	@echo "PREFIX='$(PREFIX)'" >> $@
	@echo "LIBDIR='$(LIBDIR)'" >> $@
	@echo "INCLUDEDIR='$(INCLUDEDIR)'" >> $@
	@echo "LIBNAME='$(LIBNAME)'" >> $@
	@echo "STATIC_LIBNAME='$(STATIC_LIBNAME)'" >> $@
	@cat ../resources/psyc-cc.sh >> $@
	@chmod +x $@

static: $(PSYC_STATIC_LIB)

all: $(PSYCL) $(PSYCLIB) $(PSYC_STATIC_LIB) $(PSYC_CC_UTIL)

install: all all.h
	$(call print-action,Installing,PsyC,$(LINK_ACTION_COLOR))
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)/psyc
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(SHAREDIR)/bin
	@mkdir -p $(SHAREDIR)/utils
	cp $(PSYCL) $(SHAREDIR)/bin/psycl
	ln -s -f $(SHAREDIR)/bin/psycl $(BINDIR)/psycl
	-cp $(PSYC_CC_UTIL) $(SHAREDIR)/bin/$(notdir $(PSYC_CC_UTIL))
	-ln -s -f $(SHAREDIR)/bin/$(notdir $(PSYC_CC_UTIL)) $(BINDIR)/$(notdir $(basename $(PSYC_CC_UTIL)))
	cp $(PSYCLIB) $(LIBDIR)/$(LIBNAME)
	cp $(PSYC_STATIC_LIB) $(LIBDIR)/$(STATIC_LIBNAME)
	cp *.h $(INCLUDEDIR)/psyc/
	cp -r ../resources $(SHAREDIR)/resources
	cp -r ../utils/*.rb $(SHAREDIR)/utils/
#	cp ../bin/*_demo $(SHAREDIR)/bin/
	@if [ "$(PLATFORM)" = "Linux" ]; then /sbin/ldconfig; fi

uninstall:
	$(call print-action,Uninstalling,PsyC,$(UNINSTALL_COLOR))
	rm -rvf $(SHAREDIR)
	rm -f $(BINDIR)/psycl
	rm -f $(BINDIR)/$(notdir $(basename $(PSYC_CC_UTIL)))
	rm -f $(LIBDIR)/$(LIBNAME)
	rm -rvf $(INCLUDEDIR)/psyc

show-build-conf:
	@echo "CC=$(CC)"
	@echo "IS_CLANG=$(IS_CLANG)"
	@echo "PREFIX=$(PREFIX)"
	@echo "DOUBLE_PRECISION=$(DOUBLE_PRECISION)"
	@echo "AVX=$(AVX)"
	@echo "HAS_BLAS=$(HAS_BLAS)"
	@echo "HAS_CBLAS=$(HAS_CBLAS)"
	@echo "HAS_GSL_CBLAS=$(HAS_GSL_CBLAS)"
	@echo "HAS_ACCELERATE_FRAMEWORK=$(HAS_ACCELERATE_FRAMEWORK)"
	@echo "BLAS_NEEDS_ACCELERATE=$(BLAS_NEEDS_ACCELERATE)"
	@echo "WAND_CONFIG=$(WAND_CONFIG)"
	@echo "HAS_MAGICK=$(HAS_MAGICK)"
	@echo "MAGICK_VERSION=$(MAGICK_VERSION)"
	@echo "MAGICK_VERSION_MAJOR=$(MAGICK_VERSION_MAJOR)"

list-available-options:
	@echo "Option           Argument    Description         Default"
	@echo "=============================================================="
	@echo "PREFIX           PATH        Installation path   $(PREFIX)"
	@echo "OPTIMIZATION     OPT         Optimization        $(OPTIMIZATION)"
	@echo "AVX              (on|off)    AVX support         $(AVX)"
	@echo "ACCELERATE       (on|off)    Apple Accelerate"
	@echo "                             Framework support   $(ACCELERATE)"
	@echo "BLAS             (on|off)    BLAS support        $(BLAS)"
	@echo "USE_PSYC_BLAS    (on|off)    Force native BLAS   $(USE_PSYC_BLAS)"
	@echo "MAGICK           (on|off)    ImageMagick support $(MAGICK)"
	@echo "TESTS            TESTS       Only perform specific tests"
	@echo "BUILD_ONLY                   Only build, do not execute (ie. tests)"
	@echo "COLORS           (on|off)    Colorized output (default 'on')""

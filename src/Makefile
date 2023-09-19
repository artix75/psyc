include Makefile.common.mk

LIBNAME=libpsyc
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

PSYCLIB := ../lib/$(LIBNAME)
PSYCL := ../bin/psycl

default: all

Makefile.dep:
	-$(CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NOBUILD_GOALS))))
-include Makefile.dep
endif

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

all: $(PSYCL) $(PSYCLIB)

install: all
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)/psyc
	@mkdir -p $(SHAREDIR)
	@mkdir -p $(SHAREDIR)/bin
	@mkdir -p $(SHAREDIR)/utils
	cp $(PSYCL) $(SHAREDIR)/bin/psycl
	ln -s -f $(SHAREDIR)/bin/psycl $(BINDIR)/psycl
	cp $(PSYCLIB) $(LIBDIR)/$(LIBNAME)
	cp *.h $(INCLUDEDIR)/psyc/
	cp -r ../resources $(SHAREDIR)/resources
	cp -r ../utils/*.rb $(SHAREDIR)/utils/
#	cp ../bin/*_demo $(SHAREDIR)/bin/
	@if [ "$(PLATFORM)" = "Linux" ]; then /sbin/ldconfig; fi

uninstall:
	rm -rvf $(SHAREDIR)
	rm -f $(BINDIR)/psycl
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

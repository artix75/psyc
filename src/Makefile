include Makefile.common.mk

LIBNAME=libpsyc
ifeq ($(PLATFORM), Linux)
        LIBNAME := $(LIBNAME).so
        LIBOPTS := -shared -fPIC -Wl,-soname,$(LIBNAME)
        CFLAGS+=-fPIC -Wno-unused-but-set-variable
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

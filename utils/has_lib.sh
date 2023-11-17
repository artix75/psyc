#!/bin/bash

LIBNAME=$1
if [ -z "$LIBNAME" ]; then
    echo "Usage: $0 LIBNAME [CC]" 1>&2
    exit 1
fi
CC=$2

TMPPATH="/tmp/"
if ! [ -e "$TMPPATH" ]; then
    SCRIPTPATH=""
    TMPPATH=""
    which dirname 2>/dev/null && SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    if ! [ -z "$SCRIPTPATH" ]; then
        TMPPATH="$SCRIPTPATH/tmp"
        if ! [ -e "$TMPPATH" ]; then
            mkdir -p "$TMPPATH"
        fi
    fi
    if [ -z "$TMPPATH" ]; then
        echo "Could not find a temporary path" 1>&2
        exit 1
    fi
fi

if [ -z "$CC" ]; then
    CC=gcc
    which "$CC" > /dev/null 2>&1 || CC=clang
    if ! which "$CC" > /dev/null 2>&1; then
        echo "Could not C compiler, explicitely specify it:" 1>&2
        echo "Usage: $0 LIBNAME [CC]" 1>&2
        exit 1
    fi
fi
SRCLIBNAME=$(echo "$LIBNAME" | tr '/' '-')
SRCLIBNAME=$(echo "$SRCLIBNAME" | tr ' ' '-')
EXCPATH="$TMPPATH/psyc-test-$SRCLIBNAME-$RANDOM"
SRCPATH="$EXCPATH.c"
echo "#include <$LIBNAME.h>" > "$SRCPATH"
echo "int main(int argc, char **argv) {return 0;}" >> "$SRCPATH"
HAS_LIB=yes
$CC -o "$EXCPATH" "$SRCPATH" > /dev/null 2>&1 || HAS_LIB=no
rm -f "$SRCPATH"
rm -f "$EXCPATH"
if [ "$HAS_LIB" = 'no' ]; then
    exit 1
fi

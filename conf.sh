#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OS=$(uname -s)
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
TMPDIR="$SCRIPT_DIR/tmp"
CC=`which gcc`
if ! [ -e "$TMPDIR" ]; then
    mkdir -p "$TMPDIR"
fi

printErr() {
    err=$1
    echo "ERROR[$SCRIPT_NAME]: $err" 2>&1
}

hasFramework() {
    framework=$1
    src="$TMPDIR/psyc-testcompile-$RANDOM.c"
    out="$src.o"
    res=1
    echo 'int main(int c, char **a){return 0;}' > "$src"
    if gcc -o "$out" -framework $framework "$src" 2>/dev/null; then
        res=0
    fi
    rm -rf "$src"
    rm -rf "$out"
    return $res
}

if [ -z "$CC" ]; then
    printErr "gcc was not found on your system" 
    exit 1
fi

FIND_C_HEADER="$SCRIPT_DIR/utils/find_c_headers.sh"
C_HEADERS_PATH="$SCRIPT_DIR/.c_headers"
if ! [ -f "$C_HEADERS_PATH" ]; then
    echo "$SCRIPT_NAME: Search for system C headers..."
    $FIND_C_HEADER > "$C_HEADERS_PATH"
fi

C_HEADERS=''
if ! [ -f "$C_HEADERS_PATH" ]; then
    echo  "WARN[$SCRIPT_NAME]: failed to generate $C_HEADERS_PATH"
    #exit 1
else
    C_HEADERS=$(cat "$C_HEADERS_PATH")
fi

CFLAGS=''
LDFLAGS=''

VECLIB_H_DIR=''
VECLIB_LD_DIR=''
CBLAS_H_DIR=''
CBLAS_LIB_DIR=''
HAS_ACCELERATE_FRAMEWORK='' 
HAS_CBLAS=''
HAS_GSL_CBLAS=''
ACCELERATE_CFLAGS=''
ACCELERATE_LDFLAGS=''
BLAS_LDFLAGS=''
BLAS_CFLAGS=''
BLAS_NEEDS_ACCELERATE=''
if [ "$OS" = "Darwin" ]; then
    # Check whether gcc is clang
    CLANG_VERS=$(gcc --version | grep -i clang)
    if ! [ -z "$CLANG_VERS" ]; then
        if hasFramework 'Accelerate'; then
            HAS_ACCELERATE_FRAMEWORK='true'
            HAS_CBLAS='true'
            BLAS_NEEDS_ACCELERATE='true'
            ACCELERATE_LDFLAGS='-framework Accelerate'
        fi
    else
        # Search for vecLib framework
        FRAMEWORKS_LIB="/System/Library/Frameworks"
        VECLIB_H=$(echo "$C_HEADERS" | grep -E 'vecLib\.h$' | grep 'Accelerate')
        if [ -z "$VECLIB_H" ]; then
            VECLIB_H=$(echo "$C_HEADERS" | grep -E 'vecLib\.h$')
        fi
        if ! [ -z "$VECLIB_H" ]; then
            VECLIB_H=$(echo "$VECLIB_H" | head -n 1)
            VECLIB_H_DIR=$(dirname "$VECLIB_H")
            VECLIB_LIB=$(find "$FRAMEWORKS_LIB" -type lf -name 'vecLib' | grep 'Accelerate')
            if [ -z "$VECLIB_LIB" ]; then
                VECLIB_LIB=$(find "$FRAMEWORKS_LIB" -type lf -name 'vecLib' | head -n 1)
            fi
            if [ -z "$VECLIB_LIB" ]; then
                echo "$SCRIPT_NAME: WARN: Could not find vecLib"
                VECLIB_H_DIR=''
            else
                BLAS_CFLAGS="-I${VECLIB_H_DIR}"
                VECLIB_LD_DIR=$(dirname "$VECLIB_LIB")
                BLAS_LDFLAGS="-L${VECLIB_LD_DIR} $LDFLAGS"
                CBLAS_H="$VECLIB_H_DIR/cblas.h"
                if ! [ -e "$CBLAS_H" ]; then
                    echo "$SCRIPT_PATH: WARN: cblas.h not found"
                    CBLAS_H=''
                else
                    CBLAS_LIB="$VECLIB_LD_DIR/libBLAS.dylib"
                    if ! [ -e "$CBLAS_LIB" ] && ! [ -L "$CBLAS_LIB" ]; then
                        echo "$SCRIPT_NAME: WARN: $CBLAS_LIB not found!"
                        CBLAS_H=''
                    else
                        BLAS_LDFLAGS="-lBLAS $BLAS_LDFLAGS"
                    fi
                fi
            fi
        fi
    fi
fi

# Search for BLAS library
if [ -z "$CBLAS_H" ]; then
    CBLAS_H=$(echo "$C_HEADERS" | grep -E 'cblas\.h$')
    if ! [ -z "$CBLAS_H" ]; then
        GSL_CBLAS=$(echo "$CBLAS_H" | grep gsl)
        if ! [ -z "$GSL_CBLAS" ]; then
            CBLAS_H_DIR=$(dirname "$CBLAS_H")
            GSL_ROOTDIR=$(dirname "$CBLAS_H_DIR")
            while ! [ -z $(echo $GSL_ROOTDIR | grep include) ]; do
                GSL_ROOTDIR=$(dirname "$GSL_ROOTDIR")
            done
            GSL_LIBDIR=''
            CBLAS_LIB=''
            if [ -d "$GSL_ROOTDIR/lib" ]; then
                GSL_CBLAS=$(find /usr/lib/ -iname 'libgsl*' | grep cblas  | grep -E '(\.so|\.dylib)$' | head -1 2>&1)
                if ! [ -z "$GSL_CBLAS" ]; then
                    CBLAS_LIB="$GSL_CBLAS"
                    GSL_LIBDIR="$GSL_ROOTDIR/lib"
                fi
            fi
            if ! [ -z "$CBLAS_LIB" ]; then
                CBLAS_LIBNAME=$(basename "$CBLAS_LIB")
                CBLAS_LIBNAME="${CBLAS_LIBNAME#lib}"
                CBLAS_LIBNAME="${CBLAS_LIBNAME%.so}"
                CBLAS_LIBNAME="${CBLAS_LIBNAME%.dylib}"
                BLAS_LDFLAGS="-l$CBLAS_LIBNAME $BLAS_LDFLAGS"
                BLAS_LDFLAGS="-L$GSL_LIBDIR $BLAS_LDFLAGS"
                BLAS_CFLAGS="-DHAS_GSL_CBLAS -I$CBLAS_H_DIR $BLAS_CFLAGS"
                HAS_GSL_CBLAS=true
            else
                CBLAS_H_DIR=''
            fi
        fi
    fi
fi

VARS=''
NL='\n'
if ! [ -z "$CBLAS_H" ]; then
    HAS_BLAS=true
    HAS_CBLAS=true
fi
if [ "$HAS_ACCELERATE_FRAMEWORK" = 'true' ]; then
    HAS_BLAS=true
    HAS_CBLAS=true
    ACCELERATE_CFLAGS="-DHAS_ACCELERATE_FRAMEWORK"
fi
if [ "$HAS_BLAS" = 'true' ]; then
    VARS="HAS_BLAS=true$NL$VARS"
    BLAS_CFLAGS="-DHAS_BLAS $BLAS_CFLAGS"
    if [ "$HAS_CBLAS" = 'true' ]; then
        VARS="HAS_CBLAS=true$NL$VARS"
        BLAS_CFLAGS="-DHAS_CBLAS $BLAS_CFLAGS"
    fi
    if [ "$HAS_GSL_CBLAS" = 'true' ]; then
        VARS="HAS_GSL_CBLAS=true$NL$VARS"
    fi
fi
if [ "$HAS_ACCELERATE_FRAMEWORK" = 'true' ]; then
    VARS="HAS_ACCELERATE_FRAMEWORK=true$NL$VARS"
    if [ "$BLAS_NEEDS_ACCELERATE" = 'true' ]; then
        VARS="BLAS_NEEDS_ACCELERATE=true$NL$VARS"
    fi
fi

MKFILE="$SCRIPT_DIR/src/config.mk"
echo "Generating '$MKFILE' on $(date)"
echo "# Config generated on $(date)" > $MKFILE
printf "$VARS\n" >> $MKFILE
echo "BLAS_CFLAGS=$BLAS_CFLAGS" >> $MKFILE
echo "BLAS_LDFLAGS=$BLAS_LDFLAGS" >> $MKFILE
echo "ACCELERATE_CFLAGS=$ACCELERATE_CFLAGS" >> $MKFILE
echo "ACCELERATE_LDFLAGS=$ACCELERATE_LDFLAGS" >> $MKFILE

printHelp() {
    echo '' >&2
    echo "Usage: $0 [OPTIONS] [SOURCE] [-- GCC_OPTS]" >&2
    echo "Compile a C source file against Psyc library." >&2
    echo '' >&2
    echo "OPTIONS:" >&2
    echo '' >&2
    echo "  --is-lib                    Compile a library instead of an executable" >&2
    echo "  --cflags                    Print CFLAGS and exit" >&2
    echo "  --ldflags                   Print LDFLAGS and exit" >&2
    echo "  --c++                       SOURCE is a C++ source" >&2
    echo "  -o, --output PATH           Output path" >&2
    echo "  -s, --static                Use static library" >&2
    echo "  -p, --dry-run               Only print command whithout executing it" >&2
    echo "  -q, --quiet                 Quiet mode" >&2
    echo "  -h, --help                  Print this help" >&2
    echo '' >&2
}

ARG=$1
CUR_ARG=''
GCC_OPTS=''
USE_STATIC=false
PRINT_ONLY=false
QUIET=false
SOURCE=''
OUT=''
PRINT_CFG=''
IS_CPP=false
DO_BUILD_LIB=false
PLATFORM=$(uname -s)
while ! [ -z "$ARG" ]; do
    if [ "$ARG" = "-h" ] || [ "$ARG" = "--help" ]; then
        printHelp
        exit 1
    elif ! [ -z "$CUR_ARG" ] && [ ${ARG:0:1} = "-" ]; then
        printHelp
        echo "ERROR: invalid value for '$CUR_ARG'" >&2
    fi
    if [ "$ARG" = "--" ]; then
        if [ -z "$SOURCE" ] && [ -z "$PRINT_CFG" ]; then
            printHelp
            echo "ERROR: missing SOURCE" >&2
            exit 1
        fi
        shift
        GCC_OPTS="$@"
        break
    elif [ "$ARG" = "--static" ] || [ "$ARG" = "-s" ]; then
        USE_STATIC=true
    elif [ "$ARG" = "-p" ] || [ "$ARG" = "--dry-run" ]; then
        PRINT_ONLY=true
    elif [ "$ARG" = "--cflags" ]; then
        PRINT_CFG=cflags
    elif [ "$ARG" = "--ldflags" ]; then
        PRINT_CFG=ldflags
    elif [ "$ARG" = "--c++" ]; then
        IS_CPP=true
    elif [ "$ARG" = "--is-lib" ]; then
        DO_BUILD_LIB=true
    elif [ "$ARG" = "-q" ] || [ "$ARG" = "--quiet" ]; then
        QUIET=true
    elif [ "$ARG" = "-o" ] || [ "$ARG" = "--output" ]; then
        CUR_ARG="$ARG"
    else
        #ARGNAME=${ARG##--}
        #ARGNAME=${ARG##-}
        if ! [ ${ARG:0:1} = '-' ]; then
            if [ "$CUR_ARG" = "-o" ] || [ "$CUR_ARG" = "--output" ]; then
                OUT="$ARG"
            else
                SOURCE="$ARG"
            fi
            CUR_ARG=''
        else
            printHelp
            echo "ERROR: invalid option '$ARG'" >&2
            exit 1
        fi
    fi
    shift
    ARG=$1
done

if [ -z "$SOURCE" ] && [ -z "$PRINT_CFG" ]; then
    printHelp
    exit 1
fi

if ! [ -e "$INCLUDEDIR" ]; then
    echo "FATAL: could not find INCLUDEDIR at: $INCLUDEDIR" >&2
    exit 1
fi

if ! [ -e "$LIBDIR" ]; then
    echo "FATAL: could not find LIBDIR at: $LIBDIR" >&2
    exit 1
fi

if ! [ -e "$LIBDIR/$LIBNAME" ]; then
    if ! [ "$USE_STATIC" = true ]; then
        echo "FATAL: could not find $LIBDIR/$LIBNAME" >&2
        exit 1
    else
        USE_STATIC=true
    fi
fi

STATIC_LIB_PATH=''
OBJS="$SOURCE"
if [ "$USE_STATIC" = true ]; then
    STATIC_LIB_PATH="$LIBDIR/$STATIC_LIBNAME"
    if ! [ -e "$STATIC_LIB_PATH" ]; then
        echo "FATAL: could not find $STATIC_LIB_PATH" >&2
        exit 1
    fi
    OBJS="$OBJS $STATIC_LIB_PATH"
else
    LDFLAGS="$LDFLAGS -L${LIBDIR} -lpsyc"
fi
CFLAGS="$CFLAGS -I${INCLUDEDIR}"
if [ -z "$OUT" ]; then
    OUT=${SOURCE%.cc}
    OUT=${OUT%.c}
    OUT=${OUT%.cpp}
    if [ "$DO_BUILD_LIB" = 'true' ]; then
        if [ "$PLATFORM" = 'Darwin' ]; then
            OUT="$(dirname $OUT)/lib$(basename $OUT).dylib"
        else
            OUT="$(dirname $OUT)/lib$(basename $OUT).so"
        fi
    else
        OUT="$OUT.o"
    fi
fi
if [ "$IS_CPP" = 'true' ]; then
    CC='g++'
    CFLAGS=${CFLAGS/gnu99/c++11}
    CFLAGS=${CFLAGS/c99/c++11}
    CFLAGS=${CFLAGS/c11/c++11}
fi
if [ "$DO_BUILD_LIB" = 'true' ]; then
    if [ "$PLATFORM" = 'Linux' ]; then
        CFLAGS="$CFLAGS -shared -fPIC -Wl,-soname,$(basename $OUT)"
    elif [ "$PLATFORM" = 'Darwin' ]; then
        CFLAGS="$CFLAGS -dynamiclib -install_name $OUT"
    fi
fi
if [ "$PRINT_CFG" = 'cflags' ]; then
    echo "$CFLAGS"
    exit
elif [ "$PRINT_CFG" = 'ldflags' ]; then
    echo "$LDFLAGS"
    exit
fi
CMD="$CC $CFLAGS $GCC_OPTS $OBJS -o $OUT $LDFLAGS"
if [ "$PRINT_ONLY" = true ]; then
    echo "$CMD"
    exit
fi

if ! [ "$QUIET" = true ]; then
    echo "$CMD"
else
    CMD="$CMD &> /dev/null"
fi

SUCCESS=false
eval $CMD && SUCCESS=true
STATUS=$?
if [ "$SUCCESS" = true ]; then
    if ! [ "$QUIET" = true ]; then
        echo "Successfully compiled to $OUT"
    fi
else
    if ! [ "$QUIET" = true ]; then
        echo "ERROR: failed to compile source" >&2
    fi
    exit $STATUS
fi

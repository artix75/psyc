INCLUDEDIR="$INCLUDEDIR/psyc"
printHelp() {
    echo '' >&2
    echo "Usage: $0 [OPTIONS] SOURCE [-- GCC_OPTS]" >&2
    echo "Compile a C source file against Psyc library." >&2
    echo '' >&2
    echo "OPTIONS:" >&2
    echo '' >&2
    echo "    -o, --output PATH           Output path" >&2
    echo "    -s, --static                Use static library" >&2
    echo "    -p, --dry-run               Only print command whithout executing it" >&2
    echo "    -q, --quiet                 Quiet mode" >&2
    echo "    -h, --help                  Print this help" >&2
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
while ! [ -z "$ARG" ]; do
    if [ "$ARG" = "-h" ] || [ "$ARG" = "--help" ]; then
        printHelp
        exit 1
    elif ! [ -z "$CUR_ARG" ] && [ ${ARG:0:1} = "-" ]; then
        printHelp
        echo "ERROR: invalid value for '$CUR_ARG'" >&2
    fi
    if [ "$ARG" = "--" ]; then
        if [ -z "$SOURCE" ]; then
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

if [ -z "$SOURCE" ]; then
    printHelp
    echo "ERROR: missing SOURCE" >&2
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
    OUT="$OUT.o"
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

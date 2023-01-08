#!/bin/bash

FIND=''
SORT=false
for arg in "$@"; do
    if [ "$arg" = "--sort" ]; then
        SORT=true
    else
        FIND="$arg"
    fi
done

INCPATHS=$(echo | gcc -E -Wp,-v - 2>&1 | grep -E '^\s*\/')
#echo $INCPATHS
ALL_HEADERS=''
exp=''
if ! [ -z "$FIND" ]; then
    #fname=$(basename "$hdr")
    exp="${FIND/./\.}$"
fi
#echo $exp;exit
for ipath in $INCPATHS; do
    #echo "$ipath"
    #echo "find "$ipath" -type f -iname '*.h'"
    if ! [ -d "$ipath" ]; then
        continue
    fi
    headers=$(find "$ipath" -iname '*.h')
    for hdr in $headers; do
        if ! [ -z "$FIND" ]; then
            #fname=$(basename "$hdr")
            #echo "echo \"$hdr\" | grep -E $exp > /dev/null 2>&1 && echo true"
            found=$(echo "$hdr" | grep -E $exp > /dev/null 2>&1 && echo true)
            if [ "$found" = "true" ]; then
                echo "FOUND: '$hdr'"
                exit 0
            else
                continue
            fi
        fi
        if [ "$SORT" = 'false' ]; then
            echo $hdr
        else
            ALL_HEADERS="$ALL_HEADERS\n$hdr"
        fi
    done
done
if ! [ -z "$FIND" ]; then
    echo "Not found!" 1>&2
    exit 1
fi
if [ "$SORT" = 'true' ]; then
    sorted=$(echo "$ALL_HEADERS" | sort)
    echo $sorted
fi


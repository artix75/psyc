#!/bin/bash

script_path=`dirname "$0"`
project_path=`dirname "$script_path"`
sources=""
for src in $(find "$project_path" -name *.c); do
    dir=`dirname "$src"`
    dir=`basename "$dir"`
    if ! [ "$dir" = "tmp" ]; then
        sources="$sources $src"
    fi
done
for src in $(find "$project_path" -name *.h); do
    dir=`dirname "$src"`
    dir=`basename "$dir"`
    if ! [ "$dir" = "tmp" ]; then
        sources="$sources $src"
    fi
done
#for makefile in $(find "$project_path" -name Makefile); do
#    sources="$sources $makefile"
#done
`open $sources`

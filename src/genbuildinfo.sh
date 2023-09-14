#!/bin/sh
dir=$(basename `pwd`)
if ! [ "$dir" = "src" ]; then
    exit
fi
PREV_ARG=''
OPTIMIZATION=''
PREFIX=''
for ARG in $@; do
    test "$PREV_ARG" = "--optimization" && OPTIMIZATION="$ARG"
    test "$PREV_ARG" = "--prefix" && PREFIX="$ARG"
    PREV_ARG=$ARG
done
test -z "$OPTIMIZATION" || OPTIMIZATION=${OPTIMIZATION#-O}
GIT_SHA=`(git show-ref --head --hash=8 2> /dev/null || echo 00000000) | head -n1`
GIT_DIRTY=`git diff --no-ext-diff 2> /dev/null | wc -l`
GIT_DIRTY=`echo $GIT_DIRTY | xargs`
GIT_BRANCH=`git rev-parse --abbrev-ref HEAD 2> /dev/null || echo none`
test -f buildinfo.h || touch buildinfo.h
(cat buildinfo.h | grep SHA | grep $GIT_SHA) && \
(cat buildinfo.h | grep BRANCH | grep $GIT_BRANCH) && \
(cat buildinfo.h | grep DIRTY | grep $GIT_DIRTY) && \
(cat buildinfo.h | grep OPTIMIZATION | grep $OPTIMIZATION) && \
(cat buildinfo.h | grep PREFIX | grep $PREFIX) && exit 0 # Up-to-date
DATE=`date -R`
echo "/* Generated on: $DATE */" > buildinfo.h
echo "#ifndef __PS_BUILDINFO_H__" >> buildinfo.h
echo "#define __PS_BUILDINFO_H__" >> buildinfo.h
echo "#define PSYC_GIT_SHA \"$GIT_SHA\"" >> buildinfo.h
echo "#define PSYC_GIT_DIRTY \"$GIT_DIRTY\"" >> buildinfo.h
echo "#define PSYC_GIT_BRANCH \"$GIT_BRANCH\"" >> buildinfo.h
test -z "$OPTIMIZATION" || echo "#define PS_OPTIMIZATION \"$OPTIMIZATION\"" >> buildinfo.h
test -z "$PREFIX" || echo "#define PS_PREFIX \"$PREFIX\"" >> buildinfo.h
echo "#endif /* __PS_BUILDINFO_H__ */" >> buildinfo.h

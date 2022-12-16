#!/bin/sh
GIT_SHA=`(git show-ref --head --hash=8 2> /dev/null || echo 00000000) | head -n1`
GIT_DIRTY=`git diff --no-ext-diff 2> /dev/null | wc -l`
GIT_DIRTY=`echo $GIT_DIRTY | xargs`
GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
test -f buildinfo.h || touch buildinfo.h
(cat buildinfo.h | grep SHA | grep $GIT_SHA) && \
(cat buildinfo.h | grep BRANCH | grep $GIT_BRANCH) && \
(cat buildinfo.h | grep DIRTY | grep $GIT_DIRTY) && exit 0 # Up-to-date
DATE=`date -R`
echo "/* Generated on: $DATE */" > buildinfo.h
echo "#define PSYC_GIT_SHA \"$GIT_SHA\"" >> buildinfo.h
echo "#define PSYC_GIT_DIRTY \"$GIT_DIRTY\"" >> buildinfo.h
echo "#define PSYC_GIT_BRANCH \"$GIT_BRANCH\"" >> buildinfo.h

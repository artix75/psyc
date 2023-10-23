/*
 * Copyright (C) 2016-2023 Fabio Nicotra <artix2 at gmail dot com>.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the copyright holder. The name of the
 * copyright holder may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef __PS_UTILS_H
#define __PS_UTILS_H

#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "types.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define PSDICT_HT_SIZE 4096
#define PSDICT_UPDATE_DISABLED (1 << 0)
#define PSDictHash(key) (djb33_hash(key, 4096))
#define PSDictFromInt(n) ((PSDictValue) {.as_int = n})
#define PSDictFromFloat(n) ((PSDictValue) {.as_float = n})
#define PSDictFromPointer(ptr) ((PSDictValue) {.as_ptr = ptr})
#define PSDictFromString(str) PSDictFromPointer(str)
#define PSDictSlotForKey(key) (PSDictHash(key) % PSDICT_HT_SIZE)
#define PSDictGetStr(dict, key) ((const char *) PSDictGetPointer(dict, key))

#define PSFileExists(path) (access(path, F_OK) == 0)

/* Get elapsed time in milliseconds */
#define PSGetElapsedTimeMS(st, et) ((((et.tv_sec - st.tv_sec) * 1000000) \
/* Get elapsed time in microseconds */
#define PSGetElapsedTimeUS(st, et) (((et.tv_sec - st.tv_sec) * 1000000) \
    + (et.tv_usec - st.tv_usec))

#define OPT_TIME_LONG        (1 << 0)
#define OPT_TIME_FULL        (1 << 1)
#define OPT_TIME_HUMAN       (1 << 2)

#define PS_BITMAP_OP_AND 1
#define PS_BITMAP_OP_OR  2
#define PS_BITMAP_OP_XOR 3

#define PSBitmapAnd(a, b, dest) PSBitmapOp(a, b, dest, PS_BITMAP_OP_AND)
#define PSBitmapOr(a, b, dest) PSBitmapOp(a, b, dest, PS_BITMAP_OP_OR)

/* Dictionary */

struct PSDictItem;
struct PSDict;
struct PSDictIterator;
typedef void (* PSOnDictItemRelease) (struct PSDictItem *);

typedef union {
    int64_t    as_int;
    PSFloat    as_float;
    void       *as_ptr;
} PSDictValue;

typedef struct PSDictItem {
    const char          *key;
    PSDictValue         value;
    struct PSDictItem   *prev;
    struct PSDictItem   *next;
    struct PSDict       *dict;
    int                 slot;
} PSDictItem;

typedef struct PSDict {
    int64_t             length;
    int                 flags;
    PSDictItem          *table[PSDICT_HT_SIZE];
    PSOnDictItemRelease onItemRelease;
} PSDict;

typedef struct PSDictIterator {
    PSDict      *dict;
    PSDictItem  *current;
} PSDictIterator;

PSDict *PSDictCreate(int flags);
void PSDictClear(PSDict *dict);
PSDictItem *PSDictGet(PSDict *dict, const char *key);
void *PSDictGetPointer(PSDict *dict, const char *key);
int PSDictHasKey(PSDict *dict, const char *key);
PSDictItem *PSDictSet(PSDict *dict, const char *key, PSDictValue val);
PSDictItem *PSDictGetOrSet(PSDict *dict, const char *key, PSDictValue val);
void PSDictRemove(PSDict *dict, const char *key);
const char **PSDictGetKeys(PSDict *dict);
PSDictItem **PSDictGetItems(PSDict *dict);
struct PSDictIterator *PSDictIteratorCreate(PSDict *dict);
PSDictItem *PSDictNext(PSDictIterator *iterator);
void PSDictFree(PSDict *dict);

/* Bitmaps */

typedef uint64_t *PSBitmap;
PSBitmap PSBitmapCreate(size_t size);
int PSBitmapCopy(PSBitmap dst, PSBitmap src);
PSBitmap PSBitmapDup(PSBitmap src);
void PSBitmapRelease(PSBitmap bitmap);
size_t PSBitmapSize(PSBitmap bitmap);
int PSBitmapGetBit(PSBitmap bitmap, uint64_t index);
int PSBitmapSetBit(PSBitmap bitmap, uint64_t index, int val);
void PSBitmapClear(PSBitmap bitmap);
PSBitmap PSBitmapOp(PSBitmap a, PSBitmap b, PSBitmap dest, int op);

/* Filesystem functions */
int PSIsDirectory(const char *path);
const char *PSGetHomeDirectory(void);
int PSMakeDir(const char *path, int recursive);
const char *PSWorkingDirectory(void);
char *PSPathJoin(int count, ...);

/* Strings */

char *PSStringJoin(char **strings, char *sep, int len);
int PSPrintableLength(const char *s);
unsigned int PSCalcIntStringLength(long long num);

/* Networking functions. */
int PSDownloadFile(const char *url, const char *dest_dir);

/* Neural Network Functions */

void PSAbortLayer(PSModel *model, PSLayer *layer);

/* Misc */

int PSGetTerminalColumns();
void PSFillWithBlank(int line_length);
char *PSGetElapsedTimeString(time_t elapsed_us, int long_format);

#endif /* __PS_UTILS_H */

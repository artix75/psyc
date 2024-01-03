/*
 * Copyright (C) 2016-2023 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <libgen.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <pwd.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>

#include "psyc.h"
#include "platform.h"
#include "utils.h"
#include "maths.h"
#include "utf8.h"
#include "log.h"

#if IS_UNIX
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/**** PSDict ****/

/*
 * The Dan Bernstein popuralized hash..  See
 * https://github.com/pjps/ndjbdns/blob/master/cdb_hash.c#L26 Due to hash
 * collisions it seems to be replaced with "siphash" in n-djbdns, see
 * https://github.com/pjps/ndjbdns/commit/16cb625eccbd68045737729792f09b4945a4b508
 */
uint32_t djb33_hash(const char* s, size_t len) {
    uint32_t h = 5381;
    while (len--) {
        /* h = 33 * h ^ s[i]; */
        h += (h << 5);
        h ^= *s++;
        if (!*s) break;
    }
    return h;
}

PSDictItem *PSDictItemCreate(const char *key, PSDictValue value) {
    if (key == NULL) return NULL;
    PSDictItem *item = malloc(sizeof(*item));
    if (item == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    item->key = strdup(key);
    if (item->key == NULL) {
        PSPrintMemoryErrorMsg();
        free(item);
        return NULL;
    }
    item->value = value;
    item->prev = NULL;
    item->next = NULL;
    item->dict = NULL;
    item->slot = -1;
    return item;
}

void PSDictItemRelease(PSDictItem *item) {
    if (item == NULL) return;
    if (item->dict != NULL && item->dict->onItemRelease != NULL)
        item->dict->onItemRelease(item);
    free((void *) item->key);
    free(item);
}

/* Create a new PSDict dictionary. */
PSDict *PSDictCreate(int flags) {
    PSDict *dict = calloc(1, sizeof(*dict));
    if (dict == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    dict->flags = flags;
    return dict;
}

/* Delete all items from dictionary `dict`. */
void PSDictClear(PSDict *dict) {
    if (dict == NULL) return;
    for (int i = 0; i < PSDICT_HT_SIZE; i++) {
        PSDictItem *item = dict->table[i];
        while (item) {
            PSDictItem *next = item->next;
            PSDictItemRelease(item);
            item = dict->table[i] = next;
            dict->length--;
        }
        dict->table[i] = NULL;
    }
}

/* Get the item associated to `key` from dictionary `dict`, if any.
 * Return value: the item (PSDictItem) or NULL. */
PSDictItem *PSDictGet(PSDict *dict, const char *key) {
    if (dict == NULL || key == NULL) return NULL;
    int slot = PSDictSlotForKey(key);
    PSDictItem *item = dict->table[slot], *found_item = NULL;
    while (item != NULL) {
        if (strcmp(item->key, key) == 0) {
            found_item = item;
            break;
        }
        item = item->next;
    }
    return found_item;
}

/* Get the item associated to `key` in dictionary `dict` as a pointer.
 * Return value: the item as a pointer or NULL. */
void *PSDictGetPointer(PSDict *dict, const char *key) {
    PSDictItem *item = PSDictGet(dict, key);
    if (item == NULL) return NULL;
    return item->value.as_ptr;
}

/* Check whether `dict` has the key `key`.
 * Return value: 1 if `dict` has `key`, elseway 0. */
int PSDictHasKey(PSDict *dict, const char *key) {
    PSDictItem *item = PSDictGet(dict, key);
    return (item != NULL);
}

/* Set value `val` for key `key` in dictionary `dict`. Unless flag
 * `PSDICT_UPDATE_DISABLED` is enabled in dictionary flags, value will be
 * set even If `key` is already associated to another value.
 * Return value: the item (`PSDictItem`) associated to the `key` or NULL. */
PSDictItem *PSDictSet(PSDict *dict, const char *key, PSDictValue val) {
    if (dict == NULL || key == NULL) return NULL;
    int slot = PSDictSlotForKey(key), found = 0;
    PSDictItem *item = dict->table[slot], *last_item = NULL;
    while (item != NULL) {
        if (strcmp(item->key, key) == 0) {
            if (!(dict->flags & PSDICT_UPDATE_DISABLED))
                item->value = val;
            found = 1;
            break;
        }
        last_item = item;
        item = item->next;
    }
    if (found) return item;
    item = PSDictItemCreate(key, val);
    if (item == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    item->dict = dict;
    item->slot = slot;
    if (last_item != NULL) {
        last_item->next = item;
        item->prev = last_item;
    } else dict->table[slot] = item;
    dict->length++;
    return item;
}

/* Return value `val` if it's already set for `key`, elseway set it and
 * return it.
 * Return value: the value associated with `key`. */
PSDictItem *PSDictGetOrSet(PSDict *dict, const char *key, PSDictValue val) {
    if (dict == NULL) return NULL;
    int old_flags = dict->flags;
    dict->flags |= PSDICT_UPDATE_DISABLED;
    PSDictItem *item = PSDictSet(dict, key, val);
    dict->flags = old_flags;
    return item;
}

/* Remove item associated to `key` from dictionary `dict`, if any. */
void PSDictRemove(PSDict *dict, const char *key) {
    PSDictItem *item = PSDictGet(dict, key);
    if (item == NULL) return;
    PSDictItem *prev = item->prev, *next = item->next;
    if (prev == NULL) {
        assert(item->slot >= 0);
        dict->table[item->slot] = next;
    } else prev->next = next;
    if (next != NULL) next->prev = prev;
    PSDictItemRelease(item);
    dict->length--;
}

/* Return an array containing all keys owned by dictionary `dict`.
 * The size of the array is given by `dict->length`.
 * Return value: an array of strings containing all the keys or NULL if
                 something goes wrong. */
const char **PSDictGetKeys(PSDict *dict) {
    if (dict == NULL || dict->length == 0) return NULL;
    const char **keys = malloc(dict->length * sizeof(const char *));
    if (keys == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int idx = 0, i;
    for (i = 0; i < PSDICT_HT_SIZE; i++) {
        PSDictItem *item = dict->table[i];
        while (item != NULL) {
            keys[idx++] = item->key;
            item = item->next;
        }
    }
    return keys;
}

/* Return an array containing all items owned by dictionary `dict`.
 * The size of the array is given by `dict->length`.
 * Return value: an array of `PSDictItem` containing all the values or NULL if
                 something goes wrong. */
PSDictItem **PSDictGetItems(PSDict *dict) {
    if (dict == NULL || dict->length == 0) return NULL;
    PSDictItem **items = malloc(dict->length * sizeof(PSDictItem *));
    if (items == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    int idx = 0, i;
    for (i = 0; i < PSDICT_HT_SIZE; i++) {
        PSDictItem *item = dict->table[i];
        while (item != NULL) {
            items[idx++] = item;
            item = item->next;
        }
    }
    return items;
}

/* Create a new iterator for dictionary `dict`. The dictionary will be
 * allocated in memory, so it's up to the developer to free it as soon as
 * it is no longer needed.
 * Return value: the iterator or NULL if something goes wrong. */
struct PSDictIterator *PSDictIteratorCreate(PSDict *dict) {
    PSDictIterator *iter = malloc(sizeof(*iter));
    if (iter == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    iter->dict = dict;
    iter->current = NULL;
    return iter;
}

/* Iterate over the next item using `iterator`.
 * Return value: the next item (`PSDictItem`) or NULL if there are no more
 *               items to iterate. */
PSDictItem *PSDictNext(PSDictIterator *iterator) {
    if (iterator == NULL || iterator->dict == NULL) return NULL;
    PSDictItem *item = iterator->current;
    int slot = 0;
    while (item != NULL) {
        slot = item->slot;
        item = item->next;
        if (item != NULL) break;
    }
    if (item == NULL) {
        while (++slot < PSDICT_HT_SIZE) {
            item = iterator->dict->table[slot];
            if (item != NULL) break;
        }
    }
    if (item != NULL) iterator->current = item;
    return item;
}

/* Delete the dictionary and free it's allocated memory. */
void PSDictFree(PSDict *dict) {
    if (dict == NULL) return;
    PSDictClear(dict);
    free(dict);
}

/**** Neural Network Functions ****/

void PSAbortLayer(PSModel *model, PSLayer *layer) {
    if (model->size == 0) return;
    if (layer->index == (model->size - 1)) {
        model->size--;
        if (model->size == 0) {
            model->input_size = 0;
            model->output_size = 0;
        } else {
            PSLayer *outputLayer = model->layers[model->size - 1];
            if (outputLayer) model->output_size = outputLayer->size;
            else model->output_size = 0;
            PSLayer *inputLayer = model->layers[0];
            if (inputLayer) model->input_size = inputLayer->size;
            else model->input_size = 0;
        }
        PSLayerFree(layer);
    }
}

/**** Misc ****/

int PSGetTerminalColumns() {
    static int __term_columns = -1;
    if (__term_columns < 0) {
#if IS_UNIX
        struct winsize w = {0};
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        __term_columns = w.ws_col;
#else
        __term_columns = 0;
#endif
        if (__term_columns <= 0) __term_columns = 80;
    }
    return __term_columns;
}

void PSFillWithBlank(int line_length) {
    int term_w = PSGetTerminalColumns();
    int pad = term_w - line_length, i;
    if (pad <= 0) return;
    for (i = 0; i < pad; i++) printf(" ");
    fflush(stdout);
}

/* Compare version string `vers1` with `vers2`.
 * Returns:
 *  -1 if `vers1` < `vers2`
 *  1 if `vers1` > `vers2`
 *  0 if both versions are equal. */
int PSCompareVersion(const char* vers1, const char* vers2) {
    int major1 = 0, minor1 = 0, patch1 = 0;
    int major2 = 0, minor2 = 0, patch2 = 0;
    sscanf(vers1, "%d.%d.%d", &major1, &minor1, &patch1);
    sscanf(vers2, "%d.%d.%d", &major2, &minor2, &patch2);
    if (major1 < major2) return -1;
    if (major1 > major2) return 1;
    if (minor1 < minor2) return -1;
    if (minor1 > minor2) return 1;
    if (patch1 < patch2) return -1;
    if (patch1 > patch2) return 1;
    return 0;
}

char *PSGetElapsedTimeString(time_t elapsed_us, int opts) {
    char *elapsed_str = malloc(256);
    if (elapsed_str == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    static const char *time_units_short[] = {"us", "ms", "s", "m", "h"};
    static const char *time_units_long[] = {
        "usec.", "msec.", "sec.", "min.", "hour(s)"
    };
    static const char *time_units_full[] = {
        "microsecond(s)", "millisecond(s)", "second(s)", "minute(s)", "hour(s)"
    };
    static const size_t numunits = sizeof(time_units_short) / sizeof(char *);
    int long_format = (opts & PS_OPT_TIME_LONG),
        full_format = (opts & PS_OPT_TIME_FULL),
        human = (opts & PS_OPT_TIME_HUMAN), i = 0;
    const char **units = NULL;
    char *sep = " ";
    if (full_format) units = time_units_full;
    else if (long_format) units = time_units_long;
    else {
        units = time_units_short;
        sep = "";
    }
    double elapsed = (double) elapsed_us, div = 1000, mod = 0;
    while (elapsed >= div) {
        if (++i >= (int) numunits) break;
        mod = fmod(elapsed, div);
        elapsed /= div;
        if (i > 1) div = 60;
    }
    elapsed_str[0] = '\0';
    const char *unit = units[i];
    int round = ((mod > 0) ? 1 : 0);
    if (human) {
        if (mod > 0 && i > 0) {
            const char *lower_unit = units[i - 1];
            snprintf(
                elapsed_str, 255, "%ld%s%s and %ld%s%s",
                (long) elapsed, sep, unit, (long) mod, sep, lower_unit
            );
        } else snprintf(elapsed_str, 255, "%ld%s%s", (long)elapsed, sep, unit);
    } else snprintf(elapsed_str, 255, "%.*f%s%s", round, elapsed, sep, unit);
    return elapsed_str;
}

/* Create a vector of length `len` where value at `index` is one while all the
 * other values contain zero. */
PSFloat *PSOneHotVector(uint64_t index, uint64_t len) {
    if (index >= len) {
        PSErr(__func__, "index is out-of-bounds");
        return NULL;
    }
    PSFloat *vec = PSVectorZero(len);
    if (vec == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    vec[index] = 1;
    return vec;
}

/**** Filesystem functions ****/

/* Checks whether `path` is a valid directory. */
int PSIsDirectory(const char *path) {
    if (path == NULL) return 0;
    struct stat pathstat;
    if (stat(path, &pathstat) < 0) return 0;
    return S_ISDIR(pathstat.st_mode);
}

/* Returns user HOME directory. */
const char *PSGetHomeDirectory(void) {
    struct passwd *pw = getpwuid(getuid());
    if (pw == NULL) return NULL;
    return pw->pw_dir;
}

/* Creates directory `path` is it does not exists. If `recursive` is 1,
 * the function will try to also create intermediate paths if they don't
 * exists, in a similar fashion to "mkdir -p".
 * Return vale: 1 in case of success, elseway 0. */
int PSMakeDir(const char *path, int recursive) {
    if (path == NULL) {
        PSErr(__func__, "required argument `path` cannot be NULL");
        return 0;
    }
    if (PSFileExists(path)) {
        if (!PSIsDirectory(path)) {
            PSErr(__func__, "'%s' already exists and is not a directory",path);
            return 0;
        }
        return 1;
    }
    errno = 0;
    int success = mkdir(path, 0777) == 0;
    if (!success) {
        if (recursive && errno == ENOENT) {
            char parents[PATH_MAX];
            const char *path_p = path;
            while ((path_p = strchr(path_p, '/'))) {
                if (path_p == path) goto next;
                size_t len = path_p - path;
                memcpy(parents, path, len);
                parents[len] = '\0';
                if (PSFileExists(parents)) {
                    if (!PSIsDirectory(parents)) {
                        PSErr(__func__, "'%s' already exists and is not a "
                              "directory", parents);
                        return 0;
                    }
                    goto next;
                }
                if (mkdir(parents, 0777) < 0) {
                    path = parents;
                    goto err;
                }
next:
                path_p++;
            }
            success = (mkdir(path, 0777) == 0);
            if (!success) goto err;
        } else goto err;
    }
    return success;
err:
    PSErr(__func__, "failed to create directory at path '%s': %s",
          path, strerror(errno));
    return 0;
}

/* Get PsyC working directory, that is, by default `$HOME/.psyc`.
 * A custom working directory can be specified at compile-time using
 * `PS_WORKING_DIR` macro or by setting `PS_WORKING_DIR` environment
 * variable.
 * The function will try to automatically create the working directory if
 * it doesn't exist.
 * Return value: path to the working directory or NULL in case something
 * goes wrong.
 * NOTE: it returns a static string, so it cannot be freed. */
const char *PSWorkingDirectory(void) {
#ifdef PS_WORKING_DIR
    const char *default_working_dir = PS_WORKING_DIR;
#else
    const char *default_working_dir = NULL;
#endif
    static char static_dir[PATH_MAX] = {0};
    static const char *dir = NULL;
    static int no_home_dir = 0;
    if (dir == NULL) {
        dir = getenv("PS_WORKING_DIR");
        if (dir == NULL) dir = default_working_dir;
        if (dir == NULL) {
            const char *home = PSGetHomeDirectory();
            if (home == NULL) goto no_home;
            char *path = PSPathJoin(2, home, ".psyc");
            if (path == NULL) return 0;
            if (strlen(path) >= PATH_MAX) {
                free(path);
                return NULL;
            }
            strncpy(static_dir, path, PATH_MAX);
            free(path);
            dir = static_dir;
        }
        if (!PSMakeDir(dir, 1)) return NULL;
    }
    return dir;
no_home:
    no_home_dir = 1;
    PSErr(__func__, "could not determine home directory, please explicitely "
          "set PS_WORKING_DIR env variable");
    return NULL;
}

/* Joins multiple file path components into a single path string.
 * Argument `count` is used to specify how many components will be consumed.
 * Path components must be passed as variadic arguments.
 * Return value: string containing the joined path or NULL if something goes
 *               wrong. Returned string is allocated into heap, so it's up
 *               to the developer to free it as soon as it is no longer needed.
 */
char *PSPathJoin(int count, ...) {
    if (count <= 0) return NULL;
    char *path = malloc(PATH_MAX);
    if (path == NULL) {
        PSErr(__func__, "could not allocate path");
        return NULL;
    }
    int success = 1, pathlen = 0;
    char *p = path;
    va_list args;
    va_start(args, count);
    int totcount = count;
    while(count-- > 0) {
        char *path_comp = va_arg(args, char*);
        if (path_comp == NULL) break;
        int is_first = (count == (totcount - 1));
        if (!is_first && path_comp[0] == '/') {
            if (path_comp[1] == 0) continue;
            path_comp++;
        }
        int len = snprintf(p, (PATH_MAX - pathlen), "%s", path_comp);
        if (len == 0) continue;
        pathlen += len;
        p += len;
        success = (pathlen < PATH_MAX);
        if (!success) goto exceeded;
        if (count > 0 && path[pathlen - 1] != '/') {
            len = snprintf(p, (PATH_MAX - pathlen), "%c", '/');
            pathlen += len;
            p += len;
            success = (pathlen < PATH_MAX);
            if (!success) goto exceeded;
        }
        continue;
exceeded:
        PSErr(__func__, "path length exceeded %d", PATH_MAX);
        goto final;
    }
final:
    va_end(args);
    if (!success) {
        free(path);
        path = NULL;
    }
    return path;
}

/* Try to download content from `url`. The file will be genrated into
 * `dest_dir`.
 * The functions tries to download the file by using **wget** or **curl**
 * command line utilities.
 * If those utilities are not found, download will fail.
 * Return value: 1 in case of success, elseway 0. */
int PSDownloadFile(const char *url, const char *dest_dir) {
    int exit_status = 0;
    static char *download_utility = NULL;
    static char *utilities[] = {"wget", "curl"};
    int max_cmd_len = PATH_MAX * 2;
    char cwd[PATH_MAX] = {0};
    char cmd[max_cmd_len];
    cmd[0] = '\0';
    if (download_utility == NULL) {
        size_t utilities_count = sizeof(utilities) / sizeof(char *), i;
        for (i = 0; i < utilities_count; i++) {
            char *utility = utilities[i];
            sprintf(cmd, "%s --version > /dev/null 2>&1", utility);
            exit_status = system(cmd);
            if (exit_status == 0) {
                download_utility = utility;
                break;
            }
        }
        if (download_utility == NULL) {
            PSErr(
                __func__, "could not find neither wget nor curl on "
                "your system"
            );
            return 0;
        }
    }
    if (dest_dir == NULL) {
        if (getcwd(cwd, sizeof(cwd)) == NULL) {
            PSErr(NULL, "could not determine current working dir");
            return 0;
        }
        dest_dir = cwd;
    }
    int cmdlen = 0;
    if (strcmp("curl", download_utility) == 0) {
        cmdlen = snprintf(cmd, max_cmd_len, "curl -O '%s' --output-dir '%s'",
                           url, dest_dir);
    } else if (strcmp("wget", download_utility) == 0) {
        cmdlen = snprintf(cmd, max_cmd_len, "wget -P '%s' '%s'", dest_dir, url);
    }
    if (cmdlen >= max_cmd_len) {
        PSErr(__func__, "command exceeds max length");
        return 0;
    }
    if (PSLogLevel >= PSLOGLEVEL_DEBUG) {
        PSPrint(
            PSLOGLEVEL_DEBUG,
            "Downloading file from: %s\n"
            "                   to: %s\n",
            url, dest_dir
        );
        PSPrint(PSLOGLEVEL_DEBUG, cmd);
    }
    exit_status = system(cmd);
    return exit_status == 0;
}

/**** Strings ****/

char *PSStringJoin(char **strings, char *sep, int len) {
    errno = 0;
    if (len < 2) return NULL;
    char *joined = NULL;
    if (sep == NULL) sep = "";
    int seplen = strlen(sep), totlen = 1, i;
    for (i = 0; i < len; i++) {
        char *str = strings[i];
        if (str == NULL) str = "";
        int slen = strlen(str), prepend_sep = (i > 0);
        if (prepend_sep) slen += seplen;
        totlen += slen;
        char *new = realloc(joined, totlen);
        if (new == NULL) {
            PSPrintMemoryErrorMsg();
            errno = ENOMEM;
            return NULL;
        }
        joined = new;
        char *p = joined + (totlen - slen - 1);
        if (prepend_sep) snprintf(p, slen + 1, "%s%s", sep, str);
        else snprintf(p, slen + 1, "%s", str);
    }
    return joined;
}

int PSPrintableLength(const char *s) {
    if (s == NULL) return 0;
    int len = 0;
    int in_style_seq = 0;
    while (*s) {
        char c = *s;
        if (c == '\x1b') {
            in_style_seq = 1;
            goto next;
        } else if (in_style_seq) {
            if (c == 'm') in_style_seq = 0;
            goto next;
        }
        int clen = PSUTF8CharSize(s);
        if (clen == 1) {
            if (isprint(c)) len++;
        } else if ((c & 0xC0) != 0x80) len++;
next:
        s++;
    }
    return len;
}

unsigned int PSCalcIntStringLength(long long num) {
    if (num == 0) return 1;
    int len = 1;
    if (num < 0) {
        num *= -1;
        len++;
    }
    len += log10((double) num);
    return len;
}

/* Bitmaps */

size_t PSGetBitmapArrayLength(size_t size) {
    return 1 + (size / (64 * 8));
}

int PSBitmapIndexFor(uint64_t index) {
    return (int) (index / (64 * 8));
}

PSBitmap PSBitmapCreate(size_t size) {
    if (size == 0) return NULL;
    size_t actual_size = PSGetBitmapArrayLength(size);
    actual_size += 1; /* Make room for element containing 'size': bitmap[-1] */
    PSBitmap bitmap = calloc(actual_size, 64);
    if (bitmap == NULL) {
        PSPrintMemoryErrorMsg();
        return NULL;
    }
    *bitmap = (uint64_t) size;
    return bitmap + 1;
}

void PSBitmapRelease(PSBitmap bitmap) {
    if (bitmap == NULL) return;
    bitmap--;
    free(bitmap);
}

size_t PSBitmapSize(PSBitmap bitmap) {
    if (bitmap == NULL) return 0;
    return (size_t) *(bitmap - 1);
}

int PSBitmapGetBit(PSBitmap bitmap, uint64_t index) {
    if (bitmap == NULL) return 0;
    if (index >= PSBitmapSize(bitmap)) return 0;
    int idx = PSBitmapIndexFor(index),
        bitidx = (int) (index % (64 * 8));
    return (bitmap[idx] & (1 << bitidx) ? 1 : 0);
}

int PSBitmapSetBit(PSBitmap bitmap, uint64_t index, int val) {
    if (bitmap == NULL) return 0;
    if (index >= PSBitmapSize(bitmap)) return 0;
    int idx = PSBitmapIndexFor(index),
        bitidx = (int) (index % (64 * 8));
    int old = (bitmap[idx] & (1 << bitidx) ? 1 : 0);
    if (val) bitmap[idx] |= (1 << bitidx);
    else bitmap[idx] &= ~((unsigned) (1 << bitidx));
    return old;
}

void PSBitmapClear(PSBitmap bitmap) {
    if (bitmap == NULL) return;
    size_t size = PSBitmapSize(bitmap);
    size_t actual_size = PSGetBitmapArrayLength(size);
    memset(bitmap, 0, actual_size * 64);
}

PSBitmap PSBitmapOp(PSBitmap a, PSBitmap b, PSBitmap dest, int op) {
    if (op < PS_BITMAP_OP_AND || op > PS_BITMAP_OP_XOR) {
        PSErr(__func__, "invalid operation");
        return NULL;
    }
    if (a == NULL) {
        PSErr(__func__, "mandatory argument `a` is null");
        return NULL;
    }
    if (b == NULL) {
        PSErr(__func__, "mandatory argument `b` is null");
        return NULL;
    }
    size_t size_a = PSBitmapSize(a), size_b = PSBitmapSize(b);
    if (size_a != size_b) {
        PSErr(__func__, "bitmaps have different size: %zu != %zu",
              size_a, size_b);
        return NULL;
    }
    if (dest == NULL) {
        dest = PSBitmapCreate(size_a);
        if (dest == NULL) return NULL;
    } else {
        size_t dstsize = PSBitmapSize(dest);
        if (dstsize != size_a) {
            PSErr(__func__, "dest size differs: %zu != %zu", dstsize, size_a);
            return NULL;
        }
    }
    PSBitmapClear(dest);
    size_t len = PSGetBitmapArrayLength(size_a), i;
    for (i = 0; i < len; i++) {
       if (op == PS_BITMAP_OP_AND) dest[i] = a[i] & b[i];
       else if (op == PS_BITMAP_OP_OR) dest[i] = a[i] | b[i];
       else if (op == PS_BITMAP_OP_XOR) dest[i] = a[i] ^ b[i];
    }
    return dest;
}

int PSBitmapCopy(PSBitmap dst, PSBitmap src) {
    if (dst == NULL || src == NULL) return 0;
    size_t src_size = PSBitmapSize(src), dst_size = PSBitmapSize(dst), i, count;
    if (src_size != dst_size) {
        PSErr(__func__, "source size differs from destination size: %zu != %zu",
              src_size, dst_size);
        return 0;
    }
    if (src_size == 0) return 0;
    count = PSGetBitmapArrayLength(src_size);
    for (i = 0; i < count; i++) dst[i] = src[i];
    return 1;
}

PSBitmap PSBitmapDup(PSBitmap src) {
    if (src == NULL) return NULL;
    size_t size = PSBitmapSize(src);
    PSBitmap clone = PSBitmapCreate(size);
    if (clone == NULL) return NULL;
    int success = PSBitmapCopy(clone, src);
    if (!success) {
        PSErr(__func__, "could not copy source data");
        PSBitmapRelease(clone);
        return NULL;
    }
    return clone;
}

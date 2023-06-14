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

#include "psyc.h"
#include "platform.h"
#include "utils.h"
#include "maths.h"
#include "log.h"

#if IS_UNIX
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork *network, PSLayer *layer) {
    if (network->size == 0) return;
    if (layer->index == (network->size - 1)) {
        network->size--;
        if (network->size == 0) {
            network->input_size = 0;
            network->output_size = 0;
        } else {
            PSLayer *outputLayer = network->layers[network->size - 1];
            if (outputLayer) network->output_size = outputLayer->size;
            else network->output_size = 0;
            PSLayer *inputLayer = network->layers[0];
            if (inputLayer) network->input_size = inputLayer->size;
            else network->input_size = 0;
        }
        PSDeleteLayer(layer);
    }
}

/* Misc */

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

PSFloat *PSCopyFloats(PSFloat *src, size_t length) {
    size_t size = length * sizeof(PSFloat);
    PSFloat *dup = malloc(size);
    if (dup == NULL) return NULL;
    memcpy(dup, src, size);
    return dup;
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
    static char elapsed_str[256];
    static const char *time_units_short[] = {"us", "ms", "s", "m", "h"};
    static const char *time_units_long[] = {
        "usec.", "msec.", "sec.", "min.", "hour(s)"
    };
    static const char *time_units_full[] = {
        "microsecond(s)", "millisecond(s)", "second(s)", "minute(s)", "hour(s)"
    };
    static const size_t numunits = sizeof(time_units_short) / sizeof(char *);
    int long_format = (opts & OPT_TIME_LONG),
        full_format = (opts & OPT_TIME_FULL),
        human = (opts & OPT_TIME_HUMAN), i = 0;
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

/* Filesystem functions */

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

/* Returns PsyC working directory, that is, by default '$HOME/.psyc'.
 * A custom working directory can be specified at compile-time using
 * **PS_WORKING_DIR** macro or by setting **PS_WORKING_DIR** environment
 * variable.
 * The function will try to automatically create the working directory if
 * it doesn't exist.
 * Return value: path to the working directory or NULL in case something
 *               goes wrong.
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
    va_end(args);
final:
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
            PSErr(__func__, "could not find neither wget nor curl on "
                            "your system");
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
        PSLog(PSLOGLEVEL_DEBUG, "Downloading file from: %s\n"
                                "                   to: %s\n",
                                url, dest_dir);
        PSLog(PSLOGLEVEL_DEBUG, cmd);
    }
    exit_status = system(cmd);
    return exit_status == 0;
}

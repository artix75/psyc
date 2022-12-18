/*
 * Copyright (C) 2016-2022 Fabio Nicotra <artix2 at gmail dot com>.
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
#include <string.h>
#include <strings.h>
#include "log.h"
#include "psyc.h"

const char *logLevels[] = {
    "DEBUG",
    "INFO",
    "NOTICE",
    "SUCCESS",
    "WARN",
    "ERROR",
    "FATAL"
};

static size_t log_levels_count = sizeof(logLevels) / sizeof(const char *);

int PSLogLevel = PSDEFAULT_LOGLEVEL;
FILE *PSLogFile = NULL;

void PSVLog(int level, const char *format, va_list args) {
    if (level < PSLogLevel) return;
    int use_colors = PSLogColorEnabled();
    FILE *out = PSLogFile;
    if (out == NULL) {
        if (level > PSLOGLEVEL_WARN) out = stderr;
        else out = stdout;
    } else use_colors = 0;
    char *color = NULL;
    if (use_colors) {
        switch (level) {
        case PSLOGLEVEL_DEBUG: color = PSCOLOR_DIM; break;
        case PSLOGLEVEL_INFO: color = PSCOLOR_WHITE; break;
        case PSLOGLEVEL_NOTICE: color = PSCOLOR_BOLD; break;
        case PSLOGLEVEL_SUCCESS: color = PSCOLOR_GREEN; break;
        case PSLOGLEVEL_WARN: color = PSCOLOR_YELLOW; break;
        case PSLOGLEVEL_ERROR: color = PSCOLOR_RED; break;
        case PSLOGLEVEL_FATAL: color = PSCOLOR_MAGENTA; break;
        }
    }
    if (color != NULL) fprintf(out, "%s", color);
    vfprintf(out, format, args);
    if (color != NULL) fprintf(out, "%s", PSCOLOR_RESET);
}

void PSLog(int level, const char *format, ...) {
    va_list args;
    va_start(args, format);
    PSVLog(level, format, args);
    va_end(args);
}

void PSDebug(const char *format, ...) {
    if (PSLogLevel > PSLOGLEVEL_DEBUG) return;
    PSLog(PSLOGLEVEL_DEBUG, "DEBUG: ");
    va_list args;
    va_start(args, format);
    PSVLog(PSLOGLEVEL_DEBUG, format, args);
    va_end(args);
    PSLog(PSLOGLEVEL_DEBUG, "\n");
}

void PSInfo(const char *format, ...) {
    if (PSLogLevel > PSLOGLEVEL_INFO) return;
    va_list args;
    va_start(args, format);
    PSVLog(PSLOGLEVEL_INFO, format, args);
    va_end(args);
    PSLog(PSLOGLEVEL_INFO, "\n");
}

void PSWarn(const char *format, ...) {
    if (PSLogLevel > PSLOGLEVEL_WARN) return;
    PSLog(PSLOGLEVEL_WARN, "WARN: ");
    va_list args;
    va_start(args, format);
    PSVLog(PSLOGLEVEL_WARN, format, args);
    va_end(args);
    PSLog(PSLOGLEVEL_WARN, "\n");
}

void PSErr(const char *tag, const char *format, ...) {
    if (PSLogLevel > PSLOGLEVEL_ERROR) return;
    PSLog(PSLOGLEVEL_ERROR, "ERROR");
    if (tag != NULL) PSLog(PSLOGLEVEL_ERROR, " [%s]: ", tag);
    else PSLog(PSLOGLEVEL_ERROR, ": ");
    va_list args;
    va_start(args, format);
    PSVLog(PSLOGLEVEL_ERROR, format, args);
    va_end(args);
    PSLog(PSLOGLEVEL_ERROR, "\n");
}

const char* PSLogLevelName(int level) {
    if (level >= (int)log_levels_count) level = log_levels_count - 1;
    return logLevels[level];
}

int PSLogLevelByName(const char *name) {
    int level = -1, i;
    for (i = 0; i < (int) log_levels_count; i++) {
        const char *lvlname = logLevels[i];
        if (strcasecmp(lvlname, name) == 0) {
            level = i;
            break;
        }
    }
    return level;
}

int PSGetMaxLogLevel(void) {
    return (int) log_levels_count - 1;
}

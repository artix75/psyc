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

#ifndef __LOG_H__
#define __LOG_H__
#include <stdlib.h>
#include <stdio.h>
#include "psyc.h"

#define PSLOGLEVEL_DEBUG     0
#define PSLOGLEVEL_INFO      1
#define PSLOGLEVEL_NOTICE    2
#define PSLOGLEVEL_SUCCESS   3
#define PSLOGLEVEL_WARN      4
#define PSLOGLEVEL_ERROR     5
#define PSLOGLEVEL_FATAL     6

#define PSCOLOR_RED         "\x1b[31m"
#define PSCOLOR_GREEN       "\x1b[32m"
#define PSCOLOR_YELLOW      "\x1b[33m"
#define PSCOLOR_BLUE        "\x1b[34m"
#define PSCOLOR_MAGENTA     "\x1b[35m"
#define PSCOLOR_CYAN        "\x1b[36m"
#define PSCOLOR_WHITE       "\x1b[97m"
#define PSCOLOR_BOLD        "\x1b[1m"
#define PSCOLOR_DIM         "\x1b[2m"
#define PSCOLOR_HIDDEN      "\x1b[8m"
#define PSCOLOR_RESET       "\x1b[0m"
#define PSCOLOR_RESET_BOLD  "\x1b[21m"

#define PSDEFAULT_LOGLEVEL   PSLOGLEVEL_INFO

#ifndef FLAG_LOG_COLORS
#define FLAG_LOG_COLORS (1 << 0)
#endif

#define PSLogColorEnabled() (PSGlobalFlags & FLAG_LOG_COLORS)
#define PSLogEnableColor() (PSGlobalFlags |= FLAG_LOG_COLORS)
#define PSLogDisableColor() (PSGlobalFlags &= ~((unsigned) FLAG_LOG_COLORS))
#define PSPrintMemoryErrorMsg() PSErr(NULL, "Could not allocate memory!")

extern int PSLogLevel;
extern FILE *PSLogFile;

void PSLog(int level, const char *format, ...);
void PSVLog(int level, const char *format, va_list args);
void PSDebug(const char *format, ...);
void PSInfo(const char *format, ...);
void PSNotice(const char *format, ...);
void PSWarn(const char *format, ...);
void PSErr(const char *tag, const char *format, ...);
void PSErrNN(const char *tag, PSNeuralNetwork *network, PSLayer *layer,
             const char *format, ...);
const char* PSLogLevelName(int level);
int PSLogLevelByName(const char *name);
int PSGetMaxLogLevel(void);

#endif /* __LOG_H__ */

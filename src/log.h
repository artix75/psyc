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

#define PSCOLOR_BLACK       "\x1b[30m"
#define PSCOLOR_RED         "\x1b[31m"
#define PSCOLOR_GREEN       "\x1b[32m"
#define PSCOLOR_YELLOW      "\x1b[33m"
#define PSCOLOR_BLUE        "\x1b[34m"
#define PSCOLOR_MAGENTA     "\x1b[35m"
#define PSCOLOR_CYAN        "\x1b[36m"
#define PSCOLOR_WHITE       "\x1b[37m"
#define PSCOLOR_RESET       "\x1b[0m"
#define PSCOLOR_RESET_BOLD  "\x1b[21m"
#define PSSTYLE_BOLD        "\x1b[1m"
#define PSSTYLE_DIM         "\x1b[2m"
#define PSSTYLE_HIDDEN      "\x1b[8m"
#define PSSTYLE_ITALICS     "\x1b[3m"
#define PSSTYLE_UNDERLINE   "\x1b[4m"

#define PSCOLOR_BOLD    PSSTYLE_BOLD
#define PSCOLOR_DIM     PSSTYLE_DIM
#define PSCOLOR_DARK    PSSTYLE_DIM

#define PSXTERM256_GRADIENT_BLACK_BLUE      16
#define PSXTERM256_GRADIENT_GREEN_BLUE      22
#define PSXTERM256_GRADIENT_GREEN_AZURE     28
#define PSXTERM256_GRADIENT_GREEN_CYAN1     34
#define PSXTERM256_GRADIENT_GREEN_CYAN2     40
#define PSXTERM256_GRADIENT_GREEN_CYAN3     46
#define PSXTERM256_GRADIENT_GREEN_CYAN4     112
#define PSXTERM256_GRADIENT_GREEN_CYAN5     118
#define PSXTERM256_GRADIENT_GREEN_CYAN6     148
#define PSXTERM256_GRADIENT_GREEN_CYAN7     154
#define PSXTERM256_GRADIENT_MAGENTA1        52
#define PSXTERM256_GRADIENT_MAGENTA2        88
#define PSXTERM256_GRADIENT_MAGENTA3        124
#define PSXTERM256_GRADIENT_MAGENTA4        160
#define PSXTERM256_GRADIENT_MAGENTA5        196
#define PSXTERM256_GRADIENT_ORANGE_VIOLET1  130
#define PSXTERM256_GRADIENT_ORANGE_VIOLET2  136
#define PSXTERM256_GRADIENT_ORANGE_VIOLET3  166
#define PSXTERM256_GRADIENT_ORANGE_VIOLET4  172
#define PSXTERM256_GRADIENT_ORANGE_VIOLET5  178
#define PSXTERM256_GRADIENT_ORANGE_VIOLET6  202
#define PSXTERM256_GRADIENT_ORANGE_VIOLET7  208
#define PSXTERM256_GRADIENT_ORANGE_VIOLET8  214
#define PSXTERM256_GRADIENT_YELLOW_WHITE1   190
#define PSXTERM256_GRADIENT_YELLOW_WHITE2   26
#define PSXTERM256_GRADIENT_GRAYSCALE       232

#define PSDEFAULT_LOGLEVEL   PSLOGLEVEL_INFO

#ifndef FLAG_LOG_COLORS
#define FLAG_LOG_COLORS (1 << 0)
#endif

#define PS_PROGRESS_STYLE_DOUBLE_DASH 0
#define PS_PROGRESS_STYLE_SINGLE_DASH 1
#define PS_PROGRESS_STYLE_BAR         2
#define PS_PROGRESS_STYLE_LINE        3

#define PS_PROGRESS_FLAG_PERCENT        (1 << 0)
#define PS_PROGRESS_FLAG_JUST_PERCENT   (1 << 1)
#define PS_PROGRESS_FLAG_JUST_BAR       (1 << 2)
#define PS_PROGRESS_FLAG_NO_TOTAL       (1 << 3)
#define PS_PROGRESS_FLAG_NO_GRADIENT    (1 << 4)
#define PS_PROGRESS_FLAG_NO_XTERM256    (1 << 5)
#define PS_PROGRESS_FLAG_XTERM256_CODE  (1 << 6)
#define PS_PROGRESS_FLAG_PERCENT_RIGHT  (1 << 7)

#define PS_LINE_FILL            (1 << 1)
#define PS_LINE_OVERWRITE       (1 << 2)
#define PS_LINE_PLAIN_ASCII     (1 << 3)

#define PSClearScreen() (printf("\e[1;1H\e[2J"))
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

int PSIsXTermColor256(int always_check);
int PSXTermColor256ToANSI(uint8_t color, int bgcolor);
void PSVPrintSameLine(char *format, va_list args);
void PSPrintSameLine(char *format, ...);
int PSProgressBar(int num, int tot, int style, int color, int flags,
                  int maxlen, char *label);
int PSLineStart(int opts, char *format, ...);
int PSLineAppend(int opts, char *format, ...);
int PSVLineAppend(int opts, char *format, va_list args);
int PSLineFill(void);
void PSLineEnd(void);

#endif /* __LOG_H__ */

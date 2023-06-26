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
#include <string.h>
#include <strings.h>
#include "log.h"
#include "psyc.h"
#include "utils.h"

#define CONTINUOUS_PROG_CHAR "―"

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
static int printing_on_same_line = 0;
static int current_line_length = 0;
static const char *line_overwritten_by = NULL;

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
    if (printing_on_same_line) {
        printing_on_same_line = 0;
        printf("\n");
        fflush(stdout);
        if (out != stdout) {
            fprintf(out, "\n");
            fflush(out);
        }
    }
    char *color = NULL;
    if (use_colors) {
        switch (level) {
        case PSLOGLEVEL_DEBUG: color = PSCOLOR_DIM; break;
        case PSLOGLEVEL_INFO: color = PSCOLOR_RESET; break;
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

void PSNotice(const char *format, ...) {
    if (PSLogLevel > PSLOGLEVEL_NOTICE) return;
    va_list args;
    va_start(args, format);
    PSVLog(PSLOGLEVEL_NOTICE, format, args);
    va_end(args);
    PSLog(PSLOGLEVEL_NOTICE, "\n");
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

void PSErrNN(const char *tag, PSNeuralNetwork *network, PSLayer *layer,
             const char *format, ...)
{
    if (PSLogLevel > PSLOGLEVEL_ERROR) return;
    PSLog(PSLOGLEVEL_ERROR, "ERROR");
    int null_network = network == NULL;
    if (null_network && layer != NULL) network = layer->network;
    if (tag != NULL) PSLog(PSLOGLEVEL_ERROR, " [%s]: ", tag);
    else PSLog(PSLOGLEVEL_ERROR, ": ");
    if (network != NULL || layer != NULL) {
        int printed_network = 1, printed_layer = 0;
        if (PSGetNetworkChainLength(network) > 1)
            PSLog(PSLOGLEVEL_ERROR, "Network[%d]", network->index);
        else if (network && !null_network && network->name != NULL) {
            char *ellipsis = "";
            if (strlen(network->name) > 15)
                ellipsis = "...";
            PSLog(PSLOGLEVEL_ERROR, "Network \"%.15s%s\"", network->name,
                  ellipsis);
        } else printed_network = 0;
        if (layer != NULL) {
            if (printed_network) PSLog(PSLOGLEVEL_ERROR, ", ");
            PSLog(PSLOGLEVEL_ERROR, "Layer[%d]", layer->index);
            printed_layer = 1;
        }
        if (printed_network || printed_layer)
            PSLog(PSLOGLEVEL_ERROR, ": ");
    }
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

int PSIsXTermColor256(int always_check) {
    static int is_xterm_256 = -1;
    if (always_check || is_xterm_256 < 0) {
        char *term = getenv("TERM");
        is_xterm_256 = (term != NULL && strcmp("xterm-256color",term) == 0);
    }
    return is_xterm_256;
}

int PSXTermColor256ToANSI(uint8_t color, int bgcolor) {
    if (color <= 7) color += 30;
    else if (color <= 15) color += 90;
    else if (color >= 16 && color <= 21) color = 33; /* Blue */
    else if (color >= 22 && color <= 51) {
        if (((color - 16) % 6) > 3) color = 36; /* Cyan */
        else color = 21; /* Green */
    } else if (color >= 52 && color <= 57) color = 35; /* Magenta */
    else if (color >= 58 && color <= 87) {
        if (((color - 16) % 6) > 3) color = 36; /* Cyan */
        else color = 21; /* Green */
    } else if (color >= 88 && color <= 93) color = 35; /* Magenta */
    else if (color >= 94 && color <= 99) {
        if (((color - 16) % 6) > 3) color = 35; /* Magenta */
        else color = 33; /* yellow */
    } else if (color >= 106 && color <= 123) {
        if (((color - 16) % 6) > 3) color = 36; /* Cyan */
        else color = 21; /* Green */
    } else if (color >= 124 && color <= 129) color = 35; /* Magenta */
    else if (color >= 130 && color <= 141) {
        if (((color - 16) % 6) > 3) color = 35; /* Magenta */
        else color = 33; /* yellow */
    } else if (color >= 142 && color <= 159) {
        if (((color - 16) % 6) > 3) color = 36; /* Cyan */
        else color = 21; /* Green */
    } else if (color >= 160 && color <= 165) color = 35; /* Magenta */
    else if (color >= 172 && color <= 183) {
        if (((color - 16) % 6) > 3) color = 35; /* Magenta */
        else color = 33; /* yellow */
    } else if (color >= 185 && color <= 189) {
        if (((color - 16) % 6) > 3) color = 37; /* White */
        else color = 33; /* yellow */
    } else if (color >= 190 && color <= 195) {
        if (((color - 16) % 6) > 3) color = 36; /* Cyan */
        else color = 21; /* Green */
    } else if (color >= 196 && color <= 201) color = 35; /* Magenta */
    else if (color >= 202 && color <= 225) {
        if (((color - 16) % 6) > 3) color = 35; /* Magenta */
        else color = 33; /* yellow */
    } else if (color >= 226 && color <= 231) {
        if (((color - 16) % 6) > 3) color = 37; /* White */
        else color = 33; /* yellow */
    } else {
        if ((color - 232) < 244) color = 2; /* Dark */
        else color = 37; /* White */
    }
    if (bgcolor && color >= 30) color += 10;
    return color;
}

void PSVPrintSameLine(char *format, va_list args) {
    printing_on_same_line = 1;
    line_overwritten_by = __func__;
    if (format == NULL) format = "";
    int max_w = PSGetTerminalColumns() + 1;
    char buf[max_w];
    vsnprintf(buf, max_w, format, args);
    fprintf(stdout, "\r%-*s", max_w - 1, buf);
    fflush(stdout);
}

void PSPrintSameLine(char *format, ...) {
    va_list args;
    va_start(args, format);
    PSVPrintSameLine(format, args);
    va_end(args);
}

int PSProgressBar(int num, int tot, int style, int color, int flags,
                  int maxlen, char *label)
{
    if (tot == 0) goto end_bar;
    char *c = "=";
    int color_on_bg = 0;
    if (style == PS_PROGRESS_STYLE_DOUBLE_DASH) c = "=";
    else if (style == PS_PROGRESS_STYLE_SINGLE_DASH) c = "-";
    else if (style == PS_PROGRESS_STYLE_BAR) {
        color_on_bg = 1;
        c = " ";
    } else if (style == PS_PROGRESS_STYLE_LINE) {
        c = CONTINUOUS_PROG_CHAR;
    }
    int tw = PSGetTerminalColumns();
    if (maxlen <= 0) maxlen = tw - 1;
    if (maxlen < 1) goto end_bar;
    int available = maxlen, nwritten = 0;
    int do_append = (
        printing_on_same_line && current_line_length > 0 &&
        line_overwritten_by != __func__
    );
    if (do_append) available -= current_line_length;
    if (available < 1) goto end_bar;
    int max_available = available;
    int maxwrite = (max_available > 255 ? 255 : max_available), minlen = 6;
    int clen = strlen(c);
    char buf[255] = {0};
    char *p = buf;
    float percent = ((float) num / (float) tot);
    if (label != NULL) {
        int max_label_len = 17;
        nwritten = snprintf(p, max_label_len, "%.*s ", max_label_len-2, label);
        p += nwritten;
        available -= nwritten;
        maxwrite -= nwritten;
    }
    int just_bar = (flags & PS_PROGRESS_FLAG_JUST_BAR),
        just_percent = 0, use_percent = 0, percent_align_right = 0,
        i_percent = 0;
    if (!just_bar) {
        just_percent = (flags & PS_PROGRESS_FLAG_JUST_PERCENT);
        use_percent = (just_percent || (flags & PS_PROGRESS_FLAG_PERCENT));
        if (!just_percent) {
            int pad = 1 + (int) PSMathLog10((PSFloat) tot);
            if (!(flags & PS_PROGRESS_FLAG_NO_TOTAL))
                nwritten = snprintf(p, maxwrite, "%*d/%d ", pad, num, tot);
            else
                nwritten = snprintf(p, maxwrite, "%*d ", pad, num);
            p += nwritten;
            available -= nwritten;
            maxwrite -= nwritten;
            if (available < minlen) goto end_bar;
        }
        if (use_percent) {
            i_percent = (int) roundf(percent * 100);
            percent_align_right = (flags & PS_PROGRESS_FLAG_PERCENT_RIGHT);
            if (!percent_align_right) {
                nwritten = snprintf(p, maxwrite, "- %3d%% ", i_percent);
                p += nwritten;
                available -= nwritten;
                maxwrite -= nwritten;
                if (available < minlen) goto end_bar;
            } else {
                available -= 5;
                maxwrite -= 5;
            }
        }
    }
    int max_width = available;
    int width = (int) roundf(percent * (float) max_width), i;
    int barsize = (width * clen);
    int maxsize = barsize + (max_width - width);
    int is_xterm256 = 0;
    if (!(flags & PS_PROGRESS_FLAG_NO_XTERM256))
        is_xterm256 = PSIsXTermColor256(0);
    if (color) {
        int use_gradient = !(flags & PS_PROGRESS_FLAG_NO_GRADIENT) &&
                           is_xterm256;
        if (color == 1 && (use_gradient || !is_xterm256)) {
            /* Use default color */
            color = (is_xterm256 ? 40 : 32);
        }
        if (use_gradient) {
            int gradient_size = (color >= 232 ? (255 - color) : 5);
            color += (int) roundf(percent * (float) gradient_size);
        } else if (!is_xterm256) {
            if (flags & PS_PROGRESS_FLAG_XTERM256_CODE)
                color = PSXTermColor256ToANSI(color, color_on_bg);
        }
    } else if (color_on_bg) {
        color = 7;
        is_xterm256 = 0;
    }
    if (color) {
        if (is_xterm256) {
            int target = (color_on_bg ? 48 : 38); /* 48 is background */
            nwritten = snprintf(p, maxsize, "\x1b[%d;5;%dm",target,color);
        } else nwritten = snprintf(p, maxsize, "\x1b[%dm", color);
        p += nwritten;
    }
    char bar[255] = {0};
    if (((p + maxsize) - buf) > 255) goto end_bar;
    bar[0] = '\0';
    for (i = 0; i < width; i++) {
        if (clen == 1) bar[i] = c[0];
        else memcpy(bar + (i * clen), c, clen);
    }
    if (color_on_bg) {
        nwritten = snprintf(bar + (i * clen), maxsize, "\x1b[0m");
        maxsize += nwritten;
    }
    if (clen > 1 && (maxsize - barsize <= 0)) maxsize++;
    snprintf(p, maxsize, "%-*s", maxsize, bar);
    printing_on_same_line = 1;
    line_overwritten_by = __func__;
    if (!do_append) printf("\r%-*s", max_available, buf);
    else printf("%-*s", max_available, buf);
    if (color) printf("\x1b[0m");
    current_line_length += max_available;
    if (percent_align_right) {
        current_line_length += printf(" %3d%%", i_percent);
    }
    fflush(stdout);
    return 1;
end_bar:
    printing_on_same_line = 0;
    printf("\n");
    fflush(stdout);
    return 0;
}

int PSVLineAppend(int opts, char *format, va_list args) {
    if (format == NULL) return current_line_length;
    if (!printing_on_same_line) return current_line_length;
    int len = 0, tw = PSGetTerminalColumns();
    int available = tw - current_line_length, minlen = available;
    int is_plain_ascii = (opts & PS_LINE_PLAIN_ASCII),
        fill = (opts & PS_LINE_FILL);
    char buf[255] = {0};
    char *str = NULL;
    if (!is_plain_ascii) {
        int ascii_len = vsnprintf(buf, 255, format, args);
        int printed_len = len = PSPrintableLength(buf);
        minlen += (ascii_len - printed_len);
        str = buf;
    }
    if (!fill) {
        if (str != NULL) {
            printf("%s", str);
            current_line_length += len;
        } else current_line_length += vfprintf(stdout, format, args);
    } else {
        if (!is_plain_ascii) {
             printf("%-*s", minlen, buf);
             current_line_length += len + (available - len);
        } else {
            vsnprintf(buf, 255, format, args);
            current_line_length += printf("%-*s", minlen, buf);
        }
    }
    fflush(stdout);
    return current_line_length;
}

int PSLineAppend(int opts, char *format, ...) {
    int len;
    va_list args;
    va_start(args, format);
    len = PSVLineAppend(opts, format, args);
    va_end(args);
    return len;
}

int PSLineStart(int opts, char *format, ...) {
    int overwrite = (opts & PS_LINE_OVERWRITE);
    if (printing_on_same_line && !overwrite) fflush(stdout);
    current_line_length = 0;
    if (overwrite) {
        printing_on_same_line = 1;
        line_overwritten_by = __func__;
        printf("\r");
    } else printf("\n");
    if (format != NULL) {
        va_list args;
        va_start(args, format);
        int len = PSVLineAppend(opts, format, args);
        va_end(args);
        return len;
    } else return current_line_length;
}

int PSLineFill(void) {
    if (!printing_on_same_line) PSLineStart(PS_LINE_OVERWRITE, NULL);
    return PSLineAppend(PS_LINE_FILL, "");
}

void PSLineEnd(void) {
    current_line_length = 0;
    if (printing_on_same_line) {
        printf("\n");
        fflush(stdout);
    }
    printing_on_same_line = 0;
}

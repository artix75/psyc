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

#include <stdint.h>

#ifndef __PS_UTF8_H__
#define __PS_UTF8_H__

typedef uint32_t PSUTF8Char;
#define PSUTF8CharSize(s) utf8_length[(((uint8_t *)(s))[0] & 0xFF) >> 4]

static uint8_t const utf8_length[] = {
    /* 0 1 2 3 4 5 6 7 8 9 A B C D E F */
    1,1,1,1,1,1,1,1,0,0,0,0,2,2,3,4
};

int PSUTF8StrLen(const char *s);
int PSUTF8CodepointSize(uint32_t cp);
char *PSUTF8StrNCpy(char *dest, const char *src, size_t n);
int PSUTF8IsValidChar(PSUTF8Char c);
int PSUTF8Next(char *txt, PSUTF8Char *ch);
uint32_t PSUTF8Decode(PSUTF8Char c);
PSUTF8Char PSUTF8Encode(uint32_t codepoint);
int PSUTF8IsSpace(PSUTF8Char uc);
int PSUTF8IsPunct(PSUTF8Char uc);
int PSUTF8IsDigit(PSUTF8Char uc);
int PSUTF8IsAlpha(PSUTF8Char uc);
int PSUTF8IsUpper(PSUTF8Char uc);
int PSUTF8IsLower(PSUTF8Char uc);
int PSUTF8IsAlphaNum(PSUTF8Char uc);
PSUTF8Char PSUTF8ToUpper(PSUTF8Char uc);
PSUTF8Char PSUTF8ToLower(PSUTF8Char uc);

#endif /* __PS_UTF8_H__  */

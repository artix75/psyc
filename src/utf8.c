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
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include "utf8.h"

/*
 *  Inspired by:
 *  - https://dev.to/rdentato/utf-8-strings-in-c-1-3-42a4
 *  - https://rosettacode.org/wiki/UTF-8_encode_and_decode#C
 */

#include "misc/utf8-tables.c"

int PSUTF8StrLen(const char *s) {
    if (s == NULL) return 0;
    int len = 0;
    while (*s) {
        if ((*s & 0xC0) != 0x80) len++;
        s++;
    }
    return len;
}

int PSUTF8CodepointSize(uint32_t cp) {
    int len = 0, i;
    static uint32_t beginnings[] = {0, 0000, 0200, 04000, 0200000};
    static uint32_t endings[] = {0, 0177, 03777, 0177777, 04177777};
    for(i = 0; i < 4; i++) {
        uint32_t beg = beginnings[i], end = endings[i];
        if(cp >= beg && cp <= end) break;
        ++len;
    }
    if(len > 4) return -1; /* Out of bounds */
    return len;
}

char *PSUTF8StrNCpy(char *dest, const char *src, size_t n) {
    int k = n - 1, i;
    if (n) {
        dest[k] = 0;
        strncpy(dest, src, n);
        if (dest[k] & 0x80) {
            /* Last byte has been overwritten*/
            for (i = k;(i>0) && ((k-i) < 3) && ((dest[i] & 0xC0) == 0x80);i--) ;
            switch (k - i) {
                case 0:                                 dest[i] = '\0'; break;
                case 1:  if ( (dest[i] & 0xE0) != 0xC0) dest[i] = '\0'; break;
                case 2:  if ( (dest[i] & 0xF0) != 0xE0) dest[i] = '\0'; break;
                case 3:  if ( (dest[i] & 0xF8) != 0xF0) dest[i] = '\0'; break;
            }
        }
    }
    return dest;
}

int PSUTF8IsValidChar(PSUTF8Char c) {
    if (c <= 0x7F) return 1;
    if (0xC280 <= c && c <= 0xDFBF) return ((c & 0xE0C0) == 0xC080);

    if (0xEDA080 <= c && c <= 0xEDBFBF) return 0; /* Reject UTF-16 surrogates */

    if (0xE0A080 <= c && c <= 0xEFBFBF)         /* [4] */
        return ((c & 0xF0C0C0) == 0xE08080);

    if (0xF0908080 <= c && c <= 0xF48FBFBF)     /* [5] */
        return ((c & 0xF8C0C0C0) == 0xF0808080);

    return 0;
}

int PSUTF8Next(char *txt, PSUTF8Char *ch) {
    int len;
    PSUTF8Char encoding = 0;
    len = PSUTF8CharSize(txt);

    for (int i = 0; i < len && txt[i] != '\0'; i++)
        encoding = (encoding << 8) | txt[i];

    errno = 0;
    if (len == 0 || !PSUTF8IsValidChar(encoding)) {
        encoding = txt[0];
        len = 1;
        errno = EINVAL;
    }

    if (ch) *ch = encoding;
    return encoding ? len : 0 ;
}

/* from UTF-8 encoding to Unicode Codepoint */
uint32_t PSUTF8Decode(PSUTF8Char c) {
    static int bits[] = {6, 7, 5, 4, 3};
    static int masks[] = {0x3F, 0x7F, 0x1F, 0xF, 0x7};
    int bytes = PSUTF8CharSize(&c);
    int shift = bits[0] * (bytes - 1);
    uint32_t mask = masks[bytes];
    uint8_t *p = (uint8_t *) &c;
    uint32_t codep = (*p++ & mask) << shift;
    for (int i = 1; i < bytes; ++i, ++p) {
        shift -= bits[0];
        codep |= ((*p & 0x3F) << shift);
    }
    return codep;
}

/* From Unicode Codepoint to UTF-8 encoding */
PSUTF8Char PSUTF8Encode(uint32_t codepoint) {
    static int bits[] = {6, 7, 5, 4, 3};
    static int masks[] = {0x3F, 0x7F, 0x1F, 0xF, 0x7};
    static int leadings[] = {0x80, 0, 0xC0, 0xE0, 0xF0};
    int bytes = PSUTF8CodepointSize(codepoint);
    if (bytes < 0) return 0;

    PSUTF8Char encoded = 0;
    char *enc = (char *) &encoded;
    int shift = bits[0] * (bytes - 1), i;
    enc[0] = (codepoint >> shift & masks[bytes]) | leadings[bytes];
    shift -= bits[0];
    for(i = 1; i < bytes; ++i) {
        enc[i] = (codepoint >> shift & masks[0]) | leadings[0];
        shift -= bits[0];
    }
    for (i = bytes; i < 4; ++i) enc[i] = '\0';
    return encoded;
}

int PSUTF8IsSpace(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_space_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_space_codes[i] == cp) return 1;
    return (cp >= 0x9 && cp <= 0xD);
}

int PSUTF8IsPunct(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_punct_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_punct_codes[i] == cp) return 1;
    return (cp >= 0x21 && cp <= 0x23) ||
           (cp >= 0x25 && cp <= 0x2A) || (cp >= 0x2C && cp <= 0x2F) ||
           (cp >= 0x3A && cp <= 0x3B) || (cp >= 0x3F && cp <= 0x40) ||
           (cp >= 0x5B && cp <= 0x5D) || (cp >= 0xB6 && cp <= 0xB7) ||
           (cp >= 0x55A && cp <= 0x55F) || (cp >= 0x589 && cp <= 0x58A) ||
           (cp >= 0x5F3 && cp <= 0x5F4) || (cp >= 0x609 && cp <= 0x60A) ||
           (cp >= 0x60C && cp <= 0x60D) || (cp >= 0x61D && cp <= 0x61F) ||
           (cp >= 0x66A && cp <= 0x66D) || (cp >= 0x700 && cp <= 0x70D) ||
           (cp >= 0x7F7 && cp <= 0x7F9) || (cp >= 0x830 && cp <= 0x83E) ||
           (cp >= 0x964 && cp <= 0x965) || (cp >= 0xE5A && cp <= 0xE5B) ||
           (cp >= 0xF04 && cp <= 0xF12) || (cp >= 0xF3A && cp <= 0xF3D) ||
           (cp >= 0xFD0 && cp <= 0xFD4) || (cp >= 0xFD9 && cp <= 0xFDA) ||
           (cp >= 0x104A && cp <= 0x104F) || (cp >= 0x1360 && cp <= 0x1368) ||
           (cp >= 0x169B && cp <= 0x169C) || (cp >= 0x16EB && cp <= 0x16ED) ||
           (cp >= 0x1735 && cp <= 0x1736) || (cp >= 0x17D4 && cp <= 0x17D6) ||
           (cp >= 0x17D8 && cp <= 0x17DA) || (cp >= 0x1800 && cp <= 0x180A) ||
           (cp >= 0x1944 && cp <= 0x1945) || (cp >= 0x1A1E && cp <= 0x1A1F) ||
           (cp >= 0x1AA0 && cp <= 0x1AA6) || (cp >= 0x1AA8 && cp <= 0x1AAD) ||
           (cp >= 0x1B5A && cp <= 0x1B60) || (cp >= 0x1B7D && cp <= 0x1B7E) ||
           (cp >= 0x1BFC && cp <= 0x1BFF) || (cp >= 0x1C3B && cp <= 0x1C3F) ||
           (cp >= 0x1C7E && cp <= 0x1C7F) || (cp >= 0x1CC0 && cp <= 0x1CC7) ||
           (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x2030 && cp <= 0x2043) ||
           (cp >= 0x2045 && cp <= 0x2051) || (cp >= 0x2053 && cp <= 0x205E) ||
           (cp >= 0x207D && cp <= 0x207E) || (cp >= 0x208D && cp <= 0x208E) ||
           (cp >= 0x2308 && cp <= 0x230B) || (cp >= 0x2329 && cp <= 0x232A) ||
           (cp >= 0x2768 && cp <= 0x2775) || (cp >= 0x27C5 && cp <= 0x27C6) ||
           (cp >= 0x27E6 && cp <= 0x27EF) || (cp >= 0x2983 && cp <= 0x2998) ||
           (cp >= 0x29D8 && cp <= 0x29DB) || (cp >= 0x29FC && cp <= 0x29FD) ||
           (cp >= 0x2CF9 && cp <= 0x2CFC) || (cp >= 0x2CFE && cp <= 0x2CFF) ||
           (cp >= 0x2E00 && cp <= 0x2E2E) || (cp >= 0x2E30 && cp <= 0x2E4F) ||
           (cp >= 0x2E52 && cp <= 0x2E5D) || (cp >= 0x3001 && cp <= 0x3003) ||
           (cp >= 0x3008 && cp <= 0x3011) || (cp >= 0x3014 && cp <= 0x301F) ||
           (cp >= 0xA4FE && cp <= 0xA4FF) || (cp >= 0xA60D && cp <= 0xA60F) ||
           (cp >= 0xA6F2 && cp <= 0xA6F7) || (cp >= 0xA874 && cp <= 0xA877) ||
           (cp >= 0xA8CE && cp <= 0xA8CF) || (cp >= 0xA8F8 && cp <= 0xA8FA) ||
           (cp >= 0xA92E && cp <= 0xA92F) || (cp >= 0xA9C1 && cp <= 0xA9CD) ||
           (cp >= 0xA9DE && cp <= 0xA9DF) || (cp >= 0xAA5C && cp <= 0xAA5F) ||
           (cp >= 0xAADE && cp <= 0xAADF) || (cp >= 0xAAF0 && cp <= 0xAAF1) ||
           (cp >= 0xFD3E && cp <= 0xFD3F) || (cp >= 0xFE10 && cp <= 0xFE19) ||
           (cp >= 0xFE30 && cp <= 0xFE52) || (cp >= 0xFE54 && cp <= 0xFE61) ||
           (cp >= 0xFE6A && cp <= 0xFE6B) || (cp >= 0xFF01 && cp <= 0xFF03) ||
           (cp >= 0xFF05 && cp <= 0xFF0A) || (cp >= 0xFF0C && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF1B) || (cp >= 0xFF1F && cp <= 0xFF20) ||
           (cp >= 0xFF3B && cp <= 0xFF3D) || (cp >= 0xFF5F && cp <= 0xFF65) ||
           (cp >= 0x10100 && cp <= 0x10102) ||
           (cp >= 0x10A50 && cp <= 0x10A58) ||
           (cp >= 0x10AF0 && cp <= 0x10AF6) ||
           (cp >= 0x10B39 && cp <= 0x10B3F) ||
           (cp >= 0x10B99 && cp <= 0x10B9C) ||
           (cp >= 0x10F55 && cp <= 0x10F59) ||
           (cp >= 0x10F86 && cp <= 0x10F89) ||
           (cp >= 0x11047 && cp <= 0x1104D) ||
           (cp >= 0x110BB && cp <= 0x110BC) ||
           (cp >= 0x110BE && cp <= 0x110C1) ||
           (cp >= 0x11140 && cp <= 0x11143) ||
           (cp >= 0x11174 && cp <= 0x11175) ||
           (cp >= 0x111C5 && cp <= 0x111C8) ||
           (cp >= 0x111DD && cp <= 0x111DF) ||
           (cp >= 0x11238 && cp <= 0x1123D) ||
           (cp >= 0x1144B && cp <= 0x1144F) ||
           (cp >= 0x1145A && cp <= 0x1145B) ||
           (cp >= 0x115C1 && cp <= 0x115D7) ||
           (cp >= 0x11641 && cp <= 0x11643) ||
           (cp >= 0x11660 && cp <= 0x1166C) ||
           (cp >= 0x1173C && cp <= 0x1173E) ||
           (cp >= 0x11944 && cp <= 0x11946) ||
           (cp >= 0x11A3F && cp <= 0x11A46) ||
           (cp >= 0x11A9A && cp <= 0x11A9C) ||
           (cp >= 0x11A9E && cp <= 0x11AA2) ||
           (cp >= 0x11B00 && cp <= 0x11B09) ||
           (cp >= 0x11C41 && cp <= 0x11C45) ||
           (cp >= 0x11C70 && cp <= 0x11C71) ||
           (cp >= 0x11EF7 && cp <= 0x11EF8) ||
           (cp >= 0x11F43 && cp <= 0x11F4F) ||
           (cp >= 0x12470 && cp <= 0x12474) ||
           (cp >= 0x12FF1 && cp <= 0x12FF2) ||
           (cp >= 0x16A6E && cp <= 0x16A6F) ||
           (cp >= 0x16B37 && cp <= 0x16B3B) ||
           (cp >= 0x16E97 && cp <= 0x16E9A) ||
           (cp >= 0x1DA87 && cp <= 0x1DA8B) ||
           (cp >= 0x1E95E && cp <= 0x1E95F);
}

int PSUTF8IsDigit(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_digit_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_digit_codes[i] == cp) return 1;
    return (cp >= 0x30 && cp <= 0x39) ||
           (cp >= 0xB2 && cp <= 0xB3) || (cp >= 0xBC && cp <= 0xBE) ||
           (cp >= 0x660 && cp <= 0x669) || (cp >= 0x6F0 && cp <= 0x6F9) ||
           (cp >= 0x7C0 && cp <= 0x7C9) || (cp >= 0x966 && cp <= 0x96F) ||
           (cp >= 0x9E6 && cp <= 0x9EF) || (cp >= 0x9F4 && cp <= 0x9F9) ||
           (cp >= 0xA66 && cp <= 0xA6F) || (cp >= 0xAE6 && cp <= 0xAEF) ||
           (cp >= 0xB66 && cp <= 0xB6F) || (cp >= 0xB72 && cp <= 0xB77) ||
           (cp >= 0xBE6 && cp <= 0xBF2) || (cp >= 0xC66 && cp <= 0xC6F) ||
           (cp >= 0xC78 && cp <= 0xC7E) || (cp >= 0xCE6 && cp <= 0xCEF) ||
           (cp >= 0xD58 && cp <= 0xD5E) || (cp >= 0xD66 && cp <= 0xD78) ||
           (cp >= 0xDE6 && cp <= 0xDEF) || (cp >= 0xE50 && cp <= 0xE59) ||
           (cp >= 0xED0 && cp <= 0xED9) || (cp >= 0xF20 && cp <= 0xF33) ||
           (cp >= 0x1040 && cp <= 0x1049) || (cp >= 0x1090 && cp <= 0x1099) ||
           (cp >= 0x1369 && cp <= 0x137C) || (cp >= 0x16EE && cp <= 0x16F0) ||
           (cp >= 0x17E0 && cp <= 0x17E9) || (cp >= 0x17F0 && cp <= 0x17F9) ||
           (cp >= 0x1810 && cp <= 0x1819) || (cp >= 0x1946 && cp <= 0x194F) ||
           (cp >= 0x19D0 && cp <= 0x19DA) || (cp >= 0x1A80 && cp <= 0x1A89) ||
           (cp >= 0x1A90 && cp <= 0x1A99) || (cp >= 0x1B50 && cp <= 0x1B59) ||
           (cp >= 0x1BB0 && cp <= 0x1BB9) || (cp >= 0x1C40 && cp <= 0x1C49) ||
           (cp >= 0x1C50 && cp <= 0x1C59) || (cp >= 0x2074 && cp <= 0x2079) ||
           (cp >= 0x2080 && cp <= 0x2089) || (cp >= 0x2150 && cp <= 0x2182) ||
           (cp >= 0x2185 && cp <= 0x2189) || (cp >= 0x2460 && cp <= 0x249B) ||
           (cp >= 0x24EA && cp <= 0x24FF) || (cp >= 0x2776 && cp <= 0x2793) ||
           (cp >= 0x3021 && cp <= 0x3029) || (cp >= 0x3038 && cp <= 0x303A) ||
           (cp >= 0x3192 && cp <= 0x3195) || (cp >= 0x3220 && cp <= 0x3229) ||
           (cp >= 0x3248 && cp <= 0x324F) || (cp >= 0x3251 && cp <= 0x325F) ||
           (cp >= 0x3280 && cp <= 0x3289) || (cp >= 0x32B1 && cp <= 0x32BF) ||
           (cp >= 0xA620 && cp <= 0xA629) || (cp >= 0xA6E6 && cp <= 0xA6EF) ||
           (cp >= 0xA830 && cp <= 0xA835) || (cp >= 0xA8D0 && cp <= 0xA8D9) ||
           (cp >= 0xA900 && cp <= 0xA909) || (cp >= 0xA9D0 && cp <= 0xA9D9) ||
           (cp >= 0xA9F0 && cp <= 0xA9F9) || (cp >= 0xAA50 && cp <= 0xAA59) ||
           (cp >= 0xABF0 && cp <= 0xABF9) || (cp >= 0xFF10 && cp <= 0xFF19) ||
           (cp >= 0x10107 && cp <= 0x10133) ||
           (cp >= 0x10140 && cp <= 0x10178) ||
           (cp >= 0x1018A && cp <= 0x1018B) ||
           (cp >= 0x102E1 && cp <= 0x102FB) ||
           (cp >= 0x10320 && cp <= 0x10323) ||
           (cp >= 0x103D1 && cp <= 0x103D5) ||
           (cp >= 0x104A0 && cp <= 0x104A9) ||
           (cp >= 0x10858 && cp <= 0x1085F) ||
           (cp >= 0x10879 && cp <= 0x1087F) ||
           (cp >= 0x108A7 && cp <= 0x108AF) ||
           (cp >= 0x108FB && cp <= 0x108FF) ||
           (cp >= 0x10916 && cp <= 0x1091B) ||
           (cp >= 0x109BC && cp <= 0x109BD) ||
           (cp >= 0x109C0 && cp <= 0x109CF) ||
           (cp >= 0x109D2 && cp <= 0x109FF) ||
           (cp >= 0x10A40 && cp <= 0x10A48) ||
           (cp >= 0x10A7D && cp <= 0x10A7E) ||
           (cp >= 0x10A9D && cp <= 0x10A9F) ||
           (cp >= 0x10AEB && cp <= 0x10AEF) ||
           (cp >= 0x10B58 && cp <= 0x10B5F) ||
           (cp >= 0x10B78 && cp <= 0x10B7F) ||
           (cp >= 0x10BA9 && cp <= 0x10BAF) ||
           (cp >= 0x10CFA && cp <= 0x10CFF) ||
           (cp >= 0x10D30 && cp <= 0x10D39) ||
           (cp >= 0x10E60 && cp <= 0x10E7E) ||
           (cp >= 0x10F1D && cp <= 0x10F26) ||
           (cp >= 0x10F51 && cp <= 0x10F54) ||
           (cp >= 0x10FC5 && cp <= 0x10FCB) ||
           (cp >= 0x11052 && cp <= 0x1106F) ||
           (cp >= 0x110F0 && cp <= 0x110F9) ||
           (cp >= 0x11136 && cp <= 0x1113F) ||
           (cp >= 0x111D0 && cp <= 0x111D9) ||
           (cp >= 0x111E1 && cp <= 0x111F4) ||
           (cp >= 0x112F0 && cp <= 0x112F9) ||
           (cp >= 0x11450 && cp <= 0x11459) ||
           (cp >= 0x114D0 && cp <= 0x114D9) ||
           (cp >= 0x11650 && cp <= 0x11659) ||
           (cp >= 0x116C0 && cp <= 0x116C9) ||
           (cp >= 0x11730 && cp <= 0x1173B) ||
           (cp >= 0x118E0 && cp <= 0x118F2) ||
           (cp >= 0x11950 && cp <= 0x11959) ||
           (cp >= 0x11C50 && cp <= 0x11C6C) ||
           (cp >= 0x11D50 && cp <= 0x11D59) ||
           (cp >= 0x11DA0 && cp <= 0x11DA9) ||
           (cp >= 0x11F50 && cp <= 0x11F59) ||
           (cp >= 0x11FC0 && cp <= 0x11FD4) ||
           (cp >= 0x12400 && cp <= 0x1246E) ||
           (cp >= 0x16A60 && cp <= 0x16A69) ||
           (cp >= 0x16AC0 && cp <= 0x16AC9) ||
           (cp >= 0x16B50 && cp <= 0x16B59) ||
           (cp >= 0x16B5B && cp <= 0x16B61) ||
           (cp >= 0x16E80 && cp <= 0x16E96) ||
           (cp >= 0x1D2C0 && cp <= 0x1D2D3) ||
           (cp >= 0x1D2E0 && cp <= 0x1D2F3) ||
           (cp >= 0x1D360 && cp <= 0x1D378) ||
           (cp >= 0x1D7CE && cp <= 0x1D7FF) ||
           (cp >= 0x1E140 && cp <= 0x1E149) ||
           (cp >= 0x1E2F0 && cp <= 0x1E2F9) ||
           (cp >= 0x1E4F0 && cp <= 0x1E4F9) ||
           (cp >= 0x1E8C7 && cp <= 0x1E8CF) ||
           (cp >= 0x1E950 && cp <= 0x1E959) ||
           (cp >= 0x1EC71 && cp <= 0x1ECAB) ||
           (cp >= 0x1ECAD && cp <= 0x1ECAF) ||
           (cp >= 0x1ECB1 && cp <= 0x1ECB4) ||
           (cp >= 0x1ED01 && cp <= 0x1ED2D) ||
           (cp >= 0x1ED2F && cp <= 0x1ED3D) ||
           (cp >= 0x1F100 && cp <= 0x1F10C) ||
           (cp >= 0x1FBF0 && cp <= 0x1FBF9);
}

int PSUTF8IsAlpha(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_alpha_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_alpha_codes[i] == cp) return 1;
    return (cp >= 0x41 && cp <= 0x5A) ||
           (cp >= 0x61 && cp <= 0x7A) || (cp >= 0xC0 && cp <= 0xD6) ||
           (cp >= 0xD8 && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0x2C1) ||
           (cp >= 0x2C6 && cp <= 0x2D1) || (cp >= 0x2E0 && cp <= 0x2E4) ||
           (cp >= 0x370 && cp <= 0x374) || (cp >= 0x376 && cp <= 0x377) ||
           (cp >= 0x37A && cp <= 0x37D) || (cp >= 0x388 && cp <= 0x38A) ||
           (cp >= 0x38E && cp <= 0x3A1) || (cp >= 0x3A3 && cp <= 0x3F5) ||
           (cp >= 0x3F7 && cp <= 0x481) || (cp >= 0x48A && cp <= 0x52F) ||
           (cp >= 0x531 && cp <= 0x556) || (cp >= 0x560 && cp <= 0x588) ||
           (cp >= 0x5D0 && cp <= 0x5EA) || (cp >= 0x5EF && cp <= 0x5F2) ||
           (cp >= 0x620 && cp <= 0x64A) || (cp >= 0x66E && cp <= 0x66F) ||
           (cp >= 0x671 && cp <= 0x6D3) || (cp >= 0x6E5 && cp <= 0x6E6) ||
           (cp >= 0x6EE && cp <= 0x6EF) || (cp >= 0x6FA && cp <= 0x6FC) ||
           (cp >= 0x712 && cp <= 0x72F) || (cp >= 0x74D && cp <= 0x7A5) ||
           (cp >= 0x7CA && cp <= 0x7EA) || (cp >= 0x7F4 && cp <= 0x7F5) ||
           (cp >= 0x800 && cp <= 0x815) || (cp >= 0x840 && cp <= 0x858) ||
           (cp >= 0x860 && cp <= 0x86A) || (cp >= 0x870 && cp <= 0x887) ||
           (cp >= 0x889 && cp <= 0x88E) || (cp >= 0x8A0 && cp <= 0x8C9) ||
           (cp >= 0x904 && cp <= 0x939) || (cp >= 0x958 && cp <= 0x961) ||
           (cp >= 0x971 && cp <= 0x980) || (cp >= 0x985 && cp <= 0x98C) ||
           (cp >= 0x98F && cp <= 0x990) || (cp >= 0x993 && cp <= 0x9A8) ||
           (cp >= 0x9AA && cp <= 0x9B0) || (cp >= 0x9B6 && cp <= 0x9B9) ||
           (cp >= 0x9DC && cp <= 0x9DD) || (cp >= 0x9DF && cp <= 0x9E1) ||
           (cp >= 0x9F0 && cp <= 0x9F1) || (cp >= 0xA05 && cp <= 0xA0A) ||
           (cp >= 0xA0F && cp <= 0xA10) || (cp >= 0xA13 && cp <= 0xA28) ||
           (cp >= 0xA2A && cp <= 0xA30) || (cp >= 0xA32 && cp <= 0xA33) ||
           (cp >= 0xA35 && cp <= 0xA36) || (cp >= 0xA38 && cp <= 0xA39) ||
           (cp >= 0xA59 && cp <= 0xA5C) || (cp >= 0xA72 && cp <= 0xA74) ||
           (cp >= 0xA85 && cp <= 0xA8D) || (cp >= 0xA8F && cp <= 0xA91) ||
           (cp >= 0xA93 && cp <= 0xAA8) || (cp >= 0xAAA && cp <= 0xAB0) ||
           (cp >= 0xAB2 && cp <= 0xAB3) || (cp >= 0xAB5 && cp <= 0xAB9) ||
           (cp >= 0xAE0 && cp <= 0xAE1) || (cp >= 0xB05 && cp <= 0xB0C) ||
           (cp >= 0xB0F && cp <= 0xB10) || (cp >= 0xB13 && cp <= 0xB28) ||
           (cp >= 0xB2A && cp <= 0xB30) || (cp >= 0xB32 && cp <= 0xB33) ||
           (cp >= 0xB35 && cp <= 0xB39) || (cp >= 0xB5C && cp <= 0xB5D) ||
           (cp >= 0xB5F && cp <= 0xB61) || (cp >= 0xB85 && cp <= 0xB8A) ||
           (cp >= 0xB8E && cp <= 0xB90) || (cp >= 0xB92 && cp <= 0xB95) ||
           (cp >= 0xB99 && cp <= 0xB9A) || (cp >= 0xB9E && cp <= 0xB9F) ||
           (cp >= 0xBA3 && cp <= 0xBA4) || (cp >= 0xBA8 && cp <= 0xBAA) ||
           (cp >= 0xBAE && cp <= 0xBB9) || (cp >= 0xC05 && cp <= 0xC0C) ||
           (cp >= 0xC0E && cp <= 0xC10) || (cp >= 0xC12 && cp <= 0xC28) ||
           (cp >= 0xC2A && cp <= 0xC39) || (cp >= 0xC58 && cp <= 0xC5A) ||
           (cp >= 0xC60 && cp <= 0xC61) || (cp >= 0xC85 && cp <= 0xC8C) ||
           (cp >= 0xC8E && cp <= 0xC90) || (cp >= 0xC92 && cp <= 0xCA8) ||
           (cp >= 0xCAA && cp <= 0xCB3) || (cp >= 0xCB5 && cp <= 0xCB9) ||
           (cp >= 0xCDD && cp <= 0xCDE) || (cp >= 0xCE0 && cp <= 0xCE1) ||
           (cp >= 0xCF1 && cp <= 0xCF2) || (cp >= 0xD04 && cp <= 0xD0C) ||
           (cp >= 0xD0E && cp <= 0xD10) || (cp >= 0xD12 && cp <= 0xD3A) ||
           (cp >= 0xD54 && cp <= 0xD56) || (cp >= 0xD5F && cp <= 0xD61) ||
           (cp >= 0xD7A && cp <= 0xD7F) || (cp >= 0xD85 && cp <= 0xD96) ||
           (cp >= 0xD9A && cp <= 0xDB1) || (cp >= 0xDB3 && cp <= 0xDBB) ||
           (cp >= 0xDC0 && cp <= 0xDC6) || (cp >= 0xE01 && cp <= 0xE30) ||
           (cp >= 0xE32 && cp <= 0xE33) || (cp >= 0xE40 && cp <= 0xE46) ||
           (cp >= 0xE81 && cp <= 0xE82) || (cp >= 0xE86 && cp <= 0xE8A) ||
           (cp >= 0xE8C && cp <= 0xEA3) || (cp >= 0xEA7 && cp <= 0xEB0) ||
           (cp >= 0xEB2 && cp <= 0xEB3) || (cp >= 0xEC0 && cp <= 0xEC4) ||
           (cp >= 0xEDC && cp <= 0xEDF) || (cp >= 0xF40 && cp <= 0xF47) ||
           (cp >= 0xF49 && cp <= 0xF6C) || (cp >= 0xF88 && cp <= 0xF8C) ||
           (cp >= 0x1000 && cp <= 0x102A) || (cp >= 0x1050 && cp <= 0x1055) ||
           (cp >= 0x105A && cp <= 0x105D) || (cp >= 0x1065 && cp <= 0x1066) ||
           (cp >= 0x106E && cp <= 0x1070) || (cp >= 0x1075 && cp <= 0x1081) ||
           (cp >= 0x10A0 && cp <= 0x10C5) || (cp >= 0x10D0 && cp <= 0x10FA) ||
           (cp >= 0x10FC && cp <= 0x1248) || (cp >= 0x124A && cp <= 0x124D) ||
           (cp >= 0x1250 && cp <= 0x1256) || (cp >= 0x125A && cp <= 0x125D) ||
           (cp >= 0x1260 && cp <= 0x1288) || (cp >= 0x128A && cp <= 0x128D) ||
           (cp >= 0x1290 && cp <= 0x12B0) || (cp >= 0x12B2 && cp <= 0x12B5) ||
           (cp >= 0x12B8 && cp <= 0x12BE) || (cp >= 0x12C2 && cp <= 0x12C5) ||
           (cp >= 0x12C8 && cp <= 0x12D6) || (cp >= 0x12D8 && cp <= 0x1310) ||
           (cp >= 0x1312 && cp <= 0x1315) || (cp >= 0x1318 && cp <= 0x135A) ||
           (cp >= 0x1380 && cp <= 0x138F) || (cp >= 0x13A0 && cp <= 0x13F5) ||
           (cp >= 0x13F8 && cp <= 0x13FD) || (cp >= 0x1401 && cp <= 0x166C) ||
           (cp >= 0x166F && cp <= 0x167F) || (cp >= 0x1681 && cp <= 0x169A) ||
           (cp >= 0x16A0 && cp <= 0x16EA) || (cp >= 0x16F1 && cp <= 0x16F8) ||
           (cp >= 0x1700 && cp <= 0x1711) || (cp >= 0x171F && cp <= 0x1731) ||
           (cp >= 0x1740 && cp <= 0x1751) || (cp >= 0x1760 && cp <= 0x176C) ||
           (cp >= 0x176E && cp <= 0x1770) || (cp >= 0x1780 && cp <= 0x17B3) ||
           (cp >= 0x1820 && cp <= 0x1878) || (cp >= 0x1880 && cp <= 0x1884) ||
           (cp >= 0x1887 && cp <= 0x18A8) || (cp >= 0x18B0 && cp <= 0x18F5) ||
           (cp >= 0x1900 && cp <= 0x191E) || (cp >= 0x1950 && cp <= 0x196D) ||
           (cp >= 0x1970 && cp <= 0x1974) || (cp >= 0x1980 && cp <= 0x19AB) ||
           (cp >= 0x19B0 && cp <= 0x19C9) || (cp >= 0x1A00 && cp <= 0x1A16) ||
           (cp >= 0x1A20 && cp <= 0x1A54) || (cp >= 0x1B05 && cp <= 0x1B33) ||
           (cp >= 0x1B45 && cp <= 0x1B4C) || (cp >= 0x1B83 && cp <= 0x1BA0) ||
           (cp >= 0x1BAE && cp <= 0x1BAF) || (cp >= 0x1BBA && cp <= 0x1BE5) ||
           (cp >= 0x1C00 && cp <= 0x1C23) || (cp >= 0x1C4D && cp <= 0x1C4F) ||
           (cp >= 0x1C5A && cp <= 0x1C7D) || (cp >= 0x1C80 && cp <= 0x1C88) ||
           (cp >= 0x1C90 && cp <= 0x1CBA) || (cp >= 0x1CBD && cp <= 0x1CBF) ||
           (cp >= 0x1CE9 && cp <= 0x1CEC) || (cp >= 0x1CEE && cp <= 0x1CF3) ||
           (cp >= 0x1CF5 && cp <= 0x1CF6) || (cp >= 0x1D00 && cp <= 0x1DBF) ||
           (cp >= 0x1E00 && cp <= 0x1F15) || (cp >= 0x1F18 && cp <= 0x1F1D) ||
           (cp >= 0x1F20 && cp <= 0x1F45) || (cp >= 0x1F48 && cp <= 0x1F4D) ||
           (cp >= 0x1F50 && cp <= 0x1F57) || (cp >= 0x1F5F && cp <= 0x1F7D) ||
           (cp >= 0x1F80 && cp <= 0x1FB4) || (cp >= 0x1FB6 && cp <= 0x1FBC) ||
           (cp >= 0x1FC2 && cp <= 0x1FC4) || (cp >= 0x1FC6 && cp <= 0x1FCC) ||
           (cp >= 0x1FD0 && cp <= 0x1FD3) || (cp >= 0x1FD6 && cp <= 0x1FDB) ||
           (cp >= 0x1FE0 && cp <= 0x1FEC) || (cp >= 0x1FF2 && cp <= 0x1FF4) ||
           (cp >= 0x1FF6 && cp <= 0x1FFC) || (cp >= 0x2090 && cp <= 0x209C) ||
           (cp >= 0x210A && cp <= 0x2113) || (cp >= 0x2119 && cp <= 0x211D) ||
           (cp >= 0x212A && cp <= 0x212D) || (cp >= 0x212F && cp <= 0x2139) ||
           (cp >= 0x213C && cp <= 0x213F) || (cp >= 0x2145 && cp <= 0x2149) ||
           (cp >= 0x2183 && cp <= 0x2184) || (cp >= 0x2C00 && cp <= 0x2CE4) ||
           (cp >= 0x2CEB && cp <= 0x2CEE) || (cp >= 0x2CF2 && cp <= 0x2CF3) ||
           (cp >= 0x2D00 && cp <= 0x2D25) || (cp >= 0x2D30 && cp <= 0x2D67) ||
           (cp >= 0x2D80 && cp <= 0x2D96) || (cp >= 0x2DA0 && cp <= 0x2DA6) ||
           (cp >= 0x2DA8 && cp <= 0x2DAE) || (cp >= 0x2DB0 && cp <= 0x2DB6) ||
           (cp >= 0x2DB8 && cp <= 0x2DBE) || (cp >= 0x2DC0 && cp <= 0x2DC6) ||
           (cp >= 0x2DC8 && cp <= 0x2DCE) || (cp >= 0x2DD0 && cp <= 0x2DD6) ||
           (cp >= 0x2DD8 && cp <= 0x2DDE) || (cp >= 0x3005 && cp <= 0x3006) ||
           (cp >= 0x3031 && cp <= 0x3035) || (cp >= 0x303B && cp <= 0x303C) ||
           (cp >= 0x3041 && cp <= 0x3096) || (cp >= 0x309D && cp <= 0x309F) ||
           (cp >= 0x30A1 && cp <= 0x30FA) || (cp >= 0x30FC && cp <= 0x30FF) ||
           (cp >= 0x3105 && cp <= 0x312F) || (cp >= 0x3131 && cp <= 0x318E) ||
           (cp >= 0x31A0 && cp <= 0x31BF) || (cp >= 0x31F0 && cp <= 0x31FF) ||
           (cp >= 0x9FFF && cp <= 0xA48C) || (cp >= 0xA4D0 && cp <= 0xA4FD) ||
           (cp >= 0xA500 && cp <= 0xA60C) || (cp >= 0xA610 && cp <= 0xA61F) ||
           (cp >= 0xA62A && cp <= 0xA62B) || (cp >= 0xA640 && cp <= 0xA66E) ||
           (cp >= 0xA67F && cp <= 0xA69D) || (cp >= 0xA6A0 && cp <= 0xA6E5) ||
           (cp >= 0xA717 && cp <= 0xA71F) || (cp >= 0xA722 && cp <= 0xA788) ||
           (cp >= 0xA78B && cp <= 0xA7CA) || (cp >= 0xA7D0 && cp <= 0xA7D1) ||
           (cp >= 0xA7D5 && cp <= 0xA7D9) || (cp >= 0xA7F2 && cp <= 0xA801) ||
           (cp >= 0xA803 && cp <= 0xA805) || (cp >= 0xA807 && cp <= 0xA80A) ||
           (cp >= 0xA80C && cp <= 0xA822) || (cp >= 0xA840 && cp <= 0xA873) ||
           (cp >= 0xA882 && cp <= 0xA8B3) || (cp >= 0xA8F2 && cp <= 0xA8F7) ||
           (cp >= 0xA8FD && cp <= 0xA8FE) || (cp >= 0xA90A && cp <= 0xA925) ||
           (cp >= 0xA930 && cp <= 0xA946) || (cp >= 0xA960 && cp <= 0xA97C) ||
           (cp >= 0xA984 && cp <= 0xA9B2) || (cp >= 0xA9E0 && cp <= 0xA9E4) ||
           (cp >= 0xA9E6 && cp <= 0xA9EF) || (cp >= 0xA9FA && cp <= 0xA9FE) ||
           (cp >= 0xAA00 && cp <= 0xAA28) || (cp >= 0xAA40 && cp <= 0xAA42) ||
           (cp >= 0xAA44 && cp <= 0xAA4B) || (cp >= 0xAA60 && cp <= 0xAA76) ||
           (cp >= 0xAA7E && cp <= 0xAAAF) || (cp >= 0xAAB5 && cp <= 0xAAB6) ||
           (cp >= 0xAAB9 && cp <= 0xAABD) || (cp >= 0xAADB && cp <= 0xAADD) ||
           (cp >= 0xAAE0 && cp <= 0xAAEA) || (cp >= 0xAAF2 && cp <= 0xAAF4) ||
           (cp >= 0xAB01 && cp <= 0xAB06) || (cp >= 0xAB09 && cp <= 0xAB0E) ||
           (cp >= 0xAB11 && cp <= 0xAB16) || (cp >= 0xAB20 && cp <= 0xAB26) ||
           (cp >= 0xAB28 && cp <= 0xAB2E) || (cp >= 0xAB30 && cp <= 0xAB5A) ||
           (cp >= 0xAB5C && cp <= 0xAB69) || (cp >= 0xAB70 && cp <= 0xABE2) ||
           (cp >= 0xD7B0 && cp <= 0xD7C6) || (cp >= 0xD7CB && cp <= 0xD7FB) ||
           (cp >= 0xF900 && cp <= 0xFA6D) || (cp >= 0xFA70 && cp <= 0xFAD9) ||
           (cp >= 0xFB00 && cp <= 0xFB06) || (cp >= 0xFB13 && cp <= 0xFB17) ||
           (cp >= 0xFB1F && cp <= 0xFB28) || (cp >= 0xFB2A && cp <= 0xFB36) ||
           (cp >= 0xFB38 && cp <= 0xFB3C) || (cp >= 0xFB40 && cp <= 0xFB41) ||
           (cp >= 0xFB43 && cp <= 0xFB44) || (cp >= 0xFB46 && cp <= 0xFBB1) ||
           (cp >= 0xFBD3 && cp <= 0xFD3D) || (cp >= 0xFD50 && cp <= 0xFD8F) ||
           (cp >= 0xFD92 && cp <= 0xFDC7) || (cp >= 0xFDF0 && cp <= 0xFDFB) ||
           (cp >= 0xFE70 && cp <= 0xFE74) || (cp >= 0xFE76 && cp <= 0xFEFC) ||
           (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A) ||
           (cp >= 0xFF66 && cp <= 0xFFBE) || (cp >= 0xFFC2 && cp <= 0xFFC7) ||
           (cp >= 0xFFCA && cp <= 0xFFCF) || (cp >= 0xFFD2 && cp <= 0xFFD7) ||
           (cp >= 0xFFDA && cp <= 0xFFDC) ||
           (cp >= 0x10000 && cp <= 0x1000B) ||
           (cp >= 0x1000D && cp <= 0x10026) ||
           (cp >= 0x10028 && cp <= 0x1003A) ||
           (cp >= 0x1003C && cp <= 0x1003D) ||
           (cp >= 0x1003F && cp <= 0x1004D) ||
           (cp >= 0x10050 && cp <= 0x1005D) ||
           (cp >= 0x10080 && cp <= 0x100FA) ||
           (cp >= 0x10280 && cp <= 0x1029C) ||
           (cp >= 0x102A0 && cp <= 0x102D0) ||
           (cp >= 0x10300 && cp <= 0x1031F) ||
           (cp >= 0x1032D && cp <= 0x10340) ||
           (cp >= 0x10342 && cp <= 0x10349) ||
           (cp >= 0x10350 && cp <= 0x10375) ||
           (cp >= 0x10380 && cp <= 0x1039D) ||
           (cp >= 0x103A0 && cp <= 0x103C3) ||
           (cp >= 0x103C8 && cp <= 0x103CF) ||
           (cp >= 0x10400 && cp <= 0x1049D) ||
           (cp >= 0x104B0 && cp <= 0x104D3) ||
           (cp >= 0x104D8 && cp <= 0x104FB) ||
           (cp >= 0x10500 && cp <= 0x10527) ||
           (cp >= 0x10530 && cp <= 0x10563) ||
           (cp >= 0x10570 && cp <= 0x1057A) ||
           (cp >= 0x1057C && cp <= 0x1058A) ||
           (cp >= 0x1058C && cp <= 0x10592) ||
           (cp >= 0x10594 && cp <= 0x10595) ||
           (cp >= 0x10597 && cp <= 0x105A1) ||
           (cp >= 0x105A3 && cp <= 0x105B1) ||
           (cp >= 0x105B3 && cp <= 0x105B9) ||
           (cp >= 0x105BB && cp <= 0x105BC) ||
           (cp >= 0x10600 && cp <= 0x10736) ||
           (cp >= 0x10740 && cp <= 0x10755) ||
           (cp >= 0x10760 && cp <= 0x10767) ||
           (cp >= 0x10780 && cp <= 0x10785) ||
           (cp >= 0x10787 && cp <= 0x107B0) ||
           (cp >= 0x107B2 && cp <= 0x107BA) ||
           (cp >= 0x10800 && cp <= 0x10805) ||
           (cp >= 0x1080A && cp <= 0x10835) ||
           (cp >= 0x10837 && cp <= 0x10838) ||
           (cp >= 0x1083F && cp <= 0x10855) ||
           (cp >= 0x10860 && cp <= 0x10876) ||
           (cp >= 0x10880 && cp <= 0x1089E) ||
           (cp >= 0x108E0 && cp <= 0x108F2) ||
           (cp >= 0x108F4 && cp <= 0x108F5) ||
           (cp >= 0x10900 && cp <= 0x10915) ||
           (cp >= 0x10920 && cp <= 0x10939) ||
           (cp >= 0x10980 && cp <= 0x109B7) ||
           (cp >= 0x109BE && cp <= 0x109BF) ||
           (cp >= 0x10A10 && cp <= 0x10A13) ||
           (cp >= 0x10A15 && cp <= 0x10A17) ||
           (cp >= 0x10A19 && cp <= 0x10A35) ||
           (cp >= 0x10A60 && cp <= 0x10A7C) ||
           (cp >= 0x10A80 && cp <= 0x10A9C) ||
           (cp >= 0x10AC0 && cp <= 0x10AC7) ||
           (cp >= 0x10AC9 && cp <= 0x10AE4) ||
           (cp >= 0x10B00 && cp <= 0x10B35) ||
           (cp >= 0x10B40 && cp <= 0x10B55) ||
           (cp >= 0x10B60 && cp <= 0x10B72) ||
           (cp >= 0x10B80 && cp <= 0x10B91) ||
           (cp >= 0x10C00 && cp <= 0x10C48) ||
           (cp >= 0x10C80 && cp <= 0x10CB2) ||
           (cp >= 0x10CC0 && cp <= 0x10CF2) ||
           (cp >= 0x10D00 && cp <= 0x10D23) ||
           (cp >= 0x10E80 && cp <= 0x10EA9) ||
           (cp >= 0x10EB0 && cp <= 0x10EB1) ||
           (cp >= 0x10F00 && cp <= 0x10F1C) ||
           (cp >= 0x10F30 && cp <= 0x10F45) ||
           (cp >= 0x10F70 && cp <= 0x10F81) ||
           (cp >= 0x10FB0 && cp <= 0x10FC4) ||
           (cp >= 0x10FE0 && cp <= 0x10FF6) ||
           (cp >= 0x11003 && cp <= 0x11037) ||
           (cp >= 0x11071 && cp <= 0x11072) ||
           (cp >= 0x11083 && cp <= 0x110AF) ||
           (cp >= 0x110D0 && cp <= 0x110E8) ||
           (cp >= 0x11103 && cp <= 0x11126) ||
           (cp >= 0x11150 && cp <= 0x11172) ||
           (cp >= 0x11183 && cp <= 0x111B2) ||
           (cp >= 0x111C1 && cp <= 0x111C4) ||
           (cp >= 0x11200 && cp <= 0x11211) ||
           (cp >= 0x11213 && cp <= 0x1122B) ||
           (cp >= 0x1123F && cp <= 0x11240) ||
           (cp >= 0x11280 && cp <= 0x11286) ||
           (cp >= 0x1128A && cp <= 0x1128D) ||
           (cp >= 0x1128F && cp <= 0x1129D) ||
           (cp >= 0x1129F && cp <= 0x112A8) ||
           (cp >= 0x112B0 && cp <= 0x112DE) ||
           (cp >= 0x11305 && cp <= 0x1130C) ||
           (cp >= 0x1130F && cp <= 0x11310) ||
           (cp >= 0x11313 && cp <= 0x11328) ||
           (cp >= 0x1132A && cp <= 0x11330) ||
           (cp >= 0x11332 && cp <= 0x11333) ||
           (cp >= 0x11335 && cp <= 0x11339) ||
           (cp >= 0x1135D && cp <= 0x11361) ||
           (cp >= 0x11400 && cp <= 0x11434) ||
           (cp >= 0x11447 && cp <= 0x1144A) ||
           (cp >= 0x1145F && cp <= 0x11461) ||
           (cp >= 0x11480 && cp <= 0x114AF) ||
           (cp >= 0x114C4 && cp <= 0x114C5) ||
           (cp >= 0x11580 && cp <= 0x115AE) ||
           (cp >= 0x115D8 && cp <= 0x115DB) ||
           (cp >= 0x11600 && cp <= 0x1162F) ||
           (cp >= 0x11680 && cp <= 0x116AA) ||
           (cp >= 0x11700 && cp <= 0x1171A) ||
           (cp >= 0x11740 && cp <= 0x11746) ||
           (cp >= 0x11800 && cp <= 0x1182B) ||
           (cp >= 0x118A0 && cp <= 0x118DF) ||
           (cp >= 0x118FF && cp <= 0x11906) ||
           (cp >= 0x1190C && cp <= 0x11913) ||
           (cp >= 0x11915 && cp <= 0x11916) ||
           (cp >= 0x11918 && cp <= 0x1192F) ||
           (cp >= 0x119A0 && cp <= 0x119A7) ||
           (cp >= 0x119AA && cp <= 0x119D0) ||
           (cp >= 0x11A0B && cp <= 0x11A32) ||
           (cp >= 0x11A5C && cp <= 0x11A89) ||
           (cp >= 0x11AB0 && cp <= 0x11AF8) ||
           (cp >= 0x11C00 && cp <= 0x11C08) ||
           (cp >= 0x11C0A && cp <= 0x11C2E) ||
           (cp >= 0x11C72 && cp <= 0x11C8F) ||
           (cp >= 0x11D00 && cp <= 0x11D06) ||
           (cp >= 0x11D08 && cp <= 0x11D09) ||
           (cp >= 0x11D0B && cp <= 0x11D30) ||
           (cp >= 0x11D60 && cp <= 0x11D65) ||
           (cp >= 0x11D67 && cp <= 0x11D68) ||
           (cp >= 0x11D6A && cp <= 0x11D89) ||
           (cp >= 0x11EE0 && cp <= 0x11EF2) ||
           (cp >= 0x11F04 && cp <= 0x11F10) ||
           (cp >= 0x11F12 && cp <= 0x11F33) ||
           (cp >= 0x12000 && cp <= 0x12399) ||
           (cp >= 0x12480 && cp <= 0x12543) ||
           (cp >= 0x12F90 && cp <= 0x12FF0) ||
           (cp >= 0x13000 && cp <= 0x1342F) ||
           (cp >= 0x13441 && cp <= 0x13446) ||
           (cp >= 0x14400 && cp <= 0x14646) ||
           (cp >= 0x16800 && cp <= 0x16A38) ||
           (cp >= 0x16A40 && cp <= 0x16A5E) ||
           (cp >= 0x16A70 && cp <= 0x16ABE) ||
           (cp >= 0x16AD0 && cp <= 0x16AED) ||
           (cp >= 0x16B00 && cp <= 0x16B2F) ||
           (cp >= 0x16B40 && cp <= 0x16B43) ||
           (cp >= 0x16B63 && cp <= 0x16B77) ||
           (cp >= 0x16B7D && cp <= 0x16B8F) ||
           (cp >= 0x16E40 && cp <= 0x16E7F) ||
           (cp >= 0x16F00 && cp <= 0x16F4A) ||
           (cp >= 0x16F93 && cp <= 0x16F9F) ||
           (cp >= 0x16FE0 && cp <= 0x16FE1) ||
           (cp >= 0x18800 && cp <= 0x18CD5) ||
           (cp >= 0x1AFF0 && cp <= 0x1AFF3) ||
           (cp >= 0x1AFF5 && cp <= 0x1AFFB) ||
           (cp >= 0x1AFFD && cp <= 0x1AFFE) ||
           (cp >= 0x1B000 && cp <= 0x1B122) ||
           (cp >= 0x1B150 && cp <= 0x1B152) ||
           (cp >= 0x1B164 && cp <= 0x1B167) ||
           (cp >= 0x1B170 && cp <= 0x1B2FB) ||
           (cp >= 0x1BC00 && cp <= 0x1BC6A) ||
           (cp >= 0x1BC70 && cp <= 0x1BC7C) ||
           (cp >= 0x1BC80 && cp <= 0x1BC88) ||
           (cp >= 0x1BC90 && cp <= 0x1BC99) ||
           (cp >= 0x1D400 && cp <= 0x1D454) ||
           (cp >= 0x1D456 && cp <= 0x1D49C) ||
           (cp >= 0x1D49E && cp <= 0x1D49F) ||
           (cp >= 0x1D4A5 && cp <= 0x1D4A6) ||
           (cp >= 0x1D4A9 && cp <= 0x1D4AC) ||
           (cp >= 0x1D4AE && cp <= 0x1D4B9) ||
           (cp >= 0x1D4BD && cp <= 0x1D4C3) ||
           (cp >= 0x1D4C5 && cp <= 0x1D505) ||
           (cp >= 0x1D507 && cp <= 0x1D50A) ||
           (cp >= 0x1D50D && cp <= 0x1D514) ||
           (cp >= 0x1D516 && cp <= 0x1D51C) ||
           (cp >= 0x1D51E && cp <= 0x1D539) ||
           (cp >= 0x1D53B && cp <= 0x1D53E) ||
           (cp >= 0x1D540 && cp <= 0x1D544) ||
           (cp >= 0x1D54A && cp <= 0x1D550) ||
           (cp >= 0x1D552 && cp <= 0x1D6A5) ||
           (cp >= 0x1D6A8 && cp <= 0x1D6C0) ||
           (cp >= 0x1D6C2 && cp <= 0x1D6DA) ||
           (cp >= 0x1D6DC && cp <= 0x1D6FA) ||
           (cp >= 0x1D6FC && cp <= 0x1D714) ||
           (cp >= 0x1D716 && cp <= 0x1D734) ||
           (cp >= 0x1D736 && cp <= 0x1D74E) ||
           (cp >= 0x1D750 && cp <= 0x1D76E) ||
           (cp >= 0x1D770 && cp <= 0x1D788) ||
           (cp >= 0x1D78A && cp <= 0x1D7A8) ||
           (cp >= 0x1D7AA && cp <= 0x1D7C2) ||
           (cp >= 0x1D7C4 && cp <= 0x1D7CB) ||
           (cp >= 0x1DF00 && cp <= 0x1DF1E) ||
           (cp >= 0x1DF25 && cp <= 0x1DF2A) ||
           (cp >= 0x1E030 && cp <= 0x1E06D) ||
           (cp >= 0x1E100 && cp <= 0x1E12C) ||
           (cp >= 0x1E137 && cp <= 0x1E13D) ||
           (cp >= 0x1E290 && cp <= 0x1E2AD) ||
           (cp >= 0x1E2C0 && cp <= 0x1E2EB) ||
           (cp >= 0x1E4D0 && cp <= 0x1E4EB) ||
           (cp >= 0x1E7E0 && cp <= 0x1E7E6) ||
           (cp >= 0x1E7E8 && cp <= 0x1E7EB) ||
           (cp >= 0x1E7ED && cp <= 0x1E7EE) ||
           (cp >= 0x1E7F0 && cp <= 0x1E7FE) ||
           (cp >= 0x1E800 && cp <= 0x1E8C4) ||
           (cp >= 0x1E900 && cp <= 0x1E943) ||
           (cp >= 0x1EE00 && cp <= 0x1EE03) ||
           (cp >= 0x1EE05 && cp <= 0x1EE1F) ||
           (cp >= 0x1EE21 && cp <= 0x1EE22) ||
           (cp >= 0x1EE29 && cp <= 0x1EE32) ||
           (cp >= 0x1EE34 && cp <= 0x1EE37) ||
           (cp >= 0x1EE4D && cp <= 0x1EE4F) ||
           (cp >= 0x1EE51 && cp <= 0x1EE52) ||
           (cp >= 0x1EE61 && cp <= 0x1EE62) ||
           (cp >= 0x1EE67 && cp <= 0x1EE6A) ||
           (cp >= 0x1EE6C && cp <= 0x1EE72) ||
           (cp >= 0x1EE74 && cp <= 0x1EE77) ||
           (cp >= 0x1EE79 && cp <= 0x1EE7C) ||
           (cp >= 0x1EE80 && cp <= 0x1EE89) ||
           (cp >= 0x1EE8B && cp <= 0x1EE9B) ||
           (cp >= 0x1EEA1 && cp <= 0x1EEA3) ||
           (cp >= 0x1EEA5 && cp <= 0x1EEA9) ||
           (cp >= 0x1EEAB && cp <= 0x1EEBB) ||
           (cp >= 0x2F800 && cp <= 0x2FA1D);
}

int PSUTF8IsUpper(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_upper_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_upper_codes[i] == cp) return 1;
    return (cp >= 0x41 && cp <= 0x5A) ||
           (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xDE) ||
           (cp >= 0x178 && cp <= 0x179) || (cp >= 0x181 && cp <= 0x182) ||
           (cp >= 0x186 && cp <= 0x187) || (cp >= 0x189 && cp <= 0x18B) ||
           (cp >= 0x18E && cp <= 0x191) || (cp >= 0x193 && cp <= 0x194) ||
           (cp >= 0x196 && cp <= 0x198) || (cp >= 0x19C && cp <= 0x19D) ||
           (cp >= 0x19F && cp <= 0x1A0) || (cp >= 0x1A6 && cp <= 0x1A7) ||
           (cp >= 0x1AE && cp <= 0x1AF) || (cp >= 0x1B1 && cp <= 0x1B3) ||
           (cp >= 0x1B7 && cp <= 0x1B8) || (cp >= 0x1F6 && cp <= 0x1F8) ||
           (cp >= 0x23A && cp <= 0x23B) || (cp >= 0x23D && cp <= 0x23E) ||
           (cp >= 0x243 && cp <= 0x246) || (cp >= 0x388 && cp <= 0x38A) ||
           (cp >= 0x38E && cp <= 0x38F) || (cp >= 0x391 && cp <= 0x3A1) ||
           (cp >= 0x3A3 && cp <= 0x3AB) || (cp >= 0x3D2 && cp <= 0x3D4) ||
           (cp >= 0x3F9 && cp <= 0x3FA) || (cp >= 0x3FD && cp <= 0x42F) ||
           (cp >= 0x4C0 && cp <= 0x4C1) || (cp >= 0x531 && cp <= 0x556) ||
           (cp >= 0x10A0 && cp <= 0x10C5) || (cp >= 0x13A0 && cp <= 0x13F5) ||
           (cp >= 0x1C90 && cp <= 0x1CBA) || (cp >= 0x1CBD && cp <= 0x1CBF) ||
           (cp >= 0x1F08 && cp <= 0x1F0F) || (cp >= 0x1F18 && cp <= 0x1F1D) ||
           (cp >= 0x1F28 && cp <= 0x1F2F) || (cp >= 0x1F38 && cp <= 0x1F3F) ||
           (cp >= 0x1F48 && cp <= 0x1F4D) || (cp >= 0x1F68 && cp <= 0x1F6F) ||
           (cp >= 0x1FB8 && cp <= 0x1FBB) || (cp >= 0x1FC8 && cp <= 0x1FCB) ||
           (cp >= 0x1FD8 && cp <= 0x1FDB) || (cp >= 0x1FE8 && cp <= 0x1FEC) ||
           (cp >= 0x1FF8 && cp <= 0x1FFB) || (cp >= 0x210B && cp <= 0x210D) ||
           (cp >= 0x2110 && cp <= 0x2112) || (cp >= 0x2119 && cp <= 0x211D) ||
           (cp >= 0x212A && cp <= 0x212D) || (cp >= 0x2130 && cp <= 0x2133) ||
           (cp >= 0x213E && cp <= 0x213F) || (cp >= 0x2C00 && cp <= 0x2C2F) ||
           (cp >= 0x2C62 && cp <= 0x2C64) || (cp >= 0x2C6D && cp <= 0x2C70) ||
           (cp >= 0x2C7E && cp <= 0x2C80) || (cp >= 0xA77D && cp <= 0xA77E) ||
           (cp >= 0xA7AA && cp <= 0xA7AE) || (cp >= 0xA7B0 && cp <= 0xA7B4) ||
           (cp >= 0xA7C4 && cp <= 0xA7C7) || (cp >= 0xFF21 && cp <= 0xFF3A) ||
           (cp >= 0x10400 && cp <= 0x10427) ||
           (cp >= 0x104B0 && cp <= 0x104D3) ||
           (cp >= 0x10570 && cp <= 0x1057A) ||
           (cp >= 0x1057C && cp <= 0x1058A) ||
           (cp >= 0x1058C && cp <= 0x10592) ||
           (cp >= 0x10594 && cp <= 0x10595) ||
           (cp >= 0x10C80 && cp <= 0x10CB2) ||
           (cp >= 0x118A0 && cp <= 0x118BF) ||
           (cp >= 0x16E40 && cp <= 0x16E5F) ||
           (cp >= 0x1D400 && cp <= 0x1D419) ||
           (cp >= 0x1D434 && cp <= 0x1D44D) ||
           (cp >= 0x1D468 && cp <= 0x1D481) ||
           (cp >= 0x1D49E && cp <= 0x1D49F) ||
           (cp >= 0x1D4A5 && cp <= 0x1D4A6) ||
           (cp >= 0x1D4A9 && cp <= 0x1D4AC) ||
           (cp >= 0x1D4AE && cp <= 0x1D4B5) ||
           (cp >= 0x1D4D0 && cp <= 0x1D4E9) ||
           (cp >= 0x1D504 && cp <= 0x1D505) ||
           (cp >= 0x1D507 && cp <= 0x1D50A) ||
           (cp >= 0x1D50D && cp <= 0x1D514) ||
           (cp >= 0x1D516 && cp <= 0x1D51C) ||
           (cp >= 0x1D538 && cp <= 0x1D539) ||
           (cp >= 0x1D53B && cp <= 0x1D53E) ||
           (cp >= 0x1D540 && cp <= 0x1D544) ||
           (cp >= 0x1D54A && cp <= 0x1D550) ||
           (cp >= 0x1D56C && cp <= 0x1D585) ||
           (cp >= 0x1D5A0 && cp <= 0x1D5B9) ||
           (cp >= 0x1D5D4 && cp <= 0x1D5ED) ||
           (cp >= 0x1D608 && cp <= 0x1D621) ||
           (cp >= 0x1D63C && cp <= 0x1D655) ||
           (cp >= 0x1D670 && cp <= 0x1D689) ||
           (cp >= 0x1D6A8 && cp <= 0x1D6C0) ||
           (cp >= 0x1D6E2 && cp <= 0x1D6FA) ||
           (cp >= 0x1D71C && cp <= 0x1D734) ||
           (cp >= 0x1D756 && cp <= 0x1D76E) ||
           (cp >= 0x1D790 && cp <= 0x1D7A8) ||
           (cp >= 0x1E900 && cp <= 0x1E921);
}

int PSUTF8IsLower(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    int ncodes = (int) (sizeof(utf8_lower_codes) / sizeof(uint32_t)), i;
    for (i = 0; i < ncodes; i++)
        if (utf8_lower_codes[i] == cp) return 1;
    return (cp >= 0x61 && cp <= 0x7A) ||
           (cp >= 0xDF && cp <= 0xF6) || (cp >= 0xF8 && cp <= 0xFF) ||
           (cp >= 0x137 && cp <= 0x138) || (cp >= 0x148 && cp <= 0x149) ||
           (cp >= 0x17E && cp <= 0x180) || (cp >= 0x18C && cp <= 0x18D) ||
           (cp >= 0x199 && cp <= 0x19B) || (cp >= 0x1AA && cp <= 0x1AB) ||
           (cp >= 0x1B9 && cp <= 0x1BA) || (cp >= 0x1BD && cp <= 0x1BF) ||
           (cp >= 0x1DC && cp <= 0x1DD) || (cp >= 0x1EF && cp <= 0x1F0) ||
           (cp >= 0x233 && cp <= 0x239) || (cp >= 0x23F && cp <= 0x240) ||
           (cp >= 0x24F && cp <= 0x293) || (cp >= 0x295 && cp <= 0x2AF) ||
           (cp >= 0x37B && cp <= 0x37D) || (cp >= 0x3AC && cp <= 0x3CE) ||
           (cp >= 0x3D0 && cp <= 0x3D1) || (cp >= 0x3D5 && cp <= 0x3D7) ||
           (cp >= 0x3EF && cp <= 0x3F3) || (cp >= 0x3FB && cp <= 0x3FC) ||
           (cp >= 0x430 && cp <= 0x45F) || (cp >= 0x4CE && cp <= 0x4CF) ||
           (cp >= 0x560 && cp <= 0x588) || (cp >= 0x10D0 && cp <= 0x10FA) ||
           (cp >= 0x10FD && cp <= 0x10FF) || (cp >= 0x13F8 && cp <= 0x13FD) ||
           (cp >= 0x1C80 && cp <= 0x1C88) || (cp >= 0x1D00 && cp <= 0x1D2B) ||
           (cp >= 0x1D6B && cp <= 0x1D77) || (cp >= 0x1D79 && cp <= 0x1D9A) ||
           (cp >= 0x1E95 && cp <= 0x1E9D) || (cp >= 0x1EFF && cp <= 0x1F07) ||
           (cp >= 0x1F10 && cp <= 0x1F15) || (cp >= 0x1F20 && cp <= 0x1F27) ||
           (cp >= 0x1F30 && cp <= 0x1F37) || (cp >= 0x1F40 && cp <= 0x1F45) ||
           (cp >= 0x1F50 && cp <= 0x1F57) || (cp >= 0x1F60 && cp <= 0x1F67) ||
           (cp >= 0x1F70 && cp <= 0x1F7D) || (cp >= 0x1F80 && cp <= 0x1F87) ||
           (cp >= 0x1F90 && cp <= 0x1F97) || (cp >= 0x1FA0 && cp <= 0x1FA7) ||
           (cp >= 0x1FB0 && cp <= 0x1FB4) || (cp >= 0x1FB6 && cp <= 0x1FB7) ||
           (cp >= 0x1FC2 && cp <= 0x1FC4) || (cp >= 0x1FC6 && cp <= 0x1FC7) ||
           (cp >= 0x1FD0 && cp <= 0x1FD3) || (cp >= 0x1FD6 && cp <= 0x1FD7) ||
           (cp >= 0x1FE0 && cp <= 0x1FE7) || (cp >= 0x1FF2 && cp <= 0x1FF4) ||
           (cp >= 0x1FF6 && cp <= 0x1FF7) || (cp >= 0x210E && cp <= 0x210F) ||
           (cp >= 0x213C && cp <= 0x213D) || (cp >= 0x2146 && cp <= 0x2149) ||
           (cp >= 0x2C30 && cp <= 0x2C5F) || (cp >= 0x2C65 && cp <= 0x2C66) ||
           (cp >= 0x2C73 && cp <= 0x2C74) || (cp >= 0x2C76 && cp <= 0x2C7B) ||
           (cp >= 0x2CE3 && cp <= 0x2CE4) || (cp >= 0x2D00 && cp <= 0x2D25) ||
           (cp >= 0xA72F && cp <= 0xA731) || (cp >= 0xA771 && cp <= 0xA778) ||
           (cp >= 0xA793 && cp <= 0xA795) || (cp >= 0xAB30 && cp <= 0xAB5A) ||
           (cp >= 0xAB60 && cp <= 0xAB68) || (cp >= 0xAB70 && cp <= 0xABBF) ||
           (cp >= 0xFB00 && cp <= 0xFB06) || (cp >= 0xFB13 && cp <= 0xFB17) ||
           (cp >= 0xFF41 && cp <= 0xFF5A) ||
           (cp >= 0x10428 && cp <= 0x1044F) ||
           (cp >= 0x104D8 && cp <= 0x104FB) ||
           (cp >= 0x10597 && cp <= 0x105A1) ||
           (cp >= 0x105A3 && cp <= 0x105B1) ||
           (cp >= 0x105B3 && cp <= 0x105B9) ||
           (cp >= 0x105BB && cp <= 0x105BC) ||
           (cp >= 0x10CC0 && cp <= 0x10CF2) ||
           (cp >= 0x118C0 && cp <= 0x118DF) ||
           (cp >= 0x16E60 && cp <= 0x16E7F) ||
           (cp >= 0x1D41A && cp <= 0x1D433) ||
           (cp >= 0x1D44E && cp <= 0x1D454) ||
           (cp >= 0x1D456 && cp <= 0x1D467) ||
           (cp >= 0x1D482 && cp <= 0x1D49B) ||
           (cp >= 0x1D4B6 && cp <= 0x1D4B9) ||
           (cp >= 0x1D4BD && cp <= 0x1D4C3) ||
           (cp >= 0x1D4C5 && cp <= 0x1D4CF) ||
           (cp >= 0x1D4EA && cp <= 0x1D503) ||
           (cp >= 0x1D51E && cp <= 0x1D537) ||
           (cp >= 0x1D552 && cp <= 0x1D56B) ||
           (cp >= 0x1D586 && cp <= 0x1D59F) ||
           (cp >= 0x1D5BA && cp <= 0x1D5D3) ||
           (cp >= 0x1D5EE && cp <= 0x1D607) ||
           (cp >= 0x1D622 && cp <= 0x1D63B) ||
           (cp >= 0x1D656 && cp <= 0x1D66F) ||
           (cp >= 0x1D68A && cp <= 0x1D6A5) ||
           (cp >= 0x1D6C2 && cp <= 0x1D6DA) ||
           (cp >= 0x1D6DC && cp <= 0x1D6E1) ||
           (cp >= 0x1D6FC && cp <= 0x1D714) ||
           (cp >= 0x1D716 && cp <= 0x1D71B) ||
           (cp >= 0x1D736 && cp <= 0x1D74E) ||
           (cp >= 0x1D750 && cp <= 0x1D755) ||
           (cp >= 0x1D770 && cp <= 0x1D788) ||
           (cp >= 0x1D78A && cp <= 0x1D78F) ||
           (cp >= 0x1D7AA && cp <= 0x1D7C2) ||
           (cp >= 0x1D7C4 && cp <= 0x1D7C9) ||
           (cp >= 0x1DF00 && cp <= 0x1DF09) ||
           (cp >= 0x1DF0B && cp <= 0x1DF1E) ||
           (cp >= 0x1DF25 && cp <= 0x1DF2A) ||
           (cp >= 0x1E922 && cp <= 0x1E943);
}

int PSUTF8IsAlphanum(PSUTF8Char uc) {
    return PSUTF8IsAlpha(uc) || PSUTF8IsDigit(uc);
}

PSUTF8Char PSUTF8ToUpper(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    uint32_t upper_cp = utf8_lower_to_upper[cp];
    if (upper_cp == 0) return uc;
    return PSUTF8Encode(upper_cp);
}

PSUTF8Char PSUTF8ToLower(PSUTF8Char uc) {
    uint32_t cp = PSUTF8Decode(uc);
    uint32_t lower_cp = utf8_upper_to_lower[cp];
    if (lower_cp == 0) return uc;
    return PSUTF8Encode(lower_cp);
}

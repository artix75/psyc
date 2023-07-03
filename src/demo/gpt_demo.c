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

/* This demos tries to emulate GPT-2 model developed by OpenAI
 * (https://github.com/openai/gpt-2) using PsyC library. */

#include <stdio.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <sys/time.h>
#include <pwd.h>
#include <errno.h>

#ifdef HAS_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include "../psyc.h"
#include "../attention.h"
#include "../avx.h"
#include "../activation.h"
#include "../blas.h"
#include "../buildinfo.h"
#include "../convolutional.h"
#include "../config.h"
#include "../dropout.h"
#include "../debug.h"
#include "../embedding.h"
#include "../gru.h"
#include "../image_data.h"
#include "../log.h"
#include "../lstm.h"
#include "../dataset.h"
#include "../magick_conf.h"
#include "../maths.h"
#include "../normalization.h"
#include "../optimization.h"
#include "../operator_layer.h"
#include "../positional_encoding.h"
#include "../platform.h"
#include "../recurrent.h"
#include "../types.h"
#include "../utils.h"
#include "../utf8.h"

#define PS_GPT2_MODEL_124M  124
#define PS_GPT2_MODEL_355M  355
#define PS_GPT2_MODEL_774M  774
#define PS_GPT2_MODEL_1558M 1558

#define GPT2_BLOCK_SIZE     7
#define GPT2_TOT_LAYERS     89

#define MAX_OUTPUT_TOKENS 40
#define DEFAULT_TEMPERATURE 0

#define UNUSED(V) ((void) V)

/* Globals */
int PSAvailableGPT2Models[] = {124, 355, 774, 1558};
char *python_command = "python";

int default_loglevel = PSLOGLEVEL_INFO;
int model_size = PS_GPT2_MODEL_124M;
int overwrite_downloads = 0;
int overwrite_extracted_files = 0;
char *custom_model_dir = NULL;
int verbose = 0;
int interactive = 0;
int use_binary_files = 1;
int max_output_tokens = MAX_OUTPUT_TOKENS;
PSFloat temperature = DEFAULT_TEMPERATURE;

typedef struct {
    int n_vocab;
    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
} GPT2HyperParameters;

static void onDictStringRelease(PSDictItem *item) {
    free(item->value.as_ptr);
}

/**** Encoding/Decoding ****/

#define B2U_SIZE 256
#define U2B_SIZE 324
uint32_t BytesToUnicode[B2U_SIZE] = {0};
uint32_t UnicodeToBytes[U2B_SIZE] = {0};

static int printProgressBar(PSNeuralNetwork *model) {
    if (PSLogLevel < PSLOGLEVEL_NOTICE) return 0;
    return PSProgressBar(model->size, GPT2_TOT_LAYERS, PS_PROGRESS_STYLE_LINE,
                         1, PS_PROGRESS_FLAG_XTERM256_CODE, 0, "Layer");
}

static void initBytesToUnicode(void) {
    static int initalized = 0;
    static uint32_t code_ranges[3][2] = {
        {33, 126}, {161, 172}, {174, 255}
    };
    if (initalized) return;
    int i, j, n = 0;
    for (i = 0; i < 3; i++) {
        uint32_t *range = code_ranges[i];
        for (j = range[0]; (uint32_t) j <= range[1]; j++)
            BytesToUnicode[j] = j;
    }
    for (i = 0; i < B2U_SIZE; i++)
        if (!BytesToUnicode[i]) BytesToUnicode[i] = B2U_SIZE + n++;
    for (i = 0; i < B2U_SIZE; i++)
        UnicodeToBytes[BytesToUnicode[i]] = i;
    initalized = 1;
}

static int isNewToken(char *str, int *len) {
    static const char *smatches[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
    };
    assert(len != NULL);
    if (str == NULL) return 0;
    int is_new = 0;
    *len = 0;
    size_t smatch_len = sizeof(smatches) / sizeof(char *), i;
    for (i = 0; i < smatch_len; i++) {
        is_new = (strstr(str, smatches[i]) == str);
        if (is_new) {
            *len = strlen(smatches[i]);
            break;
        }
    }
    char *p = str, *last_match = NULL;
    char c;
    int space_count = 0, match_count = 0, match_size = 0,
        type = '\0', prev_type = type, head_type = type;
    while ((c = *p)) {
        int is_first = (p - str) == 0;
        int clen = PSUTF8CharSize(p);
        PSUTF8Char uc = *((PSUTF8Char *) p);
        if (PSUTF8IsAlpha(uc)) type = 'L';
        else if (PSUTF8IsDigit(uc)) type = 'N';
        else if (isspace(c) || PSUTF8IsSpace(uc)) type = 's';
        else type = '\0';
        int matched = 0, is_real_space = 0, is_space = 0;
        if (type == 's') {
            is_space = 1;
            space_count++;
            if ((is_real_space = (c == ' '))) type = ' ';
            if (is_first) {
                if (is_real_space) matched = 1;
                else {
                    char next_c = *(p + 1);
                    matched = (next_c && isspace(next_c));
                }
            } else matched = (
                (prev_type == 's' || prev_type == ' ') &&
                (head_type == 's' || head_type == ' ')
            );
        } else if (type == 'L' || type == 'N' || type == '\0') {
            if (is_first) matched = 1;
            else {
                matched = (
                    type == prev_type ||
                    (match_count == 1 && head_type == ' ' && prev_type == ' ')
                );
                if (!matched && space_count == match_count && last_match &&
                   prev_type == ' ')
                {
                    match_count--;
                    int prev_clen = PSUTF8CharSize(last_match);
                    match_size -= prev_clen;
                    p -= prev_clen;
                }
            }
        }
        if (!matched) break;
        last_match = p;
        match_count++;
        match_size += clen;
        if (is_first) {
            is_new = !is_real_space;
            head_type = type;
        } else is_new = 1;
next:
        prev_type = type;
        p += clen;
    }
    if (is_new) *len = match_size;
    return is_new;
}

char **getTokens(char *str, int *count) {
    assert(count != NULL);
    char **tokens = NULL;
    *count = 0;
    if (str == NULL) return NULL;
    char *p = str;
    int ok = 1, n_tokens = 0, i;
    while (*p) {
        char *tok = NULL;
        int len = 0;
        if (isNewToken(p, &len) && len > 0) {
            tok = malloc(len + 1);
            ok = tok != NULL;
            if (!ok) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            memcpy(tok, p, len);
            tok[len] = '\0';
            char **new_tokens = realloc(tokens, ++n_tokens * sizeof(char *));
            ok = new_tokens != NULL;
            if (!ok) {
                PSPrintMemoryErrorMsg();
                goto final;
            }
            tokens = new_tokens;
            tokens[n_tokens - 1] = tok;
            *count = n_tokens;
            p += len;
            continue;
        }
        p++;
    }
final:
    if (!ok) {
        if (tokens != NULL) {
            for (i = 0; i < *count; i++) free(tokens[i]);
            free(tokens);
        }
        tokens = NULL;
        *count = 0;
    }
    return tokens;
}

char *encodeToken(char *token) {
    if (token == NULL) return NULL;
    char *p = token, *new = NULL;
    uint8_t c;
    int newlen = 0;
    while ((c = *p)) {
        uint32_t cp = BytesToUnicode[c];
        int clen = PSUTF8CodepointSize(cp), preceding_len = 0;
        /* Skip character until it does not require encoding. If the original
         * token doesn't contain characteres that must be encoded, the original
         * token will be returned untouched. */
        if (cp == (uint32_t) c && new == NULL && clen < 2) goto next;
        size_t sz = 0;
        if (new == NULL) {
            preceding_len = (p - token);
            sz = 1 + preceding_len + clen;
        } else {
            newlen = strlen(new);
            sz = 1 + newlen + clen;
        }
        char *nt = realloc(new, sz);
        if (nt == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        new = nt;
        PSUTF8Char new_c = PSUTF8Encode(cp);
        if (new_c == 0) {
            PSErr(NULL, "invalid codepoint %u", cp);
            goto fail;
        }
        if (preceding_len > 0) {
            memcpy(new, token, preceding_len);
            memcpy(new + preceding_len, &new_c, clen);
        } else memcpy(new + newlen, &new_c, clen);
        new[sz - 1] = '\0';
next:
        p++;
    }
    if (verbose) {
        printf("Token: '%s' (encoded from '%s')\n",
               (new != NULL ? new : token), token);
    }
    if (new != NULL) token = new;
    return token;
fail:
    PSErr(NULL, "could not encode token '%s'\n", token);
    if (new != NULL) free(new);
    return NULL;
}

char *decodeToken(char *token) {
    if (token == NULL) return NULL;
    char *p = token, *new = NULL;
    uint8_t c;
    int newlen = 0;
    while ((c = *p)) {
        PSUTF8Char uc = 0;
        int clen = PSUTF8CharSize(p);
        if (clen < 2 && new == NULL) goto next;
        memcpy(&uc, p, clen);
        uint32_t cp = PSUTF8Decode(uc);
        if (cp >= U2B_SIZE) {
            PSErr(NULL, "invalid token '%s': codepoint for character at index "
                  "%d is out-of-range (codepoint %u)", token, (int)(p - token),
                  cp);
            goto fail;
        }
        c = (char) UnicodeToBytes[cp];
        size_t sz = 0, preceding_len = 0;
        if (new == NULL) {
            preceding_len = (p - token);
            sz = 1 + preceding_len + 1;
        } else {
            newlen = strlen(new);
            sz = 1 + newlen + 1;
        }
        char *nt = realloc(new, sz);
        if (nt == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        new = nt;
        if (preceding_len > 0) memcpy(new, token, preceding_len);
        new[newlen] = c;
        new[sz - 1] = '\0';
next:
        p++;
    }
    if (verbose) {
        printf("Token: '%s' (decoded from '%s')\n",
               (new != NULL ? new : token), token);
    }
    if (new != NULL) token = new;
    return token;
fail:
    PSErr(NULL, "could not encode token '%s'\n", token);
    if (new != NULL) free(new);
    return NULL;
}

char **getTokenNgrams(char *token, int *count) {
    assert(count != NULL);
    if (token == NULL) return NULL;
    char **ngrams = NULL;
    int len = PSUTF8StrLen(token), i = 0;
    if (len == 0) return NULL;
    ngrams = calloc(len, sizeof(char *));
    if (ngrams == NULL) {
        PSPrintMemoryErrorMsg();
        goto fail;
    }
    char *p = token;
    *count = 0;
    while (*p) {
        if (i >= len) break;
        int clen = PSUTF8CharSize(p);
        char *uc = malloc(clen + 1);
        if (uc == NULL) {
            PSPrintMemoryErrorMsg();
            goto fail;
        }
        memcpy(uc, p, clen);
        uc[clen] = '\0';
        ngrams[i++] = uc;
        p += clen;
    }
    *count = i;
    return ngrams;
fail:
    *count = 0;
    if (ngrams) {
        for (i = 0; i < len; i++) free(ngrams[i]);
        free(ngrams);
    }
    PSErr(NULL, "failed to get ngrams for token '%s'", token);
    return NULL;
}

PSDict *getPairs(char **ngrams, int ncount) {
    if (ngrams == NULL) return NULL;
    errno = 0;
    PSDict *pairs = NULL;
    int n_pairs = ncount - 1, ok = 1, i;
    if (n_pairs <= 0) goto final;
    pairs = PSDictCreate(PSDICT_UPDATE_DISABLED);
    ok = pairs != NULL;
    if (!ok) {
        PSErr(__func__, "failed to allocate pairs");
        goto final;
    }
    pairs->on_item_release = onDictStringRelease;
    for (i = 0; i < n_pairs; i++) {
        char *first = ngrams[i], *second = ngrams[i + 1];
        ok = (first != NULL && second != NULL);
        if (!ok) {
            PSErr(__func__, "some ngram is null");
            goto final;
        }
        char **pair = malloc(2 * sizeof(char *));
        ok = (pair != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        pair[0] = first;
        pair[1] = second;
        char *key = PSStringJoin(pair, " ", 2);
        ok = (key != NULL);
        if (!ok) {
            free(pair);
            goto final;
        }
        ok = (PSDictSet(pairs, key, PSDictFromPointer(pair)) != NULL);
        if (!ok) {
            PSErr(__func__, "could not set pair '%s'", key);
            free(key);
            free(pair);
            goto final;
        }
    }
final:
    if (!ok) {
        PSDictRelease(pairs);
        pairs = NULL;
        if (errno == 0) errno = EINVAL;
    }
    return pairs;
}

char **getBytePairEncodings(PSDict *bpe, char *token, int *count) {
    assert(count != NULL);
    if (token == NULL) return NULL;
    int enc_count = 0;
    char **encodings = getTokenNgrams(token, &enc_count);
    int ok = (encodings != NULL && enc_count > 0);
    if (!ok) {
        PSErr(__func__, "could not get ngrams for '%s'", token);
        goto final;
    }
    char *last_pair_s = NULL;
    while (1) {
        char *pair_s = NULL;
        PSDict *pairs = getPairs(encodings, enc_count);
        if (pairs == NULL) {
            ok = (errno == 0);
            if (!ok) {
                PSErr(__func__, "could not get pairs");
                goto final;
            }
            break;
        }
        if (pairs->length <= 0) goto final;
        PSDictIterator *pair_iter = PSDictIteratorCreate(pairs);
        ok = (pair_iter != NULL);
        if (!ok) goto final;
        /* Get the pair with lower ranking */
        uint64_t min_rank = INT32_MAX;
        int first_idx = -1, bpe_found = 0, i;
        char **pair = NULL;
        PSDictItem *pair_item = NULL, *selected_item = NULL;
        while ((pair_item = PSDictNext(pair_iter))) {
            PSDictItem *bpeitem = PSDictGet(bpe, pair_item->key);
            char **item_pair = pair_item->value.as_ptr;
            assert(item_pair != NULL);
            assert(item_pair[0] != NULL && item_pair[1] != NULL);
            int rank = INT32_MAX, found = (bpeitem != NULL);
            if (found) rank = bpeitem->value.as_int;
            if (pair == NULL || (uint64_t) rank < min_rank) {
                pair = item_pair;
                bpe_found = found;
                min_rank = rank;
                selected_item = pair_item;
            }
        }
        if (!bpe_found || pair == NULL) break;
        pair_s = PSStringJoin(pair, NULL, 2);
        ok = (pair_s != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        char **new = calloc(enc_count, sizeof(char *));
        ok = (new != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            free(pair_s);
            goto final;
        }
        char *first = pair[0], *second = pair[1];
        char **enc_p = encodings, **new_p = new;
        int pair_used = 0;
        for (i = 0; i < enc_count; i++) {
            if (strcmp(first, enc_p[i]) == 0) {
                if (first_idx < 0) first_idx = i;
                if (i < (enc_count - 1) && strcmp(second, enc_p[i + 1]) == 0) {
                    pair_used = 1;
                    *(new_p++) = pair_s;
                    if (enc_p[i] == last_pair_s) enc_p[i] = NULL;
                    i += 1;
                    continue;
                }
            }
            *(new_p++) = enc_p[i];
            enc_p[i] = NULL;
        }
        int newlen =  (int) (new_p - new);
        if (newlen == 0) {
            free(new);
            break;
        }
        for (i = 0; i < enc_count; i++) free(encodings[i]);
        free(encodings);
        encodings = new;
        enc_count = newlen;
next:
        free(pair_iter);
        if (!pair_used) free(pair_s);
        else last_pair_s = pair_s;
        PSDictRelease(pairs);
    }
    *count = enc_count;
final:
    if (!ok) {
        if (encodings != NULL) {
            int i;
            for (i = 0; i < enc_count; i++) free(encodings[i]);
            free(encodings);
            encodings = NULL;
        }
        *count = 0;
    }
    return encodings;
}

PSFloat *encodePrompt(char *prompt, PSVocabulary *vocabulary, PSDict *bpe) {
    if (prompt == NULL) return NULL;
    assert(vocabulary != NULL);
    assert(bpe != NULL);
    PSFloat *inputs = NULL;
    initBytesToUnicode();
    int token_count = 0, i;
    char **tokens = getTokens(prompt, &token_count);
    int ok = tokens != NULL && token_count > 0;
    if (!ok) {
        PSErr(NULL, "failed to tokenize prompt");
        goto final;
    }
    int n_inputs = 0, input_size = 0;
    for (i = 0; i < token_count; i++) {
        char *tok = tokens[i];
        char *enc = encodeToken(tok);
        if (enc != tok) {
            free(tok);
            tokens[i] = enc;
        }
        int bpe_count = 0, j;
        char **bpe_encodings = getBytePairEncodings(bpe, enc, &bpe_count);
        ok = (bpe_encodings != NULL);
        if (!ok) {
            PSErr(NULL, "could not get byte-pair-encodings for '%s'", enc);
            goto final;
        }
        if (bpe_count <= 0) continue;
        n_inputs += bpe_count;
        if (input_size == 0) input_size++;
        int index = input_size;
        input_size += bpe_count;
        PSFloat *new = realloc(inputs, input_size * sizeof(PSFloat));
        ok = (new != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        inputs = new;
        inputs[0] = (PSFloat) n_inputs;
        PSFloat *dest = inputs + index;
        for (j = 0; j < bpe_count; j++) {
            char *bpe_token = bpe_encodings[j];
            int id = PSVocabularyGetTokenID(vocabulary, bpe_token);
            ok = (id != PS_TOKEN_NOT_FOUND);
            if (!ok) {
                PSErr(__func__, "could not find encoded ID for token '%s'",
                      bpe_token);
                goto final;
            }
            dest[j] = (PSFloat) id;
        }
    }
final:
    if (!ok) {
        free(inputs);
        inputs = NULL;
    }
    if (tokens != NULL) {
        for (i = 0; i < token_count; i++) free(tokens[i]);
        free(tokens);
    }
    return inputs;
}

/**** Utils ****/

static char *getExecutablePath(char *executable) {
    static char path[PATH_MAX + 1] = {0};
    char _realpath[PATH_MAX + 1];
    if (path[0]) return path;
    _realpath[0] = 0;
    if (realpath(executable, _realpath) != NULL) {
        char *dir = dirname(_realpath);
        if (dir == NULL) return NULL;
        int len = strlen((const char*) dir);
        if (len >= PATH_MAX) {
            fprintf(stderr, "WARN: getPsycPath(): dirname length > %d",
                    PATH_MAX);
            return NULL;
        }
        memcpy(path, dir, len);
        path[len] = 0;
        return path;
    }
    return NULL;
}

static void binaryFilePath(char *destpath, char *fpath) {
    strncpy(destpath, fpath, PATH_MAX);
    strncat(destpath, ".bin", PATH_MAX);
}

static int getBinaryParamsPathIfExists(char *params_path) {
    char binpath[PATH_MAX] = {0};
    binaryFilePath(binpath, params_path);
    int exists = PSFileExists(binpath);
    if (exists) strncpy(params_path, binpath, PATH_MAX);
    return exists;
}

static int saveBinaryParamsFor(PSLayer *layer, char *params_path) {
    char binpath[PATH_MAX] = {0};
    PSLogLevel = default_loglevel;
    if (layer == NULL) return 0;
    binaryFilePath(binpath, params_path);
    return PSSaveLayer(layer, binpath, PS_IO_BINARY_MODE);
}

int isPython3Installed(void) {
    char cmd[PATH_MAX] = {0};
    char tmpfile[PATH_MAX] = {0};
    sprintf(tmpfile, "/tmp/pyvers-%ld", time(NULL));
    sprintf(cmd, "python3 --version > %s 2>/dev/null", tmpfile);
    int status = system(cmd);
    if (status == 0) {
        python_command = "python3";
        return 3;
    }
    sprintf(tmpfile, "/tmp/pyvers-%ld", time(NULL) + 1);
    sprintf(cmd, "python --version > %s 2>/dev/null", tmpfile);
    status = system(cmd);
    if (status != 0) return 0;
    FILE *f = fopen(tmpfile, "r");
    if (f == NULL) return 0;
    int installed = 0;
    int vers_maj = 0, vers_min = 0;
    int matched = fscanf(f, "Python %d.%d", &vers_maj, &vers_min);
    if (matched == 0) goto final;
    installed = (vers_maj >= 3);
    if (!installed) goto final;
    if (vers_min < 6) {
        PSErr(NULL, "Python3 has been found but its version is < 3.6, "
                    "please install python3 >= 3.6");
        installed = 0;
    }
final:
    fclose(f);
    return installed;
}

int isPythonModuleInstalled(char *module) {
    if (python_command == NULL) python_command = "python";
    char cmd[PATH_MAX] = {0};
    sprintf(cmd, "%s -c 'import %s' > /dev/null 2>&1", python_command, module);
    /*printf("%s\n", cmd);*/
    int status = system(cmd);
    return status == 0;
}

int PSDownloadGPT2Files(int model_size, const char *model_dir, int overwrite) {
    static const char *gpt2_url =
        "https://openaipublic.blob.core.windows.net/gpt-2/models";
    static const char *files[] = {
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    };
    if (model_dir == NULL) {
        PSErr(__func__, "mandatory arg. `model_dir` is null");
        return 0;
    }
    int found_model = 0;
    size_t models_count = sizeof(PSAvailableGPT2Models) / sizeof(int), i;
    for (i = 0; i < models_count; i++) {
        int msize = PSAvailableGPT2Models[i];
        if (model_size == msize) {
            found_model = 1;
            break;
        }
    }
    if (!found_model) {
        PSErr(__func__, "invalid model_size %d", model_size);
        return 0;
    }
    size_t files_count =  sizeof(files) / sizeof(char *);
    int do_log = PSLogLevel >= PSLOGLEVEL_INFO;
    char url[PATH_MAX] = {0};
    for (i = 0; i < files_count; i++) {
        const char *file = files[i];
        char *local_path = PSPathJoin(2, model_dir, file);
        if (local_path == NULL) {
            PSErr(__func__, "local path for file %s is not valid", file);
            return 0;
        }
        int exists = PSFileExists(local_path);
        free(local_path);
        if (!overwrite && exists) continue;
        if (do_log) printf("Downloading GPT2 file: %s\n", file);
        int len = snprintf(
            url, PATH_MAX, "%s/%dM/%s", gpt2_url, model_size, file
        );
        if (len >= PATH_MAX) return 0;
        int ok = PSDownloadFile(url, model_dir);
        if (!ok) {
            PSErr(NULL, "failed to download GPT2 file: %s", file);
            return 0;
        }
    }
    return 1;
}

int PSLoadGPT2HyperParameters(char *fpath, GPT2HyperParameters *hparams) {
    if (hparams == NULL || fpath == NULL) return 0;
    FILE *json = fopen(fpath, "r");
    int success = (json != NULL);
    if (!success) {
        PSErr(__func__, "could not open '%s' for reading", fpath);
        goto final;
    }
    char buf[1024];
    buf[0] = '\0';
    char *line = NULL;
    while ((line = fgets(buf, 1024, json))) {
        char *p = strchr(line, '"');
        if (p == NULL) continue;
        char prop[50];
        prop[0] = '\0';
        int val = -1;
        int matched = sscanf(p, "\"%49[a-z_]\"", prop);
        if (matched == 0) goto invalid_line;
        while (*p && !isdigit(*p)) p++;
        if (*p == '\0') goto invalid_line;
        matched = sscanf(p, "%d", &val);
        if (matched == 0) continue;
        if (val < 0) goto invalid_line;
        int *par_p = NULL;
        if (verbose) printf("%s '%s' = %d\n", __func__, prop, val);
        if (strcmp("n_vocab", prop) == 0) par_p = &hparams->n_vocab;
        else if (strcmp("n_ctx", prop) == 0) par_p = &hparams->n_ctx;
        else if (strcmp("n_embd", prop) == 0) par_p = &hparams->n_embd;
        else if (strcmp("n_head", prop) == 0) par_p = &hparams->n_head;
        else if (strcmp("n_layer", prop) == 0) par_p = &hparams->n_layer;
        else continue;
        *par_p = val;
        continue;
invalid_line:
        PSErr(__func__, "invalid line:\n%s", line);
        success = 0;
        goto final;
    }
final:
    if (json != NULL) fclose(json);
    if (!success) PSErr(__func__, "could not parse '%s'", fpath);
    return success;
}

static int validGPT2HyperParams(GPT2HyperParameters *hpar) {
    if (hpar == NULL) return 0;
    if (hpar->n_vocab <= 0) {
        PSErr(NULL, "invalid GPT2 hyper-parameter n_vocab: %d\n",hpar->n_vocab);
        return 0;
    }
    if (hpar->n_embd <= 0) {
        PSErr(NULL, "invalid GPT2 hyper-parameter n_embd: %d\n",hpar->n_embd);
        return 0;
    }
    if (hpar->n_ctx <= 0) {
        PSErr(NULL, "invalid GPT2 hyper-parameter n_ctx: %d\n", hpar->n_ctx);
        return 0;
    }
    if (hpar->n_head <= 0) {
        PSErr(NULL, "invalid GPT2 hyper-parameter n_head: %d\n", hpar->n_head);
        return 0;
    }
    if (hpar->n_layer <= 0) {
        PSErr(NULL, "invalid GPT2 hyper-parameter n_layer: %d\n",hpar->n_layer);
        return 0;
    }
    return 1;
}

static void printdGPT2HyperParams(GPT2HyperParameters *hpar) {
    if (hpar == NULL) return;
    char *color = "", *reset = "";
    if (PSLogColorEnabled()) {
        color = PSCOLOR_CYAN;
        reset = PSCOLOR_RESET;
    }
    printf("%sn_vocab%s: %d\n", color, reset, hpar->n_vocab);
    printf("%sn_embd%s:  %d\n", color, reset, hpar->n_embd);
    printf("%sn_ctx%s:   %d\n", color, reset, hpar->n_ctx);
    printf("%sn_head%s:  %d\n", color, reset, hpar->n_head);
    printf("%sn_layer%s: %d\n", color, reset, hpar->n_layer);
}

static int hasGPT2RequiredFiles(char *model_dir, GPT2HyperParameters *hpar) {
    static char *required_files[] = {
        "extracted/wte", "extracted/wpe",
        "extracted/wte.T",
        "extracted/processed/ln_f",
        "psyc_encoder.txt",
        "vocab.bpe"
    };
    static char *block_file_patterns[] = {
        "h%d.attn", "h%d.ln_1", "h%d.ln_2", "h%d.mlp.c_fc", "h%d.mlp.c_proj"
    };
    char *fpath = NULL;
    int ok = 1, nfiles = (int) (sizeof(required_files) / sizeof(char *)),
        npatterns = (int) (sizeof(block_file_patterns) / sizeof(char *)), i, j;
    for (i = 0; i < nfiles; i++) {
        free(fpath);
        fpath = PSPathJoin(2, model_dir, required_files[i]);
        ok = PSFileExists(fpath);
        if (verbose) {
            printf("Checking file '%s': %s%s\n" PSCOLOR_RESET,
                    required_files[i], (ok ? PSCOLOR_GREEN : PSCOLOR_YELLOW),
                    (ok ? "found" : "not found"));
        }
        if (!ok) break;
    }
    if (!ok) goto final;
    for (i = 0; i < hpar->n_layer; i++) {
        char fname[PATH_MAX];
        for (j = 0; j < npatterns; j++) {
            snprintf(fname, PATH_MAX, block_file_patterns[j], i);
            free(fpath);
            fpath = PSPathJoin(3, model_dir, "extracted/processed/", fname);
            ok = fpath != NULL;
            if (!ok) goto final;
            ok = PSFileExists(fpath);
            if (verbose) {
                printf("Checking file '%s': %s%s\n" PSCOLOR_RESET,
                        fname, (ok ? PSCOLOR_GREEN : PSCOLOR_YELLOW),
                        (ok ? "found" : "not found"));
            }
            if (!ok) goto final;
        }
    }
final:
    free(fpath);
    return ok;
}

/*
    ==== GPT2 model files spec: ====

    - wte: Embedding layer
    - wpe: Positional encoding layer

    - Transformer block:
        h*.ln_1.b:        Normalization Layer 1 biases
        h*.ln_1.g:        Normalization Layer 1 weights
        h*.attn.c_attn.b: Attention layer input proj. biases (Q,K,V)
        h*.attn.c_attn.w: Attention layer input proj. weights (Q,K,V)
        h*.attn.c_proj.b: Attention layer output proj. biases
        h*.attn.c_proj.w: Attention layer output proj. weights
        h*.ln_2.b:        Normalization Layer 2 biases
        h*.ln_2.g:        Normalization Layer 2 weights
        h*.mlp.c_fc.b     Fully-Connected GeLU Layer biases
        h*.mlp.c_fc.w     Fully-Connected GeLU Layer weights
        h*.mlp.c_proj.b   Linear layer biases
        h*.mlp.c_proj.w   Linear layer weights

    - ln_f.b: Normalization layer biases
    - ln_f.g: Normalization layer weights

    ==== PsyC Model ====

    - [0]: FullyConnected input layer, size = vocab_size, onehot
    - [1]: Embedding layer, size = n_embed

    N Transformers blocks, for each block (bidx = block index):

    - [bidx * block_size + 0]: Normalization Layer 1
    - [bidx * block_size + 1]: MultiHead Causal Self-Attention Layer
    - [bidx * block_size + 2]: OperatorLayer(Add):
                               [bidx * block_size - 1] + (inputs)
                               [bidx * block_size + 1]   (attn)
    - [bidx * block_size + 3]: Normalization Layer 2
    - [bidx * block_size + 4]: Fully-Connected GeLU Layer, size = n_embed * 4
    - [bidx * block_size + 5]: Linear Layer, size = n_embed
    - [bidx * block_size + 6]: OperatorLayer(Add):
                               [bidx * block_size + 5] + (linear mlp)
                               [bidx * block_size + 2] + (add1)

    - [3 + N * block_size + 0] Normalization Layer
    - [3 + N * block_size + 1] Transposed Embedding (Output)


*/

int PSAddGPT2TransformerBlock(PSNeuralNetwork *model, int index,
                              GPT2HyperParameters *hpar, char *dir)
{
    int ok = 1, has_bin = 0;
    int print_progress = (!verbose && PSLogLevel >= PSLOGLEVEL_NOTICE);
    /*int base_index = 3 + (block_size * index);*/
    PSLayer *base_layer = NULL, *attn_layer = NULL, *l = NULL,
            *add_layer_1 = NULL, *ln_layer =  NULL;
    char fname[255];

    /* Normalization Layer 1 */
    snprintf(fname, 255, "h%d.ln_1", index);
    char *params_path = PSPathJoin(3, dir, "extracted/processed", fname);
    ok = (params_path != NULL);
    if (!ok) goto final;
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(model, Normalization, hpar->n_embd, PSLDEF(
        .flags = FLAG_USE_SEQUENCES, .load_from = params_path
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    print_progress = printProgressBar(model);
    int base_index = l->index;
    base_layer = l;

    /* Multi-Head Attention Layer */
    snprintf(fname, 255, "h%d.attn", index);
    free(params_path);
    params_path = PSPathJoin(3, dir, "extracted/processed", fname);
    ok = (params_path != NULL);
    if (!ok) goto final;
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(model, Attention, hpar->n_embd, PSLDEF(
        .flags = FLAG_USE_SEQUENCES,
        .attention_heads = hpar->n_head,
        .causal_attention = 1,
        .self_attention = 1,
        .query_provider = base_layer,
        .keys_provider = base_layer,
        .values_provider = base_layer,
        .load_from = params_path,
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    print_progress = printProgressBar(model);
    attn_layer = l;

    /* "Add" Layer 1 */
    PSLayer *input_layer = model->layers[base_index - 1];
    PSLayer *providers[] = {input_layer, attn_layer};
    l = PSAddLayer(model, OperatorLayer, 0, PSLDEF(
        .flags = FLAG_USE_SEQUENCES,
        .operator = PSAddOperator,
        .providers_count = 2,
        .providers = providers
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    print_progress = printProgressBar(model);
    add_layer_1 = l;

    /* Normalization Layer 2 */
    snprintf(fname, 255, "h%d.ln_2", index);
    free(params_path);
    params_path = PSPathJoin(3, dir, "extracted/processed", fname);
    ok = (params_path != NULL);
    if (!ok) goto final;
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(model, Normalization, hpar->n_embd, PSLDEF(
        .flags = FLAG_USE_SEQUENCES, .load_from = params_path,
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    print_progress = printProgressBar(model);

    /* Fully-Connected GeLU Layer (size: n_embd * 4) */
    snprintf(fname, 255, "h%d.mlp.c_fc", index);
    free(params_path);
    params_path = PSPathJoin(3, dir, "extracted/processed", fname);
    ok = (params_path != NULL);
    if (!ok) goto final;
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(model, FullyConnected, 4 * hpar->n_embd, PSLDEF(
        .flags = FLAG_USE_SEQUENCES,
        .activation = PSGelu,
        .load_from = params_path,
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    print_progress = printProgressBar(model);

    /* Projection Linear Layer */
    snprintf(fname, 255, "h%d.mlp.c_proj", index);
    free(params_path);
    params_path = PSPathJoin(3, dir, "extracted/processed", fname);
    ok = (params_path != NULL);
    if (!ok) goto final;
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(model, Linear, hpar->n_embd, PSLDEF(
        .flags = FLAG_USE_SEQUENCES, .load_from = params_path,
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    print_progress = printProgressBar(model);
    ln_layer = l;

    /* Add Layer 2*/
    providers[0] = add_layer_1;
    providers[1] = ln_layer;
    l = PSAddLayer(model, OperatorLayer, 0, PSLDEF(
        .flags = FLAG_USE_SEQUENCES,
        .operator = PSAddOperator,
        .providers_count = 2,
        .providers = providers
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    print_progress = printProgressBar(model);
final:
    free(params_path);
    return ok;
}

int downloadAndExtractGPT2Model(int model_size, char *model_dir,
                                GPT2HyperParameters *hpar, char *exec_path)
{
    int ok = 1, hpar_loaded = 1, i;
    char *hparams_json = PSPathJoin(2, model_dir, "hparams.json"),
         *extract_script = NULL, *extract_cfg = NULL, *process_script = NULL;
    if (hparams_json == NULL) return 0;
    int forced = overwrite_downloads || overwrite_extracted_files;
    if (!forced && (PSFileExists(model_dir) && PSFileExists(hparams_json))) {
        hpar_loaded = PSLoadGPT2HyperParameters(hparams_json, hpar);
        ok = hpar_loaded;
        if (!ok) goto final;
        ok = validGPT2HyperParams(hpar);
        if (!ok) {
            PSErr(NULL, "invalid GPT2 hyper-parameters");
            goto final;
        }
        if (hasGPT2RequiredFiles(model_dir, hpar)) {
            PSNotice("Found GPT2 model in '%s'", model_dir);
            return 1;
        }
    }
    if (!isPython3Installed()) {
        PSWarn("Python3 is required but was not found on your system\n"
               "      Please follow instructions from https://www.python.org/\n"
               "      in order to install it");
        goto final;
    }
    char *required_modules[] = {"tensorflow", "numpy"};
    int modules_count = (int) (sizeof(required_modules) / sizeof(char *));
    for (i = 0; i < modules_count; i++) {
        ok = isPythonModuleInstalled(required_modules[i]);
        if (!ok) {
            PSWarn("%s is required but was not found on your system\n"
                   "      execute `pip3 install %s` in order to "
                   "install it", required_modules[i], required_modules[i]);
            goto final;
        }
    }
    ok = PSDownloadGPT2Files(model_size, model_dir, overwrite_downloads);
    if (!ok) goto final;
    extract_script = PSPathJoin(2, exec_path, "../utils/extract_tf_model.py");
    extract_cfg = PSPathJoin(2, exec_path, "../resources/gpt_model.conf.json");
    process_script = PSPathJoin(
        2, exec_path,"../utils/process_gpt_extracted_files.py"
    );
    ok = (extract_script && extract_cfg && process_script);
    if (!ok) goto final;
    ok = PSFileExists(extract_script);
    if (!ok) {
        PSErr(NULL, "could not find tf extraction script: '%s'",extract_script);
        goto final;
   }
    ok = PSFileExists(extract_cfg);
    if (!ok) {
        PSErr(NULL, "could not find tf extraction config: '%s'", extract_cfg);
        goto final;
    }
    ok = PSFileExists(process_script);
    if (!ok) {
        PSErr(NULL, "could not find tf processing script: '%s'",process_script);
        goto final;
    }
    char cmd[PATH_MAX] = {0};
    snprintf(cmd, PATH_MAX, "%s %s--omit-shape -c '%s' '%s'",
            extract_script, (overwrite_extracted_files ? "-f " : ""),
            extract_cfg, model_dir);
    PSInfo("Extracting tensorflow GPT2 model");
    if (verbose) {
        printf(PSCOLOR_DIM "%s\n" PSCOLOR_RESET, cmd);
        fflush(stdout);
    }
    int status = system(cmd);
    ok = (status == 0);
    if (!ok) {
        fprintf(stderr, PSCOLOR_YELLOW "Command:%s\n" PSCOLOR_RESET, cmd);
        PSErr(NULL, "some error occurred while executing extraction script");
        goto final;
    }
    snprintf(cmd, PATH_MAX, "%s%s '%s'", process_script,
             (overwrite_extracted_files ? " -f" : ""), model_dir);
    if (verbose) {
        printf(PSCOLOR_DIM "%s\n" PSCOLOR_RESET, cmd);
        fflush(stdout);
    }
    status = system(cmd);
    ok = (status == 0);
    if (!ok) {
        fprintf(stderr, PSCOLOR_YELLOW "Command:%s\n" PSCOLOR_RESET, cmd);
        PSErr(NULL, "some error occurred while executing post-processing "
              "script");
        goto final;
    }
    if (!hpar_loaded) {
        ok = PSLoadGPT2HyperParameters(hparams_json, hpar);
        if (!ok) goto final;
        ok = validGPT2HyperParams(hpar);
        if (!ok) {
            PSErr(NULL, "invalid GPT2 hyper-parameters");
            goto final;
        }
    }
final:
    free(hparams_json);
    free(extract_script);
    free(extract_cfg);
    free(process_script);
    return ok;
}

static PSVocabulary *loadVocabulary(char *model_dir, uint64_t vocab_size) {
    char *encoder_path = PSPathJoin(2, model_dir, "psyc_encoder.txt");
    if (encoder_path == NULL) return NULL;
    PSVocabulary *vocabulary = NULL;
    FILE *f = NULL;
    int ok = PSFileExists(encoder_path);
    if (!ok) {
        PSErr(NULL, "could not find '%d'", encoder_path);
        goto final;
    }
    f = fopen(encoder_path, "r");
    ok = (f != NULL);
    if (!ok) {
        PSErr(NULL, "could not open '%s'", encoder_path);
        goto final;
    }
    vocabulary = PSVocabularyCreate(vocab_size);
    ok = (vocabulary != NULL);
    if (!ok) goto final;
    int lineno = 0;
    char token[1024];
    while (!feof(f)) {
        uint64_t index = 0;
        int matched = fscanf(f, "%llu: ", &index);
        if (!matched) {
            ok = 0;
            PSErr(NULL, "invalid index at line %d in file '%s'",
                  lineno, encoder_path);
            goto final;
        }
        ok = fgets(token, 1024, f) != NULL;
        if (!ok) {
            PSErr(NULL, "invalid token at line %d in file '%s'",
                  lineno, encoder_path);
            goto final;
        }
        int len = strlen(token);
        ok = (len > 0);
        if (!ok) {
            PSErr(NULL, "invalid empty token at line %d in file '%s'",
                  lineno, encoder_path);
            goto final;
        }
        if (token[len - 1] == '\n') token[len - 1] = '\0';
        int64_t id = PSVocabularyAdd(vocabulary, token);
        ok = (id != PS_INVALID_TOKEN_ID);
        if (!ok) {
            PSErr(NULL, "could not add token '%s' to vocabulary", token);
        }
        lineno++;
    }
final:
    if (f != NULL) fclose(f);
    if (!ok) {
        PSVocabularyRelease(vocabulary);
        vocabulary = NULL;
    }
    free(encoder_path);
    return vocabulary;
}

PSDict *loadBPE(char *model_dir) {
    char *bpe_path = PSPathJoin(2, model_dir, "vocab.bpe");
    if (bpe_path == NULL) return NULL;
    PSDict *bpe = NULL;
    FILE *f = NULL;
    int ok = PSFileExists(bpe_path);
    if (!ok) {
        PSErr(NULL, "could not find '%d'", bpe_path);
        goto final;
    }
    f = fopen(bpe_path, "r");
    ok = (f != NULL);
    if (!ok) {
        PSErr(NULL, "could not open '%s'", bpe_path);
        goto final;
    }
    bpe = PSDictCreate(PSDICT_UPDATE_DISABLED);
    ok = (bpe != NULL);
    if (!ok) goto final;
    int rank = 0;
    char line[1024];
    errno = 0;
    while (fgets(line, 1024, f) != NULL) {
        int len = strlen(line);
        if (len == 0) continue;
        if (line[0] == '#') continue;
        line[len - 1] = '\0';
        ok = PSDictSet(bpe, line, PSDictFromInt(rank++)) != NULL;
        if (!ok) {
            PSErr(__func__, "could not add bpe item '%s'", line);
            goto final;
        }
    }
    if (!feof(f)) {
        PSErr(NULL, "could not load whole file: '%s'", bpe_path);
        ok = 0;
        goto final;
    }
final:
    if (f != NULL) fclose(f);
    if (!ok) {
        PSDictRelease(bpe);
        bpe = NULL;
    }
    free(bpe_path);
    return bpe;
}

static int readPromptFromStdin(char *prompt, size_t max_size) {
    if (prompt == NULL || max_size == 0) return 0;
    size_t totread = 0;
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (++totread >= max_size) break;
        int i = totread - 1;
        if (c == '\n') {
            prompt[i] = '\0';
            break;
        }
        prompt[i] = c;
    }
    return totread;
}

static int getNextIndex(PSNeuralNetwork *gpt2_model, PSFloat temperature) {
    if (gpt2_model == NULL) return -1;
    PSLayer *outlayer = PSGetOutputLayer(gpt2_model);
    if (outlayer == NULL) return -1;
    if (temperature < 0) temperature = 0;
    int seqlen = PSStateSequenceLength(outlayer);
    int next_id = -1, t = seqlen - 1;
    if (temperature == 0) {
        if (!PSFindLayerMaxState(outlayer, NULL, &next_id, t)) return -1;
    } else {
        PSFloat *logits = PSGetStates(outlayer, t);
        if (logits == NULL) {
            PSErr(NULL, "output layer has not states at index %d", t);
            return -1;
        }
        PSFloat *probs = malloc(outlayer->size * sizeof(PSFloat));
        if (probs == NULL) {
            PSPrintMemoryErrorMsg();
            return -1;
        }
        PSMathOpts opts = {.acceleration = gpt2_model->acceleration};
        PSFloat temp = 1 - temperature;
        if (temp <= 0) temp = 1e-7;
        PSMultiplyVectorScalar(logits, temp, probs, outlayer->size, &opts);
        PSSoftmax(probs, probs, outlayer->size, &opts);
        int err = 0;
        next_id = PSRandomInt(outlayer->size, probs, &err, &opts);
        free(probs);
        if (err) {
            PSErr(NULL, "could not compute next ID with temperature %g",
                  temperature);
            next_id = -1;
        }
    }
    return next_id;
}

static int generate(PSNeuralNetwork *gpt2_model, char *prompt,
                    int max_output_tokens, PSVocabulary *vocabulary,
                    PSDict *bpe, GPT2HyperParameters *hparams, int verbose)
{
    if (gpt2_model == NULL) return 0;
    PSFloat *inputs = encodePrompt(prompt, vocabulary, bpe);
    int ok = (inputs != NULL), i;
    if (!ok) {
        PSErr(NULL, "failed to encode prompt");
        goto final;
    }
    int n_inputs = (int) inputs[0];
    if (verbose) {
        printf("Encoded inputs: %d\n", n_inputs);
        for (i = 0; i < n_inputs; i++)
            printf("%s%g", (i > 0 ? "," : ""), inputs[i + 1]);
        printf("\n");
    }
    PSLayer *outlayer = PSGetOutputLayer(gpt2_model);
    int generated_tokens = 0;
    printf("%s", prompt);
    while (generated_tokens < max_output_tokens) {
        ok = PSForward(gpt2_model, inputs);
        if (!ok) {
            printf("\n");
            fflush(stdout);
            PSErr(NULL, "forward step failed");
            goto final;
        }
        int seqlen = PSStateSequenceLength(outlayer);
        if (seqlen >= hparams->n_ctx) break;
        int next_id = getNextIndex(gpt2_model, temperature);
        if (!ok) goto final;
        size_t new_size = 1 + (size_t) ++n_inputs;
        PSFloat *new = realloc(inputs, new_size * sizeof(PSFloat));
        ok = (new != NULL);
        if (!ok) {
            PSPrintMemoryErrorMsg();
            goto final;
        }
        inputs = new;
        inputs[0] = n_inputs;
        inputs[n_inputs] = (PSFloat) next_id;
        if (verbose)
            printf("%s%d", (generated_tokens > 0 ? ", " : ""), next_id);
        char *token = (char *) PSVocabularyGetTokenByID(vocabulary, next_id);
        if (token == NULL) token = "!!UNKNOWN_TOKEN!!!";
        else {
            char *decoded = decodeToken(token);
            ok = decoded != NULL;
            if (!ok) {
                printf("\n");
                PSErr(NULL, "could not decode token '%s'", token);
                fflush(stdout);
                goto final;
            }
            //printf("'%s'\n", token);
            printf("%s", decoded);
        }
        fflush(stdout);
        generated_tokens++;
    }
    printf("\n");
    fflush(stdout);
final:
    free(inputs);
    return ok;
}

static void processInteractivePromptCommands(char *prompt, int max_ctx) {
    char cmd[50] = {0};
    char endchr[2] = {0};
    int matched = sscanf(prompt, "/%49[a-z-]%1[\n=]", cmd, endchr);
    if (matched == 0) {
        PSErr(NULL, "invalid command");
        return;
    }
    char *value = NULL;
    if (matched == 2 && endchr[0] == '=') value = prompt + (2 + strlen(cmd));
    fflush(stdout);
    if (strcmp("temperature", cmd) == 0) {
        if (value != NULL) {
            PSFloat temp = atof(value);
            if (temp < 0 || temp > 1) {
                PSErr(NULL, "valid temperature range: 0-1");
                return;
            }
            temperature = temp;
            printf("Set temperature to: %.1g\n", temperature);
        } else printf("%.1g\n", temperature);
    } else if (strcmp("max-output-length", cmd) == 0) {
        if (value != NULL) {
            int max = atoi(value);
            if (max < 1 || max > max_ctx) {
                PSErr(NULL, "max-output-length valid range: 1-%d", max_ctx);
                return;
            }
            max_output_tokens = max;
            printf("Set max output length to: %d\n", max_output_tokens);
        } else printf("%d\n", max_output_tokens);
    } else if (strcmp("help", cmd) == 0) {
        printf("Available commands:\n");
        printf("    /temperature            Print current temperature\n");
        printf("    /temperature=TEMP       Set temperature (range: 0-1)\n");
        printf("    /max-output-length      Print current max output length\n");
        printf("    /max-output-length=LEN  Set max output length (range: "
               "1-%d)\n", max_ctx);
        printf("    /exit                   Exit\n");
        printf("    /help                   Print this help\n");
    } else {
        printf("Invalid command '%s'\n", cmd);
        printf("Type /help for info about available commands.\n");
    }
}

static int startInteractivePrompt(PSNeuralNetwork *gpt2_model,
                                  PSVocabulary *vocabulary,
                                  PSDict *bpe, GPT2HyperParameters *hparams)
{
    PSNotice("Please, write some text: (type '/exit' to stop or /help for "
             "more commands)");
#ifndef HAS_READLINE
    char prompt[4096];
    fflush(stdout);
    printf("> ");
    int len;
    while ((len = readPromptFromStdin(prompt, 4095))) {
#else
    UNUSED(readPromptFromStdin);
    char *prompt = NULL;
    while ((prompt = readline("> "))) {
#endif
        int answered = 0;
        if (strlen(prompt) == 0) goto next;
        if (strcmp("/exit", prompt) == 0) break;
        if (prompt[0] == '/') {
            processInteractivePromptCommands(prompt, hparams->n_ctx);
            goto next;
        }
        int ok = generate(
            gpt2_model, prompt, max_output_tokens, vocabulary, bpe, hparams, 0
        );
        if (!ok) {
            printf("\n");
            PSErr(NULL, "failed to generate text");
            /*return 0;*/
        }
        answered = 1;
next:
        fflush(stdout);
#ifndef HAS_READLINE
        prompt[0] = '\0';
        printf("%s> ", (answered ? "\n" : ""));
#else
        add_history(prompt);
        free(prompt);
#endif
        /* PSResetNetworkStateSequences(gpt2_model, 0, 0); */
    }
    return 1;
}

void printHelp(char *executable) {
    static int models_count =
        (int) (sizeof(PSAvailableGPT2Models) / sizeof(int));
    int i;
    printf("Usage: %s [OPTIONS] [PROMPT]\n", executable);
    printf("\nOPTIONS:\n\n");
    printf("    --model MODELSIZE       GPT2 Model, available models:\n");
    printf("                            ");
    for (i = 0; i < models_count; i++)
        printf("%s%d", (i > 0 ? "," : ""), PSAvailableGPT2Models[i]);
    printf("\n");
    printf("                            Default: %d\n", model_size);
    printf("    --model-dir PATH        Custom model directory\n");
    printf("    --max-output-tokens N   Max tokens to generate (def. %d)\n",
        max_output_tokens);
    printf("    --temperature TEMP      Temperature (0-1) (def. %d\n",
        DEFAULT_TEMPERATURE);
    printf("    --overwrite-downloaded  Overwrite downloaded GPT2 files\n");
    printf("    --overwrite-extracted   Overwrite extracted GPT2 files\n");
    printf("    --[no-]binary-format    Enable/Disable saving layer\n"
           "                            parameters in binary format in order\n"
           "                            to save space and speed-up loading\n"
           "                            (default: %s)\n",
           (use_binary_files ? "enabled" : "disabled"));
    printf("    -i, --interactive       Interactive prompt\n");
    printf("    -v, --verbose           Verbose output\n");
    printf("    -h, --help              Print this help\n");
}

int parseOptions(int argc, char **argv) {
    int i, last_arg_idx = argc - 1, is_last;
    for (i = 1; i < argc; i++) {
        is_last = (i == last_arg_idx);
        char *arg = argv[i];
        if (strcmp("--model", arg) == 0 && !is_last) {
            model_size = atoi(argv[++i]);
            int valid_model = 0;
            switch (model_size) {
                case PS_GPT2_MODEL_124M:
                case PS_GPT2_MODEL_355M:
                case PS_GPT2_MODEL_774M:
                case PS_GPT2_MODEL_1558M: valid_model = 1;
            }
            if (!valid_model) {
                PSErr(NULL, "invalid model %s", argv[i]);
                exit(1);
            }
        } else if (strcmp("--model-dir", arg) == 0 && !is_last) {
            custom_model_dir = argv[++i];
            if (PSFileExists(custom_model_dir)) {
                if (!PSIsDirectory(custom_model_dir)) {
                    PSErr(NULL, "'%s' already exists and is not a directory",
                          custom_model_dir);
                    exit(1);
                }
            }
        } else if (strcmp("--max-output-tokens", arg) == 0 && !is_last) {
            max_output_tokens = atoi(argv[++i]);
            if (max_output_tokens < 1) {
                PSErr(NULL, "Output tokens must be at least 1");
                exit(1);
            }
        } else if (strcmp("--temperature", arg) == 0 && !is_last) {
            temperature = atof(argv[++i]);
            if (temperature < 0) temperature = 0;
            else if (temperature > 1) temperature = 1;
        } else if (strcmp("--overwrite-downloaded", arg) == 0) {
            overwrite_downloads = 1;
        } else if (strcmp("--overwrite-extracted", arg) == 0) {
            overwrite_extracted_files = 1;
        } else if (strcmp("--binary-format", arg) == 0) {
            use_binary_files = 1;
        } else if (strcmp("--no-binary-format", arg) == 0) {
            use_binary_files = 0;
        } else if (strcmp("--verbose", arg) == 0 || strcmp("-v", arg) == 0) {
            verbose = 1;
        } else if (strcmp("--interactive", arg) == 0 || strcmp("-i", arg)==0) {
            interactive = 1;
        } else if (strcmp("--help", arg) == 0 || strcmp("-h", arg) == 0) {
            printHelp(argv[0]);
            exit(1);
        } else {
            if (arg[0] == '-') {
                PSErr(NULL, "invalid argument '%s'", arg);
                exit(1);
            } else break;
        }
    }
    return i;
}

int main(int argc, char **argv) {
    PSLogEnableColor();
    default_loglevel = PSLogLevel;
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);

    /* Code here */
    int last_arg = parseOptions(argc, argv);
    char *prompt = NULL;
    if (last_arg < argc) prompt = argv[last_arg];
#ifdef HAS_READLINE
    if (interactive) rl_bind_key('\t', rl_complete);
    using_history();
#endif
    PSNeuralNetwork *gpt2_model = NULL;
    PSVocabulary *vocabulary = NULL;
    PSDict *bpe = NULL;
    int ok = 1, i;
    char *model_dir = custom_model_dir;
    char *exec_path = getExecutablePath(argv[0]);
    char *params_path = NULL;
    PSFloat *inputs = NULL;
    if (model_dir == NULL) {
        char basename[255] = {0};
        snprintf(basename, 255, "gpt2_%dM", model_size);
        const char *wdir = PSWorkingDirectory();
        ok =  (wdir != NULL);
        if (!ok) goto final;
        model_dir = PSPathJoin(3, wdir, "datasets", basename);
        ok = (model_dir != NULL);
        if (!ok) goto final;
    }
    GPT2HyperParameters hparams = {0};
    /* Eventually download, extract and process GTP2 model. */
    ok = downloadAndExtractGPT2Model(model_size, model_dir, &hparams,exec_path);
    if (!ok) {
        PSErr(NULL, "could not download and/or extract GPT2 model");
        goto final;
    }
    printdGPT2HyperParams(&hparams);
    PSLayer *l = NULL;

    /* Create Model */
    gpt2_model = PSCreateNetwork("GPT2");
    ok = (gpt2_model != NULL);
    if (!ok) goto final;
    gpt2_model->flags |= (
        FLAG_USE_SEQUENCES | FLAG_ONEHOT
    );
    gpt2_model->flags &= ~((unsigned) FLAG_RECURRENT);
    PSNotice("Loading GPT2 Model");
    if (!verbose) PSLogLevel = PSLOGLEVEL_NOTICE;
    int print_progress = (!verbose);
    /* Input Layer */
    l = PSAddLayer(gpt2_model, FullyConnected, hparams.n_vocab, PSLDEF(
        .flags = FLAG_ONEHOT | FLAG_USE_SEQUENCES
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (print_progress) print_progress = printProgressBar(gpt2_model);
    int has_bin = 0;
    /* Embedding Layer */
    params_path = PSPathJoin(2, model_dir, "extracted/wte.T");
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(gpt2_model, Embedding, hparams.n_embd, PSLDEF(
        .load_from = params_path,
        .flags = FLAG_NON_TRAINABLE | FLAG_NO_BIAS | FLAG_USE_SEQUENCES
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    PSDisablePretraining(l);
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    if (print_progress) print_progress = printProgressBar(gpt2_model);
    /* Positional Encoding Layer */
    free(params_path);
    params_path = PSPathJoin(2, model_dir, "extracted/wpe");
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(gpt2_model, PositionalEncoding, hparams.n_embd, PSLDEF(
        .load_from = params_path,
        .flags = FLAG_NON_TRAINABLE | FLAG_NO_BIAS | FLAG_USE_SEQUENCES
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    if (print_progress) print_progress = printProgressBar(gpt2_model);
    /* Transformer Blocks */
    for (i = 0; i < hparams.n_layer; i++) {
        ok = PSAddGPT2TransformerBlock(gpt2_model, i, &hparams, model_dir);
        if (!ok) {
            PSErr(NULL, "failed to add transformer block %d", i);
            goto final;
        }
    }
    /* Normalization Layer */
    free(params_path);
    params_path = PSPathJoin(3, model_dir, "extracted/processed", "ln_f");
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(gpt2_model, Normalization, hparams.n_embd, PSLDEF(
        .load_from = params_path,
        .flags = FLAG_USE_SEQUENCES
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    if (print_progress) print_progress = printProgressBar(gpt2_model);
    /* Output Layer */
    free(params_path);
    params_path = PSPathJoin(3, model_dir, "extracted", "wte");
    if (use_binary_files) has_bin = getBinaryParamsPathIfExists(params_path);
    l = PSAddLayer(gpt2_model, Linear, hparams.n_vocab, PSLDEF(
        .load_from = params_path,
        .flags = FLAG_USE_SEQUENCES | FLAG_NO_BIAS
    ));
    ok = (l != NULL);
    if (!ok) goto final;
    if (use_binary_files && !has_bin) saveBinaryParamsFor(l, params_path);
    if (print_progress) print_progress = printProgressBar(gpt2_model);
    printf("\n");
    fflush(stdout);

    ok = PSBuildNetwork(gpt2_model);
    if (!ok) goto final;
    if (prompt == NULL) interactive = 1;
    if (verbose || (prompt == NULL && !interactive))
        PSPrintNetworkInfo(gpt2_model);

make_input:
    vocabulary = loadVocabulary(model_dir, hparams.n_vocab);
    ok = (vocabulary != NULL);
    if (!ok) {
        PSErr(NULL, "failed to load vocabulary");
        goto final;
    }
    bpe = loadBPE(model_dir);
    ok = (bpe != NULL);
    if (!ok) {
        PSErr(NULL, "failed to load BPE");
        goto final;
    }
    /*PSClearScreen();*/
    if (!interactive) {
        printf("\n");
        fflush(stdout);
        ok = generate(gpt2_model, prompt, max_output_tokens, vocabulary, bpe,
                      &hparams, verbose);
        if (!ok) {
            PSErr(NULL, "failed to generate text");
            goto final;
        }
    } else {
        ok = startInteractivePrompt(gpt2_model, vocabulary, bpe, &hparams);
    }
final:
    PSLogLevel = default_loglevel;
    if (model_dir != custom_model_dir) free(model_dir);
    PSDeleteNetwork(gpt2_model);
    PSVocabularyRelease(vocabulary);
    PSDictRelease(bpe);
    free(params_path);
    free(inputs);
    return (ok ? 0 : 1);
}

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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include "../psyc.h"
#include "../utils.h"
#include "../debug.h"
#include "test.h"

#define MAX_ERROR_LEN   4096 * 10

int stdout_fd = -999;

PSFloat getRoundedFloatDec(PSFloat n, unsigned int decimals) {
    PSFloat rounder = (PSFloat) PSPow(10.0, (PSFloat) decimals);
    return (PSRound(n * rounder) / rounder);
}

TestCase *createTest(char *name) {
    TestCase *test_case = malloc(sizeof(TestCase));
    test_case->name = name;
    test_case->setup = NULL;
    test_case->teardown = NULL;
    test_case->count = 0;
    test_case->tests = NULL;
    test_case->data = NULL;
    return test_case;
}

Test *addTest(TestCase *test_case, char *name, char *errmsg,
               TestFunction func) {
    Test test;
    test.name = name;
    test.error_message = errmsg;
    test.run = func;
    test.status = NOT_PERFORMED;
    test_case->count++;
    if (test_case->tests == NULL) {
        test_case->tests = malloc(sizeof(Test));
    } else {
        test_case->tests = realloc(test_case->tests,
                                   sizeof(Test) * test_case->count);
    }
    test_case->tests[test_case->count - 1] = test;
    return &(test_case->tests[test_case->count - 1]);
}


int performTests(TestCase *test_case) {
    printf("\n");
    printf(BOLD "Performing tests on %s\n", test_case->name);
    printf(RESET);
    if (test_case->setup != NULL) {
        printf(" -> setup\n");
        printf(DIM);
        int ok = test_case->setup(test_case);
        printf(RESET);
        if (!ok) {
            printf(RED "Setup failed!\n" RESET);
            return 1;
        }
    }
    int i, errors = 0, count = test_case->count;
    time_t start_t, end_t;
    time(&start_t);
    for (i = 0; i < count; i++) {
        Test *test = &(test_case->tests[i]);
        printf(" -> [%d] ", i);
        printf(CYAN "%s", test->name); printf(":");
        printf(RESET);
        /* printf(HIDDEN); */
#ifndef PS_VERBOSE_TESTS
        stdout_fd = dup(fileno(stdout));
        PSOriginalStdOutFD = stdout_fd;
        freopen("/dev/null", "w", stdout);
#endif
        test->status = test->run(test_case, test);
#ifndef PS_VERBOSE_TESTS
        fflush(stdout);
        fclose(stdout);
        stdout = fdopen(stdout_fd, "w");
#endif
        /* printf(RESET); */
        if (!test->status) {
            printf(RED "    FAILED");
            if (test->error_message != NULL)
                printf(YELLOW "\n    %s\n", test->error_message);
            errors++;
        } else printf(GREEN "    OK");
        printf("\n" RESET);
    }
    if (test_case->teardown != NULL) {
        printf(" -> teardown\n");
        int ok = test_case->teardown(test_case);
        if (!ok) {
            printf(RED "Teardown failed!\n" RESET);
            return 1;
        }
    }
    time(&end_t);
    printf("Tests performed in %d sec.\n", (int) (end_t - start_t));
    printf("Found ");
    if (errors > 0) printf(RED "%d errors.\n", errors);
    else printf(GREEN "no errors.\n");
    printf(RESET);
    return errors;
}

void deleteTest(TestCase *test_case) {
    if (test_case->data != NULL) free(test_case->data);
    if (test_case->tests != NULL) {
        int i;
        for (i = 0; i < test_case->count; i++) {
            Test *test = &(test_case->tests[i]);
            if (test->error_message != NULL) free(test->error_message);
        }
        free(test_case->tests);
    }
    free(test_case);
}

void setTestErrorMessage(Test *test, char *fmt, ...) {
    assert(test != NULL);
    if (test->error_message != NULL) free(test->error_message);
    test->error_message = malloc(MAX_ERROR_LEN);
    if (test->error_message == NULL) {
        fprintf(
            stderr,"FATAL: could not allocate memory for test error message\n"
        );
        abort();
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(test->error_message, MAX_ERROR_LEN, fmt, ap);
    va_end(ap);
}

void appendTestErrorMessage(Test *test, char *fmt, ...) {
    assert(test != NULL);
    int len = 0, maxlen;
    if (test->error_message == NULL) {
        test->error_message = malloc(MAX_ERROR_LEN);
        if (test->error_message == NULL) {
            fprintf(
                stderr,
                "FATAL: could not allocate memory for test error message\n"
            );
            abort();
        }
    } else len = strlen(test->error_message);
    if (len >= MAX_ERROR_LEN) return;
    char *msg = test->error_message + len;
    maxlen = MAX_ERROR_LEN - len;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, maxlen, fmt, ap);
    va_end(ap);
}

void buildAssertionMessage(Test *test, char *file, int line, const char *func,
                           char *expr, char *msg, ...)
{
    appendTestErrorMessage(test,
        "Test assertion failed for test \"%s\":\n", test->name);
    appendTestErrorMessage(test, "    %s\n", expr);
    appendTestErrorMessage(test,
        "    In %s:%d (%s)\n", file, line, func);
    if (msg != NULL) {
        appendTestErrorMessage(test, "    ");
        int len = strlen(test->error_message),
            maxlen = MAX_ERROR_LEN - len - 1;
        va_list ap;
        va_start(ap, msg);
        vsnprintf(test->error_message + len, maxlen, msg, ap);
        va_end(ap);
    }
}

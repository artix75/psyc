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

#ifndef __PS_TEST_H
#define __PS_TEST_H

#include "../types.h"
#define NOT_PERFORMED -1

#define testAssert(expr, test) do {\
    if (!(expr)){\
        buildAssertionMessage(test, __FILE__, __LINE__, __func__, #expr,NULL);\
        return 0;\
    }\
} while (0)

#define testAssertWithMessage(expr, test, msg, ...) do {\
    if (!(expr)){\
        buildAssertionMessage(test, __FILE__, __LINE__, __func__, #expr,\
            msg, __VA_ARGS__);\
        return 0;\
    }\
} while (0)

#define testAssertWithMessageOrGoto(expr, gotolabel, test, msg, ...) do {\
    if (!(expr)){\
        buildAssertionMessage(test, __FILE__, __LINE__, __func__, #expr,\
            msg, __VA_ARGS__);\
        goto gotolabel;\
    }\
} while (0)

#define testAssertEqual(a, b, test) testAssert((a == b), test)
#define testAssertEqualWithMsg(a, b, test, msg, ...) \
    testAssertWithMessage((a == b), test, msg, __VA_ARGS__)
#define testAssertNotEqual(a, b, test) testAssert((a != b), test)
#define testAssertNotEqualWithMsg(a, b, test, msg, ...) \
    testAssertWithMessage((a != b), test, msg, __VA_ARGS__)
#define testAssertNull(o, test) testAssert((o == NULL), test)
#define testAssertNotNull(o, test) testAssert((o != NULL), test)

struct Test;
struct TestCase;

typedef int (* TestFunction) (struct TestCase *test_case, struct Test *test);
typedef int (* SetupFunction) (struct TestCase *test_case);
typedef int (* TeardownFunction) (struct TestCase *test_case);

typedef struct Test {
    char *name;
    char *error_message;
    int status;
    TestFunction run;
} Test;

typedef struct TestCase {
    char *name;
    SetupFunction setup;
    TeardownFunction teardown;
    int count;
    int failed_count;
    Test *tests;
    void **data;
} TestCase;

TestCase *createTest(char *name);
Test *addTest(TestCase *test_case, char *name, char *errmsg, TestFunction func);
int  performTests(TestCase *test_case);
void deleteTest(TestCase *test_case);
void setTestErrorMessage(Test *test, char *fmt, ...);
void appendTestErrorMessage(Test *test, char *fmt, ...);
void buildAssertionMessage(Test *test, char *file, int line, const char *func,
                           char *expr, char *msg, ...);
PSFloat getRoundedFloatDec(PSFloat n, unsigned int decimals);

#endif /*  __PS_TEST_H */

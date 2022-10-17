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

#ifndef __PS_TEST_H
#define __PS_TEST_H

#define NOT_PERFORMED -1

#define testAssert(test, expr) do {\
    if (!(expr)){\
        buildAssertionMessage(test, __FILE__, __LINE__, __func__, #expr,NULL);\
        return 0;\
    }\
} while (0)

#define testAssertWithMessage(test, expr, msg, ...) do {\
    if (!(expr)){\
        buildAssertionMessage(test, __FILE__, __LINE__, __func__, #expr,\
            msg, __VA_ARGS__);\
        return 0;\
    }\
} while (0)

#define testAssertEqual(test, a, b) testAssert(test, (a == b))
#define testAssertNotEqual(test, a, b) testAssert(test, (a != b))
#define testAssertNull(test, o) testAssert(test, (o == NULL))
#define testAssertNotNull(test, o) testAssert(test, (o != NULL))

typedef int (* TestFunction) (void* test_case, void* test);
typedef int (* SetupFunction) (void* test_case);
typedef int (* TeardownFunction) (void* test_case);

typedef struct {
    char *name;
    char *error_message;
    int status;
    TestFunction run;
} Test;

typedef struct {
    char *name;
    SetupFunction setup;
    TeardownFunction teardown;
    int count;
    Test *tests;
    void **data;
} TestCase;

TestCase *createTest(char *name);
Test *addTest(TestCase *test_case, char *name, char *errmsg,
               TestFunction func);
int performTests(TestCase *test_case);
void deleteTest(TestCase *test_case);
void setTestErrorMessage(Test *test, char *fmt, ...);
void appendTestErrorMessage(Test *test, char *fmt, ...);
void buildAssertionMessage(Test *test, char *file, int line, const char *func,
                           char *expr, char *msg, ...);

#endif /*  __PS_TEST_H */

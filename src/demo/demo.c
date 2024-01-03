/*
 * Copyright (C) 2016-2024 Giuseppe Fabio Nicotra <artix2 at gmail dot com>.
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
#include <time.h>
#include "psyc.h"

#define INPUTS_SIZE (28 * 28)


PSFloat normalized_rand() {
    srand(time(NULL));
    int r = rand();
    return ((PSFloat) r / (PSFloat) RAND_MAX);
}

int main(int argc, char** argv) {
    PSModel *model = PSModelCreate(NULL);
    PSAddLayer(model, FullyConnected, INPUTS_SIZE, NULL);
    PSAddLayer(model, FullyConnected, 30, NULL);
    PSAddLayer(model, FullyConnected, 10, NULL);

    PSFloat values[INPUTS_SIZE];
    int i;
    for (i = 0; i < INPUTS_SIZE; i++) {
        values[i] = normalized_rand();
    }
    PSForward(model, values);

    PSModelFree(model);

    PSFloat nums[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    testShuffle(nums, 6, 2);
    exit(0);
}

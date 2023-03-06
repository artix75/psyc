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
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <dirent.h>

#include "cifar.h"
#include "convolutional.h"
#include "utils.h"
#include "log.h"

#define CIFAR_FILE_IMG_COUNT 10000
#define CIFAR_IMAGE_BYTESIZE 3072
#define CIFAR_DATAFILE_COUNT 6

static int compareFilenames(const void* a, const void* b) {
    return strcmp((const char*)a, (const char*)b);
}

int PSLoadCIFARData(int type, int classes, const char *dataset_path,
                    PSFloat **data, int max_files, int max_elements)
{
    if (classes != 10 && classes != 100) {
        PSErr(__func__, "Invalid classes %d: only 10 or 100 allowed.", classes);
        return 0;
    }
    int label_size = (classes == 100 ? 2 : 1);
    int img_count = CIFAR_FILE_IMG_COUNT;
    if (max_elements > 0) img_count = max_elements;
    int expected_fsize = (CIFAR_IMAGE_BYTESIZE + label_size) * img_count;
    int fcount = 0, dataset_size = 0, i, j, k;
    char datafiles[CIFAR_DATAFILE_COUNT][255];

    DIR *dir;
    struct dirent *finfo;
    dir = opendir(dataset_path);
    if (dir == NULL) {
        PSErr(__func__, "Invalid path %s", dataset_path);
        return 0;
    }

    char *prfx = (type == DATA_TYPE_TRAINING ? "data_batch" : "test_batch");
    while ((finfo = readdir(dir))) {
        if (strstr(finfo->d_name, prfx) == NULL) continue;
        sprintf(datafiles[fcount++], "%s/%s", dataset_path, finfo->d_name);
    }
    if (fcount == 0) {
        PSErr(__func__, "Empty directory: %s", dataset_path);
        return 0;
    }
    qsort(datafiles, fcount, 255, compareFilenames);
    if (max_files > 0 && max_files < fcount) fcount = max_files;

    int datasize = (fcount * img_count * (classes + CIFAR_IMAGE_BYTESIZE));
    dataset_size = datasize * sizeof(PSFloat);
    *data = calloc(dataset_size, 1);
    if (*data == NULL) return 0;
    PSFloat *data_p = *data;
    for (i = 0; i < fcount; i++) {
        char *fname = datafiles[i];
        printf("Reading %s\n", fname);
        FILE *f = fopen(fname, "r");
        if (f == NULL) {
            PSErr(__func__, "Could not open file %s", fname);
            free(*data);
            data = NULL;
            return 0;
        }
        fseek(f, 0, SEEK_END);
        int pos = ftell(f);
        if (pos < expected_fsize) {
            PSErr(
                __func__, "Invalid file size: %d != %d (expected)", pos,
                expected_fsize
            );
            free(*data);
            data = NULL;
            fclose(f);
            return 0;
        }
        fseek(f, 0, SEEK_SET);
        for (j = 0; j < img_count; j++) {
            int label = 0;
            if (classes == 10) {
                label = fgetc(f);
            } else {
                label = 100 * fgetc(f);
                label += fgetc(f);
            }
            /* printf("Label: %d\n", label); */
            for (k = 0; k < CIFAR_IMAGE_BYTESIZE; k++) {
                float b = (float)(fgetc(f)) / 255.0f;
                /* printf("%d ", b); */
                *(data_p++) = (PSFloat) b;
            }
            for (k = 0; k < classes; k++) {
                PSFloat y = (k == label ? 1.0 : 0.0);
                *(data_p++) = y;
            }
        }
        fclose(f);
    }
    (void) closedir (dir);
    return dataset_size;
}

PSLayer *PSAddCIFARInputLayer(PSNeuralNetwork *network) {
    if (network->size > 0) {
        PSErr(__func__, "CIFAR layer must be input layer!\n");
        return NULL;
    }
    PSLayerDef ldef = {
        .output_depth = 3, .output_columns = 32, .output_rows = 32
    };
    return PSAddLayer(network, FullyConnected, CIFAR_IMAGE_SIZE, &ldef);
}

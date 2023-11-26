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
#include <stdio.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <sys/time.h>

#include "../psyc.h"
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
#include "../log.h"
#include "../lstm.h"
#include "../maths.h"
#include "../normalization.h"
#include "../operator-layer.h"
#include "../attention.h"
#include "../optimization.h"
#include "../platform.h"
#include "../recurrent.h"
#include "../types.h"
#include "../utils.h"

#define EOS  0

#define HIDDEN_SIZE  20
#define EMBED_SIZE  40
#define INIT_RANGE  0.3
#define BATCH_SIZE 1

#define LEARNING_RATE  0.001
#define MULTABLE_NROWS 10
#define MULTABLE_NCOLS 10
#define MOMENTUM       0.9
#define CLIP_GRAD      5.0
#define EPOCHS         1800
#define DEFAULT_OUTPUT_FILE "/tmp/pretrained.attention_encoder_decoder.psmodel"
#define OPTIMIZATION PSAdamOptimization

#define UNUSED(V) ((void) V)

char *output_path = DEFAULT_OUTPUT_FILE;
char *load_model_file = NULL;
int epochs = EPOCHS, batch_size = BATCH_SIZE;
PSLayerType rnn_type = GRU;
PSFloat lr = LEARNING_RATE, momentum = MOMENTUM, clip = CLIP_GRAD;
PSOptimization optimization = OPTIMIZATION;
int train_mul_table = 1;
int link2attn = 0;
int no_attention = 0;
int force_epochs = 0, force_lr = 0;
int rand_autoregression = 0;
int n_heads = 0;
PSAttentionType attn_type = PSDotAttention;
int hidden_size = HIDDEN_SIZE, embed_size = EMBED_SIZE;

char *PSGetRecurrentModeLabel(PSRecurrentNetworkMode mode);
char *getOptimizationName(PSOptimization optimization);

static PSFloat *getMultiplicationTable(int max_x) {
    if (max_x > MULTABLE_NROWS) max_x = MULTABLE_NROWS;
    static PSFloat table[MULTABLE_NROWS * MULTABLE_NCOLS] = {0};
    PSFloat *table_p = table;
    int i, j;
    for (i = 0; i < max_x; i++) {
        PSFloat *row = table_p + (i * MULTABLE_NCOLS);
        int n_i = i + 1;
        for (j = 0; j < MULTABLE_NCOLS; j++) {
            int n_j = j + 1;
            row[j] = n_i * n_j;
        }
    }
    return table_p;
}

void printHelp(char *progname) {
    char optimization_name[255] = {0};
    char *uc_optimization_name = getOptimizationName(OPTIMIZATION);
    int namelen = strlen(uc_optimization_name), i;
    for (i = 0; i < namelen; i++)
        optimization_name[i] = tolower(uc_optimization_name[i]);
    printf("Usage %s OPTIONS\n", progname);
    printf("    OPTIONS:\n");
    printf("        -l, --load TRAINED_DT_FILE      Load pretrained model\n");
    printf("        -s, --save TRAINED_DT_FILE      Save trained model\n"
           "                                        "
           "(default: %s)\n", DEFAULT_OUTPUT_FILE);
    printf("        --gru                           Use GRU (default)\n");
    printf("        --lstm                          Use LSTM\n");
    printf("        --rnn                           Use RNN\n");
    printf("        --hidden-size SIZE              Recurrent layers size\n");
    printf("                                        (def. %d)\n", HIDDEN_SIZE);
    printf("        --embed-size SIZE               Emedding layers size\n");
    printf("                                        (def. %d)\n", EMBED_SIZE);
    printf("        --learning-rate RATE            Learnig Rate "
        "(def. %g)\n", LEARNING_RATE);
    printf("        --momentum MOMENTUM             Momentum "
        "(def. %g)\n", MOMENTUM);
    printf("        --optimization                  Training Optimization \n"
          "                                        "
          "(adagrad,adadelta,adam,windowgrad,\n"
          "                                        "
          "nesterov,rmsprop)\n"
          "                                        "
          "Default: %s\n", optimization_name
    );
    printf("        --train-multiplication-table    Train on multiplication "
        "table\n");
    printf("        -a, --attention-type TYPE       Attention Type: dot|add\n");
    printf("                                        (def. dot)\n");
    printf("        --heads NUM                     Attention Heads\n");
    printf("        --link-attention-layer          Link to attention layer\n");
    printf("        --no-attention                  Disable Attention\n");
    printf("        --randomized-autoregression     Randomize autoregression\n"
    );
    printf("        --epochs EPOCHS                 Epochs (def. %d)\n",
           EPOCHS);
    printf("        --batch-size SIZE               Batch size (def. %d)\n",
        BATCH_SIZE);
#ifdef USE_AVX
    printf("        --disable-avx                   Disable AVX\n");
#endif
    printf("        --no-shuffle                    Don't shuffle data\n");
    printf("        -h, --help                      Print this help\n");
}

void parseOptions(int argc, char **argv) {
    int i, last_arg, last_idx = argc - 1;
    for (i = 1; i < argc; i++) {
        last_arg = (i == last_idx);
        char *arg = argv[i];
        if (strcmp("--learning-rate", arg) == 0 && !last_arg) {
            lr = atof(argv[++i]);
            if (lr <= 0.0) {
                fprintf(stderr, "ERROR: learning rate must > 0\n");
                exit(1);
            }
            force_lr = 1;
        } else if (strcmp("--clip", arg) == 0 && !last_arg) {
            clip = atof(argv[++i]);
            if (clip < 0.0) clip *= -1;
        } else if (strcmp("--epochs", arg) == 0 && !last_arg) {
            epochs = atoi(argv[++i]);
            if (epochs <= 0) {
                fprintf(stderr, "ERROR: epochs must > 0\n");
                exit(1);
            }
            force_epochs = 1;
        } else if (strcmp("--hidden-size", arg) == 0 && !last_arg) {
            hidden_size = atoi(argv[++i]);
            if (hidden_size < 2) {
                fprintf(stderr, "ERROR: --hidden-size must >= 2\n");
                exit(1);
            }
        } else if (strcmp("--embed-size", arg) == 0 && !last_arg) {
            embed_size = atoi(argv[++i]);
            if (embed_size < 2) {
                fprintf(stderr, "ERROR: --embed-size must >= 2\n");
                exit(1);
            }
        } else if (strcmp("--heads", arg) == 0 && !last_arg) {
            n_heads = atoi(argv[++i]);
            if (n_heads < 0) n_heads = 0;
        } else if ((strcmp("--attention-type", arg) == 0 ||
                    strcmp("-a", arg) == 0) && !last_arg)
        {
            char *type = argv[++i];
            if (strcmp("add", type) == 0) attn_type = PSAdditiveAttention;
            else if (strcmp("dot", type) == 0) attn_type = PSDotAttention;
            else {
                PSErr(NULL, "invalid attention type '%s'", type);
                exit(1);
            }
        } else if (strcmp("--optimization", arg) == 0 && !last_arg) {
            char *optname = argv[++i];
            if (strcmp("adam", optname) == 0)
                optimization = PSAdamOptimization;
            else if (strcmp("adagrad", optname) == 0)
                optimization = PSAdaGradOptimization;
            else if (strcmp("adadelta", optname) == 0)
                optimization = PSAdaDeltaOptimization;
            else if (strcmp("windowgrad", optname) == 0)
                optimization = PSWindowGradOptimization;
            else if (strcmp("nesterov", optname) == 0)
                optimization = PSNesterovOptimization;
            else if (strcmp("rmsprop", optname) == 0)
                optimization = PSRMSPropOptimization;
            else if (strcmp("none", optname) == 0)
                optimization = PSDefaultOptimization;
            else {
                fprintf(stderr, "Invalid optimization `%s`\n", optname);
                fprintf(
                    stderr, "Valid values: adam, adagrad, adadelta, "
                    "windowgrad, nesterov\n"
                );
                exit(1);
            }
        } else if (strcmp("--lstm", arg) == 0) {
            rnn_type = LSTM;
        } else if (strcmp("--gru", arg) == 0) {
            rnn_type = GRU;
        } else if (strcmp("--rnn", arg) == 0) {
            rnn_type = RNNLayer;
        } else if (strcmp("--train-multiplication-table", arg) == 0) {
            train_mul_table = 1;
        } else if (strcmp("--link-attention-layer", arg) == 0) {
            link2attn = 1;
        } else if (strcmp("--no-attention", arg) == 0) {
            no_attention = 1;
        } else if (strcmp("--randomized-autoregression", arg) == 0) {
            rand_autoregression = 1;
        } else if (strcmp("--load", arg) == 0 || strcmp("-l", arg) == 0) {
            if (last_arg) {
                fprintf(stderr, "ERROR: missing model file\n");
                exit(1);
            }
            load_model_file = argv[++i];
            continue;
        } else if (strcmp("--save", arg) == 0 || strcmp("-s", arg) == 0) {
            if (last_arg) {
                fprintf(stderr, "ERROR: missing output_path\n");
                exit(1);
            }
            output_path = argv[++i];
            continue;
        } else if (strcmp("--help", arg) == 0 || strcmp("-h", arg) == 0) {
            printHelp(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "ERROR: Invalid option `%s`\n", arg);
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    PSLogEnableColor();
#ifdef CATCH_FPE
    PSCatchFloatingPointExceptions(FE_OVERFLOW | FE_DIVBYZERO);
#endif
    PSHandleSignals(NULL);
    int input_size, output_size;
    parseOptions(argc, argv);
    if (no_attention) link2attn = 0;
    PSFloat *training_data = NULL;
    int data_size = 0;

    int success = 1;
    PSFloat *table_train_data = NULL;
    PSFloat *table = NULL;
    if (train_mul_table) {
        int max_mul_x = MULTABLE_NROWS;
        input_size = output_size = (max_mul_x * MULTABLE_NCOLS) + 1;
        table = getMultiplicationTable(max_mul_x);
        int sent_len = 3, x_sent_count = (max_mul_x / sent_len);
        int sent_count = x_sent_count * max_mul_x;
        int xlen, ylen, r, c;
        xlen = ylen = sent_len + 1;
        data_size = (1 + (sent_count * (xlen + ylen)));
        PSFloat *row = table;
        table_train_data = calloc(data_size, sizeof(PSFloat));
        if (table_train_data == NULL) {
            PSPrintMemoryErrorMsg();
            return 1;
        }
        PSFloat *data_p = table_train_data;
        *(data_p++) = (PSFloat) sent_count;
        for (r = 0; r < max_mul_x; r++) {
            for (c = 0; c < 10; c += sent_len) {
                int xidx = c, yidx = c + sent_len, x, y;
                if (yidx >= 10) break;
                PSFloat *xsrc = row + xidx, *ysrc = row + yidx;
                *(data_p++) = (PSFloat) x_sent_count;
                for (x = 0; x < sent_len; x++) {
                    PSFloat xval = xsrc[x];
                    *(data_p++) = xval;
                }
                *(data_p++) = (PSFloat) x_sent_count;
                for (y = 0; y < sent_len; y++) {
                    PSFloat yval = 0;
                    if ((yidx + y) < 10) yval = ysrc[y];
                    *(data_p++) = yval;
                }
            }
            row += 10;
        }
        training_data = table_train_data;
    }

    PSLayerDef common_ldef = {.init_range = INIT_RANGE};
    PSModel *encoder = NULL, *decoder = NULL;
    PSLayer *encoder_rnn = NULL, *decoder_rnn = NULL,
            *decoder_embed = NULL, *decoder_attn = NULL;
    int do_train = 1;

    if (load_model_file == NULL) {
        /* Encoder */
        encoder = PSModelCreate("Encoder");
        success = encoder != NULL;
        if (!success) goto final;
        PSLayer *l = PSAddLayer(encoder, FullyConnected, input_size, PSLDEF(
            .flags = PS_FLAG_ONEHOT, .init_range = INIT_RANGE
        ));
        success = l != NULL;
        if (!success) goto final;
        l = PSAddLayer(encoder, Embedding, embed_size, &common_ldef);
        success = l != NULL;
        PSDisablePretraining(l);
        if (!success) goto final;
        l = PSAddLayer(encoder, rnn_type, hidden_size, &common_ldef);
        success = l != NULL;
        if (!success) goto final;
        encoder_rnn = l;

        /* Decoder */
        decoder = PSModelCreate("Decoder");
        success = decoder != NULL;
        if (!success) goto final;
        l = PSAddLayer(decoder, FullyConnected, output_size, PSLDEF(
            .flags = PS_FLAG_ONEHOT
        ));
        success = l != NULL;
        if (!success) goto final;
        l = PSAddLayer(decoder, Embedding, embed_size, &common_ldef);
        success = l != NULL;
        if (!success) goto final;
        decoder_embed = l;
        PSDisablePretraining(l);
        if (!no_attention) {
            l = PSAddLayer(decoder, Attention, hidden_size, PSLDEF(
                .init_range = INIT_RANGE,
                .flags = PS_FLAG_RECURRENT,
                .attention_type = attn_type,
                .keys_provider = encoder_rnn,
                .attention_heads = n_heads,
            ));
            success = l != NULL;
            if (!success) goto final;
            decoder_attn = l;
            PSLayer *providers[] = {decoder_attn, decoder_embed};
            l = PSAddLayer(decoder, OperatorLayer, 0, PSLDEF(
                .providers_count = 2,
                .providers = providers,
                .operator = PSConcatenateOperator
            ));
            success = l != NULL;
            if (!success) goto final;
        }
        l = PSAddLayer(decoder, rnn_type, hidden_size, &common_ldef);
        success = l != NULL;
        if (!success) goto final;
        decoder_rnn = l;
        PSSetAttentionQueryProvider(decoder_attn, decoder_rnn);
        l = PSAddLayer(decoder, SoftMax, output_size, &common_ldef);
        success = l != NULL;
        l->flags |= PS_FLAG_ONEHOT;
        if (!success) goto final;
        decoder->flags |= PS_FLAG_AUTOREGRESSION;
        decoder->sequence_settings.end = 0;


        PSModelLink link = {
            .layer = decoder_rnn,
            .previous_layer = encoder_rnn
        };
        if (link2attn) link.layer = decoder_attn;
        success = PSAddModel(encoder, decoder, &link);
    } else {
        encoder = PSModelCreate("Encoder");
        success = encoder != NULL;
        if (!success) goto final;
        success = PSModelLoad(encoder, load_model_file);
        if (success) success = (encoder != NULL && encoder->layers != NULL);
        if (!success) goto final;
        decoder = encoder->next;
        success = decoder != NULL;
        if (!success) {
            PSErr(NULL, "could not load decoder");
            goto final;
        }
        do_train = 0;
        int input_size = PSGetOneHotLayerVectorSize(encoder->layers[0]);
        train_mul_table =
            (input_size == ((MULTABLE_NROWS * MULTABLE_NCOLS) + 1));
        if (train_mul_table && table == NULL)
            table = getMultiplicationTable(MULTABLE_NROWS);
    }
    if (!PSModelBuild(encoder)) {
        success = 0;
        goto final;
    }
    if (!PSModelBuild(decoder)) {
        success = 0;
        goto final;
    }
    PSModelPrintInfo(encoder);
    PSTrainingOptions opts = {
        .flags = PS_TRAINING_FLAG_TEACHER_FORCING | PS_TRAINING_FLAG_SEQ2SEQ |
                 PS_TRAINING_NO_SHUFFLE,
        .optimization = optimization,
        .epochs = epochs,
        .learning_rate = lr,
        .batch_size = batch_size,
        .bptt_truncate = 0,
    };
    if (do_train) {
        PSTrain(encoder, training_data, data_size, NULL, 0, &opts);
        PSResetModelStateSequences(encoder, 0, 0);
        PSResetModelStateSequences(decoder, 0, 0);
    }
    PSFloat x_table[4] = {3, 0, 0, 0};
    PSFloat y_table[4] = {3, 0, 0, 0};
    PSFloat *x = NULL, *y = NULL;
    if (train_mul_table && table != NULL) {
        int row = PSRandomInt(10, NULL, NULL, NULL);
        if (row >= 10) row = 9;
        PSFloat *xrow = table + (10 * row);
        if (PSNormalizedRandom() > 0.5) xrow += 3;
        x = x_table;
        y = y_table;
        PSVectorCopy(x + 1, xrow, 3);
        PSVectorCopy(y + 1, xrow + 3, 3);
    }
    PSNotice("Testing Autoregression");
    printf("Inputs:\n");
    PSVectorPrint(x + 1, ((int) *x), ", ");
    if (y != NULL) {
        printf("Expected:\n");
        PSVectorPrint(y + 1, ((int) *y), ", ");
    }
    decoder->sequence_settings.max_length = 3;
    success = PSAutoregression(encoder, x, rand_autoregression, NULL);
    if (!success) {
        PSErr(NULL, "autoregression failed");
        goto final;
    }
    PSLayer *output_layer = decoder->layers[decoder->size - 1];
    int seqlen = PSStateSequenceLength(output_layer);
    printf("Produced %d output(s)\n", seqlen);
    for (int t = 0; t < seqlen; t++) {
        int max_idx = -1;
        if (!PSFindLayerMaxState(output_layer, NULL, &max_idx, t)) {
            PSErr(NULL, "PSFindLayerMaxState failed at t=%d", t);
            success = 0;
            goto final;
        }
        printf("Prediction[%d] = ", t);
        if (y != NULL && (PSFloat) max_idx == *(y + 1 + t))
            printf(PSCOLOR_GREEN);
        printf("%d\n", max_idx);
        printf(PSCOLOR_RESET);
    }
    if (output_path != NULL && do_train) PSModelSave(encoder, output_path);
final:
    if (encoder != NULL) PSModelFree(encoder);
    free(table_train_data);
    if (!success) PSErr(NULL, "some error occurred");
    return (success ? 0 : 1);
}

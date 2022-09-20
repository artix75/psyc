#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "../psyc.h"
#include "char_training_data.h"

#define EPOCHS 300
#define LEARNING_RATE 0.0025
//#define LEARNING_RATE 0.01
#define BATCHES 1

#define strEq(s1,s2) (strcmp(s1, s2) == 0)
#define UNUSED(V) ((void) V)

void CompleteText (PSNeuralNetwork * network, char* text, int len,
                   float randomicity, int max_words)
{
    int i = 0, wcount = 0;
    double inputs[len + 1];
    if (text == NULL) {
        srand ( time(NULL) - i);
        inputs[0] = 1.0;
        inputs[1] = (double)(rand() % INPUT_SIZE);
        printf("\nSample:\n%s", characters[(int) inputs[1]]);
    } else {
        int txtlen = strlen(text), j;
        if (txtlen >= len) {
            fprintf(stderr, "text legth >= length!\n");
            return;
        }
        inputs[0] = (double) txtlen;
        printf("\nSample:\n");
        for (i = 0; i < txtlen; i++) {
            char c = text[i];
            if (c == 0) break;
            int idx = -1;
            for (j = 0; j < INPUT_SIZE; j++) {
                char * chr = characters[j];
                if (chr[0] == c) {
                    idx = j;
                    break;
                }
            }
            if (idx < 0) {
                char invalid[2];
                invalid[1] = 0;
                invalid[0] = c;
                fprintf(stderr, "Invalid char: %s\n", invalid);
                return;
            }
            printf("%s", characters[idx]);
            inputs[i + 1] = (double) idx;
        }
    }
    char last_char = 0;
    int is_sep = 0;
    for (i = 0; i < len - 1; i++) {
        srand ( time(NULL) + i);
        float p = (rand() % 10) / 10.0f;
        int idx = PSClassify(network, inputs);
        is_sep = (last_char == ' ' || last_char == ',' || last_char == '.');
        if (max_words > 0 && is_sep && (++wcount > max_words)) break;
        if (p <= randomicity && is_sep) {
            PSLayer * out = network->layers[network->size - 1];
            int o = 0;
            double omax = 0.0;
            int oidx = 0;
            for (; o < out->size; o++) {
                if (o == idx) continue;
                double a = out->neurons[o]->activation;
                if (a > omax) {
                    omax = a;
                    oidx = o;
                }
            }
            idx = oidx;
            //printf("RND(%.2f):(%s)", p, characters[idx]);
            //idx = (rand() % INPUT_SIZE);
        }
        if (idx >= INPUT_SIZE) {
            fprintf(stderr, "Index %d >= %d", idx, INPUT_SIZE);
            return;
        }
        printf("%s", characters[idx]);
        inputs[0] += 1.0;
        last_char = characters[idx][0];
        inputs[(int) inputs[0]] = (double) idx;
    }
    printf("\n");
}

void TrainCallback (void * _net, int epoch, int epochs, double loss,
                    double previous_loss, float accuracy,
                    double * rate, double *training_data)
{
    UNUSED(epoch);
    UNUSED(epochs);
    UNUSED(loss);
    UNUSED(previous_loss);
    UNUSED(accuracy);
    UNUSED(rate);
    UNUSED(training_data);
    //if ((epoch % 2) != 0) return;
    PSNeuralNetwork * network = (PSNeuralNetwork*) _net;
    CompleteText(network, NULL, 255, 2.0f, 0);
}

int main(int argc, char**argv){
    PSNeuralNetwork * network = PSCreateNetwork("TEST CHAR RNN");
    network->onEpochTrained = TrainCallback;
    
    int epochs = EPOCHS;
    int batch_size = BATCHES;
    double learning_rate = LEARNING_RATE;
    double l2_decay = 0.0;
    PSLayerType type = LSTM;
    char * save_to = NULL;
    char * load_from = NULL;
    char * complete_text = NULL;
    float randomicity = 2.0f;
    int max_words = 0;
    int hidden_size = INPUT_SIZE / 2;
    /*double * vdataset = validation_data;
    int vdlen = EVAL_DATALEN;
    double * tdataset = test_data;
    int tdlen = TEST_DATALEN;*/
    int pretest = 0 ;
    
    int i;
    for (i = 0; i < argc; i++) {
        char * arg = argv[i];
        int next_idx = i + 1;
        if (strEq("--use-rnn", arg)) type = Recurrent;
        /*if (strEq("--same-dataset", arg)) {
            vdataset = training_data;
            vdlen = TRAIN_DATALEN;
            tdataset = training_data;
            tdlen = TRAIN_DATALEN;
        }*/
        if (strEq("--pre-test", arg)) pretest = 1;
        if (next_idx < argc) {
            char * next = argv[next_idx];
            if (strEq("--epochs", arg) || strEq("-e", arg)) {
                epochs = atoi(next);
                if (!epochs) {
                    fputs("Invalid epochs!", stderr);
                    return 1;
                }
            }
            if (strEq("--batch-size", arg) || strEq("-b", arg)) {
                batch_size = atoi(next);
                if (!batch_size) {
                    fputs("Invalid batch-size!", stderr);
                    return 1;
                }
            }
            if (strEq("--hidden-size", arg) || strEq("-s", arg)) {
                hidden_size = atoi(next);
                if (!hidden_size) {
                    fputs("Invalid hidden-size!", stderr);
                    return 1;
                }
            }
            if (strEq("--learning-rate", arg) || strEq("-r", arg)) {
                learning_rate = (double) atof(next);
                if (learning_rate == 0.0) {
                    fputs("Invalid learing rate!", stderr);
                    return 1;
                }
            }
            if (strEq("--l2-decay", arg))
                l2_decay = (double) atof(next);
            if (strEq("--save", arg))
                save_to = next;
            if (strEq("--load", arg))
                load_from = next;
            if (strEq("--complete", arg))
                complete_text = next;
            if (strEq("--randomicity", arg)) {
                int matched = sscanf(next, "%f", &randomicity);
                if (!matched)
                    fprintf(stderr, "Invalid randomicity %s\n", next);
            }
            if (strEq("--max-words", arg)) {
                int matched = sscanf(next, "%d", &max_words);
                if (!matched)
                    fprintf(stderr, "Invalid max-words %s\n", next);
            }
        }
    }
    //printf("CHAR: %s\n", characters[6]);return 0;
    network->flags |= FLAG_ONEHOT;
    
    PSAddLayer(network, FullyConnected, INPUT_SIZE, NULL);
    PSAddLayer(network, type, hidden_size, NULL);
    PSAddLayer(network, SoftMax, INPUT_SIZE, NULL);

    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
    
    if (load_from != NULL) {
        PSLoadNetwork(network, load_from);
        if (complete_text)
            CompleteText(network, complete_text, 255, randomicity, max_words);
    }
    else {
        printf("Epochs: %d\n", epochs);
        printf("Rate: %f\n", learning_rate);
        
        if (pretest) {
            PSTest(network, training_data, TRAIN_DATALEN);
            TrainCallback (network, 0, 0, 0.0,
                           0.0, 0.0,
                           NULL, NULL);
        }
        //epochs = 2;
        PSTrainingOptions options = {
            .flags = TRAINING_NO_SHUFFLE,
            .l2_decay = l2_decay
        };
        printf("L2 Decay: %.2f\n", (float) l2_decay);
        PSTrain(network, training_data, TRAIN_DATALEN, epochs, learning_rate,
                batch_size, &options, training_data, TRAIN_DATALEN);
        
        PSTest(network, training_data, TRAIN_DATALEN);
        
        if (save_to != NULL)
            PSSaveNetwork(network, save_to);
    }

    PSDeleteNetwork(network);
    return 0;
}

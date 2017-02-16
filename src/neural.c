#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "neural.h"

static unsigned char randomSeeded = 0;

typedef struct {
    double bias;
    double * weights;
} Delta;

/* Activation functions */

double sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double sigmoid_prime(double val) {
    double s = sigmoid(val);
    return s * (1 -s);
}

/* Feedforward functions */

void fullFeedforward(NeuralNetwork * net, Layer * layer) {
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    Layer * previous = net->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, previous_size = previous->size;
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        double sum = 0;
        for (j = 0; j < previous_size; j++) {
            Neuron * prev_neuron = previous->neurons[j];
            assert(prev_neuron != NULL);
            double a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        neuron->activation = layer->activate(neuron->z_value);
    }
}

/* Utils */

static double normalized_random() {
    if (!randomSeeded) {
        randomSeeded = 1;
        srand(time(NULL));
    }
    int r = rand();
    return ((double) r / (double) RAND_MAX);
}

static double gaussian_random(double mean, double stddev) {
    double theta = 2 * M_PI * normalized_random();
    double rho = sqrt(-2 * log(1 - normalized_random()));
    double scale = stddev * rho;
    double x = mean + scale * cos(theta);
    double y = mean + scale * sin(theta);
    double r = normalized_random();
    return (r > 0.5 ? y : x);
}

static void shuffle ( double * array, int size, int element_size )
{

    srand ( time(NULL) );
    int byte_size = element_size * sizeof(double);
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        //printf("Shuffle cycle %d: random is %d\n", i, j);
        double tmp_a[element_size];
        double tmp_b[element_size];
        int idx_a = i * element_size;
        int idx_b = j * element_size;
        //printf("-> idx_a: %d\n", idx_a);
        //printf("-> idx_b: %d\n", idx_b);
        memcpy(tmp_a, array + idx_a, byte_size);
        memcpy(tmp_b, array + idx_b, byte_size);
        memcpy(array + idx_a, tmp_b, byte_size);
        memcpy(array + idx_b, tmp_a, byte_size);
    }
}

int arrayMax(double * array, int len) {
    int i;
    double max = 0;
    int max_idx = 0;
    for (i = 0; i < len; i++) {
        double v = array[i];
        if (v > max) {
            max = v;
            max_idx = i;
        }
    }
    return max_idx;
}

/* NN functions */

NeuralNetwork * createNetwork() {
    NeuralNetwork *network = (malloc(sizeof(NeuralNetwork)));
    network->size = 0;
    network->layers = NULL;
    network->input_size = 0;
    network->output_size = 0;
    network->status = STATUS_UNTRAINED;
    network->current_epoch = 0;
    return network;
}

int loadNetwork(NeuralNetwork * network, const char* filename) {
    FILE * f = fopen(filename, "r");
    printf("Loading network from %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s!\n", filename);
        return 0;
    }
    int netsize, i, j, k;
    int empty = (network->size == 0);
    int matched = fscanf(f, "%d:", &netsize);
    if (!matched) {
        fprintf(stderr, "Invalid file %s!\n", filename);
        return 0;
    }
    if (empty && network->size != netsize) {
        fprintf(stderr, "Network size differs!\n");
        return 0;
    }
    char sep[] = ",";
    char eol[] = "\n";
    Layer * layer = NULL;
    for (i = 0; i < netsize; i++) {
        int lsize = 0;
        char * last = (i == (netsize - 1) ? eol : sep);
        char fmt[5];
        sprintf(fmt, "%%d%s", last);
        matched = fscanf(f, fmt, &lsize);
        if (!matched) {
            fputs("Invalid header!\n", stderr);
            return 0;
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                fprintf(stderr, "Layer %d size %d differs from %d!\n", i,
                        layer->size, lsize);
                return 0;
            }
        } else {
            layer = addLayer(network, FullyConnected, lsize, NULL);
            if (layer == NULL) {
                fprintf(stderr, "Could not create layer %d\n", i);
                return 0;
            }
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = layer->size;
        for (j = 0; j < lsize; j++) {
            double bias = 0;
            Neuron * neuron = layer->neurons[j];
            int wsize = neuron->weights_size;
            matched = fscanf(f, "%lf|", &bias);
            if (!matched) {
                fprintf(stderr, "Layer %d, neuron %d: invalid bias!\n", i, j);
                return 0;
            }
            neuron->bias = bias;
            for (k = 0; k < wsize; k++) {
                double w = 0;
                char * last = (k == (wsize - 1) ? eol : sep);
                char fmt[5];
                sprintf(fmt, "%%lf%s", last);
                matched = fscanf(f, fmt, &w);
                if (!matched) {
                    fprintf(stderr,"\nLayer %d neuron %d: invalid weight[%d]\n",
                            i, j, k);
                    return 0;
                }
                neuron->weights[k] = w;
                printf("\rLoading layer %d, neuron %d                ", i, j);
                fflush(stdout);
            }
        }
    }
    printf("\n");
    fclose(f);
    return 1;
}

void deleteNetwork(NeuralNetwork * network) {
    int size = network->size;
    int i;
    for (i = 0; i < size; i++) {
        Layer * layer = network->layers[i];
        deleteLayer(layer);
    }
    free(network->layers);
    free(network);
}

void deleteNeuron(Neuron * neuron) {
    if (neuron->weights != NULL) free(neuron->weights);
    free(neuron);
}

Layer * addLayer(NeuralNetwork * network, LayerType type, int size,
                 LayerParameters* params) {
    if (network->size == 0 && type != FullyConnected) {
        fprintf(stderr, "First layer type must be FullyConnected\n");
        return NULL;
    }
    Layer * layer = malloc(sizeof(Layer));
    layer->index = network->size++;
    layer->type = type;
    layer->size = size;
    Layer * previous = NULL;
    int previous_size = 0;
    printf("Adding layer %d\n", layer->index);
    if (network->layers == NULL) {
        network->layers = malloc(sizeof(Layer*));
        network->input_size = size;
    } else {
        network->layers = realloc(network->layers,
                                  sizeof(Layer*) * network->size);
        previous = network->layers[layer->index - 1];
        assert(previous != NULL);
        previous_size = previous->size;
        network->output_size = size;
    }
    layer->neurons = malloc(sizeof(Neuron*) * size);
    int i, j;
    for (i = 0; i < size; i++) {
        Neuron * neuron = malloc(sizeof(Neuron));
        neuron->index = 0;
        if (layer->index > 0) {
            neuron->weights_size = previous_size;
            neuron->bias = gaussian_random(0, 1);
            neuron->weights = malloc(sizeof(double) * previous_size);
            for (j = 0; j < previous_size; j++) {
                neuron->weights[j] = gaussian_random(0, 1);
            }
        } else {
            neuron->bias = 0;
            neuron->weights_size = 0;
            neuron->weights = NULL;
        }
        neuron->activation = 0;
        neuron->z_value = 0;
        //printf("Adding neuron %d\n", i);
        layer->neurons[i] = neuron;
    }
    if (type == FullyConnected) {
        layer->activate = sigmoid;
        layer->prime = sigmoid_prime;
        layer->feedforward = fullFeedforward;
    } else if (type == Convolutional) {
        layer->activate = sigmoid;
        // TODO
    }
    network->layers[layer->index] = layer;
    return layer;
}

void deleteLayer(Layer* layer) {
    int size = layer->size;
    int i;
    for (i = 0; i < size; i++) {
        Neuron* neuron = layer->neurons[i];
        deleteNeuron(neuron);
    }
    free(layer);
}

void feedforward(NeuralNetwork * network, double * values) {
    if (network->size == 0) {
        printf("Empty network");
        exit(1);
    }
    Layer * first = network->layers[0];
    int input_size = first->size;
    int i;
    for (i = 0; i < input_size; i++) {
        first->neurons[i]->activation = values[i];
    }
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        if (layer == NULL) {
            printf("Layer %d is NULL\n", i);
            exit(1);
        }
        if (layer->feedforward == NULL) {
            printf("Layer %d feedforward function is NULL\n", i);
            exit(1);
        }
        layer->feedforward(network, layer);
    }
}

Delta * emptyLayer(Layer * layer) {
    Delta * delta = malloc(sizeof(Delta) * layer->size);
    int i;
    for (i = 0; i < layer->size; i++) {
        Neuron * neuron = layer->neurons[i];
        int ws = neuron->weights_size;
        int memsize = sizeof(double) * ws;
        delta[i].weights = malloc(memsize);
        memset(delta[i].weights, 0, memsize);
        delta[i].bias = 0;
    }
    return delta;
}

Delta ** emptyDeltas(NeuralNetwork * network) {
    Delta ** deltas = malloc(sizeof(Delta*) * network->size - 1);
    int i;
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        deltas[i - 1] = emptyLayer(layer);
    }
    return deltas;
}

void deleteDelta(Delta * delta, int size) {
    int i;
    for (i = 0; i < size; i++) {
        Delta d = delta[i];
        free(d.weights);
    }
    free(delta);
}

void deleteDeltas(Delta ** deltas, NeuralNetwork * network) {
    int i;
    for (i = 1; i < network->size; i++) {
        Delta * delta = deltas[i - 1];
        Layer * layer = network->layers[i];
        deleteDelta(delta, layer->size);
    }
    free(deltas);
}

Delta ** backprop(NeuralNetwork * network, double * x, double * y) {
    Delta ** deltas = emptyDeltas(network);
    int netsize = network->size;
    Layer * inputLayer = network->layers[0];
    Layer * outputLayer = network->layers[netsize - 1];
    int isize = inputLayer->size;
    int osize = outputLayer->size;
    Delta * layer_delta = deltas[netsize - 2]; // Deltas have no input layer
    Layer * previousLayer = network->layers[outputLayer->index - 1];
    Layer * nextLayer = NULL;
    double * delta_v;
    double * last_delta_v;
    delta_v = malloc(sizeof(double) * osize);
    memset(delta_v, 0, sizeof(double) * osize);
    last_delta_v = delta_v;
    int i, o, w, j, k;
    feedforward(network, x);
    for (o = 0; o < osize; o++) {
        Neuron * neuron = outputLayer->neurons[o];
        double o_val = neuron->activation;
        double y_val = y[o];
        double d = o_val - y_val;
        d *= outputLayer->prime(neuron->z_value);
        delta_v[o] = d;
        Delta * n_delta = &(layer_delta[o]);
        n_delta->bias = d;
        for (w = 0; w < neuron->weights_size; w++) {
            double prev_a = previousLayer->neurons[w]->activation;
            n_delta->weights[w] = d * prev_a;
        }
    }
    for (i = previousLayer->index; i > 0; i--) {
        Layer * layer = network->layers[i];
        previousLayer = network->layers[i - 1];
        nextLayer = network->layers[i + 1];
        layer_delta = deltas[i - 1];
        int lsize = layer->size;
        delta_v = malloc(sizeof(double) * lsize);
        memset(delta_v, 0, sizeof(double) * lsize);
        for (j = 0; j < lsize; j++) {
            Neuron * neuron = layer->neurons[j];
            double sum = 0;
            for (k = 0; k < nextLayer->size; k++) {
                Neuron * nextNeuron = nextLayer->neurons[k];
                double weight = nextNeuron->weights[j];
                double d = last_delta_v[k];
                sum += (d * weight);
            }
            double dv = sum * layer->prime(neuron->z_value);
            delta_v[j] = dv;
            Delta * n_delta = &(layer_delta[j]);
            n_delta->bias = dv;
            for (w = 0; w < neuron->weights_size; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                n_delta->weights[w] = dv * prev_a;
            }
        }
        free(last_delta_v);
        last_delta_v = delta_v;
    }
    free(delta_v);
    return deltas;
}

void updateWeights(NeuralNetwork * network, double * training_data,
                   int batch_size, double rate) {
    double r = rate / (double) batch_size;
    int i, j, k, w, netsize = network->size, dsize = netsize - 1;
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    int element_size = training_data_size + label_data_size;
    Delta ** deltas = emptyDeltas(network);
    //double * data_p = training_data;
    for (i = 0; i < batch_size; i++) {
        double * x = training_data;
        double * y = training_data + training_data_size;
        training_data += element_size;
        Delta ** bp_deltas = backprop(network, x, y);
        for (j = 0; j < dsize; j++) {
            Layer * layer = network->layers[j + 1];
            Delta * layer_delta_bp = bp_deltas[j];
            Delta * layer_delta = deltas[j];
            int lsize = layer->size;
            for (k = 0; k < lsize; k++) {
                Neuron * neuron = layer->neurons[k];
                Delta * n_delta_bp = &(layer_delta_bp[k]);
                Delta * n_delta = &(layer_delta[k]);
                n_delta->bias += n_delta_bp->bias;
                for (w = 0; w < neuron->weights_size; w++) {
                    n_delta->weights[w] += n_delta_bp->weights[w];
                }
            }
        }
        deleteDeltas(bp_deltas, network);
    }
    for (i = 0; i < dsize; i++) {
        Delta * l_delta = deltas[i];
        Layer * layer = network->layers[i + 1];
        int l_size = layer->size;
        for (j = 0; j < l_size; j++) {
            Delta * d = &(l_delta[j]);
            Neuron * neuron = layer->neurons[j];
            neuron->bias = neuron->bias - r * d->bias;
            //printf("Layer %d n. %d: bias delta %lf\n", layer->index, j, d->bias);
            for (k = 0; k < neuron->weights_size; k++) {
                double w = neuron->weights[k];
                neuron->weights[k] = w - r * d->weights[k];
                //printf("  -> weight[%d]: delta %lf\n", k, d->weights[k]);
                //printf("%.3lf ", d->weights[k]);
            }
            //printf("\n");
        }
    }
    deleteDeltas(deltas, network);
}

void gradientDescent(NeuralNetwork * network,
                     double * training_data,
                     int element_size,
                     int elements_count,
                     double learning_rate,
                     int batch_size) {
    int batches_count = elements_count / batch_size;
    shuffle(training_data, elements_count, element_size);
    int offset = (element_size * batch_size), i;
    for (i = 0; i < batches_count; i++) {
        printf("\rEpoch %d: batch %d/%d", network->current_epoch + 1,
               i + 1, batches_count);
        fflush(stdout);
        updateWeights(network, training_data, batch_size, learning_rate);
        training_data += offset;
    }
}

void train(NeuralNetwork * network,
           double * training_data,
           int data_size,
           int epochs,
           double learning_rate,
           int batch_size) {
    int i;
    int element_size = network->input_size + network->output_size;
    int elements_count = data_size / element_size;
    printf("Training data elements: %d\n", elements_count);
    network->status = STATUS_TRAINING;
    time_t start_t, end_t, epoch_t;
    char timestr[80];
    struct tm * tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    printf("Training started at %s\n", timestr);
    epoch_t = start_t;
    time_t e_t = epoch_t;
    for (i = 0; i < epochs; i++) {
        network->current_epoch = i;
        gradientDescent(network, training_data, element_size,
                        elements_count, learning_rate, batch_size);
        time(&epoch_t);
        time_t elapsed_t = epoch_t - e_t;
        e_t = epoch_t;
        printf(" (%ld sec.)\n", elapsed_t);
    }
    time(&end_t);
    printf("Completed in %ld sec.\n", end_t - start_t);
    network->status = STATUS_TRAINED;
}

float test(NeuralNetwork * network, double * test_data, int data_size) {
    int i, j;
    float accuracy = 0.0f;
    int correct_results = 0;
    int input_size = network->input_size;
    int output_size = network->output_size;
    int element_size = input_size + output_size;
    int elements_count = data_size / element_size;
    double outputs[output_size];
    Layer * output_layer = network->layers[network->size - 1];
    printf("Test data elements: %d\n", elements_count);
    time_t start_t, end_t;
    char timestr[80];
    struct tm * tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    printf("Testing started at %s\n", timestr);
    for (i = 0; i < elements_count; i++) {
        printf("\rTesting %d/%d                ", i + 1, elements_count);
        fflush(stdout);
        double * inputs = test_data;
        test_data += input_size;
        double * expected = test_data;
        feedforward(network, inputs);
        for (j = 0; j < output_size; j++) {
            Neuron * neuron = output_layer->neurons[j];
            outputs[j] = neuron->activation;
        }
        int omax = arrayMax(outputs, output_size);
        int emax = arrayMax(expected, output_size);
        if (omax == emax) correct_results++;
        test_data += output_size;
    }
    printf("\n");
    time(&end_t);
    printf("Completed in %ld sec.\n", end_t - start_t);
    accuracy = (float) correct_results / (float) elements_count;
    printf("Accuracy (%d/%d): %.2f\n", correct_results, elements_count,
           accuracy);
    return accuracy;
}

/* Test */

void testShuffle(double * array, int size, int element_size) {
    printf("Original array:\n");
    //int total_size = size * element_size;
    int i, j;
    for (i = 0; i < size; i++) {
        int offset = i * element_size;
        for (j = offset; j <  offset + element_size; j++) {
            printf("%f ", array[j]);
        }
        printf("\n");
    }
    shuffle(array, size, element_size);
    printf("Shuffled array:\n");
    for (i = 0; i < size; i++) {
        int offset = i * element_size;
        for (j = offset; j <  offset + element_size; j++) {
            printf("%f ", array[j]);
        }
        printf("\n");
    }
}


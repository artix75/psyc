/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "neural.h"

#ifdef USE_AVX
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#endif

#define calculateConvolutionalSide(s,rs,st,pad) ((s - rs + 2 * pad) / st + 1)
#define calculatePoolingSide(s, rs) ((s - rs) / rs + 1)
#define getColumn(index, width) (index % width)
#define getRow(index, width) ((int) ((int) index / (int) width))

static unsigned char randomSeeded = 0;

/* Function Prototypes */

RecurrentCell * createRecurrentCell(int lsize);
void addRecurrentState(Neuron * neuron, double state, int times, int t);

/* Activation functions */

double sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double sigmoid_prime(double val) {
    double s = sigmoid(val);
    return s * (1 - s);
}

double relu(double val) {
    return (val >= 0.0 ? val : 0.0);
}

double relu_prime(double val) {
    return (double)(val > 0.0);
}

double tanh_prime(double val) {
    return (1 - (val * val));
}

/* Feedforward functions */

void fullFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * network = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    Layer * previous = network->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, previous_size = previous->size;
    int is_recurrent = (network->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
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
        if (is_recurrent)
            addRecurrentState(neuron, neuron->activation, times, t);
    }
}

void softmaxFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    Layer * previous = net->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, previous_size = previous->size;
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    double max = 0.0, esum = 0.0;
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
        if (i == 0)
            max = neuron->z_value;
        else if (neuron->z_value > max)
            max = neuron->z_value;
        //printf("[%d] -> %lf\n", i, neuron->z_value);
    }
    //printf("Max: %lf\n", max);
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        double z = neuron->z_value;
        double e = exp(z - max);
        esum += e;
        neuron->activation = e;
        //printf("[%d] -> %lf\n", i, neuron->activation);
    }
    //printf("Sum: %lf\n", esum);
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        neuron->activation /= esum;
        //printf("[%d] -> %lf\n", i, neuron->activation);
        if (is_recurrent)
            addRecurrentState(neuron, neuron->activation, times, t);
    }
}

void convolve(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    Layer * previous = net->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, x, y, row, col, previous_size = previous->size;
    LayerParameters * parameters = layer->parameters;
    LayerParameters * previous_parameters = previous->parameters;
    assert(parameters != NULL);
    assert(previous_parameters != NULL);
    double * params = parameters->parameters;
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[FEATURE_COUNT]);
    int stride = (int) (params[STRIDE]);
    double region_size = params[REGION_SIZE];
    double region_area = region_size * region_size;
    double input_w = previous_params[OUTPUT_WIDTH];
    double input_h = previous_params[OUTPUT_HEIGHT];
    double output_w = params[OUTPUT_WIDTH];
    double output_h = params[OUTPUT_HEIGHT];
    int feature_size = layer->size / feature_count;
    ConvolutionalSharedParams * shared;
    shared = (ConvolutionalSharedParams*) layer->extra;
    assert(shared != NULL);
    for (i = 0; i < feature_count; i++) {
        double bias = shared->biases[i];
        double * weights = shared->weights[i];
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            Neuron * neuron = layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * stride;
            int r_col = col * stride;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            double sum = 0;
            int widx = 0;
            //printf("Neuron %d,%d: r: %d, b: %d\n", col, row, max_x, max_y);
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = (y * input_w) + x;
                    //printf("  -> %d,%d [%d]\n", x, y, nidx);
                    Neuron * prev_neuron = previous->neurons[nidx];
                    double a = prev_neuron->activation;
                    sum += (a * weights[widx++]);
                }
            }
            neuron->z_value = sum + bias;
            neuron->activation = layer->activate(neuron->z_value);
        }
    }
}

void pool(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    //assert(layer->type == Pooling);
    Layer * previous = net->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, x, y, row, col, previous_size = previous->size;
    LayerParameters * parameters = layer->parameters;
    LayerParameters * previous_parameters = previous->parameters;
    assert(parameters != NULL);
    assert(previous_parameters != NULL);
    double * params = parameters->parameters;
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[FEATURE_COUNT]);
    double region_size = params[REGION_SIZE];
    double region_area = region_size * region_size;
    double input_w = previous_params[OUTPUT_WIDTH];
    double input_h = previous_params[OUTPUT_HEIGHT];
    double output_w = params[OUTPUT_WIDTH];
    double output_h = params[OUTPUT_HEIGHT];
    int feature_size = layer->size / feature_count;
    int prev_size = previous->size / feature_count;
    for (i = 0; i < feature_count; i++) {
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            Neuron * neuron = layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * region_size;
            int r_col = col * region_size;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            double max = 0.0, max_z = 0.0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size * i);
                    Neuron * prev_neuron = previous->neurons[nidx];
                    double a = prev_neuron->activation;
                    double z = prev_neuron->z_value;
                    if (a > max) {
                        max = a;
                        max_z = z;
                    }
                }
            }
            neuron->z_value = max_z;
            neuron->activation = max;
        }
    }
}

void recurrentFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    va_list args;
    va_start(args, _layer);
    int times = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    assert(times > 0);
    int size = layer->size;
    assert(layer->neurons != NULL);
    assert(layer->index > 0);
    Layer * previous = net->layers[layer->index - 1];
    assert(previous != NULL);
    int i, j, w, previous_size = previous->size;
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        RecurrentCell * cell = (RecurrentCell*) neuron->extra;
        assert(cell != NULL);
        double sum = 0, bias = 0;
        if (previous->flags & FLAG_ONEHOT) {
            LayerParameters * params = (LayerParameters*) layer->extra;
            assert(params != NULL && params->count > 0);
            int vector_size = (int) (params->parameters[0]);
            assert(vector_size > 0);
            Neuron * prev_neuron = previous->neurons[0];
            int a = (int) (prev_neuron->activation);
            assert(a < vector_size);
            sum = neuron->weights[a];
        } else {
            for (j = 0; j < previous_size; j++) {
                Neuron * prev_neuron = previous->neurons[j];
                assert(prev_neuron != NULL);
                double a = prev_neuron->activation;
                sum += (a * neuron->weights[j]);
            }
        }
        if (t > 0) {
            int last_t = t - 1;
            for (w = 0; w < cell->weights_size; w++) {
                Neuron * n = layer->neurons[w];
                RecurrentCell * rc = (RecurrentCell*) n->extra;
                assert(rc != NULL);
                double weight = cell->weights[w];
                double last_state = rc->states[last_t];
                bias += (weight * last_state);
            }
        } else {
            if (cell->states != NULL) free(cell->states);
            cell->states_count = times;
            cell->states = malloc(times * sizeof(double));
        }
        neuron->z_value = sum + bias;
        neuron->activation = layer->activate(neuron->z_value);
        cell->states[t] = neuron->activation;
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

double norm(double* matrix, int size) {
    double r = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        double v = matrix[i];
        r += (v * v);
    }
    return sqrt(r);
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

void addRecurrentState(Neuron * neuron, double state, int times, int t) {
    RecurrentCell * cell = (RecurrentCell*) neuron->extra;
    if (cell == NULL) {
        cell = createRecurrentCell(0);
        neuron->extra = cell;
    }
    if (t == 0) {
        cell->states_count = times;
        if (cell->states != NULL) free(cell->states);
        cell->states = malloc(times * sizeof(double));
    }
    cell->states[t] = state;
}

static double ** getRecurrentSeries(double * array, int series_count,
                                    int x_size, int y_size) {
    double ** series = malloc(series_count * sizeof(double**));
    if (series == NULL) {
        return NULL;
    }
    int i;
    double * p = array;
    for (i = 0; i < series_count; i++) {
        int series_size = (int) (p[0]);
        if (!series_size) {
            fprintf(stderr, "Invalid series size 0 at %d", (int) (p - array));
            return NULL;
        }
        series[i] = p;
        p += ((series_size * x_size) + (series_size * y_size));
    }
    return series;
}

static void shuffleSeries ( double ** series, int size)
{
    srand ( time(NULL) );
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        //printf("Shuffle cycle %d: random is %d\n", i, j);
        double * tmp_a = series[i];
        double * tmp_b = series[j];
        series[i] = tmp_b;
        series[j] = tmp_a;
    }
}

int arrayMaxIndex(double * array, int len) {
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

double arrayMax(double * array, int len) {
    int i;
    double max = 0;
    for (i = 0; i < len; i++) {
        double v = array[i];
        if (v > max) {
            max = v;
        }
    }
    return max;
}

char * getLayerTypeLabel(Layer * layer) {
    switch (layer->type) {
        case FullyConnected:
            return "fully_connected";
            break;
        case Convolutional:
            return "convolutional";
            break;
        case Pooling:
            return "pooling";
            break;
        case Recurrent:
            return "recurrent";
            break;
        case LSTM:
            return "lstm";
            break;
        case SoftMax:
            return "softmax";
            break;
    }
    return "UNKOWN";
}

void printLayerInfo(Layer * layer) {
    LayerType ltype = layer->type;
    char * type_name = getLayerTypeLabel(layer);
    LayerParameters * lparams = layer->parameters;
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
        double * params = lparams->parameters;
        int fcount = (int) (params[FEATURE_COUNT]);
        int rsize = (int) (params[REGION_SIZE]);
        int input_w = (int) (params[INPUT_WIDTH]);
        int input_h = (int) (params[INPUT_HEIGHT]);
        int stride = (int) (params[STRIDE]);
        int use_relu = (int) (params[USE_RELU]);
        char * actv = (use_relu ? "relu" : "sigmoid");
        printf(", input size = %dx%d, features = %d", input_w, input_h, fcount);
        printf(", region = %dx%d, stride = %d, activation = %s\n",
               rsize, rsize, stride, actv);
    } else printf("\n");
}

/* Loss functions */

static double loss(double * outputs, double * desired, int size) {
    int i;
    double diffs[size];
    for (i = 0; i < size; i++) {
        double d = outputs[i] - desired[i];
        diffs[i] = d;
    }
    double n = norm(diffs, size);
    return 0.5 * (n * n);
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
    network->current_batch = 0;
    network->flags = FLAG_NONE;
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
        fclose(f);
        return 0;
    }
    if (!empty && network->size != netsize) {
        fprintf(stderr, "Network size differs!\n");
        fclose(f);
        return 0;
    }
    char sep[] = ",";
    char eol[] = "\n";
    Layer * layer = NULL;
    for (i = 0; i < netsize; i++) {
        int lsize = 0;
        LayerType ltype = FullyConnected;
        int args[20];
        int argc = 0, aidx = 0;
        char * last = (i == (netsize - 1) ? eol : sep);
        char fmt[50];
        char buff[255];
        sprintf(fmt, "%%d%s", last);
        //fputs(fmt, stderr);
        matched = fscanf(f, fmt, &lsize);
        if (!matched) {
            int type = 0, arg = 0;
            argc = 0;
            matched = fscanf(f, "[%d,%d", &type, &argc);
            if (!matched) {
                fprintf(stderr, "Invalid header: layer[%d], col. %ld!\n",
                        i, ftell(f));
                fclose(f);
                return 0;
            }
            if (argc == 0) {
                fputs("Layer must have at least 1 argument (size)\n", stderr);
                fclose(f);
                return 0;
            }
            ltype = (LayerType) type;
            for (aidx = 0; aidx < argc; aidx++) {
                matched = fscanf(f, ",%d", &arg);
                if (!matched) {
                    fprintf(stderr, "Invalid header: l%d, arg. %d, col. %ld!\n",
                            i, aidx, ftell(f));
                    fclose(f);
                    return 0;
                }
                if (aidx == 0) lsize = arg;
                else args[aidx - 1] = arg;
            }
            argc--;
            sprintf(fmt, "]%s", last);
            fscanf(f, fmt, buff);
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                fprintf(stderr, "Layer %d size %d differs from %d!\n", i,
                        layer->size, lsize);
                fclose(f);
                return 0;
            }
            if (ltype != layer->type) {
                fprintf(stderr, "Layer %d type %d differs from %d!\n", i,
                        (int) (layer->type), (int) ltype);
                fclose(f);
                return 0;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                LayerParameters * params = layer->parameters;
                assert(params != NULL);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= params->count) break;
                    int arg = args[aidx];
                    double val = params->parameters[aidx];
                    if (arg != (int) val) {
                        fprintf(stderr, "Layer %d arg[%d] %d diff. from %d!\n",
                                i, aidx,(int) val, arg);
                        fclose(f);
                        return 0;
                    }
                }
            }
        } else {
            layer = NULL;
            if (ltype == FullyConnected || ltype == SoftMax) {
                layer = addLayer(network, ltype, lsize, NULL);
            } else if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                LayerParameters * params = createLayerParamenters(param_c);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= param_c) break;
                    int arg = args[aidx];
                    params->parameters[aidx] = (double) arg;
                }
                layer = addLayer(network, ltype, lsize, params);
            }
            if (layer == NULL) {
                fprintf(stderr, "Could not create layer %d\n", i);
                fclose(f);
                return 0;
            }
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = 0;
        ConvolutionalSharedParams * shared = NULL;
        if (layer->type == FullyConnected || layer->type == SoftMax) {
            lsize = layer->size;
        } else if (layer->type == Convolutional) {
            shared = (ConvolutionalSharedParams*) layer->extra;
            if (shared == NULL) {
                fprintf(stderr, "Layer %d, missing shared params!\n", i);
                fclose(f);
                return 0;
            }
            lsize = shared->feature_count;
        } else if (layer->type == Pooling) {
            continue;
        }
        for (j = 0; j < lsize; j++) {
            double bias = 0;
            int wsize = 0;
            double * weights = NULL;
            matched = fscanf(f, "%lf|", &bias);
            if (!matched) {
                fprintf(stderr, "Layer %d, neuron %d: invalid bias!\n", i, j);
                fclose(f);
                return 0;
            }
            if (shared == NULL) {
                Neuron * neuron = layer->neurons[j];
                wsize = neuron->weights_size;
                neuron->bias = bias;
                weights = neuron->weights;
            } else {
                shared->biases[j] = bias;
                wsize = shared->weights_size;
                weights = shared->weights[j];
            }
            for (k = 0; k < wsize; k++) {
                double w = 0;
                char * last = (k == (wsize - 1) ? eol : sep);
                char fmt[5];
                sprintf(fmt, "%%lf%s", last);
                matched = fscanf(f, fmt, &w);
                if (!matched) {
                    fprintf(stderr,"\nLayer %d neuron %d: invalid weight[%d]\n",
                            i, j, k);
                    fclose(f);
                    return 0;
                }
                weights[k] = w;
                printf("\rLoading layer %d, neuron %d   ", i, j);
                fflush(stdout);
            }
        }
    }
    printf("\n");
    fclose(f);
    return 1;
}

int saveNetwork(NeuralNetwork * network, const char* filename) {
    if (network->size == 0) {
        fprintf(stderr, "Empty network!\n");
        return 0;
    }
    FILE * f = fopen(filename, "w");
    printf("Saving network to %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    int netsize, i, j, k;
    fprintf(f, "%d:", network->size);
    for (i = 0; i < network->size; i++) {
        Layer * layer = network->layers[i];
        LayerType ltype = layer->type;
        if (i > 0) fprintf(f, ",");
        if (FullyConnected == ltype) fprintf(f, "%d", layer->size);
        else if (Convolutional == ltype || Pooling == ltype){
            LayerParameters * params = layer->parameters;
            int argc = params->count;
            fprintf(f, "[%d,%d,%d", (int) ltype, 1 + argc, layer->size);
            for (j = 0; j < argc; j++) {
                fprintf(f, ",%d", (int) (params->parameters[j]));
            }
            fprintf(f, "]");
        } else {
            fprintf(f, "[%d,1,%d]", (int) ltype, layer->size);
        }
    }
    fprintf(f, "\n");
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        LayerType ltype = layer->type;
        int lsize = layer->size;
        if (FullyConnected == ltype || SoftMax == ltype) {
            for (j = 0; j < lsize; j++) {
                Neuron * neuron = layer->neurons[j];
                fprintf(f, "%.15e|", neuron->bias);
                for (k = 0; k < neuron->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    double w = neuron->weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        } else if (Convolutional == ltype) {
            LayerParameters * params = layer->parameters;
            ConvolutionalSharedParams * shared;
            shared = (ConvolutionalSharedParams*) layer->extra;
            assert(shared != NULL);
            int feature_count = shared->feature_count;
            assert(feature_count > 0);
            for (j = 0; j < feature_count; j++) {
                double bias = shared->biases[j];
                double * weights = shared->weights[j];
                fprintf(f, "%.15e|", bias);
                for (k = 0; k < shared->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    double w = weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        } else if (Pooling == ltype) continue;
    }
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

void deleteNeuron(Neuron * neuron, Layer * layer) {
    if (neuron->weights != NULL) free(neuron->weights);
    if (neuron->extra != NULL) {
        if (layer->flags & FLAG_RECURRENT) {
            RecurrentCell * cell = (RecurrentCell*) neuron->extra;
            if (cell->states != NULL) free(cell->states);
            if (cell->weights != NULL) free(cell->weights);
            free(cell);
        } else free(neuron->extra);
    }
    free(neuron);
}

void abortLayer(NeuralNetwork * network, Layer * layer) {
    if (layer->index == (network->size - 1)) {
        network->size--;
        deleteLayer(layer);
    }
}

int initConvolutionalLayer(NeuralNetwork * network, Layer * layer,
                           LayerParameters * parameters) {
    int index = layer->index;
    Layer * previous = network->layers[index - 1];
    if (previous->type != FullyConnected) {
        fprintf(stderr,
                "FullyConnected -> Convolutional trans. not supported ATM :(");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        fputs("Layer parameters is NULL!\n", stderr);
        abortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        fprintf(stderr, "Convolutional Layer parameters count must be %d\n",
                CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    int feature_count = (int) (params[FEATURE_COUNT]);
    if (feature_count <= 0) {
        fprintf(stderr, "FEATURE_COUNT must be > 0 (given: %d)", feature_count);
        abortLayer(network, layer);
        return 0;
    }
    double region_size = params[REGION_SIZE];
    if (region_size <= 0) {
        fprintf(stderr, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        abortLayer(network, layer);
        return 0;
    }
    int previous_size = previous->size;
    LayerParameters * previous_params = previous->parameters;
    double input_w, input_h, output_w, output_h;
    int use_relu = (int) (params[USE_RELU]);
    if (previous_params == NULL) {
        double w = sqrt(previous_size);
        input_w = w; input_h = w;
        previous_params = createConvolutionalParameters(1, 0, 0, 0, 0);
        previous_params->parameters[OUTPUT_WIDTH] = input_w;
        previous_params->parameters[OUTPUT_HEIGHT] = input_h;
        previous->parameters = previous_params;
    } else {
        double input_w, input_h, prev_area;
        input_w = previous_params->parameters[OUTPUT_WIDTH];
        input_h = previous_params->parameters[OUTPUT_HEIGHT];
        prev_area = input_w * input_h;
        if ((int) prev_area != previous_size) {
            fprintf(stderr, "Previous size %d != %lfx%lf\n",
                    previous_size, input_w, input_h);
            abortLayer(network, layer);
            return 0;
        }
    }
    params[INPUT_WIDTH] = input_w;
    params[INPUT_HEIGHT] = input_h;
    int stride = (int) params[STRIDE];
    int padding = (int) params[PADDING];
    if (stride == 0) stride = 1;
    output_w =  calculateConvolutionalSide(input_w, region_size,
                                           (double) stride, (double) padding);
    output_h =  calculateConvolutionalSide(input_h, region_size,
                                           (double) stride, (double) padding);
    params[OUTPUT_WIDTH] = output_w;
    params[OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(Neuron*) * size);
    ConvolutionalSharedParams * shared;
    shared = malloc(sizeof(ConvolutionalSharedParams));
    shared->feature_count = feature_count;
    shared->weights_size = (int)(region_size * region_size);
    shared->biases = malloc(feature_count * sizeof(double));
    shared->weights = malloc(feature_count * sizeof(double*));
    layer->extra = shared;
    int i, j, w;
    for (i = 0; i < feature_count; i++) {
        shared->biases[i] = gaussian_random(0, 1);
        shared->weights[i] = malloc(shared->weights_size * sizeof(double));
        for (w = 0; w < shared->weights_size; w++) {
            shared->weights[i][w] = gaussian_random(0, 1);
        }
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            Neuron * neuron = malloc(sizeof(Neuron));
            neuron->index = idx;
            neuron->weights_size = shared->weights_size;
            neuron->bias = shared->biases[i];
            neuron->weights = shared->weights[i];
            layer->neurons[idx] = neuron;
        }
    }
    if (!use_relu) {
        layer->activate = sigmoid;
        layer->delta = sigmoid_prime;
    } else {
        layer->activate = relu;
        layer->delta = relu_prime;
    }
    layer->feedforward = convolve;
    return 1;
}

int initPoolingLayer(NeuralNetwork * network, Layer * layer,
                     LayerParameters * parameters) {
    int index = layer->index;
    Layer * previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        fprintf(stderr,
                "Pooling's previous layer must be a Convolutional layer!");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        fputs("Layer parameters is NULL!\n", stderr);
        abortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        fprintf(stderr, "Convolutional Layer parameters count must be %d\n",
                CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    LayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        fputs("Previous layer parameters is NULL!\n", stderr);
        abortLayer(network, layer);
        return 0;
    }
    if (previous_parameters->count < CONV_PARAMETER_COUNT) {
        fprintf(stderr, "Convolutional Layer parameters count must be %d\n",
                CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (previous_params[FEATURE_COUNT]);
    params[FEATURE_COUNT] = (double) feature_count;
    double region_size = params[REGION_SIZE];
    if (region_size <= 0) {
        fprintf(stderr, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        abortLayer(network, layer);
        return 0;
    }
    int previous_size = previous->size;
    double input_w, input_h, output_w, output_h;
    input_w = previous_params[OUTPUT_WIDTH];
    input_h = previous_params[OUTPUT_HEIGHT];
    params[INPUT_WIDTH] = input_w;
    params[INPUT_HEIGHT] = input_h;
    
    output_w = calculatePoolingSide(input_w, region_size);
    output_h = calculatePoolingSide(input_h, region_size);
    params[OUTPUT_WIDTH] = output_w;
    params[OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(Neuron*) * size);
    int i, j, w;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            Neuron * neuron = malloc(sizeof(Neuron));
            neuron->index = idx;
            neuron->weights_size = 0;
            neuron->bias = NULL_VALUE;
            neuron->weights = NULL;
            layer->neurons[idx] = neuron;
        }
    }
    layer->activate = NULL;
    layer->delta = previous->delta;
    layer->feedforward = pool;
    return 1;
}

RecurrentCell * createRecurrentCell(int lsize) {
    RecurrentCell * cell = malloc(sizeof(RecurrentCell));
    cell->states_count = 0;
    cell->states = NULL;
    cell->weights_size = lsize;
    cell->weights = malloc(lsize * sizeof(double));
    int i;
    for (i = 0; i < lsize; i++) {
        cell->weights[i] = gaussian_random(0, 1);
    }
    return cell;
}

int initRecurrentLayer(NeuralNetwork * network, Layer * layer) {
    int lsize = layer->size, i, j;
    for (i = 0; i < lsize; i++) {
        Neuron * neuron = layer->neurons[i];
        neuron->extra = createRecurrentCell(lsize);
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = tanh;
    layer->delta = tanh_prime;
    layer->feedforward = recurrentFeedforward;
    network->flags |= FLAG_RECURRENT;
    return 1;
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
    layer->parameters = params;
    layer->extra = NULL;
    layer->flags = FLAG_NONE;
    Layer * previous = NULL;
    int previous_size = 0;
    //printf("Adding layer %d\n", layer->index);
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
    if (type == FullyConnected || type == SoftMax) {
        layer->neurons = malloc(sizeof(Neuron*) * size);
        int i, j;
        for (i = 0; i < size; i++) {
            Neuron * neuron = malloc(sizeof(Neuron));
            neuron->index = i;
            neuron->extra = NULL;
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
        if (type != SoftMax) {
            layer->activate = sigmoid;
            layer->delta = sigmoid_prime;
            layer->feedforward = fullFeedforward;
        } else {
            layer->activate = NULL;
            layer->delta = NULL;
            layer->feedforward = softmaxFeedforward;
        }
    } else if (type == Convolutional) {
        initConvolutionalLayer(network, layer, params);
    } else if (type == Pooling) {
        initPoolingLayer(network, layer, params);
    } else if (type == Recurrent) {
        initRecurrentLayer(network, layer);
    }
    network->layers[layer->index] = layer;
    printLayerInfo(layer);
    return layer;
}

Layer * addConvolutionalLayer(NeuralNetwork * network, LayerParameters* params){
    return addLayer(network, Convolutional, 0, params);
}

Layer * addPoolingLayer(NeuralNetwork * network, LayerParameters* params) {
    return addLayer(network, Pooling, 0, params);
}

void deleteLayer(Layer* layer) {
    int size = layer->size;
    int i;
    for (i = 0; i < size; i++) {
        Neuron* neuron = layer->neurons[i];
        if (layer->type != Convolutional)
            deleteNeuron(neuron, layer);
        else
            free(neuron);
    }
    LayerParameters * params = layer->parameters;
    if (params != NULL) deleteLayerParamenters(params);
    void * extra = layer->extra;
    if (extra != NULL) {
        if (layer->type == Convolutional) {
            ConvolutionalSharedParams * shared;
            shared = (ConvolutionalSharedParams*) extra;
            int fc = shared->feature_count;
            //int ws = shared->weights_size;
            if (shared->biases != NULL) free(shared->biases);
            if (shared->weights != NULL) {
                int i;
                for (i = 0; i < fc; i++) free(shared->weights[i]);
                free(shared->weights);
            }
        } else free(extra);
    }
    free(layer);
}

LayerParameters * createLayerParamenters(int count, ...) {
    LayerParameters * params = malloc(sizeof(LayerParameters));
    params->count = count;
    if (count == 0) params->parameters = NULL;
    else {
        params->parameters = malloc(sizeof(double) * count);
        va_list args;
        va_start(args, count);
        int i;
        for (i = 0; i < count; i++)
            params->parameters[i] = va_arg(args, double);
        va_end(args);
    }
    return params;
}

LayerParameters * createConvolutionalParameters(double feature_count,
                                                double region_size,
                                                int stride,
                                                int padding,
                                                int use_relu) {
    return createLayerParamenters(CONV_PARAMETER_COUNT, feature_count,
                                  region_size, (double) stride,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  (double) padding, (double) use_relu);
}

void setLayerParameter(LayerParameters * params, int param, double value) {
    if (params->parameters == NULL) {
        int len = param + 1;
        params->parameters = malloc(sizeof(double) * len);
        memset(params->parameters, 0.0f, sizeof(double) * len);
        params->count = len;
    } else if (param >= params->count) {
        int len = params->count;
        int new_len = param + 1;
        double * old_params = params->parameters;
        size_t size = sizeof(double) * new_len;
        params->parameters = malloc(sizeof(double) * size);
        memset(params->parameters, 0.0f, sizeof(double) * size);
        memcpy(params->parameters, old_params, len * sizeof(double));
        free(old_params);
    }
    params->parameters[param] = value;
}

void addLayerParameter(LayerParameters * params, double val) {
    setLayerParameter(params, params->count + 1, val);
}

void deleteLayerParamenters(LayerParameters * params) {
    if (params == NULL) return;
    if (params->parameters != NULL) free(params->parameters);
    free(params);
}

void feedforwardThroughTime(NeuralNetwork * network, double * values,
                            int times) {
    Layer * first = network->layers[0];
    int input_size = first->size;
    int i, t;
    for (t = 0; t < times; t++) {
        for (i = 0; i < input_size; i++) {
            Neuron * neuron = first->neurons[i];
            neuron->activation = values[i];
            RecurrentCell * cell = (RecurrentCell*) neuron->extra;
            if (cell == NULL) {
                cell = createRecurrentCell(0);
                neuron->extra = cell;
            }
            if (t == 0) {
                cell->states_count = times;
                if (cell->states != NULL) free(cell->states);
                cell->states = malloc(times * sizeof(double));
            }
            cell->states[t] = values[i];
        }
        for (i = 1; i < network->size; i++) {
            Layer * layer = network->layers[i];
            if (layer == NULL) {
                printf("Layer %d is NULL\n", i);
                deleteNetwork(network);
                exit(1);
            }
            if (layer->feedforward == NULL) {
                printf("Layer %d feedforward function is NULL\n", i);
                deleteNetwork(network);
                exit(1);
            }
            layer->feedforward(network, layer);
        }
        values += input_size;
    }
}

void feedforward(NeuralNetwork * network, double * values) {
    if (network->size == 0) {
        printf("Empty network");
        exit(1);
    }
    if (network->flags & FLAG_RECURRENT) {
        int times = (int) values[0];
        assert(times > 0);
        feedforwardThroughTime(network, values + 1, times);
        return;
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
            deleteNetwork(network);
            exit(1);
        }
        if (layer->feedforward == NULL) {
            printf("Layer %d feedforward function is NULL\n", i);
            deleteNetwork(network);
            exit(1);
        }
        layer->feedforward(network, layer);
    }
}

Delta * emptyLayer(Layer * layer) {
    Delta * delta;
    LayerType ltype = layer->type;
    if (ltype == Pooling) return NULL;
    int size = layer->size;
    LayerParameters * parameters = NULL;
    if (ltype == Convolutional) {
        parameters = layer->parameters;
        assert(parameters != NULL);
        size = (int) (parameters->parameters[FEATURE_COUNT]);
    }
    delta = malloc(sizeof(Delta) * size);
    int i, ws = 0;
    for (i = 0; i < size; i++) {
        if (ltype == Convolutional) {
            if (!ws) {
                int region_size = (int) (parameters->parameters[REGION_SIZE]);
                ws = region_size * region_size;
            }
        } else {
            Neuron * neuron = layer->neurons[i];
            ws = neuron->weights_size;
        }
        delta[i].bias = 0;
        int memsize = sizeof(double) * ws;
        delta[i].weights = malloc(memsize);
        memset(delta[i].weights, 0, memsize);
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
        if (delta == NULL) continue;
        Layer * layer = network->layers[i];
        int lsize;
        if (layer->type == Convolutional) {
            LayerParameters * params = layer->parameters;
            lsize = (int) (params->parameters[FEATURE_COUNT]);
        } else lsize = layer->size;
        deleteDelta(delta, lsize);
    }
    free(deltas);
}

double * backpropPoolingToConv(NeuralNetwork * network, Layer * pooling_layer,
                               Layer * convolutional_layer, double * delta_v) {
    int conv_size = convolutional_layer->size;
    double * new_delta_v = malloc(sizeof(double) * conv_size);
    memset(new_delta_v, 0, sizeof(double) * conv_size);
    LayerParameters * pool_params = pooling_layer->parameters;
    LayerParameters * conv_params = convolutional_layer->parameters;
    int feature_count = (int) (conv_params->parameters[FEATURE_COUNT]);
    int pool_size = (int) (pool_params->parameters[REGION_SIZE]);
    int feature_size = pooling_layer->size / feature_count;
    double input_w = pool_params->parameters[INPUT_WIDTH];
    double output_w = pool_params->parameters[OUTPUT_WIDTH];
    int prev_size = convolutional_layer->size / feature_count;
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            double d = delta_v[idx];
            Neuron * neuron = pooling_layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * pool_size;
            int r_col = col * pool_size;
            int max_x = pool_size + r_col;
            int max_y = pool_size + r_row;
            double max = 0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size * i);
                    Neuron * prev_neuron = convolutional_layer->neurons[nidx];
                    double a = prev_neuron->activation;
                    new_delta_v[nidx] = (a < neuron->activation ? 0 : d);
                }
            }
            
        }
    }
    return new_delta_v;
}

double * backpropConvToFull(NeuralNetwork * network, Layer* convolutional_layer,
                            Layer * full_layer, double * delta_v,
                            Delta * ldelta) {
    int size = convolutional_layer->size;
    LayerParameters * params = convolutional_layer->parameters;
    int feature_count = (int) (params->parameters[FEATURE_COUNT]);
    int region_size = (int) (params->parameters[REGION_SIZE]);
    int stride = (int) (params->parameters[STRIDE]);
    double input_w = params->parameters[INPUT_WIDTH];
    double output_w = params->parameters[OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int wsize = region_size * region_size;
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        Delta * feature_delta = &(ldelta[i]);
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            double d = delta_v[idx];
            feature_delta->bias += d;
            
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * stride;
            int r_col = col * stride;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            int widx = 0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = (y * input_w) + x;
                    //printf("  -> %d,%d [%d]\n", x, y, nidx);
                    Neuron * prev_neuron = full_layer->neurons[nidx];
                    double a = prev_neuron->activation;
                    feature_delta->weights[widx++] += (a * d);
                }
            }
        }
    }
    return NULL;
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
    double softmax_sum = 0.0;
    for (o = 0; o < osize; o++) {
        Neuron * neuron = outputLayer->neurons[o];
        double o_val = neuron->activation;
        double y_val = y[o];
        double d = 0.0;
        if (outputLayer->type != SoftMax) {
            d = o_val - y_val;
            d *= outputLayer->delta(neuron->z_value);
        } else {
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            d *= o_val;
            softmax_sum += d;
            //printf("SoftMax D[%d](y=%lf) -> %lf\n", o, y_val, d);
        }
        delta_v[o] = d;
        if (outputLayer->type != SoftMax) {
            Delta * n_delta = &(layer_delta[o]);
            n_delta->bias = d;
            for (w = 0; w < neuron->weights_size; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                n_delta->weights[w] = d * prev_a;
            }
        }
    }
    if (outputLayer->type == SoftMax) {
        for (o = 0; o < osize; o++) {
            Neuron * neuron = outputLayer->neurons[o];
            double o_val = neuron->activation;
            delta_v[o] -= (o_val * softmax_sum);
            double d = delta_v[o];
            Delta * n_delta = &(layer_delta[o]);
            n_delta->bias = d;
            for (w = 0; w < neuron->weights_size; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                n_delta->weights[w] = d * prev_a;
            }
        }
    }
    for (i = previousLayer->index; i > 0; i--) {
        Layer * layer = network->layers[i];
        previousLayer = network->layers[i - 1];
        nextLayer = network->layers[i + 1];
        layer_delta = deltas[i - 1];
        int lsize = layer->size;
        LayerType ltype = layer->type;
        LayerType prev_ltype = previousLayer->type;
        if (FullyConnected == ltype) {
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
                double dv = sum * layer->delta(neuron->z_value);
                delta_v[j] = dv;
                Delta * n_delta = &(layer_delta[j]);
                n_delta->bias = dv;
                for (w = 0; w < neuron->weights_size; w++) {
                    double prev_a = previousLayer->neurons[w]->activation;
                    n_delta->weights[w] = dv * prev_a;
                }
            }
        } else if (Pooling == ltype && Convolutional == prev_ltype) {
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
                double dv = sum * layer->delta(neuron->z_value);
                delta_v[j] = dv;
            }
            free(last_delta_v);
            last_delta_v = delta_v;
            delta_v = backpropPoolingToConv(network, layer,
                                            previousLayer, last_delta_v);
            //free(delta_v);
            //continue;
        } else if (Convolutional == ltype && FullyConnected == prev_ltype) {
            delta_v = backpropConvToFull(network, layer, previousLayer,
                               last_delta_v, layer_delta);
        } else {
            fprintf(stderr, "Backprop from %s to %s not suported!\n",
                    getLayerTypeLabel(layer),
                    getLayerTypeLabel(previousLayer));
            free(last_delta_v);
            free(delta_v);
            exit(1);
        }
        free(last_delta_v);
        last_delta_v = delta_v;
    }
    if (delta_v != NULL) free(delta_v);
    return deltas;
}

double updateWeights(NeuralNetwork * network, double * training_data,
                     int batch_size, double rate) {
    double r = rate / (double) batch_size;
    int i, j, k, w, netsize = network->size, dsize = netsize - 1;
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    int element_size = training_data_size + label_data_size;
    Delta ** deltas = emptyDeltas(network);
    //double * data_p = training_data;
    double * x;
    double * y;
    for (i = 0; i < batch_size; i++) {
        x = training_data;
        y = training_data + training_data_size;
        training_data += element_size;
        Delta ** bp_deltas = backprop(network, x, y);
        for (j = 0; j < dsize; j++) {
            Layer * layer = network->layers[j + 1];
            Delta * layer_delta_bp = bp_deltas[j];
            Delta * layer_delta = deltas[j];
            if (layer_delta == NULL) continue;
            int lsize = layer->size;
            int wsize = 0;
            if (layer->type == Convolutional) {
                LayerParameters * params = layer->parameters;
                lsize = (int) (params->parameters[FEATURE_COUNT]);
                int rsize = (int) (params->parameters[REGION_SIZE]);
                wsize = rsize * rsize;
            }
            for (k = 0; k < lsize; k++) {
                if (!wsize) {
                    Neuron * neuron = layer->neurons[k];
                    wsize = neuron->weights_size;
                }
                Delta * n_delta_bp = &(layer_delta_bp[k]);
                Delta * n_delta = &(layer_delta[k]);
                n_delta->bias += n_delta_bp->bias;
                for (w = 0; w < wsize; w++) {
                    n_delta->weights[w] += n_delta_bp->weights[w];
                }
            }
        }
        deleteDeltas(bp_deltas, network);
    }
    for (i = 0; i < dsize; i++) {
        Delta * l_delta = deltas[i];
        if (l_delta == NULL) continue;
        Layer * layer = network->layers[i + 1];
        LayerType ltype = layer->type;
        int l_size;
        ConvolutionalSharedParams * shared = NULL;
        if (ltype == Convolutional) {
            LayerParameters * params = layer->parameters;
            l_size = (int) (params->parameters[FEATURE_COUNT]);
            shared = (ConvolutionalSharedParams*) layer->extra;
        } else l_size = layer->size;
        for (j = 0; j < l_size; j++) {
            Delta * d = &(l_delta[j]);
            if (shared == NULL) {
                Neuron * neuron = layer->neurons[j];
                neuron->bias = neuron->bias - r * d->bias;
                for (k = 0; k < neuron->weights_size; k++) {
                    double w = neuron->weights[k];
                    neuron->weights[k] = w - r * d->weights[k];
                }
            } else {
                double b = shared->biases[j];
                shared->biases[j] = b - r * d->bias;
                double * weights = shared->weights[j];
                for (k = 0; k < shared->weights_size; k++) {
                    double w = weights[k];
                    weights[k] = w - r * d->weights[k];
                }
            }
        }
    }
    deleteDeltas(deltas, network);
    Layer * out = network->layers[netsize - 1];
    double outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        outputs[i] = out->neurons[i]->activation;
    }
    return loss(outputs, y, label_data_size);
}

double gradientDescent(NeuralNetwork * network,
                       double * training_data,
                       int element_size,
                       int elements_count,
                       double learning_rate,
                       int batch_size) {
    int batches_count = elements_count / batch_size;
    shuffle(training_data, elements_count, element_size);
    int offset = (element_size * batch_size), i;
    double err = 0.0;
    for (i = 0; i < batches_count; i++) {
        network->current_batch = i;
        printf("\rEpoch %d: batch %d/%d", network->current_epoch + 1,
               i + 1, batches_count);
        fflush(stdout);
        err += updateWeights(network, training_data, batch_size, learning_rate);
        training_data += offset;
    }
    return err / (double) batches_count;
}

float validate(NeuralNetwork * network, double * test_data, int data_size,
               int log) {
    int i, j;
    float accuracy = 0.0f;
    int correct_results = 0;
    int input_size = network->input_size;
    int output_size = network->output_size;
    int element_size = input_size + output_size;
    int elements_count = data_size / element_size;
    double outputs[output_size];
    Layer * output_layer = network->layers[network->size - 1];
    if (log) printf("Test data elements: %d\n", elements_count);
    time_t start_t, end_t;
    char timestr[80];
    struct tm * tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    if (log) printf("Testing started at %s\n", timestr);
    for (i = 0; i < elements_count; i++) {
        if (log) printf("\rTesting %d/%d", i + 1, elements_count);
        fflush(stdout);
        double * inputs = test_data;
        test_data += input_size;
        double * expected = test_data;
        feedforward(network, inputs);
        for (j = 0; j < output_size; j++) {
            Neuron * neuron = output_layer->neurons[j];
            outputs[j] = neuron->activation;
        }
        int omax = arrayMaxIndex(outputs, output_size);
        int emax = arrayMaxIndex(expected, output_size);
        if (omax == emax) correct_results++;
        test_data += output_size;
    }
    if (log) printf("\n");
    time(&end_t);
    if (log) printf("Completed in %ld sec.\n", end_t - start_t);
    accuracy = (float) correct_results / (float) elements_count;
    if (log) printf("Accuracy (%d/%d): %.2f\n", correct_results, elements_count,
                    accuracy);
    return accuracy;
}

void train(NeuralNetwork * network,
           double * training_data,
           int data_size,
           int epochs,
           double learning_rate,
           int batch_size,
           double * test_data,
           int test_size) {
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
        double err = gradientDescent(network, training_data, element_size,
                                     elements_count, learning_rate, batch_size);
        char accuracy_msg[255] = "";
        if (test_data != NULL) {
            int batches_count = elements_count / batch_size;
            printf("\rEpoch %d: batch %d/%d, validating...",
                   network->current_epoch + 1,
                   network->current_batch + 1,
                   batches_count);
            float acc = validate(network, test_data, test_size, 0);
            printf("\rEpoch %d: batch %d/%d",
                   network->current_epoch + 1,
                   network->current_batch + 1,
                   batches_count);
            sprintf(accuracy_msg, ", acc = %.2f,", acc);
        }
        time(&epoch_t);
        time_t elapsed_t = epoch_t - e_t;
        e_t = epoch_t;
        printf(", loss = %.2lf%s (%ld sec.)\n", err, accuracy_msg, elapsed_t);
    }
    time(&end_t);
    printf("Completed in %ld sec.\n", end_t - start_t);
    network->status = STATUS_TRAINED;
}

float test(NeuralNetwork * network, double * test_data, int data_size) {
    return validate(network, test_data, data_size, 1);
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


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

#ifdef USE_AVX
#include "avx.h"
#endif

#include "neural.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define calculateConvolutionalSide(s,rs,st,pad) ((s - rs + 2 * pad) / st + 1)
#define calculatePoolingSide(s, rs) ((s - rs) / rs + 1)
#define getColumn(index, width) (index % width)
#define getRow(index, width) ((int) ((int) index / (int) width))
#define getConvSharedParams(layer) ((ConvolutionalSharedParams*) layer->extra)
#define getRecurrentCell(neuron) ((RecurrentCell*) neuron->extra)
#define getNeuronLayer(neuron) ((Layer*) neuron->layer)
#define getLayerNetwork(layer) ((NeuralNetwork*) layer->network)
#define shouldApplyDerivative(network) (network->loss != crossEntropyLoss)
#define printMemoryErrorMsg() logerr(NULL, "Could not allocate memory!")

static unsigned char randomSeeded = 0;

static LossFunction loss_functions[] = {
    NULL,
    quadraticLoss,
    crossEntropyLoss
};

static size_t loss_functions_count = sizeof(loss_functions) /
                                     sizeof(LossFunction);

/* Private Functions */

static void logerr(const char* tag, char* fmt, ...) {
    va_list args;
    
    fflush (stdout);
    fprintf(stderr, "ERROR");
    if (tag != NULL) fprintf(stderr, " [%s]: ", tag);
    else fprintf(stderr, ": ");
    
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    fprintf(stderr, "\n");
}

/* Function Prototypes */

RecurrentCell * createRecurrentCell(Neuron * neuron, int lsize);
void addRecurrentState(Neuron * neuron, double state, int times, int t);
void deleteLayerGradients(Gradient * lgradients, int size);
void deleteGradients(Gradient ** gradients, NeuralNetwork * network);

/* Activation Functions */

double sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double sigmoid_derivative(double val) {
    double s = sigmoid(val);
    return s * (1 - s);
}

double relu(double val) {
    return (val >= 0.0 ? val : 0.0);
}

double relu_derivative(double val) {
    return (double)(val > 0.0);
}

double tanh_derivative(double val) {
    return (1 - (val * val));
}

/* Feedforward Functions */

int fullFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * network = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    char * func = "fullFeedforward";
    if (layer->neurons == NULL) {
        logerr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        logerr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    Layer * previous = network->layers[layer->index - 1];
    if (previous == NULL) {
        logerr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
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
        j = 0;
#ifdef USE_AVX
        int avx_step_len = AVXGetDotStepLen(previous_size);
        avx_dot_product dot_product = AVXGetDotProductFunc(previous_size);
        int avx_steps = previous_size / avx_step_len, avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            double * x_vector = previous->avx_activation_cache + j;
            if (is_recurrent) x_vector += (t * previous_size);
            double * y_vector = neuron->weights + j;
            sum += dot_product(x_vector, y_vector);
            j += avx_step_len;
        }
#endif
        for (; j < previous_size; j++) {
            Neuron * prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                logerr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                       layer->index, j);
                return 0;
            }
            double a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        neuron->activation = layer->activate(neuron->z_value);
#ifdef USE_AVX
        if (!is_recurrent)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            addRecurrentState(neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                logerr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
    }
    return 1;
}

int softmaxFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    char * func = "softmaxFeedforward";
    if (layer->neurons == NULL) {
        logerr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        logerr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    Layer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        logerr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
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
        j = 0;
#ifdef USE_AVX
        int avx_step_len = AVXGetDotStepLen(previous_size);
        avx_dot_product dot_product = AVXGetDotProductFunc(previous_size);
        int avx_steps = previous_size / avx_step_len, avx_step;
        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
            double * x_vector = previous->avx_activation_cache + j;
            if (is_recurrent) x_vector += (t * previous_size);
            double * y_vector = neuron->weights + j;
            sum += dot_product(x_vector, y_vector);
            j += avx_step_len;
        }
#endif
        for (; j < previous_size; j++) {
            Neuron * prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                logerr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                       layer->index, j);
                return 0;
            }
            double a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        if (i == 0)
            max = neuron->z_value;
        else if (neuron->z_value > max)
            max = neuron->z_value;
    }
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        double z = neuron->z_value;
        double e = exp(z - max);
        esum += e;
        neuron->activation = e;
    }
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        neuron->activation /= esum;
#ifdef USE_AVX
        if (!is_recurrent)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            addRecurrentState(neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                logerr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
    }
    return 1;
}

int convolve(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    if (layer->neurons == NULL) {
        logerr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        logerr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    Layer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        logerr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, x, y, row, col, previous_size = previous->size;
    LayerParameters * parameters = layer->parameters;
    if (parameters == NULL) {
        logerr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    LayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        logerr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
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
    ConvolutionalSharedParams * shared = getConvSharedParams(layer);
    if (shared == NULL) {
        logerr(NULL, "Layer[%d]: shared params are NULL!", layer->index);
        return 0;
    }
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
#ifdef USE_AVX
            layer->avx_activation_cache[idx] = neuron->activation;
#endif
        }
    }
    return 1;
}

int pool(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    int size = layer->size;
    if (layer->neurons == NULL) {
        logerr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        logerr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    Layer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        logerr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, x, y, row, col, previous_size = previous->size;
    LayerParameters * parameters = layer->parameters;
    if (parameters == NULL) {
        logerr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    LayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        logerr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
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
#ifdef USE_AVX
            layer->avx_activation_cache[idx] = neuron->activation;
#endif
        }
    }
    return 1;
}

int recurrentFeedforward(void * _net, void * _layer, ...) {
    NeuralNetwork * net = (NeuralNetwork*) _net;
    Layer * layer = (Layer*) _layer;
    char * func = "recurrentFeedforward";
    va_list args;
    va_start(args, _layer);
    int times = va_arg(args, int);
    int t = va_arg(args, int);
    va_end(args);
    if (times < 1) {
        logerr(func, "Layer[%d]: times must be >= 1 (found %d)",
                layer->index, times);
        return 0;
    }
    int size = layer->size;
    if (layer->neurons == NULL) {
        logerr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        logerr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    Layer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        logerr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int onehot = previous->flags & FLAG_ONEHOT;
    LayerParameters * params = NULL;
    int vector_size = 0, vector_idx = 0;
    if (onehot) {
        params = previous->parameters;
        if (params == NULL) {
            logerr(NULL, "Layer[%d]: prev. onehot layer params are NULL!",
                   layer->index);
            return 0;
        }
        if (params->count < 1) {
            logerr(NULL, "Layer[%d]: prev. onehot layer params < 1!",
                   layer->index);
            return 0;
        }
        vector_size = (int) (params->parameters[0]);
        Neuron * prev_neuron = previous->neurons[0];
        vector_idx = (int) (prev_neuron->activation);
        if (vector_size == 0 && vector_idx >= vector_size) {
            logerr(NULL, "Layer[%d]: invalid vector index %d (max. %d)!",
                   previous->index, vector_idx, vector_size - 1);
            return 0;
        }
    }
    int i, j, w, previous_size = previous->size;
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        RecurrentCell * cell = getRecurrentCell(neuron);
        if (cell == NULL) {
            logerr(NULL, "Layer[%d]: neuron[%d] cell is NULL!",
                   layer->index, i);
            return 0;
        }
        double sum = 0, bias = 0;
        if (onehot) sum = neuron->weights[vector_idx];
        else {
            j = 0;
#ifdef USE_AVX
            int avx_step_len = AVXGetDotStepLen(previous_size);
            avx_dot_product dot_product = AVXGetDotProductFunc(previous_size);
            int avx_steps = previous_size / avx_step_len, avx_step;
            for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                double * x_vector = previous->avx_activation_cache + j;
                x_vector += (t * previous_size);
                double * y_vector = neuron->weights + j;
                sum += dot_product(x_vector, y_vector);
                j += avx_step_len;
            }
#endif
            for (; j < previous_size; j++) {
                Neuron * prev_neuron = previous->neurons[j];
                if (prev_neuron == NULL) return 0;
                double a = prev_neuron->activation;
                sum += (a * neuron->weights[j]);
            }
        }
        if (t > 0) {
            int last_t = t - 1;
            w = 0;
#ifdef USE_AVX
            int avx_step_len = AVXGetDotStepLen(size);
            avx_dot_product dot_product = AVXGetDotProductFunc(size);
            int avx_steps = cell->weights_size / avx_step_len, avx_step;
            for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                double * x_vector = layer->avx_activation_cache + w;
                x_vector += (last_t * size);
                double * y_vector = cell->weights + w;
                bias += dot_product(x_vector, y_vector);
                w += avx_step_len;
            }
#endif
            for (; w < cell->weights_size; w++) {
                Neuron * n = layer->neurons[w];
                RecurrentCell * rc = getRecurrentCell(n);
                if (rc == NULL) return 0;
                double weight = cell->weights[w];
                double last_state = rc->states[last_t];
                bias += (weight * last_state);
            }
        } else {
            if (cell->states != NULL) free(cell->states);
            cell->states_count = times;
            cell->states = calloc(times, sizeof(double));
#ifdef USE_AVX
            if (neuron->index == 0) {
                if (layer->avx_activation_cache != NULL)
                    free(layer->avx_activation_cache);
                layer->avx_activation_cache = calloc(times * size,
                                                     sizeof(double));
                if (layer->avx_activation_cache == NULL) {
                    printMemoryErrorMsg();
                    return 0;
                }
            }
#endif
        }
        neuron->z_value = sum + bias;
        neuron->activation = layer->activate(neuron->z_value);
        cell->states[t] = neuron->activation;
#ifdef USE_AVX
        layer->avx_activation_cache[(t * size) + i] = neuron->activation;
#endif
    }
    return 1;
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

void addRecurrentState(Neuron * neuron, double state, int times, int t) {
    RecurrentCell * cell = getRecurrentCell(neuron);
    if (cell == NULL) {
        cell = createRecurrentCell(neuron, 0);
        neuron->extra = cell;
        if (cell == NULL) return;
    }
    if (t == 0) {
        cell->states_count = times;
        if (cell->states != NULL) free(cell->states);
        cell->states = malloc(times * sizeof(double));
        if (cell->states == NULL) {
            free(cell);
            return;
        }
    }
    cell->states[t] = state;
#ifdef USE_AVX
    Layer * layer = getNeuronLayer(neuron);
    assert(layer != NULL);
    int lsize = layer->size;
    if (t == 0 && neuron->index == 0) {
        if (layer->avx_activation_cache != NULL)
            free(layer->avx_activation_cache);
        layer->avx_activation_cache = calloc(lsize * times, sizeof(double));
    }
    if (layer->avx_activation_cache == NULL) {
        printMemoryErrorMsg();
        return; //TODO: handle
    }
    layer->avx_activation_cache[(t * lsize) + neuron->index] = state;
#endif
}

static double ** getRecurrentSeries(double * array, int series_count,
                                    int x_size, int y_size)
{
    double ** series = malloc(series_count * sizeof(double**));
    if (series == NULL) {
        logerr(NULL, "Could not allocate memory for recurrent series!");
        return NULL;
    }
    int i;
    double * p = array;
    for (i = 0; i < series_count; i++) {
        int series_size = (int) *p;
        if (!series_size) {
            logerr(NULL, "Invalid series size 0 at %d", (int) (p - array));
            free(series);
            return NULL;
        }
        series[i] = p++;
        p += ((series_size * x_size) + (series_size * y_size));
    }
    return series;
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

void fetchRecurrentOutputState(Layer * out, double * outputs,
                               int i, int onehot)
{
    int t = (onehot ? i : i % out->size), j;
    int max_idx = 0;
    double max = 0.0;
    for (j = 0; j < out->size; j++) {
        Neuron * neuron = out->neurons[j];
        RecurrentCell * cell = getRecurrentCell(neuron);
        double s = cell->states[t];
        if (onehot) {
            if (s > max) {
                max = s;
                max_idx = j;
            }
        } else {
            outputs[i] = s;
        }
    }
    if (onehot) outputs[i] = max_idx;
}

static int compareVersion(const char* vers1, const char* vers2) {
    int major1 = 0, minor1 = 0, patch1 = 0;
    int major2 = 0, minor2 = 0, patch2 = 0;
    sscanf(vers1, "%d.%d.%d", &major1, &minor1, &patch1);
    sscanf(vers2, "%d.%d.%d", &major2, &minor2, &patch2);
    if (major1 < major2) return -1;
    if (major1 > major2) return 1;
    if (minor1 < minor2) return -1;
    if (minor1 > minor2) return 1;
    if (patch1 < patch2) return -1;
    if (patch1 > patch2) return 1;
    return 0;
}

char * getLabelForType(LayerType type) {
    switch (type) {
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

char * getLayerTypeLabel(Layer * layer) {
    return getLabelForType(layer->type);
}

char * getLossFunctionName(LossFunction function) {
    if (function == NULL) return "null";
    if (function == quadraticLoss) return "quadratic";
    else if (function == crossEntropyLoss) return "cross_entropy";
    return "UNKOWN";
}

void printLayerInfo(Layer * layer) {
    LayerType ltype = layer->type;
    char * type_name = getLayerTypeLabel(layer);
    LayerParameters * lparams = layer->parameters;
    char onehot_info[50];
    onehot_info[0] = 0;
    if (layer->index == 0 && layer->flags & FLAG_ONEHOT) {
        LayerParameters * params = layer->parameters;
        int onehot_sz = (int) (params->parameters[0]);
        sprintf(onehot_info, " (vector size: %d)", onehot_sz);
    }
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if (onehot_info[0]) printf(" %s", onehot_info);
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

/* Loss Functions */

double quadraticLoss(double * outputs, double * desired, int size,
                     int onehot_size)
{
    double * d;
    double diffs[size];
    if (!onehot_size) {
        int i;
        
        for (i = 0; i < size; i++) {
            double d = outputs[i] - desired[i];
            diffs[i] = d;
        }
        d = diffs;
    } else d = outputs;
    double n = norm(diffs, size);
    double loss = 0.5 * (n * n);
    if (onehot_size) loss /= (double) onehot_size;
    return loss;
}

double crossEntropyLoss(double * outputs, double * desired, int size,
                        int onehot_size)
{
    double loss = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        double o = outputs[i];
        if (o == 0.0) continue;
        if (onehot_size) loss += (log(o));
        else {
            if (o == 1) continue; // log(1 - 1) would be NaN
            double y = desired[i];
            loss += (y * log(o) + (1 - y) * log(1 - o));
        }
    }
    loss *= -1;
    if (onehot_size)
        loss = (loss / (double) size);// / log((double) onehot_size);
    return loss;
}

/* NN Functions */

NeuralNetwork * createNetwork() {
    NeuralNetwork *network = (malloc(sizeof(NeuralNetwork)));
    if (network == NULL) {
        logerr("createNetwork", "Could not allocate memory for Network!");
        return NULL;
    }
    network->size = 0;
    network->layers = NULL;
    network->input_size = 0;
    network->output_size = 0;
    network->status = STATUS_UNTRAINED;
    network->current_epoch = 0;
    network->current_batch = 0;
    network->flags = FLAG_NONE;
    network->loss = quadraticLoss;
    return network;
}

NeuralNetwork * cloneNetwork(NeuralNetwork * network, int layout_only) {
    NeuralNetwork * clone = createNetwork();
    if (clone == NULL) return NULL;
    char * func = "cloneNetwork";
    if (!layout_only) {
        clone->status = network->status;
        clone->current_epoch = network->current_epoch;
        clone->current_batch = network->current_batch;
    }
    clone->flags = network->flags;
    clone->loss = network->loss;
    
    int i, j, k, w;
    for (i = 0; i < network->size; i++) {
        Layer * layer = network->layers[i];
        LayerType type = layer->type;
        LayerParameters * oparams = layer->parameters;
        LayerParameters * cparams = NULL;
        if (oparams) {
            cparams = malloc(sizeof(LayerParameters));
            if (cparams == NULL) {
                logerr(func, "Layer[%d]: Could not allocate layer params!", i);
                deleteNetwork(clone);
                return NULL;
            }
            cparams->count = oparams->count;
            cparams->parameters = malloc(cparams->count * sizeof(double));
            if (cparams->parameters == NULL) {
                logerr(func, "Layer[%d]: Could not allocate layer params!", i);
                deleteNetwork(clone);
                return NULL;
            }
            for (j = 0; j < cparams->count; j++)
                cparams->parameters[j] = oparams->parameters[j];
        }
        Layer * cloned_layer = addLayer(clone, type, layer->size, cparams);
        if (cloned_layer == NULL) {
            deleteNetwork(clone);
            return NULL;
        }
        cloned_layer->flags = layer->flags;
        if (!layout_only) {
            void * extra = layer->extra;
            if (Convolutional == type && extra) {
                ConvolutionalSharedParams * oshared;
                ConvolutionalSharedParams * cshared;
                oshared = getConvSharedParams(layer);
                cshared = getConvSharedParams(cloned_layer);
                cshared->feature_count = oshared->feature_count;
                cshared->weights_size = oshared->weights_size;
                for (k = 0; k < cshared->feature_count; k++) {
                    cshared->biases[k] = oshared->biases[k];
                    for (w = 0; w < cshared->weights_size; w++)
                        cshared->weights[k][w] = oshared->weights[k][w];
                }
            }
            for (j = 0; j < layer->size; j++) {
                Neuron * orig_n = layer->neurons[j];
                Neuron * clone_n = cloned_layer->neurons[j];
                clone_n->activation = orig_n->activation;
                clone_n->z_value = orig_n->z_value;
                //if (Pooling == type) continue;
                clone_n->bias = orig_n->bias;

                if (Convolutional != type && Pooling != type) {
                    double * oweights = orig_n->weights;
                    double * cweights = clone_n->weights;
                    for (w = 0; w < orig_n->weights_size; w++)
                        cweights[w] = oweights[w];
                }
                if (layer->flags & FLAG_RECURRENT) {
                    RecurrentCell * ocell = getRecurrentCell(orig_n);
                    RecurrentCell * ccell = getRecurrentCell(clone_n);
                    int sc = ocell->states_count;
                    ccell->states_count = sc;
                    if (sc > 0) {
                        ccell->states = malloc(sc * sizeof(double));
                        if (ccell->states == NULL) {
                            printMemoryErrorMsg();
                            deleteNetwork(clone);
                            return NULL;
                        }
                        for (k = 0; k < sc; k++)
                            ccell->states[k] = ocell->states[k];
                    }
                }
            }
        }
    }
    return clone;
}

int loadNetwork(NeuralNetwork * network, const char* filename) {
    FILE * f = fopen(filename, "r");
    printf("Loading network from %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s!\n", filename);
        return 0;
    }
    char * func = "loadNetwork";
    int netsize, i, j, k;
    int empty = (network->size == 0);
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int epochs = 0, batch_count = 0;
    int matched = fscanf(f, "--v%d.%d.%d", &v0, &v1, &v2);
    if (matched) {
        sprintf(vers, "%d.%d.%d", v0, v1, v2);
        printf("File version is %s (current: %s).\n", vers, NN_VERSION);
        int idx = 0, val = 0;
        LossFunction loss = NULL;
        while ((matched = fscanf(f, ",%d", &val))) {
            switch (idx++) {
                case 0:
                    network->flags |= val; break;
                case 1:
                    if (val < loss_functions_count) {
                        loss = loss_functions[val];
                        network->loss = loss;
                        printf("Loss Function: %s\n",getLossFunctionName(loss));
                    }
                    break;
                case 2: epochs = val; break;
                case 3: batch_count = val; break;
                default:
                    break;
            }
        }
        fscanf(f, "\n");
    }
    matched = fscanf(f, "%d:", &netsize);
    if (!matched) {
        logerr(func, "Invalid file %s!", filename);
        fclose(f);
        return 0;
    }
    if (!empty && network->size != netsize) {
        logerr(func, "Network size differs!");
        fclose(f);
        return 0;
    }
    char sep[] = ",";
    char eol[] = "\n";
    int min_argc = (compareVersion(vers, "0.0.0") == 1 ? 2 : 1);
    Layer * layer = NULL;
    for (i = 0; i < netsize; i++) {
        int lsize = 0;
        int lflags = 0;
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
                logerr(func, "Invalid header: layer[%d], col. %ld!",
                       i, ftell(f));
                fclose(f);
                return 0;
            }
            if (argc == 0) {
                logerr(func, "Layer must have at least 1 argument (size)");
                fclose(f);
                return 0;
            }
            ltype = (LayerType) type;
            for (aidx = 0; aidx < argc; aidx++) {
                matched = fscanf(f, ",%d", &arg);
                if (!matched) {
                    logerr(func, "Invalid header: l%d, arg. %d, col. %ld!",
                           i, aidx, ftell(f));
                    fclose(f);
                    return 0;
                }
                if (aidx == 0) lsize = arg;
                else if (min_argc > 1 && aidx == 1) lflags = arg;
                else args[aidx - min_argc] = arg;
            }
            argc -= min_argc;
            sprintf(fmt, "]%s", last);
            fscanf(f, fmt, buff);
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                logerr(func, "Layer %d size %d differs from %d!", i,
                       layer->size, lsize);
                fclose(f);
                return 0;
            }
            if (ltype != layer->type) {
                logerr(func, "Layer %d type %d differs from %d!", i,
                       (int) (layer->type), (int) ltype);
                fclose(f);
                return 0;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                LayerParameters * params = layer->parameters;
                if (params == NULL) {
                    logerr(func, "Layer %d params are NULL!", i);
                    fclose(f);
                    return 0;
                }
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= params->count) break;
                    int arg = args[aidx];
                    double val = params->parameters[aidx];
                    if (arg != (int) val) {
                        logerr(func, "Layer %d arg[%d] %d diff. from %d!",
                               i, aidx,(int) val, arg);
                        fclose(f);
                        return 0;
                    }
                }
            }
        } else {
            layer = NULL;
            LayerParameters * params = NULL;
            if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                params = createLayerParamenters(param_c);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= param_c) break;
                    int arg = args[aidx];
                    params->parameters[aidx] = (double) arg;
                }
                layer = addLayer(network, ltype, lsize, params);
            } else {
                if (network->size == 0 && (lflags & FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    network->flags |= FLAG_ONEHOT;
                } else if (argc > 0) {
                    params = createLayerParamenters(argc);
                    for (aidx = 0; aidx < argc; aidx++) {
                        int arg = args[aidx];
                        params->parameters[aidx] = (double) arg;
                    }
                }
                layer = addLayer(network, ltype, lsize, params);
            }
            if (layer == NULL) {
                logerr(func, "Could not create layer %d", i);
                fclose(f);
                return 0;
            }
            layer->flags |= lflags;
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = 0;
        ConvolutionalSharedParams * shared = NULL;
        if (layer->type == Convolutional) {
            shared = getConvSharedParams(layer);
            if (shared == NULL) {
                logerr(func, "Layer %d, missing shared params!", i);
                fclose(f);
                return 0;
            }
            lsize = shared->feature_count;
        } else if (layer->type == Pooling) {
            continue;
        } else lsize = layer->size;
        for (j = 0; j < lsize; j++) {
            double bias = 0;
            int wsize = 0;
            double * weights = NULL;
            matched = fscanf(f, "%lf|", &bias);
            if (!matched) {
                logerr(func, "Layer %d, neuron %d: invalid bias!", i, j);
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
                    logerr(func,"Layer %d neuron %d: invalid weight[%d]",
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
    char * func = "saveNetwork";
    if (network->size == 0) {
        logerr(func, "Empty network!");
        return 0;
    }
    FILE * f = fopen(filename, "w");
    printf("Saving network to %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    int netsize, i, j, k, loss_function = 0;
    // Header
    fprintf(f, "--v%s", NN_VERSION);
    for (i = 0; i < loss_functions_count; i++) {
        if (network->loss == loss_functions[i]) {
            loss_function = i;
            break;
        }
    }
    fprintf(f, ",%d,%d,%d,%d\n", network->flags, loss_function,
            network->current_epoch, network->current_batch);
    
    fprintf(f, "%d:", network->size);
    for (i = 0; i < network->size; i++) {
        Layer * layer = network->layers[i];
        LayerType ltype = layer->type;
        if (i > 0) fprintf(f, ",");
        int flags = layer->flags;
        LayerParameters * params = layer->parameters;
        if (FullyConnected == ltype && !flags && !params)
            fprintf(f, "%d", layer->size);
        else if (params) {
            int argc = params->count;
            fprintf(f, "[%d,%d,%d,%d", (int) ltype, 2 + argc, layer->size,
                    layer->flags);
            for (j = 0; j < argc; j++) {
                fprintf(f, ",%d", (int) (params->parameters[j]));
            }
            fprintf(f, "]");
        } else {
            fprintf(f, "[%d,2,%d,%d]", (int) ltype, layer->size, flags);
        }
    }
    fprintf(f, "\n");
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        LayerType ltype = layer->type;
        int lsize = layer->size;
        if (Convolutional == ltype) {
            LayerParameters * params = layer->parameters;
            ConvolutionalSharedParams * shared = getConvSharedParams(layer);
            if (shared == NULL) {
                logerr(func, "Layer[%d]: shared params are NULL!", i);
                fclose(f);
                return 0;
            }
            int feature_count = shared->feature_count;
            if (feature_count < 1) {
                logerr(func, "Layer[%d]: feature count must be >= 1!", i);
                fclose(f);
                return 0;
            }
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
        }
        else if (Pooling == ltype) continue;
        else {
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
        }
    }
    fclose(f);
    return 1;
}

void deleteNetwork(NeuralNetwork * network) {
    int size = network->size;
    int i, is_recurrent = (network->flags & FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        Layer * layer = network->layers[i];
        if (is_recurrent) layer->flags |= FLAG_RECURRENT;
        deleteLayer(layer);
    }
    free(network->layers);
    free(network);
}

void deleteNeuron(Neuron * neuron, Layer * layer) {
    if (neuron->weights != NULL) free(neuron->weights);
    if (neuron->extra != NULL) {
        if (layer->flags & FLAG_RECURRENT) {
            RecurrentCell * cell = getRecurrentCell(neuron);
            if (cell->states != NULL) free(cell->states);
            //if (cell->weights != NULL) free(cell->weights);
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
    char * func = "initConvolutionalLayer";
    Layer * previous = network->layers[index - 1];
    if (previous->type != FullyConnected) {
        fprintf(stderr,
                "FullyConnected -> Convolutional trans. not supported ATM :(");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        logerr(func, "Layer parameters is NULL!");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        logerr(func, "Convolutional Layer parameters count must be %d",
               CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    int feature_count = (int) (params[FEATURE_COUNT]);
    if (feature_count <= 0) {
        logerr(func, "FEATURE_COUNT must be > 0 (given: %d)", feature_count);
        abortLayer(network, layer);
        return 0;
    }
    double region_size = params[REGION_SIZE];
    if (region_size <= 0) {
        logerr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
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
        input_w = previous_params->parameters[OUTPUT_WIDTH];
        input_h = previous_params->parameters[OUTPUT_HEIGHT];
        double prev_area = input_w * input_h;
        if ((int) prev_area != previous_size) {
            logerr(func, "Previous size %d != %lfx%lf",
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
    if (layer->neurons == NULL) {
        logerr(func, "Layer[%d]: Could not allocate neurons!", index);
        abortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    layer->avx_activation_cache = calloc(size, sizeof(double));
    if (layer->avx_activation_cache == NULL) {
        printMemoryErrorMsg();
        abortLayer(network, layer);
        return 0;
    }
#endif
    ConvolutionalSharedParams * shared;
    shared = malloc(sizeof(ConvolutionalSharedParams));
    if (shared == NULL) {
        logerr(func, "Layer[%d]: Couldn't allocate shared params!", index);
        abortLayer(network, layer);
        return 0;
    }
    shared->feature_count = feature_count;
    shared->weights_size = (int)(region_size * region_size);
    shared->biases = malloc(feature_count * sizeof(double));
    shared->weights = malloc(feature_count * sizeof(double*));
    if (shared->biases == NULL || shared->weights == NULL) {
        logerr(func, "Layer[%d]: Could not allocate memory!", index);
        abortLayer(network, layer);
        return 0;
    }
    layer->extra = shared;
    int i, j, w;
    for (i = 0; i < feature_count; i++) {
        shared->biases[i] = gaussian_random(0, 1);
        shared->weights[i] = malloc(shared->weights_size * sizeof(double));
        if (shared->weights[i] == NULL) {
            logerr(func, "Layer[%d]: Could not allocate weights!", index);
            abortLayer(network, layer);
            return 0;
        }
        for (w = 0; w < shared->weights_size; w++) {
            shared->weights[i][w] = gaussian_random(0, 1);
        }
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            Neuron * neuron = malloc(sizeof(Neuron));
            if (neuron == NULL) {
                logerr(func, "Layer[%d]: Couldn't allocate neuron!",index);
                abortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = shared->weights_size;
            neuron->bias = shared->biases[i];
            neuron->weights = shared->weights[i];
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    if (!use_relu) {
        layer->activate = sigmoid;
        layer->derivative = sigmoid_derivative;
    } else {
        layer->activate = relu;
        layer->derivative = relu_derivative;
    }
    layer->feedforward = convolve;
    return 1;
}

int initPoolingLayer(NeuralNetwork * network, Layer * layer,
                     LayerParameters * parameters) {
    int index = layer->index;
    char * func = "initPoolingLayer";
    Layer * previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        fprintf(stderr,
                "Pooling's previous layer must be a Convolutional layer!\n");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        logerr(func, "Layer parameters is NULL!");
        abortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        logerr(func, "Convolutional Layer parameters count must be %d",
               CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    LayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        logerr(func, "Previous layer parameters is NULL!");
        abortLayer(network, layer);
        return 0;
    }
    if (previous_parameters->count < CONV_PARAMETER_COUNT) {
        logerr(func, "Convolutional Layer parameters count must be %d",
               CONV_PARAMETER_COUNT);
        abortLayer(network, layer);
        return 0;
    }
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (previous_params[FEATURE_COUNT]);
    params[FEATURE_COUNT] = (double) feature_count;
    double region_size = params[REGION_SIZE];
    if (region_size <= 0) {
        logerr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
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
    if (layer->neurons == NULL) {
        logerr(func, "Layer[%d]: Could not allocate neurons!", index);
        abortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    layer->avx_activation_cache = calloc(size, sizeof(double));
    if (layer->avx_activation_cache == NULL) {
        printMemoryErrorMsg();
        abortLayer(network, layer);
        return 0;
    }
#endif
    int i, j, w;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            Neuron * neuron = malloc(sizeof(Neuron));
            if (neuron == NULL) {
                logerr(func, "Layer[%d]: Couldn't allocate neuron!", index);
                abortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = 0;
            neuron->bias = NULL_VALUE;
            neuron->weights = NULL;
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    layer->activate = NULL;
    layer->derivative = previous->derivative;
    layer->feedforward = pool;
    return 1;
}

RecurrentCell * createRecurrentCell(Neuron * neuron, int lsize) {
    RecurrentCell * cell = malloc(sizeof(RecurrentCell));
    if (cell == NULL) return NULL;
    cell->states_count = 0;
    cell->states = NULL;
    cell->weights_size = lsize;
    if (!lsize) cell->weights = NULL;
    else cell->weights = neuron->weights + (neuron->weights_size - lsize);
    return cell;
}

int initRecurrentLayer(NeuralNetwork * network, Layer * layer, int size,int ws){
    int index = layer->index, i, j;
    ws += size;
    char * func = "initRecurrentLayer";
    layer->neurons = malloc(sizeof(Neuron*) * size);
/*#ifdef USE_AVX
    layer->avx_activation_cache = calloc(size, sizeof(double));
#endif*/
    if (layer->neurons == NULL) {
        logerr(func, "Could not allocate layer neurons!");
        abortLayer(network, layer);
        return 0;
    }
    for (i = 0; i < size; i++) {
        Neuron * neuron = malloc(sizeof(Neuron));
        if (neuron == NULL) {
            logerr(func, "Could not allocate neuron!");
            abortLayer(network, layer);
            return 0;
        }
        neuron->index = i;
        neuron->weights_size = ws;
        neuron->bias = gaussian_random(0, 1);
        neuron->weights = malloc(sizeof(double) * ws);
        if (neuron->weights ==  NULL) {
            abortLayer(network, layer);
            logerr(func, "Could not allocate neuron weights!");
            return 0;
        }
        for (j = 0; j < ws; j++) {
            neuron->weights[j] = gaussian_random(0, 1);
        }
        neuron->activation = 0;
        neuron->z_value = 0;
        layer->neurons[i] = neuron;
        neuron->extra = createRecurrentCell(neuron, size);
        if (neuron->extra == NULL) {
            abortLayer(network, layer);
            return 0;
        }
        neuron->layer = layer;
    }
    layer->flags |= FLAG_RECURRENT;
    layer->activate = tanh;
    layer->derivative = tanh_derivative;
    layer->feedforward = recurrentFeedforward;
    network->flags |= FLAG_RECURRENT;
    return 1;
}

Layer * addLayer(NeuralNetwork * network, LayerType type, int size,
                 LayerParameters* params) {
    char * func = "addLayer";
    if (network->size == 0 && type != FullyConnected) {
        logerr(func, "First layer type must be FullyConnected");
        return NULL;
    }
    Layer * layer = malloc(sizeof(Layer));
    if (layer == NULL) {
        logerr(func, "Could not allocate layer %d!", network->size);
        return NULL;
    }
    layer->network = network;
    layer->index = network->size++;
    layer->type = type;
    layer->size = size;
    layer->parameters = params;
    layer->extra = NULL;
    layer->flags = FLAG_NONE;
#ifdef USE_AVX
    layer->avx_activation_cache = NULL;
#endif
    Layer * previous = NULL;
    int previous_size = 0;
    int initialized = 0;
    //printf("Adding layer %d\n", layer->index);
    if (network->layers == NULL) {
        network->layers = malloc(sizeof(Layer*));
        if (network->layers == NULL) {
            abortLayer(network, layer);
            logerr(func, "Could not allocate network layers!");
            return NULL;
        }
        if ((network->flags & FLAG_ONEHOT) && params == NULL) {
            layer->flags |= FLAG_ONEHOT;
            LayerParameters * params = createLayerParamenters(1, (double) size);
            layer->parameters = params;
            size = 1;
            layer->size = 1;
        }
        network->input_size = size;
    } else {
        network->layers = realloc(network->layers,
                                  sizeof(Layer*) * network->size);
        if (network->layers == NULL) {
            abortLayer(network, layer);
            logerr(func, "Could not reallocate network layers!");
            return NULL;
        }
        previous = network->layers[layer->index - 1];
        if (previous == NULL) {
            abortLayer(network, layer);
            logerr(func, "Previous layer is NULL!");
            return NULL;
        }
        previous_size = previous->size;
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT) {
            LayerParameters * params = previous->parameters;
            if (params == NULL) {
                abortLayer(network, layer);
                logerr(func, "Missing layer params on onehot layer[0]!");
                return NULL;
            }
            previous_size = (int) (params->parameters[0]);
        }
        network->output_size = size;
    }
    if (type == FullyConnected || type == SoftMax) {
        layer->neurons = malloc(sizeof(Neuron*) * size);
        if (layer->neurons == NULL) {
            logerr(func, "Layer[%d]: could not allocate neurons!",
                   layer->index);
            abortLayer(network, layer);
            return NULL;
        }
#ifdef USE_AVX
        layer->avx_activation_cache = calloc(size, sizeof(double));
        if (layer->avx_activation_cache == NULL) {
            printMemoryErrorMsg();
            abortLayer(network, layer);
            return NULL;
        }
#endif
        int i, j;
        for (i = 0; i < size; i++) {
            Neuron * neuron = malloc(sizeof(Neuron));
            if (neuron == NULL) {
                abortLayer(network, layer);
                logerr(func, "Could not allocate neuron!");
                return NULL;
            }
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
            neuron->layer = layer;
            layer->neurons[i] = neuron;
        }
        if (type != SoftMax) {
            layer->activate = sigmoid;
            layer->derivative = sigmoid_derivative;
            layer->feedforward = fullFeedforward;
        } else {
            layer->activate = NULL;
            layer->derivative = NULL;
            layer->feedforward = softmaxFeedforward;
            //network->loss = crossEntropyLoss;
        }
        initialized = 1;
    } else if (type == Convolutional) {
        initialized = initConvolutionalLayer(network, layer, params);
    } else if (type == Pooling) {
        initialized = initPoolingLayer(network, layer, params);
    } else if (type == Recurrent) {
        initialized = initRecurrentLayer(network, layer, size, previous_size);
        if (initialized) network->loss = crossEntropyLoss;
    }
    if (!initialized) {
        abortLayer(network, layer);
        logerr(func, "Could not initialize layer %d!", network->size + 1);
        return NULL;
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
    free(layer->neurons);
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
            free(extra);
        } else free(extra);
    }
#ifdef USE_AVX
    if (layer->avx_activation_cache != NULL) free(layer->avx_activation_cache);
#endif
    free(layer);
}

LayerParameters * createLayerParamenters(int count, ...) {
    LayerParameters * params = malloc(sizeof(LayerParameters));
    if (params == NULL) {
        logerr(NULL, "Could not allocate Layer Parameters!");
        return NULL;
    }
    params->count = count;
    if (count == 0) params->parameters = NULL;
    else {
        params->parameters = malloc(sizeof(double) * count);
        if (params->parameters == NULL) {
            logerr(NULL, "Could not allocate Layer Parameters!");
            free(params);
            return NULL;
        }
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
        //TODO: handle memory failure
        memset(params->parameters, 0.0f, sizeof(double) * len);
        params->count = len;
    } else if (param >= params->count) {
        int len = params->count;
        int new_len = param + 1;
        double * old_params = params->parameters;
        size_t size = sizeof(double) * new_len;
        params->parameters = malloc(sizeof(double) * size);
        //TODO: handle memory failure
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

int feedforwardThroughTime(NeuralNetwork * network, double * values, int times)
{
    Layer * first = network->layers[0];
    int input_size = first->size;
    char * func = "feedforwardThroughTime";
    int i, t;
    for (t = 0; t < times; t++) {
        for (i = 0; i < input_size; i++) {
            Neuron * neuron = first->neurons[i];
            neuron->activation = values[i];
            addRecurrentState(neuron, values[i], times, t);
            if (neuron->extra == NULL) {
                logerr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
        for (i = 1; i < network->size; i++) {
            Layer * layer = network->layers[i];
            if (layer == NULL) {
                logerr(func, "Layer %d is NULL", i);
                return 0;
            }
            if (layer->feedforward == NULL) {
                logerr(func, "Layer %d feedforward function is NULL", i);
                return 0;
            }
            int ok = layer->feedforward(network, layer, times, t);
            if (!ok) return 0;
        }
        values += input_size;
    }
    return 1;
}

int feedforward(NeuralNetwork * network, double * values) {
    char * func = "feedforward";
    if (network->size == 0) {
        logerr(func, "Empty network!");
        return 0;
    }
    if (network->flags & FLAG_RECURRENT) {
        int times = (int) values[0];
        if (times <= 0) {
            logerr(func, "Recurrent times must be > 0 (found %d)", times);
            return 0;
        }
        return feedforwardThroughTime(network, values + 1, times);
    }
    Layer * first = network->layers[0];
    int input_size = first->size;
    int i;
    for (i = 0; i < input_size; i++) {
        first->neurons[i]->activation = values[i];
#ifdef USE_AVX
        first->avx_activation_cache[i] = values[i];
#endif
    }
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        if (layer == NULL) {
            logerr(func, "Layer %d is NULL!", i);
            return 0;
        }
        if (layer->feedforward == NULL) {
            logerr(func, "Layer %d feedforward function is NULL", i);
            return 0;
        }
        int success = layer->feedforward(network, layer);
        if (!success) return 0;
    }
    return 1;
}

Gradient * createLayerGradients(Layer * layer) {
    Gradient * gradients;
    char * func = "createLayerGradients";
    LayerType ltype = layer->type;
    if (ltype == Pooling) return NULL;
    int size = layer->size;
    LayerParameters * parameters = NULL;
    if (ltype == Convolutional) {
        parameters = layer->parameters;
        if (parameters == NULL) {
            logerr(func, "Layer %d parameters are NULL!", layer->index);
            return NULL;
        }
        size = (int) (parameters->parameters[FEATURE_COUNT]);
    }
    gradients = malloc(sizeof(Gradient) * size);
    if (gradients == NULL) {
        logerr(func, "Could not allocate memory!");
        return NULL;
    }
    int i, ws = 0;
    for (i = 0; i < size; i++) {
        Neuron * neuron = layer->neurons[i];
        if (ltype == Convolutional) {
            if (!ws) {
                int region_size = (int) (parameters->parameters[REGION_SIZE]);
                ws = region_size * region_size;
            }
        } else {
            ws = neuron->weights_size;
        }
        gradients[i].bias = 0;
        int memsize = sizeof(double) * ws;
        gradients[i].weights = malloc(memsize);
        if (gradients[i].weights == NULL) {
            logerr(func, "Could not allocate memory!");
            deleteLayerGradients(gradients, size);
            return NULL;
        }
        memset(gradients[i].weights, 0, memsize);
    }
    return gradients;
}

Gradient ** createGradients(NeuralNetwork * network) {
    Gradient ** gradients = malloc(sizeof(Gradient*) * network->size - 1);
    if (gradients == NULL) {
        printMemoryErrorMsg();
        return NULL;
    }
    int i;
    for (i = 1; i < network->size; i++) {
        Layer * layer = network->layers[i];
        int idx = i - 1;
        gradients[idx] = createLayerGradients(layer);
        if (gradients[idx] == NULL && layer->type != Pooling) {
            printMemoryErrorMsg();
            deleteGradients(gradients, network);
            return NULL;
        }
    }
    return gradients;
}

void deleteLayerGradients(Gradient * gradient, int size) {
    int i;
    for (i = 0; i < size; i++) {
        Gradient g = gradient[i];
        free(g.weights);
    }
    free(gradient);
}

void deleteGradients(Gradient ** gradients, NeuralNetwork * network) {
    int i;
    for (i = 1; i < network->size; i++) {
        Gradient * lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        Layer * layer = network->layers[i];
        int lsize;
        if (layer->type == Convolutional) {
            LayerParameters * params = layer->parameters;
            lsize = (int) (params->parameters[FEATURE_COUNT]);
        } else lsize = layer->size;
        deleteLayerGradients(lgradients, lsize);
    }
    free(gradients);
}

double * backpropPoolingToConv(NeuralNetwork * network, Layer * pooling_layer,
                               Layer * convolutional_layer, double * delta) {
    int conv_size = convolutional_layer->size;
    double * new_delta = malloc(sizeof(double) * conv_size);
    if (new_delta == NULL) {
        printMemoryErrorMsg();
        return NULL;
    }
    memset(new_delta, 0, sizeof(double) * conv_size);
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
            double d = delta[idx];
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
                    new_delta[nidx] = (a < neuron->activation ? 0 : d);
                }
            }
            
        }
    }
    return new_delta;
}

double * backpropConvToFull(NeuralNetwork * network, Layer* convolutional_layer,
                            Layer * full_layer, double * delta,
                            Gradient * lgradients) {
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
        Gradient * feature_gradient = &(lgradients[i]);
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            double d = delta[idx];
            feature_gradient->bias += d;
            
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
                    feature_gradient->weights[widx++] += (a * d);
                }
            }
        }
    }
    return NULL;
}

Gradient ** backprop(NeuralNetwork * network, double * x, double * y) {
    Gradient ** gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    Layer * inputLayer = network->layers[0];
    Layer * outputLayer = network->layers[netsize - 1];
    int isize = inputLayer->size;
    int osize = outputLayer->size;
    Gradient * lgradients = gradients[netsize - 2];//No gradient for input layer
    Layer * previousLayer = network->layers[outputLayer->index - 1];
    Layer * nextLayer = NULL;
    double * delta;
    double * last_delta;
    delta = malloc(sizeof(double) * osize);
    if (delta == NULL) {
        printMemoryErrorMsg();
        deleteGradients(gradients, network);
        return NULL;
    }
    memset(delta, 0, sizeof(double) * osize);
    last_delta = delta;
    int i, o, w, j, k;
    int ok = feedforward(network, x);
    if (!ok) {
        deleteGradients(gradients, network);
        free(delta);
        return NULL;
    }
    int apply_derivative = shouldApplyDerivative(network);
    double softmax_sum = 0.0;
    for (o = 0; o < osize; o++) {
        Neuron * neuron = outputLayer->neurons[o];
        double o_val = neuron->activation;
        double y_val = y[o];
        double d = 0.0;
        if (outputLayer->type != SoftMax) {
            d = o_val - y_val;
            if (apply_derivative) d *= outputLayer->derivative(neuron->z_value);
        } else {
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
        }
        delta[o] = d;
        if (outputLayer->type != SoftMax) {
            Gradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            int avx_step_len = AVXGetStepLen(wsize);
            int avx_steps = wsize / avx_step_len, avx_step;
            avx_multiply_value multiply_val = AVXGetMultiplyValFunc(wsize);
            for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                double * x_vector = previousLayer->avx_activation_cache + w;
                multiply_val(x_vector, d, gradient->weights + w, 0);
                w += avx_step_len;
            }
#endif
            for (; w < neuron->weights_size; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
            }
        }
    }
    if (outputLayer->type == SoftMax) {
        for (o = 0; o < osize; o++) {
            Neuron * neuron = outputLayer->neurons[o];
            double o_val = neuron->activation;
            if (apply_derivative) delta[o] -= (o_val * softmax_sum);
            double d = delta[o];
            Gradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            int avx_step_len = AVXGetStepLen(wsize);
            int avx_steps = wsize / avx_step_len, avx_step;
            avx_multiply_value multiply_val = AVXGetMultiplyValFunc(wsize);
            for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                double * x_vector = previousLayer->avx_activation_cache + w;
                multiply_val(x_vector, d, gradient->weights + w, 0);
                w += avx_step_len;
            }
#endif
            for (; w < neuron->weights_size; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
            }
        }
    }
    for (i = previousLayer->index; i > 0; i--) {
        Layer * layer = network->layers[i];
        previousLayer = network->layers[i - 1];
        nextLayer = network->layers[i + 1];
        lgradients = gradients[i - 1];
        int lsize = layer->size;
        LayerType ltype = layer->type;
        LayerType prev_ltype = previousLayer->type;
        if (FullyConnected == ltype) {
            delta = malloc(sizeof(double) * lsize);
            if (delta == NULL) {
                printMemoryErrorMsg();
                if (last_delta != NULL) free(last_delta);
                return NULL;
            }
            memset(delta, 0, sizeof(double) * lsize);
            for (j = 0; j < lsize; j++) {
                Neuron * neuron = layer->neurons[j];
                double sum = 0;
                for (k = 0; k < nextLayer->size; k++) {
                    Neuron * nextNeuron = nextLayer->neurons[k];
                    double weight = nextNeuron->weights[j];
                    double d = last_delta[k];
                    sum += (d * weight);
                }
                double dv = sum;
                if (shouldApplyDerivative(network))
                    dv *= layer->derivative(neuron->z_value);
                delta[j] = dv;
                Gradient * gradient = &(lgradients[j]);
                gradient->bias = dv;
                w = 0;
                int wsize = neuron->weights_size;
#ifdef USE_AVX
                int avx_step_len = AVXGetStepLen(wsize);
                int avx_steps = wsize / avx_step_len, avx_step;
                avx_multiply_value multiply_val = AVXGetMultiplyValFunc(wsize);
                for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                    double * x_vector = previousLayer->avx_activation_cache + w;
                    multiply_val(x_vector, dv, gradient->weights + w, 0);
                    w += avx_step_len;
                }
#endif
                for (; w < neuron->weights_size; w++) {
                    double prev_a = previousLayer->neurons[w]->activation;
                    gradient->weights[w] = dv * prev_a;
                }
            }
        } else if (Pooling == ltype && Convolutional == prev_ltype) {
            delta = malloc(sizeof(double) * lsize);
            if (delta == NULL) {
                printMemoryErrorMsg();
                if (last_delta != NULL) free(last_delta);
                return NULL;
            }
            memset(delta, 0, sizeof(double) * lsize);
            for (j = 0; j < lsize; j++) {
                Neuron * neuron = layer->neurons[j];
                double sum = 0;
                for (k = 0; k < nextLayer->size; k++) {
                    Neuron * nextNeuron = nextLayer->neurons[k];
                    double weight = nextNeuron->weights[j];
                    double d = last_delta[k];
                    sum += (d * weight);
                }
                double dv = sum;
                if (shouldApplyDerivative(network))
                    dv *= layer->derivative(neuron->z_value);
                delta[j] = dv;
            }
            free(last_delta);
            last_delta = delta;
            delta = backpropPoolingToConv(network, layer,
                                          previousLayer, last_delta);
        } else if (Convolutional == ltype && FullyConnected == prev_ltype) {
            delta = backpropConvToFull(network, layer, previousLayer,
                               last_delta, lgradients);
        } else {
            fprintf(stderr, "Backprop from %s to %s not suported!\n",
                    getLayerTypeLabel(layer),
                    getLayerTypeLabel(previousLayer));
            deleteGradients(gradients, network);
            free(last_delta);
            if (delta != last_delta) free(delta);
            return NULL;
        }
        free(last_delta);
        last_delta = delta;
        if (delta == NULL && Convolutional != ltype) {
            deleteGradients(gradients, network);
            return NULL;
        }
    }
    if (delta != NULL) free(delta);
    return gradients;
}

Gradient ** backpropThroughTime(NeuralNetwork * network, double * x,
                                double * y, int times)
{
    // TODO: RECURRENT OUTPUT LAYER MUST BE SOFTMAX!!!

    Gradient ** gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    Layer * inputLayer = network->layers[0];
    Layer * outputLayer = network->layers[netsize - 1];
    int onehot = (outputLayer->flags & FLAG_ONEHOT);
    int isize = inputLayer->size;
    int osize = outputLayer->size;
    int bptt_truncate = BPTT_TRUNCATE;

    int i, o, w, j, k, t, tt;
    int ok = feedforwardThroughTime(network, x, times);
    if (!ok) {
        deleteGradients(gradients, network);
        return NULL;
    }
    
    int last_t = times - 1;
    for (t = last_t; t >= 0; t--) {
        int lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        int ysize = (onehot ? 1 : osize);
        int time_offset = t * ysize;
        double * time_y = y + time_offset;
        
        Gradient * lgradients = gradients[netsize - 2];//No grad. in input layer
        Layer * previousLayer = network->layers[outputLayer->index - 1];
        Layer * nextLayer = NULL;
        double * delta;
        double * last_delta;
        delta = malloc(sizeof(double) * osize);
        if (delta == NULL) {
            printMemoryErrorMsg();
            return NULL;
        }
        memset(delta, 0, sizeof(double) * osize);
        last_delta = delta;

        double softmax_sum = 0.0;
        int apply_derivative = shouldApplyDerivative(network);
        for (o = 0; o < osize; o++) {
            Neuron * neuron = outputLayer->neurons[o];
            RecurrentCell * cell = getRecurrentCell(neuron);
            double o_val = cell->states[t];
            double y_val;
            if (onehot)
                y_val = ((int) *(time_y) == o);
            else
                y_val = time_y[o];
            double d = 0.0;
            
            // Recurrent Output Layer must be SoftMax ??
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
            delta[o] = d;
        }
        // SoftMax
        for (o = 0; o < osize; o++) {
            Neuron * neuron = outputLayer->neurons[o];
            RecurrentCell * cell = getRecurrentCell(neuron);
            double o_val = cell->states[t];
            if (apply_derivative) delta[o] -= (o_val * softmax_sum);
            double d = delta[o];
            Gradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            w = 0;
#ifdef USE_AVX
            int avx_step_len = AVXGetStepLen(neuron->weights_size);
            int avx_steps = neuron->weights_size / avx_step_len,
                avx_step;
            avx_multiply_value multiply;
            multiply = AVXGetMultiplyValFunc(neuron->weights_size);
            for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                double * xv = previousLayer->avx_activation_cache + w;
                xv += (t * previousLayer->size);
                double * weights = gradient->weights + w;
                multiply(xv, d, weights, AVX_STORE_MODE_ADD);
                w += avx_step_len;
            }
#endif
            for (; w < neuron->weights_size; w++) {
                Neuron * prev_neuron = previousLayer->neurons[w];
                RecurrentCell * prev_cell = getRecurrentCell(prev_neuron);
                double prev_a = prev_cell->states[t];
                gradient->weights[w] += (d * prev_a);
            }
        }
        
        for (i = previousLayer->index; i > 0; i--) {
            Layer * layer = network->layers[i];
            previousLayer = network->layers[i - 1];
            nextLayer = network->layers[i + 1];
            lgradients = gradients[i - 1];
            int lsize = layer->size;
            LayerType ltype = layer->type;
            LayerType prev_ltype = previousLayer->type;
            
            delta = malloc(sizeof(double) * lsize);
            if (delta == NULL) {
                printMemoryErrorMsg();
                if (last_delta != NULL) free(last_delta);
                return NULL;
            }
            memset(delta, 0, sizeof(double) * lsize);
            for (j = 0; j < lsize; j++) {
                Neuron * neuron = layer->neurons[j];
                RecurrentCell * cell = getRecurrentCell(neuron);
                double sum = 0;
                for (k = 0; k < nextLayer->size; k++) {
                    Neuron * nextNeuron = nextLayer->neurons[k];
                    double weight = nextNeuron->weights[j];
                    double d = last_delta[k];
                    sum += (d * weight);
                }
                double dv = sum * layer->derivative(cell->states[t]);
                delta[j] = dv;
            }
            free(last_delta);
            last_delta = delta;
            delta = malloc(sizeof(double) * lsize);
            if (delta == NULL) {
                printMemoryErrorMsg();
                if (last_delta != NULL) free(last_delta);
                return NULL;
            }
            memset(delta, 0, sizeof(double) * lsize);
            
            for (tt = t; tt >= lowest_t; tt--) {
                for (j = 0; j < lsize; j++) {
                    Neuron * neuron = layer->neurons[j];
                    RecurrentCell * cell = getRecurrentCell(neuron);
                    Gradient * gradient = &(lgradients[j]);
                    double dv = last_delta[j];
                    gradient->bias += dv;
                    
                    if (previousLayer->flags & FLAG_ONEHOT) {
                        LayerParameters * params = previousLayer->parameters;
                        if (params == NULL) {
                            fprintf(stderr, "Layer %d params are NULL!\n",
                                    previousLayer->index);
                            if (last_delta != NULL) free(last_delta);
                            return NULL;
                        }
                        int vector_size = (int) params->parameters[0];
                        assert(vector_size > 0);
                        Neuron * prev_n = previousLayer->neurons[0];
                        RecurrentCell * prev_c = getRecurrentCell(prev_n);
                        double prev_a = prev_c->states[tt];
                        assert(prev_a < vector_size);
                        w = (int) prev_a;
                        gradient->weights[w] += dv;
                    } else {
                        int ws = neuron->weights_size;
                        if (Recurrent == ltype) ws -= cell->weights_size;
                        for (w = 0; w < ws; w++) {
                            Neuron * prev_n = previousLayer->neurons[w];
                            RecurrentCell * prev_c = getRecurrentCell(prev_n);
                            double prev_a = prev_c->states[t];
                            gradient->weights[w] += (dv * prev_a);
                        }
                    }
                    
                    if (Recurrent == ltype && tt > 0) {
                        int w_offs = neuron->weights_size - cell->weights_size;
                        double rsum = 0.0;
                        w = 0;
#ifdef USE_AVX                        
                        int avx_step_len = AVXGetStepLen(cell->weights_size);
                        int avx_steps = cell->weights_size / avx_step_len,
                            avx_step;
                        avx_multiply_value multiply;
                        multiply = AVXGetMultiplyValFunc(cell->weights_size);
                        for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                            double * xv = layer->avx_activation_cache + w;
                            xv += ((tt - 1) * cell->weights_size);
                            double * weights = gradient->weights + w_offs + w;
                            multiply(xv, dv, weights, AVX_STORE_MODE_ADD);
                            w += avx_step_len;
                        }
#endif
                        for (; w < cell->weights_size; w++) {
                            Neuron * rn = layer->neurons[w];
                            RecurrentCell * rc = getRecurrentCell(rn);
                            double a = rc->states[tt - 1];
                            gradient->weights[w_offs + w] += (dv * a);
                            //sum += (delta[w] * );
                        }
                        for (w = 0; w < cell->weights_size; w++) {
                            Neuron * rn = layer->neurons[w];
                            RecurrentCell * rc = getRecurrentCell(rn);
                            double rw = rc->weights[neuron->index];
                            rsum += (last_delta[rn->index] * rw);
                        }
                        double prev_a = cell->states[tt - 1];
                        delta[neuron->index] =
                            rsum * layer->derivative(prev_a);
                    }

                }
                
                free(last_delta);
                last_delta = delta;
                delta = malloc(sizeof(double) * lsize);
                if (delta == NULL) {
                    printMemoryErrorMsg();
                    if (last_delta != NULL) free(last_delta);
                    return NULL;
                }
                memset(delta, 0, sizeof(double) * lsize);
            }
            
            free(last_delta);
            last_delta = delta;
        }
        if (delta != NULL) free(delta);
    }
    return gradients;
}

double updateWeights(NeuralNetwork * network, double * training_data,
                     int batch_size, double rate, ...) {
    double r = rate / (double) batch_size;
    int i, j, k, w, netsize = network->size, dsize = netsize - 1, times;
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    Gradient ** gradients = createGradients(network);
    if (gradients == NULL) {
        network->status = STATUS_ERROR;
        return -999.0;
    }
    char * func = "updateWeights";
    Gradient ** bp_gradients = NULL;
    double ** series = NULL;
    int is_recurrent = network->flags & FLAG_RECURRENT;
    if (is_recurrent) {
        va_list args;
        va_start(args, rate);
        series = va_arg(args, double**);
        va_end(args);
        if (series == NULL) {
            logerr(func, "Series is NULL");
            network->status = STATUS_ERROR;
            deleteGradients(gradients, network);
            return -999.0;
        }
    }
    double * x;
    double * y;
    for (i = 0; i < batch_size; i++) {
        if (series == NULL) {
            int element_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += element_size;
            bp_gradients = backprop(network, x, y);
        } else {
            x = series[i];
            times = (int) *(x++);
            if (times == 0) {
                logerr(func, "Series len must b > 0. (batch = %d)", i);
                deleteGradients(gradients, network);
                return -999.0;
            }
            y = x + (times * training_data_size);
            bp_gradients = backpropThroughTime(network, x, y, times);
        }
        if (bp_gradients == NULL) {
            network->status = STATUS_ERROR;
            deleteGradients(gradients, network);
            return -999.0;
        }
        for (j = 0; j < dsize; j++) {
            Layer * layer = network->layers[j + 1];
            Gradient * lgradients_bp = bp_gradients[j];
            Gradient * lgradients = gradients[j];
            if (lgradients == NULL) continue;
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
                Gradient * gradient_bp = &(lgradients_bp[k]);
                Gradient * gradient = &(lgradients[k]);
                gradient->bias += gradient_bp->bias;
                for (w = 0; w < wsize; w++) {
                    gradient->weights[w] += gradient_bp->weights[w];
                }
            }
        }
        deleteGradients(bp_gradients, network);
    }
    for (i = 0; i < dsize; i++) {
        Gradient * lgradients = gradients[i];
        if (lgradients == NULL) continue;
        Layer * layer = network->layers[i + 1];
        LayerType ltype = layer->type;
        int l_size;
        ConvolutionalSharedParams * shared = NULL;
        if (ltype == Convolutional) {
            LayerParameters * params = layer->parameters;
            l_size = (int) (params->parameters[FEATURE_COUNT]);
            shared = getConvSharedParams(layer);
        } else l_size = layer->size;
        for (j = 0; j < l_size; j++) {
            Gradient * g = &(lgradients[j]);
            if (shared == NULL) {
                Neuron * neuron = layer->neurons[j];
                neuron->bias = neuron->bias - r * g->bias;
                for (k = 0; k < neuron->weights_size; k++) {
                    double w = neuron->weights[k];
                    neuron->weights[k] = w - r * g->weights[k];
                }
            } else {
                double b = shared->biases[j];
                shared->biases[j] = b - r * g->bias;
                double * weights = shared->weights[j];
                for (k = 0; k < shared->weights_size; k++) {
                    double w = weights[k];
                    weights[k] = w - r * g->weights[k];
                }
            }
        }
    }
    deleteGradients(gradients, network);
    Layer * out = network->layers[netsize - 1];
    int onehot = out->flags & FLAG_ONEHOT;
    if (onehot) label_data_size = 1;
    if (is_recurrent) label_data_size *= times;
    double outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (!is_recurrent)
            outputs[i] = out->neurons[i]->activation;
        else {
            if (onehot) {
                int idx = (int) *(y + i);
                Neuron * n = out->neurons[idx];
                RecurrentCell * cell = getRecurrentCell(n);
                outputs[i] = cell->states[i];
            } else fetchRecurrentOutputState(out, outputs, i, 0);
        }
    }
    //TODO: configurable loss function
    int onehot_s = (onehot ? out->size : 0);
    return network->loss(outputs, y, label_data_size, onehot_s);
}

double gradientDescent(NeuralNetwork * network,
                       double * training_data,
                       int element_size,
                       int elements_count,
                       double learning_rate,
                       int batch_size,
                       int flags,
                       int epochs) {
    int batches_count = elements_count / batch_size;
    double ** series = NULL;
    if (network->flags & FLAG_RECURRENT) {
        if (series == NULL) {
            Layer * out = network->layers[network->size - 1];
            int o_size = (out->flags & FLAG_ONEHOT ? 1 : network->output_size);
            series = getRecurrentSeries(training_data,
                                        elements_count,
                                        network->input_size,
                                        o_size);
            if (series == NULL) {
                network->status = STATUS_ERROR;
                return -999.00;
            }
        }
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffleSeries(series, elements_count);
    } else {
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffle(training_data, elements_count, element_size);
    }
    int offset = (element_size * batch_size), i;
    double err = 0.0;
    for (i = 0; i < batches_count; i++) {
        network->current_batch = i;
        printf("\rEpoch %d/%d: batch %d/%d", network->current_epoch + 1, epochs,
               i + 1, batches_count);
        fflush(stdout);
        err += updateWeights(network, training_data, batch_size,
                             learning_rate, series);
        if (network->status == STATUS_ERROR) {
            if (series != NULL) free(series - (i * batches_count));
            return -999.00;
        }
        if (series == NULL) training_data += offset;
        else series += batch_size;
    }
    if (series != NULL) free(series - (batch_size * batches_count));
    return err / (double) batches_count;
}

float validate(NeuralNetwork * network, double * test_data, int data_size,
               int log) {
    int i, j;
    float accuracy = 0.0f;
    int correct_results = 0;
    float correct_amount = 0.0f;
    Layer * output_layer = network->layers[network->size - 1];
    int input_size = network->input_size;
    int output_size = network->output_size;
    int y_size = output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int element_size = input_size + output_size;
    int elements_count;
    double ** series = NULL;
    if (network->flags & FLAG_RECURRENT) {
        // First training data number for Recurrent networks must indicate
        // the data elements count
        elements_count = (int) *(test_data++);
        data_size--;
        if (onehot) y_size = 1;
        series = getRecurrentSeries(test_data,
                                    elements_count,
                                    input_size,
                                    y_size);
        if (series == NULL) {
            network->status = STATUS_ERROR;
            return -999.0f;
        }
    } else elements_count = data_size / element_size;
    //double outputs[output_size];
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
        double * inputs = NULL;
        double * expected = NULL;
        int times = 0;
        if (series == NULL) {
            // Not Recurrent
            inputs = test_data;
            test_data += input_size;
            expected = test_data;
            
            int ok = feedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            
            double max = 0.0;
            int omax = 0;
            int emax = 0;
            for (j = 0; j < output_size; j++) {
                Neuron * neuron = output_layer->neurons[j];
                if (neuron->activation > max) {
                    max = neuron->activation;
                    omax = j;
                }
            }
            if (!onehot)
                emax = arrayMaxIndex(expected, output_size);
            else
                emax = *(expected + (times - 1));
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            // Recurrent
            inputs = series[i];
            times = (int) (*inputs);
            if (times == 0) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            expected = inputs + 1 + (times * input_size);
            
            int ok = feedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            
            int label_data_size = y_size * times;
            int correct_states = 0;
            double outputs[label_data_size];
            for (j = 0; j < label_data_size; j++) {
                fetchRecurrentOutputState(output_layer, outputs, j, onehot);
                if (onehot && (outputs[j] == expected[j])) correct_states++;
                else if (!onehot && j > 0 && (j % y_size) == 0) {
                    int t = (j / y_size) - 1;
                    int omax = arrayMaxIndex(outputs + (t * y_size), y_size);
                    int emax = arrayMaxIndex(expected + (t * y_size), y_size);
                    if (emax == omax) correct_states++;
                }
            }
            correct_amount += (float) correct_states / (float) times;
        }
    }
    if (log) printf("\n");
    time(&end_t);
    if (log) printf("Completed in %ld sec.\n", end_t - start_t);
    if (series == NULL) {
        accuracy = (float) correct_results / (float) elements_count;
    } else {
        accuracy = correct_amount / (float) elements_count;
        free(series);
    }
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
           int flags,
           double * test_data,
           int test_size) {
    int i, elements_count;
    int element_size = network->input_size + network->output_size;
    int valid = verifyNetwork(network);
    if (!valid) {
        network->status = STATUS_ERROR;
        return;
    }
    if (network->flags & FLAG_RECURRENT) {
        // First training data number for Recurrent networks must indicate
        // the data elements count
        elements_count = (int) *(training_data++);
        data_size--;
    } else elements_count = data_size / element_size;
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
    double prev_err = 0.0;
    for (i = 0; i < epochs; i++) {
        network->current_epoch = i;
        double err = gradientDescent(network, training_data, element_size,
                                     elements_count, learning_rate,
                                     batch_size, flags, epochs);
        if (network->status == STATUS_ERROR) {
            fprintf(stderr, "\nAn error occurred while training, aborting!\n");
            return;
        }
        char accuracy_msg[255] = "";
        if (test_data != NULL) {
            int batches_count = elements_count / batch_size;
            printf("\rEpoch %d/%d: batch %d/%d, validating...",
                   network->current_epoch + 1,
                   epochs,
                   network->current_batch + 1,
                   batches_count);
            float acc = validate(network, test_data, test_size, 0);
            printf("\rEpoch %d/%d: batch %d/%d",
                   network->current_epoch + 1,
                   epochs,
                   network->current_batch + 1,
                   batches_count);
            sprintf(accuracy_msg, ", acc = %.2f,", acc);
        }
        time(&epoch_t);
        time_t elapsed_t = epoch_t - e_t;
        e_t = epoch_t;
        if (i > 0 && err > prev_err && (flags & TRAINING_ADJUST_RATE))
            learning_rate *= 0.5;
        prev_err = err;
        printf(", loss = %.2lf%s (%ld sec.)\n", err, accuracy_msg, elapsed_t);
    }
    time(&end_t);
    printf("Completed in %ld sec.\n", end_t - start_t);
    network->status = STATUS_TRAINED;
}

float test(NeuralNetwork * network, double * test_data, int data_size) {
    return validate(network, test_data, data_size, 1);
}

int verifyNetwork(NeuralNetwork * network) {
    if (network == NULL) {
        logerr("verifyNetwork", "Network is NULL");
        return 0;
    }
    int size = network->size, i;
    Layer * previous = NULL;
    for (i = 0; i < size; i++) {
        Layer * layer = network->layers[i];
        if (layer == NULL) {
            logerr("verifyNetwork", "Layer[%d] is NULL", i);
            return 0;
        }
        int lsize = layer->size;
        int ltype = layer->type;
        if (i == 0) {
            if (ltype != FullyConnected) {
                logerr("verifyNetwork", "Layer[%d] type must be '%s'",
                       i, getLabelForType(FullyConnected));
                return 0;
            }
            if (layer->flags & FLAG_ONEHOT) {
                LayerParameters * params = layer->parameters;
                if (params == NULL) {
                    logerr("verifyNetwork",
                           "Layer[%d] uses a onehot vector index as input, "
                           "but it has no parameters", i);
                    return 0;
                }
                if (params->count < 1) {
                    logerr("verifyNetwork",
                           "Layer[%d] uses a onehot vector index as input, "
                           "but parameters count is < 1", i);
                    return 0;
                }
            }
        }
        if (ltype == Pooling && previous->type != Convolutional) {
            logerr("verifyNetwork", "Layer[%d] type is Pooling, "
                   "but previous type is not Convolutional", i);
            return 0;
        }
        if (layer->activate == sigmoid &&
            layer->derivative != sigmoid_derivative) {
            logerr("verifyNetwork",
                   "Layer[%d] activate function is sigmoid, "
                   "but derivative function is not sigmoid_derivative", i);
            return 0;
        }
        if (layer->activate == relu && layer->derivative != relu_derivative) {
            logerr("verifyNetwork",
                   "Layer[%d] activate function is relu, "
                   "but derivative function is not relu_derivative", i);
            return 0;
        }
        if (layer->activate == tanh && layer->derivative != tanh_derivative) {
            logerr("verifyNetwork",
                   "Layer[%d] activate function is tanh, "
                   "but derivative function is not tanh_derivative", i);
            return 0;
        }
        previous = layer;
    }
    return 1;
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


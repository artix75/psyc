#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural.h"

#define INPUTS_SIZE (28 * 28)


double normalized_rand() {
    srand(time(NULL));
    int r = rand();
    return ((double) r / (double) RAND_MAX);
}

int main(int argc, char** argv) {
    NeuralNetwork * network = createNetwork();
    addLayer(network, FullyConnected, INPUTS_SIZE, NULL);
    addLayer(network, FullyConnected, 30, NULL);
    addLayer(network, FullyConnected, 10, NULL);
    
    double values[INPUTS_SIZE];
    int i;
    for (i = 0; i < INPUTS_SIZE; i++) {
        values[i] = normalized_rand();
    }
    feedforward(network, values);
    
    deleteNetwork(network);
    
    double nums[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    testShuffle(nums, 6, 2);
    exit(0);
}

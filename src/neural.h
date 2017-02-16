#define PARAM_INPUT_SIZE_X    0
#define PARAM_INPUT_SIZE_Y    1
#define PARAM_OUTPUT_SIZE_X   2
#define PARAM_OUTPUT_SIZE_Y   3
#define PARAM_REGION_SIZE     4
#define PARAM_STRIDE          5

#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2

#define CONV_PARAMETER_COUNT 6


typedef double (*ActivationFunction)(double);
typedef void (*FeedforwardFunction)(void * network, void * layer);

typedef enum {
    FullyConnected,
    Convolutional,
    Pooling,
    Recurrent,
    LSTM
} LayerType;

typedef struct {
    int count;
    double * parameters;
} LayerParameters;

typedef struct {
    int index;
    int weights_size;
    double bias;
    double * weights;
    double activation;
    double z_value;
} Neuron;

typedef struct {
    LayerType type;
    int index;
    int size;
    ActivationFunction activate;
    ActivationFunction prime;
    FeedforwardFunction feedforward;
    Neuron ** neurons;
} Layer;

typedef struct {
    int size;
    Layer ** layers;
    unsigned char status;
    int input_size;
    int output_size;
    int current_epoch;
} NeuralNetwork;

NeuralNetwork * createNetwork();
int loadNetwork(NeuralNetwork * network, const char* filename);
Layer * addLayer(NeuralNetwork * network, LayerType type, int size,
                 LayerParameters* params);
void feedforward(NeuralNetwork * network, double * values);

void deleteNetwork(NeuralNetwork * network);
void deleteLayer(Layer * layer);
void deleteNeuron(Neuron * neuron);
void train(NeuralNetwork * network,
           double * training_data,
           int data_size,
           int epochs,
           double learning_rate,
           int batch_size);
float test(NeuralNetwork * network, double * test_data, int data_size);
int arrayMax(double * array, int len);

void testShuffle(double * array, int size, int element_size);


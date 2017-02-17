#define FEATURE_COUNT   0
#define REGION_SIZE     1
#define STRIDE          2
#define INPUT_WIDTH     3
#define INPUT_HEIGHT    4
#define OUTPUT_WIDTH    5
#define OUTPUT_HEIGHT   6
#define PADDING         7

#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2

#define CONV_PARAMETER_COUNT 8
#define NULL_VALUE -9999999.99


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
    int feature_count;
    int weights_size;
    double * biases;
    double ** weights;
} ConvolutionalSharedParams;

typedef struct {
    int index;
    int weights_size;
    double bias;
    double * weights;
    double activation;
    double z_value;
    void * extra;
} Neuron;

typedef struct {
    LayerType type;
    int index;
    int size;
    LayerParameters * parameters;
    ActivationFunction activate;
    ActivationFunction prime;
    FeedforwardFunction feedforward;
    Neuron ** neurons;
    void * extra;
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
Layer * addConvolutionalLayer(NeuralNetwork * network, LayerParameters* params);
Layer * addPoolingLayer(NeuralNetwork * network, LayerParameters* params);
LayerParameters * createLayerParamenters(int count, ...);
void setLayerParameter(LayerParameters * params, int param, double value);
void addLayerParameter(LayerParameters * params, double val);
LayerParameters * createConvolutionalParameters(double feature_count,
                                                double region_size,
                                                int stride,
                                                int padding);
void deleteLayerParamenters(LayerParameters * params);
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
//int arrayMaxIndex(double * array, int len);
char * getLayerTypeLabel(Layer * layer);

void testShuffle(double * array, int size, int element_size);


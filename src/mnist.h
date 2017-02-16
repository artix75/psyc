#define TRAINING_DATA   0
#define TEST_DATA       1

int loadMNISTData(int type,
                  const char * images_file,
                  const char * labels_file,
                  double ** data);

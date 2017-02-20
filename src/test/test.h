#define NOT_PERFORMED -1

typedef int (* TestFunction) (void* test_case, void* test);
typedef void (* SetupFunction) (void* test_case);
typedef void (* TeardownFunction) (void* test_case);

typedef struct {
    char * name;
    char * error_message;
    int status;
    TestFunction run;
} Test;

typedef struct {
    char * name;
    SetupFunction setup;
    TeardownFunction teardown;
    int count;
    Test * tests;
    void ** data;
} TestCase;

TestCase * createTest(char * name);
Test * addTest(TestCase * test_case, char * name, char * errmsg,
               TestFunction func);
int performTests(TestCase * test_case);
void deleteTest(TestCase * test_case);

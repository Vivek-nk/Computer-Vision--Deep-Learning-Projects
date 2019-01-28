//  DNN - Deep Neural Network Score system
//
//  Hardware Software Interface - ELEC 5511 - Midterm 2017
//  
//  Subtmitted by Vivek Nambidi 
//  Student ID : 107636214

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <unistd.h>     /* getopt() */
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include<sys/time.h>
#include<time.h>
#include<emmintrin.h> /* SSE2 */


// To get number of CPU CYCLES
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


// Defining 3 types of layers = INPUT, HIDDEN, OUTPUT
typedef enum {LR_INPUT=1, LR_HIDDEN, LR_OUTPUT} layer_t;

#define INFO_ACTIVATION 0x01
#define INFO_WEIGHTS    0x02
#define INFO_BIASES     0x04
#define INFO_Z          0x08

typedef struct Layer Layer;
typedef struct Network Network;

// Defining a structure of layer
struct Layer
{
    int type;
    int level;
    int nodes;
    
    struct Layer *prev;
    struct Layer *next;
    
    // Description
    float *weights;
    float *biases;
    
    // Output of computation: Activation or Result
    float *z;              // Calculation of sum
    float *activation;
    
};

// Defining structure of neural network
struct Network
{
    int OutputNodes;
    int InputNodes;
    int HiddenNodes;
    int HiddenLayers;
    
    Layer *input;
    Layer *output;
};


#define MAX_INPUT 10000
float Input[MAX_INPUT];
int InputSize = 0;

Layer *Create_Layer(int type, int level, int number_nodes, Layer *previous)
{
    Layer *current = (Layer *) malloc(sizeof(Layer));
    
    if (current == NULL) {
        printf("Error: Allocation of layer\n");
        exit(-1);
    }
    current->type = type;
    current->level = level;
    current->nodes = number_nodes;
    
    current->prev = NULL;
    current->next = NULL;
    
    if (type == LR_INPUT) {
        current->biases = NULL;
        current->weights = NULL;
    }
    
    
    if (type == LR_OUTPUT || type == LR_HIDDEN) {
        current->biases = (float *) malloc(sizeof(float)*current->nodes);
        
        if (previous == NULL) {
            printf("Error: Allocation of layer\n");
            exit(-1);
        }
        
        current->prev = previous;
        current->weights = (float *) malloc(sizeof(float)*current->nodes* previous->nodes);
        current->biases = (float *) malloc(sizeof(float)*current->nodes);
        
        // z is used as a temporary before the activation function is applied
        current->z = (float *) malloc(sizeof(float)*current->nodes);
    }
    
    // All layers get an activation array (Input's will be stored in activation)
    current->activation = (float *) malloc(sizeof(float)*current->nodes);
    
    return current;
}


// Function to assign Layer weights 
void Assign_Layer_Weights(char *str, Layer *layer)
{
    char *pch;
    int level;
    
    // Split into tokens
    //printf("Working on Weights %s\n", str);
    pch = strtok (str," :");
    
    pch = strtok (NULL," :");
    level = atoi(pch);
    layer->level = level;
    
    int index = 0;
    pch = strtok (NULL, " :");
    while (pch != NULL)
    {
        //printf ("Weight %d -> %s\n",index, pch);
        layer->weights[index] = atof(pch);
        index++;
        pch = strtok (NULL, " :");
        
    }
}

// Function to assign Layer Biases
void Assign_Layer_Biases(char *str, Layer *layer)
{
    char *pch;
    int level;
    
    // Split into tokens
    //printf("Working on Biases %s\n", str);
    pch = strtok (str," :");
    
    pch = strtok (NULL," :");
    level = atoi(pch);
    
    int index = 0;
    pch = strtok (NULL, " :");
    while (pch != NULL)
    {
        //printf ("Biases %d -> %s\n",index, pch);
        layer->biases[index] = atof(pch);
        index++;
        pch = strtok (NULL, " :");
    }
}


void Assign_Layer_Attributes(FILE *file, Layer *layer)
{
    char line[100000];
    
    // Read weights
    fgets(line, 100000, file);
    Assign_Layer_Weights(line, layer);
    
    // Read Biases
    fgets(line, 100000, file);
    Assign_Layer_Biases(line, layer);
}

// Creating Neural Network Architecture
Network *Create_Neural_Network(char *filename)
{
    Layer *layer, *newLayer, *prevLayer;
    int InputNodes, HiddenNodes, HiddenLayers, OutputNodes;
    int layerIndex;
    Network *network = (Network *) malloc(sizeof(Network));
    
    if (network == NULL) {
        printf("Error: Allocation of network\n");
        exit(-1);
    }
    
    FILE* file = fopen (filename, "r");
    if (file == NULL) {
        printf("Error: opening of network file\n");
        exit(-1);
    }
    
    // Reat the input format
    fscanf(file, "Input Nodes : %d\n", &InputNodes);
    
    // Create the input layer
    network->input = Create_Layer(LR_INPUT,0,InputNodes,NULL);
    
    // Read the hidden format
    fscanf(file, "Hidden Layers : %d\n", &HiddenLayers);
    fscanf(file, "Hidden Nodes : %d\n", &HiddenNodes);
    
    prevLayer = network->input;
    
    // Create the hidden layers
    for (layerIndex=0; layerIndex < HiddenLayers; layerIndex++) {
        newLayer = Create_Layer(LR_HIDDEN,layerIndex,HiddenNodes,prevLayer);
        prevLayer->next = newLayer;
        prevLayer = newLayer;
    }
    
    // Read the output format
    fscanf(file, "Output Nodes : %d\n", &OutputNodes);
    
    // Create the output layers
    network->output= Create_Layer(LR_OUTPUT,0,OutputNodes,prevLayer);
    prevLayer->next = network->output;
    
    // Assign the data structure information
    network->InputNodes  = InputNodes;
    network->HiddenLayers= HiddenLayers;
    network->HiddenNodes = HiddenNodes;
    network->OutputNodes = OutputNodes;
    
    // Read in the weights and biases
    for (layer = network->input->next; layer != network->output; layer = layer->next) {
        Assign_Layer_Attributes(file, layer);
    }
    
    // Read and assign the output
    Assign_Layer_Attributes(file, network->output);
    
    return network;
}

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ";");
         tok && *tok;
         tok = strtok(NULL, ";\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

// Function to read Inputs from a INPUT FILE 
void Read_Input(const char* filename, Network *network)
{
    float fValue;
    int index;
    char *delimiter = ",";
    char line[100000];
    
    FILE* file = fopen (filename, "r");
    if (file == NULL) {
        printf("Error: opening of input file\n");
        exit(-1);
    }
    
    InputSize = 0;
    while (fgets(line, 100000, file))
    {
        const char* tok;
        for (tok = strtok(line, delimiter);
             tok && *tok;
             tok = strtok(NULL, "\n"))
        {
            fValue = atof(tok);
            //printf("%f ",fValue);
            Input[InputSize++] = fValue;
        }
    }
    
    // Assign input to the Input nodes
    for (index=0; index < InputSize; index++) {
        network->input->activation[index] = Input[index];
    }
    
#if 0
    for (index=0; index < InputSize; index++) {
        printf ("%f ", Input[index]);
    }
    printf("\n");
#endif
    
}

// Function to calculate the activation = sigmoid function
float Sigmoid(float z)
{
    return 1 / (1 + expf(-z));
}


void Forward(float *L0_Input, int L0_Size, float *L1_Weights, float *L1_Bias, int L1_Size,
             float *L1_Z, float *L1_Output)
{
    int inputNode;
    int index;
    float sum;
    
    
    for (index=0; index < L1_Size; index++) {
        sum = 0.0;
        for (inputNode = 0; inputNode < L0_Size; inputNode++) {
            sum = sum + L0_Input[inputNode] * L1_Weights[L1_Size*index + inputNode];
        }
        sum = sum + L1_Bias[index];
        L1_Z[index] = sum;
    }
    
    // Apply activation function
    for (index=0; index < L1_Size; index++) {
        L1_Output[index] = Sigmoid(L1_Z[index]);
    }
}

// Optmizing the Code using SSE instructions
void Forward_SSE(float *L0_Input, int L0_Size, float *L1_Weights, float *L1_Bias, int L1_Size,
                 float *L1_Z, float *L1_Output)
{

}


void Execute_Layer_Forward(Layer *L0, Layer *L1)
{
    Forward(L0->activation, L0->nodes, L1->weights, L1->biases, L1->nodes, L1->z, L1->activation);
}

void Execute_Layer_Forward_SSE(Layer *L0, Layer *L1)
{
    Forward_SSE(L0->activation, L0->nodes, L1->weights, L1->biases, L1->nodes, L1->z, L1->activation);
}

void Run_Network_Forward(Network *network)
{
    Layer *layer;
    for (layer = network->input->next; layer != NULL; layer = layer->next) {
        // printf("Running layer %d -> %d\n", layer->prev->type, layer->type);
        Execute_Layer_Forward(layer->prev, layer);
    }
}
void Run_Network_Forward_SSE(Network *network)
{
    Layer *layer;
    for (layer = network->input->next; layer != NULL; layer = layer->next) {
        // printf("Running layer %d -> %d\n", layer->prev->type, layer->type);
        Execute_Layer_Forward_SSE(layer->prev, layer);
    }
}

// Function to Print the values of the neural network architecture
void Print_Layer(Layer *layer, int output)
{
    int index;
    int weights;
    
    switch(layer->type)
    {
        case LR_INPUT:
            printf("Input Layer\n");
            break;
        case LR_HIDDEN:
            printf("Hidden Layer %d\n",layer->level);
            break;
        case LR_OUTPUT:
            printf("OUTPUT LAYER\n");
            break;
    }
    
    for (index=0 ; index < layer->nodes; index++) {
        printf("Node %d\n", index);
        
        if (output & INFO_BIASES) {
            printf("  Bias %f\n", layer->biases[index]);
        }
        
        if (output & INFO_WEIGHTS) {
            if (layer->prev) {
                printf("weights : ");
                for (weights=0 ; weights < layer->prev->nodes; weights++) {
                    printf(" %f",layer->prev->weights[index*layer->nodes + weights]);
                }
                printf("\n");
            }
        }
        
        
        if (output & INFO_Z) {
            printf("Z : %3.2f\n",layer->z[index]);
        }
        if (output & INFO_ACTIVATION) {
            printf("ACTIVATION = %3.2f\n",layer->activation[index]);
        }
        
    }
    
}

int main(int argc, char * argv[])
{

    char *inputFilename = NULL;
    char *networkFilename = NULL;
    char *outputFilename = NULL;
    
    Network *network;
    
    if (argc != 4) {
        printf("Arguments not set\n");
        exit(-1);
    }
    inputFilename= argv[1];
    networkFilename= argv[2];
    outputFilename= argv[3];
    
    if (inputFilename == NULL || networkFilename == NULL || outputFilename == NULL ){
        printf("Cannot read input files\n");
        exit(-1);
    }
    
    network = Create_Neural_Network(networkFilename);
    
    // Reads into a global array, then assigns to Input nodes of network
    Read_Input(inputFilename, network);
    
    // Timing support
    unsigned long long baseStart, baseStop, sseStart,sseStop;
    
    
    //Regular section
    baseStart = rdtsc();
    Run_Network_Forward(network);
    baseStop = rdtsc();
    
    // Print Output
    Print_Layer(network->output, INFO_ACTIVATION);
    
    //SSE section
    sseStart = rdtsc();
    Run_Network_Forward_SSE(network);
    sseStop = rdtsc();
    Print_Layer(network->output, INFO_ACTIVATION);
    
    
    // Open file output
    FILE*  outputFile = fopen(outputFilename, "a");
    fprintf(outputFile,"%d, %d, %d, %d, %llu, %llu\n", 
            network->InputNodes, network->HiddenLayers, network->HiddenNodes, network->OutputNodes,
            baseStop-baseStart, sseStop-sseStart);
    fclose(outputFile);
    
    return 0;

}



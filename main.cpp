#include <iostream>

using namespace std;

#include "neuralNetwork.h"

int main() {
    // hyperparams.
    float learningRate = 0.3;
    int numHiddenNodes = 5; //size of single hidden layer.
    int numEpochs = 500;
    float testRatio = 0.2; //proportion of data to use for testing.
    // fake training data for testing.
    float dataset[10][3] = {{2.7810836,2.550537003,0},
                            {1.465489372,2.362125076,0},
                            {3.396561688,4.400293529,0},
                            {1.38807019,1.850220317,0},
                            {3.06407232,3.005305973,0},
                            {7.627531214,2.759262235,1},
                            {5.332441248,2.088626775,1},
                            {6.922596716,1.77106367,1},
                            {8.675418651,-0.242068655,1},
                            {7.673756466,3.508563011,1}};
    // params that get set automatically from data. TODO
    // int testInd = 7;
    float trainingData[7][3] = {{2.7810836,2.550537003,0},
                                {1.465489372,2.362125076,0},
                                {3.396561688,4.400293529,0},
                                {1.38807019,1.850220317,0},
                                {3.06407232,3.005305973,0},
                                {7.627531214,2.759262235,1},
                                {5.332441248,2.088626775,1}};
    float testingData[3][3] = {{6.922596716,1.77106367,1},
                               {8.675418651,-0.242068655,1},
                               {7.673756466,3.508563011,1}};
    int numInputs = 2;
    int numOutputs = 2;
    // init the NN.
    NeuralNetwork net (learningRate, numInputs, numHiddenNodes, numOutputs);
    net.train(trainingData, numEpochs);
    net.print();
    int *predictions = net.predictList(testingData);
    // TODO check accuracy.
    return 0;
}

// Steps to compile:
// g++ *.cpp -o output
// OR
// g++ -c neuralNetwork.cpp
// g++ -c main.cpp
// g++ neuralNetwork.o main.o
// ./a.out

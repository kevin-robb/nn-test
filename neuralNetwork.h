#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
using namespace std;
#pragma once

class NeuralNetwork {
    public:
    float learningRate;
    int numOutputs;
    Neuron **network;
    int numLayers;

    NeuralNetwork(float learningRate, int numInputs, int numHidden, int numOutputs);
    void print();
    float* forwardPropagate(float *inputs);
    void backpropagateError(float *expected);
    void updateWeights(float *row);
    void train(float **trainingData, int numEpochs);
    int predict(float *row);
    int* predictList(float **rows);
};

class Neuron {
    public:
    float *weights;
    int lenWeights;
    float output;
    float error;

    Neuron();
    Neuron(int numNodesInPrevLayer);
    void print();
    float activate(float *inputs);
    void transfer(float *inputs);
    float transferDerivative();
    void setError(float error);
    void updateWeights(float *inputs, float learningRate);
};
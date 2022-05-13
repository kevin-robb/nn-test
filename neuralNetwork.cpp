#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
// #include <cstring>
#include <chrono>
#include <thread>

#include "neuralNetwork.h"

using namespace std;


// Create the NeuralNetwork object and init structure.
NeuralNetwork::NeuralNetwork(float learningRate, int numInputs, int numHidden, int numOutputs) {
    // assign global vals for the NN.
    this->learningRate = learningRate;
    this->numLayers = 2;
    // this->layerSizes = new int[2] {numHidden, numOutputs};
    this->layerSizes[0] = numHidden; this->layerSizes[1] = numOutputs;;
    // init network.
    this->network = new Neuron *[2];
    // create hidden layer nodes.
    this->network[0] = new Neuron[numHidden];
    for (int i=0; i<numHidden; ++i) {
        // each neuron will store links to all nodes in prev layer.
        this->network[0][i] = Neuron(numInputs);
    }
    // create output layer nodes.
    this->network[1] = new Neuron[numOutputs];
    for (int i=0; i<numOutputs; ++i) {
        // each neuron will store links to all nodes in prev layer.
        this->network[1][i] = Neuron(numHidden);
    }
}

// Print NN to console.
void NeuralNetwork::print() {
    cout << "\n-- Network:";
    // iterate through network.
    for (int i=0; i<this->numLayers; ++i) { // choose a layer.
        for (int j=0; j<this->layerSizes[i]; ++j) { // choose a node.
            this->network[i][j].print();
        }
        cout << "\n--";
    }
}

// Obtain the output of the network for a particular input.
float* NeuralNetwork::forwardPropagate(float *inputs, float *outputs) {
    // cout << inputs[0] << "," << inputs[1] << flush;
    outputs[0] = 0.1;
    outputs[1] = 0.9;
    return outputs;
    // return;
    // DEBUG print out inputs
    // cout << "\nforwardPropagate called with inputs: ";
    // for (int i=0; i<sizeof(inputs)/sizeof(inputs[0]); ++i) {
    //     cout << inputs[i] << ",";
    // }

    // float *outputs;
    float *nextInputs = inputs;
    // memcpy(nextInputs, inputs, *inputs * sizeof(inputs)/sizeof(inputs[0]));
    cout << "????";
    int l = this->numLayers;
    cout << "ugh";
    cout << l;
    // cout << "\n" << l << " ";
    // iterate through network.
    for (int i=0; i<this->numLayers; ++i) { // choose a layer.
        // create necessary number of outputs for this layer.
        cout << this->layerSizes[i] << " ";
        // float outputs[this->layerSizes[i]];

        // // DEBUG print out inputs
        // cout << "\nnextInputs: ";
        // for (int i=0; i<sizeof(nextInputs)/sizeof(nextInputs[0]); ++i) {
        //     cout << nextInputs[i] << ",";
        // }

        // cout << "\noutputs: ";
        // float *nextInputs = inputs;
        cout << "test";
        for (int j=0; j<this->layerSizes[i]; ++j) { // choose a node.
            cout << "iter";
            this->network[i][j].transfer(nextInputs);
            // store each node's output.
            outputs[j] = this->network[i][j].output;
            cout << outputs[j] << ",";
        }
        // this output is the input for the next layer.
        delete[] nextInputs;
        // nextInputs = new float[this->layerSizes[i]];
        nextInputs = outputs;
        // memcpy(nextInputs, outputs, *outputs * this->layerSizes[i]);
    }
    // store the outputs in a global var to access them later.
    // this->outputs = outputs;
    return outputs;
}

// Update weights by propagating error backwards through network.
void NeuralNetwork::backpropagateError(float *expected) {
    float errors[this->layerSizes[this->numLayers-1]];
    // iterate through network in reverse.
    for (int i=this->numLayers-1; i>=0; --i) {
        // handle error differently for output vs hidden layers.
        if (i == this->numLayers-1) { // output layer.
            for (int j=0; j<this->layerSizes[i]; ++j) {
                // get the error for each node.
                errors[j] = expected[j] - this->network[i][j].output;
            }
        } else { // hidden layer.
            for (int j=0; j<this->layerSizes[i]; ++j) {
                float err = 0.0;
                // accumulate error for links to all nodes in layer after this.
                for (int k=0; k<this->layerSizes[i+1]; ++k) {
                    err += this->network[i+1][k].weights[j] * this->network[i+1][k].error;
                }
                // set error for this node.
                errors[j] = err;
            }
        }
        // set the error attribute on all nodes in this layer.
        for (int j=0; j<this->layerSizes[i]; ++j) {
            this->network[i][j].setError(errors[j]);
        }
    }
}

// Given a row of inputs, update weights of all nodes in the NN.
void NeuralNetwork::updateWeights(float *row) {
    // we use online learning, updating after every training example.
    float *inputs = new float[sizeof(row)/sizeof(row[0])-1];
    for (int i=0; i<this->numLayers; ++i) {
        if (i == 0) { // first layer.
            // row contains data + true label, so remove the label.
            for (int r=0; r<sizeof(row)/sizeof(row[0]); ++r) {
                inputs[r] = row[r];
            }
        } else {
            // reinitialize inputs array to potentially different length.
            delete[] inputs;
            inputs = new float[this->layerSizes[i-1]];
            // inputs are the outputs from prev layer.
            for (int j=0; j<this->layerSizes[i-1]; ++j) {
                inputs[j] = this->network[i-1][j].output;
            }
        }
        // have each node update its weights.
        for (int j=0; j<this->layerSizes[i]; ++j) {
            this->network[i][j].updateWeights(inputs, this->learningRate);
        }
    }
}

// Train the network for a given number of epochs.
void NeuralNetwork::train(float trainingData[][3], int numEpochs) {
    // int numTrainingRows = sizeof(*trainingData)/sizeof(*trainingData[0])*3-2;
    int numTrainingRows = 7;
    for (int epoch=0; epoch<numEpochs; ++epoch) {
        float sumError = 0;
        for (int row=0; row<numTrainingRows; ++row) {
            // get our model's predictions.
            // define outputs big enough that there won't be a problem for any layer. fill with zeros.
            float outputs_temp[*max_element(this->layerSizes , this->layerSizes + this->numLayers)] = {};
            float *outputs = this->forwardPropagate(trainingData[row], outputs_temp);
            // get the true labels (using one-hot encoding).
            float expected[this->layerSizes[this->numLayers-1]] = {}; // init all 0.
            expected[(int) trainingData[row][2]] = 1; // correct spot is set to 1.
            // add up errors.
            cout << "\nexpected/output: " << flush;
            for (int j=0; j<this->layerSizes[this->numLayers-1]; ++j) {
                cout << expected[j] << "/" << outputs[j] << ", " << flush;
                sumError += (expected[j] - outputs[j]) * (expected[j] - outputs[j]);
            }
            // update the network based on the error.
            this->backpropagateError(expected);
            this->updateWeights(trainingData[row]);
        }
        // print some details to console.
        cout << fixed << setprecision(4);
        cout << "\n>epoch " << epoch << ": error=" << sumError;
    }
}

// Make a predicton using the trained network.
int NeuralNetwork::predict(float *row) {
    // define outputs big enough that there won't be a problem for any layer.
    float outputs_temp[*max_element(this->layerSizes , this->layerSizes + this->numLayers)];
    float *outputs = this->forwardPropagate(row, outputs_temp);
    // undo the one-hot encoding to get back a categorical prediction.
    int prediction = distance(outputs, max_element(outputs, outputs + this->layerSizes[this->numLayers-1]));
    // print to console.
    cout << "\nPredicting " << prediction << ", Actual is " << (int) row[sizeof(row)/sizeof(row[0])];
    return prediction;
}

// Make several predictions from a list of rows.
int* NeuralNetwork::predictList(float rows[][3]) {
    static int predictions[sizeof(*rows)/sizeof(*rows[0])];
    for (int r=0; r<sizeof(*rows)/sizeof(*rows[0]); ++r) {
        predictions[r] = this->predict(rows[r]);
    }
    cout << endl;
    return predictions;
}

// default constructor.
Neuron::Neuron() {
    return;
}

// Create a node in the neural network with given weights.
Neuron::Neuron(int numNodesInPrevLayer){
    this->output = 0;
    this->error = 0;
    // create list of weights at random in range (0,1).
    this->lenWeights = numNodesInPrevLayer + 1;
    this->weights = new float [this->lenWeights];
    for (int i=0; i<this->lenWeights; ++i) {
        weights[i] = (float) rand() / RAND_MAX;
    }
}

// Print Node details to console.
void Neuron::print() {
    cout << fixed << setprecision(4);
    cout << "\nweights: [";
    // int lenWeights = sizeof(this->weights)/sizeof(this->weights[0]);
    for (int i=0; i<this->lenWeights-1; ++i) {
        cout << this->weights[i] << ",";
    } // don't put comma at end of list.
    cout << this->weights[this->lenWeights-1] << "], output: ";
    cout << this->output << ", error: " << this->error;;
}

float Neuron::activate(float *inputs) {
    // length of weights = length of inputs + 1.
    // int lenWeights = sizeof(this->weights)/sizeof(this->weights[0]);
    float activation = this->weights[this->lenWeights-1]; // the bias.
    for (int i=0; i<this->lenWeights-1; ++i) {
        // compute weighted sum of inputs.
        activation += this->weights[i] * inputs[i];
    }
    return activation;
}

void Neuron::transfer(float *inputs) {
    // sigmoid/logistic activation function used for forward propagation.
    this->output = 1.0 / (1.0 + exp(-1 * this->activate(inputs)));
}

float Neuron::transferDerivative() {
    // derivative of sigmoid used for backpropagation.
    return this->output * (1.0 - this->output);
}

void Neuron::setError(float error) {
    this->error = error * this->transferDerivative();
}

void Neuron::updateWeights(float *inputs, float learningRate) {
    for (int i=0; i<this->lenWeights; ++i) {
        this->weights[i] += learningRate * this->error * inputs[i];
    }
    // account for bias in non-output layers.
    this->weights[this->lenWeights-1] += learningRate * this->error;
}

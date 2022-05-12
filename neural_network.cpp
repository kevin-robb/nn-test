#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>

using namespace std;

class NeuralNetwork {
    // Neural Network with one hidden layer.
    public:
    float learningRate;
    int numOutputs; // number of nodes in the output layer.
    Neuron **network;
    int numLayers; // number of layers, not including input layer.

    // Create the NeuralNetwork object and init structure.
    NeuralNetwork(float learningRate, int numInputs, int numHidden, int numOutputs) {
        // assign global vals for the NN.
        this->learningRate = learningRate;
        this->numOutputs = numOutputs;
        this->numLayers = 2;
        // create hidden layer nodes.
        Neuron hiddenLayer[numHidden];
        for (int i=0; i<numHidden; ++i) {
            // each neuron will store links to all nodes in prev layer.
            hiddenLayer[i] = Neuron(numInputs);
        }
        // create output layer nodes.
        Neuron outputLayer[numOutputs];
        for (int i=0; i<numOutputs; ++i) {
            // each neuron will store links to all nodes in prev layer.
            hiddenLayer[i] = Neuron(numHidden);
        }
        // add the layers to the network.
        this->network[0] = hiddenLayer;
        this->network[1] = outputLayer;
    }

    // Print NN to console.
    void print() {
        cout << "--";
        // iterate through network.
        for (int i=0; i<this->numLayers; ++i) { // choose a layer.
            for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) { // choose a node.
                this->network[i][j].print();
            }
            cout << "\n--";
        }
    }

    // Obtain the output of the network for a particular input.
    float* forwardPropagate(float *inputs) {
        static float outputs[sizeof(inputs)/sizeof(inputs[0])];
        float *nextInputs = inputs;
        // iterate through network.
        for (int i=0; i<this->numLayers; ++i) { // choose a layer.
            for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) { // choose a node.
                this->network[i][j].transfer(nextInputs);
                // store each node's output.
                outputs[j] = this->network[i][j].output;
            }
            // this output is the input for the next layer.
            nextInputs = outputs;
        }
        return outputs;
    }

    // Update weights by propagating error backwards through network.
    void backpropagateError(float *expected) {
        float errors[sizeof(expected)/sizeof(expected[0])];
        // iterate through network in reverse.
        for (int i=this->numLayers-1; i>=0; --i) {
            // handle error differently for output vs hidden layers.
            if (i == this->numLayers-1) { // output layer.
                for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) {
                    // get the error for each node.
                    errors[j] = expected[j] - this->network[i][j].output;
                }
            } else { // hidden layer.
                for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) {
                    float err = 0.0;
                    // accumulate error for links to all nodes in layer after this.
                    for (int k=0; k<sizeof(this->network[i+1])/sizeof(this->network[i+1][0]); ++k) {
                        err += this->network[i+1][k].weights[j] * this->network[i+1][k].error;
                    }
                    // set error for this node.
                    errors[j] = err;
                }
            }
            // set the error attribute on all nodes in this layer.
            for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) {
                this->network[i][j].setError(errors[j]);
            }
        }
    }

    // Given a row of inputs, update weights of all nodes in the NN.
    void updateWeights(float *row) {
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
                inputs = new float[sizeof(this->network[i-1])/sizeof(this->network[i-1][0])];
                // inputs are the outputs from prev layer.
                for (int j=0; j<sizeof(this->network[i-1])/sizeof(this->network[i-1][0]); ++j) {
                    inputs[j] = this->network[i-1][j].output;
                }
            }
            // have each node update its weights.
            for (int j=0; j<sizeof(this->network[i])/sizeof(this->network[i][0]); ++j) {
                this->network[i][j].updateWeights(inputs, this->learningRate);
            }
        }
    }

    // Train the network for a given number of epochs.
    void train(float **trainingData, int numEpochs) {
        for (int epoch=0; epoch<numEpochs; ++epoch) {
            float sumError = 0;
            for (int row=0; row<sizeof(trainingData)/sizeof(trainingData[0]); ++row) {
                // get our model's predictions.
                float *outputs = this->forwardPropagate(trainingData[row]);
                // get the true labels (using one-hot encoding).
                float expected[this->numOutputs] = {0}; // init all 0.
                expected[(int) trainingData[row][sizeof(trainingData[row])/sizeof(trainingData[row][0])-0]] = 1; // correct spot is set to 1.
                // add up errors.
                for (int j=0; j<this->numOutputs; ++j) {
                    sumError += (expected[j] - outputs[j]) * (expected[j] - outputs[j]);
                }
                // update the network based on the error.
                this->backpropagateError(expected);
                this->updateWeights(trainingData[row]);
                // print some details to console.
                cout << fixed << setprecision(4);
                cout << "\n>epoch " << epoch << ": error=" << sumError;
            }
        }
    }

    // Make a predicton using the trained network.
    int predict(float *row) {
        float *outputs = this->forwardPropagate(row);
        // undo the one-hot encoding to get back a categorical prediction.
        int prediction = distance(outputs, max_element(outputs, outputs + this->numOutputs));
        // print to console.
        cout << "\nPredicting " << prediction << ", Actual is " << row[sizeof(row)/sizeof(row[0])];
        return prediction;
    }

    // TODO predict_list function
};

class Neuron {
    // a node in the neural network.
    // each node has nInputs+1 weights that are initially random.
    // -> one for each node in the previous layer +1 for the bias.
    public:
    // TODO array of weights
    float *weights;
    int lenWeights;
    float output;
    float error;

    // default constructor.
    Neuron() {
        return;
    }

    // Create a node in the neural network with given weights.
    Neuron(int numNodesInPrevLayer){
        this->output = 0;
        this->error = 0;
        // create list of weights at random in range (0,1).
        this->lenWeights = numNodesInPrevLayer + 1;
        float weights[this->lenWeights];
        for (int i=0; i<this->lenWeights; ++i) {
            weights[i] = (float) rand() / RAND_MAX;
        }
        this->weights = weights;
    }

    // Print Node details to console.
    void print() {
        cout << fixed << setprecision(4);
        cout << "\nweights: [";
        // int lenWeights = sizeof(this->weights)/sizeof(this->weights[0]);
        for (int i=0; i<this->lenWeights-1; ++i) {
            cout << this->weights[i] << ",";
        } // don't put comma at end of list.
        cout << this->weights[this->lenWeights-1] << "], output: ";
        cout << this->output << ", error: " << this->error;;
    }

    float activate(float *inputs) {
        // length of weights = length of inputs + 1.
        // int lenWeights = sizeof(this->weights)/sizeof(this->weights[0]);
        float activation = this->weights[this->lenWeights-1]; // the bias.
        for (int i=0; i<this->lenWeights-1; ++i) {
            // compute weighted sum of inputs.
            activation += this->weights[i] * inputs[i];
        }
        return activation;
    }

    void transfer(float *inputs) {
        // sigmoid/logistic activation function used for forward propagation.
        this->output = 1.0 / (1.0 + exp(-1 * this->activate(inputs)));
    }

    float transferDerivative() {
        // derivative of sigmoid used for backpropagation.
        return this->output * (1.0 - this->output);
    }

    void setError(float error) {
        this->error = error * this->transferDerivative();
    }

    void updateWeights(float *inputs, float learningRate) {
        for (int i=0; i<this->lenWeights; ++i) {
            this->weights[i] += learningRate * this->error * inputs[i];
        }
        // account for bias in non-output layers.
        this->weights[this->lenWeights-1] += learningRate * this->error;
    }
};
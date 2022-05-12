#include <iostream>

using namespace std;


// Obtain the output of the network for a particular input.
float* forwardPropagate(float *inputs) {
    static float outputs[sizeof(inputs)/sizeof(inputs[0])];
    float *nextInputs = inputs;
    // iterate through network.
    for (int i=0; i<2; ++i) { // choose a layer.
        for (int j=0; j<10; ++j) { // choose a node.
            // store each node's output.
            outputs[j] = nextInputs[j] * 2;
        }
        // this output is the input for the next layer.
        nextInputs = outputs;
    }
    return outputs;
}

int main() {
    // test
    float inputs[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float *outputs = forwardPropagate(inputs);
    cout << "inputs: ";
    for (int i=0; i<10; ++i) {
        cout << inputs[i] << ",";
    }
    cout << "\noutputs: ";
    for (int i=0; i<10; ++i) {
        cout << outputs[i] << ",";
    }



    // hyperparams.
    float learningRate = 0.3;
    int numHiddenNodes = 5; //size of single hidden layer.
    int numEpochs = 500;
    float testRatio = 0.2; //proportion of data to use for testing.
    return 0;
}

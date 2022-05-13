#include <iostream>

using namespace std;

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


float* forwardPropagate(float *inputs, float *outputs) {
    cout << "DEBUG2 ";
    cout << inputs[0] << "," << inputs[1];
    cout << "\nSet outputs.";

    outputs[0] = 0.5;
    outputs[1] = 6.9;
    outputs[3] = 1.4;

    return outputs;

    // return;
    // DEBUG print out inputs
    // cout << "\nforwardPropagate called with inputs: ";
    // for (int i=0; i<sizeof(inputs)/sizeof(inputs[0]); ++i) {
    //     cout << inputs[i] << ",";
    // }

    // // float *outputs;
    // float *nextInputs = inputs;
    // // memcpy(nextInputs, inputs, *inputs * sizeof(inputs)/sizeof(inputs[0]));
    // cout << "????";
    // int l = this->numLayers;
    // cout << "ugh";
    // cout << l;
    // // cout << "\n" << l << " ";
    // // iterate through network.
    // for (int i=0; i<this->numLayers; ++i) { // choose a layer.
    //     // create necessary number of outputs for this layer.
    //     cout << this->layerSizes[i] << " ";
    //     // float outputs[this->layerSizes[i]];

    //     // // DEBUG print out inputs
    //     // cout << "\nnextInputs: ";
    //     // for (int i=0; i<sizeof(nextInputs)/sizeof(nextInputs[0]); ++i) {
    //     //     cout << nextInputs[i] << ",";
    //     // }

    //     // cout << "\noutputs: ";
    //     // float *nextInputs = inputs;
    //     cout << "test";
    //     for (int j=0; j<this->layerSizes[i]; ++j) { // choose a node.
    //         cout << "iter";
    //         this->network[i][j].transfer(nextInputs);
    //         // store each node's output.
    //         outputs[j] = this->network[i][j].output;
    //         cout << outputs[j] << ",";
    //     }
    //     // this output is the input for the next layer.
    //     delete[] nextInputs;
    //     // nextInputs = new float[this->layerSizes[i]];
    //     nextInputs = outputs;
    //     // memcpy(nextInputs, outputs, *outputs * this->layerSizes[i]);
    // }
    // // store the outputs in a global var to access them later.
    // // this->outputs = outputs;
    // return outputs;
}


int main() {
    // define outputs big enough that there won't be a problem for any layer.
    float outputs_temp[5] = {}; // fill with zeros.
    int row = 0;
    float *outputs = forwardPropagate(trainingData[row], outputs_temp);

    cout << "\noutputs: ";

    for (int i=0; i<5; ++i) {
        cout << outputs[i] << ",";
    }

    return 0;
}


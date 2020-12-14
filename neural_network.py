from random import random
from math import exp
from typing import List
class NeuralNetwork:
    def __init__(self,n_in:int,n_hidden:int,n_out:int):
        # network is a list of layers
        self.network = []
        # each layer is a list of nodes
        hidden_layer = [Neuron(weights=[random() for i in range(n_in+1)]) for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [Neuron(weights=[random() for i in range(n_hidden+1)]) for i in range(n_out)]
        self.network.append(output_layer)

    def __str__(self) -> str:
        result = "--"
        for layer in self.network:
            for neuron in layer:
                result += str(neuron)
            result += "\n--"
        return result

    def forward_propagate(self, inputs:List[float]) -> List[float]:
        for layer in self.network:
            outputs = []
            for neuron in layer:
                neuron.transfer(inputs)
                outputs.append(neuron.output)
            inputs = outputs
        return outputs

class Neuron:
    # each node has n_in+1 (initially random) weights,
    # one for each node in the previous layer + 1 for the bias
    def __init__(self, weights:List[float]):
        self.weights = weights
        self.output = 0

    def __str__(self) -> str:
        result = "\nweights: ["
        for w in range(len(self.weights)):
            txt = "{weight:.4f}"
            result += txt.format(weight = self.weights[w])
            if w < len(self.weights)-1: result += ","
        result += "], output: " + "{out:.4f}".format(out = self.output)
        return result
        #return "\nweights: " + str(self.weights) + "\noutput: " + str(self.output)

    def activate(self, inputs:List[float]) -> float:
        activation = self.weights[-1] # the bias
        # weighted sum of inputs
        for i in range(len(inputs)):
            activation += self.weights[i] * inputs[i]
        return activation

    def transfer(self, inputs:List[float]):
        # sigmoid/logistic activation function
        self.output = 1.0 / (1.0 + exp(-self.activate(inputs)))
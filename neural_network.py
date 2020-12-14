from random import random
class NeuralNetwork:
    def __init__(self,n_in:int,n_hidden:int,n_out:int):
        # network is a list of layers
        self.network = []
        # each layer is a list of nodes
        # each node is a dictionary containing n_in+1 (random) weights (the +1 is for the bias)
        hidden_layer = [{"weights":[random() for i in range(n_in+1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{"weights":[random() for i in range(n_in+1)]} for i in range(n_out)]
        self.network.append(output_layer)
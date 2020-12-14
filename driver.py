from neural_network import NeuralNetwork

net = NeuralNetwork(2,1,2)
print(net)

print(net.forward_propagate([1,0]))

print(net)
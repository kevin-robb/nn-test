from neural_network import NeuralNetwork

net = NeuralNetwork(2,1,2)
print(net)

net.forward_propagate([1,0])

net.backpropagate_error(expected=[0,1])

print(net)
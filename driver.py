from neural_network import NeuralNetwork
import data_util

#net = NeuralNetwork(learning_rate, 2,1,2)
#print(net)
#net.forward_propagate([1,0])
#net.backpropagate_error(expected=[0,1])
#print(net)
# fake training data to test
# dataset = [[2.7810836,2.550537003,0],
#     [1.465489372,2.362125076,0],
#     [3.396561688,4.400293529,0],
#     [1.38807019,1.850220317,0],
#     [3.06407232,3.005305973,0],
#     [7.627531214,2.759262235,1],
#     [5.332441248,2.088626775,1],
#     [6.922596716,1.77106367,1],
#     [8.675418651,-0.242068655,1],
#     [7.673756466,3.508563011,1]]
#net.train(dataset, 20)

def make_and_train_nn():
    # params to set manually
    learning_rate = 0.3
    n_hidden = 5
    n_epochs = 500
    test_ratio = 0.2 #proportion of data for testing
    # params that are automatically set
    dataset = data_util.get_data()
    test_ind = int(len(dataset) * (1 - test_ratio))
    training_data = dataset[:test_ind]
    testing_data = dataset[test_ind:]
    n_inputs = len(dataset[0])-1
    n_outputs = len(set([row[-1] for row in dataset]))
    net = NeuralNetwork(learning_rate,n_inputs,n_hidden,n_outputs)
    net.train(training_data, n_epochs)
    print(net)
    predictions = net.predict_list(testing_data)
    print("Accuracy is " + str(data_util.accuracy(testing_data,predictions)))

make_and_train_nn()

from neural_network import NeuralNetwork
import numpy

# Setting up the value for the neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

# Initializing the neural network
neural_network_object = NeuralNetwork(input_nodes, hidden_nodes,
                                       output_nodes, learning_rate)

training_file = open('mnist_train_100.csv', 'r')
training_list = training_file.readlines()
training_file.close()

for record in training_list:
    values = record.split(',')
    input_array = (numpy.asfarray(values[1:])/255.0*0.99)+0.01
    target = numpy.zeros(output_nodes)+0.01
    target[int(values[0])] = 0.99
    neural_network_object.train(input_array, target)

test_file = open('mnist_test_10.csv', 'r')
test_list = test_file.readlines()
test_file.close()

test_value = test_list[1].split(',')
print test_value[0]
print neural_network_object.query(
            (numpy.asfarray(test_value[1:])/255.0*0.99)+0.01)

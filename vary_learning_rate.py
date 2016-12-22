from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import numpy

# Setting up the value for the neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

scorecard = []
def scorecard_generation(record):
    test_value = record.split(',')
    network_lable = neural_network_object.query(
                                (numpy.asfarray(test_value[1:])/255.0*0.99)+0.01)
    if (int(test_value[0]) == numpy.argmax(network_lable)):
        return 1
    else:
        return 0


training_file = open('mnist_train.csv', 'r')
training_list = training_file.readlines()
training_file.close()
score = []
learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i in range(1, 10):

    # Initializing the neural network
    neural_network_object = NeuralNetwork(input_nodes, hidden_nodes,
                                        output_nodes, learning_rate[i-1])

    for record in training_list:
        values = record.split(',')
        input_array = (numpy.asfarray(values[1:])/255.0*0.99)+0.01
        target = numpy.zeros(output_nodes)+0.01
        target[int(values[0])] = 0.99
        neural_network_object.train(input_array, target)

    test_file = open('mnist_test.csv', 'r')
    test_list = test_file.readlines()
    test_file.close()
    scorecard = map(lambda record: scorecard_generation(record), test_list)
    score.append(sum(scorecard)/10000.0)


print score
print learning_rate

# The graph is Accuracy against the learning rate which actually gives total
# justification as the accuracy decreases with learning rate increase
y = score
x = learning_rate
plt.plot(x, y)
plt.yscale('linear')
plt.title('Varying Learning Rate')
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

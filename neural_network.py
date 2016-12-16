import numpy
import scipy.special


class NeuralNetwork:
    """ This is the first implementation of the Neural_Network
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """ Initializing the neural network
        """
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.learning_rate = learning_rate

        # This is the heart of  the neural network where the link-weight are
        # assigned to the nodes for the first time
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))

        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))
        # This is the activation funtion i.e sigmond function for crossing the
        # thereshold value just like in neurons
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        """ To train the neural network
        """
        input_2darray = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.wih, input_2darray)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        # This is the error calculation part from the output calculated by the
        # neural network
        target = numpy.array(target_list, ndmin=2).T
        output_errors = target - final_output

        # backpropogation of error to the hidden layer for error division
        # according to the weight of the link
        hidden_error = numpy.dot(self.who.T, output_errors)

        # This is the heart of training the machine it calculates the error and
        # updates the weight accordingly hence making the machine more accurate

        self.who += self.learning_rate*numpy.dot((output_errors*final_output*(1-final_output)),
                                                numpy.transpose(hidden_output))

        self.wih += self.learning_rate*numpy.dot((hidden_error*hidden_output*(1-hidden_output)),
                                                numpy.transpose(input_2darray))

    def query(self, input_list):
        """ To query the neural network
        """
        input_2darray = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.wih, input_2darray)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        return final_output

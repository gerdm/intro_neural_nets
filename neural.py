"""
eed Forward Neural Network class based on the book by Michael A. Nielsen, 
"Neural Networks and Deep Learning", Determination Press, 2015
"""

import numpy as np
from numpy.random import randn, seed
from numpy import vectorize
from numpy import ones, exp

sigmoid = vectorize(lambda z: 1 / (1 +  exp(-z)))
sigmoid_prime = vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))

class NNet:
    def __init__(self, layers, cost_function="rss", set_seed=None):
        seed(set_seed)
        self.layers_len = len(layers)
        self.layers = layers
        self.biases = self.init_biases()
        self.weights = self.init_weights()
        self.cost_funcion = cost_function

    def init_weights(self):
        # Mapping from layer l to l-1 for all l in 2, ..., L-1
        layers_map = zip(self.layers[1:], self.layers[:-1])
        return [randn(output, input) for output, input in layers_map]

    def init_biases(self):
        return [randn(output, 1) for output in self.layers[1:]]

    def forwardpropagate(self, inputs):
        """
        Forward propagate the neural net where 'inputs'
        represent the values given in the input layer
        """
        for w, b in zip(self.weights, self.biases):
            inputs = sigmoid(w @ inputs + b)
        
        return inputs

    def update_mini_batch(self, mini_batch, alpha):
        """
        Update a mini batch training set applying
        gradient descent:

        Returns
        -------
        A tuple with the changes in weights and biases
        """
        pass

    def cost_deriv(self, output, target):
        #TODO: Add different cost functions
        return (output - target)

    def backpropagate(self, inputs, targets):
        """
        For a given pair of inputs / outputs, backpropagate
        through the net and return the a tuple with a 
        set of changes of the output w.r.t. each weigth
        and a set of changes in in the outputs w.r.t. each
        bias
        """
        # List to hold the gradient in biases and weights
        # for each layer in the net
        gradient_bias = [np.empty(b.shape) for b in self.biases]
        gradient_weight = [np.empty(w.shape) for w in self.weights]

        ## Forward-Propagate ##
        activation = inputs
        # Lists to store the activation function at each layer
        # (the activation at l=0 is simply the inputs)
        activations = [activation]
        # Lists where we will store the inputs for every hidden unit
        # and the output layer
        zs = []
        # Forward-propagate the input units; save the
        # inputs for each activation function for each hidden and
        # out layer; save the activation at each layer
        for bias, weight in zip(self.biases, self.weights):
            z = weight @ activation + bias
            activation = sigmoid(z)

            zs.append(z)
            activations.append(activation) 

        ## Back-Propagate ##
        # We first compute the outermost delta factor which involves computing
        # the change in error w.r.t. the output times the change in the
        # output w.r.t. the input at that layer (z terms)
        change_error_wrt_output = self.cost_deriv(targets, activations[-1])
        change_output_wrt_input = sigmoid_prime(zs[-1])
        delta = change_error_wrt_output * change_output_wrt_input 
        print(delta.shape)
        print(activations[-2].T.shape)
        gradient_bias[-1] = delta
        #gradient_weight[-1] = delta * 

if __name__ == "__main__":
    #train_data = np.array([1, 1], ndmin=2).T
    data = np.load("train.npy")
    input_train = data[:, :-1].T
    output_train = data[:, -1:].T

    net = NNet([2, 3, 1], set_seed=23)
    net.backpropagate(input_train, output_train)

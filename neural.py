"""
Feed Forward Neural Network class based on the book by Michael A. Nielsen, 
"Neural Networks and Deep Learning", Determination Press, 2015
"""

import numpy as np
from numpy.random import randn, seed, shuffle
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

    def make_mini_batches(self,training_data, len_batch):
        n= len(training_data)
        shuffle(training_data)
        mini_batches = [training_data[ix: ix + len_batch]
                        for ix in range(0, n, len_batch)]
        return mini_batches

    def SGD(self, training_data, epochs, mini_batch_size,
            alpha, test_data=None):
        # If there is test data, define the test_data
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for epoch in range(epochs):
            mini_batches = self.make_mini_batches(training_data, mini_batch_size)
            for batch in mini_batches:
                self.update_mini_batch(batch, alpha)
            if test_data:
                correct = self.evaluate(test_data)
                print(f"Epoch {epoch + 1}: {correct} / {n_test}, ({correct / n_test:0.4f})", end="\r")
            else:
                # Print end of training
                print(f"Epoch {epoch + 1} / {epochs}")

    def update_mini_batch(self, mini_batch, alpha):
        """
        Update a mini batch training set applying
        gradient descent:

        Returns
        -------
        A tuple with the changes in weights and biases
        """
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        gradient_weight = [np.zeros(w.shape) for w in self.weights]

        # compounding the changes in weights for every
        # training element
        for x, y in mini_batch:
            change_bias, change_weight = self.backpropagate(x, y)

            gradient_bias = [grad_bias + change_grad_bias for
                             grad_bias, change_grad_bias in
                             zip(gradient_bias, change_bias)]
            
            gradient_weight = [grad_weight + change_grad_weight for
                             grad_weight, change_grad_weight in
                             zip(gradient_weight, change_weight)]

        # Update weights and biases
        nbatch = len(mini_batch)
        self.weights = [w - alpha / nbatch * grad_w for w, grad_w
                        in zip(self.weights, gradient_weight)]
        self.biases = [b - alpha / nbatch * grad_b for b, grad_b
                        in zip(self.biases, gradient_bias)]


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

        for l in range(1,self.layers_len):
            change_output_wrt_input = sigmoid_prime(zs[-l])
            if l == 1:
                # If we are at the ouput layer
                change_error_wrt_output = self.cost_deriv(activations[-l], targets)
                delta = change_error_wrt_output * change_output_wrt_input
            else:
                # If we are in every other hidden layer
                delta = (self.weights[-l + 1].T @ delta) * change_output_wrt_input

            gradient_bias[-l] = delta
            gradient_weight[-l] = delta @ activations[-(l+1)].T

        return gradient_bias, gradient_weight

    def evaluate(self, test_data):
        results = [(np.argmax(self.forwardpropagate(x)), np.argmax(y)) for
                   (x, y) in test_data]

        return sum([int(x == y) for (x, y) in results])


if __name__ == "__main__":
    import pickle
    with open("train.pkl", "rb") as file:
        train_data= pickle.load(file)

    with open("test.pkl", "rb") as file:
        test_data = pickle.load(file)

    net = NNet([2, 3, 2], set_seed=23)
    net.SGD(train_data, 100, 13, 0.03, test_data)

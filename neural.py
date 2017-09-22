import numpy as np
from numpy.random import randn, seed
from numpy import vectorize
from numpy import ones, exp

sigmoid = vectorize(lambda z: 1 / (1 +  exp(-z)))
sigmoid_prime = (lambda z: sigmoid(z) * (1 - sigmoid(z)))

class NNet:
    def __init__(self, layers, cost_function="rss", set_seed=None):
        seed(set_seed)
        self.layers_len = len(layers)
        self.layers = layers
        self.biases = self.init_biases()
        self.weights = self.init_weights()
        self.cost_funcion = cost_function

    def init_weights(self):
        layers_map = zip(self.layers[1:], self.layers[:-1])
        return [randn(t0, t1) for t0, t1 in layers_map]

    def init_biases(self):
        return [randn(b, 1) for b in self.layers[1:]]

    def forwardpropagate(self, inputs):
        """
        Forward propagate the neural net where 'inputs'
        represent the values given in the input layer
        """
        for w, b in zip(self.weights, self.biases):
            inputs = sigmoid(w @ inputs + b)
        
        return inputs

    def train(self, targets):
        pass

    @vectorize
    def cost_deriv(self, target):
        #TODO: Add different cost functions
        return (output - target)

    def backpropagate(self, targets):
        """
        For a given list of outputs, backpropagate
        through the net and return the a tuple with a 
        set of changes of the output w.r.t. each weigth
        and a set of changes in in the outputs w.r.t. each
        bias
        """
        pass

if __name__ == "__main__":
    #train_data = np.load("train.npy")
    train_data = np.array([1, 1], ndmin=2).T
    print(train_data)

    net = NNet([2, 3, 1], set_seed=23)
    output = net.forwardpropagate(train_data)
    print(output)

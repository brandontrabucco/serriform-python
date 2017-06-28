"""

Author: Brandon Trabucco.
Date: 2017.06.27.
Program Name: Serriform Neural Network.
Program Description: This project contains a Python implementation of the serriform neural network algorithm, using NumPy as a linear algebra system.

"""


"""
The NumPy library is used extensively in this python application.
This library provides a comprehensive set of linear algebra utility classes and functions.
"""
import numpy as np


"""
This class contains all the resources and utility functions used by the serriform algorithm
"""
class serriform:

    """
    The activation function transforms an input betwene positive and negatitive infinity between zero and one.
    """
    def activate(x):
        return 1 / (1 + np.exp(-x))

    """
    The slope of the activation function is used for calculating partial derivatives for machine learning.
    """
    def aprime(x):
        return x * (1 - x)

    """
    This class represents a mneural network layer, which accepts an input, produces an output, and can backpropogate error in order to learn.
    The serriform algorithm usese backpropagation through time, but is not a recurrent neural network.
    """
    class layer:

        """
        Reset the algorithm after a full forward and backward pass cycle has been completed, and the unrolled network has been collapsed
        """
        def reset(self):
            self.unrolled = []
            self.time = -1
            self.cell.reset()

        """
        Initialize the hidden layer, and create the serriform intermediate layers
        """
        def __init__(self, params):
            self.alpha = params['alpha']
            self.cell = serriform.cell(params)
            self.reset()

        """
        Compute a single forward pass and obtain an output
        """
        def forward(self, stimulus, delta):
            self.time += 1
            self.output = self.cell.forward(stimulus)
            self.cell.set(delta)
            self.unrolled.append(self.cell)
            return self.output

        """
        Prepare the algorithm to begin backpropogation through time, and update the weight parameters
        """
        def initBackward(self):
            self.cell.inputWeights *= -1 * (self.time - 1)
            self.cell.stateWeights *= -1 * (self.time - 1)

        """
        Compute a single backward pass and obtain a delta to send to earlier hidden layers in the network
        """
        def backward(self):
            self.delta = self.unrolled[self.time].backward(self.alpha)
            self.cell.inputWeights += self.unrolled[self.time].inputWeights
            self.cell.stateWeights += self.unrolled[self.time].stateWeights
            if self.time > 0:
                self.unrolled[self.time - 1].statePartial = self.unrolled[self.time].statePartial
                self.time -= 1
            else:
                self.reset()
            return self.delta

    """
    This class contains the heavy-lifting mathematics of the serriform algorithm.
    The intermediate layers and weights are represented by NumPy matrices.
    """
    class cell:

        """
        This section creates the matrices used by this algorithm, includig the weight matrices, and an adjacency matrix
        """
        def create(self):
            self.reset()
            self.adjacency = np.triu(np.ones((self.depth, self.depth)), 1) * np.tril(np.ones(((self.depth, self.depth))), self.overlap)
            self.adjacency[:(self.depth - self.offset), :] = self.adjacency[self.offset:, :]
            self.adjacency[(self.depth - self.offset):self.depth, :] = np.zeros((self.offset, self.depth))
            self.adjacency = self.adjacency.reshape(1, self.depth, 1, self.depth).repeat(self.outputWidth, axis=0).repeat(self.outputWidth, axis=2)
            self.inputWeights = np.random.normal(0, 1, (self.inputWidth, self.outputWidth, self.depth))
            self.stateWeights = np.random.normal(0, 1, (self.outputWidth, self.depth, self.outputWidth, self.depth))

        """
        This section will reset the serriform cell after a complete backpropapgation cycle has occured
        """
        def reset(self):
            self.stimulus = np.zeros((1, self.inputWidth))
            self.previous = np.zeros((1, self.outputWidth, self.depth))
            self.state = np.zeros((1, self.outputWidth, self.depth))
            self.statePartial = np.zeros((1, self.outputWidth, self.depth))

        """
        This will set the error to be used by backpropagation in the unrolled network.
        """
        def set(self, error):
            self.error = error

        """
        This will create the serriform cell based a set of configurable parameters.
        Note that the parameters such as depth, overlap, and offset will most significantly change the shape and structure of teh adjacency matrix.
        """
        def __init__(self, params):
            self.inputWidth = params['inputWidth']
            self.outputWidth = params['outputWidth']
            self.depth = params['depth']
            self.overlap = params['overlap']
            self.offset = params['offset']
            self.create()

        """
        Compute a single forward pass within this serriform cell, and output a resulting serriform state.
        """
        def forward(self, stimulus):
            self.stimulus = stimulus
            self.previous = self.state
            self.state = serriform.activate(np.tensordot(self.stimulus, self.inputWeights, axes=1) + np.tensordot(self.previous, (self.adjacency * self.stateWeights), axes=2))
            return self.state

        """
        Compute a single backward pass within this serriform cell, and output a weighted error for earlier hidden layers.
        """
        def backward(self, alpha):
            self.delta = (self.error + self.statePartial) * serriform.aprime(self.state)
            self.inputPartial = np.tensordot(self.delta, self.inputWeights.transpose(1, 2, 0), axes=2)
            self.statePartial = np.tensordot(self.delta, (self.adjacency * self.stateWeights).transpose(2, 3, 0, 1), axes=2)
            self.inputWeightPartial = np.tensordot(self.stimulus.transpose(1, 0), self.delta, axes=1)
            self.stateWeightPartial = np.tensordot(np.tensordot(self.previous.transpose(1, 2, 0), self.delta, axes=1), self.adjacency, axes=2)
            self.inputWeights -= alpha * self.inputWeightPartial
            self.stateWeights -= alpha * self.stateWeightPartial
            return self.inputPartial


"""
This section calls the utility functions of the serriform algorithm to ensure that all features work properly.
"""
def main():
    layer = serriform.layer({'inputWidth': 3, 'outputWidth': 2, 'depth': 5, 'overlap': 3, 'offset': 1, 'alpha': 0.01})
    print(layer.forward(np.ones((1, 3)), np.ones((1, 2, 5))))
    layer.initBackward()
    print(layer.backward())

main()
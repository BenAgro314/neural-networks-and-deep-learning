"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

np.random.seed(0)
random.seed(0)

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """
        self.biases = [
            [[a_11], ... [[a_1n]
            [a_21],
            [a_31],
            ...
            [a_n1]]       [a_nn]]
        ]
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # weights is a list of matricies of of size
        # (number of neurons in layer l-1) x (number of neurons in layer l)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        # training data is a list of tuples of the form
        # ([image pixel data (784, 1)], digit guess [10, 1])
        n = len(training_data)
        for j in xrange(epochs): # for each epoch
            start_time = time.time()
            # shuffle the training data 
            random.shuffle(training_data)
            # minibatches is a list of lists of training data
            # this fully covers the image data, but is in 
            # randomized order
            """
            [
                [
                    (image_list, guess_list)
                ]
                ...
            ]
            """
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
                print "Completed in {0} seconds".format(time.time() - start_time)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # copy of self.biases and self.weights with zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x, y = zip(*mini_batch)
        x,y = np.array(x), np.array(y)
        nabla_b, nabla_w = self.backprop(x, y)
        """
        m = len(mini_batch)
        eta = learning rate
        X_j = all training examples in current minibatch
        w_k -> w_k - \frac{\eta}{m}\sum_j \frac{\partial C_{X_j}}{\partial w_k}
        """
        self.weights = [w-(eta/len(mini_batch))*np.sum(nw, axis = 0)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*np.sum(nb, axis = 0)
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #nabla_b = np.array([np.zeros(b.shape) for b in self.biases]*len(x))
        #nabla_w = np.array([np.zeros(w.shape) for w in self.weights]*len(x))
        nabla_b = []
        nabla_w = []
        # feedforward
        activation = x
        # the activation for the first (input) layer is just the input
        activations = [x] # list to store all the activations, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            w = np.repeat(w[np.newaxis, :, :], len(activation), axis = 0)
            b = np.repeat(b[np.newaxis, :, :], len(activation), axis = 0)
            #w = np.array([w]*len(activation))
            #b = np.array([b]*len(activation))
            nabla_b.append(np.zeros(b.shape))
            nabla_w.append(np.zeros(w.shape))
            # iterate through the layers and calculate z and a
            z = np.matmul(w, activation)+b # of same dim as neurons in first hidden layer
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # last layer delta is equal to 
        """
        (\nabla_a C) \odot \sigma'(z^L)
        """
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        """
        \frac{\partial C}{\partial b^l} = \delta^l
        """
        nabla_b[-1] = delta 
        """
        \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
        """
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose(0, 2, 1))
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            # we are at layer 
            # l = L - l'
            z = zs[-l]
            sp = sigmoid_prime(z)
            """
            \delta^l = ((w^{l+1}_{jk})^T \delta^{l+1}) \odot \sigma'(z^l)
            """
            w = np.repeat(self.weights[-l + 1][np.newaxis, :, :], len(activations[-l+1]), axis = 0)
            delta = np.matmul(w.transpose(0, 2, 1), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.matmul(delta, activations[-l-1].transpose(0, 2, 1))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

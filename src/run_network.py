#!/usr/bin/env python2.7

MATRIX = True
import mnist_loader

if MATRIX:
    import matrix_network
    net = matrix_network.Network([784, 30, 10])
else:
    import network
    net = network.Network([784, 30, 10])

training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

net.SGD(training_data, 30, 10, 3.0, test_data = test_data)



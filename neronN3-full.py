import random
import math


class Neuron:
    def __init__(self, activation_function, num_inputs, learning_rate, weights=None):
        self.activation_function = activation_function
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = weights or [random.random() for _ in range(num_inputs + 1)]
        self.inputs = [0] * (num_inputs + 1)
        self.output = 0
        self.partial_derivatives = [0] * (num_inputs + 1)

    def activate(self, value):
        net = sum(x * y for x, y in zip(self.weights, self.inputs))
        self.output = self.activation_function(net)
        return self.output

    def calculate(self, inputs):
        self.inputs = inputs + [1]
        return self.activate(self.inputs)

    def activation_derivative(self):
        return self.activation_function(self.output, derivative=True)

    def calc_partial_derivative(self, delta):
        self.partial_derivatives = [self.activation_derivative() * delta * x for x in self.inputs]
        return [self.weights[i] * delta for i in range(self.num_inputs)]

    def update_weights(self):
        for i in range(self.num_inputs + 1):
            self.weights[i] -= self.learning_rate * self.partial_derivatives[i]

import numpy as np


class ConvolutionalLayer:

    def __init__(self, num_kernels, kernel_size, activation_func, input_dim, learning_rate, weights=None):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.activation_func = activation_func
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Initialize weights randomly if not provided
        if weights is None:
            self.weights = np.random.rand(num_kernels, kernel_size, kernel_size, input_dim[2])
        else:
            self.weights = weights

        # Initialize neurons for each kernel
        self.neurons = []
        for k in range(num_kernels):
            neuron_weights = self.weights[k, :, :, :]
            neuron = Neuron(activation_func, kernel_size ** 2 * input_dim[2], learning_rate, neuron_weights)
            self.neurons.append(neuron)

    def calculate(self, input_data):
        """
        Given input data of shape (batch_size, height, width, num_channels),
        calculate the output of all neurons in the layer.
        """
        self.input_data = input_data
        batch_size, height, width, num_channels = input_data.shape

        # Calculate output for each kernel
        outputs = np.zeros((batch_size, height - self.kernel_size + 1, width - self.kernel_size + 1, self.num_kernels))
        for k in range(self.num_kernels):
            for i in range(height - self.kernel_size + 1):
                for j in range(width - self.kernel_size + 1):
                    receptive_field = input_data[:, i:i + self.kernel_size, j:j + self.kernel_size, :]
                    output = self.neurons[k].calculate(receptive_field.reshape(batch_size, -1))
                    outputs[:, i, j, k] = output

        self.outputs = outputs
        return outputs

    def calculatewdeltas(self, next_layer_wdeltas):
        """
        Given the sum of w*deltas from the next layer, calculate the partial derivative for
        each neuron and update the weights. Return the sum of w*deltas for this layer.
        """
        batch_size, next_height, next_width, next_channels = next_layer_wdeltas.shape
        wdeltas = np.zeros((batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2]))

        for k in range(self.num_kernels):
            kernel_wdeltas = np.zeros((batch_size, self.kernel_size, self.kernel_size, self.input_dim[2]))
            for i in range(next_height):
                for j in range(next_width):
                    kernel_wdeltas += self.weights[k] * next_layer_wdeltas[:, i, j, k].reshape(batch_size, 1, 1, -1)

                    # Calculate partial derivative for each neuron
                    receptive_field = self.input_data[:, i:i + self.kernel_size, j:j + self.kernel_size, :]
                    neuron_pd = self.neurons[k].calcpartialderivative(receptive_field.reshape(batch_size, -1),
                                                                      next_layer_wdeltas[:, i, j, k])

                    # Add neuron partial derivative to kernel_wdeltas
                    kernel_wdeltas += neuron_pd.reshape(batch_size, self.kernel_size, self.kernel_size, -1)

            # Update weights for this kernel
            self.weights[k] -= self.learning_rate * kernel_wdeltas.mean(axis=0)

            # Add kernel_wdeltas to wdeltas
            wdeltas[:, :, :, :] += kernel_wdeltas

        return wdeltas

import numpy as np


class MaxPoolingLayer:
    def __init__(self, kernel_size, input_dim):
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.max_locs = None

    def calculate(self, input_data):
        batch_size, input_h, input_w, input_c = input_data.shape
        output_h = input_h // self.kernel_size
        output_w = input_w // self.kernel_size
        output_c = input_c

        # Reshape input data to allow for easy max pooling
        input_data = np.reshape(input_data, (batch_size, input_h, input_w, self.kernel_size, self.kernel_size, input_c))

        # Calculate max pooling
        output_data = np.max(input_data, axis=(3, 4))

        # Save the locations of the max values for use in backpropagation
        self.max_locs = np.zeros((batch_size, output_h, output_w, input_c, 2), dtype=int)
        for h in range(output_h):
            for w in range(output_w):
                h_start = h * self.kernel_size
                w_start = w * self.kernel_size
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                for c in range(input_c):
                    max_loc = np.unravel_index(np.argmax(input_data[:, h_start:h_end, w_start:w_end, c]),
                                               (self.kernel_size, self.kernel_size))
                    self.max_locs[:, h, w, c] = [h_start + max_loc[0], w_start + max_loc[1]]

        return output_data

    def calculatewdeltas(self, wdeltas):
        batch_size, output_h, output_w, input_c = wdeltas.shape
        input_h = output_h * self.kernel_size
        input_w = output_w * self.kernel_size

        # Reshape wdeltas to allow for easy insertion of max values
        wdeltas = np.reshape(wdeltas, (batch_size, output_h, output_w, 1, 1, input_c))

        # Initialize output wdeltas
        output_wdeltas = np.zeros((batch_size, input_h, input_w, input_c))

        # Insert max values into output wdeltas
        for h in range(output_h):
            for w in range(output_w):
                for c in range(input_c):
                    h_loc, w_loc = self.max_locs[:, h, w, c]
                    output_wdeltas[:, h_loc, w_loc, c] = wdeltas[:, h, w, 0, 0, c]

        return output_wdeltas

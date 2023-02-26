class FlattenLayer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size[0] * input_size[1] * input_size[2]

    def calculate(self, input_data):
        return input_data.reshape(self.output_size)

    def calculatewdeltas(self, next_layer_wdeltas):
        return next_layer_wdeltas.reshape(self.input_size)

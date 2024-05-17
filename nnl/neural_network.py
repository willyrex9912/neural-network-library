import numpy as np
from nnl.activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self, num_inputs, num_outputs, hidden_layers, activation_hidden='sigmoid', activation_output='step'):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.weights = []
        self.biases = []
        self.initialize_weights()

        self.layer_inputs = []
        self.layer_outputs = []
        self.activation_function = ActivationFunction()

    def initialize_weights(self):
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]
        for i in range(len(layers) - 1):
            weight_matrix = np.random.rand(layers[i], layers[i+1]) - 0.5
            bias_vector = np.random.rand(layers[i+1]) - 0.5
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, inputs):
        self.layer_inputs = [inputs]
        self.layer_outputs = []

        for i in range(len(self.weights)):
            net_input = np.dot(self.layer_inputs[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                activation = self.activation_function.activation(net_input, self.activation_output)
            else:
                activation = self.activation_function.activation(net_input, self.activation_hidden)
            self.layer_outputs.append(activation)
            self.layer_inputs.append(activation)

        return self.layer_outputs[-1]

    def backward_propagation(self, expected_output, learning_rate):
        errors = [expected_output - self.layer_outputs[-1]]
        deltas = [errors[-1] * self.activation_function.activation_derivative(self.layer_outputs[-1], self.activation_output)]

        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * self.activation_function.activation_derivative(self.layer_outputs[i], self.activation_hidden)
            errors.append(error)
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            input_layer = np.atleast_2d(self.layer_inputs[i])
            delta_layer = np.atleast_2d(deltas[i])

            self.weights[i] += input_layer.T.dot(delta_layer) * learning_rate
            self.biases[i] += np.sum(delta_layer, axis=0) * learning_rate

    def train(self, inputs, outputs, learning_rate=0.2, times=1000):
        print("##### TRAIN START #####")
        for time in range(times):
            for input_vector, output_vector in zip(inputs, outputs):
                self.forward_propagation(input_vector)
                self.backward_propagation(output_vector, learning_rate)

            if time % 100 == 0:
                mse = np.mean(np.square(outputs - self.forward_propagation(inputs)))
                print(f"Time {time} / {times} - MSE: {mse}")
        print("##### TRAIN END #####")

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            prediction = self.forward_propagation(input_vector)
            predictions.append(prediction)
        return np.array(predictions)

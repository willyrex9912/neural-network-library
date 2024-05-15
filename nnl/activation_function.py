import numpy as np


class ActivationFunction:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def derivative_tanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def derivative_identity(x):
        return np.ones_like(x)

    @staticmethod
    def step(x):
        return np.where(x >= 0.5, 1, 0)

    def activation(self, x, func='sigmoid'):
        if func == 'sigmoid':
            return self.sigmoid(x)
        elif func == 'tanh':
            return self.tanh(x)
        elif func == 'identity':
            return self.identity(x)
        else:
            return self.sigmoid(x)

    def activation_derivative(self, x, func='sigmoid'):
        if func == 'sigmoid':
            return self.derivative_sigmoid(x)
        elif func == 'tanh':
            return self.derivative_tanh(x)
        elif func == 'identity':
            return self.derivative_identity(x)
        else:
            return self.derivative_sigmoid(x)

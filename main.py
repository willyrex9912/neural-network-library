# This is a sample Python script.
from nnl.neural_network import NeuralNetwork
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    nn = NeuralNetwork(2, 1, [2, 2], 'tanh', 'sigmoid')
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    outputs = [[0], [1], [1], [0]]
    nn.train(inputs, outputs, 0.2, 10000)

    while True:
        input_values = []
        for i in range(2):
            value = float(input(f"Type characteristic value {i + 1}: "))
            input_values.append(value)
        predicted_output = nn.predict([input_values])
        print("The output is:", round(predicted_output[0][0]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

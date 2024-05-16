# This is a sample Python script.
from nnl.neural_network import NeuralNetwork


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    num_inputs = int(input("Inputs number: "))
    num_outputs = int(input("Outputs number: "))
    num_training_inputs = int(input("Training inputs/outputs number: "))

    inputs = []
    outputs = []

    for ti in range(num_training_inputs):
        print("Inputs and outputs " + str(ti+1) + ": ")
        inputs_aux = []
        outputs_aux = []
        for i in range(num_inputs):
            inp = int(input("Type input " + str(i+1) + ": "))
            inputs_aux.append(inp)
        for o in range(num_outputs):
            inp = int(input("Type output " + str(o+1) + ": "))
            outputs_aux.append(inp)
        inputs.append(inputs_aux)
        outputs.append(outputs_aux)

    num_hidden_layers = int(input("Number of hidden layers: "))
    hidden_layers = []
    for hl in range(num_hidden_layers):
        neurons_num = int(input("Neurons number on hidden layer " + str(hl+1) + ": "))
        hidden_layers.append(neurons_num)

    print("Available functions for hidden layers: sigmoid, tanh")
    activation_hidden = input("Type function for hidden layers: ")
    if activation_hidden == "":
        activation_hidden = "sigmoid"
    print("Available functions for output layer: identity, step")
    activation_output = input("Type function for output layer: ")
    if activation_output == "":
        activation_output = "identity"

    learning_rate = float(input("Learning rate: "))
    training_times = int(input("Times for training: "))

    nn = NeuralNetwork(num_inputs, num_outputs, hidden_layers, activation_hidden, activation_output)
    nn.train(inputs, outputs, learning_rate, training_times)

    counter = 0
    while counter < 100:
        input_values = []
        for i in range(num_inputs):
            value = float(input(f"Type characteristic value {i + 1}: "))
            input_values.append(value)
        predicted_output = nn.predict([input_values])
        print("The output is:", predicted_output)
        counter += 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

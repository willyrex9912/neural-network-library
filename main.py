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

    nn = NeuralNetwork(num_inputs, num_outputs, hidden_layers, 'tanh', 'sigmoid')
    nn.train(inputs, outputs, 0.1, 10000)

    counter = 0
    while counter < 100:
        input_values = []
        for i in range(2):
            value = float(input(f"Type characteristic value {i + 1}: "))
            input_values.append(value)
        predicted_output = nn.predict([input_values])
        print("The output is:", round(predicted_output[0][0]))
        counter += 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

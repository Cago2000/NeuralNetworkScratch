import math
import random
from enum import Enum
import functions

import numpy
from matplotlib import pyplot as plt


class Model(Enum):
    SIN = 0
    XOR = 1


class Act_Func(Enum):
    SIGMOID = 1
    TANH = 2
    IDENTITY = 3


def return_consistent_weights(input_size: int, hidden_size: int, value: float) -> tuple:
    weights1 = [[value+((i+j)/10) for i in range(input_size)] for j in range(hidden_size)]
    weights2 = [value+(i/10) for i in range(hidden_size)]
    return weights1, weights2


def return_random_weights(input_size: int, hidden_size: int) -> tuple:
    weights1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
    weights2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
    return weights1, weights2


def tensor_product(A: list, B: list) -> list:
    product = []
    for a in A:
        for b in B:
            product.append(a * b)
    return product


def matrix_multiplication(X: list, Y: list) -> list:
    result = [[0] * len(Y)] * len(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Y[0])):
                result[i][j] += X[i][k] * Y[j][k]
    return result


def activation_function(act_func: Act_Func, data: list):
    match act_func:
        case Act_Func.SIGMOID:
            return functions.sigmoid_list(data)
        case Act_Func.TANH:
            return functions.tanh_list(data)
        case Act_Func.IDENTITY:
            return data

def activation_function_derivative(act_func: Act_Func, data: list):
    match act_func:
        case Act_Func.SIGMOID:
            return functions.sigmoid_derivative_list(data)
        case Act_Func.TANH:
            return functions.tanh_derivative_list(data)
        case Act_Func.IDENTITY:
            return [1] * len(data)

def forward_pass_hidden_layer(h: list, weights: list, act_func: Act_Func) -> tuple:
    z = matrix_multiplication(h, weights)
    new_h = activation_function(act_func, z[0])
    return z, new_h


def forward_pass_output_layer(h: list, weights: list, act_func: Act_Func) -> tuple:
    z = matrix_multiplication(h, weights)
    return z, z


def calculate_sin_output_error(x: float, h: list) -> list:
    error = []
    for y_pred in h[0]:
        y_true = math.sin(x)
        e = y_true - y_pred
        error.append(e)
    return error


def calculate_xor_output_error(x1: int, x2: int, h: list) -> list:
    error = []
    for y_pred in h[0]:
        y_true = functions.xor(x1, x2)
        e = y_true - y_pred
        error.append(e)
    return error


def output_delta(error: list, z: list, act_func: Act_Func) -> list:
    delta = []
    derivative = activation_function_derivative(act_func, z)
    for derivative_val, error_val in zip(derivative, error):
        delta.append(derivative_val * error_val)
    return delta


def backpropagation(z: list, weights: list, delta: list, act_func: Act_Func) -> list:
    error = matrix_multiplication([delta], weights)
    new_delta = []
    derivative = activation_function_derivative(act_func, z[0])
    for derivative_val, error_val in zip(derivative, error[0]):
        new_delta.append(derivative_val * error_val)
    return new_delta


def adjust_weights(weights: list, delta: list, h: list, alpha: float) -> list:
    weight_delta = tensor_product(delta, h)
    adjusted_weights = weights.copy()
    index = 0
    for i, row in enumerate(weights):
        for j, weight in enumerate(row):
            adjusted_weights[i][j] += (alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def transpose_matrix(X: list):
    result = [[0] * len(X)] * len(X[0])
    for i, row in enumerate(X):
        for j, val in enumerate(row):
            result[j][i] = val
    return result


def fit(iterations: int, data: list, input_size: int, hidden_size: int, alpha: float, error_threshold: float, model: Model,
        hidden_act_func: Act_Func, output_act_func: Act_Func) -> tuple:
    weights1, weights2 = return_random_weights(input_size, hidden_size)
    errors_all_iterations = []
    print(f'Fitting {model.name} model, max iterations: {iterations}, learning rate: {alpha}')
    print(f'Hidden activation function: {hidden_act_func.name}, output activation function: {output_act_func.name}')
    print(f'Neurons in input layer: {input_size}, neurons in hidden layer: {hidden_size}, neurons in output layer: {1}')
    for i in range(0, iterations):
        errors = []
        for row in data:
            h1 = row
            #print("h1:", h1)
            z1, h2 = forward_pass_hidden_layer([h1], weights1, hidden_act_func)
            #print("z1:", z1)
            #print("h2:", h2)
            z2, y = forward_pass_output_layer([h2], [weights2], output_act_func)
            #print("z2:", z2)
            #print("y:", y)
            error = []
            match model:
                case model.SIN:
                    error = calculate_sin_output_error(h1[0], y)
                case model.XOR:
                    error = calculate_xor_output_error(h1[0], h1[1], y)
            errors.append(math.sqrt(error[0]**2))
            #print("error:", error)
            delta3 = output_delta(error, z2[0], output_act_func)
            #print("delta3:", delta3)
            t_weights2 = transpose_matrix([weights2])
            delta2 = backpropagation(z1, t_weights2, delta3, hidden_act_func)
            #print("delta2:", delta2)
            weights2 = adjust_weights([weights2], delta3, h2, alpha)[0]
            weights1 = adjust_weights(weights1, delta2, h1, alpha)
            #print("weights1:", weights1)
            #print("weights2:", weights2)
        if i % 1000 == 0:
            print(f'Iteration {i}')
        met_threshold = True
        err_temp = 0.0
        for error in errors:
            err_temp += error
            if error > error_threshold:
                met_threshold = False
        errors_all_iterations.append(err_temp/len(errors))
        if met_threshold:
            print(f'Threshold met after {i} iterations.')
            return weights1, weights2, errors_all_iterations
    return weights1, weights2, errors_all_iterations


def predict(h1: list, weights1: list, weights2: list, model: Model, print_output: bool,
            hidden_act_func: Act_Func, output_act_func: Act_Func):
    _, h2 = forward_pass_hidden_layer(h1, weights1, hidden_act_func)
    z2, y = forward_pass_output_layer([h2], [weights2], output_act_func)
    if print_output:
        match model:
            case Model.SIN:
                print(f'pred: {y[0]}, x: {h1[0][0]}, y: {math.sin(h1[0][0])}')
            case Model.XOR:
                print(f'pred: {y[0]}, x1: {h1[0][0]}, x2: {h1[0][1]}, y: {functions.xor(h1[0][0], h1[0][1])}')
    return y[0]


def predict_all(samples: list, weights1: list, weights2: list, model: Model, print_output: bool,
                hidden_act_func: Act_Func, output_act_func: Act_Func):
    predictions = []
    if print_output:
        print(f'--------------------Predict {model.name}--------------------')
    for sample in samples:
        predictions.append(predict([sample], weights1, weights2, model, print_output,
                                   hidden_act_func, output_act_func))
    if print_output:
        print("---------------------------------------------------\n")
    return predictions


def plot_result(model: Model, errors: list):
    plt.plot(errors)
    plt.ylabel(f'Model: {model.name}')
    plt.show()


def main() -> None:

    xor_sample = [[1, -1, 1],
                 [-1, 1, 1],
                  [1, 1, 1],
                 [-1, -1, 1]]

    sin_sample = []
    x_vals_sin = numpy.linspace(-math.pi*1, math.pi*1, 100)
    for x_val in x_vals_sin:
        sin_sample.append([x_val, 1.0])

    weights1_xor, weights2_xor, errors_xor = (
        fit(iterations=10000,
            data=xor_sample,
            input_size=3,
            hidden_size=3,
            alpha=0.05,
            error_threshold=1e-6,
            model=Model.XOR,
            hidden_act_func=Act_Func.TANH,
            output_act_func=Act_Func.TANH))
    predict_all(xor_sample, weights1_xor, weights2_xor, Model.XOR, True,
                Act_Func.TANH, Act_Func.TANH)
    plot_result(Model.XOR, errors_xor)

    weights1_sin, weights2_sin, errors_sin = (
        fit(iterations=10000,
            data=sin_sample,
            input_size=2,
            hidden_size=7,
            alpha=0.05,
            error_threshold=1e-3,
            model=Model.SIN,
            hidden_act_func=Act_Func.TANH,
            output_act_func=Act_Func.TANH))
    y_predictions_sin = predict_all(sin_sample, weights1_sin, weights2_sin, Model.SIN, False,
                                    Act_Func.TANH, Act_Func.IDENTITY)
    plot_result(Model.SIN, errors_sin)
    plt.plot(x_vals_sin, y_predictions_sin)
    plt.show()


if __name__ == "__main__":
    main()

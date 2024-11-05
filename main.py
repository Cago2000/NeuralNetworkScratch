import math
import random

import numpy
from keras.src.datasets import mnist

import enums
from enums import Model, Act_Func
import functions
from matplotlib import pyplot as plt


def forward_pass(h: list, weights: list, act_func: Act_Func) -> tuple:
    z = functions.matrix_multiplication(h, weights)
    new_h = act_func.get_function()(z[0])
    return z, new_h


def calculate_output_error(x: list, h: list, model: Model) -> list:
    x = x[0][:-1] if len(x[0]) > 1 else x[0]
    error = []
    for y_pred in h[0]:
        y_true = model.get_function()(*x)
        e = y_true - y_pred
        error.append(e)
    return [error]


def backpropagation(z: list, weights: list, delta: list, act_func: Act_Func) -> list:
    new_delta = []
    derivative = act_func.get_derivative_function()(z[0])
    error = functions.matrix_multiplication(delta, weights)
    for derivative_val, error_val in zip(derivative[0], error[0]):
        new_delta.append(derivative_val * error_val)
    return [new_delta]


def adjust_weights(weights: list, delta: list, h: list, alpha: float) -> list:
    weight_delta = functions.tensor_product(delta[0], h)
    adjusted_weights = weights.copy()
    index = 0
    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            adjusted_weights[i][j] += (alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def fit(iterations: int, iteration_update: int, data: list, layer_sizes: list,
        alpha: float, error_threshold: float, model: Model, act_functions: list, y_train: list) -> tuple:
    w_list = functions.return_random_weights(layer_sizes)
    w_list.append([[1 for _ in range(layer_sizes[-1])]])
    all_errors = []
    for i in range(0, iterations):
        if i % iteration_update == 0:
            print(f'Iteration {i + 1}')
        errors = []
        for j, row in enumerate(data):
            h_list = [[row]]
            z_list = []
            d_list = []
            for (k, h), w, act_func in zip(enumerate(h_list), w_list, act_functions):
                if k < len(layer_sizes) - 1:
                    z, h = forward_pass(h, w, act_func)
                    h_list.append(h)
                    z_list.append(z)
            y = functions.argmax(h_list[-1])
            # print(f'h_list: {h_list}')
            # print(f'w_list: {w_list}')
            # print(f'z_list: {z_list}')
            match model:
                case Model.DIGIT:
                    error = functions.get_digit_error(h_list[-1], y_train[j])
                case _:
                    error = calculate_output_error(h_list[0], h_list[-1], model)
            errors.append(abs(error[0][0]))
            d_list.append(error)
            for z, (k, w), d, act_func in zip(reversed(z_list), enumerate(reversed(w_list)), d_list,
                                              reversed(act_functions)):
                new_d = backpropagation(z, functions.transpose_matrix(w), d, act_func)
                d_list.append(new_d)
            # print("d_list:", d_list)
            for (k, w), d, h in zip(enumerate(reversed(w_list)), d_list, reversed(h_list)):
                if k > 0:
                    w_list[len(w_list) - 1 - k] = adjust_weights(w, d, h, alpha)
            # print(f'w_list after: {w_list}')
        met_threshold = True
        err_temp = 0.0
        for error in errors:
            err_temp += error
            if error > error_threshold:
                met_threshold = False
        all_errors.append(err_temp / len(errors))
        if met_threshold:
            print(f'Threshold met after {i} iterations.')
            return w_list, all_errors
    return w_list, all_errors


def predict(h1: list, weights: list, model: Model, print_output: bool,
            act_functions: list):
    h_list = [h1]
    z_list = []
    for h, w, act_func in zip(h_list, weights, act_functions):
        z, h = forward_pass(h, w, act_func)
        h_list.append(h)
        z_list.append(z)
    y = h_list[-1]
    if print_output:
        match model:
            case Model.XOR:
                print(f'pred: {y[0]}, x1: {h1[0][0]}, x2: {h1[0][1]}, y: {enums.xor(h1[0][0], h1[0][1])}')
            case Model.SIN:
                print(f'pred: {y[0]}, x: {h1[0][0]}, y: {math.sin(h1[0][0])}')
            case Model.COS:
                print(f'pred: {y[0]}, x: {h1[0][0]}, y: {math.cos(h1[0][0])}')
            case Model.DIGIT:
                print(f'pred: {functions.argmax(y)}')
    return y[0]


def predict_all(samples: list, weights: list, model: Model, print_output: bool,
                act_functions: list):
    predictions = []
    if print_output:
        print(f'--------------------Predict {model.name}--------------------')
    for sample in samples:
        predictions.append(predict([sample], weights, model, print_output,
                                   act_functions))
    if print_output:
        print("---------------------------------------------------\n")
    return predictions


def main() -> None:
    '''x_train, y_train, x_test, y_test = functions.transform_digit_data(1.0)
    x_train = functions.rescale_data(x_train)
    x_test = functions.rescale_data(x_test)

    matrix = [[random.uniform(-0.5, 0.5) for _ in range(9)] for _ in range(9)]
    kernel = [[0.0, 0.4, 0.0],
              [0.4, 1.0, 0.4],
              [0.0, 0.4, 0.0]]
    functions.print_matrix(matrix)
    conv_matrix = functions.conv(matrix, kernel)
    functions.print_matrix(conv_matrix)


    #prints true value distribution
    digit_counts_train = [0] * 10
    for true_val in y_train:
        digit_counts_train[true_val] += 1
    print(digit_counts_train)
    digit_counts_test = [0] * 10
    for true_val in y_test:
        digit_counts_test[true_val] += 1
    print(digit_counts_test)

    for i, digit in enumerate(x_train):
        x_train[i] = functions.flatten_matrix(digit)

    for i, digit in enumerate(x_test):
        x_test[i] = functions.flatten_matrix(digit)

    digit_act_functions = [Act_Func.SIGMOID, Act_Func.SIGMOID, Act_Func.SIGMOID]
    digit_layer_sizes = [len(x_train[0]), 56, 28, 10]
    weights_digit, errors_digit = (
        fit(iterations=10,
            data=x_train,
            layer_sizes=digit_layer_sizes,
            alpha=0.001,
            error_threshold=1e-3,
            model=Model.DIGIT,
            act_functions=digit_act_functions,
            y_train=y_train,
            iteration_update=1))
    digit_predictions = predict_all(x_test, weights_digit,
                                    Model.DIGIT, False, digit_act_functions)

    hits = 0
    for pred, y_true in zip(digit_predictions, y_test):
        if functions.argmax([pred]) == y_true:
            hits+=1
        print(f'pred_arr: {pred}, pred_digit: {functions.argmax([pred])},y_true: {y_true}')
    print(f'{hits/len(digit_predictions)}%')
    plt.plot(errors_digit)
    plt.ylabel(f'Model: {Model.DIGIT.name}')
    plt.show()'''

    xor_bias = 1.0
    xor_sample = [[1, -1, xor_bias],
                 [-1, 1, xor_bias],
                  [1, 1, xor_bias],
                 [-1, -1, xor_bias]]
    xor_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    xor_layer_sizes = [3, 3, 1]
    weights_xor, errors_xor = (
        fit(iterations=10000,
            iteration_update=10000,
            data=xor_sample,
            layer_sizes=xor_layer_sizes,
            alpha=0.05,
            error_threshold=1e-15,
            model=Model.XOR,
            act_functions=xor_act_functions,
            y_train=[]))
    predict_all(xor_sample, weights_xor, Model.XOR, True,
                xor_act_functions)
    plt.plot(errors_xor)
    plt.title(f'Model: {Model.XOR.name}')
    plt.show()

    x_vals = numpy.linspace(-math.pi*1, math.pi*1, 400)

    sin_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    sin_layer_sizes = [2, 7, 1]
    sin_bias = 1.0
    sin_sample = []
    for x_val in x_vals:
        sin_sample.append([x_val, sin_bias])

    weights_sin, errors_sin = (

        fit(iterations=5000,
            iteration_update=1000,
            data=sin_sample,
            layer_sizes=sin_layer_sizes,
            alpha=0.05,
            error_threshold=1e-3,
            model=Model.SIN,
            act_functions=sin_act_functions,
            y_train=[]))
    y_predictions_sin = predict_all(sin_sample, weights_sin, Model.SIN, True,
                                    sin_act_functions)
    plt.plot(errors_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.show()
    plt.plot(x_vals, y_predictions_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.show()

    cos_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    cos_layer_sizes = [2, 7, 1]
    cos_bias = 1
    cos_sample = []
    for x_val in x_vals:
        cos_sample.append([x_val, cos_bias])
    weights_cos, errors_cos = (

        fit(iterations=5000,
            iteration_update=1000,
            data=cos_sample,
            layer_sizes=cos_layer_sizes,
            alpha=0.05,
            error_threshold=1e-3,
            model=Model.COS,
            act_functions=cos_act_functions,
            y_train=[]))
    y_predictions_cos = predict_all(cos_sample, weights_cos, Model.COS, True,
                                    cos_act_functions)

    plt.plot(errors_cos)
    plt.ylabel(f'Model: {Model.COS.name}')
    plt.show()
    plt.plot(x_vals, y_predictions_cos)
    plt.title(f'Model: {Model.COS.name}')
    plt.show()


if __name__ == "__main__":
    main()

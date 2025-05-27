import math
from copy import deepcopy
import numpy
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
    adjusted_weights = deepcopy(weights)
    index = 0
    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            adjusted_weights[i][j] += (alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def fit(iterations: int, iteration_update: int, data: list, layer_sizes: list,
        alpha: float, error_threshold: float, model: Model, act_functions: list, y_train: list) -> tuple:
    w_list = functions.return_random_weights(layer_sizes)
    w_list.append([[1.0 for _ in range(layer_sizes[-1])]])
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
            # print(f'h_list: {h_list}')
            # print(f'w_list: {w_list}')
            # print(f'z_list: {z_list}')
            match model:
                case Model.DIGIT:
                    error = functions.get_digit_error(h_list[-1], y_train[j])
                case _:
                    error = calculate_output_error(h_list[0], h_list[-1], model)
            # print(error)
            for err in error[0]:
                errors.append(err ** 2)
            d_list.append(error)
            for z, w, d, act_func in zip(reversed(z_list), reversed(w_list), d_list,
                                              reversed(act_functions)):
                new_d = backpropagation(z, functions.transpose_matrix(w), d, act_func)
                d_list.append(new_d)
            # print("d_list:", d_list)
            for (k, w), d, h in zip(enumerate(reversed(w_list)), d_list, reversed(h_list)):
                if k > 0:
                    w_list[len(w_list) - 1 - k] = adjust_weights(w, d, h, alpha)
            # print(f'w_list after: {w_list}')
            # print(f'error: {error[0]}')
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


def predict(h: list, w_list: list, model: Model, print_output: bool,
            act_functions: list, layer_sizes: list, y_true: int):
    h_list = [h]
    for (k, h), w, act_func in zip(enumerate(h_list), w_list, act_functions):
        if k < len(layer_sizes) - 1:
            _, h = forward_pass(h, w, act_func)
            h_list.append(h)
    y = h_list[-1]
    if print_output:
        match model:
            case Model.XOR:
                print(f'pred: {y[0][0]}, x1: {h_list[0][0][0]}, x2: {h_list[0][0][1]}, y: {enums.xor(h_list[0][0][0], h_list[0][0][1])}')
            case Model.SIN:
                print(f'pred: {y[0][0]}, x: {h_list[0][0][0]}, y: {math.sin(h_list[0][0][0])}')
            case Model.COS:
                print(f'pred: {y[0][0]}, x: {h_list[0][0][0]}, y: {math.cos(h_list[0][0][0])}')
            case Model.DIGIT:
                print(f'pred: {y}, predicted_digit: {functions.argmax(y)}, true: {y_true}')
                y[0] = functions.argmax(y)
    return y[0]


def predict_all(samples: list, weights: list, model: Model, print_output: bool,
                act_functions: list, layer_sizes: list, y_true: list):
    predictions = []
    if print_output:
        print(f'--------------------Predict {model.name}--------------------')
    for i, sample in enumerate(samples):
        match model:
            case Model.DIGIT:
                predictions.append(predict([sample], weights, model, print_output,
                                   act_functions, layer_sizes, y_true[i]))
            case _:
                predictions.append(predict([sample], weights, model, print_output,
                                           act_functions, layer_sizes, 0))
    if print_output:
        print("---------------------------------------------------\n")
    return predictions


def main() -> None:
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
                xor_act_functions, xor_layer_sizes, [])
    plt.plot(errors_xor)
    plt.title(f'Model: {Model.XOR.name}')
    plt.show()

    sin_x_vals = numpy.linspace(0, 7, 200)
    sin_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    sin_layer_sizes = [2, 13, 1]
    sin_bias = 1.0
    sin_sample = []

    for x_val in sin_x_vals:
        sin_sample.append([x_val, sin_bias])

    weights_sin, errors_sin = (
        fit(iterations=5000,
            iteration_update=100,
            data=sin_sample,
            layer_sizes=sin_layer_sizes,
            alpha=0.001,
            error_threshold=1e-5,
            model=Model.SIN,
            act_functions=sin_act_functions,
            y_train=[]))

    y_predictions_sin = predict_all(sin_sample, weights_sin, Model.SIN, True,
                                    sin_act_functions, sin_layer_sizes, [])
    plt.plot(errors_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.show()
    plt.plot(sin_x_vals, y_predictions_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.show()

    cos_x_vals = numpy.linspace(0, 7, 200)
    cos_act_functions = [Act_Func.SIN, Act_Func.IDENTITY]
    cos_layer_sizes = [2, 10, 1]
    cos_bias = 1.0
    cos_sample = []

    for x_val in cos_x_vals:
        cos_sample.append([x_val, cos_bias])

    weights_cos, errors_cos = (

        fit(iterations=5000,
            iteration_update=100,
            data=cos_sample,
            layer_sizes=cos_layer_sizes,
            alpha=0.001,
            error_threshold=1e-5,
            model=Model.COS,
            act_functions=cos_act_functions,
            y_train=[]))

    y_predictions_cos = predict_all(cos_sample, weights_cos, Model.COS, True,

                                    cos_act_functions, cos_layer_sizes, [])
    plt.plot(errors_cos)
    plt.ylabel(f'Model: {Model.COS.name}')
    plt.show()
    plt.plot(cos_x_vals, y_predictions_cos)
    plt.title(f'Model: {Model.COS.name}')
    plt.show()


if __name__ == "__main__":
    main()

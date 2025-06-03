import math
from models import Model
import functions
import math_functions
from copy import deepcopy


def forward_pass(h_list: list, weights: list, act_functions: list) -> tuple:
    z_list = []
    for w, act_func in zip(weights, act_functions):
        z = functions.matrix_multiplication(h_list[-1], w)
        h = act_func.get_function()(z[0])
        h_list.append(h)
        z_list.append(z)
    return z_list, h_list


def backpropagation(z_list: list, w_list: list, d_list: list, act_functions: list) -> list:
    for z, w, d, act_func in zip(reversed(z_list), reversed(w_list), d_list, reversed(act_functions)):
        derivative = act_func.get_derivative_function()(z[0])
        error = functions.matrix_multiplication(d, functions.transpose_matrix(w))
        new_delta = [d_val * e_val for d_val, e_val in zip(derivative[0], error[0])]
        d_list.append([new_delta])
    return d_list


def adjust_weights(w_list: list, d_list: list, h_list: list, alpha: float) -> list:
    adjusted_weights = deepcopy(w_list)
    for k, (w, d, h) in enumerate(zip(reversed(w_list), d_list, reversed(h_list))):
        if k == 0:
            continue
        idx = len(w_list) - 1 - k
        weight_delta = functions.tensor_product(d[0], h)
        index = 0
        for i, row in enumerate(w):
            for j, _ in enumerate(row):
                adjusted_weights[idx][i][j] += alpha * weight_delta[index]
                index += 1
    return adjusted_weights


def fit(iterations: int, iteration_update: int, data: list, layer_sizes: list,
        alpha: float, error_threshold: float, model: Model, act_functions: list, y_train: list, seed: int) -> tuple:
    w_list = functions.return_random_weights(layer_sizes, seed)
    w_list.append([[1.0 for _ in range(layer_sizes[-1])]])
    all_errors = []
    error_threshold **= 2

    print(f'--------------------Fitting {model.name}--------------------')
    for i in range(0, iterations):
        if i % iteration_update == 0:
            print(f'    Iteration {i + 1}')
        errors = []

        for j, row in enumerate(data):
            h_list = [[row]]
            d_list = []
            z_list, h_list = forward_pass(h_list, w_list, act_functions)
            error = model.get_error_function(h_list, y_train[j])
            errors.extend(err ** 2 for err in error[0])
            d_list.append(error)
            d_list = backpropagation(z_list, w_list, d_list, act_functions)
            w_list = adjust_weights(w_list, d_list, h_list, alpha)

        met_threshold = True
        err_temp = 0.0
        for error in errors:
            err_temp += error
            if error > error_threshold:
                met_threshold = False
        all_errors.append(err_temp / len(errors))

        if met_threshold:
            print(f'Threshold met after {i} iterations.')
            print("---------------------------------------------------\n")
            return w_list, all_errors
    print("---------------------------------------------------\n")
    return w_list, all_errors


def predict(h: list, w_list: list, model: Model, print_output: bool,
            act_functions: list, y_true: float):
    _, h_list = forward_pass([h], w_list, act_functions)
    y = h_list[-1]
    if print_output:
        match model:
            case Model.XOR:
                print(f'x: {h_list[0][0][:-1]}, y: {y_true}, pred: {y[0][0]}')
            case Model.SIN:
                print(f'x: {h_list[0][0][0]}, y: {y_true}, pred: {y[0][0]}')
            case Model.COS:
                print(f'x: {h_list[0][0][0]}, y: {y_true}, pred: {y[0][0]}')
            case Model.DIGIT:
                print(f'predicted_digit: {functions.argmax(y)}, true: {y_true}, raw_pred: {y}')
                y[0] = functions.argmax(y)
    return y[0]


def predict_all(samples: list, weights: list, model: Model, print_output: bool,
                act_functions: list, y_true: list):
    predictions = []
    if print_output:
        print(f'--------------------Predict {model.name}--------------------')
    for i, sample in enumerate(samples):
        match model:
            case Model.DIGIT:
                predictions.append(predict([sample], weights, model, print_output,
                                           act_functions, y_true[i]))
            case _:
                predictions.append(predict([sample], weights, model, print_output,
                                           act_functions, y_true[i]))
    if print_output:
        print("---------------------------------------------------\n")
    return predictions

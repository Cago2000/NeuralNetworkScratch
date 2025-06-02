import math
from enums import Model, Act_Func
import enums
import functions
from copy import deepcopy


def forward_pass(h: list, weights: list, act_func: Act_Func) -> tuple:
    z = functions.matrix_multiplication(h, weights)
    new_h = act_func.get_function()(z[0])
    return z, new_h

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
            z_list = []
            d_list = []
            for (k, h), w, act_func in zip(enumerate(h_list), w_list, act_functions):
                if k < len(layer_sizes) - 1:
                    z, h = forward_pass(h, w, act_func)
                    h_list.append(h)
                    z_list.append(z)
            error = model.get_error_function(h_list, y_train[j])
            for err in error[0]:
                errors.append(err ** 2)
            d_list.append(error)
            for z, w, d, act_func in zip(reversed(z_list), reversed(w_list), d_list, reversed(act_functions)):
                new_d = backpropagation(z, functions.transpose_matrix(w), d, act_func)
                d_list.append(new_d)
            for (k, w), d, h in zip(enumerate(reversed(w_list)), d_list, reversed(h_list)):
                if k > 0:
                    w_list[len(w_list) - 1 - k] = adjust_weights(w, d, h, alpha)
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
                print(f'x: {h_list[0][0][:-1]}, y: {enums.xor(h_list[0][0][0], h_list[0][0][1])}, pred: {y[0][0]}')
            case Model.SIN:
                print(f'x: {h_list[0][0][0]}, y: {math.sin(h_list[0][0][0])}, pred: {y[0][0]}')
            case Model.COS:
                print(f'x: {h_list[0][0][0]}, y: {math.cos(h_list[0][0][0])}, pred: {y[0][0]}')
            case Model.DIGIT:
                print(f'predicted_digit: {functions.argmax(y)}, true: {y_true}, raw_pred: {y}')
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
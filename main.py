import math
import random
import numpy
from enums import Model, Act_Func
import functions

from matplotlib import pyplot as plt
from keras.datasets import mnist


def return_consistent_weights(input_size: int, hidden_size: int, output_size: int, value: float) -> tuple:
    weights1 = [[value + ((i + j) / 10) for i in range(input_size)] for j in range(hidden_size)]
    weights2 = [[value + (i / 10) for i in range(hidden_size)] for _ in range(output_size)]
    return weights1, weights2


def return_random_weights(input_size: int, hidden_size: int, output_size: int) -> tuple:
    weights1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
    weights2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
    return weights1, weights2


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


def output_delta(error: list, z: list, act_func: Act_Func) -> list:
    delta = []
    derivative = act_func.get_derivative_function()(z[0])
    for derivative_val, error_val in zip(derivative[0], error):
        delta.append(derivative_val * error_val)
    return [delta]


def backpropagation(z: list, weights: list, delta: list, act_func: Act_Func) -> list:
    error = functions.matrix_multiplication(delta, weights)
    new_delta = []
    derivative = act_func.get_derivative_function()(z[0])
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


def fit(iterations: int, iteration_update: int, data: list, input_size: int, hidden_size: int, output_size: int,
        alpha: float, error_threshold: float, model: Model, act_functions: list, y_train: list) -> tuple:
    weights1, weights2 = return_consistent_weights(input_size, hidden_size, output_size, 0.5)
    all_errors = []
    #print(f'Weights1: {weights1}')
    #print(f'Weights2: {weights2}')
    for i in range(0, iterations):
        if i % iteration_update == 0:
            print(f'Iteration {i+1}')
        errors = []
        for j, row in enumerate(data):
            h_list = [[row]]
            w_list = [weights1, weights2]
            z_list = []
            d_list = []
            for h, w, act_func in zip(h_list, w_list, act_functions):
                z, h = forward_pass(h, w, act_func)
                h_list.append(h)
                z_list.append(z)
            print(f'h_list:{h_list}')
            print(f'z_list: {z_list}')
            match model:
                case Model.DIGIT:
                    error = functions.get_digit_error(h_list[-1], y_train[j])
                case _:
                    error = calculate_output_error(h_list[0], h_list[-1], model)
            errors.append(abs(error[0][0]))
            d_list.append(error)
            w_list.append(functions.transpose_matrix([[1] * output_size]))
            for z, w, d, act_func in zip(reversed(z_list), reversed(w_list), d_list, reversed(act_functions)):
                print(f'z:{z}, w: {w}, t_w: {functions.transpose_matrix(w)}, d:{d}, act_func:{act_func}')
                d_list.append(backpropagation(z, w, d, act_func))
            print("delta_list:", d_list)
            '''
            t_weights2 = functions.transpose_matrix(weights2)
            delta2 = backpropagation(z1, t_weights2, delta3, hidden_act_func)
            #print("delta2:", delta2)
            weights2 = adjust_weights(weights2, delta3, h2, alpha)
            weights1 = adjust_weights(weights1, delta2, h1, alpha)
            #print("weights1:", weights1)
            #print("weights2:", weights2)'''
        met_threshold = True
        err_temp = 0.0
        for error in errors:
            err_temp += error
            if error > error_threshold:
                met_threshold = False
        all_errors.append(err_temp / len(errors))
        if met_threshold:
            print(f'Threshold met after {i} iterations.')
            return weights1, weights2, all_errors
    return weights1, weights2, all_errors


def predict(h1: list, weights1: list, weights2: list, model: Model, print_output: bool,
            hidden_act_func: Act_Func, output_act_func: Act_Func):
    _, h2 = forward_pass(h1, weights1, hidden_act_func)
    z2, y = forward_pass(h2, weights2, output_act_func)
    if print_output:
        match model:
            case Model.XOR:
                print(f'pred: {y[0]}, x1: {h1[0][0]}, x2: {h1[0][1]}, y: {functions.xor(h1[0][0], h1[0][1])}')
            case Model.SIN:
                print(f'pred: {y[0]}, x: {h1[0][0]}, y: {math.sin(h1[0][0])}')
            case Model.COS:
                print(f'pred: {y[0]}, x: {h1[0][0]}, y: {math.cos(h1[0][0])}')
            case Model.DIGIT:
                print(f'pred: {functions.argmax(y)}')
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


def transform_digit_data(percentage: float):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:int(len(x_train) * percentage)]
    y_train = y_train[:int(len(y_train) * percentage)]
    x_test = x_test[:int(len(x_test) * percentage)]
    y_test = y_test[:int(len(y_test) * percentage)]
    x_train_flattened = []
    for digit in x_train:
        x_train_flattened.append(functions.flatten_matrix(digit))
    x_test_flattened = []
    for digit in x_test:
        x_test_flattened.append(functions.flatten_matrix(digit))
    return x_train_flattened, y_train, x_test_flattened, y_test


def main() -> None:
    xor_bias = 1.0
    xor_sample = [[1, -1, xor_bias],
                  [-1, 1, xor_bias],
                  [1, 1, xor_bias],
                  [-1, -1, xor_bias]]

    weights1_xor, weights2_xor, errors_xor = (
        fit(iterations=1,
            iteration_update=10000,
            data=xor_sample,
            input_size=3,
            hidden_size=3,
            output_size=1,
            alpha=0.05,
            error_threshold=1e-9,
            model=Model.XOR,
            act_functions=[Act_Func.SIGMOID, Act_Func.IDENTITY],
            y_train=[]))
    predict_all(xor_sample, weights1_xor, weights2_xor, Model.XOR, True,
                Act_Func.TANH, Act_Func.IDENTITY)
    plot_result(Model.XOR, errors_xor)

    '''
    sin_bias = 1.0
    sin_sample = []
    x_vals = numpy.linspace(-math.pi*1, math.pi*1, 1000)
    for x_val in x_vals:
        sin_sample.append([x_val, sin_bias])

    weights1_sin, weights2_sin, errors_sin = (
        fit(iterations=10000,
            iteration_update=1000,
            data=sin_sample,
            input_size=2,
            hidden_size=7,
            output_size=1,
            alpha=0.05,
            error_threshold=1e-5,
            model=Model.SIN,
            hidden_act_func=Act_Func.TANH,
            output_act_func=Act_Func.IDENTITY,
            y_train=[]))
    y_predictions_sin = predict_all(sin_sample, weights1_sin, weights2_sin, Model.SIN, False,
                                    Act_Func.TANH, Act_Func.IDENTITY)
    plot_result(Model.SIN, errors_sin)
    plt.plot(x_vals, y_predictions_sin)
    plt.show()

    cos_bias = 1
    cos_sample = []
    for x_val in x_vals:
        cos_sample.append([x_val, cos_bias])
    weights1_cos, weights2_cos, errors_cos = (
        fit(iterations=10000,
            iteration_update=1000,
            data=cos_sample,
            input_size=2,
            hidden_size=7,
            output_size=1,
            alpha=0.05,
            error_threshold=1e-5,
            model=Model.COS,
            hidden_act_func=Act_Func.TANH,
            output_act_func=Act_Func.IDENTITY,
            y_train=[]))
    y_predictions_cos = predict_all(cos_sample, weights1_cos, weights2_cos, Model.COS, False,
                                    Act_Func.TANH, Act_Func.IDENTITY)

    plot_result(Model.COS, errors_cos)
    plt.plot(x_vals, y_predictions_cos)
    plt.show()
    
    # displaying true value counts
    digit_train_sample, y_train, digit_test_sample, y_test = transform_digit_data(0.1)
    print(len(digit_train_sample))
    print(len(y_train))
    digit_counts_train = [0] * 10
    for true_val in y_train:
        digit_counts_train[true_val] += 1
    print(digit_counts_train)
    digit_counts_test = [0] * 10
    for true_val in y_test:
        digit_counts_test[true_val] += 1
    print(digit_counts_test)

    # rescaling 0-255 int to 0-1 float
    for i, digit in enumerate(digit_train_sample):
        for j, _ in enumerate(digit):
            digit_train_sample[i][j] /= 255

    weights1_digit, weights2_digit, errors_digit = (
        fit(iterations=10,
            data=digit_train_sample,
            input_size=len(digit_train_sample[0]),
            hidden_size=28,
            output_size=10,
            alpha=0.05,
            error_threshold=1e-3,
            model=Model.DIGIT,
            hidden_act_func=Act_Func.SIGMOID,
            output_act_func=Act_Func.SIGMOID,
            y_train=y_train,
            iteration_update=1))
    digit_predictions = predict_all(digit_test_sample, weights1_digit, weights2_digit,
                                    Model.DIGIT, False, Act_Func.SIGMOID, Act_Func.SIGMOID)

    hits = 0
    for pred, true in zip(digit_predictions, y_test):
        if functions.argmax([pred]) == true:
            hits+=1
        print(f'pred_arr: {pred}, pred_digit: {functions.argmax([pred])},true: {true}')
    print(f'{hits/len(digit_predictions)}%')
    plot_result(Model.DIGIT, errors_digit)'''


if __name__ == "__main__":
    main()

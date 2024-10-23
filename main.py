import random

import functions

alpha = 0.005
input_size = 3
hidden_size = 3


def return_consistent_weights():
    w1_11, w1_12, w1_13, w1_21, w1_22, w1_23, w1_31, w1_32, w1_33 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    weights1 = [[w1_11, w1_12],
                [w1_21, w1_22],
                [w1_31, w1_32]]
    w2_21, w2_22, w2_23 = 0.5, 0.5, 0.5
    weights2 = [[w2_21],
                [w2_22],
                [w2_23]]
    return weights1, weights2


def return_new_weights():
    w1_11, w1_12, w1_13, w1_21, w1_22, w1_23, w1_31, w1_32, w1_33 = tuple(random.random() for _ in range(input_size * hidden_size))
    weights1 = [[w1_11, w1_12, w1_13],
                [w1_21, w1_22, w1_23],
                [w1_31, w1_32, w1_33]]
    w2_21, w2_22, w2_23 = tuple(random.random() for _ in range(hidden_size))
    weights2 = [[w2_21],
                [w2_22],
                [w2_23]]
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


def forward_pass(h: list, weights: list) -> tuple:
    z = matrix_multiplication(h, weights)
    new_h = list(map(lambda x: functions.sigmoid(x, 0, 5), z[0]))
    return z, new_h


def calculate_output_error(x1: int, x2: int, h: list) -> list:
    error = []
    for y_pred in h:
        y_true = functions.sigmoid(functions.xor(x1, x2), 0, 5)
        e = y_true - y_pred
        error.append(e)
    return error


def output_delta(x1: int, x2: int, z: list, h: list) -> list:
    delta = []
    error = calculate_output_error(x1, x2, [h])
    derivative = list(map(lambda x: functions.sigmoid_derivative(x, 0, 5), z))
    for derivative_val, error_val in zip(derivative, error):
        delta.append(derivative_val * error_val)
    return delta


def backpropagation(z: list, weights: list, delta: list) -> list:
    error = matrix_multiplication([delta], weights)
    new_delta = []
    derivative = list(map(lambda x: functions.sigmoid_derivative(x, 0, 5), z[0]))
    for derivative_val, error_val in zip(derivative, error[0]):
        new_delta.append(derivative_val * error_val)
    return new_delta


def adjust_weights(weights: list, delta: list, layer: list) -> list:
    weight_delta = tensor_product(delta, layer)
    adjusted_weights = weights.copy()
    index = 0
    for i, row in enumerate(weights):
        for j, val in enumerate(row):
            adjusted_weights[i][j] = weights[i][j] + (alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def predict(input: list, weights1: list, weights2: list):
    _, h2 = forward_pass(input, weights1)
    z2, y = forward_pass([h2], weights2)
    return y[0]


def fit(data: list, weights1: list, weights2: list):
    weights1, weights2 = return_consistent_weights()  # return_new_weights()
    for i in range(0, 10000):
        for row in data:
            h1 = row
            z1, h2 = forward_pass([h1], weights1)
            z2, y = forward_pass([h2], weights2)
            delta3 = output_delta(h1[0], h1[1], z2[0], y[0])
            delta2 = backpropagation(z1, weights2, delta3)
            delta1 = backpropagation([h1], weights1, delta2)
            weights2 = adjust_weights(weights2, delta3, h2)
            weights1 = adjust_weights(weights1, delta2, h1)
    return weights1, weights2


print_stuff = True

def main() -> None:
    samples = [[-1, -1],
               [ 1,  1],
               [-1,  1],
               [ 1, -1]]

    weights1, weights2 = return_consistent_weights()
    if print_stuff:
        print(predict([[-1, -1]], weights1, weights2), functions.xor(samples[0][0], samples[0][1]))
        print(predict([[1, 1]], weights1, weights2), functions.xor(samples[1][0], samples[1][1]))
        print(predict([[-1, 1]], weights1, weights2), functions.xor(samples[2][0], samples[2][1]))
        print(predict([[1, -1]], weights1, weights2), functions.xor(samples[3][0], samples[3][1]))
    weights1, weights2 = fit(samples, weights1, weights2)
    if print_stuff:
        print(predict([[-1, -1]], weights1, weights2), functions.xor(samples[0][0], samples[0][1]))
        print(predict([[1, 1]], weights1, weights2), functions.xor(samples[1][0], samples[1][1]))
        print(predict([[-1, 1]], weights1, weights2), functions.xor(samples[2][0], samples[2][1]))
        print(predict([[1, -1]], weights1, weights2), functions.xor(samples[3][0], samples[3][1]))



if __name__ == "__main__":
    main()

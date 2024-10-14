import random

import functions

alpha = 0.25
input_size = 3
hidden_size = 3

samples = [[-1, -1, -1],
           [-1, 1, -1],
           [1, -1, -1],
           [1, 1, -1]]

x1, x2, bias = tuple(random.random() for _ in range(input_size))
h1 = [[x1, x2, bias]]

w11, w12, w13, w21, w22, w23, w31, w32, w33 = tuple(random.random() for _ in range(input_size * hidden_size))
weights1 = [[w11, w12, w13],
            [w21, w22, w23],
            [w31, w32, w33]]
z1 = []

h2 = []

w21, w22, w23 = tuple(random.random() for _ in range(hidden_size))
weights2 = [[w21],
            [w22],
            [w23]]
z2 = []

y = []


def tensor_product(A: list, B: list) -> list:
    product = []
    for a in A:
        for b in B:
            product.append(a * b)
    return product


def matrix_multiplication(A: list, B: list) -> list:
    output_size = len(B[0])
    result = [0] * output_size
    for n, row in zip(A, B):
        for i in range(output_size):
            result[i] += n * row[i]
    return result


def forward_pass(h: list, weights: list) -> tuple:
    z = matrix_multiplication(h, weights)
    h = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    return z, h


def calculate_output_error(h: list) -> list:
    error = []
    for layer_value in h:
        e = -(functions.sigmoid(layer_value, 0))
        error.append(e)
    return error


def output_delta(z: list, h: list) -> list:
    delta = []
    error = calculate_output_error(h)
    derivative = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    for derivative_val, error_val in zip(derivative, error):
        delta.append(derivative_val * error_val)
    return delta


def backpropagation(z: list, weights: list, delta: list) -> list:
    error = matrix_multiplication(delta, weights)
    new_delta = []
    derivative = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    for derivative_val, error_val in zip(derivative, error):
        new_delta.append(derivative_val * error_val)
    return new_delta


def adjust_weights(weights: list, delta: list, layer: list) -> list:
    weight_delta = tensor_product(delta, layer)
    adjusted_weights = weights.copy()
    index = 0
    for i, row in enumerate(weights):
        for j, val in enumerate(row):
            adjusted_weights[i][j] = weights[i][j] + (-alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def predict(input: list):
    _, h2 = forward_pass(input, weights1)
    z2, y = forward_pass(h2, weights2)
    print(y)


def fit(data: list, weights1, weights2):
    for row in data:
        h1 = row
        z1, h2 = forward_pass(h1, weights1)
        z2, y = forward_pass(h2, weights2)
        delta3 = output_delta(z2, y)
        delta2 = backpropagation(z1, weights2, delta3)
        delta1 = backpropagation([1, 1, 1], weights1, delta2)
        weights2 = adjust_weights(weights2, delta2, h2)
        weights1  = adjust_weights(weights1, delta1, h1)


def main() -> None:
    for i in range(0, 100000):
        fit(h1, weights1, weights2)
    predict([-1, -1, -1])
    predict([1, 1, -1])
    predict([-1, 1, 1])


if __name__ == "__main__":
    main()

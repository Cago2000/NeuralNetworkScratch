import functions
import random

alpha = 0.05
input_size = 3
hidden_size = 3

x1, x2, bias = tuple(random.random() for _ in range(input_size))
h1 = [x1, x2, bias]

w11, w12, w13, w21, w22, w23, w31, w32, w33 = tuple(random.random() for _ in range(input_size*hidden_size))
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
        e = -(functions.sigmoid(layer_value, bias))
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


def adjust_weights(weights: list, deltas: list, layers: list) -> None:
    pass


def main() -> None:
    z1, h2 = forward_pass(h1, weights1)
    z2, y = forward_pass(h2, weights2)
    delta3 = output_delta(z2, y)
    delta2 = backpropagation(z1, weights2, delta3)
    delta1 = backpropagation([1, 1, 1], weights1, delta2)
    adjust_weights([[], weights1, weights2], [[], delta1, delta2, delta3], [h1, h2, y])


if __name__ == "__main__":
    main()

import random
from keras.src.datasets import mnist

import functions


def print_matrix(matrix):
    for row in matrix:
        print(row, sep=', ', end=',\n')
    print('\n')


def transpose_matrix(X: list) -> list:
    result = [[1.0 for _ in range(len(X))] for _ in range(len(X[0]))]
    for i, row in enumerate(X):
        for j, val in enumerate(row):
            result[j][i] = val
    return result


def tensor_product(A: list, B: list) -> list:
    product = []
    for a in A:
        for b in B[0]:
            product.append(a * b)
    return product


def matrix_multiplication(X: list, Y: list) -> list:
    result = [[0.0 for _ in range(len(Y))] for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Y[0])):
                result[i][j] += X[i][k] * Y[j][k]
    return result


def conv(matrix, kernel) -> list:
    conv_matrix = [[0.0 for _ in range(len(matrix) - 2)] for _ in range(len(matrix[0]) - 2)]
    for i, matrix_row in enumerate(matrix):
        for j, _ in enumerate(matrix_row):
            conv_val = 0.0
            for n, kernel_row in enumerate(kernel):
                for m, _ in enumerate(kernel_row):
                    if i + n + 2 < len(matrix) and j + m + 2 < len(matrix[0]):
                        conv_val += matrix[i + n][j + m] * kernel[n][m]
            if i < len(conv_matrix) and j < len(conv_matrix[0]):
                conv_matrix[i][j] = conv_val / (len(kernel) * len(kernel[0]))
    return conv_matrix


def argmax(data: list) -> int:
    index = 0
    for i, val in enumerate(data[0]):
        if val > data[0][index]:
            index = i
    return index


def flatten_matrix(matrix: list) -> list:
    flattened_array = []
    for row in matrix:
        for val in row:
            flattened_array.append(val)
    return flattened_array


def get_digit_error(y: list, y_true: int) -> list:
    error = []
    digit = [0.0]*10
    digit[y_true] = 1
    for i, y_pred in enumerate(y[0]):
        err = digit[i] - y_pred
        error.append(err)
    return [error]


def get_digit_data(percentage: float) -> tuple:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:int(len(x_train) * percentage)]
    y_train = y_train[:int(len(y_train) * percentage)]
    x_test = x_test[:int(len(x_test) * percentage)]
    y_test = y_test[:int(len(y_test) * percentage)]
    return x_train, y_train, x_test, y_test


def rescale_data(data: list) -> list:
    digits = []
    for data_row in data:
        digit = [[0.0 for _ in range(len(data[0][0]))] for _ in range(len(data[0]))]
        for j, row in enumerate(data_row):
            for k, val in enumerate(row):
                digit[j][k] = float(val)/255.0
        digits.append(digit)
    return digits


def return_consistent_weights(layer_sizes: list, value: float) -> list:
    weights = []
    for i, _ in enumerate(layer_sizes):
        if i >= len(layer_sizes)-1:
            return weights
        weights.append([[value + ((i + j) / 10) for i in range(layer_sizes[i])] for j in range(layer_sizes[i+1])])
    return weights


def return_random_weights(layer_sizes: list) -> list:
    weights = []
    for i, _ in enumerate(layer_sizes):
        if i >= len(layer_sizes)-1:
            return weights
        weights.append([[random.uniform(-0.5, 0.5) for _ in range(layer_sizes[i])] for _ in range(layer_sizes[i+1])])
    return weights

import random
from keras.src.datasets import mnist
from matplotlib import pyplot as plt


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1


def transpose_matrix(X: list):
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
    result = [[0] * len(Y)] * len(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Y[0])):
                result[i][j] += X[i][k] * Y[j][k]
    return result


def argmax(data: list):
    index = 0
    for i, val in enumerate(data[0]):
        if val > data[0][index]:
            index = i
    return index


def flatten_matrix(matrix: list):
    flattened_array = []
    for row in matrix:
        for val in row:
            flattened_array.append(val)
    return flattened_array


def get_digit_error(y: list, y_true: int):
    error = []
    digit = [0]*10
    digit[y_true] = 1
    for index, y_pred in enumerate(y[0]):
        err = digit[index] - y_pred
        error.append(err)
    return [error]


def transform_digit_data(percentage: float):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:int(len(x_train) * percentage)]
    y_train = y_train[:int(len(y_train) * percentage)]
    x_test = x_test[:int(len(x_test) * percentage)]
    y_test = y_test[:int(len(y_test) * percentage)]
    x_train_flattened = []
    for digit in x_train:
        x_train_flattened.append(flatten_matrix(digit))
    x_test_flattened = []
    for digit in x_test:
        x_test_flattened.append(flatten_matrix(digit))
    return x_train_flattened, y_train, x_test_flattened, y_test


def return_consistent_weights(input_size: int, hidden_size: int, output_size: int, value: float) -> tuple:
    weights1 = [[value + ((i + j) / 10) for i in range(input_size)] for j in range(hidden_size)]
    weights2 = [[value + (i / 10) for i in range(hidden_size)] for _ in range(output_size)]
    return weights1, weights2


def return_random_weights(input_size: int, hidden_size: int, output_size: int) -> tuple:
    weights1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
    weights2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
    return weights1, weights2

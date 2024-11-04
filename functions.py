import math
from math import e


def sigmoid_list(values: list) -> list:
    return [list(map(lambda x: sigmoid(x), values))]


def sigmoid_derivative_list(values: list) -> list:
    return [list(map(lambda x: sigmoid_derivative(x), values))]


def sigmoid(x: float) -> float:
    return 1 / (1 + (e ** -x))


def sigmoid_derivative(x: float) -> float:
    try:
        return sigmoid(x) * (1 - sigmoid(x))
    except:
        print(f'input: {x} killed it')
        return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    return 1-(tanh(x)**2)


def tanh_list(values: list) -> list:
    return [list(map(lambda x: tanh(x), values))]


def tanh_derivative_list(values: list) -> list:
    return [list(map(lambda x: tanh_derivative(x), values))]


def identity(x: float) -> float:
    return x


def identity_derivative(x: float) -> float:
    return 1.0


def identity_list(values: list) -> list:
    return [list(map(lambda x: identity(x), values))]


def identity_derivative_list(values: list) -> list:
    return [list(map(lambda x: identity_derivative(x), values))]


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1


def transpose_matrix(X: list):
    result = [[2] * len(X)] * len(X[0])
    print(f'result: {result}')
    print(f'len(result): {len(result)}')
    print(f'len(result[0]): {len(result[0])}')
    print(f'X:{X}')
    print(f'len(X): {len(X)}')
    print(f'len(row): {len(X[0])}')
    for i, row in enumerate(X):
        print(f'row: {row}')
        for j, val in enumerate(row):
            result[j][i] = val
            print("val assigned!")
            print(f'result: {result}')
    print(f'result: {result}')
    return result


def tensor_product(A: list, B: list) -> list:
    product = []
    for a in A:
        for b in B[0]:
            product.append(a * b)
    return product


def matrix_multiplication(X: list, Y: list) -> list:
    print(f'X:{X}')
    print(f'Y: {Y}')
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

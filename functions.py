from math import e


def sigmoid_list(values, bias):
    return list(map(lambda x: sigmoid(x, bias), values))


def sigmoid_derivative_list(values, bias):
    return list(map(lambda x: sigmoid_derivative(x, bias), values))


def sigmoid(x, bias):
    return 1 / (1 + (e ** -(x - bias)))


def sigmoid_derivative(x, bias):
    return 1 - (1 / (1 + (e ** -(x - bias))))


def relu():
    return 1


def tanh():
    return 1


def xor(a, b):
    if a != b:
        return 1
    else:
        return 0

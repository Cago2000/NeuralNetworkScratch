import math
from math import e


def sigmoid_list(values: list) -> list:
    return list(map(lambda x: sigmoid(x), values))


def sigmoid_derivative_list(values: list) -> list:
    return list(map(lambda x: sigmoid_derivative(x), values))


def sigmoid(x: float) -> float:
    return 1 / (1 + (e ** -x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: float) -> float:
    return max(0, int(x))


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    return 1-(tanh(x)**2)


def tanh_list(values: list) -> list:
    return list(map(lambda x: tanh(x), values))


def tanh_derivative_list(values: list) -> list:
    return list(map(lambda x: tanh_derivative(x), values))


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1

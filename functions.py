from math import e


def sigmoid_list(values: list, bias: float) -> list:
    return list(map(lambda x: sigmoid(x, bias), values))


def sigmoid_derivative_list(values: list, bias: float) -> list:
    return list(map(lambda x: sigmoid_derivative(x, bias), values))


def sigmoid(x: float, bias: float) -> float:
    return 1 / (1 + (e ** -(x - bias)))


def sigmoid_derivative(x: float, bias: float) -> float:
    return 1 - (1 / (1 + (e ** -(x - bias))))


def relu(x: float) -> float:
    return max(0, x)


def tanh(x: float) -> float:
    return (e ** x - e ** -x) / (e ** x + e ** -x)


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1

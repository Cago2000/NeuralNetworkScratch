from math import e


def sigmoid_list(values: list, bias: float) -> list:
    return list(map(lambda x: sigmoid(x, bias, 5), values))


def sigmoid_derivative_list(values: list, bias: float) -> list:
    return list(map(lambda x: sigmoid_derivative(x, bias, 5), values))


def sigmoid(x: float, bias: float, stretch: float) -> float:
    return 1 / (1 + (e ** -(stretch*(x - bias))))


def sigmoid_derivative(x: float, bias: float, stretch: float) -> float:
    return 1 - (1 / (1 + (e ** -(stretch*(x - bias)))))


def relu(x: float) -> float:
    return max(0, int(x))


def tanh(x: float) -> float:
    return (e ** x - e ** -x) / (e ** x + e ** -x)


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1

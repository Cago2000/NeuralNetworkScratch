import math


def sigmoid_list(values: list) -> list:
    return [list(map(lambda x: sigmoid(x), values))]


def sigmoid_derivative_list(values: list) -> list:
    return [list(map(lambda x: sigmoid_derivative(x), values))]


def sigmoid(x: float) -> float:
    return 1 / (1 + (math.e ** -x))


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    return 1 - (tanh(x) ** 2)


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


def relu(x: float) -> float:
    return max(0, int(x))


def relu_derivative(x: float) -> float:
    return 1 if x > 0 else 0


def relu_list(values: list) -> list:
    return [list(map(lambda x: relu(x), values))]


def relu_derivative_list(values: list) -> list:
    return [list(map(lambda x: relu_derivative(x), values))]


def sin(x: float) -> float:
    return math.sin(x)


def sin_derivative(x: float) -> float:
    return math.cos(x)


def sin_list(values: list) -> list:
    return [list(map(lambda x: sin(x), values))]


def sin_derivative_list(values: list) -> list:
    return [list(map(lambda x: sin_derivative(x), values))]


def cos(x: float) -> float:
    return math.cos(x)


def cos_derivative(x: float) -> float:
    return -math.sin(x)


def cos_list(values: list) -> list:
    return [list(map(lambda x: cos(x), values))]


def cos_derivative_list(values: list) -> list:
    return [list(map(lambda x: cos_derivative(x), values))]
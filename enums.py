from enum import Enum
import functions
import math

class Model(Enum):
    SIN = 0
    COS = 1
    XOR = 2
    DIGIT = 3

    def get_function(self):
        match self:
            case self.SIN:
                return math.sin
            case self.COS:
                return math.cos
            case self.XOR:
                return xor
            case self.DIGIT:
                return get_digit_error
            case _:
                return None

    def get_error_function(self, h_list, y_labels):
        match self:
            case Model.DIGIT:
                return get_digit_error(h_list[-1], y_labels)
            case _:
                return calculate_output_error(h_list[0], h_list[-1], self)


class Act_Func(Enum):
    SIGMOID = 0
    TANH = 1
    IDENTITY = 2
    RELU = 3
    SIN = 4

    def get_function(self):
        match self:
            case self.SIGMOID:
                return sigmoid_list
            case self.TANH:
                return tanh_list
            case self.IDENTITY:
                return identity_list
            case self.RELU:
                return relu_list
            case self.SIN:
                return sin_list
            case _:
                return None

    def get_derivative_function(self):
        match self:
            case self.SIGMOID:
                return sigmoid_derivative_list
            case self.TANH:
                return tanh_derivative_list
            case self.IDENTITY:
                return identity_derivative_list
            case self.RELU:
                return relu_derivative_list
            case self.SIN:
                return sin_derivative_list
            case _:
                return None


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1

def xor_list(values: list) -> list:
    return [(lambda row: xor(row[0], row[1]))(row) for row in values]

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
    return max(0, x)

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


def calculate_output_error(x: list, h: list, model: Model) -> list:
    x = x[0][:-1] if len(x[0]) > 1 else x[0]
    error = []
    for y_pred in h[0]:
        y_true = model.get_function()(*x)
        e = y_true - y_pred
        error.append(e)
    return [error]


def get_digit_error(y: list, y_true: int) -> list:
    error = []
    digit = [0.0]*10
    digit[y_true] = 1
    for i, y_pred in enumerate(y[0]):
        err = digit[i] - y_pred
        error.append(err)
    return [error]


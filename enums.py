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
                return functions.get_digit_error


class Act_Func(Enum):
    SIGMOID = 0
    TANH = 1
    IDENTITY = 2
    RELU = 3

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


def xor(a: int, b: int) -> int:
    return 1 if a != b else -1

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
from enum import Enum
import math_functions


class Model(Enum):
    SIN = 0
    COS = 1
    XOR = 2
    DIGIT = 3

    def get_function(self):
        match self:
            case self.SIN:
                return math_functions.sin
            case self.COS:
                return math_functions.cos
            case self.XOR:
                return math_functions.xor
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
    digit = [0.0] * 10
    digit[y_true] = 1
    for i, y_pred in enumerate(y[0]):
        err = digit[i] - y_pred
        error.append(err)
    return [error]
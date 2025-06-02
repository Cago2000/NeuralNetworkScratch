from enum import Enum
import math_functions


class Act_Func(Enum):
    SIGMOID = 0
    TANH = 1
    IDENTITY = 2
    RELU = 3
    SIN = 4

    def get_function(self):
        match self:
            case self.SIGMOID:
                return math_functions.sigmoid_list
            case self.TANH:
                return math_functions.tanh_list
            case self.IDENTITY:
                return math_functions.identity_list
            case self.RELU:
                return math_functions.relu_list
            case self.SIN:
                return math_functions.sin_list
            case _:
                return None

    def get_derivative_function(self):
        match self:
            case self.SIGMOID:
                return math_functions.sigmoid_derivative_list
            case self.TANH:
                return math_functions.tanh_derivative_list
            case self.IDENTITY:
                return math_functions.identity_derivative_list
            case self.RELU:
                return math_functions.relu_derivative_list
            case self.SIN:
                return math_functions.sin_derivative_list
            case _:
                return None

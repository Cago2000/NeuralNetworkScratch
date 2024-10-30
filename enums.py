from enum import Enum
import functions
import math


class Model(Enum):
    SIN = 0
    COS = 1
    XOR = 2

    def get_function(self):
        match self:
            case self.SIN:
                return math.sin
            case self.COS:
                return math.cos
            case self.XOR:
                return functions.xor


class Act_Func(Enum):
    SIGMOID = 0
    TANH = 1
    IDENTITY = 2

    def get_function(self):
        match self:
            case self.SIGMOID:
                return functions.sigmoid_list
            case self.TANH:
                return functions.tanh_list
            case self.IDENTITY:
                return functions.identity_list

    def get_derivative_function(self):
        match self:
            case self.SIGMOID:
                return functions.sigmoid_derivative_list
            case self.TANH:
                return functions.tanh_derivative_list
            case self.IDENTITY:
                return functions.identity_derivative_list

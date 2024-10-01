from math import e


def sigmoid(values):
    result = list(map(lambda x: 1/(1+(e**-x)), values))
    return result


def relu():
    return 1


def tanh():
    return 1


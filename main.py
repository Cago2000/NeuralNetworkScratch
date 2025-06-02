import math
import numpy

import enums
from enums import Model, Act_Func
import neural_network as nn
from matplotlib import pyplot as plt


def main() -> None:
    xor_bias = 1.0
    xor_sample = [[1, -1, xor_bias],
                  [-1, 1, xor_bias],
                  [1, 1, xor_bias],
                  [-1, -1, xor_bias]]

    xor_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    xor_layer_sizes = [3, 3, 1]
    weights_xor, errors_xor = (
        nn.fit(iterations=10000,
            iteration_update=10000,
            data=xor_sample,
            layer_sizes=xor_layer_sizes,
            alpha=0.01,
            error_threshold=1e-5,
            model=Model.XOR,
            act_functions=xor_act_functions,
            y_train=enums.xor_list(xor_sample),
            seed=42))

    nn.predict_all(xor_sample, weights_xor, Model.XOR, True,
                xor_act_functions, xor_layer_sizes, [])
    plt.plot(errors_xor)
    plt.title(f'Model: {Model.XOR.name}')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    sin_x_vals = numpy.linspace(0, 7, 100)
    sin_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    sin_layer_sizes = [2, 3, 1]
    sin_bias = 1.0
    sin_sample = []

    for x_val in sin_x_vals:
        sin_sample.append([x_val, sin_bias])

    weights_sin, errors_sin = (
        nn.fit(iterations=5000,
            iteration_update=100,
            data=sin_sample,
            layer_sizes=sin_layer_sizes,
            alpha=0.01,
            error_threshold=1e-5,
            model=Model.SIN,
            act_functions=sin_act_functions,
            y_train=enums.sin_list(list(sin_x_vals))[0],
            seed=42))

    y_predictions_sin = nn.predict_all(sin_sample, weights_sin, Model.SIN, True,
                                    sin_act_functions, sin_layer_sizes, [])
    plt.plot(errors_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()
    plt.plot(sin_x_vals, y_predictions_sin)
    plt.title(f'Model: {Model.SIN.name}')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

    cos_x_vals = numpy.linspace(0, 7, 100)
    cos_act_functions = [Act_Func.TANH, Act_Func.IDENTITY]
    cos_layer_sizes = [2, 3, 1]
    cos_bias = math.pi/2
    cos_sample = []

    for x_val in cos_x_vals:
        cos_sample.append([x_val, cos_bias])

    weights_cos, errors_cos = (

        nn.fit(iterations=5000,
            iteration_update=100,
            data=cos_sample,
            layer_sizes=cos_layer_sizes,
            alpha=0.01,
            error_threshold=1e-5,
            model=Model.COS,
            act_functions=cos_act_functions,
            y_train=enums.cos_list(list(cos_x_vals))[0],
            seed=42))

    y_predictions_cos = nn.predict_all(cos_sample, weights_cos, Model.COS, True,

                                    cos_act_functions, cos_layer_sizes, [])
    plt.plot(errors_cos)
    plt.ylabel(f'Model: {Model.COS.name}')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()
    plt.plot(cos_x_vals, y_predictions_cos)
    plt.title(f'Model: {Model.COS.name}')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()


if __name__ == "__main__":
    main()

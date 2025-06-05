import math
import numpy
from act_functions import Act_Func
from models import Model
import math_functions
import neural_network as nn
from matplotlib import pyplot as plt

def run_model(name: str,
              model,
              sample_data,
              y_values,
              layer_sizes,
              act_functions,
              iterations=5000,
              iteration_update=100,
              alpha=0.01,
              error_threshold=1e-5,
              seed=42,
              plot_prediction=False,
              x_vals=None):
    weights, errors = nn.fit(
        iterations=iterations,
        iteration_update=iteration_update,
        data=sample_data,
        layer_sizes=layer_sizes,
        alpha=alpha,
        error_threshold=error_threshold,
        model=model,
        act_functions=act_functions,
        y_train=y_values,
        seed=seed
    )

    predictions = nn.predict_all(
        sample_data, weights, model, True, act_functions, y_values
    )

    plt.plot(errors)
    plt.title(f'Model: {name}')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    if plot_prediction and x_vals is not None:
        plt.plot(x_vals, predictions)
        plt.title(f'Model: {name} Prediction')
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show()

    return predictions, errors



def main() -> None:
    xor_bias = 1.0
    xor_sample = [[1, -1, xor_bias],
                  [-1, 1, xor_bias],
                  [1, 1, xor_bias],
                  [-1, -1, xor_bias]]
    xor_y = math_functions.xor_list(xor_sample)
    run_model(name="XOR",
              model=Model.XOR,
              sample_data=xor_sample,
              y_values=xor_y,
              layer_sizes=[3, 3, 1],
              act_functions=[Act_Func.TANH, Act_Func.IDENTITY],
              iterations=10000,
              iteration_update=10000)

    sin_x_vals = numpy.linspace(0, 7, 100)
    sin_sample = [[x, 1.0] for x in sin_x_vals]
    sin_y = math_functions.sin_list(list(sin_x_vals))[0]
    run_model(name="SIN",
              model=Model.SIN,
              sample_data=sin_sample,
              y_values=sin_y,
              layer_sizes=[2, 3, 1],
              act_functions=[Act_Func.TANH, Act_Func.IDENTITY],
              plot_prediction=True,
              x_vals=sin_x_vals)

    cos_x_vals = numpy.linspace(0, 7, 100)
    cos_sample = [[x, math.pi / 2] for x in cos_x_vals]
    cos_y = math_functions.cos_list(list(cos_x_vals))[0]
    run_model(name="COS",
              model=Model.COS,
              sample_data=cos_sample,
              y_values=cos_y,
              layer_sizes=[2, 3, 1],
              act_functions=[Act_Func.TANH, Act_Func.IDENTITY],
              plot_prediction=True,
              x_vals=cos_x_vals)



if __name__ == "__main__":
    main()

import math
import numpy
from act_functions import Act_Func
from models import Model
import math_functions
import neural_network as nn
from matplotlib import pyplot as plt

def run_model(name: str,
              model: Model,
              sample_data: list,
              x_train: list,
              y_train: list,
              layer_sizes: list,
              act_functions: list,
              iterations: int,
              iteration_update: int,
              alpha: float,
              error_threshold: float,
              seed: int,
              plot_prediction: bool,
              x_test: list):
    weights, errors = nn.fit(
        iterations=iterations,
        iteration_update=iteration_update,
        data=sample_data,
        layer_sizes=layer_sizes,
        alpha=alpha,
        error_threshold=error_threshold,
        model=model,
        act_functions=act_functions,
        y_train=y_train,
        seed=seed
    )

    predictions = nn.predict_all(
        x_test, weights, model, True, act_functions, y_train
    )

    plt.plot(errors)
    plt.title(f'Model: {name}')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    if plot_prediction and x_test is not None:
        x_test = [row[:-1] for row in x_test]  # remove bias term for plotting
        plt.plot(x_test, predictions)
        plt.title(f'Model: {name} Prediction')
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show()

    return predictions, errors


def main() -> None:
    xor_bias = 1.0
    xor_x = [[1, -1],
             [-1, 1],
             [1, 1],
             [-1, -1]]
    xor_y = math_functions.xor_list(xor_x)
    xor_sample = [[1, -1, xor_bias],
                  [-1, 1, xor_bias],
                  [1, 1, xor_bias],
                  [-1, -1, xor_bias]]
    run_model(name="XOR",
              model=Model.XOR,
              sample_data=xor_sample,
              x_train=xor_x,
              y_train=xor_y,
              layer_sizes=[3, 3, 1],
              act_functions=[Act_Func.TANH, Act_Func.IDENTITY],
              iterations=10000,
              iteration_update=1000,
              alpha=0.01,
              error_threshold=1e-8,
              seed=42,
              plot_prediction=False,
              x_test=xor_sample)

    sin_bias = 1.0
    sin_x = list(numpy.linspace(0, 7, 300))
    sin_y = math_functions.sin_list(list(sin_x))[0]
    sin_sample = [[x, sin_bias] for x in sin_x]
    sin_test = [[x+0.05, sin_bias] for x in sin_x]
    run_model(name="SIN",
              model=Model.SIN,
              sample_data=sin_sample,
              x_train=sin_x,
              y_train=sin_y,
              layer_sizes=[2, 3, 1],
              act_functions=[Act_Func.SIN, Act_Func.IDENTITY],
              iterations=2000,
              iteration_update=100,
              alpha=0.01,
              error_threshold=1e-5,
              seed=42,
              plot_prediction=True,
              x_test=sin_test)

    cos_bias = math.pi / 2
    cos_x = list(numpy.linspace(0, 7, 300))
    cos_y = math_functions.cos_list(list(cos_x))[0]
    cos_sample = [[x, cos_bias] for x in cos_x]
    cos_test = [[x, cos_bias] for x in cos_x]
    run_model(name="COS",
              model=Model.COS,
              sample_data=cos_sample,
              x_train=cos_x,
              y_train=cos_y,
              layer_sizes=[2, 3, 1],
              act_functions=[Act_Func.SIN, Act_Func.IDENTITY],
              iterations=5000,
              iteration_update=100,
              alpha=0.01,
              error_threshold=0.01,
              seed=42,
              plot_prediction=True,
              x_test=cos_test)


if __name__ == "__main__":
    main()

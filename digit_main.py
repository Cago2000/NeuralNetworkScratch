from matplotlib import pyplot as plt
from enums import Model, Act_Func
import functions


def forward_pass(h: list, weights: list, act_func: Act_Func) -> tuple:
    z = functions.matrix_multiplication(h, weights)
    new_h = act_func.get_function()(z[0])
    return z, new_h


def calculate_output_error(x: list, h: list, model: Model) -> list:
    x = x[0][:-1] if len(x[0]) > 1 else x[0]
    error = []
    for y_pred in h[0]:
        y_true = model.get_function()(*x)
        e = y_true - y_pred
        error.append(e)
    return [error]


def backpropagation(z: list, weights: list, delta: list, act_func: Act_Func) -> list:
    new_delta = []
    derivative = act_func.get_derivative_function()(z[0])
    print(f'delta: {delta}')
    print(f'weights:{weights}')
    error = functions.matrix_multiplication(delta, weights)
    print(f'derivative: {derivative}')
    print(f'error: {error}')
    for derivative_val, error_val in zip(derivative[0], error[0]):
        new_delta.append(derivative_val * error_val)
    print(f'new_delta: {new_delta}')
    return [new_delta]


def adjust_weights(weights: list, delta: list, h: list, alpha: float) -> list:
    print(f'delta: {delta}')
    print(f'h: {h}')
    weight_delta = functions.tensor_product(delta[0], h)
    adjusted_weights = weights.copy()
    index = 0
    print(f'weight_delta: {weight_delta}')
    #print(f'weights:{weights}')

    for i, row in enumerate(weights):
        for j, _ in enumerate(row):
            adjusted_weights[i][j] += (alpha * weight_delta[index])
            index += 1
    return adjusted_weights


def fit(iterations: int, iteration_update: int, data: list, layer_sizes: list,
        alpha: float, error_threshold: float, act_functions: list, y_train: list, kernel: list) -> tuple:
    w_list = functions.return_random_weights(layer_sizes)
    w_list.append([[1 for _ in range(layer_sizes[-1])]])
    all_errors = []
    for i in range(0, iterations):
        if i % iteration_update == 0:
            print(f'Iteration {i + 1}')
        errors = []
        for j, row in enumerate(data):
            conv_row = functions.conv(row, kernel)
            flat_conv_row = functions.flatten_matrix(conv_row)
            h_list = [[flat_conv_row]]
            z_list = []
            d_list = []
            for (k, h), w, act_func in zip(enumerate(h_list), w_list, act_functions):
                if k < len(layer_sizes) - 1:
                    z, h = forward_pass(h, w, act_func)
                    h_list.append(h)
                    z_list.append(z)
            # print(f'h_list: {h_list}')
            # print(f'w_list: {w_list}')
            # print(f'z_list: {z_list}')
            # print(f'y: {h_list[-1]}')
            error = functions.get_digit_error(h_list[-1], y_train[j])
            for err in error[0]:
                errors.append(abs(err))
            # print(f'error: {error}')
            d_list.append(error)
            for z, w, d, act_func in zip(reversed(z_list), reversed(w_list), d_list,
                                         reversed(act_functions)):
                new_d = backpropagation(z, functions.transpose_matrix(w), d, act_func)
                d_list.append(new_d)
            # print("d_list:", d_list)
            for (k, w), d, h in zip(enumerate(reversed(w_list)), d_list, reversed(h_list)):
                if k > 0:
                    w_list[len(w_list) - 1 - k] = adjust_weights(w, d, h, alpha)
            # print(f'w_list after: {w_list}')
        met_threshold = True
        err_temp = 0.0
        for error in errors:
            err_temp += error
            if error > error_threshold:
                met_threshold = False
        all_errors.append(err_temp / len(errors))
        if met_threshold:
            print(f'Threshold met after {i} iterations.')
            return w_list, all_errors
    return w_list, all_errors


def predict(h1: list, weights: list, print_output: bool,
            act_functions: list, y_true: list):
    h_list = [h1]
    z_list = []

    for (k, h), w, act_func in zip(enumerate(h_list), weights, act_functions):
        if k < len(act_functions) - 1:
            z, h = forward_pass(h, w, act_func)
            h_list.append(h)
            z_list.append(z)
    y = h_list[-1]
    if print_output:
        print(f'y: {y}, predicted_digit: {functions.argmax(y)}, true: {y_true}')
    return y[0]


def predict_all(samples: list, weights: list, print_output: bool,
                act_functions: list, kernel: list, y_true: list):
    predictions = []
    if print_output:
        print(f'--------------------Predict Digit Model--------------------')
    for i, sample in enumerate(samples):
        conv_h = functions.conv(sample, kernel)
        flat_conv_h = functions.flatten_matrix(conv_h)
        predictions.append(predict([flat_conv_h], weights, print_output,
                                   act_functions, y_true[i]))
    if print_output:
        print("---------------------------------------------------\n")
    return predictions


def main():
    x_train, y_train, x_test, y_test = functions.transform_digit_data(0.1)
    x_train = functions.rescale_data(x_train)
    x_test = functions.rescale_data(x_test)

    digit_kernel = [[1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0]]
    digit_act_functions = [Act_Func.TANH, Act_Func.SIGMOID, Act_Func.SIGMOID]
    digit_layer_sizes = [((len(x_train[0])-2)**2), 56, 10]
    x_train = [x_train[i] for i in range(0, 1)]

    #prints true value distribution
    digit_counts_train = [0] * 10
    for true_val in y_train:
        digit_counts_train[true_val] += 1
    print(digit_counts_train)
    digit_counts_test = [0] * 10
    for true_val in y_test:
        digit_counts_test[true_val] += 1
    print(digit_counts_test)

    weights_digit, errors_digit = (
        fit(iterations=1,
            data=x_train,
            layer_sizes=digit_layer_sizes,
            alpha=0.1,
            error_threshold=1e-2,
            act_functions=digit_act_functions,
            y_train=y_train,
            iteration_update=10,
            kernel=digit_kernel))
    digit_predictions = predict_all(x_train, weights_digit, True, digit_act_functions, digit_kernel, y_train)

    '''hits = 0
    for pred, y_true in zip(digit_predictions, y_test):
        if functions.argmax([pred]) == y_true:
            hits+=1
        print(f'pred_arr: {pred}, pred_digit: {functions.argmax([pred])},y_true: {y_true}')
    print(f'{hits/len(digit_predictions)}%')'''
    plt.plot(errors_digit)
    plt.ylabel(f'Model: {Model.DIGIT.name}')
    plt.show()


if __name__ == "__main__":
    main()

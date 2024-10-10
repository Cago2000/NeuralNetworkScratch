import functions

alpha = 0.05
bias = 0.5

x1, x2 = 1, 1
layer1 = [x1, x2]  # 1x2
w11, w12, w13, w21, w22, w23 = 0.6, 0.7, 0.5, 0.4, 0.9, 0.6
weights1 = [[w11, w12, w13],  # 2x3
            [w21, w22, w23]]
z1 = []

layer2 = []

w21, w22, w23 = 0.7, 1.8, 0.9
weights2 = [[w21],
            [w22],
            [w23]]
z2 = []


def matrix_multiplication(layer, weights):
    output_size = len(weights[0])
    result = [0] * output_size
    for n, row in zip(layer, weights):
        for i in range(output_size):
            result[i] += n * row[i]
    return result


def forward_pass(layer, weights):
    z = matrix_multiplication(layer, weights)
    result = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    return z, result


def calculate_error(layer):
    error = []
    for layer_value in layer:
        e = -(functions.sigmoid(layer_value, bias))
        error.append(e)
    return error


def output_delta(z, layer):
    delta = []
    error = calculate_error(layer)
    g = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    for g_val, error_val in zip(g, error):
        delta.append(g_val * error_val)
    return delta


def backpropagation(z, weights, delta):
    error = matrix_multiplication(delta, weights)
    new_delta = []
    g = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    for g_val, error_val in zip(g, error):
        new_delta.append(g_val * error_val)
    return new_delta
    # backwards to neuron with derivative with input = e*w


# TODO
def adjust_weights(deltas, weights):
    new_weights = []
    for i in range(0, 2):
        alpha_delta = []
        for val in deltas[i]:
            alpha_delta.append(-alpha * val)
        weight_delta = matrix_multiplication(alpha_delta, weights[i - 1])
        for j in enumerate(weights[i]):
            weights[i][j] += weight_delta[j]
        new_weights.append(weights[i])
    return new_weights


def main():
    z1, layer2 = forward_pass(layer1, weights1)
    z2, layer3 = forward_pass(layer2, weights2)
    delta3 = output_delta(z2, layer3)
    delta2 = backpropagation(z1, weights2, delta3)
    # ones since input does not have weights going into first neuron
    delta1 = backpropagation([1, 1, 1], weights1, delta2)
    print(delta1)


if __name__ == "__main__":
    main()

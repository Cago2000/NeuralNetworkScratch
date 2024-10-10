import functions

alpha = 0.05
bias = 0.5

h0 = [1, 1]
x1, x2 = 1, 1
h1 = [x1, x2]  # 1x2
w11, w12, w13, w21, w22, w23 = 0.6, 0.7, 0.5, 0.4, 0.9, 0.6
weights1 = [[w11, w12, w13],  # 2x3
            [w21, w22, w23]]
z1 = []

h2 = []

w21, w22, w23 = 0.7, 1.8, 0.9
weights2 = [[w21],
            [w22],
            [w23]]
z2 = []


def tensor_product(A, B):
    T = []
    for a in A:
        for b in B:
            T.append(a*b)
    return T


def matrix_multiplication(A, B):
    output_size = len(B[0])
    result = [0] * output_size
    for n, row in zip(A, B):
        for i in range(output_size):
            result[i] += n * row[i]
    return result


def forward_pass(h, weights):
    z = matrix_multiplication(h, weights)
    h = list(map(lambda x: functions.sigmoid_derivative(x, bias), z))
    return z, h


def calculate_error(h):
    error = []
    for layer_value in h:
        e = -(functions.sigmoid(layer_value, bias))
        error.append(e)
    return error


def output_delta(z, h):
    delta = []
    error = calculate_error(h)
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
def adjust_weights(weights, deltas, layers):
    pass



def main():
    z1, h2 = forward_pass(h1, weights1)
    z2, h3 = forward_pass(h2, weights2)
    delta3 = output_delta(z2, h3)
    delta2 = backpropagation(z1, weights2, delta3)
    # ones since input does not have weights going into first neuron
    delta1 = backpropagation([1, 1, 1], weights1, delta2)
    adjust_weights([[], weights1, weights2],[[], delta1, delta2, delta3],[h0, h1, h2, h3])


if __name__ == "__main__":
    main()

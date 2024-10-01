import functions

n1, n2 = 0.2, 0.4
layer1 = [n1, n2]

w11, w12, w13, w21, w22, w23 = 0.6, 0.7, 0.5, 0.4, 0.9, 0.6
weights1 = [[w11, w12, w13],
            [w21, w22, w23]]

w21, w22, w23, w31, w32, w33 = 0.7, 1.8, 0.9, 0.3, 0.3, 0.6
weights2 = [[w21, w31],
            [w22, w32],
            [w23, w33]]


def matrix_multiplication(layer, weights):
    output_size = len(weights[0])
    result = [0] * output_size
    for n, row in zip(layer, weights):
        for i in range(output_size):
            result[i] += n * row[i]
    return result


def forward_pass(layer, weights):
    result = matrix_multiplication(layer, weights)
    result = functions.sigmoid(result)
    return result


def main():
    layer2 = forward_pass(layer1, weights1)
    layer3 = forward_pass(layer2, weights2)
    print(layer3)


if __name__ == "__main__":
    main()

import numpy as np


def relu(x):
    return np.maximum(0, x)


class ACONeuralNet:
    def __init__(self, input_size, hidden_size, output_size, bias_init='random'):
        if bias_init not in ('zero', 'random'):
            raise ValueError("bias_init must be either 'zero' or 'random'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias_init = bias_init
        self.pesi = self.crea_pesi_random()

    def crea_pesi_random(self):
        if self.bias_init == 'zero':
            b1 = np.zeros((1, self.hidden_size))
            b2 = np.zeros((1, self.output_size))
        else:
            b1 = np.random.randn(1, self.hidden_size)
            b2 = np.random.randn(1, self.output_size)

        return [
            np.random.randn(self.input_size, self.hidden_size),
            np.random.randn(self.hidden_size, self.output_size),
            b1,
            b2,
        ]

    def forward(self, X):
        z1 = X @ self.pesi[0] + self.pesi[2]
        a1 = relu(z1)
        z2 = a1 @ self.pesi[1] + self.pesi[3]
        return z2

    def loss(self, X, y_true):
        y_pred = self.forward(X)
        return np.mean((y_pred - y_true) ** 2)

    def get_shapes(self):
        return [
            (self.input_size, self.hidden_size),
            (self.hidden_size, self.output_size),
            (1, self.hidden_size),
            (1, self.output_size),
        ]

    def to_vector(self):
        return np.concatenate([p.ravel() for p in self.pesi])

    def from_vector(self, vector):
        vector = np.asarray(vector, dtype=np.float64).ravel()
        nuovo = []
        start = 0

        for shape in self.get_shapes():
            size = int(np.prod(shape))
            chunk = vector[start:start + size]
            if chunk.size != size:
                raise ValueError('Vector size is incompatible with network shapes')
            nuovo.append(chunk.reshape(shape).copy())
            start += size

        if start != vector.size:
            raise ValueError('Vector has extra values not used by network shapes')

        self.pesi = nuovo

    def copy(self):
        clone = ACONeuralNet(
            self.input_size,
            self.hidden_size,
            self.output_size,
            bias_init=self.bias_init,
        )
        clone.pesi = [p.copy() for p in self.pesi]
        return clone

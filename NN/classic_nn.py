import numpy as np

def relu(x): 
    return np.maximum(0, x)

def mexican_hat(x, center, scale):
    z = (x - center) / scale
    return (1 - z**2) * np.exp(-0.5 * z**2)

def parabola(x):
    return x**2


class ClassicNeuralNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', seed=None, bias_init='zero'):
        if seed is not None:
            np.random.seed(seed)

        if bias_init not in ('zero', 'random'):
            raise ValueError("bias_init must be either 'zero' or 'random'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias_init = bias_init
        self.activation = activation

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = self._init_bias((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = self._init_bias((1, output_size))

    def _init_bias(self, shape):
        if self.bias_init == 'zero':
            return np.zeros(shape)
        return np.random.randn(*shape)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        if self.activation == 'mexican_hat':
            A1 = mexican_hat(Z1, center=0, scale=1)
        elif self.activation == 'relu':
            A1 = relu(Z1)
        elif self.activation == 'parabola':
            A1 = parabola(Z1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        Z2 = A1 @ self.W2 + self.b2
        return Z1, A1, Z2

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

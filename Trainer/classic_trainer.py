import numpy as np

from NN.classic_nn import ClassicNeuralNet


def relu_derivative(x):
    return (x > 0).astype(float)

def mexican_hat_derivative(x, center, scale):
    z = (x - center) / scale
    exp_term = np.exp(-0.5 * z**2)
    d_psi_dz = (-3 * z + z**3) * exp_term
    return d_psi_dz / scale


class ClassicTrainer:
    def __init__(self, model, activation='relu', learning_rate=0.01):
        if not isinstance(model, ClassicNeuralNet):
            raise TypeError("model must be an instance of ClassicNeuralNet")

        self.model = model
        self.activation = activation
        self.learning_rate = learning_rate
        self.losses = []
        self.forward_calls = 0
        self.backward_calls = 0

    def backward(self, X, y, Z1, A1, y_pred):
        m = len(X)

        dZ2 = (y_pred - y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.model.W2.T
        if self.activation == 'mexican_hat':
            dZ1 = dA1 * mexican_hat_derivative(Z1, center=0, scale=1)
        elif self.activation == 'relu':
            dZ1 = dA1 * relu_derivative(Z1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self, X, y, print_every=10, soglia=0.05, max_iter=100000):
        self.losses = []
        self.forward_calls = 0
        self.backward_calls = 0
        loss = float("inf")
        y_pred = None
        should_print = isinstance(print_every, int) and print_every > 0

        while loss > soglia and len(self.losses) < max_iter:
            Z1, A1, Z2 = self.model.forward(X)
            self.forward_calls += 1
            y_pred = Z2

            loss = self.model.loss(y_pred, y)
            self.losses.append(loss)

            dW1, db1, dW2, db2 = self.backward(X, y, Z1, A1, y_pred)
            self.backward_calls += 1

            self.model.W1 -= self.learning_rate * dW1
            self.model.b1 -= self.learning_rate * db1
            self.model.W2 -= self.learning_rate * dW2
            self.model.b2 -= self.learning_rate * db2

            if should_print and len(self.losses) % print_every == 0:
                print(f"Iterazione MLP  {len(self.losses)}, loss: {loss:.4f}")

        return y_pred, self.losses

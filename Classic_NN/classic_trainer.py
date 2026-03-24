<<<<<<< HEAD
import numpy as np

from Classic_NN.classic_nn import ClassicNeuralNet

def relu_derivative(x):
    return (x > 0).astype(float)

class ClassicTrainer:
    def __init__(self, model, learning_rate=0.01):
        if not isinstance(model, ClassicNeuralNet):
            raise TypeError("model must be an instance of ClassicNeuralNet")

        self.model = model
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
        dZ1 = dA1 * relu_derivative(Z1)
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
                print(f"Iterazione {len(self.losses)}, loss: {loss:.4f}")

        return y_pred, self.losses
=======
import numpy as np

from Classic_NN.classic_nn import ClassicNeuralNet

def relu_derivative(x):
    return (x > 0).astype(float)

class ClassicTrainer:
    def __init__(self, model, learning_rate=0.01):
        if not isinstance(model, ClassicNeuralNet):
            raise TypeError("model must be an instance of ClassicNeuralNet")

        self.model = model
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
        dZ1 = dA1 * relu_derivative(Z1)
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
                print(f"Iterazione {len(self.losses)}, loss: {loss:.4f}")

        return y_pred, self.losses
>>>>>>> 45b7e75 (Sync with remote)

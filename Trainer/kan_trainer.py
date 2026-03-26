import numpy as np
from NN.kan import KAN


class KANTrainer:
    def __init__(self, model, learning_rate=0.01):
        if not isinstance(model, KAN):
            raise TypeError("model must be an instance of KAN")

        self.model = model
        self.learning_rate = learning_rate
        self.losses = []
        self.forward_calls = 0
        self.backward_calls = 0

    def backward(self, y_pred, y_true, cache):
        X = cache["X"]
        b1 = cache["b1"]
        b2 = cache["b2"]
        db2_da1 = cache["db2_da1"]

        n = X.shape[0]
        out_dim = y_pred.shape[1]

        # dL/dY for mean over all output entries.
        dY = (2.0 / (n * out_dim)) * (y_pred - y_true)

        grad_b2 = dY.sum(axis=0, keepdims=True)
        grad_c2 = np.einsum("no,nhk->hok", dY, b2)

        # dY/da1 = sum_k c2[h,o,k] * d basis(a1_h)_k / da1_h
        dY_da1 = np.einsum("nhk,hok->nho", db2_da1, self.model.c2)
        dA1 = np.einsum("no,nho->nh", dY, dY_da1)

        # a1 = z1 (identity), so dZ1 = dA1.
        dZ1 = dA1

        grad_b1 = dZ1.sum(axis=0, keepdims=True)
        grad_c1 = np.einsum("nh,nik->ihk", dZ1, b1)

        return grad_c1, grad_b1, grad_c2, grad_b2

    def train(self, X, y, print_every=10, soglia=0.05, max_iter=100000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.model.input_size:
            raise ValueError(f"X must have shape (n_samples, {self.model.input_size})")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2 or y.shape[1] != self.model.output_size or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape (n_samples, {self.model.output_size}), got {y.shape}"
            )

        self.losses = []
        self.forward_calls = 0
        self.backward_calls = 0
        loss = float("inf")
        y_pred = None
        should_print = isinstance(print_every, int) and print_every > 0

        while loss > soglia and len(self.losses) < max_iter:
            y_pred, cache = self.model.forward(X)
            self.forward_calls += 1

            loss = self.model._mse(y_pred, y)
            self.losses.append(loss)

            grad_c1, grad_b1, grad_c2, grad_b2 = self.backward(y_pred, y, cache)
            self.backward_calls += 1

            self.model.c1 -= self.learning_rate * grad_c1
            self.model.b1 -= self.learning_rate * grad_b1
            self.model.c2 -= self.learning_rate * grad_c2
            self.model.b2 -= self.learning_rate * grad_b2

            if should_print and len(self.losses) % print_every == 0:
                print(f"Iterazione {len(self.losses)}, loss: {loss:.6f}")

        return y_pred, self.losses

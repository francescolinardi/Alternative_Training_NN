<<<<<<< HEAD
import numpy as np

def relu(x):
    return np.maximum(0, x) 

class GeneticNeuralNet:
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
            b2
        ]

    def forward(self, X):
        Z1 = X @ self.pesi[0] + self.pesi[2]
        A1 = relu(Z1)
        Z2 = A1 @ self.pesi[1] + self.pesi[3]
        return Z2

    def loss(self, X, y_true):
        y_pred = self.forward(X)
        return np.mean((y_pred - y_true) ** 2)

    def crossover(self, other):
        nuovi_pesi = [(p1 + p2) / 2 for p1, p2 in zip(self.pesi, other.pesi)]
        figlio = GeneticNeuralNet(
            self.input_size,
            self.hidden_size,
            self.output_size,
            bias_init=self.bias_init,
        )
        figlio.pesi = nuovi_pesi
        return figlio

    def muta(self, mutation_rate=0.1, mutation_strength=0.1):
        nuovi_pesi = []
        for param in self.pesi:
            if np.random.rand() < mutation_rate:
                noise = mutation_strength * np.random.randn(*param.shape)
                nuovi_pesi.append(param + noise)
            else:
                nuovi_pesi.append(param.copy())
        self.pesi = nuovi_pesi
























=======
import numpy as np

def relu(x):
    return np.maximum(0, x) 

class GeneticNeuralNet:
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
            b2
        ]

    def forward(self, X):
        Z1 = X @ self.pesi[0] + self.pesi[2]
        A1 = relu(Z1)
        Z2 = A1 @ self.pesi[1] + self.pesi[3]
        return Z2

    def loss(self, X, y_true):
        y_pred = self.forward(X)
        return np.mean((y_pred - y_true) ** 2)

    def crossover(self, other):
        nuovi_pesi = [(p1 + p2) / 2 for p1, p2 in zip(self.pesi, other.pesi)]
        figlio = GeneticNeuralNet(
            self.input_size,
            self.hidden_size,
            self.output_size,
            bias_init=self.bias_init,
        )
        figlio.pesi = nuovi_pesi
        return figlio

    def muta(self, mutation_rate=0.1, mutation_strength=0.1):
        nuovi_pesi = []
        for param in self.pesi:
            if np.random.rand() < mutation_rate:
                noise = mutation_strength * np.random.randn(*param.shape)
                nuovi_pesi.append(param + noise)
            else:
                nuovi_pesi.append(param.copy())
        self.pesi = nuovi_pesi
























>>>>>>> 45b7e75 (Sync with remote)

import numpy as np
from ACO_NN.aco_nn import ACONeuralNet


class ACOTrainer:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_ants,
        elite_fraction=0.25,
        evaporation_rate=0.2,
        pheromone_learning_rate=0.4,
        init_sigma=1.0,
        min_sigma=1e-3,
        bias_init='random',
        seed=None,
    ):
        if bias_init not in ('zero', 'random'):
            raise ValueError("bias_init must be either 'zero' or 'random'")
        if n_ants < 2:
            raise ValueError('n_ants must be at least 2')
        if not (0 < elite_fraction <= 1):
            raise ValueError('elite_fraction must be in (0, 1]')
        if not (0 <= evaporation_rate <= 1):
            raise ValueError('evaporation_rate must be in [0, 1]')
        if not (0 < pheromone_learning_rate <= 1):
            raise ValueError('pheromone_learning_rate must be in (0, 1]')
        if init_sigma <= 0:
            raise ValueError('init_sigma must be > 0')
        if min_sigma <= 0:
            raise ValueError('min_sigma must be > 0')

        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.n_ants = n_ants
        self.elite_fraction = elite_fraction
        self.evaporation_rate = evaporation_rate
        self.pheromone_learning_rate = pheromone_learning_rate
        self.min_sigma = min_sigma
        self.bias_init = bias_init
        self.forward_calls = 0
        self.losses = []

        self.template = ACONeuralNet(input_size, hidden_size, output_size, bias_init=bias_init)
        self.dimension = self.template.to_vector().size

        # In continuous ACO, pheromone is represented by a normal distribution.
        self.mu = self.template.to_vector().copy()
        self.sigma = np.full(self.dimension, init_sigma, dtype=np.float64)

        self.population = []
        self.best_individual = self.template.copy()
        self.best_loss = float('inf')

    def _sample_ant(self):
        sampled = self.mu + self.sigma * self.rng.standard_normal(self.dimension)
        ant = ACONeuralNet(
            self.template.input_size,
            self.template.hidden_size,
            self.template.output_size,
            bias_init=self.bias_init,
        )
        ant.from_vector(sampled)
        return ant

    def _evaluate_population(self, X, y):
        self.population = [self._sample_ant() for _ in range(self.n_ants)]
        evaluated = [(ant, ant.loss(X, y)) for ant in self.population]
        self.forward_calls += len(evaluated)
        evaluated.sort(key=lambda item: item[1])
        return evaluated

    def _update_pheromones(self, elite_vectors):
        elite_mean = elite_vectors.mean(axis=0)
        elite_std = elite_vectors.std(axis=0)

        alpha = self.pheromone_learning_rate
        rho = self.evaporation_rate

        self.mu = (1 - alpha) * self.mu + alpha * elite_mean
        sigma_target = np.maximum(elite_std, self.min_sigma)
        self.sigma = (1 - rho) * self.sigma + rho * sigma_target
        self.sigma = np.maximum(self.sigma, self.min_sigma)

    def evoluzione(self, X, y):
        evaluated = self._evaluate_population(X, y)

        current_best, current_loss = evaluated[0]
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_individual = current_best.copy()

        n_elite = max(1, int(round(self.n_ants * self.elite_fraction)))
        elite = evaluated[:n_elite]
        elite_vectors = np.vstack([ant.to_vector() for ant, _ in elite])
        self._update_pheromones(elite_vectors)

        return current_loss

    def train(self, X, y, print_every=10, soglia=0.01, max_iter=10000):
        self.losses = []
        should_print = isinstance(print_every, int) and print_every > 0

        acc = float('inf')
        while acc > soglia and len(self.losses) < max_iter:
            acc = self.evoluzione(X, y)
            self.losses.append(acc)
            if should_print and len(self.losses) % print_every == 0:
                print(f'Evoluzione {len(self.losses)}, loss: {acc:.4f}')
        return self.losses

import copy
import numpy as np

from NN.genetic_nn import GeneticNeuralNet


class GeneticTrainer:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        population_size,
        mutation_rate=0.1,
        mutation_strength=0.1,
        bias_init='random',
    ):
        if bias_init not in ('zero', 'random'):
            raise ValueError("bias_init must be either 'zero' or 'random'")

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.bias_init = bias_init
        self.forward_calls = 0
        self.population = [
            GeneticNeuralNet(input_size, hidden_size, output_size, bias_init=bias_init)
            for _ in range(population_size)
        ]
        self.losses = []

    def evoluzione(self, X, y):
        valutati = [(ind, ind.loss(X, y)) for ind in self.population]
        self.forward_calls += len(valutati)
        valutati.sort(key=lambda x: x[1])

        acc = valutati[0][1]
        k = self.population_size // 2
        selezionati = [ind for ind, _ in valutati[:k]]

        nuova_gen = copy.deepcopy(selezionati)

        for i in range(k - 1):
            figlio = selezionati[i].crossover(selezionati[i + 1])
            figlio.muta(self.mutation_rate, self.mutation_strength)
            nuova_gen.append(figlio)

        figlio = selezionati[-1].crossover(selezionati[0])
        figlio.muta(self.mutation_rate, self.mutation_strength)
        nuova_gen.append(figlio)

        self.population = nuova_gen
        return acc

    def torneo(self, valutati, k):
        indici = np.random.choice(len(valutati), size=k, replace=False)
        partecipanti = [valutati[i] for i in indici]
        vincitore = min(partecipanti, key=lambda x: x[1])
        return vincitore[0]

    def evoluzione_torneo(self, X, y, k_torneo=4):
        valutati = [(ind, ind.loss(X, y)) for ind in self.population]
        self.forward_calls += len(valutati)

        k = self.population_size // 2
        idx_best = int(np.argmin([v[1] for v in valutati]))
        best = valutati[idx_best][0]
        acc = valutati[idx_best][1]

        if k > 1:
            selezionati = [best] + [self.torneo(valutati, k=k_torneo) for _ in range(k - 1)]
        else:
            selezionati = [best]

        nuova_gen = copy.deepcopy(selezionati)

        for i in range(k - 1):
            figlio = selezionati[i].crossover(selezionati[i + 1])
            figlio.muta(self.mutation_rate, self.mutation_strength)
            nuova_gen.append(figlio)

        figlio = selezionati[-1].crossover(selezionati[0])
        figlio.muta(self.mutation_rate, self.mutation_strength)
        nuova_gen.append(figlio)

        self.population = nuova_gen
        return acc

    def stochastic_universal_sampling(self, fitness, num_to_select):
        fitness = np.array(fitness, dtype=np.float64)
        fitness /= fitness.sum()

        cumulative_fitness = np.cumsum(fitness)
        step = 1.0 / num_to_select
        start = np.random.uniform(0, step)
        pointers = [start + i * step for i in range(num_to_select)]

        selected_indices = []
        i, j = 0, 0
        while i < num_to_select and j < len(cumulative_fitness):
            if pointers[i] < cumulative_fitness[j]:
                selected_indices.append(j)
                i += 1
            else:
                j += 1
        return selected_indices

    def evoluzione_roulette(self, X, y):
        valutati = [(ind, ind.loss(X, y)) for ind in self.population]
        self.forward_calls += len(valutati)
        errori = np.array([v[1] for v in valutati])
        epsilon = 1e-8
        fitness = 1 / (errori + epsilon)
        fitness /= fitness.sum()

        idx_best = np.argmin(errori)
        best = valutati[idx_best][0]
        acc = errori[idx_best]

        indici = np.random.choice(
            len(self.population),
            size=(self.population_size // 2) - 1,
            replace=False,
            p=fitness,
        )
        selezionati = [best] + [valutati[i][0] for i in indici]

        nuova_gen = copy.deepcopy(selezionati)

        for i in range(len(selezionati) - 1):
            figlio = selezionati[i].crossover(selezionati[i + 1])
            figlio.muta(self.mutation_rate, self.mutation_strength)
            nuova_gen.append(figlio)

        figlio = selezionati[-1].crossover(selezionati[0])
        figlio.muta(self.mutation_rate, self.mutation_strength)
        nuova_gen.append(figlio)

        self.population = nuova_gen
        return acc

    def evoluzione_roulette_sus(self, X, y):
        valutati = [(ind, ind.loss(X, y)) for ind in self.population]
        self.forward_calls += len(valutati)
        errori = np.array([v[1] for v in valutati])

        epsilon = 1e-8
        fitness = 1 / (errori + epsilon)
        fitness /= fitness.sum()

        idx_best = np.argmin(errori)
        best = valutati[idx_best][0]
        acc = errori[idx_best]

        n_sus = (self.population_size // 2) - 1
        if n_sus > 0:
            indici = self.stochastic_universal_sampling(fitness, n_sus)
            selezionati = [best] + [valutati[i][0] for i in indici]
        else:
            selezionati = [best]

        nuova_gen = copy.deepcopy(selezionati)

        for i in range(len(selezionati) - 1):
            figlio = selezionati[i].crossover(selezionati[i + 1])
            figlio.muta(self.mutation_rate, self.mutation_strength)
            nuova_gen.append(figlio)

        figlio = selezionati[-1].crossover(selezionati[0])
        figlio.muta(self.mutation_rate, self.mutation_strength)
        nuova_gen.append(figlio)

        self.population = nuova_gen
        return acc

    def train(self, X, y, print_every=10, soglia=0.01, max_iter=10000, strategy='roulette', k_torneo=4):
        self.losses = []
        should_print = isinstance(print_every, int) and print_every > 0

        strategy_map = {
            'sort': self.evoluzione,
            'torneo': self.evoluzione_torneo,
            'roulette': self.evoluzione_roulette,
            'roulette_sus': self.evoluzione_roulette_sus,
        }

        if strategy not in strategy_map:
            disponibili = ', '.join(strategy_map.keys())
            raise ValueError(f"Strategia '{strategy}' non valida. Strategie disponibili: {disponibili}")

        acc = float('inf')
        while acc > soglia and len(self.losses) < max_iter:
            if strategy == 'torneo':
                acc = strategy_map[strategy](X, y, k_torneo=k_torneo)
            else:
                acc = strategy_map[strategy](X, y)
            self.losses.append(acc)
            if should_print and len(self.losses) % print_every == 0:
                print(f'Evoluzione GA {strategy} {len(self.losses)}, loss: {acc:.4f}')
        return self.losses

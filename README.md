# Alternative_Training_NN

Comparison of different feed-forward neural network training methods implemented from scratch in NumPy:

- `Backpropagation` (classic baseline)
- `Genetic Algorithm (GA)` with multiple selection strategies
- `Ant Colony Optimization (ACO)` in a continuous formulation

The goal of this project is to evaluate how alternatives to backprop perform in terms of:

- final quality (`MSE`)
- computational cost (`forward_calls` / `compute_units` counters)
- convergence speed

## Repository structure

```text
Alternative_Training_NN/
|- Classic_NN/
|  |- classic_nn.py
|  |- classic_trainer.py
|- Genetic_NN/
|  |- genetic_nn.py
|  |- genetic_trainer.py
|- ACO_NN/
|  |- aco_nn.py
|  |- aco_trainer.py
|- experiments.py
|- Backpropagation_VS_Genetic.ipynb
|- pictures/
|- README.md
```

## Main modules

### `Classic_NN`

- `ClassicNeuralNet`: single-hidden-layer network with ReLU activation
- `ClassicTrainer`: gradient-based training (backprop), tracking:
  - `forward_calls`
  - `backward_calls`
  - `losses` history

### `Genetic_NN`

- `GeneticNeuralNet`: genetic individual with mutable weights/biases
- `GeneticTrainer`: population evolution with strategies:
  - `sort`
  - `torneo`
  - `roulette`
  - `roulette_sus`

### `ACO_NN`

- `ACONeuralNet`: network representable as a continuous vector
- `ACOTrainer`: continuous ACO-style optimization with:
  - pheromone mean `mu`
  - standard deviation `sigma`
  - evaporation and elite-based learning

### `experiments.py`

Contains a repeatable pipeline for multi-run benchmarks:

- `run_single_experiment(...)`
- `run_experiments(...)`
- `summarize_results(df)`

Main output: a `pandas.DataFrame` with comparable metrics across methods/strategies.

## Environment setup

Minimum requirements:

- Python 3.10+
- packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `jupyter`

Example (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn jupyter
```

## How to run

### 1. Notebook mode (recommended)

Open `Backpropagation_VS_Genetic.ipynb` and run the cells in order.

The notebook:

- generates a synthetic 3D dataset
- trains Backprop, GA (multiple strategies), and ACO
- compares losses, predictions, and computational cost
- uses `experiments.py` for hyperparameter sweeps

### 2. Script mode (batch experiments)

You can import the classes and call `run_experiments(...)` from your own Python script.

Minimal example:

```python
from experiments import run_experiments
from Classic_NN.classic_nn import ClassicNeuralNet
from Classic_NN.classic_trainer import ClassicTrainer
from Genetic_NN.genetic_trainer import GeneticTrainer
from ACO_NN.aco_trainer import ACOTrainer

# Define X, y
df = run_experiments(
    X=X,
    y=y,
    seeds=[42],
    hidden_sizes=[3, 4, 5],
    soglie=[0.05, 0.02],
    max_iter=20000,
    bias_init='random',
    learning_rate=0.01,
    population_sizes=[8, 16, 32],
    mutation_rate=0.1,
    mutation_strength=0.1,
    ga_strategies=['sort', 'torneo', 'roulette', 'roulette_sus'],
    elite_fraction=0.25,
    evaporation_rate=1.0,
    pheromone_learning_rate=1.0,
    ClassicNeuralNet=ClassicNeuralNet,
    ClassicTrainer=ClassicTrainer,
    GeneticTrainer=GeneticTrainer,
    ACOTrainer=ACOTrainer,
)
```

## Metrics used

- `final_loss`: final MSE
- `iterations`: number of iterations/evolutions
- `forward_calls`: direct estimate of inference cost during training
- `compute_units`:
  - Backprop: `forward_calls + 2 * backward_calls`
  - GA/ACO: `forward_calls`
- `converged`: whether `soglia` (threshold) was reached
- `time`: total experiment runtime

## Practical notes

- `venv/`, `__pycache__/`, and `*.pyc` files should not be versioned (already in `.gitignore`).
- If you see `SyntaxError` messages with `<<<<<<<` / `>>>>>>>`, a Git merge conflict was left unresolved in a Python file.
- For reproducible comparisons, always set a fixed `seed`.

## Possible extensions

- multi-layer support
- new objective functions and real datasets
- structured logging to CSV/Parquet
- statistical analysis over multiple runs/seeds

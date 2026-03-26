import numpy as np
import pandas as pd
import time


def run_single_experiment(
    X, y, seed, hidden_size, soglia, max_iter, bias_init, 
    learning_rate, population_size, mutation_rate,
    mutation_strength, ga_strategies, elite_fraction, 
    evaporation_rate, pheromone_learning_rate,
    ClassicNeuralNet, ClassicTrainer, GeneticTrainer, ACOTrainer,
    KAN, KANTrainer, kan_n_wavelets, kan_wavelet,
):
    np.random.seed(seed)
    results = []
    # =======================
    # BACKPROP
    # =======================
    start = time.time()

    model = ClassicNeuralNet(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        seed=seed,
        bias_init=bias_init,
    )

    trainer = ClassicTrainer(model, learning_rate=learning_rate)

    y_pred, losses = trainer.train(
        X, y,
        print_every=None,
        soglia=soglia,
        max_iter=max_iter
    )

    elapsed = time.time() - start

    results.append({
        "method": "backprop",
        "strategy": None,
        "seed": seed,
        "hidden_size": hidden_size,
        "soglia": soglia,
        "population_size": population_size,
        "iterations": len(losses),
        "final_loss": losses[-1],
        "compute_units": trainer.forward_calls + 2 * trainer.backward_calls,
        "forward_calls": trainer.forward_calls,
        "time": elapsed,
        "converged": losses[-1] <= soglia,
        "learning_rate": learning_rate
    })

    # =======================
    # KAN
    # =======================
    start = time.time()

    kan_model = KAN(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        n_wavelets=kan_n_wavelets,
        wavelet=kan_wavelet,
        seed=seed,
    )

    kan_trainer = KANTrainer(kan_model, learning_rate=learning_rate)

    y_pred, losses = kan_trainer.train(
        X, y,
        print_every=None,
        soglia=soglia,
        max_iter=max_iter
    )

    elapsed = time.time() - start

    results.append({
        "method": "KAN",
        "strategy": kan_wavelet,
        "seed": seed,
        "hidden_size": hidden_size,
        "soglia": soglia,
        "population_size": population_size,
        "iterations": len(losses),
        "final_loss": losses[-1],
        "compute_units": kan_trainer.forward_calls + 2 * kan_trainer.backward_calls,
        "forward_calls": kan_trainer.forward_calls,
        "time": elapsed,
        "converged": losses[-1] <= soglia,
        "learning_rate": learning_rate,
        "n_wavelets": kan_n_wavelets,
    })

    # =======================
    # GA
    # =======================
    for strategy in ga_strategies:
        start = time.time()

        ga_trainer = GeneticTrainer(
            input_size=X.shape[1],
            hidden_size=hidden_size,
            output_size=1,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            bias_init=bias_init,
        )

        losses = ga_trainer.train(
            X,
            y,
            print_every=None,
            soglia=soglia,
            max_iter=max_iter,
            strategy=strategy,
            k_torneo=4,
        )

        elapsed = time.time() - start

        results.append({
            "method": "GA",
            "strategy": strategy,
            "seed": seed,
            "hidden_size": hidden_size,
            "soglia": soglia,
            "iterations": len(losses),
            "final_loss": losses[-1],
            "compute_units": ga_trainer.forward_calls,
            "forward_calls": ga_trainer.forward_calls,
            "time": elapsed,
            "converged": losses[-1] <= soglia,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
        })

    # =======================
    # ACO
    # =======================
    start = time.time()

    aco_trainer = ACOTrainer(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        output_size=1,
        n_ants=population_size,  # Using population_size as n_ants for consistency
        elite_fraction=elite_fraction,
        evaporation_rate=evaporation_rate,
        pheromone_learning_rate=pheromone_learning_rate,
        init_sigma=1.0,
        min_sigma=1e-3,
        bias_init=bias_init,
        seed=seed,
    )

    losses = aco_trainer.train(
        X,
        y,
        print_every=None,
        soglia=soglia,
        max_iter=max_iter,
    )

    elapsed = time.time() - start

    results.append({
        "method": "ACO",
        "strategy": None,
        "seed": seed,
        "hidden_size": hidden_size,
        "soglia": soglia,
        "population_size": population_size,
        "iterations": len(losses),
        "final_loss": losses[-1],
        "compute_units": aco_trainer.forward_calls,
        "forward_calls": aco_trainer.forward_calls,
        "n_ants": population_size,
        "time": elapsed,
        "converged": losses[-1] <= soglia,
        "elite_fraction": elite_fraction,
        "evaporation_rate": evaporation_rate,
        "pheromone_learning_rate": pheromone_learning_rate,
    })

    return results


def run_experiments(
    X,
    y,
    seeds,
    hidden_sizes,
    soglie,
    max_iter,
    bias_init,
    learning_rate,
    population_sizes,
    mutation_rate,
    mutation_strength,
    ga_strategies,
    elite_fraction,
    evaporation_rate,
    pheromone_learning_rate,
    ClassicNeuralNet,
    ClassicTrainer,
    GeneticTrainer,
    ACOTrainer,
    KAN,
    KANTrainer,
    kan_n_wavelets=8,
    kan_wavelet="mexican_hat",
):
    all_results = []

    for seed in seeds:
        for hidden_size in hidden_sizes:
            for soglia in soglie:
                for population_size in population_sizes:

                    res = run_single_experiment(
                        X,
                        y,
                        seed,
                        hidden_size,
                        soglia,
                        max_iter,
                        bias_init,
                        learning_rate,
                        population_size,
                        mutation_rate,
                        mutation_strength,
                        ga_strategies,
                        elite_fraction,
                        evaporation_rate,
                        pheromone_learning_rate,
                        ClassicNeuralNet,
                        ClassicTrainer,
                        GeneticTrainer,
                        ACOTrainer,
                        KAN,
                        KANTrainer,
                        kan_n_wavelets,
                        kan_wavelet,
                    )

                    all_results.extend(res)

    df = pd.DataFrame(all_results)
    return df


def summarize_results(df):
    summary = df.copy()

    # Keep methods without an explicit strategy (Backprop/ACO) in grouped output.
    summary["strategy"] = summary["strategy"].fillna("-")

    grouped = summary.groupby(["method", "strategy"], dropna=False).agg({
        "final_loss": ["mean", "std"],
        "compute_units": "mean",
        "converged": "mean",
        "time": "mean"
    }).reset_index()

    grouped.columns = [
        "method",
        "strategy",
        "final_loss_mean",
        "final_loss_std",
        "compute_units_mean",
        "converged_rate",
        "time_mean",
    ]

    return grouped
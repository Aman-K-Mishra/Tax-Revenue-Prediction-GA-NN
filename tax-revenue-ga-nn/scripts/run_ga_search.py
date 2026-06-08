import random
import numpy as np
import torch, json, os
from deap import base, creator, tools

from src.utils.config import load_config
from src.data.load_data import load_dataset
from src.training.train_eval import train_and_eval

# DEAP setup
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", dict, fitness=creator.FitnessMin)

def create_individual():
    return creator.Individual({
        "layers": random.choice([
            [64, 64],
            [128, 64],
            [128, 128],
            [256, 128],
            [256, 128, 64]
        ]),
        "activation": random.choice(["relu", "tanh"])
    })

def mutate(ind):
    if random.random() < 0.5:
        ind["layers"] = random.choice([
            [64, 64],
            [128, 64],
            [128, 128],
            [256, 128],
            [256, 128, 64]
        ])
    if random.random() < 0.5:
        ind["activation"] = random.choice(["relu", "tanh"])
    return ind

def main():
    cfg = load_config()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_scaler, y_scaler
    ) = load_dataset(cfg.csv_path, cfg.target_column)

    # INITIAL POPULATION
    population = [create_individual() for _ in range(cfg.population_size)]

    for gen in range(cfg.generations):
        # Evaluate population
        fitnesses = [train_and_eval(ind, X_train, y_train, X_val, y_val)
                     for ind in population]

        # Sort by fitness
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])]

        # Keep elites
        population = sorted_pop[: cfg.population_size // 2]

        # Breed
        while len(population) < cfg.population_size:
            p1, p2 = random.sample(sorted_pop[:5], 2)
            child = {
                "layers": random.choice([p1["layers"], p2["layers"]]),
                "activation": random.choice([p1["activation"], p2["activation"]])
            }
            child = mutate(child)
            population.append(creator.Individual(child))

    # ⚠️ FINAL EVALUATION HERE (IMPORTANT)
    final_fitnesses = [train_and_eval(ind, X_train, y_train, X_val, y_val)
                       for ind in population]

    best_idx = np.argmin(final_fitnesses)
    best = population[best_idx]

    # Save best
    os.makedirs("models", exist_ok=True)
    with open("models/best_architecture.json", "w") as f:
        json.dump(best, f, indent=4)

    print("\nBEST FOUND:", best)
    print("Saved best architecture to models/best_architecture.json")

if __name__ == "__main__":
    main()

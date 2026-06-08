from typing import List, Callable
from .chromosome import random_chromosome, Chromosome


def initialize_population(size: int) -> List[Chromosome]:
    """Create initial population of random chromosomes."""
    return [random_chromosome() for _ in range(size)]


def evaluate_population(
    population: List[Chromosome],
    eval_fn: Callable[[Chromosome], float]
) -> List[float]:
    """
    Evaluate every individual using eval_fn, which should
    return lower values for better individuals (e.g., MSE).
    """
    fitnesses = []
    for chrom in population:
        fitness = eval_fn(chrom)
        fitnesses.append(fitness)
    return fitnesses

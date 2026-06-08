import random
from typing import Tuple
from .chromosome import ACTIVATIONS, Chromosome, HIDDEN_LAYER_SIZES


def mutate(chrom: Chromosome, p_layer: float = 0.5) -> Chromosome:
    """
    Mutate a chromosome in place: either change a layer size or an activation.
    """
    if not chrom.get("layers"):
        return chrom

    idx = random.randrange(len(chrom["layers"]))

    # mutate layer size
    if random.random() < p_layer:
        chrom["layers"][idx] = random.choice(HIDDEN_LAYER_SIZES)
    # mutate activation
    else:
        chrom["activations"][idx] = random.choice(ACTIVATIONS)

    return chrom


def crossover(c1: Chromosome, c2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """
    Single-point crossover between two parents.
    """
    min_len = min(len(c1["layers"]), len(c2["layers"]))
    if min_len < 1:
        return c1, c2

    point = random.randint(1, min_len)

    child1 = {
        "layers": c1["layers"][:point] + c2["layers"][point:],
        "activations": c1["activations"][:point] + c2["activations"][point:]
    }

    child2 = {
        "layers": c2["layers"][:point] + c1["layers"][point:],
        "activations": c2["activations"][:point] + c1["activations"][point:]
    }

    return child1, child2

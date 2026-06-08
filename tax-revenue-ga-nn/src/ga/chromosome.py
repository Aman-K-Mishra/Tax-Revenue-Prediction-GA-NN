import random
from typing import Dict, List

ACTIVATIONS = ["relu", "tanh", "sigmoid"]

HIDDEN_LAYER_SIZES = [16, 32, 64, 128, 256]
MIN_LAYERS = 1
MAX_LAYERS = 4

# Chromosome is just a dict with "layers" and "activations" lists
Chromosome = Dict[str, List]


def random_chromosome() -> Chromosome:
    """
    Example chromosome:
    {
        "layers": [64, 32],
        "activations": ["relu", "tanh"]
    }
    """
    n_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    layers = [random.choice(HIDDEN_LAYER_SIZES) for _ in range(n_layers)]
    activations = [random.choice(ACTIVATIONS) for _ in range(n_layers)]
    return {"layers": layers, "activations": activations}

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class GAConfig:
    # GA hyperparameters
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.9
    mutation_prob: float = 0.4
    random_seed: int = 42

    # Training hyperparameters
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-3

    # Data
    target_column: str = "tax_revenue"
    csv_path: str = "data/raw/tax_sample.csv"

    # Prediction feature order
    predict_features: list = field(default_factory=lambda: [
        "gdp",
        "inflation",
        "population",
        "imports",
        "exports",
        "corporate_tax_rate"
    ])


def load_config(path: str = "configs/config.yaml") -> GAConfig:
    """
    Load YAML config and override defaults in GAConfig.
    """
    cfg = GAConfig()   # create with default values

    p = Path(path)
    if p.exists():
        with p.open("r") as f:
            data = yaml.safe_load(f) or {}

        # Override fields dynamically
        for field in cfg.__dataclass_fields__:
            if field in data:
                setattr(cfg, field, data[field])

    return cfg

from typing import List, Optional
import matplotlib.pyplot as plt
import os


def plot_fitness(
    history: List[float],
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot GA fitness (e.g., validation loss) over generations.
    """
    if not history:
        return

    plt.figure()
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Validation Loss (MSE)")
    plt.title("GA Evolution of Validation Loss")
    plt.grid(True)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

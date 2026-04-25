"""Training module — GRPO training for Solver and Adversary."""
from .train_solver import train_solver, SolverTrainingConfig
from .train_adversary import train_adversary, AdversaryTrainingConfig

__all__ = [
    "train_solver", "SolverTrainingConfig",
    "train_adversary", "AdversaryTrainingConfig",
]

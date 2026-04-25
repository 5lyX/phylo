"""
Phylo — Sim2Reason Physics Reasoning Environment.

A MuJoCo-backed physics reasoning environment with adversarial self-play,
implemented as an OpenEnv environment for training and evaluating LLMs
on physics reasoning tasks.

Architecture:
  - Solver (Qwen2.5-3B): reasons about physics problems
  - Adversary (Qwen2.5-0.5B): proposes scenes to challenge the Solver
  - PhyloEnvironment: MuJoCo simulation via Sim2Reason pipeline

Modules:
  - server/: OpenEnv HTTP server
  - adversary/: Adversary LLM + curriculum
  - solver/: Solver inference + reward
  - self_play/: Flywheel orchestrator + replay buffer
  - training/: GRPO training for Solver and Adversary
  - evaluation/: Benchmarking suite
"""

from .client import PhyloEnv
from .models import (
    PhysicsAction,
    PhysicsObservation,
    AdversaryAction,
    AdversaryObservation,
    # Backward-compatible aliases
    PhyloAction,
    PhyloObservation,
)

__all__ = [
    "PhysicsAction",
    "PhysicsObservation",
    "AdversaryAction",
    "AdversaryObservation",
    "PhyloAction",
    "PhyloObservation",
    "PhyloEnv",
]

"""Curriculum tracking for the adversarial self-play loop."""

from collections import defaultdict
from typing import Dict, List, Tuple


class CurriculumTracker:
    """
    EMA-based curriculum tracker for the adversarial self-play system.

    Tracks solver success rate per (scene_type, difficulty) pair using
    Exponential Moving Average. Provides difficulty sampling weights
    so the adversary focuses on the solver's weakest areas.
    """

    def __init__(self, alpha: float = 0.1, initial_value: float = 0.5):
        """
        Args:
            alpha: EMA smoothing factor (0=no update, 1=instant).
            initial_value: Starting accuracy assumption (0.5 = unknown).
        """
        self.alpha = alpha
        self.initial_value = initial_value
        self._accuracy: Dict[Tuple[str, str], float] = defaultdict(lambda: initial_value)
        self._update_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def update(self, scene_type: str, difficulty: str, solver_succeeded: bool) -> None:
        """Update EMA accuracy for a (scene_type, difficulty) pair."""
        key = (scene_type, difficulty.upper())
        old = self._accuracy[key]
        self._accuracy[key] = (1 - self.alpha) * old + self.alpha * float(solver_succeeded)
        self._update_counts[key] += 1

    def get_accuracy(self, scene_type: str, difficulty: str) -> float:
        """Get current EMA accuracy for a pair."""
        return self._accuracy[(scene_type, difficulty.upper())]

    def get_sampling_weight(self, scene_type: str, difficulty: str) -> float:
        """
        Sampling weight for the adversary: inversely proportional to accuracy.
        Higher weight = solver struggles more here = adversary should focus here.
        """
        acc = self.get_accuracy(scene_type, difficulty)
        return max(1.0 - acc, 0.01)  # Avoid zero weight

    def get_difficulty_map(self) -> Dict[str, float]:
        """Return average accuracy per scene_type."""
        from adversary.adversary_config import VALID_SCENE_TYPES, VALID_DIFFICULTIES
        result = {}
        for st in VALID_SCENE_TYPES:
            accs = [self._accuracy.get((st, d), self.initial_value) for d in VALID_DIFFICULTIES]
            result[st] = sum(accs) / len(accs)
        return result

    def summary(self) -> str:
        """Human-readable summary of current curriculum state."""
        from adversary.adversary_config import VALID_SCENE_TYPES, VALID_DIFFICULTIES
        lines = []
        for st in VALID_SCENE_TYPES:
            parts = [f"{d}={self._accuracy.get((st, d), self.initial_value):.2f}" for d in VALID_DIFFICULTIES]
            count = sum(self._update_counts.get((st, d), 0) for d in VALID_DIFFICULTIES)
            lines.append(f"  {st:45s} [{', '.join(parts)}] (n={count})")
        return "\n".join(lines)

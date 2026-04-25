"""
Adversary Agent — proposes physics scene configurations to challenge the Solver.

Architecture:
  - Uses Qwen2.5-0.5B as the proposal LLM.
  - The model is prompted with the current curriculum state (per-scene accuracy)
    and must output a valid JSON scene config.
  - The adversary is trained via GRPO: it gets reward=1 when the Solver fails
    (and the scene was physically valid), reward=-1 for invalid scenes.
"""

from .adversary_agent import AdversaryAgent
from .adversary_config import ADVERSARY_ACTION_SCHEMA, SCENE_TYPE_TO_CATEGORY, AdversarySceneConfig

__all__ = [
    "AdversaryAgent",
    "AdversarySceneConfig",
    "ADVERSARY_ACTION_SCHEMA",
    "SCENE_TYPE_TO_CATEGORY",
]

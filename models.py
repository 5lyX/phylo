"""
Data models for the Phylo Physics Reasoning Environment.

Replaces the original echo environment models with physics-aware
Action/Observation types that support the adversarial self-play loop.

Two agent roles:
  - Solver: receives a physics problem, returns an answer.
  - Adversary: proposes scene configurations, receives difficulty feedback.
"""

from typing import Any, Dict, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ─────────────────────────────────────────────
# Solver Models
# ─────────────────────────────────────────────

class PhysicsAction(Action):
    """
    Action submitted by the Solver agent.
    The solver reads a physics problem and produces a numeric answer
    (optionally with chain-of-thought reasoning).
    """
    answer: str = Field(..., description="The solver's numeric answer, e.g. '3.14' or '\\\\boxed{3.14}'")
    reasoning: str = Field(default="", description="Chain-of-thought reasoning trace (optional)")


class PhysicsObservation(Observation):
    """
    Observation returned to the Solver agent.
    Contains the full physics problem text (natural language).
    The ground truth is NOT included here — it lives server-side.
    """
    problem_text: str = Field(default="", description="Full physics problem in natural language")
    scene_id: str = Field(default="", description="Unique scene identifier")
    scene_type: str = Field(default="", description="Scene category, e.g. 'BasicPulley'")
    difficulty: str = Field(default="EASY", description="Scene difficulty: EASY/MEDIUM/HARD")
    question_type: str = Field(default="numeric", description="QA type: numeric / symbolic / reverse")
    step_num: int = Field(default=0, description="Episode step count")


# ─────────────────────────────────────────────
# Adversary Models
# ─────────────────────────────────────────────

class AdversaryAction(Action):
    """
    Action submitted by the Adversary agent.
    The adversary proposes a scene configuration as a JSON-serializable dict.
    The server uses this to parameterize SceneGenerator.
    """
    scene_type: str = Field(
        ...,
        description="Scene type key from SCENE_CONFIGS, e.g. 'BasicPulley', 'SpringBlockSystems'"
    )
    difficulty: str = Field(
        default="EASY",
        description="Difficulty level: EASY / MEDIUM / HARD"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible scene generation"
    )
    question_type: str = Field(
        default="numeric",
        description="QA type to generate: numeric / symbolic / reverse"
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra scene parameters (reserved for future use)"
    )


class AdversaryObservation(Observation):
    """
    Observation returned to the Adversary agent after the Solver attempts the scene.
    Used to compute the adversary's reward and update its difficulty curriculum.
    """
    scene_id: str = Field(default="", description="Scene identifier")
    scene_type: str = Field(default="", description="Scene type that was proposed")
    difficulty: str = Field(default="EASY", description="Difficulty that was proposed")
    scene_valid: bool = Field(default=True, description="False if MuJoCo simulation crashed")
    solver_succeeded: bool = Field(default=False, description="True if solver answered within 5% tolerance")
    solver_answer: str = Field(default="", description="Solver's raw answer string")
    ground_truth: float = Field(default=0.0, description="Simulator ground truth value")
    adversary_reward: float = Field(default=0.0, description="Reward for the adversary")


# ─────────────────────────────────────────────
# Backward-compatible aliases (keep old names importable)
# ─────────────────────────────────────────────
PhyloAction = PhysicsAction
PhyloObservation = PhysicsObservation

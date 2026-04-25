"""
Adversary Configuration — valid scene types, difficulty levels, and JSON schema.

The adversary outputs a JSON matching ADVERSARY_ACTION_SCHEMA. The server
validates this JSON before passing it to SceneGenerator.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Valid scene types (from SCENE_CONFIGS in scene_generator.py) ──────────────
VALID_SCENE_TYPES: List[str] = [
    "BasicPulley",
    "IntermediatePulley",
    "BasicInclinedPlaneFriction",
    "IntermediateInclinedPlaneFriction",
    "AdvancedInclinedPlaneFriction",
    "IntermediateHybrid",
    "AdvancedHybrid",
    "BasicCollision",
    "IntermediateCollision",
    "AdvancedCollision",
    "Rotation",
    "RigidBodyRotation",
    "SpringBlockSystems",
    "DifficultPulley",
    "DifficultSpringMass",
    "DifficultOrbitalMotion",
    "DifficultRocket",
    "DifficultElectroMagnetic",
]

VALID_DIFFICULTIES: List[str] = ["EASY", "MEDIUM", "HARD"]
VALID_QUESTION_TYPES: List[str] = ["numeric", "symbolic", "reverse"]

# ── Category mapping (scene_type → physics category) ─────────────────────────
SCENE_TYPE_TO_CATEGORY: Dict[str, str] = {
    "BasicPulley": "pulley",
    "IntermediatePulley": "pulley",
    "BasicInclinedPlaneFriction": "pulley",
    "IntermediateInclinedPlaneFriction": "pulley",
    "AdvancedInclinedPlaneFriction": "pulley",
    "IntermediateHybrid": "pulley",
    "AdvancedHybrid": "pulley",
    "DifficultPulley": "pulley",
    "BasicCollision": "collision",
    "IntermediateCollision": "collision",
    "AdvancedCollision": "collision",
    "DifficultProjectile": "collision",
    "Rotation": "rotation",
    "RigidBodyRotation": "rotation",
    "SpringBlockSystems": "spring",
    "DifficultSpringMass": "spring",
    "DifficultOrbitalMotion": "orbital",
    "DifficultRocket": "orbital",
    "DifficultElectroMagnetic": "em",
}

# ── JSON Schema for adversary output ─────────────────────────────────────────
ADVERSARY_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_type": {
            "type": "string",
            "enum": VALID_SCENE_TYPES,
            "description": "Physics scene category to generate",
        },
        "difficulty": {
            "type": "string",
            "enum": VALID_DIFFICULTIES,
            "description": "Difficulty level for parameter randomization",
        },
        "seed": {
            "type": "integer",
            "description": "Random seed for reproducibility",
            "minimum": 0,
        },
        "question_type": {
            "type": "string",
            "enum": VALID_QUESTION_TYPES,
            "description": "Type of QA to generate",
        },
    },
    "required": ["scene_type", "difficulty"],
    "additionalProperties": False,
}


@dataclass
class AdversarySceneConfig:
    """Validated, typed scene configuration produced by the Adversary."""
    scene_type: str
    difficulty: str = "EASY"
    seed: int = 42
    question_type: str = "numeric"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "question_type": self.question_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdversarySceneConfig":
        """Parse and validate an adversary output dict."""
        scene_type = d.get("scene_type", "BasicPulley")
        if scene_type not in VALID_SCENE_TYPES:
            raise ValueError(f"Invalid scene_type: {scene_type!r}. Must be one of {VALID_SCENE_TYPES}")

        difficulty = d.get("difficulty", "EASY").upper()
        if difficulty not in VALID_DIFFICULTIES:
            difficulty = "EASY"

        question_type = d.get("question_type", "numeric")
        if question_type not in VALID_QUESTION_TYPES:
            question_type = "numeric"

        return cls(
            scene_type=scene_type,
            difficulty=difficulty,
            seed=int(d.get("seed", 42)),
            question_type=question_type,
        )

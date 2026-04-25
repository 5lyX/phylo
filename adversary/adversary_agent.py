"""
Adversary Agent — Qwen2.5-0.5B LLM that proposes physics scene configurations.

The adversary is an LLM that:
1. Receives a curriculum state (per-scene-type solver accuracy).
2. Outputs a JSON scene config.
3. Gets reward=1 if the solver fails on the generated scene (and scene was valid).
4. Gets reward=-1 if the scene is physically invalid (MuJoCo crash).
5. Gets reward=0 if the solver succeeds.

Training: The adversary is trained with GRPO (same as the solver) to maximize
solver failure rate while generating valid scenes. This is the asymmetric
self-play / L2D (Learning-to-Difficulty) objective.

The system prompt instructs the adversary to output ONLY valid JSON. Responses
are parsed and validated against AdversarySceneConfig before use.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adversary_config import (
    ADVERSARY_ACTION_SCHEMA,
    VALID_SCENE_TYPES,
    VALID_DIFFICULTIES,
    AdversarySceneConfig,
)

logger = logging.getLogger(__name__)

# ── Default adversary model ───────────────────────────────────────────────────
DEFAULT_ADVERSARY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ── System prompt for the adversary ──────────────────────────────────────────
ADVERSARY_SYSTEM_PROMPT = """You are an adversarial physics problem generator. Your goal is to propose physics simulation scenarios that will CHALLENGE the solver — problems the solver is likely to get WRONG.

You will be given the solver's current performance statistics (accuracy per scene type). Choose a scene type and difficulty where the solver is WEAKEST.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{schema}

Valid scene_type values: {scene_types}
Valid difficulty values: EASY, MEDIUM, HARD
Valid question_type values: numeric, symbolic, reverse

Rules:
- Choose the scene type where solver accuracy is LOWEST (to maximize difficulty)
- Prefer HARD difficulty for scene types where solver is already decent
- NEVER include any text outside the JSON object
- The seed should be a random integer between 0 and 999999

Respond ONLY with JSON. No explanations, no markdown, just the JSON object.
""".format(
    schema=json.dumps(ADVERSARY_ACTION_SCHEMA, indent=2),
    scene_types=VALID_SCENE_TYPES,
)

ADVERSARY_USER_TEMPLATE = """Current solver performance (accuracy per scene type, lower = harder for solver):
{performance_summary}

Propose a physics scene that the solver will likely FAIL on. Respond with JSON only."""


class AdversaryAgent:
    """
    LLM-based adversary that proposes physics scene configurations.

    Maintains an EMA (Exponential Moving Average) accuracy tracker per
    (scene_type, difficulty) pair to guide the curriculum.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ADVERSARY_MODEL,
        device: str = "auto",
        ema_alpha: float = 0.1,
        target_success_rate: float = 0.5,
        load_in_4bit: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model ID for the adversary LLM.
            device: torch device ('auto', 'cuda', 'cpu').
            ema_alpha: EMA smoothing factor for accuracy tracking.
            target_success_rate: Target solver success rate (0.5 = frontier).
            load_in_4bit: Load model in 4-bit quantization (saves VRAM).
        """
        self.model_name = model_name
        self.ema_alpha = ema_alpha
        self.target_success_rate = target_success_rate

        # Per-(scene_type, difficulty) EMA accuracy tracker
        # Initialized to 0.5 (unknown) for all scene types
        self._accuracy: Dict[Tuple[str, str], float] = {
            (st, diff): 0.5
            for st in VALID_SCENE_TYPES
            for diff in VALID_DIFFICULTIES
        }

        # History for training
        self._proposal_history: List[dict] = []

        # Load model lazily
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device = device
        self._load_in_4bit = load_in_4bit

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the adversary LLM."""
        if self._model is not None:
            return

        logger.info("Loading adversary model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self._device,
        }

        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        logger.info("Adversary model loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────────

    def propose(self, use_llm: bool = True) -> AdversarySceneConfig:
        """
        Propose a scene configuration.

        If use_llm=True, the LLM generates the config conditioned on the
        current accuracy curriculum. Falls back to heuristic if LLM fails.

        Returns:
            AdversarySceneConfig validated and ready for SceneGenerator.
        """
        if use_llm:
            config = self._propose_via_llm()
            if config is not None:
                return config

        # Fallback: heuristic curriculum sampling
        return self._propose_heuristic()

    def update(
        self,
        scene_type: str,
        difficulty: str,
        solver_succeeded: bool,
    ) -> None:
        """
        Update EMA accuracy tracker after a Solver attempt.

        Args:
            scene_type: The scene type that was proposed.
            difficulty: The difficulty level used.
            solver_succeeded: Whether the solver got it right (within 5%).
        """
        key = (scene_type, difficulty.upper())
        if key not in self._accuracy:
            self._accuracy[key] = 0.5

        old = self._accuracy[key]
        self._accuracy[key] = (1 - self.ema_alpha) * old + self.ema_alpha * float(solver_succeeded)

        logger.debug(
            "Adversary EMA update: %s/%s → %.3f (was %.3f, succeeded=%s)",
            scene_type, difficulty, self._accuracy[key], old, solver_succeeded,
        )

    def get_reward(self, scene_valid: bool, solver_succeeded: bool) -> float:
        """
        Compute adversary reward.

        Reward structure:
          - scene_valid=False: -1.0 (heavy penalty for broken scenes)
          - scene_valid=True, solver_succeeded=True: 0.0 (problem was too easy)
          - scene_valid=True, solver_succeeded=False: +1.0 (Solver stumped!)

        Args:
            scene_valid: Did MuJoCo successfully simulate the scene?
            solver_succeeded: Did the solver answer within 5% tolerance?

        Returns:
            Float reward for the adversary.
        """
        if not scene_valid:
            return -1.0
        if solver_succeeded:
            return 0.0
        return 1.0

    def get_performance_summary(self) -> str:
        """
        Format the current accuracy tracker as a readable summary.
        Used in the adversary's prompt.
        """
        lines = []
        for scene_type in VALID_SCENE_TYPES:
            accs = []
            for diff in VALID_DIFFICULTIES:
                acc = self._accuracy.get((scene_type, diff), 0.5)
                accs.append(f"{diff}={acc:.2f}")
            lines.append(f"  {scene_type}: {', '.join(accs)}")
        return "\n".join(lines)

    def get_accuracy_map(self) -> Dict[str, float]:
        """Return average accuracy per scene type (across difficulties)."""
        result = {}
        for scene_type in VALID_SCENE_TYPES:
            accs = [self._accuracy.get((scene_type, d), 0.5) for d in VALID_DIFFICULTIES]
            result[scene_type] = sum(accs) / len(accs)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _propose_via_llm(self) -> Optional[AdversarySceneConfig]:
        """Use the LLM to propose a scene config. Returns None on parse failure."""
        self._load_model()

        performance_summary = self.get_performance_summary()
        user_message = ADVERSARY_USER_TEMPLATE.format(
            performance_summary=performance_summary
        )

        messages = [
            {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_json_response(generated)

    def _parse_json_response(self, text: str) -> Optional[AdversarySceneConfig]:
        """Extract and validate JSON from the LLM response."""
        # Try to find JSON in the text
        json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not json_match:
            logger.warning("Adversary LLM: no JSON found in response: %r", text[:200])
            return None

        try:
            raw = json.loads(json_match.group(0))
            return AdversarySceneConfig.from_dict(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Adversary LLM: JSON parse failed: %s", e)
            return None

    def _propose_heuristic(self) -> AdversarySceneConfig:
        """
        Heuristic fallback: sample scene type weighted by (1 - accuracy).
        Scene types where the solver struggles get higher weight.
        """
        import random

        acc_map = self.get_accuracy_map()
        # Weight inversely proportional to accuracy
        weights = [max(1.0 - acc, 0.01) for acc in acc_map.values()]
        scene_type = random.choices(list(acc_map.keys()), weights=weights, k=1)[0]

        # Choose difficulty based on accuracy for this scene type
        avg_acc = acc_map[scene_type]
        if avg_acc < 0.3:
            difficulty = "HARD"
        elif avg_acc < 0.6:
            difficulty = "MEDIUM"
        else:
            difficulty = "EASY"

        import random as _r
        seed = _r.randint(0, 999999)

        # Allow some probability of picking reverse
        question_type = _r.choices(["numeric", "reverse"], weights=[0.8, 0.2])[0]

        return AdversarySceneConfig(
            scene_type=scene_type,
            difficulty=difficulty,
            seed=seed,
            question_type=question_type,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Serialization for training data collection
    # ──────────────────────────────────────────────────────────────────────────

    def record_proposal(
        self,
        config: AdversarySceneConfig,
        scene_valid: bool,
        solver_succeeded: bool,
        reward: float,
    ) -> None:
        """Store a proposal + outcome for GRPO training data collection."""
        self._proposal_history.append({
            "scene_type": config.scene_type,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "question_type": config.question_type,
            "scene_valid": scene_valid,
            "solver_succeeded": solver_succeeded,
            "reward": reward,
        })

    def get_training_samples(self, clear: bool = True) -> List[dict]:
        """Get collected proposal history for GRPO training."""
        samples = list(self._proposal_history)
        if clear:
            self._proposal_history = []
        return samples

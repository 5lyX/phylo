"""
Solver Agent — Qwen2.5-3B reasoning model for physics problems.

Used during the self-play loop to:
1. Generate responses to physics problems (inference).
2. Provide data for GRPO training (via training/train_solver.py).

For training, use training/train_solver.py which wraps GRPOTrainer.
This class is for inference-time rollouts during self-play.
"""

import logging
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reward import extract_boxed_answer

logger = logging.getLogger(__name__)

DEFAULT_SOLVER_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SOLVER_SYSTEM_PROMPT = """You are a physics expert. Solve the given physics problem step-by-step.

Rules:
1. Read the problem carefully and identify all given quantities.
2. Write out the relevant physics equations.
3. Solve systematically, showing each step.
4. Always end with your final numeric answer in the format: \\boxed{your_answer}
5. Include appropriate units in your reasoning but give only the numeric value in the box.

Example format:
Given: mass m = 5 kg, force F = 10 N
Using Newton's second law: F = ma → a = F/m = 10/5 = 2 m/s²
\\boxed{2}"""


class SolverAgent:
    """
    Inference wrapper for the Qwen2.5-3B solver.

    Used during self-play rollouts to generate answers for physics problems.
    The model weights are updated externally via GRPOTrainer (train_solver.py).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SOLVER_MODEL,
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self._device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load_in_4bit = load_in_4bit

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _load_model(self):
        if self._model is not None:
            return

        logger.info("Loading solver model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "device_map": self._device,
        }

        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        logger.info("Solver model loaded.")

    def answer(self, problem_text: str) -> str:
        """
        Generate an answer for a physics problem.

        Args:
            problem_text: Natural language physics problem.

        Returns:
            Full completion string including chain-of-thought and boxed answer.
        """
        self._load_model()

        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
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
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        completion = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return completion

    def answer_batch(self, problems: List[str]) -> List[str]:
        """Answer a batch of physics problems (sequential for now)."""
        return [self.answer(p) for p in problems]

    def extract_answer(self, completion: str) -> Optional[float]:
        """Extract the numeric answer from a completion."""
        return extract_boxed_answer(completion)

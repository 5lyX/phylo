"""
Adversary Training — GRPO training for Qwen2.5-0.5B scene proposer.

The adversary learns to generate scene configs (JSON) that maximize
the probability of stumping the Solver.

Reward signal:
  +1.0 → Solver failed (adversary succeeded)
   0.0 → Solver succeeded (problem was too easy)
  -1.0 → Scene was physically invalid (MuJoCo crashed)

Usage:
    python training/train_adversary.py --replay_path replay_buffer/replay_iter_00100.parquet
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

import json
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from adversary.adversary_config import ADVERSARY_ACTION_SCHEMA, VALID_SCENE_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AdversaryTrainingConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "./adversary_grpo_output"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_generations: int = 4
    logging_steps: int = 1
    log_file: str = "adversary_training_log.csv"
    disable_tracking: bool = False
    replay_path: Optional[str] = None


ADVERSARY_SYSTEM_PROMPT = (
    "You are an adversarial physics problem generator. "
    "Output ONLY a valid JSON object with keys: scene_type, difficulty, seed, question_type. "
    f"scene_type must be one of: {VALID_SCENE_TYPES}. "
    "difficulty must be EASY, MEDIUM, or HARD. "
    "No explanations, just the JSON."
)


def _adversary_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Reward for adversary GRPO:
    ground_truth column contains the adversary's reward (str: -1.0, 0.0, or 1.0).
    We also add a format bonus for producing valid JSON.
    """
    import re
    rewards = []
    for completion, gt_str in zip(completions, ground_truth):
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)

        # Base reward from ground truth
        try:
            base_reward = float(gt_str)
        except (TypeError, ValueError):
            base_reward = 0.0

        # Format bonus: +0.3 if valid JSON with correct scene_type
        format_bonus = 0.0
        json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if parsed.get("scene_type") in VALID_SCENE_TYPES:
                    format_bonus = 0.3
            except json.JSONDecodeError:
                pass

        rewards.append(base_reward + format_bonus)

    return rewards


class CSVLoggingCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_data: List[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step}
            for k, v in logs.items():
                if any(kw in k.lower() for kw in ["reward", "loss", "lr"]):
                    entry[k] = v
            self.log_data.append(entry)
            pd.DataFrame(self.log_data).to_csv(self.log_path, index=False)


def _build_adversary_dataset(path: str) -> Optional[Dataset]:
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if "adversary_reward" not in df.columns:
        return None

    # Compute performance stats for the prompt
    type_stats = df.groupby("scene_type")["solver_reward"].mean().to_dict()
    perf_lines = "\n".join(f"  {st}: solver_accuracy={acc:.2f}" for st, acc in type_stats.items())

    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt": [
                {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Solver performance:\n{perf_lines}\n\nPropose a challenging scene (JSON only):"},
            ],
            "ground_truth": str(row.get("adversary_reward", 0.0)),
        })

    logger.info("Adversary dataset: %d samples", len(records))
    return Dataset.from_list(records)


def _build_dummy_adversary_dataset() -> Dataset:
    """Dummy dataset for smoke testing."""
    records = []
    import random
    for _ in range(50):
        st = random.choice(VALID_SCENE_TYPES[:5])
        records.append({
            "prompt": [
                {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
                {"role": "user", "content": "Propose a challenging physics scene (JSON only):"},
            ],
            "ground_truth": str(random.choice([-1.0, 0.0, 1.0])),
        })
    return Dataset.from_list(records)


def train_adversary(cfg: Optional[AdversaryTrainingConfig] = None) -> None:
    if cfg is None:
        cfg = AdversaryTrainingConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("Starting Adversary GRPO training: %s", cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if cfg.replay_path:
        dataset = _build_adversary_dataset(cfg.replay_path)
        if dataset is None:
            dataset = _build_dummy_adversary_dataset()
    else:
        dataset = _build_dummy_adversary_dataset()

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    log_path = os.path.join(cfg.output_dir, cfg.log_file)
    
    import time
    from omegaconf import OmegaConf
    from utils.tracking import Tracking, TrackingCallback
    
    default_backend = [] if cfg.disable_tracking else ["console", "tensorboard"]
    
    tracker = Tracking(
        project_name="Sim2Reason-Phylo",
        experiment_name=f"adversary_train_{int(time.time())}",
        default_backend=default_backend,
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True) if not isinstance(cfg, dict) else cfg,
    )
    tracking_callback = TrackingCallback(tracker=tracker, output_dir=cfg.output_dir)
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_adversary_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[CSVLoggingCallback(log_path=log_path), tracking_callback],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Adversary model saved to %s", cfg.output_dir)
    
    tracker.finish()


def main():
    parser = argparse.ArgumentParser(description="GRPO Adversary Training")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="./adversary_grpo_output")
    parser.add_argument("--replay_path", default=None)
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--disable_tracking", action="store_true", help="Turn off tensorboard/wandb tracking")
    args = parser.parse_args()

    cfg = AdversaryTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        replay_path=None if args.dummy_data else args.replay_path,
        max_steps=args.max_steps,
        disable_tracking=args.disable_tracking,
    )
    train_adversary(cfg)


if __name__ == "__main__":
    main()

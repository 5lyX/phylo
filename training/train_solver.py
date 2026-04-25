"""
Solver Training — GRPO training for Qwen2.5-3B on physics problems.

This script is the production training pipeline for the Solver agent.
It extends Sim2Reason's train_trl_sample.py with:

1. Real physics reward (5% tolerance, from solver/reward.py)
2. Format reward (\\boxed answer encouragement)
3. Loading from replay buffer / parquet dataset
4. LoRA fine-tuning via PEFT
5. CSV logging compatible with plot_metrics.py
6. HuggingFace Hub model pushing

Designed to run on a single T4 GPU (Google Colab / HuggingFace Spaces).

Usage:
    python training/train_solver.py --replay_path replay_buffer/replay_iter_00100.parquet
    python training/train_solver.py --dummy_data  # quick smoke test
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from solver.reward import format_reward, physics_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverTrainingConfig:
    """Configuration for GRPO Solver training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "./solver_grpo_output"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 1
    max_steps: int = -1             # -1 = use epochs
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 1024

    # GRPO specific
    num_generations: int = 8        # G rollouts per prompt

    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    log_file: str = "solver_training_log.csv"
    disable_tracking: bool = False  # If True, disable WandB/TensorBoard tracking

    # Data
    replay_path: Optional[str] = None   # path to parquet replay buffer
    min_samples: int = 10

    # HF Hub push
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# CSV Callback (reused from train_trl_sample.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

class CSVLoggingCallback(TrainerCallback):
    """Saves training metrics to CSV after each logging step."""

    def __init__(self, log_path: str = "training_log.csv"):
        self.log_path = log_path
        self.log_data: List[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step}
        for k, v in logs.items():
            if any(kw in k.lower() for kw in ["reward", "loss", "lr", "epoch"]):
                entry[k] = v
        self.log_data.append(entry)
        pd.DataFrame(self.log_data).to_csv(self.log_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

SOLVER_SYSTEM_PROMPT = (
    "You are a physics expert. Solve the problem step-by-step and end with "
    r"\boxed{your_numeric_answer}."
)


def _build_dummy_dataset() -> Dataset:
    """Small dummy dataset for smoke testing."""
    problems = [
        ("A block of mass 5 kg is on a frictionless surface. A force of 20 N is applied. What is the acceleration?", "4.0"),
        ("A spring with k=200 N/m is compressed by 0.1 m. What is the elastic potential energy?", "1.0"),
        ("A 2 kg ball is dropped from 10 m. What is its velocity just before hitting the ground? g=9.81 m/s²", "14.0"),
        ("Two masses of 3 kg and 5 kg are connected by a string over a frictionless pulley. What is the acceleration?", "2.45"),
    ] * 25  # 100 samples

    records = []
    for problem, gt in problems:
        records.append({
            "prompt": [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ground_truth": gt,
        })
    return Dataset.from_list(records)


def _build_dataset_from_replay(path: str) -> Optional[Dataset]:
    """Load dataset from a replay buffer parquet file."""
    if not os.path.exists(path):
        logger.error("Replay path does not exist: %s", path)
        return None

    df = pd.read_parquet(path)
    # Filter valid scenes with real problem text
    df = df[df["scene_valid"] == True]
    df = df[df["problem_text"] != "[SIMULATION FAILED]"]
    df = df[df["problem_text"].str.len() > 10]

    if len(df) == 0:
        logger.error("No valid samples in replay buffer: %s", path)
        return None

    logger.info("Loaded %d samples from replay buffer", len(df))

    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt": [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": row["problem_text"]},
            ],
            "ground_truth": str(row["ground_truth"]),
        })

    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────

def physics_reward_wrapper(completions, prompts, ground_truth, **kwargs):
    """
    TRL GRPOTrainer reward wrapper.
    ground_truth comes from the dataset column of the same name.
    """
    return physics_reward(completions, ground_truth)


def format_reward_wrapper(completions, prompts, **kwargs):
    return format_reward(completions)

def dynamic_sampling_filter(dataset: Dataset) -> Dataset:
    """
    Spec §6: Dynamic sampling — filter low-variance prompts.
    Provides a lightweight heuristic filter to remove trivial (0 ground truth)
    problems prior to GRPO training to ensure gradient efficiency.
    """
    logger.info("Running dynamic sampling pre-filter on dataset of size %d", len(dataset))
    kept_indices = []
    for i, item in enumerate(dataset):
        try:
            gt = float(item.get("ground_truth", 0.0))
        except (ValueError, TypeError):
            gt = 0.0
            
        if abs(gt) > 1e-6:
            kept_indices.append(i)
            
    if len(kept_indices) == 0:
        logger.warning("Dynamic sampling removed all prompts — bypassing filter")
        return dataset
        
    filtered = dataset.select(kept_indices)
    logger.info("Dynamic sampling kept %d/%d prompts", len(filtered), len(dataset))
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_solver(cfg: Optional[SolverTrainingConfig] = None) -> None:
    """
    Run GRPO training on the Solver model.

    Args:
        cfg: SolverTrainingConfig. If None, uses defaults.
    """
    if cfg is None:
        cfg = SolverTrainingConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info("Starting Solver GRPO training with model: %s", cfg.model_name)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model (bfloat16)…")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    logger.info("Applying LoRA (r=%d)…", cfg.lora_r)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        # Target key attention + MLP layers
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if cfg.replay_path:
        dataset = _build_dataset_from_replay(cfg.replay_path)
        if dataset is None:
            logger.warning("Could not load replay dataset, falling back to dummy data")
            dataset = _build_dummy_dataset()
    else:
        logger.info("No replay path specified — using dummy dataset")
        dataset = _build_dummy_dataset()

    logger.info("Dataset size: %d samples", len(dataset))
    dataset = dynamic_sampling_filter(dataset)

    # ── GRPO Config ───────────────────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        # ── Spec §6: KL regularization (reference model penalty) ──────────────
        # Prevents policy from deviating too far from the reference model.
        # TRL uses beta as the KL penalty coefficient in the GRPO objective:
        # L = -mean(min(ρ*A, clip(ρ)*A)) - beta * KL(π || π_ref)
        beta=0.04,
        # ── Optimization ──────────────────────────────────────────────────────
        warmup_ratio=0.05,
        weight_decay=0.01,
        # Mixed precision
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
        experiment_name=f"solver_train_{int(time.time())}",
        default_backend=default_backend,  # Enable wandb if desired
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True) if not isinstance(cfg, dict) else cfg,
    )
    tracking_callback = TrackingCallback(tracker=tracker, output_dir=cfg.output_dir)

    # ── Trainer ───────────────────────────────────────────────────────────────
    logger.info("Initializing GRPOTrainer…")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[physics_reward_wrapper, format_reward_wrapper],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[CSVLoggingCallback(log_path=log_path), tracking_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training…")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Model saved to %s", cfg.output_dir)
    
    tracker.finish()

    if cfg.push_to_hub and cfg.hub_model_id:
        logger.info("Pushing to HuggingFace Hub: %s", cfg.hub_model_id)
        trainer.push_to_hub(cfg.hub_model_id)

    logger.info("Training complete. Metrics saved to %s", log_path)
    logger.info("Run `python plot_metrics.py` to visualize results.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Solver Training (Sim2Reason + Self-Play)")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", default="./solver_grpo_output")
    parser.add_argument("--replay_path", default=None, help="Path to replay buffer parquet")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy dataset (smoke test)")
    parser.add_argument("--disable_tracking", action="store_true", help="Turn off tensorboard/wandb tracking")
    args = parser.parse_args()

    cfg = SolverTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        replay_path=None if args.dummy_data else args.replay_path,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        lora_r=args.lora_r,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        disable_tracking=args.disable_tracking,
    )

    train_solver(cfg)


if __name__ == "__main__":
    main()

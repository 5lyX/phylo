#!/usr/bin/env python3
"""
Sim2Reason Adversarial Self-Play — HuggingFace Training Notebook
================================================================

This script is designed to run on HuggingFace Spaces / Google Colab with a T4 GPU.
It runs the complete training flywheel:

  1. Run self-play loop to collect physics QA pairs into replay buffer
  2. Train Solver (Qwen2.5-3B) with GRPO on replay buffer
  3. Train Adversary (Qwen2.5-0.5B) with GRPO on replay buffer
  4. Evaluate Solver on held-out scenes
  5. Repeat

Usage on HuggingFace (T4, 16GB VRAM):
    python scripts/hf_training_notebook.py --iterations 500 --batch_size 4

Or adapt as a Jupyter notebook cells.
"""

import os
import sys
import logging

# ── Environment setup for headless MuJoCo on Linux T4 ────────────────────────
os.environ["MUJOCO_GL"] = "egl"        # Headless OpenGL on Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PHYLO_ROOT)
# Also add Sim2Reason root for direct imports if needed
_SIM2REASON_ROOT = os.path.join(os.path.dirname(_PHYLO_ROOT), "Sim2Reason")
if os.path.exists(_SIM2REASON_ROOT):
    sys.path.insert(0, _SIM2REASON_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


def cell_1_install_deps():
    """Cell 1: Install dependencies."""
    import subprocess
    packages = [
        "mujoco>=3.1.0",
        "trl>=0.9.0",
        "peft>=0.11.0",
        "transformers>=4.42.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.17.0",
        "openenv-core>=0.2.2",
        "ipdb>=0.13.13",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
    print("✅ Dependencies installed")


def cell_2_self_play(num_iterations=200, batch_size=4, dry_run=False):
    """Cell 2: Run self-play loop to collect replay buffer data."""
    from self_play.self_play_loop import SelfPlayConfig, run_self_play_loop

    cfg = SelfPlayConfig(
        adversary_model="Qwen/Qwen2.5-0.5B-Instruct",
        solver_model="Qwen/Qwen2.5-3B-Instruct",
        num_iterations=num_iterations,
        batch_size=batch_size,
        solver_update_interval=999999,   # Only collect data, don't train yet
        adversary_update_interval=999999,
        output_dir="./self_play_output",
        replay_dir="./replay_buffer",
        save_interval=50,
        log_interval=5,
        dry_run=dry_run,
        adversary_use_llm=not dry_run,
    )

    logger.info("Starting self-play data collection (%d iterations, batch=%d)…", num_iterations, batch_size)
    replay = run_self_play_loop(cfg)

    if replay:
        stats = replay.stats()
        logger.info("Replay buffer stats: %s", stats)
        replay.save(num_iterations)

    return replay


def cell_3_train_solver(replay_path=None, max_steps=200, push_to_hub=False, hub_model_id=None):
    """Cell 3: Train the Solver with GRPO on replay buffer data."""
    from training.train_solver import SolverTrainingConfig, train_solver

    cfg = SolverTrainingConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir="./solver_grpo_output",
        replay_path=replay_path,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        lora_r=16,
        learning_rate=5e-6,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        log_file="solver_training_log.csv",
    )

    logger.info("Starting Solver GRPO training (max_steps=%d)…", max_steps)
    train_solver(cfg)
    logger.info("✅ Solver training complete")


def cell_4_train_adversary(replay_path=None, max_steps=100):
    """Cell 4: Train the Adversary with GRPO on replay buffer data."""
    from training.train_adversary import AdversaryTrainingConfig, train_adversary

    cfg = AdversaryTrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="./adversary_grpo_output",
        replay_path=replay_path,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        log_file="adversary_training_log.csv",
    )

    logger.info("Starting Adversary GRPO training (max_steps=%d)…", max_steps)
    train_adversary(cfg)
    logger.info("✅ Adversary training complete")


def cell_5_evaluate(solver_checkpoint=None):
    """Cell 5: Evaluate the trained Solver on held-out scenes."""
    from evaluation.eval_runner import EvalConfig, EvalRunner

    cfg = EvalConfig(
        solver_model="Qwen/Qwen2.5-3B-Instruct",
        solver_checkpoint=solver_checkpoint,
        output_dir="./eval_output",
        num_problems_per_type=5,    # Keep small for quick eval
        scene_types=[
            "BasicPulley", "IntermediatePulley",
            "BasicCollision", "SpringBlockSystems", "Rotation",
        ],
        difficulties=["EASY", "MEDIUM", "HARD"],
        load_in_4bit=True,
    )

    logger.info("Running evaluation…")
    runner = EvalRunner(cfg)
    df = runner.run()
    logger.info("✅ Evaluation complete. Overall accuracy: %.3f", df["reward"].mean())
    return df


def cell_6_plot_metrics():
    """Cell 6: Plot training metrics (reuses Sim2Reason's plot_metrics.py pattern)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sim2Reason Adversarial Self-Play — Training Metrics", fontsize=14)

    # Self-play metrics
    sp_logs = glob.glob("self_play_output/self_play_log.csv")
    if sp_logs:
        df = pd.read_csv(sp_logs[0])
        axes[0].plot(df["iteration"], df["solver_reward_mean"], label="Solver Reward", color="#4CAF50")
        axes[0].plot(df["iteration"], df["adversary_reward_mean"], label="Adversary Reward", color="#F44336")
        axes[0].set_title("Self-Play Rewards")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Solver training loss
    solver_logs = glob.glob("solver_grpo_output/solver_training_log.csv")
    if solver_logs:
        df = pd.read_csv(solver_logs[0])
        reward_cols = [c for c in df.columns if "reward" in c.lower()]
        for col in reward_cols[:2]:
            axes[1].plot(df["step"], df[col], label=col)
        axes[1].set_title("Solver GRPO Training")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    # Eval results
    eval_logs = glob.glob("eval_output/eval_results.csv")
    if eval_logs:
        df = pd.read_csv(eval_logs[0])
        type_acc = df.groupby("scene_type")["reward"].mean().sort_values()
        axes[2].barh(type_acc.index, type_acc.values, color="#2196F3")
        axes[2].set_title("Eval Accuracy by Scene Type")
        axes[2].set_xlabel("Accuracy")
        axes[2].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("✅ Metrics plot saved to training_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    num_self_play_iterations: int = 200,
    batch_size: int = 4,
    solver_max_steps: int = 200,
    adversary_max_steps: int = 100,
    num_cycles: int = 3,
    push_to_hub: bool = False,
    hub_model_id: str = None,
    dry_run: bool = False,
):
    """
    Run the complete adversarial self-play training pipeline for N cycles.

    Each cycle:
      1. Self-play data collection
      2. Solver GRPO training
      3. Adversary GRPO training
      4. Evaluation

    Args:
        num_self_play_iterations: Self-play iterations per cycle.
        batch_size: Scenes per self-play iteration.
        solver_max_steps: GRPO training steps for solver per cycle.
        adversary_max_steps: GRPO training steps for adversary per cycle.
        num_cycles: Number of full cycles to run.
        push_to_hub: Push trained solver to HuggingFace Hub.
        hub_model_id: HuggingFace Hub repo ID (e.g. "username/sim2reason-solver-3b").
        dry_run: Skip actual LLM inference (for testing).
    """
    import glob

    for cycle in range(num_cycles):
        logger.info("=" * 70)
        logger.info("CYCLE %d / %d", cycle + 1, num_cycles)
        logger.info("=" * 70)

        # Determine if solver checkpoint from previous cycle exists
        solver_ckpt = "./solver_grpo_output" if cycle > 0 and os.path.exists("./solver_grpo_output") else None

        # Step 1: Self-play
        replay = cell_2_self_play(
            num_iterations=num_self_play_iterations,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        # Find latest replay buffer parquet
        replay_files = sorted(glob.glob("./replay_buffer/replay_iter_*.parquet"))
        replay_path = replay_files[-1] if replay_files else None

        if replay_path is None:
            logger.warning("No replay buffer found, skipping training for cycle %d", cycle)
            continue

        logger.info("Using replay buffer: %s (%d entries)", replay_path, len(replay) if replay else 0)

        # Step 2: Solver training
        cell_3_train_solver(
            replay_path=replay_path,
            max_steps=solver_max_steps,
            push_to_hub=push_to_hub and cycle == num_cycles - 1,  # Push only on last cycle
            hub_model_id=hub_model_id,
        )

        # Step 3: Adversary training
        cell_4_train_adversary(
            replay_path=replay_path,
            max_steps=adversary_max_steps,
        )

        # Step 4: Evaluation
        cell_5_evaluate(solver_checkpoint="./solver_grpo_output")

    # Final metrics plot
    cell_6_plot_metrics()
    logger.info("🎉 Full pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sim2Reason Adversarial Self-Play Pipeline")
    parser.add_argument("--iterations", type=int, default=200, help="Self-play iterations per cycle")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--solver_max_steps", type=int, default=200)
    parser.add_argument("--adversary_max_steps", type=int, default=100)
    parser.add_argument("--num_cycles", type=int, default=3)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument("--dry_run", action="store_true", help="Test pipeline without real LLM inference")
    args = parser.parse_args()

    run_full_pipeline(
        num_self_play_iterations=args.iterations,
        batch_size=args.batch_size,
        solver_max_steps=args.solver_max_steps,
        adversary_max_steps=args.adversary_max_steps,
        num_cycles=args.num_cycles,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dry_run=args.dry_run,
    )

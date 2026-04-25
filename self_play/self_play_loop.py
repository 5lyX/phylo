"""
Self-Play Loop — the adversarial flywheel orchestrator.

Implements the Automated Curriculum Learning loop:

  For each iteration:
    1. Adversary proposes N scene configs (JSON via Qwen2.5-0.5B)
    2. PhyloEnvironment simulates them → N QA pairs (invalid → penalize adversary)
    3. Solver answers the valid QA pairs (Qwen2.5-3B inference)
    4. Rewards computed:
       - Solver: 1.0 if |pred-gt|/|gt| ≤ 5%
       - Adversary: 1.0 if solver failed & scene valid, -1.0 if scene invalid
    5. Replay buffer updated
    6. Every solver_update_interval steps: run GRPO on solver
    7. Every adversary_update_interval steps: run GRPO on adversary

The loop writes all metrics to a CSV log for use with plot_metrics.py.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

import pandas as pd
from omegaconf import OmegaConf

from adversary.adversary_agent import AdversaryAgent
from adversary.adversary_config import AdversarySceneConfig
from self_play.replay_buffer import ReplayBuffer, ReplayEntry
from server.phylo_environment import PhyloEnvironment, _DEFAULT_MAIN_CFG, _DEFAULT_RECORDER_CFG, _get_category_for_scene_type, _get_recorder_cfg_for_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for the self-play loop."""
    # Adversary
    adversary_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adversary_use_llm: bool = False

    # Solver (inference only during self-play; training happens in train_solver.py)
    solver_model: str = "Qwen/Qwen2.5-3B-Instruct"
    solver_checkpoint: Optional[str] = None   # path to fine-tuned solver

    # Loop parameters
    num_iterations: int = 1000
    batch_size: int = 8               # scenes proposed per iteration
    solver_update_interval: int = 50  # run GRPO every N iterations
    adversary_update_interval: int = 100
    replay_buffer_capacity: int = 50_000
    save_interval: int = 100          # save replay buffer every N iterations
    save_replay_at_end_only: bool = False  # If True, skip periodic saves and save only once at end
    log_interval: int = 10            # log metrics every N iterations

    # Directories
    output_dir: str = "self_play_output"
    replay_dir: str = "replay_buffer"
    log_file: str = "self_play_log.csv"

    # Misc
    dry_run: bool = False             # If True, skip actual model inference
    seed: int = 42
    disable_tracking: bool = False    # If True, disable WandB/TensorBoard tracking

    @classmethod
    def from_yaml(cls, path: str) -> "SelfPlayConfig":
        cfg = OmegaConf.load(path)
        return cls(**OmegaConf.to_container(cfg, resolve=True))


def _generate_physics_qa(
    config: AdversarySceneConfig,
    episode_seed: int,
) -> Optional[dict]:
    """
    Run Sim2Reason pipeline for a given scene config.

    Returns:
        dict with keys: text, answer, simulation_mapping, scene_id
        None if simulation failed.
    """
    from server.phylo_environment import PhyloEnvironment

    env = PhyloEnvironment(
        scene_type=config.scene_type,
        difficulty=config.difficulty,
        seed=config.seed + episode_seed,
        question_type=config.question_type,
    )

    obs = env.reset()
    if obs.metadata.get("error", False):
        return None

    # The environment stores the full raw generated problem internally, 
    # we can just extract it to return the dict expected by the loop
    return env._current_problem


def run_self_play_loop(cfg: Optional[SelfPlayConfig] = None) -> None:
    """
    Main self-play flywheel.

    Args:
        cfg: SelfPlayConfig. If None, uses defaults.
    """
    if cfg is None:
        cfg = SelfPlayConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Initialize agents ──────────────────────────────────────────────────────
    logger.info("Initializing Adversary Agent (%s)…", cfg.adversary_model)
    adversary = AdversaryAgent(
        model_name=cfg.adversary_model,
        load_in_4bit=True,
    )

    logger.info("Initializing Solver Agent (%s)…", cfg.solver_model)
    if not cfg.dry_run:
        from solver.solver_agent import SolverAgent
        solver = SolverAgent(
            model_name=cfg.solver_checkpoint or cfg.solver_model,
            load_in_4bit=True,
        )
    else:
        solver = None

    # ── Replay buffer ──────────────────────────────────────────────────────────
    replay = ReplayBuffer(capacity=cfg.replay_buffer_capacity, save_dir=cfg.replay_dir)

    # ── Metrics log & Tracking ────────────────────────────────────────────────
    metrics_rows = []
    log_path = os.path.join(cfg.output_dir, cfg.log_file)
    
    from utils.tracking import Tracking
    
    # ── Handle Tracking Flag ──────────────────────────────────────────────────
    default_backend = [] if cfg.disable_tracking else ["console", "tensorboard"]
    
    tracker = Tracking(
        project_name="Sim2Reason-Phylo",
        experiment_name=f"self_play_{int(time.time())}",
        default_backend=default_backend,  # Enable wandb if desired
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True) if not isinstance(cfg, dict) else cfg,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    global_step = 0

    for iteration in range(cfg.num_iterations):
        iter_start = time.time()
        iter_metrics = {
            "iteration": iteration,
            "global_step": global_step,
            "valid_scenes": 0,
            "solver_reward_mean": 0.0,
            "adversary_reward_mean": 0.0,
            "batch_size": cfg.batch_size,
        }

        # ── Step 1: Adversary proposes scene configs ───────────────────────────
        proposals = []
        for _ in range(cfg.batch_size):
            config = adversary.propose(use_llm=cfg.adversary_use_llm and not cfg.dry_run)
            proposals.append(config)

        logger.info(
            "[Iter %d] Adversary proposed %d scenes: %s",
            iteration,
            len(proposals),
            [p.scene_type for p in proposals],
        )

        # ── Step 2: Simulate each scene ────────────────────────────────────────
        valid_qas = []
        for idx, config in enumerate(proposals):
            qa = _generate_physics_qa(config, episode_seed=global_step + idx)
            scene_valid = qa is not None

            if not scene_valid:
                # Penalize adversary for invalid scene
                adv_reward = adversary.get_reward(scene_valid=False, solver_succeeded=False)
                adversary.update(config.scene_type, config.difficulty, solver_succeeded=False)
                adversary.record_proposal(config, scene_valid=False, solver_succeeded=False, reward=adv_reward)

                entry = ReplayEntry(
                    scene_id=f"{config.scene_type}_{global_step}_{idx}",
                    scene_type=config.scene_type,
                    difficulty=config.difficulty,
                    question_type=config.question_type,
                    seed=config.seed,
                    problem_text="[SIMULATION FAILED]",
                    ground_truth=0.0,
                    scene_valid=False,
                    solver_reward=0.0,
                    adversary_reward=adv_reward,
                    iteration=iteration,
                )
                replay.push(entry)
            else:
                valid_qas.append((config, qa))

        iter_metrics["valid_scenes"] = len(valid_qas)
        logger.info("[Iter %d] %d/%d scenes valid", iteration, len(valid_qas), cfg.batch_size)

        # ── Step 3: Solver answers valid QA pairs ─────────────────────────────
        solver_rewards = []
        adversary_rewards = []

        for config, qa in valid_qas:
            problem_text = qa["text"]
            ground_truth = float(qa["answer"])

            # Solver inference
            if cfg.dry_run or solver is None:
                import random
                completion = f"The answer is \\boxed{{{ground_truth * (1 + random.uniform(-0.1, 0.1)):.4f}}}"
            else:
                completion = solver.answer(problem_text)

            # Compute solver reward
            from solver.reward import physics_reward, extract_boxed_answer
            solver_reward_list = physics_reward(
                [[{"role": "assistant", "content": completion}]],
                [ground_truth],
            )
            solver_reward = solver_reward_list[0]
            solver_succeeded = solver_reward >= 1.0

            # Compute adversary reward
            adv_reward = adversary.get_reward(scene_valid=True, solver_succeeded=solver_succeeded)

            # Update adversary curriculum
            adversary.update(config.scene_type, config.difficulty, solver_succeeded=solver_succeeded)
            adversary.record_proposal(config, scene_valid=True, solver_succeeded=solver_succeeded, reward=adv_reward)

            solver_rewards.append(solver_reward)
            adversary_rewards.append(adv_reward)

            # Store in replay buffer
            entry = ReplayEntry(
                scene_id=f"{config.scene_type}_{global_step}",
                scene_type=config.scene_type,
                difficulty=config.difficulty,
                question_type=config.question_type,
                seed=config.seed,
                problem_text=problem_text,
                ground_truth=ground_truth,
                simulation_mapping=qa.get("simulation_mapping", ""),
                solver_completion=completion,
                solver_answer=extract_boxed_answer(completion),
                solver_reward=solver_reward,
                adversary_reward=adv_reward,
                scene_valid=True,
                iteration=iteration,
                time_series=qa.get("time_series", ""),
            )
            replay.push(entry)

        if solver_rewards:
            iter_metrics["solver_reward_mean"] = sum(solver_rewards) / len(solver_rewards)
        if adversary_rewards:
            iter_metrics["adversary_reward_mean"] = sum(adversary_rewards) / len(adversary_rewards)

        global_step += cfg.batch_size

        # ── Step 4: Log metrics ────────────────────────────────────────────────
        if iteration % cfg.log_interval == 0:
            buf_stats = replay.stats()
            iter_metrics.update({
                "replay_buffer_size": buf_stats.get("size", 0),
                "iteration_time_s": time.time() - iter_start,
            })
            
            # Use unified tracking
            tracker.log(iter_metrics, step=iteration)
            
            metrics_rows.append(iter_metrics)
            pd.DataFrame(metrics_rows).to_csv(log_path, index=False)

        # ── Step 5: Save replay buffer ─────────────────────────────────────────
        if (not cfg.save_replay_at_end_only) and (iteration % cfg.save_interval == 0) and len(replay) > 0:
            save_path = replay.save(iteration)
            logger.info("Saved replay buffer → %s", save_path)

    logger.info("Self-play loop complete. Total steps: %d", global_step)
    logger.info("Replay buffer size: %d", len(replay))
    if len(replay) > 0:
        final_save_path = replay.save(cfg.num_iterations)
        logger.info("Saved final replay buffer → %s", final_save_path)
    pd.DataFrame(metrics_rows).to_csv(log_path, index=False)
    logger.info("Metrics saved to %s", log_path)
    
    tracker.plot_learning_curves(log_path, os.path.join(cfg.output_dir, "learning_curves.png"))
    tracker.finish()

    return replay


def main():
    parser = argparse.ArgumentParser(description="Sim2Reason Adversarial Self-Play Loop")
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--adversary_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--solver_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--solver_checkpoint", default=None)
    parser.add_argument("--output_dir", default="self_play_output")
    parser.add_argument("--save_replay_at_end_only", action="store_true", help="If set, save replay buffer only once at the end")
    parser.add_argument("--dry_run", action="store_true", help="Skip actual LLM inference (for testing)")
    parser.add_argument("--disable_tracking", action="store_true", help="Turn off tensorboard/wandb tracking")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    args = parser.parse_args()

    if args.config:
        cfg = SelfPlayConfig.from_yaml(args.config)
    else:
        cfg = SelfPlayConfig(
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            adversary_model=args.adversary_model,
            solver_model=args.solver_model,
            solver_checkpoint=args.solver_checkpoint,
            output_dir=args.output_dir,
            save_replay_at_end_only=args.save_replay_at_end_only,
            dry_run=args.dry_run,
            disable_tracking=args.disable_tracking,
        )

    run_self_play_loop(cfg)


if __name__ == "__main__":
    main()

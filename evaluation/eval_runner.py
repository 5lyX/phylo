"""
Evaluation Runner — benchmarks the Solver on held-out synthetic scenes.

Generates a fixed test set (seeded) for each scene type/difficulty combo
and measures:
  - Overall accuracy (5% tolerance)
  - Per-scene-type accuracy breakdown
  - Average response length (reasoning verbosity)
  - Reward variance
  - Generalization gap vs. training distribution

Usage:
    python evaluation/eval_runner.py --solver_checkpoint ./solver_grpo_output
    python evaluation/eval_runner.py --dry_run
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

import pandas as pd
import numpy as np

from adversary.adversary_config import VALID_SCENE_TYPES, VALID_DIFFICULTIES
from solver.reward import extract_boxed_answer, physics_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    solver_model: str = "Qwen/Qwen2.5-3B-Instruct"
    solver_checkpoint: Optional[str] = None
    output_dir: str = "./eval_output"
    output_file: str = "eval_results.csv"

    # Test set config
    num_problems_per_type: int = 10     # problems per (scene_type, difficulty)
    scene_types: List[str] = field(
        default_factory=lambda: [
            "BasicPulley", "IntermediatePulley", "BasicCollision",
            "SpringBlockSystems", "Rotation",
        ]
    )
    difficulties: List[str] = field(default_factory=lambda: ["EASY", "MEDIUM", "HARD"])
    seed_offset: int = 9999             # Offset from training seeds

    # Inference
    max_new_tokens: int = 1024
    temperature: float = 0.0            # Greedy for eval
    load_in_4bit: bool = False
    dry_run: bool = False
    train_replay_path: Optional[str] = None  # Used for computing generalization gap
    eval_real_world: bool = False       # Run on real-world OlympiadBench questions


def _generate_eval_problem(
    scene_type: str,
    difficulty: str,
    seed: int,
) -> Optional[dict]:
    """Generate a single eval problem using Sim2Reason pipeline."""
    import traceback
    from sim.scene_generator import SceneGenerator, SCENE_CONFIGS
    from sim.qa_gen_rule import data_gen
    from server.phylo_environment import (
        _DEFAULT_MAIN_CFG,
        _get_category_for_scene_type,
        _get_recorder_cfg_for_category,
    )

    if scene_type not in SCENE_CONFIGS:
        return None

    try:
        gen = SceneGenerator(subtype=scene_type, seed=seed)
        scene_yaml = gen.generate_scene_yaml()
    except Exception:
        logger.debug("SceneGenerator failed for %s/%s: %s", scene_type, difficulty, traceback.format_exc())
        return None

    category = _get_category_for_scene_type(scene_type)
    if category is None:
        return None

    recorder_cfg = _get_recorder_cfg_for_category(category)

    try:
        qa = data_gen(
            scene_yaml=scene_yaml,
            cfg=_DEFAULT_MAIN_CFG,
            recorder_cfg=recorder_cfg,
            seed=seed,
        )
        qa["scene_type"] = scene_type
        qa["difficulty"] = difficulty
        qa["seed"] = seed
        return qa
    except Exception:
        logger.debug("data_gen failed: %s", traceback.format_exc())
        return None


class EvalRunner:
    """Runs evaluation benchmarks on the Solver."""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self._solver = None

    def _load_solver(self):
        if self._solver is not None:
            return
        from solver.solver_agent import SolverAgent
        model = self.cfg.solver_checkpoint or self.cfg.solver_model
        self._solver = SolverAgent(
            model_name=model,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            load_in_4bit=self.cfg.load_in_4bit,
        )

    def run(self) -> pd.DataFrame:
        """
        Run full evaluation suite.

        Returns:
            DataFrame with columns: scene_type, difficulty, problem_text,
            ground_truth, predicted, reward, completion_length, seed.
        """
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if not self.cfg.dry_run:
            self._load_solver()

        all_results = []
        global_seed = self.cfg.seed_offset
        
        # Spec §7: Real-world transfer evaluation
        if self.cfg.eval_real_world:
            real_world_path = os.path.join(
                _PHYLO_ROOT, "..", "Sim2Reason", "verl_v4", "utils", "reward_score", 
                "testing_reward_functions", "qa_pairs_with_predicted_answers_sample.json"
            )
            real_world_path = os.path.normpath(real_world_path)
            if os.path.exists(real_world_path):
                logger.info("Evaluating Real-World Transfer Benchmark: %s", real_world_path)
                import json
                with open(real_world_path, "r", encoding="utf-8") as f:
                    rw_data = json.load(f)
                
                successes = 0
                total = 0
                for item in rw_data[:self.cfg.num_problems_per_type]: # Limit for speed
                    # Use existing predictions in JSON for dry_run, else run solver
                    problem_text = item.get("text", "")
                    # Try to extract a ground truth answer (often in boxed or as a string)
                    ground_truth_str = item.get("answer", "")
                    
                    if self.cfg.dry_run:
                        completion = item.get("LLM_01_predicted_answer", "")
                        reward = 1.0 if "\\boxed" in completion else 0.0  # Dummy reward for dry run
                    else:
                        completion = self._solver.answer(problem_text)
                        # We use simple string match or extraction for real world if numerical isn't perfect
                        pred = str(extract_boxed_answer(completion))
                        reward = 1.0 if pred in ground_truth_str or ground_truth_str in pred else 0.0
                    
                    if reward >= 1.0: successes += 1
                    total += 1
                    
                    all_results.append({
                        "scene_type": "RealWorld",
                        "difficulty": "Olympiad",
                        "seed": 0,
                        "ground_truth": ground_truth_str,
                        "predicted": completion[-50:], # Truncated
                        "reward": float(reward),
                        "completion_length": len(completion),
                        "problem_length": len(problem_text),
                    })
                if total > 0:
                    logger.info("  RealWorld / Olympiad: accuracy=%.2f (%d/%d)", successes / total, successes, total)
            else:
                logger.warning("eval_real_world requested but dataset not found at %s", real_world_path)

        for scene_type in self.cfg.scene_types:
            for difficulty in self.cfg.difficulties:
                logger.info("Evaluating %s / %s…", scene_type, difficulty)
                successes = 0
                total = 0

                for i in range(self.cfg.num_problems_per_type):
                    seed = global_seed + i

                    qa = _generate_eval_problem(scene_type, difficulty, seed)
                    if qa is None:
                        logger.debug("Skipping failed problem %s/%s seed=%d", scene_type, difficulty, seed)
                        continue

                    problem_text = qa["text"]
                    ground_truth = float(qa["answer"])

                    if self.cfg.dry_run:
                        # Return exact answer (upper bound)
                        completion = f"The answer is \\boxed{{{ground_truth:.4f}}}"
                    else:
                        completion = self._solver.answer(problem_text)

                    predicted = extract_boxed_answer(completion)
                    reward_list = physics_reward(
                        [[{"role": "assistant", "content": completion}]],
                        [ground_truth],
                    )
                    reward = reward_list[0]

                    if reward >= 1.0:
                        successes += 1
                    total += 1

                    all_results.append({
                        "scene_type": scene_type,
                        "difficulty": difficulty,
                        "seed": seed,
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "reward": reward,
                        "completion_length": len(completion),
                        "problem_length": len(problem_text),
                    })

                if total > 0:
                    logger.info(
                        "  %s / %s: accuracy=%.2f (%d/%d)",
                        scene_type, difficulty, successes / total, successes, total,
                    )

                global_seed += self.cfg.num_problems_per_type

        df = pd.DataFrame(all_results)

        # ── Summary metrics ────────────────────────────────────────────────────
        if len(df) > 0:
            logger.info("═" * 60)
            logger.info("EVALUATION SUMMARY")
            logger.info("═" * 60)
            
            eval_acc = df["reward"].mean()
            logger.info("Overall accuracy: %.3f", eval_acc)
            
            # Spec §7: Generalization gap
            if self.cfg.train_replay_path and os.path.exists(self.cfg.train_replay_path):
                train_df = pd.read_parquet(self.cfg.train_replay_path)
                train_acc = train_df["solver_reward"].mean()
                gap = train_acc - eval_acc
                logger.info("Train accuracy:   %.3f", train_acc)
                logger.info("Generalization gap: %.3f (Train - Eval)", gap)
            
            logger.info("Reward variance:  %.4f", df["reward"].var())
            logger.info("Mean completion length: %.0f tokens", df["completion_length"].mean())
            logger.info("\nPer scene type:")
            type_summary = df.groupby("scene_type")["reward"].agg(["mean", "count"])
            logger.info("\n%s", type_summary.to_string())
            logger.info("\nPer difficulty:")
            diff_summary = df.groupby("difficulty")["reward"].agg(["mean", "count"])
            logger.info("\n%s", diff_summary.to_string())

        out_path = os.path.join(self.cfg.output_dir, self.cfg.output_file)
        df.to_csv(out_path, index=False)
        logger.info("Results saved to %s", out_path)

        return df


def main():
    parser = argparse.ArgumentParser(description="Solver Evaluation Runner")
    parser.add_argument("--solver_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--solver_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./eval_output")
    parser.add_argument("--num_problems_per_type", type=int, default=10)
    parser.add_argument("--scene_types", nargs="+",
                        default=["BasicPulley", "IntermediatePulley", "BasicCollision",
                                 "SpringBlockSystems", "Rotation"])
    parser.add_argument("--difficulties", nargs="+", default=["EASY", "MEDIUM", "HARD"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--train_replay_path", default=None, help="Path to train replay buffer for generalization gap")
    parser.add_argument("--eval_real_world", action="store_true", help="Run real-world transfer benchmark")
    args = parser.parse_args()

    cfg = EvalConfig(
        solver_model=args.solver_model,
        solver_checkpoint=args.solver_checkpoint,
        output_dir=args.output_dir,
        num_problems_per_type=args.num_problems_per_type,
        scene_types=args.scene_types,
        difficulties=args.difficulties,
        dry_run=args.dry_run,
        load_in_4bit=args.load_in_4bit,
        train_replay_path=args.train_replay_path,
        eval_real_world=args.eval_real_world,
    )

    runner = EvalRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()

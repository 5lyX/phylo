"""
Parallel replay buffer generation without LLM inference.

This script is designed for fast synthetic data generation to train the solver.
It runs the physics scene generation + QA pipeline in parallel workers and writes
one combined replay parquet file compatible with training/train_solver.py.

Example:
    python self_play/generate_replay_parallel.py \
        --num_iterations 500 \
        --batch_size 8 \
        --num_workers 4 \
        --replay_dir replay_buffer_parallel
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

from adversary.adversary_config import AdversarySceneConfig, VALID_SCENE_TYPES
from self_play.self_play_loop import _generate_physics_qa
from self_play.replay_buffer import ReplayEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BLENDER_DEPENDENT_SCENES = {"RigidBodyRotation"}
_VALID_DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]
_QUESTION_TYPES = ["numeric", "reverse"]


@dataclass
class ParallelReplayConfig:
    num_iterations: int = 500
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    include_invalid: bool = False
    include_blender_dependent: bool = True
    output_dir: str = "self_play_output"
    replay_dir: str = "replay_buffer"


def _build_scene_pool(include_blender_dependent: bool) -> List[str]:
    if include_blender_dependent:
        return list(VALID_SCENE_TYPES)
    return [s for s in VALID_SCENE_TYPES if s not in _BLENDER_DEPENDENT_SCENES]


def _build_tasks(cfg: ParallelReplayConfig) -> List[Tuple[int, str, str, str, int, int]]:
    total = cfg.num_iterations * cfg.batch_size
    rng = random.Random(cfg.seed)
    scene_types = _build_scene_pool(cfg.include_blender_dependent)
    if not scene_types:
        raise ValueError("No scene types available after filtering.")

    tasks = []
    for idx in range(total):
        # Round-robin base for coverage + random jitter via shuffled list per cycle
        scene_type = scene_types[idx % len(scene_types)]
        difficulty = rng.choices(_VALID_DIFFICULTIES, weights=[0.4, 0.4, 0.2], k=1)[0]
        question_type = rng.choices(_QUESTION_TYPES, weights=[0.85, 0.15], k=1)[0]
        scene_seed = cfg.seed + idx
        tasks.append((idx, scene_type, difficulty, question_type, scene_seed, cfg.batch_size))
    return tasks


def _worker_generate(task: Tuple[int, str, str, str, int, int]) -> Dict:
    idx, scene_type, difficulty, question_type, scene_seed, batch_size = task

    config = AdversarySceneConfig(
        scene_type=scene_type,
        difficulty=difficulty,
        seed=scene_seed,
        question_type=question_type,
    )

    qa = _generate_physics_qa(config, episode_seed=idx)
    iteration = idx // batch_size
    scene_id = f"{scene_type}_{idx}"

    if qa is None:
        entry = ReplayEntry(
            scene_id=scene_id,
            scene_type=scene_type,
            difficulty=difficulty,
            question_type=question_type,
            seed=scene_seed,
            problem_text="[SIMULATION FAILED]",
            ground_truth=0.0,
            simulation_mapping="",
            solver_completion="",
            solver_answer=None,
            solver_reward=0.0,
            adversary_reward=-1.0,
            scene_valid=False,
            iteration=iteration,
            time_series="",
        )
        return asdict(entry)

    entry = ReplayEntry(
        scene_id=scene_id,
        scene_type=scene_type,
        difficulty=difficulty,
        question_type=question_type,
        seed=scene_seed,
        problem_text=qa["text"],
        ground_truth=float(qa["answer"]),
        simulation_mapping=qa.get("simulation_mapping", ""),
        solver_completion="",
        solver_answer=None,
        solver_reward=0.0,
        adversary_reward=0.0,
        scene_valid=True,
        iteration=iteration,
        time_series=qa.get("time_series", ""),
    )
    return asdict(entry)


def run_parallel_generation(cfg: ParallelReplayConfig) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.replay_dir, exist_ok=True)

    tasks = _build_tasks(cfg)
    total = len(tasks)
    logger.info(
        "Starting parallel replay generation: total_samples=%d, workers=%d",
        total,
        cfg.num_workers,
    )

    t0 = time.time()
    records: List[Dict] = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
        for i, record in enumerate(pool.map(_worker_generate, tasks, chunksize=max(1, cfg.batch_size // 2)), start=1):
            if cfg.include_invalid or record.get("scene_valid", False):
                records.append(record)
            if i % 100 == 0 or i == total:
                logger.info("Progress: %d/%d", i, total)

    if not records:
        raise RuntimeError("No records generated. Check scene generation setup.")

    df = pd.DataFrame(records).sort_values(by=["iteration", "scene_id"]).reset_index(drop=True)
    replay_path = os.path.join(cfg.replay_dir, f"replay_iter_{cfg.num_iterations:05d}.parquet")
    df.to_parquet(replay_path, index=False)

    elapsed = time.time() - t0
    stats = {
        "num_iterations": cfg.num_iterations,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "total_requested": total,
        "total_saved": int(len(df)),
        "valid_saved": int(df["scene_valid"].sum()) if "scene_valid" in df.columns else 0,
        "invalid_saved": int((~df["scene_valid"]).sum()) if "scene_valid" in df.columns else 0,
        "elapsed_seconds": round(elapsed, 2),
        "replay_path": replay_path,
    }
    stats_path = os.path.join(cfg.output_dir, "parallel_replay_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Replay saved to %s", replay_path)
    logger.info("Stats saved to %s", stats_path)
    logger.info("Done in %.2f seconds", elapsed)
    return replay_path


def main():
    parser = argparse.ArgumentParser(description="Parallel replay generation (no LLM inference).")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_invalid", action="store_true", help="Keep simulation-failed rows in output.")
    parser.add_argument(
        "--include_blender_dependent",
        action="store_true",
        help="Include Blender-dependent scenes like RigidBodyRotation (enabled by default).",
    )
    parser.add_argument(
        "--exclude_blender_dependent",
        action="store_true",
        help="Exclude Blender-dependent scenes like RigidBodyRotation.",
    )
    parser.add_argument("--output_dir", default="self_play_output")
    parser.add_argument("--replay_dir", default="replay_buffer")
    args = parser.parse_args()

    cfg = ParallelReplayConfig(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        include_invalid=args.include_invalid,
        include_blender_dependent=(args.include_blender_dependent or not args.exclude_blender_dependent),
        output_dir=args.output_dir,
        replay_dir=args.replay_dir,
    )
    run_parallel_generation(cfg)


if __name__ == "__main__":
    main()


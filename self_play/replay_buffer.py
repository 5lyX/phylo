"""
Replay Buffer — stores (scene_config, problem, completion, reward) tuples.

Used to:
1. Feed solver GRPO training batches.
2. Feed adversary GRPO training batches.
3. Persist data to disk for resumable training.
"""

import json
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, List, Optional

import pandas as pd


@dataclass
class ReplayEntry:
    """A single experience tuple from the self-play loop."""
    # Scene metadata
    scene_id: str
    scene_type: str
    difficulty: str
    question_type: str
    seed: int

    # Physics problem
    problem_text: str
    ground_truth: float
    simulation_mapping: str = ""

    # Solver response
    solver_completion: str = ""
    solver_answer: Optional[float] = None
    solver_reward: float = 0.0

    # Adversary info
    adversary_reward: float = 0.0
    scene_valid: bool = True

    # Episode metadata
    iteration: int = 0
    
    # Spec §3: Time series data
    time_series: str = ""


class ReplayBuffer:
    """
    Fixed-capacity replay buffer with FIFO eviction.

    Provides:
      - solver_dataset(): HuggingFace Dataset for GRPO training
      - adversary_dataset(): HuggingFace Dataset for adversary GRPO
      - save/load to parquet
    """

    def __init__(self, capacity: int = 50_000, save_dir: str = "replay_buffer"):
        self._buffer: Deque[ReplayEntry] = deque(maxlen=capacity)
        self._capacity = capacity
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def push(self, entry: ReplayEntry) -> None:
        """Add an experience to the buffer."""
        self._buffer.append(entry)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, n: int) -> List[ReplayEntry]:
        """Sample n entries (random, with replacement if needed)."""
        import random
        buf = list(self._buffer)
        if len(buf) <= n:
            return buf
        return random.sample(buf, n)

    def get_recent(self, n: int) -> List[ReplayEntry]:
        """Get the n most recent entries."""
        buf = list(self._buffer)
        return buf[-n:]

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset export for GRPO training
    # ──────────────────────────────────────────────────────────────────────────

    def solver_dataset(self, min_samples: int = 10):
        """
        Export buffer as a HuggingFace Dataset for GRPO Solver training.

        Format expected by GRPOTrainer:
          - "prompt": list of message dicts
          - "ground_truth": float (passed to reward fn via kwargs)
        """
        from datasets import Dataset

        entries = list(self._buffer)
        if len(entries) < min_samples:
            return None

        records = []
        for e in entries:
            records.append({
                "prompt": [
                    {"role": "system", "content": "You are a physics expert. Solve the problem step-by-step and end with \\boxed{your_answer}."},
                    {"role": "user", "content": e.problem_text},
                ],
                "ground_truth": str(e.ground_truth),
                "scene_id": e.scene_id,
                "scene_type": e.scene_type,
                "difficulty": e.difficulty,
            })

        return Dataset.from_list(records)

    def adversary_dataset(self, min_samples: int = 10):
        """
        Export buffer as a HuggingFace Dataset for GRPO Adversary training.

        The adversary's "prompt" asks it to propose a scene config.
        The "ground_truth" is the adversary's reward (1 if solver failed, else 0).
        """
        from datasets import Dataset

        entries = list(self._buffer)
        if len(entries) < min_samples:
            return None

        # Build a simple performance summary from recent entries
        from collections import defaultdict
        type_results = defaultdict(list)
        for e in entries[-500:]:  # Use last 500 entries for stats
            type_results[e.scene_type].append(e.solver_reward)

        perf_lines = []
        for st, rewards in type_results.items():
            avg = sum(rewards) / len(rewards) if rewards else 0.5
            perf_lines.append(f"  {st}: accuracy={avg:.2f}")
        perf_summary = "\n".join(perf_lines) if perf_lines else "  (no data yet)"

        records = []
        for e in entries:
            records.append({
                "prompt": [
                    {"role": "system", "content": "You are an adversarial physics problem generator. Output ONLY valid JSON."},
                    {"role": "user", "content": f"Current solver performance:\n{perf_summary}\n\nPropose a hard scene config as JSON:"},
                ],
                "ground_truth": str(e.adversary_reward),
                "scene_type": e.scene_type,
            })

        return Dataset.from_list(records)

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, iteration: int) -> str:
        """Save buffer to parquet."""
        path = os.path.join(self._save_dir, f"replay_iter_{iteration:05d}.parquet")
        records = [asdict(e) for e in self._buffer]
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
        return path

    def load(self, path: str) -> None:
        """Load buffer from parquet."""
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            entry = ReplayEntry(**row.to_dict())
            self._buffer.append(entry)

    def stats(self) -> dict:
        """Compute buffer statistics."""
        if not self._buffer:
            return {"size": 0}

        buf = list(self._buffer)
        solver_rewards = [e.solver_reward for e in buf]
        adv_rewards = [e.adversary_reward for e in buf]
        valid_pct = sum(1 for e in buf if e.scene_valid) / len(buf)

        from collections import Counter
        type_counts = Counter(e.scene_type for e in buf)

        return {
            "size": len(buf),
            "capacity": self._capacity,
            "solver_reward_mean": sum(solver_rewards) / len(solver_rewards),
            "adversary_reward_mean": sum(adv_rewards) / len(adv_rewards),
            "valid_scene_pct": valid_pct,
            "top_scene_types": type_counts.most_common(5),
        }

"""
Physics reward functions for the Solver agent.

The primary reward signal follows the Sim2Reason paper:
  reward = 1.0  if |predicted - ground_truth| / |ground_truth| ≤ 5%
  reward = 0.0  otherwise

This module provides both the TRL-compatible reward function signature
and helper utilities for answer extraction.
"""

import re
from typing import List, Optional, Union


def extract_boxed_answer(text: str) -> Optional[float]:
    """
    Extract the numeric answer from a model completion.

    Handles multiple common formats:
      - \\boxed{3.14}
      - <answer>3.14</answer>
      - "the answer is 3.14"
      - bare number at the end of the text

    Returns:
        Float value if found, None otherwise.
    """
    if not text:
        return None

    # 1. LaTeX boxed: \boxed{3.14} or \boxed{3.14 \text{ m/s}}
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        inner = boxed.group(1).strip()
        # Remove LaTeX formatting like \text{...} or units
        inner = re.sub(r"\\text\{[^}]*\}", "", inner).strip()
        inner = re.sub(r"[a-zA-Z/²³]+$", "", inner).strip()  # remove trailing units
        inner = inner.replace(",", "").replace(" ", "")
        try:
            return float(inner)
        except ValueError:
            pass

    # 2. XML-tagged answer
    tagged = re.search(r"<answer>\s*([\-\+]?\d*\.?\d+(?:[eE][\-\+]?\d+)?)\s*</answer>", text)
    if tagged:
        try:
            return float(tagged.group(1))
        except ValueError:
            pass

    # 3. "= X" at the end of a sentence (common in CoT answers)
    eq_match = re.findall(r"=\s*([\-\+]?\d*\.?\d+(?:[eE][\-\+]?\d+)?)\s*(?:m|s|kg|N|J|W|Pa|rad|°|$)", text)
    if eq_match:
        try:
            return float(eq_match[-1])
        except ValueError:
            pass

    # 4. Last float in the text
    all_nums = re.findall(r"[\-\+]?\d*\.?\d+(?:[eE][\-\+]?\d+)?", text)
    for n in reversed(all_nums):
        try:
            val = float(n)
            # Sanity: skip trivially small numbers that are likely years/counts
            if abs(val) > 1e-12 or val == 0.0:
                return val
        except ValueError:
            continue

    return None


def physics_reward(
    completions: List[List[dict]],
    ground_truths: List[Union[str, float]],
    tolerance: float = 0.05,
    **kwargs,
) -> List[float]:
    """
    TRL-compatible reward function for GRPO training.

    Args:
        completions: List of completion message lists (TRL format).
                     Each item is [{"role": "assistant", "content": "..."}].
        ground_truths: Simulator ground truth values (as str or float).
        tolerance: Relative error tolerance (default 5% as in Sim2Reason paper).

    Returns:
        List of float rewards, one per completion.
    """
    rewards = []

    for completion, gt_raw in zip(completions, ground_truths):
        # Extract text from TRL message format
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "")
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)

        # Parse ground truth
        try:
            gt = float(gt_raw)
        except (TypeError, ValueError):
            rewards.append(0.0)
            continue

        # Extract predicted value
        predicted = extract_boxed_answer(text)

        if predicted is None:
            rewards.append(0.0)
            continue

        # Relative error check
        if abs(gt) > 1e-9:
            rel_err = abs(predicted - gt) / abs(gt)
            reward = 1.0 if rel_err <= tolerance else 0.0
        else:
            # Ground truth is ~0: use absolute tolerance
            reward = 1.0 if abs(predicted - gt) <= 1e-4 else 0.0

        rewards.append(reward)

    return rewards


def format_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    Format reward: checks that the completion contains a valid numeric answer
    in a recognizable format (\\boxed or <answer> tags).

    Encourages the model to always produce a clearly extractable answer.
    Returns 0.5 if format is correct, 0.0 otherwise (combined with physics_reward).
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "")
        else:
            text = str(completion)

        has_boxed = bool(re.search(r"\\boxed\{", text))
        has_answer_tag = bool(re.search(r"<answer>", text))

        rewards.append(0.5 if (has_boxed or has_answer_tag) else 0.0)

    return rewards

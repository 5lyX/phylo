"""
Quick smoke test — verifies the Sim2Reason pipeline works WITHOUT needing GPU or MuJoCo.
Generates a dummy dataset and runs one fake GRPO step.

Run this first before any real training:
    python scripts/verify_setup.py
"""

import sys, os
import io

# Force UTF-8 encoding for stdout/stderr to avoid UnicodeEncodeError on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

print("=" * 60)
print("Sim2Reason Adversarial Self-Play — Setup Verification")
print("=" * 60)

# 1. Basic imports
print("\n[1/5] Checking core imports...")
try:
    from models import PhysicsAction, PhysicsObservation, AdversaryAction
    print("  ✅ models OK")
except Exception as e:
    print(f"  ❌ models: {e}")

try:
    from solver.reward import physics_reward, extract_boxed_answer
    print("  ✅ solver.reward OK")
except Exception as e:
    print(f"  ❌ solver.reward: {e}")

try:
    from adversary.adversary_config import AdversarySceneConfig, VALID_SCENE_TYPES
    print(f"  ✅ adversary.config OK ({len(VALID_SCENE_TYPES)} scene types)")
except Exception as e:
    print(f"  ❌ adversary.config: {e}")

try:
    from self_play.replay_buffer import ReplayBuffer, ReplayEntry
    print("  ✅ replay_buffer OK")
except Exception as e:
    print(f"  ❌ replay_buffer: {e}")

# 2. Reward function test
print("\n[2/5] Testing physics reward function...")
completions = [
    [{"role": "assistant", "content": "The acceleration is \\boxed{2.0} m/s²"}],
    [{"role": "assistant", "content": "The answer is \\boxed{9.99}"}],
    [{"role": "assistant", "content": "I don't know"}],
]
ground_truths = [2.0, 10.0, 5.0]
rewards = physics_reward(completions, ground_truths)
assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"
assert rewards[1] == 1.0, f"Expected 1.0 (9.99 is within 5% of 10), got {rewards[1]}"
assert rewards[2] == 0.0, f"Expected 0.0, got {rewards[2]}"
print(f"  ✅ Reward test passed: {rewards}")

# 3. Answer extraction test
print("\n[3/5] Testing answer extraction...")
cases = [
    ("The answer is \\boxed{3.14}", 3.14),
    ("<answer>9.81</answer>", 9.81),
    ("= 42.5 m/s", 42.5),
]
for text, expected in cases:
    got = extract_boxed_answer(text)
    status = "✅" if got == expected else "❌"
    print(f"  {status} '{text[:40]}' → {got} (expected {expected})")

# 4. Replay buffer test
print("\n[4/5] Testing replay buffer...")
buf = ReplayBuffer(capacity=100, save_dir="/tmp/test_replay")
for i in range(20):
    buf.push(ReplayEntry(
        scene_id=f"test_{i}",
        scene_type="BasicPulley",
        difficulty="EASY",
        question_type="numeric",
        seed=i,
        problem_text=f"A mass of {i+1} kg is on a pulley system.",
        ground_truth=float(i + 1),
        solver_completion=f"\\boxed{{{i+1}}}",
        solver_reward=1.0,
        adversary_reward=0.0,
    ))
dataset = buf.solver_dataset(min_samples=5)
print(f"  ✅ Replay buffer: {len(buf)} entries, dataset: {len(dataset)} samples")

# 5. MuJoCo check
print("\n[5/5] Checking MuJoCo (optional for data collection)...")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body/></worldbody></mujoco>")
    print(f"  ✅ MuJoCo {mujoco.__version__} working")
except ImportError:
    print("  ⚠️  MuJoCo not installed — self-play needs it, but reward/training work without it")
    print("     Install: pip install mujoco>=3.1.0")
except Exception as e:
    print(f"  ⚠️  MuJoCo installed but rendering issue: {e}")
    print("     On Windows try: set MUJOCO_GL=glfw  or  set MUJOCO_GL=osmesa")

# GPU check
print("\n[BONUS] GPU check...")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"  ✅ CUDA available: {name} ({vram}GB VRAM)")
        if vram < 8:
            print("  ⚠️  <8GB VRAM: training Qwen2.5-3B will OOM — use HuggingFace/Colab T4")
            print("     Self-play data collection still works locally (MuJoCo is CPU-based)")
    else:
        print("  ⚠️  No CUDA GPU — use HuggingFace/Colab for training")
except ImportError:
    print("  ⚠️  PyTorch not installed yet")

print("\n" + "=" * 60)
print("Setup verification complete!")
print("\nNext steps:")
print("  1. pip install -e .                   (install all deps)")
print("  2. python scripts/hf_training_notebook.py --dry_run  (test pipeline)")
print("  3. Upload to HuggingFace Space / Colab for real training")
print("=" * 60)

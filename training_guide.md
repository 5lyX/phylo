# Training Guide: Sim2Reason Adversarial Self-Play

> **Your hardware**: RTX 3050 Laptop (4GB VRAM)  
> **Training target**: Qwen2.5-3B Solver + Qwen2.5-0.5B Adversary

---

## TL;DR — Recommended Approach

```
Your Machine (Windows RTX 3050)          HuggingFace / Colab T4
────────────────────────────────         ──────────────────────────────
Step 1: pip install + verify         →   Step 3: Upload replay buffer
Step 2: Run self-play dry run             Step 4: Run train_solver.py (GRPO)
        (generates training data)         Step 5: Run train_adversary.py (GRPO)
                                     ←   Step 6: Download trained checkpoint
                                          Step 7: Eval runs locally
```

---

## Part 1 — Local Setup (Do This First)

### Install dependencies

```powershell
cd d:\frontendsProd\hackathon\phylo
.venv\Scripts\activate

# Core deps (no torch yet — install separately below)
pip install openenv-core hydra-core omegaconf pandas scipy pyyaml tqdm ipdb imageio

# MuJoCo (Windows — uses glfw by default)
pip install mujoco>=3.1.0

# PyTorch (CUDA 12.1 — adjust cu121 for your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training stack
pip install transformers>=4.42.0 trl>=0.9.0 peft>=0.11.0 accelerate>=0.30.0
pip install bitsandbytes>=0.43.0 datasets>=2.19.0 wandb
```

### Set MuJoCo display backend (Windows)

```powershell
# In PowerShell before running anything
$env:MUJOCO_GL = "glfw"   # needs a display (your local screen) — works on Windows
# OR
$env:MUJOCO_GL = "osmesa"  # software rendering, slower but no display needed
```

### Verify everything works

```powershell
python scripts/verify_setup.py
```

Expected output:
```
✅ models OK
✅ solver.reward OK
✅ adversary.config OK (18 scene types)
✅ replay_buffer OK
✅ Reward test passed: [1.0, 1.0, 0.0]
✅ MuJoCo 3.x working
⚠️  <8GB VRAM: training Qwen2.5-3B will OOM — use HuggingFace/Colab
```

---

## Part 2 — Local Self-Play Data Collection

> [!NOTE]
> MuJoCo simulation runs on **CPU** — no GPU needed here.
> This generates the replay buffer that HuggingFace will train on.

### Option A: Dry run (no LLM inference, instant, just tests the pipeline)

```powershell
python self_play/self_play_loop.py --num_iterations 10 --batch_size 4 --dry_run
```

### Option B: Real self-play with heuristic adversary (no LLM, still useful)

This skips the Qwen2.5-0.5B adversary and uses the heuristic curriculum sampler instead. Runs on CPU alone, generates real physics QA data.

```powershell
# Edit self_play/self_play_loop.py: adversary_use_llm=False 
python self_play/self_play_loop.py ^
    --num_iterations 200 ^
    --batch_size 8 ^
    --output_dir self_play_output ^
    --dry_run
```

This creates `replay_buffer/replay_iter_00200.parquet` — the training data file.

### Option C: Full self-play with both LLMs (4GB VRAM — tight but possible)

```powershell
# Load adversary (0.5B) and solver (3B) both in 4-bit
# Runs one at a time — not simultaneous
python self_play/self_play_loop.py ^
    --num_iterations 100 ^
    --batch_size 4 ^
    --adversary_model Qwen/Qwen2.5-0.5B-Instruct ^
    --solver_model Qwen/Qwen2.5-3B-Instruct
```

> [!WARNING]
> On 4GB VRAM, loading Qwen2.5-3B (even 4-bit ~3.5GB) plus MuJoCo + Python overhead
> will likely OOM during the solver inference step. If it crashes, use Option B + train on HF.

---

## Part 3 — Training on HuggingFace (Recommended for 3B Model)

> [!IMPORTANT]
> **Free T4 GPU = 15GB VRAM** — easily fits Qwen2.5-3B + LoRA + GRPO

### Method A: HuggingFace Spaces (Persistent, No Time Limit)

1. **Create a new HF Space** → `New Space` → `Docker` SDK
2. **Upload your phylo directory** (or push via git):
   ```bash
   cd d:\frontendsProd\hackathon\phylo
   git init
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/sim2reason-training
   git add .
   git commit -m "Initial self-play training setup"
   git push hf main
   ```
3. **Run the training script** in the Space terminal:
   ```bash
   export MUJOCO_GL=egl  # Headless on Linux
   python scripts/hf_training_notebook.py \
       --iterations 500 \
       --batch_size 8 \
       --solver_max_steps 500 \
       --adversary_max_steps 200 \
       --num_cycles 3 \
       --push_to_hub \
       --hub_model_id "YOUR_USERNAME/sim2reason-solver-3b"
   ```

### Method B: Google Colab (Free T4, 12-hour sessions)

Create a new Colab notebook with these cells:

**Cell 1 — Clone and install**
```python
!git clone https://github.com/YOUR_REPO/hackathon /content/hackathon
%cd /content/hackathon/phylo

!pip install mujoco>=3.1.0 trl>=0.9.0 peft>=0.11.0 transformers>=4.42.0 \
    accelerate bitsandbytes datasets hydra-core omegaconf pandas wandb
```

**Cell 2 — Set environment**
```python
import os, sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/content/hackathon/phylo")
```

**Cell 3 — Self-play data collection (CPU)**
```python
from scripts.hf_training_notebook import cell_2_self_play
replay = cell_2_self_play(num_iterations=200, batch_size=8, dry_run=False)
```

**Cell 4 — Train solver (T4 GPU)**
```python
from scripts.hf_training_notebook import cell_3_train_solver
import glob
replay_files = sorted(glob.glob("./replay_buffer/replay_iter_*.parquet"))
cell_3_train_solver(
    replay_path=replay_files[-1],
    max_steps=300,
    push_to_hub=True,
    hub_model_id="YOUR_USERNAME/sim2reason-solver-3b"
)
```

**Cell 5 — Train adversary**
```python
from scripts.hf_training_notebook import cell_4_train_adversary
cell_4_train_adversary(replay_path=replay_files[-1], max_steps=100)
```

**Cell 6 — Evaluate**
```python
from scripts.hf_training_notebook import cell_5_evaluate, cell_6_plot_metrics
df = cell_5_evaluate(solver_checkpoint="./solver_grpo_output")
cell_6_plot_metrics()
```

---

## Part 4 — What to Do on 4GB VRAM Locally

You can still do meaningful things locally:

### ✅ Collect self-play data (most important!)

```powershell
# This generates your training corpus on CPU — no GPU needed
python self_play/self_play_loop.py ^
    --num_iterations 500 ^
    --batch_size 8 ^
    --dry_run ^
    --output_dir self_play_output
```

Remove `--dry_run` to use real MuJoCo simulation (still CPU). The adversary will use its **heuristic mode** (no LLM) which is actually fine for initial data collection.

### ✅ Train adversary only (0.5B fits in 4GB with 4-bit)

```powershell
python training/train_adversary.py ^
    --model_name Qwen/Qwen2.5-0.5B-Instruct ^
    --replay_path replay_buffer/replay_iter_00200.parquet ^
    --max_steps 100 ^
    --output_dir adversary_grpo_output
```

### ✅ Smoke test the full solver training (1 step, verify it doesn't crash)

```powershell
python training/train_solver.py ^
    --model_name Qwen/Qwen2.5-0.5B-Instruct ^
    --max_steps 1 ^
    --dummy_data ^
    --output_dir solver_smoke_test
```

> [!TIP]
> Use `Qwen2.5-0.5B` as the solver locally for debugging. It's the same code —
> just swap the model name. Once you confirm everything works, switch to 3B on T4.

---

## Part 5 — Training Config Reference

### Solver (Qwen2.5-3B, T4 recommended)

```powershell
python training/train_solver.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --replay_path replay_buffer/replay_iter_00200.parquet \
    --max_steps 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --lora_r 16 \
    --learning_rate 5e-6 \
    --output_dir solver_grpo_output \
    --push_to_hub \
    --hub_model_id "you/sim2reason-solver-3b"
```

### Adversary (Qwen2.5-0.5B, fits locally)

```powershell
python training/train_adversary.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --replay_path replay_buffer/replay_iter_00200.parquet \
    --max_steps 100 \
    --output_dir adversary_grpo_output
```

### Evaluation

```powershell
python evaluation/eval_runner.py \
    --solver_checkpoint solver_grpo_output \
    --num_problems_per_type 20 \
    --dry_run
```

---

## Part 6 — Tracking and Monitoring

The codebase includes an integrated tracking utility that automatically logs metrics to Console, TensorBoard, and WandB (if configured). 

### Viewing Real-Time Graphs
During self-play or training, open a new terminal in the `phylo` folder and start TensorBoard:
```powershell
tensorboard --logdir tensorboard_log
```
Then, navigate to `http://localhost:6006` in your browser.

### Static PNG Curves
When any of the main loops finish, a `.png` file is automatically saved into the output directory.
- **Self-Play:** `self_play_output/learning_curves.png`
- **Solver Training:** `solver_grpo_output/training_curves.png`
- **Adversary Training:** `adversary_grpo_output/training_curves.png`

### Disabling Tracking
If you want to reduce overhead and disable TensorBoard/WandB logging entirely, append the `--disable_tracking` flag to any of your scripts:
```powershell
python self_play/self_play_loop.py --disable_tracking
python training/train_solver.py --disable_tracking
python training/train_adversary.py --disable_tracking
```

---

## Part 7 — VRAM Budget Reference (T4, 16GB)

| Component | VRAM Usage |
|---|---|
| Qwen2.5-3B (bfloat16) | ~6 GB |
| + LoRA adapters | ~0.2 GB |
| + GRPO 8 generations | ~4 GB |
| + Optimizer states | ~3 GB |
| **Total** | **~13 GB** ✅ fits T4 |

| Component | VRAM Usage |
|---|---|
| Qwen2.5-0.5B (bfloat16) | ~1.5 GB |
| + LoRA + GRPO | ~2 GB |
| **Total** | **~3.5 GB** ✅ fits RTX 3050 |

---

## Summary Decision Tree

```
Do you want to train?
│
├── Just collect data / test pipeline?
│   └── ✅ Local (RTX 3050) — python self_play/self_play_loop.py --dry_run
│
├── Train Adversary (0.5B)?
│   └── ✅ Local — python training/train_adversary.py
│
├── Train Solver (3B)?
│   ├── HuggingFace Space (free, persistent) ← Recommended
│   ├── Google Colab (free T4, 12hr limit)
│   └── ❌ RTX 3050 (4GB) — OOM
│
└── Full pipeline?
    └── ✅ Data collection local → Upload parquet → Train on HF → Download checkpoint
```

---
title: Phylo Physics Environment Server
emoji: ⚛️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - physics
  - sim2reason
  - reinforcement-learning
---

# Phylo — Sim2Reason Physics Reasoning Environment

A **MuJoCo-backed physics reasoning environment** with **adversarial self-play**, built on top of the [Sim2Reason](https://arxiv.org/abs/...) framework and the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) interface.

The environment generates physics problems from simulations and trains LLMs to reason about them using RLVR (Reinforcement Learning with Verifiable Rewards).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL SELF-PLAY LOOP                    │
├───────────────────────────┬─────────────────────────────────────┤
│    Adversary Agent        │         Solver Agent                │
│  (Qwen2.5-0.5B)          │       (Qwen2.5-3B)                 │
│                           │                                     │
│  Proposes scene configs   │  Answers physics problems           │
│  (JSON → SceneGenerator) │  (CoT + \boxed{answer})            │
│                           │                                     │
│  Reward: +1 if solver    │  Reward: +1 if |pred-gt|/|gt|≤5%   │
│  fails, -1 if invalid    │         0 otherwise                  │
└───────────────────────────┴─────────────────────────────────────┘
                       ↕ MuJoCo Simulation
               (Sim2Reason SceneGenerator + data_gen)
```

### Data Flow
1. **Adversary** proposes a scene config as JSON (e.g. `{"scene_type": "IntermediatePulley", "difficulty": "HARD"}`)
2. **SceneGenerator** builds the physics scene (DSL → YAML)
3. **MuJoCo** simulates it and records time-series data
4. **QA Generator** produces a natural language physics problem
5. **Solver** reads the problem and returns a chain-of-thought + numeric answer
6. **Reward** computed: 5% relative error tolerance
7. Both agents are updated via **GRPO** (Group Relative Policy Optimization)

---

## Quick Start

### Run the Environment Server

```bash
# Clone and install
cd phylo
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .

# Set headless MuJoCo (Linux/T4)
export MUJOCO_GL=egl

# Start the server
uvicorn server.app:app --port 8000
```

### Interact via Python Client

```python
from phylo import PhyloEnv, PhysicsAction

with PhyloEnv(base_url="http://localhost:8000") as env:
    # Reset gets a fresh physics problem
    result = env.reset()
    print(result.observation.problem_text)

    # Step submits your answer
    result = env.step(PhysicsAction(
        answer="3.14",
        reasoning="Using F=ma: a = F/m = 10/5 = 2 m/s²"
    ))
    print(f"Reward: {result.reward}")  # 1.0 if within 5% of ground truth
```

---

## Training Pipeline

### Full Pipeline (HuggingFace T4)

```bash
# Run the complete adversarial self-play pipeline
python scripts/hf_training_notebook.py \
    --iterations 500 \
    --batch_size 4 \
    --solver_max_steps 300 \
    --adversary_max_steps 150 \
    --num_cycles 3 \
    --push_to_hub \
    --hub_model_id "your-org/sim2reason-solver-3b"
```

### Individual Components

```bash
# 1. Self-play data collection
python self_play/self_play_loop.py \
    --num_iterations 200 \
    --batch_size 8 \
    --output_dir self_play_output

# 2. Train Solver
python training/train_solver.py \
    --replay_path replay_buffer/replay_iter_00200.parquet \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --max_steps 200 \
    --output_dir solver_grpo_output

# 3. Train Adversary
python training/train_adversary.py \
    --replay_path replay_buffer/replay_iter_00200.parquet \
    --max_steps 100

# 4. Evaluate
python evaluation/eval_runner.py \
    --solver_checkpoint ./solver_grpo_output \
    --num_problems_per_type 20

# 5. Plot metrics
python ../Sim2Reason/plot_metrics.py
```

---

## Environment Details

### Action
**PhysicsAction** — submitted by the Solver agent
| Field | Type | Description |
|---|---|---|
| `answer` | str | Numeric answer, e.g. `"3.14"` or `"\\boxed{3.14}"` |
| `reasoning` | str | Chain-of-thought (optional) |

### Observation
**PhysicsObservation** — returned to the Solver
| Field | Type | Description |
|---|---|---|
| `problem_text` | str | Full physics problem in natural language |
| `scene_id` | str | Unique scene identifier |
| `scene_type` | str | e.g. `"BasicPulley"`, `"SpringBlockSystems"` |
| `difficulty` | str | `EASY` / `MEDIUM` / `HARD` |
| `question_type` | str | `numeric` / `symbolic` / `reverse` |
| `reward` | float | 1.0 if correct, 0.0 otherwise (after step) |
| `done` | bool | True after step (single-turn QA) |

### Reward
```
R = 1.0  if |predicted - ground_truth| / |ground_truth| ≤ 5%
R = 0.0  otherwise
```
Ground truth comes directly from MuJoCo simulation — no human labels.

---

## Supported Physics Domains

| Category | Scene Types |
|---|---|
| Pulley Systems | BasicPulley, IntermediatePulley, AdvancedHybrid, DifficultPulley |
| Inclined Planes | BasicInclinedPlaneFriction, IntermediateInclinedPlaneFriction |
| Collisions | BasicCollision, IntermediateCollision, AdvancedCollision |
| Rotation | Rotation, RigidBodyRotation |
| Springs | SpringBlockSystems, DifficultSpringMass |
| Orbital Mechanics | DifficultOrbitalMotion, DifficultRocket |
| Electromagnetism | DifficultElectroMagnetic |

---

## Project Structure

```
phylo/
├── __init__.py            # Module exports
├── models.py              # PhysicsAction, PhysicsObservation, AdversaryAction
├── client.py              # PhyloEnv client
├── pyproject.toml         # Project + dependencies
│
├── sim/                   # Junction → Sim2Reason/sim/ (DSL + SceneGenerator)
├── recorder/              # Junction → Sim2Reason/recorder/ (MuJoCo recorder)
├── config/                # Junction → Sim2Reason/config/
│
├── server/
│   ├── phylo_environment.py   # Core PhyloEnvironment (MuJoCo + data_gen)
│   └── app.py                 # FastAPI HTTP + WebSocket server
│
├── adversary/
│   ├── adversary_agent.py     # Qwen2.5-0.5B scene proposer + EMA curriculum
│   └── adversary_config.py    # Valid scene types, JSON schema
│
├── solver/
│   ├── solver_agent.py        # Qwen2.5-3B inference wrapper
│   └── reward.py              # Physics reward + answer extraction
│
├── self_play/
│   ├── self_play_loop.py      # Flywheel orchestrator
│   └── replay_buffer.py       # Experience buffer → HF Dataset
│
├── training/
│   ├── train_solver.py        # GRPO for Solver (Qwen2.5-3B + LoRA)
│   ├── train_adversary.py     # GRPO for Adversary (Qwen2.5-0.5B + LoRA)
│   └── curriculum.py          # EMA difficulty tracker
│
├── evaluation/
│   └── eval_runner.py         # Benchmarking suite
│
└── scripts/
    └── hf_training_notebook.py  # Full pipeline for HuggingFace T4
```

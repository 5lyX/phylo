"""
PhyloEnvironment — core OpenEnv environment for Sim2Reason.

Replaces the echo environment with a real MuJoCo-backed physics simulation.
Each episode:
  1. SceneGenerator builds a random physics scene (DSL → YAML).
  2. data_gen() runs the MuJoCo simulation and produces a QA pair.
  3. The Solver agent reads the problem text and submits a numeric answer.
  4. The reward is 1.0 if |pred - gt| / |gt| ≤ 5%, else 0.0.

MuJoCo rendering is headless — set MUJOCO_GL=egl on Linux (T4) or
MUJOCO_GL=osmesa as fallback. No display required.
"""

import os
import re
import sys
import logging
import traceback
from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# ── Headless MuJoCo (matches Sim2Reason practice) ─────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

# ── Path bootstrap so "sim" and "recorder" packages are findable ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)          # phylo/
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

# ── Import Sim2Reason modules (via symlinked directories) ──────────────────────
from sim.scene_generator import SceneGenerator, SCENE_CONFIGS          # noqa: E402
from sim.qa_gen_rule import data_gen, POTENTIAL_FIND_QUANTITIES        # noqa: E402
from omegaconf import OmegaConf                                        # noqa: E402

try:
    from ..models import PhysicsAction, PhysicsObservation             # package import
except ImportError:
    from models import PhysicsAction, PhysicsObservation               # direct run

logger = logging.getLogger(__name__)

# ── Default recorder config (no render, headless, fast) ───────────────────────
_DEFAULT_RECORDER_CFG = OmegaConf.create({
    "dt": 1e-3,
    "duration": 4,
    "render": False,
    "parallel": False,
    "num_workers": 1,
    "fps": 60,
    "height": 1080,
    "width": 1920,
    "lookat": [2, 0, 1.5],
    "custom_camera": False,
    "distance": 15,
    "azimuth": 45,
    "elevation": -15,
    "prune_timesteps": True,
    "prune_first_contact": False,
    "prune_derivative": True,
    "prune_tendon_length_change": True,
    "threshold_tendon_length_change": 1e-3,
    "threshold_derivative": 1e-3,
    "plot_data": False,
    "debug": False,
    "disable_trail": True,
    "input_dir": "",
    "mode": "normal",
    "orbit_camera": False,
    "adaptive_camera_distance": False,
    "debug_adaptive_camera": False,
    "enable_smart_focus": False,
    "debug_focus_camera": False,
})

_DEFAULT_QUESTION_GEN_CFG = OmegaConf.create({
    "numerical": True,
    "symbolic": False,
    "reverse": False,
    "num_generations_per_problem": 1,
    "model_name": "heuristic",
    "solve_locally": True,
    "use_openrouter": False,
    "gpu_memory_utilization": 0.9,
    "reuse": False,
    "build_child_scenes": False,
})

_DEFAULT_MAIN_CFG = OmegaConf.create({
    "question_generation": _DEFAULT_QUESTION_GEN_CFG,
    "recorder": _DEFAULT_RECORDER_CFG,
    "seed": 42,
    "debug": False,
    "solve_locally": True,
    "model_name": "heuristic",
    "max_samples": -1,
    "num_factors": -1,
    "factor_id": -1,
})


def _get_category_for_scene_type(scene_type: str) -> Optional[str]:
    """Map scene_type → physics category (pulley/collision/spring/rotation/orbital/em)."""
    for cat, cfg in POTENTIAL_FIND_QUANTITIES.items():
        if scene_type in cfg.get("categories", []):
            return cat
    return None


def _get_recorder_cfg_for_category(category: str) -> OmegaConf:
    """Load the category-specific recorder overrides, exactly as Sim2Reason does."""
    cfg = OmegaConf.create(dict(_DEFAULT_RECORDER_CFG))
    override_path = os.path.join(_PHYLO_ROOT, "config", "recorder", f"{category}.yaml")
    if os.path.exists(override_path):
        override = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(cfg, override)
    return cfg


def _extract_number(text: str) -> Optional[float]:
    """
    Extract the first numeric value from the solver's answer string.
    Handles formats: '3.14', '\\boxed{3.14}', 'the answer is 3.14 m/s', etc.
    """
    # Try boxed first
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        try:
            return float(boxed.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try <answer>...</answer>
    tagged = re.search(r"<answer>\s*([\d.eE+\-]+)\s*</answer>", text)
    if tagged:
        try:
            return float(tagged.group(1))
        except ValueError:
            pass

    # Fallback: first float/int in the string
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    for n in nums:
        try:
            return float(n)
        except ValueError:
            continue
    return None


class PhyloEnvironment(Environment):
    """
    MuJoCo-backed physics reasoning environment.

    Each episode generates a fresh physics scene using Sim2Reason's
    SceneGenerator + data_gen pipeline and presents it as a natural-language
    QA problem to the Solver agent.

    Supports concurrent WebSocket sessions — each client gets its own instance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scene_type: str = "BasicPulley",
        difficulty: str = "EASY",
        seed: int = 42,
        question_type: str = "numeric",
    ):
        self._scene_type = scene_type
        self._difficulty = difficulty
        self._base_seed = seed
        self._question_type = question_type

        # Episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_problem: Optional[dict] = None   # raw data_gen output
        self._ground_truth: Optional[float] = None
        self._scene_id: str = ""
        self._episode_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> PhysicsObservation:
        """
        Generate a new physics scene and QA pair.
        Returns a PhysicsObservation with the problem text.
        """
        self._episode_count += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scene_id = f"{self._scene_type}_{self._episode_count}_{uuid4().hex[:8]}"

        problem = self._generate_problem()

        if problem is None:
            # Simulation failed — return an error observation with zero reward
            return PhysicsObservation(
                problem_text="[SIMULATION ERROR] Could not generate a valid physics problem.",
                scene_id=self._scene_id,
                scene_type=self._scene_type,
                difficulty=self._difficulty,
                question_type=self._question_type,
                step_num=0,
                done=False,
                reward=0.0,
                metadata={"error": True},
            )

        self._current_problem = problem
        self._ground_truth = float(problem["answer"])

        return PhysicsObservation(
            problem_text=problem["text"],
            scene_id=self._scene_id,
            scene_type=self._scene_type,
            difficulty=self._difficulty,
            question_type=self._question_type,
            step_num=0,
            done=False,
            reward=0.0,
            metadata={
                "scene_id": self._scene_id,
                "simulation_mapping": problem.get("simulation_mapping", ""),
            },
        )

    def step(self, action: PhysicsAction) -> PhysicsObservation:  # type: ignore[override]
        """
        Evaluate the Solver's answer against the physics simulator ground truth.

        Reward:
          1.0 if |predicted - ground_truth| / |ground_truth| ≤ 5%
          0.0 otherwise
        """
        self._state.step_count += 1

        if self._ground_truth is None:
            # No active problem (step called before reset)
            return PhysicsObservation(
                problem_text="",
                scene_id=self._scene_id,
                scene_type=self._scene_type,
                difficulty=self._difficulty,
                question_type=self._question_type,
                step_num=self._state.step_count,
                done=True,
                reward=0.0,
                metadata={"error": "step called before reset"},
            )

        predicted = _extract_number(action.answer)
        gt = self._ground_truth

        if predicted is not None and abs(gt) > 1e-9:
            rel_err = abs(predicted - gt) / abs(gt)
            reward = 1.0 if rel_err <= 0.05 else 0.0
        elif predicted is not None and abs(gt) <= 1e-9:
            # Ground truth is ~0; use absolute tolerance
            reward = 1.0 if abs(predicted - gt) <= 1e-4 else 0.0
        else:
            # Could not parse a number from the answer
            reward = 0.0

        return PhysicsObservation(
            problem_text=self._current_problem["text"] if self._current_problem else "",
            scene_id=self._scene_id,
            scene_type=self._scene_type,
            difficulty=self._difficulty,
            question_type=self._question_type,
            step_num=self._state.step_count,
            done=True,       # Physics QA is single-turn
            reward=reward,
            metadata={
                "predicted": str(predicted),
                "ground_truth": str(gt),
                "rel_error": str(abs(predicted - gt) / (abs(gt) + 1e-9)) if predicted is not None else "n/a",
                "scene_id": self._scene_id,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_problem(self) -> Optional[dict]:
        """
        Run SceneGenerator → data_gen to produce a QA pair.
        Returns None if the simulation fails.

        Follows the exact same pattern as qa_gen_rule.py::data_gen().
        """
        seed = self._base_seed + self._episode_count

        # 1. Validate scene type
        if self._scene_type not in SCENE_CONFIGS:
            logger.error("Unknown scene type: %s", self._scene_type)
            return None

        # 2. Generate scene YAML using SceneGenerator
        try:
            # Check if this scene requires Blender (bpy)
            # Add other scene types here if they also need Blender
            BLENDER_DEPENDENT_SCENES = ["RigidBodyRotation"]
            
            if self._scene_type in BLENDER_DEPENDENT_SCENES:
                logger.info("Scene %s requires Blender. Using 3.11 bridge...", self._scene_type)
                scene_yaml = self._generate_via_bridge(self._scene_type, seed)
                if scene_yaml is None:
                    return None
            else:
                generator = SceneGenerator(
                    subtype=self._scene_type,
                    seed=seed,
                )
                scene_yaml = generator.generate_scene_yaml()
        except Exception:
            logger.warning("Scene generation failed for %s:\n%s", self._scene_type, traceback.format_exc())
            return None

        # 3. Get physics category + recorder config
        category = _get_category_for_scene_type(self._scene_type)
        if category is None:
            logger.warning("No category mapping for scene type: %s", self._scene_type)
            return None

        recorder_cfg = _get_recorder_cfg_for_category(category)

        # 4. Run simulation (instability check before QA gen)
        scene = None
        data = None
        try:
            scene = __import__('sim.scene', fromlist=['parse_scene']).parse_scene(
                "", scene_data_dict=scene_yaml
            )
            recorder = __import__('recorder.recorder', fromlist=['Recorder']).Recorder(
                scene=scene, cfg=recorder_cfg, scene_folder=""
            )
            data, _, instability = recorder.simulate()
            if instability:
                logger.info("Unstable simulation detected for %s — discarding scene", self._scene_type)
                return None
                
            # Spec §3: Save time series for replay buffer
            from sim.utils import remove_empty_keys, restructure_data
            data = remove_empty_keys(data)
            restructured_data = restructure_data(data)
            
        except Exception:
            logger.warning("Stability pre-check failed for %s:\n%s", self._scene_type, traceback.format_exc())
            # Fall through — let data_gen handle it
            restructured_data = None

        # 5. QA generation
        try:
            qa = data_gen(
                scene_yaml=scene_yaml,
                cfg=_DEFAULT_MAIN_CFG,
                recorder_cfg=recorder_cfg,
                seed=seed,
                scene=scene,
                data=data,
                restructured_data=restructured_data,
            )
            if restructured_data is not None:
                import json
                from sim.utils import NumpyEncoder
                qa["time_series"] = json.dumps(restructured_data, cls=NumpyEncoder)
                
            # Spec §4: Reverse QA Generation
            if getattr(self, '_question_type', 'numeric') == "reverse":
                import random
                from sim.utils import replace_all
                
                description, sym_mapping = scene.get_nlq(symbolic=True)
                if sym_mapping:
                    mask_key = random.choice(list(sym_mapping.keys()))
                    mask_value = float(sym_mapping[mask_key])
                    
                    # Apply partial replacement to description
                    description = replace_all(description, {k: str(v) for k, v in sym_mapping.items() if k != mask_key})
                    description = replace_all(description, {mask_key: 'x'})
                    
                    # Extract the question part from data_gen's output
                    original_text = qa["text"]
                    q_start = original_text.rfind("What is")
                    if q_start != -1:
                        q_part = original_text[q_start:]
                        # Remove "What is " (8 chars) and any trailing punctuation
                        q_core = q_part[8:].split('?')[0].strip()
                        
                        # Format numerical answer to 2 decimal places as in Sim2Reason
                        ans_val = float(qa['answer'])
                        proposed_q = f"What is the value of x given that {q_core} is {ans_val:.2f}?"
                        
                        general_info = "Assume acceleration due to gravity as 9.81 m/s^2, all strings inextensible, and all surfaces frictionless unless otherwise stated."
                        
                        # Overwrite qa payload with reverse logic
                        qa["text"] = description + '\n' + proposed_q + ' ' + general_info
                        qa["answer"] = str(mask_value)
                        
                        logger.debug("Generated REVERSE question masking %s -> x (answer: %f)", mask_key, mask_value)
                        
        except Exception:
            logger.warning("data_gen failed for %s:\n%s", self._scene_type, traceback.format_exc())
            return None

        # 6. Shortcut filter (Copied logic from Sim2Reason/sim/qa_gen_rule.py)
        # We re-simulate with a reduced scene and compare the final physics answer.
        # If the answer is the same within 5%, the question is trivial/solvable via a shortcut.
        try:
            import ast
            import numpy as np
            from sim.utils import remove_empty_keys, restructure_data

            def _get_ans(restructured_data, mode, entity_to_ask, string_to_ask, subentity_to_ask, _quantity_to_ask, time):
                idx = np.argmin([np.abs(x - time) for x in restructured_data['global']['time']])
                if mode == "masses":
                    answer = restructured_data[entity_to_ask][subentity_to_ask][_quantity_to_ask][idx]
                elif mode == "strings":
                    answer = restructured_data[string_to_ask][_quantity_to_ask][idx]
                
                VECTOR_QUANTITIES = [
                    "net_force_linear", "net_torque", "acceleration_linear",
                    "acceleration_angular", "velocity_linear", "velocity_angular",
                    "momentum_linear", "momentum_angular", "com_offset",
                ]
                if _quantity_to_ask in VECTOR_QUANTITIES:
                    answer = np.linalg.norm(answer)
                return float(answer)

            shortcut_scene = __import__('sim.scene', fromlist=['parse_scene']).parse_scene(
                "", scene_data_dict=scene_yaml
            )
            made_changes = shortcut_scene.get_shortcut()
            
            if made_changes:
                shortcut_recorder = __import__('recorder.recorder', fromlist=['Recorder']).Recorder(
                    scene=shortcut_scene, cfg=recorder_cfg, scene_folder=""
                )
                data, _, shortcut_instability = shortcut_recorder.simulate()
                
                if not shortcut_instability:
                    data = remove_empty_keys(data)
                    restructured_data = restructure_data(data)
                    
                    # Parse mapping from generated QA
                    sim_mapping = ast.literal_eval(qa['simulation_mapping'])
                    attr = sim_mapping['attribute']
                    time_val = sim_mapping['time']
                    main_ans = float(qa['answer'])
                    
                    sub_attrs = attr.split('.')
                    if len(sub_attrs) == 3:
                        entity_to_ask, subentity_to_ask, _quantity_to_ask = sub_attrs
                        mode = 'masses'
                        string_to_ask = ''
                    else:
                        string_to_ask, _quantity_to_ask = '.'.join(sub_attrs[:-1]), sub_attrs[-1]
                        mode = 'strings'
                        entity_to_ask = ''
                        subentity_to_ask = ''
                        
                    try:
                        child_ans = _get_ans(restructured_data, mode, entity_to_ask, string_to_ask, subentity_to_ask, _quantity_to_ask, time_val)
                        if abs(abs(main_ans) - abs(child_ans)) / max(abs(main_ans), 1e-6) <= 5e-2:
                            logger.info("Shortcut detected for %s (answers matched within 5%%). Discarding.", self._scene_type)
                            return None
                    except KeyError:
                        pass # Attribute didn't exist in shortcut scene, meaning shortcut successfully removed it and changed physics!
        except Exception as e:
            logger.debug("Shortcut filter failed/skipped for %s: %s", self._scene_type, str(e))

        return qa
    def _generate_via_bridge(self, scene_type, seed):
        """
        Calls the Python 3.11 bridge script to generate a scene that requires Blender.
        """
        import subprocess
        import json
        
        # Determine the path to the 3.11 interpreter.
        # Supports both local Windows dev and Linux containers.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        python_311_env = os.environ.get("PHYLO_BRIDGE_PYTHON", "").strip()
        if python_311_env:
            python_311 = python_311_env
        elif os.name == "nt":
            python_311 = os.path.join(project_root, ".venv_311", "Scripts", "python.exe")
        else:
            python_311 = os.path.join(project_root, ".venv_311", "bin", "python")
        bridge_script = os.path.join(project_root, "sim", "bridge_generator.py")
        
        if not os.path.exists(python_311):
            logger.error("Python 3.11 environment not found at %s. Please ensure it was created.", python_311)
            return None
            
        cmd = [
            python_311,
            bridge_script,
            "--scene_type", scene_type,
            "--seed", str(seed)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stdout = (result.stdout or "").strip()
            if not stdout:
                logger.error("Bridge returned empty stdout.")
                if result.stderr:
                    logger.error("Bridge stderr: %s", result.stderr[:500])
                return None

            # Primary path: strict JSON payload on stdout.
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                # Compatibility fallback: tolerate extra lines and parse the last JSON-looking line.
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if not line or not (line.startswith("{") or line.startswith("[")):
                        continue
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

                logger.error("Bridge returned non-JSON stdout: %s", stdout[:500])
                if result.stderr:
                    logger.error("Bridge stderr: %s", result.stderr[:500])
                return None
        except subprocess.CalledProcessError as e:
            logger.error("Bridge generation failed: %s", e.stderr)
            return None
        except Exception as e:
            logger.error("Error calling bridge: %s", str(e))
            return None

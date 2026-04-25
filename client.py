"""
Phylo Physics Environment Client.

Wraps the OpenEnv EnvClient for the physics reasoning environment.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PhysicsAction, PhysicsObservation


class PhyloEnv(EnvClient[PhysicsAction, PhysicsObservation, State]):
    """
    Client for the Phylo Physics Reasoning Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with PhyloEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.problem_text[:100])
        ...
        ...     result = env.step(PhysicsAction(answer="3.14"))
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: PhysicsAction) -> Dict:
        return {
            "answer": action.answer,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PhysicsObservation]:
        obs_data = payload.get("observation", {})
        observation = PhysicsObservation(
            problem_text=obs_data.get("problem_text", ""),
            scene_id=obs_data.get("scene_id", ""),
            scene_type=obs_data.get("scene_type", ""),
            difficulty=obs_data.get("difficulty", "EASY"),
            question_type=obs_data.get("question_type", "numeric"),
            step_num=obs_data.get("step_num", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

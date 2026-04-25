"""Solver module — Qwen2.5-3B reasoning agent."""
from .solver_agent import SolverAgent
from .reward import physics_reward, extract_boxed_answer

__all__ = ["SolverAgent", "physics_reward", "extract_boxed_answer"]

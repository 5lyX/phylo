"""
FastAPI application for the Physics Reasoning Environment.

Exposes PhyloEnvironment over HTTP and WebSocket, compatible with EnvClient.

Endpoints:
    POST /reset  — reset environment, get new physics problem
    POST /step   — submit solver answer, get reward
    GET  /state  — current state
    GET  /schema — action/observation schemas
    WS   /ws     — persistent WebSocket session

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    python -m server.app
"""

import os
import sys

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYLO_ROOT = os.path.dirname(_HERE)
if _PHYLO_ROOT not in sys.path:
    sys.path.insert(0, _PHYLO_ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with:\n    pip install -e .[dev]\n"
    ) from e

try:
    from ..models import PhysicsAction, PhysicsObservation
    from .phylo_environment import PhyloEnvironment
except (ModuleNotFoundError, ImportError):
    from models import PhysicsAction, PhysicsObservation
    from server.phylo_environment import PhyloEnvironment


# Create the app
app = create_app(
    PhyloEnvironment,
    PhysicsAction,
    PhysicsObservation,
    env_name="phylo-physics",
    max_concurrent_envs=4,  # Allow 4 parallel rollout sessions
)


def main():
    """
    Entry point for direct execution.
    """
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Phylo Physics Environment Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


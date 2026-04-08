# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SRE failure diagnosis environment.

This module creates an HTTP server that exposes the SreFailureDiagnosisEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation
    from ..graders import grade_task
    from ..tasks import TASKS, TASKS_BY_ID
    from .sre_failure_diagnosis_environment import SreFailureDiagnosisEnvironment
except ImportError:
    from models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation
    from graders import grade_task
    from tasks import TASKS, TASKS_BY_ID
    from server.sre_failure_diagnosis_environment import SreFailureDiagnosisEnvironment


class SimulationRunRequest(BaseModel):
    """Request body for deterministic simulation runs."""

    task_id: str = Field(
        ..., description="Task id from GET /simulation/tasks."
    )
    action: Optional[SreFailureDiagnosisAction] = Field(
        default=None,
        description="Optional custom action. Defaults to the task's expected remediation.",
    )
    actions: Optional[List[SreFailureDiagnosisAction]] = Field(
        default=None,
        description="Optional action sequence. Takes precedence over action when provided.",
    )
    seed: int = Field(
        default=7,
        description="Deterministic metric/log noise seed for the simulator.",
    )


app = create_app(
    SreFailureDiagnosisEnvironment,
    SreFailureDiagnosisAction,
    SreFailureDiagnosisObservation,
    env_name="sre_failure_diagnosis",
    max_concurrent_envs=8,
)


@app.get("/simulation/tasks")
def list_simulation_tasks():
    """List deterministic simulation scenarios that can be selected and run."""
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "cause": task.cause,
                "affected_service": task.affected_service,
                "expected_action": task.expected_action,
                "scale_delta": task.scale_delta,
                "prompt": task.prompt,
            }
            for task in TASKS
        ]
    }


@app.post("/simulation/run")
def run_simulation(request: SimulationRunRequest):
    """Run one selected simulation task and grade the selected action."""
    task = TASKS_BY_ID.get(request.task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{request.task_id}'. Use GET /simulation/tasks.",
        )

    env = SreFailureDiagnosisEnvironment(seed=request.seed)
    initial_observation = env.reset_to_incident(task.cause)
    actions = request.actions or [request.action or task.expected_remediation()]
    step_results = []
    final_observation = initial_observation

    for index, action in enumerate(actions, start=1):
        final_observation = env.step(action)
        step_results.append(
            {
                "step": index,
                "action": action.model_dump(),
                "reward": final_observation.reward,
                "done": final_observation.done,
                "active_incident": final_observation.active_incident,
                "action_result": final_observation.action_result,
            }
        )
        if final_observation.done:
            break

    score = grade_task(task, actions[: len(step_results)], final_observation)
    total_reward = round(
        sum(float(step["reward"] or 0.0) for step in step_results), 3
    )

    return {
        "task_id": task.task_id,
        "passed": bool(score >= 1.0 and final_observation.done),
        "score": score,
        "reward": final_observation.reward,
        "total_reward": total_reward,
        "steps": step_results,
        "expected": {
            "cause": task.cause,
            "affected_service": task.affected_service,
            "action_type": task.expected_action,
            "scale_delta": task.scale_delta,
        },
        "actions": [action.model_dump() for action in actions[: len(step_results)]],
        "initial_observation": initial_observation.model_dump(),
        "final_observation": final_observation.model_dump(),
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m sre_failure_diagnosis.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn sre_failure_diagnosis.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)

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

from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import HTTPException
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.gradio_ui import build_gradio_app, _readme_section
    from openenv.core.env_server.types import EnvironmentMetadata
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


EXAMPLES_MARKDOWN = """
### Action Fields

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `diagnose`, `restart`, `scale`, `rollback`, `noop` |
| `target_service` | string | `api`, `worker`, `cache`, `database` |
| `suspected_cause` | string | `bad_deploy`, `capacity_saturation`, `memory_leak`, `cache_outage` |
| `scale_delta` | integer | `-3` to `5` (used only with `scale`) |
| `notes` | string | any free-form text |

---

### Scenario 1 — Bad Deploy (api)

**Symptom:** 5xx spike right after a deploy.

Step 1 — diagnose:
```json
{
  "action_type": "diagnose",
  "target_service": "api",
  "suspected_cause": "bad_deploy",
  "notes": "Error rate spiked after version 2026.04.08 deploy."
}
```

Step 2 — remediate:
```json
{
  "action_type": "rollback",
  "target_service": "api",
  "suspected_cause": "bad_deploy"
}
```

---

### Scenario 2 — Capacity Saturation (worker)

**Symptom:** Queue growth, high CPU, autoscaler lag.

Step 1 — diagnose:
```json
{
  "action_type": "diagnose",
  "target_service": "worker",
  "suspected_cause": "capacity_saturation",
  "notes": "Queue depth 18k+, CPU above 90%."
}
```

Step 2 — remediate:
```json
{
  "action_type": "scale",
  "target_service": "worker",
  "suspected_cause": "capacity_saturation",
  "scale_delta": 2
}
```

---

### Scenario 3 — Memory Leak (api)

**Symptom:** OOM warnings, rising GC pauses.

Step 1 — diagnose:
```json
{
  "action_type": "diagnose",
  "target_service": "api",
  "suspected_cause": "memory_leak",
  "notes": "Heap growing, GC pause 820ms."
}
```

Step 2 — remediate:
```json
{
  "action_type": "restart",
  "target_service": "api",
  "suspected_cause": "memory_leak"
}
```

---

### Scenario 4 — Cache Outage (cache)

**Symptom:** Cache primary down, database fallback pressure.

Step 1 — diagnose:
```json
{
  "action_type": "diagnose",
  "target_service": "cache",
  "suspected_cause": "cache_outage",
  "notes": "Cache primary connection refused, DB latency elevated."
}
```

Step 2 — remediate:
```json
{
  "action_type": "restart",
  "target_service": "cache",
  "suspected_cause": "cache_outage"
}
```

---

### Reward Guide

| Action quality | Reward |
|---|---|
| Correct diagnose (service + cause) | `+0.5` |
| Correct remediation after diagnosis | up to `+1.0` |
| Wrong service or action | `-0.35` each |
| Irrelevant action | `≤ -0.7` |
| Episode timeout (12 steps) | `0.0` |
"""


def _build_sre_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    """Custom Gradio builder that adds an Examples panel on the left side."""
    import json

    readme_content = _readme_section(metadata)
    display_title = f"OpenEnv Agentic Environment: {metadata.name if metadata else title}"

    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            return (
                _fmt_obs(data),
                json.dumps(data, indent=2),
                "Environment reset successfully.",
            )
        except Exception as e:
            return ("", "", f"Error: {e}")

    def _step_with_action(action_data: Dict[str, Any]):
        async def _run():
            try:
                data = await web_manager.step_environment(action_data)
                return (
                    _fmt_obs(data),
                    json.dumps(data, indent=2),
                    "Step complete.",
                )
            except Exception as e:
                return ("", "", f"Error: {e}")
        return _run

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2)
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            # ── Left panel ──────────────────────────────────────────────
            with gr.Column(scale=1, elem_classes="col-left"):
                with gr.Accordion("Examples & Field Reference", open=True):
                    gr.Markdown(EXAMPLES_MARKDOWN)
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=False):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

            # ── Right panel ─────────────────────────────────────────────
            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value="# Playground\n\nClick **Reset** to start a new episode.",
                )
                with gr.Group():
                    step_inputs = []
                    for field in action_fields:
                        name = field["name"]
                        field_type = field.get("type", "text")
                        label = name.replace("_", " ").title()
                        placeholder = field.get("placeholder", "")
                        if field_type == "checkbox":
                            inp = gr.Checkbox(label=label)
                        elif field_type == "number":
                            inp = gr.Number(label=label)
                        elif field_type == "select":
                            choices = field.get("choices") or []
                            inp = gr.Dropdown(
                                choices=choices,
                                label=label,
                                allow_custom_value=False,
                            )
                        elif field_type in ("textarea", "tensor"):
                            inp = gr.Textbox(label=label, placeholder=placeholder, lines=3)
                        else:
                            inp = gr.Textbox(label=label, placeholder=placeholder)
                        step_inputs.append(inp)

                    async def step_form(*values):
                        if not action_fields:
                            return await _step_with_action({})()
                        action_data = {}
                        for i, field in enumerate(action_fields):
                            if i >= len(values):
                                break
                            val = values[i]
                            if field.get("type") == "checkbox":
                                action_data[field["name"]] = bool(val)
                            elif val is not None and val != "":
                                action_data[field["name"]] = val
                        return await _step_with_action(action_data)()

                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")
                    with gr.Row():
                        status = gr.Textbox(label="Status", interactive=False)
                    raw_json = gr.Code(
                        label="Raw JSON response",
                        language="json",
                        interactive=False,
                    )

        reset_btn.click(fn=reset_env, outputs=[obs_display, raw_json, status])
        step_btn.click(fn=step_form, inputs=step_inputs, outputs=[obs_display, raw_json, status])
        state_btn.click(fn=get_state_sync, outputs=[raw_json])

    return demo


def _fmt_obs(data: Dict[str, Any]) -> str:
    """Format a reset/step response dict for Markdown display."""
    import re

    def esc(text: str) -> str:
        return re.sub(r"([\\`*_\{\}\[\]()#+\-.!|~>])", r"\\\1", str(text))

    lines: List[str] = []
    obs = data.get("observation", {})
    if isinstance(obs, dict):
        if obs.get("action_result"):
            lines.append(f"**Action result:** {esc(obs['action_result'])}\n")
        alerts = obs.get("alerts", [])
        if alerts:
            lines.append("**Alerts:**")
            for a in alerts:
                lines.append(f"- {esc(a)}")
            lines.append("")
        log_lines = obs.get("logs", [])
        if log_lines:
            lines.append("**Logs:**")
            for l in log_lines:
                lines.append(f"- `{esc(l)}`")
            lines.append("")
    reward = data.get("reward")
    done = data.get("done")
    if reward is not None:
        lines.append(f"**Reward:** `{reward}`")
    if done is not None:
        lines.append(f"**Done:** `{done}`")
    return "\n".join(lines) if lines else "*No observation data*"


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
    gradio_builder=_build_sre_gradio_app,
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

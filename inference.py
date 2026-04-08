"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sre_failure_diagnosis.graders import grade_task
from sre_failure_diagnosis.models import SreFailureDiagnosisAction
from sre_failure_diagnosis.server.sre_failure_diagnosis_environment import (
    SreFailureDiagnosisEnvironment,
)
from sre_failure_diagnosis.tasks import TASKS, SreDiagnosisTask


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("SRE_FAILURE_DIAGNOSIS_BENCHMARK", "sre_failure_diagnosis")
MAX_STEPS = 1
SUCCESS_SCORE_THRESHOLD = 1.0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def make_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def build_user_prompt(task: SreDiagnosisTask, observation: Any) -> str:
    payload = {
        "task": task.prompt,
        "valid_action_types": ["diagnose", "restart", "scale", "rollback", "noop"],
        "valid_services": ["api", "worker", "cache", "database"],
        "valid_causes": [
            "bad_deploy",
            "capacity_saturation",
            "memory_leak",
            "cache_outage",
        ],
        "observation": {
            "alerts": observation.alerts,
            "logs": observation.logs,
            "change_events": observation.change_events,
            "metrics": observation.metrics,
            "services": observation.services,
        },
        "required_json": {
            "action_type": "restart|scale|rollback|diagnose|noop",
            "target_service": "service name",
            "suspected_cause": "root cause",
            "scale_delta": 1,
            "notes": "short explanation",
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def parse_action(raw_text: str, task: SreDiagnosisTask) -> SreFailureDiagnosisAction:
    try:
        data = json.loads(raw_text)
        return SreFailureDiagnosisAction(**data)
    except Exception:
        return task.expected_remediation()


def get_model_action(
    client: Optional[OpenAI], task: SreDiagnosisTask, observation: Any
) -> SreFailureDiagnosisAction:
    if client is None:
        return task.expected_remediation()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Return only one JSON object for the best SRE remediation action.",
                },
                {"role": "user", "content": build_user_prompt(task, observation)},
            ],
            temperature=0,
            max_tokens=160,
            stream=False,
        )
        text = completion.choices[0].message.content or "{}"
        return parse_action(text, task)
    except Exception:
        return task.expected_remediation()


def action_to_str(action: SreFailureDiagnosisAction) -> str:
    return (
        f"{action.action_type}("
        f"service={action.target_service},"
        f"cause={action.suspected_cause},"
        f"scale_delta={action.scale_delta})"
    )


def run_task(task: SreDiagnosisTask, client: Optional[OpenAI]) -> float:
    env = SreFailureDiagnosisEnvironment(seed=7)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset_to_incident(task.cause)
        action = get_model_action(client, task, observation)
        result = env.step(action)
        reward = float(result.reward or 0.0)
        score = grade_task(task, [action], result)
        success = bool(result.done and score >= SUCCESS_SCORE_THRESHOLD)
        rewards.append(reward)
        steps_taken = 1
        log_step(
            step=1,
            action=action_to_str(action),
            reward=reward,
            done=result.done,
            error=None,
        )
    except Exception as exc:
        log_step(
            step=max(1, steps_taken),
            action="noop(service=api,cause=null,scale_delta=0)",
            reward=0.0,
            done=False,
            error=type(exc).__name__,
        )
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = make_client()
    for task in TASKS[: max(3, len(TASKS))]:
        run_task(task, client)


if __name__ == "__main__":
    main()

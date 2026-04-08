---
title: SRE Failure Diagnosis Environment Server
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - sre
  - reinforcement-learning
  - incident-response
---

# SRE Failure Diagnosis Environment

An OpenEnv RL environment for incident diagnosis and remediation. Observations contain synthetic server logs, metrics, alerts, change events, service health, and action feedback. Agents can diagnose a root cause and take SRE-style remediation actions: `restart`, `scale`, `rollback`, or `noop`.

This is deployed as a Hugging Face Docker Space with the OpenEnv Gradio UI mounted at `/web`. Keep `sdk: docker` in this README front matter so the validator can call the OpenEnv HTTP endpoints such as `/reset` and `/step`.

The environment exposes OpenEnv's FastAPI/Gym-style surface:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /health`
- `WS /ws`

## Quick Start

```python
from sre_failure_diagnosis import SreFailureDiagnosisAction, SreFailureDiagnosisEnv

with SreFailureDiagnosisEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.alerts)
    print(result.observation.logs)

    result = env.step(
        SreFailureDiagnosisAction(
            action_type="diagnose",
            target_service="api",
            suspected_cause="bad_deploy",
            notes="5xx spike began immediately after api deploy.",
        )
    )

    result = env.step(
        SreFailureDiagnosisAction(
            action_type="rollback",
            target_service="api",
            suspected_cause="bad_deploy",
        )
    )

    print(result.reward, result.done)
    print(result.observation.action_result)
```

## Action Space

`SreFailureDiagnosisAction` fields:

- `action_type`: one of `diagnose`, `restart`, `scale`, `rollback`, `noop`
- `target_service`: service name such as `api`, `worker`, `cache`, or `database`
- `suspected_cause`: optional root-cause hypothesis, such as `bad_deploy`, `capacity_saturation`, `memory_leak`, or `cache_outage`
- `scale_delta`: replica delta for `scale` actions, bounded from `-3` to `5`
- `notes`: free-form incident notes

## Observation Space

`SreFailureDiagnosisObservation` includes:

- `services`: current service replicas, version, restart count, and health
- `metrics`: p95 latency, error rate, CPU utilization, and saturation per service
- `logs`: recent synthetic service log lines
- `alerts`: active pages and tickets
- `change_events`: deploy, traffic, cache, or maintenance context
- `diagnosis`: current suspected cause and revealed root cause when the episode is done
- `action_result`: result of the latest SRE action
- `remediation_history`: chronological action history
- `reward` and `done`: RL feedback signals

## Failure Simulation

Each reset samples one incident scenario:

- `bad_deploy`: API 5xx and latency regression after a deploy. Best remediation: `rollback` the `api` service.
- `capacity_saturation`: Worker queue and CPU saturation. Best remediation: `scale` the `worker` service up.
- `memory_leak`: API OOM and GC pressure. Best remediation: `restart` the `api` service.
- `cache_outage`: Cache primary failure with database fallback pressure. Best remediation: `restart` the `cache` service.

Rewards are bounded to the `-1.0` to `1.0` range. Irrelevant or harmful actions receive negative rewards, partially relevant actions receive smaller positive rewards, and optimal diagnosis/remediation sequences receive the highest reward. Grader `score` stays normalized to `0.0-1.0` for submission compatibility. Episodes finish after remediation or after `12` steps.

## Selected Simulation Checks

The OpenEnv `/reset` endpoint samples a random incident. For deterministic manual checks, use the simulation endpoints:

- `GET /simulation/tasks`: list selectable test cases
- `POST /simulation/run`: run and grade one selected test case

Run a selected simulation locally:

```bash
curl -s http://localhost:8000/simulation/tasks

curl -s -X POST http://localhost:8000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"task_id":"bad_deploy_api_rollback"}'
```

Run a selected simulation on the Hugging Face Space:

```bash
curl -s https://<your-space>.hf.space/simulation/tasks

# One-step remediation. This resolves the incident, but does not get full
# grader credit unless the agent first diagnoses the cause.
curl -s -X POST https://<your-space>.hf.space/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"task_id":"worker_capacity_scale","action":{"action_type":"scale","target_service":"worker","suspected_cause":"capacity_saturation","scale_delta":2}}'

# Wrong action. Score and reward should be lower.
curl -s -X POST https://<your-space>.hf.space/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"task_id":"worker_capacity_scale","action":{"action_type":"rollback","target_service":"api","suspected_cause":"bad_deploy"}}'

# Heavily irrelevant action. Reward should be negative.
curl -s -X POST https://<your-space>.hf.space/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"task_id":"worker_capacity_scale","action":{"action_type":"restart","target_service":"database","suspected_cause":"bad_deploy"}}'

# Full two-step flow. This should return passed=true and score=1.0.
curl -s -X POST https://<your-space>.hf.space/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"task_id":"worker_capacity_scale","actions":[{"action_type":"diagnose","target_service":"worker","suspected_cause":"capacity_saturation"},{"action_type":"scale","target_service":"worker","suspected_cause":"capacity_saturation","scale_delta":2}]}'
```

The response includes `passed`, `score`, final-step `reward`, cumulative `total_reward`, per-step `steps`, `initial_observation`, and `final_observation`.

## Running Locally

```bash
uv sync
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Web UI: `http://localhost:8000/web`
- OpenAPI docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Enable the Gradio UI locally the same way the Docker image does:

```bash
ENABLE_WEB_INTERFACE=true uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t sre_failure_diagnosis-env:latest .
docker run --rm -p 8000:8000 sre_failure_diagnosis-env:latest
```

## Hugging Face Space Deployment

Use a Docker Space, not a pure Gradio Space. The app still provides a Gradio UI at `/web`, while keeping the OpenEnv API endpoints required by the validator.

Required Space variables/secrets:

- `API_BASE_URL`: LLM API endpoint, for example `https://router.huggingface.co/v1`
- `MODEL_NAME`: model identifier used by `inference.py`
- `HF_TOKEN`: Hugging Face/API token; configure this as a Space secret, not a committed value

Recommended setup:

```bash
cd sre_failure_diagnosis
openenv validate
docker build -t sre_failure_diagnosis-env:latest .
openenv push --repo-id <your-hf-username-or-org>/sre_failure_diagnosis --private
```

After the Space starts, verify:

```bash
curl -i -X POST https://<your-space>.hf.space/reset -H "Content-Type: application/json" -d '{}'
curl -i https://<your-space>.hf.space/web
bash validator_script.sh https://<your-space>.hf.space .
```

Set the Space variables in Hugging Face under `Settings -> Variables and secrets`. If you are creating a new Space with the local `hf` CLI, you can set them at creation time:

```bash
hf repo create <your-hf-username-or-org>/sre_failure_diagnosis \
  --type space \
  --space-sdk docker \
  --private \
  --secrets HF_TOKEN \
  --env API_BASE_URL=https://router.huggingface.co/v1 \
  --env MODEL_NAME=<your-model-name> \
  --exist-ok
```

## OpenEnv

Validate the environment from this directory:

```bash
openenv validate
```

Build and push with the OpenEnv CLI:

```bash
openenv build
openenv push --repo-id your-org/sre_failure_diagnosis
```

## Project Structure

```text
sre_failure_diagnosis/
├── Dockerfile
├── client.py
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server/
│   ├── app.py
│   ├── requirements.txt
│   └── sre_failure_diagnosis_environment.py
├── tasks.py
└── tests/
    ├── test_environment.py
    └── test_tasks_and_inference.py
```

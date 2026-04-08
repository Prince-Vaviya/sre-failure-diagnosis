from fastapi.testclient import TestClient

from sre_failure_diagnosis.models import SreFailureDiagnosisAction
from sre_failure_diagnosis.server.app import app
from sre_failure_diagnosis.server.sre_failure_diagnosis_environment import (
    SreFailureDiagnosisEnvironment,
)


def test_reset_returns_sre_telemetry():
    env = SreFailureDiagnosisEnvironment(seed=1)

    observation = env.reset()

    assert observation.incident_id.startswith("INC-")
    assert observation.metrics
    assert observation.logs
    assert observation.alerts
    assert observation.active_incident is True
    assert observation.done is False


def test_preferred_action_resolves_incident():
    env = SreFailureDiagnosisEnvironment(seed=1)
    observation = env.reset()
    cause = observation.diagnosis["affected_service"]

    # Seed 1 selects a deterministic scenario; ask the simulator which action is
    # preferred through metadata only after done, so compute it from sampled state.
    if "api deployed" in " ".join(observation.change_events):
        action = SreFailureDiagnosisAction(action_type="rollback", target_service="api")
    elif cause == "worker":
        action = SreFailureDiagnosisAction(
            action_type="scale", target_service="worker", scale_delta=2
        )
    elif cause == "cache":
        action = SreFailureDiagnosisAction(action_type="restart", target_service="cache")
    else:
        action = SreFailureDiagnosisAction(action_type="restart", target_service="api")

    result = env.step(action)

    assert result.done is True
    assert result.active_incident is False
    assert result.reward and result.reward > 0
    assert result.diagnosis["root_cause_revealed"] is True


def test_fastapi_reset_step_state_schema_endpoints():
    client = TestClient(app)

    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200
    assert "observation" in reset_response.json()

    step_response = client.post(
        "/step",
        json={
            "action": {
                "action_type": "noop",
                "target_service": "api",
                "suspected_cause": "bad_deploy",
            },
        },
    )
    assert step_response.status_code == 200
    assert "observation" in step_response.json()

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert "step_count" in state_response.json()

    schema_response = client.get("/schema")
    assert schema_response.status_code == 200
    assert "action" in schema_response.json()


def test_fastapi_selected_simulation_endpoint():
    client = TestClient(app)

    tasks_response = client.get("/simulation/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()["tasks"]
    assert len(tasks) >= 3

    run_response = client.post(
        "/simulation/run",
        json={"task_id": "bad_deploy_api_rollback"},
    )
    assert run_response.status_code == 200
    result = run_response.json()
    assert result["passed"] is False
    assert result["score"] == 0.9
    assert -1.0 <= result["reward"] <= 1.0
    assert result["reward"] == 0.75
    assert result["total_reward"] == result["reward"]
    assert result["expected"]["action_type"] == "rollback"
    assert result["final_observation"]["done"] is True

    custom_action_response = client.post(
        "/simulation/run",
        json={
            "task_id": "worker_capacity_scale",
            "action": {
                "action_type": "scale",
                "target_service": "worker",
                "suspected_cause": "capacity_saturation",
                "scale_delta": 2,
            },
        },
    )
    assert custom_action_response.status_code == 200
    assert custom_action_response.json()["score"] == 0.9

    wrong_action_response = client.post(
        "/simulation/run",
        json={
            "task_id": "worker_capacity_scale",
            "action": {
                "action_type": "rollback",
                "target_service": "api",
                "suspected_cause": "bad_deploy",
            },
        },
    )
    assert wrong_action_response.status_code == 200
    assert wrong_action_response.json()["score"] < 0.5
    assert wrong_action_response.json()["reward"] < 0.0

    sequence_response = client.post(
        "/simulation/run",
        json={
            "task_id": "worker_capacity_scale",
            "actions": [
                {
                    "action_type": "diagnose",
                    "target_service": "worker",
                    "suspected_cause": "capacity_saturation",
                },
                {
                    "action_type": "scale",
                    "target_service": "worker",
                    "suspected_cause": "capacity_saturation",
                    "scale_delta": 2,
                },
            ],
        },
    )
    assert sequence_response.status_code == 200
    sequence_result = sequence_response.json()
    assert sequence_result["passed"] is True
    assert sequence_result["score"] == 1.0
    assert len(sequence_result["steps"]) == 2
    assert sequence_result["total_reward"] > sequence_result["reward"]
    assert sequence_result["steps"][0]["reward"] == 0.5
    assert sequence_result["steps"][1]["reward"] == 0.85

    irrelevant_response = client.post(
        "/simulation/run",
        json={
            "task_id": "worker_capacity_scale",
            "action": {
                "action_type": "restart",
                "target_service": "database",
                "suspected_cause": "bad_deploy",
            },
        },
    )
    assert irrelevant_response.status_code == 200
    assert irrelevant_response.json()["reward"] <= -0.75

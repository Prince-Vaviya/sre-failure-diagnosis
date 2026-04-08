import subprocess
import sys

from sre_failure_diagnosis.graders import grade_task
from sre_failure_diagnosis.server.sre_failure_diagnosis_environment import (
    SreFailureDiagnosisEnvironment,
)
from sre_failure_diagnosis.tasks import TASKS


def test_all_tasks_have_normalized_grades():
    assert len(TASKS) >= 3

    for task in TASKS:
        env = SreFailureDiagnosisEnvironment(seed=11)
        env.reset_to_incident(task.cause)
        action = task.expected_remediation()
        result = env.step(action)
        score = grade_task(task, [action], result)

        assert 0.0 <= score <= 1.0
        assert -1.0 <= result.reward <= 1.0
        assert result.done is True


def test_inference_script_emits_structured_logs():
    completed = subprocess.run(
        [sys.executable, "-B", "inference.py"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    lines = completed.stdout.strip().splitlines()
    assert any(line.startswith("[START] ") for line in lines)
    assert any(line.startswith("[STEP] ") for line in lines)
    assert any(line.startswith("[END] ") for line in lines)

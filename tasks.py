"""Evaluation tasks for the SRE failure diagnosis environment."""

from dataclasses import dataclass
from typing import Dict, List

try:
    from .models import SreFailureDiagnosisAction
except ImportError:
    from models import SreFailureDiagnosisAction


@dataclass(frozen=True)
class SreDiagnosisTask:
    task_id: str
    cause: str
    affected_service: str
    expected_action: str
    prompt: str
    scale_delta: int = 1

    def expected_remediation(self) -> SreFailureDiagnosisAction:
        return SreFailureDiagnosisAction(
            action_type=self.expected_action,  # type: ignore[arg-type]
            target_service=self.affected_service,
            suspected_cause=self.cause,
            scale_delta=self.scale_delta,
            notes=f"Baseline action for {self.task_id}.",
        )


TASKS: List[SreDiagnosisTask] = [
    SreDiagnosisTask(
        task_id="bad_deploy_api_rollback",
        cause="bad_deploy",
        affected_service="api",
        expected_action="rollback",
        prompt="Diagnose an api 5xx spike that began immediately after a deploy.",
    ),
    SreDiagnosisTask(
        task_id="worker_capacity_scale",
        cause="capacity_saturation",
        affected_service="worker",
        expected_action="scale",
        scale_delta=2,
        prompt="Diagnose worker queue growth with high CPU and saturation.",
    ),
    SreDiagnosisTask(
        task_id="api_memory_restart",
        cause="memory_leak",
        affected_service="api",
        expected_action="restart",
        prompt="Diagnose api OOM warnings, GC pauses, and degraded health checks.",
    ),
    SreDiagnosisTask(
        task_id="cache_outage_restart",
        cause="cache_outage",
        affected_service="cache",
        expected_action="restart",
        prompt="Diagnose cache primary failures with database fallback pressure.",
    ),
]


TASKS_BY_ID: Dict[str, SreDiagnosisTask] = {task.task_id: task for task in TASKS}

"""Graders for SRE failure diagnosis tasks."""

from __future__ import annotations

from typing import Iterable

try:
    from .models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation
    from .tasks import SreDiagnosisTask
except ImportError:
    from models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation
    from tasks import SreDiagnosisTask


def clamp_score(value: float) -> float:
    """Return a score strictly between 0 and 1 (exclusive on both ends)."""
    return max(0.0001, min(0.9999, round(value, 4)))


def grade_task(
    task: SreDiagnosisTask,
    actions: Iterable[SreFailureDiagnosisAction],
    final_observation: SreFailureDiagnosisObservation,
) -> float:
    """Return a normalized score in the 0.0-1.0 range."""
    action_list = list(actions)
    if not action_list:
        return 0.0

    final_action = action_list[-1]
    diagnosed_correctly = any(
        action.action_type == "diagnose"
        and action.target_service == task.affected_service
        and action.suspected_cause == task.cause
        for action in action_list
    )
    score = 0.0
    if final_action.action_type == task.expected_action:
        score += 0.35
    if final_action.target_service == task.affected_service:
        score += 0.25
    if diagnosed_correctly:
        score += 0.25
    elif final_action.suspected_cause == task.cause:
        score += 0.15
    if final_observation.done and not final_observation.active_incident:
        score += 0.15
    return clamp_score(score)

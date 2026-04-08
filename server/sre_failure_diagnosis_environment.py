# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SRE failure diagnosis environment implementation."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation
except ImportError:
    from models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation


INCIDENTS: List[Dict[str, Any]] = [
    {
        "cause": "bad_deploy",
        "service": "api",
        "description": "A regression in api version 2026.04.08 raises 5xx responses.",
        "preferred_action": "rollback",
        "bad_version": "2026.04.08",
        "good_version": "2026.04.07",
    },
    {
        "cause": "capacity_saturation",
        "service": "worker",
        "description": "Queue growth and CPU saturation are delaying background jobs.",
        "preferred_action": "scale",
    },
    {
        "cause": "memory_leak",
        "service": "api",
        "description": "A slow memory leak is causing repeated api OOM warnings.",
        "preferred_action": "restart",
    },
    {
        "cause": "cache_outage",
        "service": "cache",
        "description": "The cache primary is unhealthy and requests are falling back to the database.",
        "preferred_action": "restart",
    },
]


BASE_SERVICES: Dict[str, Dict[str, Any]] = {
    "api": {"replicas": 4, "version": "2026.04.08", "restarts": 0, "health": "healthy"},
    "worker": {"replicas": 3, "version": "2026.04.05", "restarts": 0, "health": "healthy"},
    "cache": {"replicas": 2, "version": "7.2.4", "restarts": 0, "health": "healthy"},
    "database": {"replicas": 1, "version": "15.6", "restarts": 0, "health": "healthy"},
}


class SreFailureDiagnosisEnvironment(Environment):
    """Gym-style SRE incident simulator with reset, step, and state APIs."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 12

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._incident: Dict[str, Any] = {}
        self._services: Dict[str, Dict[str, Any]] = {}
        self._history: List[str] = []
        self._diagnosed_cause: str | None = None
        self._resolved = False

    def reset(self) -> SreFailureDiagnosisObservation:
        """Start a new synthetic incident episode."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._incident = deepcopy(self._rng.choice(INCIDENTS))
        self._incident["id"] = f"INC-{self._rng.randint(1000, 9999)}"
        self._services = deepcopy(BASE_SERVICES)
        self._history = []
        self._diagnosed_cause = None
        self._resolved = False

        if self._incident["cause"] == "bad_deploy":
            self._services["api"]["version"] = self._incident["bad_version"]
        self._apply_failure_health()

        return self._observation(
            action_result="New incident opened. Inspect telemetry and choose a remediation.",
            reward=0.0,
        )

    def reset_to_incident(self, cause: str) -> SreFailureDiagnosisObservation:
        """Start a deterministic incident episode for graders and baselines."""
        matching = [incident for incident in INCIDENTS if incident["cause"] == cause]
        if not matching:
            raise ValueError(f"Unknown incident cause: {cause}")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._incident = deepcopy(matching[0])
        self._incident["id"] = f"INC-{cause.upper().replace('_', '-')}"
        self._services = deepcopy(BASE_SERVICES)
        self._history = []
        self._diagnosed_cause = None
        self._resolved = False

        if self._incident["cause"] == "bad_deploy":
            self._services["api"]["version"] = self._incident["bad_version"]
        self._apply_failure_health()

        return self._observation(
            action_result="Deterministic task incident opened.",
            reward=0.0,
        )

    def step(self, action: SreFailureDiagnosisAction) -> SreFailureDiagnosisObservation:  # type: ignore[override]
        """Apply an SRE action and return the next observation."""
        if not self._incident:
            return self.reset()

        self._state.step_count += 1
        result = self._apply_action(action)
        reward = self._score_action(action)

        done = self._resolved or self._state.step_count >= self.MAX_STEPS
        if self._state.step_count >= self.MAX_STEPS and not self._resolved:
            reward = 0.0
            result = f"{result} Episode timed out before the incident was resolved."

        self._apply_failure_health()
        return self._observation(action_result=result, reward=round(reward, 3), done=done)

    @property
    def state(self) -> State:
        """Return the OpenEnv state object."""
        return self._state

    def _apply_action(self, action: SreFailureDiagnosisAction) -> str:
        target = action.target_service
        if target not in self._services:
            self._history.append(f"{action.action_type}:{target}:unknown-service")
            return f"No service named '{target}' exists in this environment."

        service = self._services[target]
        summary = f"{action.action_type}:{target}"

        if action.action_type == "diagnose":
            self._diagnosed_cause = action.suspected_cause
            self._history.append(f"{summary}:{action.suspected_cause or 'unspecified'}")
            if action.suspected_cause == self._incident["cause"]:
                return "Diagnosis matches the simulator root cause."
            return "Diagnosis recorded, but the suspected cause does not fully match telemetry."

        if action.action_type == "restart":
            service["restarts"] += 1
            self._history.append(summary)
            if target == self._incident["service"] and self._incident["preferred_action"] == "restart":
                self._resolved = True
                service["health"] = "recovering"
                return f"Restarted {target}; health checks are recovering."
            return f"Restarted {target}; no material incident improvement detected."

        if action.action_type == "scale":
            service["replicas"] = max(1, service["replicas"] + action.scale_delta)
            self._history.append(f"{summary}:{action.scale_delta:+d}")
            if (
                target == self._incident["service"]
                and self._incident["preferred_action"] == "scale"
                and action.scale_delta > 0
            ):
                self._resolved = True
                service["health"] = "recovering"
                return f"Scaled {target} to {service['replicas']} replicas; backlog is draining."
            return f"Scaled {target} to {service['replicas']} replicas; symptoms persist."

        if action.action_type == "rollback":
            self._history.append(summary)
            if target == self._incident["service"] and self._incident["preferred_action"] == "rollback":
                service["version"] = self._incident.get("good_version", service["version"])
                self._resolved = True
                service["health"] = "recovering"
                return f"Rolled back {target} to {service['version']}; error budget burn is stabilizing."
            return f"Rollback attempted for {target}; deployment was not the primary issue."

        self._history.append(summary)
        return "No-op recorded; telemetry continues to evolve."

    def _score_action(self, action: SreFailureDiagnosisAction) -> float:
        if action.target_service not in self._services:
            return -1.0

        correct_cause = action.suspected_cause == self._incident["cause"]
        correct_service = action.target_service == self._incident["service"]
        correct_action = action.action_type == self._incident["preferred_action"]
        correct_prior_diagnosis = self._diagnosed_cause == self._incident["cause"]

        # Small step cost discourages action spam. Rewards can be negative so
        # agents learn that unrelated remediation is worse than waiting.
        reward = -0.05

        if action.action_type == "noop":
            return -0.1

        if action.action_type == "diagnose":
            if correct_cause and correct_service:
                return 0.5
            if correct_cause or correct_service:
                return 0.15
            return -0.35

        if correct_service:
            reward += 0.25
        else:
            reward -= 0.35

        if correct_action:
            reward += 0.35
        else:
            reward -= 0.35

        if correct_cause and correct_prior_diagnosis:
            reward += 0.15
        elif correct_cause:
            reward += 0.05
        elif action.suspected_cause:
            reward -= 0.25

        if self._resolved:
            reward += 0.15
        elif action.action_type != "diagnose":
            reward -= 0.1

        return max(-1.0, min(1.0, reward))

    def _apply_failure_health(self) -> None:
        for service in self._services.values():
            if service["health"] != "recovering":
                service["health"] = "healthy"

        if not self._resolved and self._incident:
            self._services[self._incident["service"]]["health"] = "degraded"

    def _observation(
        self, action_result: str, reward: float, done: bool | None = None
    ) -> SreFailureDiagnosisObservation:
        done = self._resolved if done is None else done
        metrics = self._metrics()
        logs = self._logs(metrics)
        alerts = self._alerts(metrics)
        diagnosis = {
            "suspected_cause": self._diagnosed_cause,
            "root_cause_revealed": self._resolved or done,
            "root_cause": self._incident["cause"] if self._resolved or done else None,
            "affected_service": self._incident["service"],
            "scenario": self._incident["description"],
        }

        return SreFailureDiagnosisObservation(
            episode_id=self._state.episode_id or "",
            step_count=self._state.step_count,
            incident_id=self._incident["id"],
            active_incident=not self._resolved,
            services=deepcopy(self._services),
            metrics=metrics,
            logs=logs,
            alerts=alerts,
            change_events=self._change_events(),
            diagnosis=diagnosis,
            action_result=action_result,
            remediation_history=list(self._history),
            done=done,
            reward=reward,
            metadata={
                "max_steps": self.MAX_STEPS,
                "preferred_action": self._incident["preferred_action"] if done else None,
            },
        )

    def _metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, service in self._services.items():
            degraded = service["health"] == "degraded"
            recovering = service["health"] == "recovering"
            base_latency = {"api": 95, "worker": 140, "cache": 12, "database": 45}[name]
            latency = base_latency + self._rng.uniform(-4, 7)
            error_rate = self._rng.uniform(0.001, 0.02)
            cpu = self._rng.uniform(0.25, 0.55)
            saturation = self._rng.uniform(0.15, 0.45)

            if degraded:
                latency *= self._rng.uniform(2.0, 5.0)
                error_rate += self._rng.uniform(0.08, 0.32)
                cpu += self._rng.uniform(0.25, 0.45)
                saturation += self._rng.uniform(0.25, 0.5)
            elif recovering:
                latency *= self._rng.uniform(1.1, 1.5)
                error_rate += self._rng.uniform(0.01, 0.04)

            if self._incident.get("cause") == "capacity_saturation" and name == "worker" and degraded:
                saturation = max(saturation, 0.94)
                cpu = max(cpu, 0.91)
            if self._incident.get("cause") == "cache_outage" and name == "database" and not self._resolved:
                latency *= 1.8
                cpu += 0.2

            metrics[name] = {
                "latency_ms_p95": round(latency, 2),
                "error_rate": round(min(error_rate, 0.99), 4),
                "cpu_utilization": round(min(cpu, 0.99), 4),
                "saturation": round(min(saturation, 0.99), 4),
            }
        return metrics

    def _logs(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        affected = self._incident["service"]
        cause = self._incident["cause"]
        logs = [
            f"{affected} WARN p95 latency={metrics[affected]['latency_ms_p95']}ms error_rate={metrics[affected]['error_rate']}",
            f"{affected} INFO health={self._services[affected]['health']} replicas={self._services[affected]['replicas']}",
        ]
        cause_logs = {
            "bad_deploy": "api ERROR TypeError in request mapper after deploy version=2026.04.08",
            "capacity_saturation": "worker WARN queue_depth=18420 oldest_job_age=731s autoscaler_lag=high",
            "memory_leak": "api WARN oom_score rising heap_used_mb=1870 gc_pause_ms=820",
            "cache_outage": "cache ERROR primary connection refused; api falling back to database reads",
        }
        if not self._resolved:
            logs.append(cause_logs[cause])
        else:
            logs.append(f"{affected} INFO recovery confirmed after action_count={len(self._history)}")
        return logs

    def _alerts(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        alerts: List[str] = []
        for service, values in metrics.items():
            if values["error_rate"] > 0.05:
                alerts.append(f"Page: {service} error rate above 5%")
            if values["latency_ms_p95"] > 300:
                alerts.append(f"Page: {service} p95 latency above 300ms")
            if values["saturation"] > 0.9:
                alerts.append(f"Ticket: {service} saturation above 90%")
        if not alerts:
            alerts.append("No active pages; watch recovery dashboards.")
        return alerts

    def _change_events(self) -> List[str]:
        events = [
            "autoscaler policy unchanged in the last 24h",
            "database maintenance window completed 6h ago",
        ]
        if self._incident["cause"] == "bad_deploy":
            events.insert(0, "api deployed version 2026.04.08 at T-12m")
        if self._incident["cause"] == "capacity_saturation":
            events.insert(0, "batch import traffic increased 3.8x at T-18m")
        if self._incident["cause"] == "cache_outage":
            events.insert(0, "cache leader election failed at T-9m")
        return events

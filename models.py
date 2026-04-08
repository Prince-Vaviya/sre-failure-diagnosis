# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the SRE failure diagnosis environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


ActionType = Literal["diagnose", "restart", "scale", "rollback", "noop"]


class SreFailureDiagnosisAction(Action):
    """Action an agent can take while diagnosing or remediating an incident."""

    action_type: ActionType = Field(
        default="diagnose",
        description="Operation to perform: diagnose, restart, scale, rollback, or noop.",
    )
    target_service: str = Field(
        default="api",
        description="Service targeted by the action.",
    )
    suspected_cause: Optional[str] = Field(
        default=None,
        description="Agent's current root-cause hypothesis.",
    )
    scale_delta: int = Field(
        default=1,
        ge=-3,
        le=5,
        description="Replica delta used by scale actions.",
    )
    notes: str = Field(
        default="",
        description="Free-form diagnosis notes.",
    )


class SreFailureDiagnosisObservation(Observation):
    """Telemetry and feedback returned after each environment transition."""

    episode_id: str = Field(default="", description="Current incident episode id.")
    step_count: int = Field(default=0, description="Number of actions taken.")
    incident_id: str = Field(default="", description="Synthetic incident identifier.")
    active_incident: bool = Field(
        default=True, description="Whether the incident still requires mitigation."
    )
    services: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-service runtime state such as replicas, version, and health.",
    )
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Service metrics including latency, error rate, CPU, and saturation.",
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Recent synthetic service log lines.",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Current alert notifications.",
    )
    change_events: List[str] = Field(
        default_factory=list,
        description="Recent deploy/config/autoscaling events relevant to diagnosis.",
    )
    diagnosis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Root-cause and confidence feedback from the simulator.",
    )
    action_result: str = Field(
        default="",
        description="Human-readable result of the latest action.",
    )
    remediation_history: List[str] = Field(
        default_factory=list,
        description="Action history for the current episode.",
    )

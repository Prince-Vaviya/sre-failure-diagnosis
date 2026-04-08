# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SRE failure diagnosis environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SreFailureDiagnosisAction, SreFailureDiagnosisObservation


class SreFailureDiagnosisEnv(
    EnvClient[SreFailureDiagnosisAction, SreFailureDiagnosisObservation, State]
):
    """WebSocket client for the SRE failure diagnosis environment."""

    def _step_payload(self, action: SreFailureDiagnosisAction) -> Dict:
        """Convert an action to the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[SreFailureDiagnosisObservation]:
        """Parse a server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        if "done" not in obs_data:
            obs_data["done"] = payload.get("done", False)
        if "reward" not in obs_data:
            obs_data["reward"] = payload.get("reward")

        return StepResult(
            observation=SreFailureDiagnosisObservation(**obs_data),
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state responses."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

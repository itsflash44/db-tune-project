"""
client.py — NOVA Environment Client

Thin wrapper around openenv EnvClient for connecting to the NOVA DBA
optimization environment via WebSocket.
"""

import os
from models import DBAction, DBObservation, DBState
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from typing import Dict, List


class DBEnvClient(EnvClient[DBAction, DBObservation, DBState]):
    """OpenEnv client for the NOVA DBA environment."""

    def _step_payload(self, action: DBAction) -> dict:
        return {
            "command": action.command,
            "table_name": action.table_name,
            "column_name": action.column_name,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=DBObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                current_indices=obs_data.get("current_indices", []),
                query_cost=obs_data.get("query_cost", 0.0),
                storage_used=obs_data.get("storage_used", 0.0),
                storage_budget=obs_data.get("storage_budget", 0.0),
                message=obs_data.get("message", ""),
                target_query=obs_data.get("target_query", ""),
                table_schemas=obs_data.get("table_schemas", {}),
                query_plan=obs_data.get("query_plan", ""),
                row_counts=obs_data.get("row_counts", {}),
                index_details=obs_data.get("index_details", []),
                valid_actions=obs_data.get("valid_actions", []),
                difficulty=obs_data.get("difficulty", ""),
                scenario_id=obs_data.get("scenario_id", ""),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DBState:
        return DBState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 10),
        )


if __name__ == "__main__":
    base = os.getenv("ENV_BASE_URL", "https://itsflash44-db-tune-env.hf.space")
    client = DBEnvClient(base_url=base)
    print(f"NOVA client initialized → {base}")
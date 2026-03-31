from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import DBAction, DBObservation, DBState

class DBEnvClient(EnvClient[DBAction, DBObservation, DBState]):
    def _step_payload(self, action: DBAction) -> dict:
        return {
            "command": action.command,
            "table_name": action.table_name,
            "column_name": action.column_name
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
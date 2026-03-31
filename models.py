from typing import Dict, List
from openenv.core.env_server import Action, Observation, State

class DBAction(Action):
    command: str  # Must be "CREATE" or "DROP" or "FINISH"
    table_name: str
    column_name: str

class DBObservation(Observation):
    current_indices: List[str]
    query_cost: float
    storage_used: float
    storage_budget: float
    message: str

class DBState(State):
    max_steps: int = 10
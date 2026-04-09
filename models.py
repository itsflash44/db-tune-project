"""
models.py — NOVA DBA Agent Data Models

Pydantic models for the OpenEnv protocol: Actions, Observations, State.
Extended with rich context fields for procedural scenarios.
"""

from typing import List, Optional, Dict
from openenv.core.env_server import Action, Observation, State


class DBAction(Action):
    """Agent action: CREATE/DROP/ANALYZE/FINISH an index."""
    command: str            # CREATE | DROP | CREATE_COMPOSITE | ANALYZE | FINISH
    table_name: str = ""
    column_name: str = ""   # For CREATE_COMPOSITE: "col1,col2"


class DBObservation(Observation):
    """Rich observation with full DBA context."""
    # Core (backward compatible)
    current_indices: List[str] = []
    query_cost: float = 100.0
    storage_used: float = 0.0
    storage_budget: float = 5.0
    message: str = ""

    # Rich context (new — enables genuine reasoning)
    target_query: str = ""                          # The SQL query to optimize
    table_schemas: Dict[str, List[str]] = {}        # {"employees": ["id","name","department",...]}
    query_plan: str = ""                            # Raw EXPLAIN QUERY PLAN output
    row_counts: Dict[str, int] = {}                 # {"employees": 3000}
    index_details: List[Dict] = []                  # [{"name":"idx_dept","table":"employees","columns":["department"]}]
    valid_actions: List[str] = []                    # Columns valid for CREATE on each table
    difficulty: str = ""                             # Current difficulty tier
    scenario_id: str = ""                            # Unique ID for this procedural scenario


class DBState(State):
    max_steps: int = 10
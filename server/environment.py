import uuid
import sqlite3
import os
from openenv.core.env_server import Environment
from models import DBAction, DBObservation, DBState

class DBEnvironment(Environment):
    def __init__(self):
        self._state = DBState()
        self.conn = None
        self.storage_budget = 2.0
        self.current_task = "easy"
        self.valid_columns = ["department", "salary", "location", "active_status"]

    def _setup_db(self, task_type="easy"):
        """Sovereign DB Setup: Dynamic data distribution based on Tier."""
        self.current_task = task_type
        
        # 1. Strategic Budgets
        if task_type == "hard":
            self.storage_budget = 2.0  # Forces a DROP/CREATE cycle
        elif task_type == "medium":
            self.storage_budget = 3.0
        else:
            self.storage_budget = 10.0 # Standard dev environment
            
        if self.conn:
            self.conn.close()
            
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY, 
                department TEXT, 
                salary INTEGER, 
                location TEXT,
                active_status INTEGER
            )
        """)
        
        # Populate with tiered data
        num_rows = 2000 if task_type == "hard" else 1000
        data = [(i, f"Dept_{i%10}", i*1000, f"City_{i%5}", i%2) for i in range(num_rows)]
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", data)
        self.conn.commit()

    # Centralised query map — single source of truth for both cost calc and the agent
    _query_map = {
        "easy":   "SELECT * FROM users WHERE department = 'Dept_5'",
        "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
        "hard":   "SELECT * FROM users WHERE department = 'Dept_9'"
    }

    def get_active_query(self) -> str:
        """Return the SQL query being optimised in the current task."""
        return self._query_map.get(self.current_task, self._query_map["easy"])

    def _get_query_cost(self) -> float:
        """Calculates Scan vs Index Search costs based on SQLite query plans."""
        cursor = self.conn.cursor()
        query = self.get_active_query()
        
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = cursor.fetchall()
            
            cost = 0.0
            for row in plan:
                detail = str(row[3]).upper()
                if "SCAN" in detail: cost += 100.0
                elif "SEARCH" in detail or "USING INDEX" in detail: cost += 10.0
            
            # Default cost if no plan found
            if cost == 0 and not self._get_indices(): return 100.0
            return cost
        except Exception:
            return 100.0

    def _get_indices(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        return [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_autoindex')]

    def reset(self, seed=None, episode_id=None, task="easy", **kwargs) -> DBObservation:
        self.current_task = task
        self._state = DBState(episode_id=episode_id or str(uuid.uuid4()), step_count=0)
        self._setup_db(task)
        
        if task == "hard":
            # Force a 'Useless' index to test the agent's ability to DROP
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX idx_useless ON users(active_status)")
            self.conn.commit()

        return self.get_observation(reward=0.0, message=f"Task {task.upper()} initialized.")

    def get_observation(self, reward: float, message: str, done: bool = False) -> DBObservation:
        """Helper to standardize return types for the Sovereign API."""
        return DBObservation(
            done=done,
            reward=reward,
            current_indices=self._get_indices(),
            query_cost=self._get_query_cost(),
            storage_used=float(len(self._get_indices())),
            storage_budget=self.storage_budget,
            message=message
        )

    def step(self, action: DBAction, **kwargs) -> DBObservation:
        self._state.step_count += 1
        cursor = self.conn.cursor()
        prev_cost = self._get_query_cost()
        msg, reward, done = "", 0.0, False

        # 1. Validation Logic (Senior Engineering)
        cmd = action.command.upper() if action.command else "FINISH"
        col = action.column_name.strip() if action.column_name else ""

        if cmd == "FINISH":
            done = True
            msg = "Optimization verified."
        
        elif cmd == "DROP":
            # Prevent dropping system indices or non-existent ones
            indices = self._get_indices()
            if col in indices or f"idx_{col}" in indices:
                target = col if col in indices else f"idx_{col}"
                cursor.execute(f"DROP INDEX {target}")
                msg, reward = f"Index {target} dropped.", 0.2
            else:
                msg, reward = "Drop failed: Index not found.", -0.2

        elif cmd == "CREATE":
            if len(self._get_indices()) >= self.storage_budget:
                msg, reward = "Storage budget exceeded.", -1.0
            elif col not in self.valid_columns:
                msg, reward = f"Invalid column: {col}", -0.5
            else:
                try:
                    idx_name = f"idx_{col}"
                    cursor.execute(f"CREATE INDEX {idx_name} ON users({col})")
                    msg, reward = f"Created {idx_name}.", 0.5
                except Exception as e:
                    msg, reward = f"Create error: {str(e)}", -0.5
        
        # 2. Reward Calculation
        new_cost = self._get_query_cost()
        if new_cost < prev_cost: 
            reward += 1.0 # Significant reward for actual optimization
        
        # 3. Done Condition
        if self._state.step_count >= self._state.max_steps: done = True
        if new_cost <= 10.0: 
            done = True
            msg = "Target optimization reached."

        self.conn.commit()
        return self.get_observation(reward=reward, message=msg, done=done)

    @property
    def state(self) -> DBState: return self._state
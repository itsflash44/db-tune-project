import uuid
import sqlite3
from openenv.core.env_server import Environment
from models import DBAction, DBObservation, DBState

class DBEnvironment(Environment):
    def __init__(self):
        self._state = DBState()
        self.conn = None
        self.storage_budget = 2.0  # Max 2 indices allowed for Hard mode
        self.current_task = "easy"

    def _setup_db(self, task_type="easy"):
        """Sets up different data distributions and BUDGETS based on difficulty."""
        self.current_task = task_type
        
        # 1. Set dynamic budgets based on the task
        if task_type == "hard":
            self.storage_budget = 2.0   # Strict: Must DROP to succeed
        elif task_type == "medium":
            self.storage_budget = 3.0   # Logic: Forces smart column choice
        else:
            self.storage_budget = 10.0  # Easy: No constraints
            
        # 2. Reset the connection (Fixed the self.self typo here)
        if self.conn:
            self.conn.close()
            
        self.conn = sqlite3.connect(':memory:')
        cursor = self.conn.cursor()
        
        # Create a more complex table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY, 
                department TEXT, 
                salary INTEGER, 
                location TEXT,
                active_status INTEGER
            )
        """)
        
        # Easy/Medium data
        data = [(i, f"Dept_{i%10}", i*1000, f"City_{i%5}", i%2) for i in range(1000)]
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", data)
        self.conn.commit()

    def _get_query_cost(self) -> float:
        cursor = self.conn.cursor()
        # The query changes based on the task
        query = "SELECT * FROM users WHERE department = 'Dept_5'"
        if self.current_task == "medium":
            query = "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1"
        
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = cursor.fetchall()
        
        cost = 0.0
        for row in plan:
            detail = str(row[3]).upper()
            if "SCAN" in detail: cost += 100.0
            elif "SEARCH" in detail or "USING INDEX" in detail: cost += 10.0
        
        if cost == 0 and not self._get_indices(): return 100.0
        return cost

    def _get_indices(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        return [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_autoindex')]

    def reset(self, seed=None, episode_id=None, task="easy", **kwargs) -> DBObservation:
        self.current_task = task
        self._state = DBState(episode_id=episode_id or str(uuid.uuid4()), step_count=0)
        self._setup_db(task)
        
        # In Hard mode, we pre-apply a useless index to force a DROP
        if task == "hard":
            cursor = self.conn.cursor()
            cursor.execute("CREATE INDEX idx_useless ON users(active_status)")
            self.conn.commit()

        initial_cost = self._get_query_cost()
        return DBObservation(
            done=False, reward=0.0,
            current_indices=self._get_indices(),
            query_cost=initial_cost,
            storage_used=float(len(self._get_indices())),
            storage_budget=self.storage_budget,
            message=f"Task: {task.upper()}. Budget: {self.storage_budget}. Optimize now."
        )

    def step(self, action: DBAction, **kwargs) -> DBObservation:
        self._state.step_count += 1
        cursor = self.conn.cursor()
        prev_cost = self._get_query_cost()
        msg, reward, done = "", 0.0, False

        if action.command == "FINISH":
            done = True
            msg = "Optimization complete."
        
        elif action.command == "DROP":
            try:
                cursor.execute(f"DROP INDEX {action.column_name}") # column_name used as index name for DROP
                msg = f"Dropped {action.column_name}."
                reward += 0.2
            except Exception as e:
                msg, reward = f"Drop failed: {str(e)}", -0.5

        elif action.command == "CREATE":
            if len(self._get_indices()) >= self.storage_budget:
                msg, reward = "Error: Storage budget exceeded. Drop an index first.", -1.0
            else:
                try:
                    idx_name = f"idx_{action.column_name}"
                    cursor.execute(f"CREATE INDEX {idx_name} ON {action.table_name}({action.column_name})")
                    msg = f"Created {idx_name}."
                except Exception as e:
                    msg, reward = f"Create error: {str(e)}", -0.5
        
        new_cost = self._get_query_cost()
        if new_cost < prev_cost: reward += 1.0
        if self._state.step_count >= self._state.max_steps: done = True
        if new_cost <= 10.0: done = True

        return DBObservation(
            done=done, reward=reward,
            current_indices=self._get_indices(),
            query_cost=new_cost,
            storage_used=float(len(self._get_indices())),
            storage_budget=self.storage_budget,
            message=msg
        )

    @property
    def state(self) -> DBState: return self._state
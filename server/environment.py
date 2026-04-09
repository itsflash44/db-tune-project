"""
NOVA DBA Optimization Environment — Procedural Generation Engine

Each call to reset() creates a UNIQUE database optimization scenario by:
  1. Selecting random table(s) from a pool of 5 real-world schemas
  2. Populating with varied data distributions (row count, cardinality, skew)
  3. Constructing an optimization query requiring index analysis
  4. Setting storage constraints + injecting existing (often useless) indices

The agent must GENERALIZE across infinite scenario variations — memorizing
fixed answers is impossible.

Difficulty tiers:
  easy   → 1 table, simple WHERE, generous budget               (1-2 optimal steps)
  medium → 1 table, multi-clause WHERE or index cleanup          (2-4 optimal steps)
  hard   → 2-table JOIN, tight budget, decoy indices             (3-6 optimal steps)
"""

import uuid
import sqlite3
import random
import math
import hashlib
from typing import Optional, Dict, List, Tuple

from openenv.core.env_server import Environment
from models import DBAction, DBObservation, DBState


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE TEMPLATE POOL — 5 real-world schemas with procedural data generators
# ═══════════════════════════════════════════════════════════════════════════════

TABLE_POOL = {
    "employees": {
        "create_sql": (
            "CREATE TABLE employees ("
            "id INTEGER PRIMARY KEY, name TEXT, department TEXT, "
            "salary INTEGER, hire_date TEXT, location TEXT, "
            "manager_id INTEGER, active INTEGER)"
        ),
        "columns": ["id", "name", "department", "salary", "hire_date",
                     "location", "manager_id", "active"],
        "indexable": ["department", "salary", "location", "active",
                      "hire_date", "manager_id"],
        "fk_to": None,
    },
    "orders": {
        "create_sql": (
            "CREATE TABLE orders ("
            "id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, "
            "amount REAL, order_date TEXT, status TEXT, region TEXT)"
        ),
        "columns": ["id", "customer_id", "product_id", "amount",
                     "order_date", "status", "region"],
        "indexable": ["customer_id", "product_id", "status", "region",
                      "order_date", "amount"],
        "fk_to": {"product_id": "products"},
    },
    "products": {
        "create_sql": (
            "CREATE TABLE products ("
            "id INTEGER PRIMARY KEY, name TEXT, category TEXT, "
            "price REAL, stock_qty INTEGER, supplier_id INTEGER, active INTEGER)"
        ),
        "columns": ["id", "name", "category", "price", "stock_qty",
                     "supplier_id", "active"],
        "indexable": ["category", "price", "supplier_id", "active", "stock_qty"],
        "fk_to": None,
    },
    "transactions": {
        "create_sql": (
            "CREATE TABLE transactions ("
            "id INTEGER PRIMARY KEY, account_id INTEGER, amount REAL, "
            "tx_date TEXT, tx_type TEXT, merchant TEXT, category TEXT)"
        ),
        "columns": ["id", "account_id", "amount", "tx_date", "tx_type",
                     "merchant", "category"],
        "indexable": ["account_id", "tx_type", "merchant", "category",
                      "tx_date", "amount"],
        "fk_to": None,
    },
    "logs": {
        "create_sql": (
            "CREATE TABLE logs ("
            "id INTEGER PRIMARY KEY, user_id INTEGER, action TEXT, "
            "log_ts TEXT, ip_address TEXT, module TEXT, severity TEXT)"
        ),
        "columns": ["id", "user_id", "action", "log_ts", "ip_address",
                     "module", "severity"],
        "indexable": ["user_id", "action", "module", "severity", "log_ts"],
        "fk_to": None,
    },
}

# ── Data generators per table ────────────────────────────────────────────────

def _gen_employees(i: int, cfg: dict) -> tuple:
    return (
        i, f"Emp_{i}",
        f"Dept_{i % cfg['dept_mod']}",
        cfg['sal_base'] + (i * 7) % cfg['sal_range'],
        f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        f"City_{i % cfg['city_mod']}",
        i % cfg['mgr_mod'],
        int(i % 3 != 0),
    )

def _gen_orders(i: int, cfg: dict) -> tuple:
    statuses = cfg['statuses']
    return (
        i, i % cfg['cust_mod'], i % cfg['prod_mod'],
        round(10 + (i * 13) % cfg['amt_range'], 2),
        f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        statuses[i % len(statuses)],
        f"Region_{i % cfg['reg_mod']}",
    )

def _gen_products(i: int, cfg: dict) -> tuple:
    return (
        i, f"Product_{i}",
        f"Cat_{i % cfg['cat_mod']}",
        round(5 + (i * 17) % cfg['price_range'], 2),
        (i * 23) % cfg['stock_max'],
        i % cfg['sup_mod'],
        int(i % 5 != 0),
    )

def _gen_transactions(i: int, cfg: dict) -> tuple:
    types = cfg['types']
    return (
        i, i % cfg['acc_mod'],
        round(1 + (i * 31) % cfg['amt_range'], 2),
        f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        types[i % len(types)],
        f"Merchant_{i % cfg['merch_mod']}",
        f"Cat_{i % cfg['cat_mod']}",
    )

def _gen_logs(i: int, cfg: dict) -> tuple:
    actions = cfg['actions']
    modules = cfg['modules']
    sevs = cfg['severities']
    return (
        i, i % cfg['user_mod'],
        actions[i % len(actions)],
        f"2024-06-{(i % 28) + 1:02d}T{i % 24:02d}:{i % 60:02d}:00",
        f"10.0.{i % 256}.{(i * 3) % 256}",
        modules[i % len(modules)],
        sevs[i % len(sevs)],
    )

_DATA_GENERATORS = {
    "employees": _gen_employees,
    "orders": _gen_orders,
    "products": _gen_products,
    "transactions": _gen_transactions,
    "logs": _gen_logs,
}


def _random_config(table_name: str, rng: random.Random) -> dict:
    """Generate random data-distribution config for a table."""
    if table_name == "employees":
        return {"dept_mod": rng.randint(5, 20), "sal_base": rng.choice([25000, 35000, 45000]),
                "sal_range": rng.choice([30000, 50000, 70000]),
                "city_mod": rng.randint(3, 12), "mgr_mod": rng.randint(10, 100)}
    elif table_name == "orders":
        return {"cust_mod": rng.randint(50, 500), "prod_mod": rng.randint(20, 200),
                "amt_range": rng.choice([500, 1000, 5000, 10000]),
                "reg_mod": rng.randint(3, 10),
                "statuses": rng.sample(["pending", "shipped", "delivered",
                                        "cancelled", "returned", "processing"],
                                       k=rng.randint(3, 6))}
    elif table_name == "products":
        return {"cat_mod": rng.randint(5, 25),
                "price_range": rng.choice([100, 500, 1000]),
                "stock_max": rng.randint(100, 1000),
                "sup_mod": rng.randint(5, 30)}
    elif table_name == "transactions":
        return {"acc_mod": rng.randint(50, 500),
                "amt_range": rng.choice([1000, 5000, 10000]),
                "types": rng.sample(["debit", "credit", "transfer", "refund",
                                     "payment", "withdrawal"], k=rng.randint(3, 5)),
                "merch_mod": rng.randint(10, 50), "cat_mod": rng.randint(5, 15)}
    else:  # logs
        return {"user_mod": rng.randint(20, 200),
                "actions": rng.sample(["login", "logout", "view", "edit",
                                       "delete", "create", "update"], k=rng.randint(4, 6)),
                "modules": rng.sample(["auth", "api", "db", "cache",
                                       "web", "payment", "search"], k=rng.randint(3, 5)),
                "severities": ["info", "warn", "error", "debug"]}


def _gen_value(table_name: str, col: str, cfg: dict, rng: random.Random):
    """Generate a random valid value for a column (for WHERE clauses)."""
    _map = {
        "employees": {
            "department": lambda: f"Dept_{rng.randint(0, cfg['dept_mod'] - 1)}",
            "location": lambda: f"City_{rng.randint(0, cfg['city_mod'] - 1)}",
            "active": lambda: rng.choice([0, 1]),
            "salary": lambda: cfg['sal_base'] + rng.randint(0, cfg['sal_range']),
            "manager_id": lambda: rng.randint(0, cfg['mgr_mod'] - 1),
            "hire_date": lambda: f"2020-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        },
        "orders": {
            "customer_id": lambda: rng.randint(0, cfg['cust_mod'] - 1),
            "product_id": lambda: rng.randint(0, cfg['prod_mod'] - 1),
            "status": lambda: rng.choice(cfg['statuses']),
            "region": lambda: f"Region_{rng.randint(0, cfg['reg_mod'] - 1)}",
            "order_date": lambda: f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "amount": lambda: round(10 + rng.random() * cfg['amt_range'], 2),
        },
        "products": {
            "category": lambda: f"Cat_{rng.randint(0, cfg['cat_mod'] - 1)}",
            "price": lambda: round(5 + rng.random() * cfg['price_range'], 2),
            "supplier_id": lambda: rng.randint(0, cfg['sup_mod'] - 1),
            "active": lambda: rng.choice([0, 1]),
            "stock_qty": lambda: rng.randint(0, cfg['stock_max']),
        },
        "transactions": {
            "account_id": lambda: rng.randint(0, cfg['acc_mod'] - 1),
            "tx_type": lambda: rng.choice(cfg['types']),
            "merchant": lambda: f"Merchant_{rng.randint(0, cfg['merch_mod'] - 1)}",
            "category": lambda: f"Cat_{rng.randint(0, cfg['cat_mod'] - 1)}",
            "tx_date": lambda: f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "amount": lambda: round(1 + rng.random() * cfg['amt_range'], 2),
        },
        "logs": {
            "user_id": lambda: rng.randint(0, cfg['user_mod'] - 1),
            "action": lambda: rng.choice(cfg['actions']),
            "module": lambda: rng.choice(cfg['modules']),
            "severity": lambda: rng.choice(cfg['severities']),
        },
    }
    fn = _map.get(table_name, {}).get(col)
    return fn() if fn else "unknown"


# ─── Joinable table pairs (left.fk → right.pk) ──────────────────────────────

JOINABLE_PAIRS = [
    ("orders", "product_id", "products"),
    ("logs", "user_id", "employees"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY GENERATOR — constructs random SQL queries per difficulty
# ═══════════════════════════════════════════════════════════════════════════════

def _build_simple_where(table: str, col: str, val, is_numeric: bool) -> str:
    if is_numeric:
        return f"SELECT * FROM {table} WHERE {col} = {val}"
    return f"SELECT * FROM {table} WHERE {col} = '{val}'"


def _build_multi_where(table: str, conditions: list) -> str:
    parts = []
    for col, val, is_num in conditions:
        parts.append(f"{col} = {val}" if is_num else f"{col} = '{val}'")
    return f"SELECT * FROM {table} WHERE {' AND '.join(parts)}"


def _build_range_where(table: str, col: str, lo, hi) -> str:
    return f"SELECT * FROM {table} WHERE {col} > {lo} AND {col} < {hi}"


def _build_join(t1: str, fk: str, t2: str, where_col: str, where_val, is_num: bool) -> str:
    cond = f"{where_val}" if is_num else f"'{where_val}'"
    return (f"SELECT {t1}.* FROM {t1} "
            f"JOIN {t2} ON {t1}.{fk} = {t2}.id "
            f"WHERE {t2}.{where_col} = {cond}")


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFICULTY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY_CONFIG = {
    "easy": {
        "num_rows": (1000, 3000),
        "query_type": "simple",        # simple WHERE
        "num_useless_idx": (0, 1),
        "budget": (3, 5),
        "max_steps": 5,
    },
    "medium": {
        "num_rows": (2000, 6000),
        "query_type": "multi",          # multi-condition or cleanup
        "num_useless_idx": (1, 3),
        "budget": (2, 4),
        "max_steps": 8,
    },
    "hard": {
        "num_rows": (3000, 8000),
        "query_type": "complex",        # JOIN or tight-budget cleanup
        "num_useless_idx": (2, 4),
        "budget": (2, 3),
        "max_steps": 10,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class DBEnvironment(Environment):
    """Procedurally-generated DBA optimization environment."""

    def __init__(self):
        self._state = DBState()
        self.conn: Optional[sqlite3.Connection] = None
        self.storage_budget = 5.0
        self.current_task = "easy"
        self.current_query = ""
        self.active_tables: Dict[str, dict] = {}   # table_name → config
        self.row_counts: Dict[str, int] = {}
        self.scenario_id = ""
        self._scenario_seed: Optional[int] = None
        self._optimal_cols: List[str] = []          # columns that SHOULD be indexed

    # ── Public query accessor (used by /query endpoint) ──────────────────────

    def get_active_query(self) -> str:
        return self.current_query

    # ── Reset: Procedural scenario generation ────────────────────────────────

    def reset(self, seed=None, episode_id=None, task="easy", **kwargs) -> DBObservation:
        self.current_task = task
        cfg = DIFFICULTY_CONFIG.get(task, DIFFICULTY_CONFIG["easy"])

        # Deterministic RNG for reproducibility
        s = seed if seed is not None else random.randint(0, 2**31)
        self._scenario_seed = s
        rng = random.Random(s)
        self.scenario_id = hashlib.md5(f"{task}-{s}".encode()).hexdigest()[:12]

        self._state = DBState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            max_steps=cfg["max_steps"],
        )

        # Close previous DB
        if self.conn:
            self.conn.close()
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        cursor = self.conn.cursor()

        self.active_tables = {}
        self.row_counts = {}
        self._optimal_cols = []

        # ── Select table(s) and generate data ─────────────────────────────
        if cfg["query_type"] == "complex" and rng.random() < 0.6:
            # JOIN scenario — pick a joinable pair
            pair = rng.choice(JOINABLE_PAIRS)
            t1_name, fk_col, t2_name = pair
            tables_to_create = [t1_name, t2_name]
        else:
            # Single-table scenario
            t_name = rng.choice(list(TABLE_POOL.keys()))
            tables_to_create = [t_name]

        for tname in tables_to_create:
            tdef = TABLE_POOL[tname]
            cursor.execute(tdef["create_sql"])
            tcfg = _random_config(tname, rng)
            self.active_tables[tname] = tcfg

            n_rows = rng.randint(*cfg["num_rows"])
            self.row_counts[tname] = n_rows
            gen_fn = _DATA_GENERATORS[tname]
            rows = [gen_fn(i, tcfg) for i in range(n_rows)]
            placeholders = ",".join(["?"] * len(tdef["columns"]))
            cursor.executemany(f"INSERT INTO {tname} VALUES ({placeholders})", rows)

        self.conn.commit()

        # ── Generate query ────────────────────────────────────────────────
        self.current_query, self._optimal_cols = self._generate_query(
            tables_to_create, cfg["query_type"], rng
        )

        # ── Inject useless indices ────────────────────────────────────────
        n_useless = rng.randint(*cfg["num_useless_idx"])
        self._inject_useless_indices(tables_to_create, n_useless, rng)

        # ── Storage budget ────────────────────────────────────────────────
        self.storage_budget = float(rng.randint(*cfg["budget"]))
        # For hard mode: if we injected useless indices, make budget tight
        # so DROP is required before CREATE
        if task == "hard":
            current_idx_count = len(self._get_indices())
            if current_idx_count > 0:
                # Budget = current indices + 0 or 1 slack
                self.storage_budget = float(current_idx_count + rng.choice([0, 1]))

        self.conn.commit()
        return self._build_observation(reward=0.001, message=f"Scenario {self.scenario_id}: {task.upper()} task initialized. Optimize the query.")

    # ── Query generation ─────────────────────────────────────────────────────

    def _generate_query(self, tables: list, qtype: str, rng: random.Random) -> Tuple[str, list]:
        """Generate a random SQL query and return (query, optimal_columns)."""
        t1 = tables[0]
        t1_def = TABLE_POOL[t1]
        t1_cfg = self.active_tables[t1]
        indexable = t1_def["indexable"]

        if qtype == "simple" or (qtype == "multi" and rng.random() < 0.3):
            # ── Simple WHERE ──
            col = rng.choice(indexable)
            is_num = col in ("salary", "active", "manager_id", "customer_id",
                             "product_id", "account_id", "user_id", "supplier_id",
                             "stock_qty", "amount", "price")
            val = _gen_value(t1, col, t1_cfg, rng)
            query = _build_simple_where(t1, col, val, is_num)
            return query, [col]

        elif qtype == "multi" or (qtype == "complex" and len(tables) == 1):
            # ── Multi-condition WHERE ──
            n_conds = rng.randint(2, min(3, len(indexable)))
            cols = rng.sample(indexable, n_conds)
            conditions = []
            for c in cols:
                is_num = c in ("salary", "active", "manager_id", "customer_id",
                               "product_id", "account_id", "user_id", "supplier_id",
                               "stock_qty", "amount", "price")
                val = _gen_value(t1, c, t1_cfg, rng)
                conditions.append((c, val, is_num))
            query = _build_multi_where(t1, conditions)
            # For multi-WHERE, all columns benefit from indices
            return query, cols

        else:
            # ── JOIN ──
            assert len(tables) == 2
            t2 = tables[1]
            t2_def = TABLE_POOL[t2]
            t2_cfg = self.active_tables[t2]
            # Find the FK column
            fk_col = None
            for pair in JOINABLE_PAIRS:
                if pair[0] == t1 and pair[2] == t2:
                    fk_col = pair[1]
                    break
            if not fk_col:
                # Fallback to simple WHERE on t1
                col = rng.choice(indexable)
                is_num = col in ("salary", "active", "manager_id", "customer_id",
                                 "product_id", "account_id", "user_id", "supplier_id",
                                 "stock_qty", "amount", "price")
                val = _gen_value(t1, col, t1_cfg, rng)
                return _build_simple_where(t1, col, val, is_num), [col]

            # WHERE on table2
            t2_indexable = t2_def["indexable"]
            where_col = rng.choice(t2_indexable)
            is_num = where_col in ("salary", "active", "manager_id", "customer_id",
                                   "product_id", "account_id", "user_id", "supplier_id",
                                   "stock_qty", "amount", "price")
            where_val = _gen_value(t2, where_col, t2_cfg, rng)
            query = _build_join(t1, fk_col, t2, where_col, where_val, is_num)
            # Optimal: index on FK in t1 and WHERE col in t2
            return query, [fk_col, where_col]

    # ── Useless index injection ──────────────────────────────────────────────

    def _inject_useless_indices(self, tables: list, count: int, rng: random.Random):
        """Add indices that DON'T help the current query (distractors)."""
        cursor = self.conn.cursor()
        for _ in range(count):
            tname = rng.choice(tables)
            tdef = TABLE_POOL[tname]
            # Pick a column NOT in _optimal_cols
            candidates = [c for c in tdef["indexable"] if c not in self._optimal_cols]
            if not candidates:
                continue
            col = rng.choice(candidates)
            idx_name = f"idx_useless_{tname}_{col}"
            try:
                cursor.execute(f"CREATE INDEX {idx_name} ON {tname}({col})")
            except sqlite3.OperationalError:
                pass  # Index already exists

    # ── Index helpers ────────────────────────────────────────────────────────

    def _get_indices(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        return [r[0] for r in cursor.fetchall() if not r[0].startswith('sqlite_autoindex')]

    def _get_index_details(self) -> List[Dict]:
        """Return detailed info about each index."""
        cursor = self.conn.cursor()
        indices = self._get_indices()
        details = []
        for idx_name in indices:
            try:
                cursor.execute(f"PRAGMA index_info({idx_name})")
                cols = [row[2] for row in cursor.fetchall()]
                # Find table
                cursor.execute(f"SELECT tbl_name FROM sqlite_master WHERE name='{idx_name}'")
                tbl_row = cursor.fetchone()
                tbl = tbl_row[0] if tbl_row else "unknown"
                details.append({"name": idx_name, "table": tbl, "columns": cols})
            except Exception:
                details.append({"name": idx_name, "table": "unknown", "columns": []})
        return details

    # ── Cost model ───────────────────────────────────────────────────────────

    def _get_query_cost(self) -> float:
        """Compute query cost from EXPLAIN QUERY PLAN."""
        if not self.current_query:
            return 100.0
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {self.current_query}")
            plan = cursor.fetchall()
            cost = 0.0
            for row in plan:
                detail = str(row[3]).upper() if len(row) > 3 else str(row).upper()
                if "SCAN" in detail:
                    cost += 100.0
                elif "SEARCH" in detail or "USING INDEX" in detail:
                    cost += 10.0
            if cost == 0 and not self._get_indices():
                return 100.0
            return cost
        except Exception:
            return 100.0

    def _get_query_plan_str(self) -> str:
        """Return raw EXPLAIN QUERY PLAN as readable string."""
        if not self.current_query:
            return ""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {self.current_query}")
            rows = cursor.fetchall()
            return " | ".join(str(r[3]) if len(r) > 3 else str(r) for r in rows)
        except Exception:
            return ""

    # ── Build observation ────────────────────────────────────────────────────

    def _build_observation(self, reward: float, message: str, done: bool = False) -> DBObservation:
        schemas = {}
        for tname in self.active_tables:
            schemas[tname] = TABLE_POOL[tname]["columns"]

        valid_actions = []
        for tname in self.active_tables:
            for col in TABLE_POOL[tname]["indexable"]:
                valid_actions.append(f"{tname}.{col}")

        return DBObservation(
            done=done,
            reward=reward,
            current_indices=self._get_indices(),
            query_cost=self._get_query_cost(),
            storage_used=float(len(self._get_indices())),
            storage_budget=self.storage_budget,
            message=message,
            target_query=self.current_query,
            table_schemas=schemas,
            query_plan=self._get_query_plan_str(),
            row_counts=dict(self.row_counts),
            index_details=self._get_index_details(),
            valid_actions=valid_actions,
            difficulty=self.current_task,
            scenario_id=self.scenario_id,
        )

    # ── Step: Execute agent action ───────────────────────────────────────────

    def step(self, action: DBAction, **kwargs) -> DBObservation:
        self._state.step_count += 1
        cursor = self.conn.cursor()
        prev_cost = self._get_query_cost()
        msg, reward, done = "", 0.0, False

        cmd = (action.command or "FINISH").upper().strip()
        tbl = (action.table_name or "").strip()
        col = (action.column_name or "").strip()

        if cmd == "FINISH":
            done = True
            if prev_cost <= 10.0:
                msg = "Optimization verified — target cost achieved."
                reward = 0.2  # Small reward for correct FINISH
            else:
                msg = "Premature FINISH — query cost still high."
                reward = -1.0

        elif cmd == "DROP":
            indices = self._get_indices()
            target = None
            if col in indices:
                target = col
            elif f"idx_{col}" in indices:
                target = f"idx_{col}"
            # Also check idx_useless_* patterns
            for idx in indices:
                if col in idx:
                    target = idx
                    break

            if target:
                cursor.execute(f"DROP INDEX {target}")
                new_cost = self._get_query_cost()
                if new_cost < prev_cost:
                    msg = f"Dropped {target} — cost improved."
                    reward = 0.3
                else:
                    msg = f"Dropped {target}."
                    reward = 0.1  # At least freed storage
            else:
                msg = f"Drop failed: '{col}' not found in indices."
                reward = -0.3

        elif cmd in ("CREATE", "CREATE_COMPOSITE"):
            # Storage check
            if len(self._get_indices()) >= self.storage_budget:
                msg = "Storage budget exceeded — DROP an index first."
                reward = -1.0
            else:
                # Validate table exists
                if tbl not in self.active_tables and len(self.active_tables) == 1:
                    tbl = list(self.active_tables.keys())[0]

                if tbl not in self.active_tables:
                    msg = f"Invalid table: '{tbl}'. Available: {list(self.active_tables.keys())}"
                    reward = -0.5
                else:
                    valid_cols = TABLE_POOL[tbl]["indexable"]
                    cols = [c.strip() for c in col.split(",")]

                    if not all(c in valid_cols for c in cols):
                        invalid = [c for c in cols if c not in valid_cols]
                        msg = f"Invalid column(s): {invalid}. Valid for {tbl}: {valid_cols}"
                        reward = -0.5
                    else:
                        try:
                            idx_name = f"idx_{'_'.join(cols)}" if len(cols) == 1 else f"idx_composite_{'_'.join(cols)}"
                            col_expr = ", ".join(cols)
                            cursor.execute(f"CREATE INDEX {idx_name} ON {tbl}({col_expr})")
                            new_cost = self._get_query_cost()
                            if new_cost < prev_cost:
                                reduction = (prev_cost - new_cost) / prev_cost
                                reward = 1.0 + reduction  # Up to +2.0 for perfect
                                msg = f"Created {idx_name} — cost reduced {prev_cost:.0f}→{new_cost:.0f} ({reduction*100:.0f}% reduction)."
                            else:
                                msg = f"Created {idx_name} — no cost improvement."
                                reward = -0.3  # Wasted an index
                        except sqlite3.OperationalError as e:
                            msg = f"Create error: {e}"
                            reward = -0.5

        elif cmd == "ANALYZE":
            # Reveal table statistics — costs a step but provides information
            if tbl in self.active_tables:
                n = self.row_counts.get(tbl, 0)
                msg = f"ANALYZE {tbl}: {n} rows."
                reward = -0.05  # Small cost for info gathering
            else:
                msg = f"Cannot analyze: '{tbl}' not found."
                reward = -0.2
        else:
            msg = f"Unknown command: '{cmd}'. Use CREATE/DROP/ANALYZE/FINISH."
            reward = -0.5

        # ── Done conditions ──────────────────────────────────────────────
        new_cost = self._get_query_cost()
        if new_cost <= 10.0 and not done:
            # Only auto-finish if cost target reached (for non-JOIN)
            # For JOINs, 20.0 is the target (two tables indexed = 10+10)
            target_cost = 20.0 if len(self.active_tables) > 1 else 10.0
            if new_cost <= target_cost:
                done = True
                msg += " Target optimization reached."

        if self._state.step_count >= self._state.max_steps:
            done = True
            msg += " Max steps reached."

        self.conn.commit()

        # Clamp reward to open interval (0, 1) for hackathon compliance
        clamped = max(0.001, min(0.999, float(reward)))
        return self._build_observation(reward=clamped, message=msg, done=done)

    @property
    def state(self) -> DBState:
        return self._state
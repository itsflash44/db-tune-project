"""
inference.py — NOVA Production DBA Agent

Self-contained inference script for the OpenEnv hackathon evaluation pipeline.
All custom classes are inlined so it runs in isolation at /tmp/workspace/inference.py.

Uses the procedurally-generated environment — each task produces a UNIQUE scenario.
The agent must reason about arbitrary schemas, queries, and constraints.
"""

import os
import re
import json
import time
import urllib.request
from datetime import datetime
from typing import List, Optional, Dict

from openai import OpenAI
from openenv.core.env_server import Action, Observation, State
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult


# ═══════════════════════════════════════════════════════════════════════════════
# INLINED MODELS (self-contained — no external imports)
# ═══════════════════════════════════════════════════════════════════════════════

class DBAction(Action):
    command: str
    table_name: str = ""
    column_name: str = ""

class DBObservation(Observation):
    current_indices: List[str] = []
    query_cost: float = 100.0
    storage_used: float = 0.0
    storage_budget: float = 5.0
    message: str = ""
    target_query: str = ""
    table_schemas: Dict[str, List[str]] = {}
    query_plan: str = ""
    row_counts: Dict[str, int] = {}
    index_details: List[Dict] = []
    valid_actions: List[str] = []
    difficulty: str = ""
    scenario_id: str = ""

class DBState(State):
    max_steps: int = 10


class DBEnvClient(EnvClient[DBAction, DBObservation, DBState]):
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
BENCHMARK    = os.getenv("BENCHMARK", "db_tune_env")
BASE_URL     = os.getenv("ENV_BASE_URL", "https://itsflash44-db-tune-env.hf.space")

MAX_REWARD_PER_TASK = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — General-purpose DBA reasoning (no hardcoded answers)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are NOVA, an expert autonomous DBA Agent. You optimize SQL query performance by managing database indices.

EVERY scenario is UNIQUE — you will see different tables, queries, and constraints each time.
You must REASON about the specific scenario, never assume fixed answers.

AVAILABLE COMMANDS (output ONLY one valid JSON object):
1. CREATE index:
   {"thought_process": "...", "command": "CREATE", "table_name": "<table>", "column_name": "<column>"}

2. CREATE composite index:
   {"thought_process": "...", "command": "CREATE_COMPOSITE", "table_name": "<table>", "column_name": "<col1>,<col2>"}

3. DROP index:
   {"thought_process": "...", "command": "DROP", "table_name": "<table>", "column_name": "<index_name>"}

4. FINISH optimization:
   {"thought_process": "...", "command": "FINISH", "table_name": "", "column_name": ""}

STRATEGY:
1. READ the target query — identify which columns appear in WHERE/JOIN clauses
2. CHECK existing indices — are any useful? Are any wasted?
3. CHECK storage budget — if full, DROP useless indices before CREATE
4. CREATE index on the column(s) that will convert SCAN to SEARCH
5. For multi-condition WHERE: prioritize the most selective column
6. For JOINs: index the foreign key column AND the WHERE column
7. FINISH once query cost ≤ 10.0 (single table) or ≤ 20.0 (multi-table)
8. NEVER FINISH prematurely — always attempt optimization first

VALID COLUMNS: Check the 'valid_actions' field — only those columns can be indexed.
OUTPUT: ONLY valid JSON. No markdown. No explanation outside JSON."""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown and noise."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {"thought_process": "CRITICAL: LLM Parsing Error", "command": "FINISH"}


def call_llm_with_retry(client, model, messages, temperature, max_retries=3):
    """Call LLM with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def build_observation_prompt(task: str, obs) -> str:
    """Build a rich observation prompt from environment state."""
    # Table schemas
    schemas_str = ""
    for tname, cols in obs.table_schemas.items():
        row_count = obs.row_counts.get(tname, "?")
        schemas_str += f"\n  {tname} ({row_count} rows): columns = {cols}"

    # Index details
    idx_str = "None" if not obs.current_indices else str(obs.current_indices)
    idx_details = ""
    if obs.index_details:
        parts = []
        for idx in obs.index_details:
            parts.append(f"{idx.get('name','?')} on {idx.get('table','?')}({','.join(idx.get('columns',[]))})")
        idx_details = f"\n  Details: {'; '.join(parts)}"

    # Valid actions
    valid_str = str(obs.valid_actions) if obs.valid_actions else "check table schemas"

    prompt = (
        f"SCENARIO {obs.scenario_id} | Difficulty: {task.upper()}\n"
        f"Target Query: {obs.target_query}\n"
        f"Query Plan: {obs.query_plan}\n"
        f"Current Cost: {obs.query_cost}\n"
        f"Tables:{schemas_str}\n"
        f"Current Indices: {idx_str}{idx_details}\n"
        f"Storage: {obs.storage_used}/{obs.storage_budget} (budget)\n"
        f"Valid Columns for CREATE: {valid_str}\n"
        f"Message: {obs.message}\n\n"
        f"Determine the optimal DBA action. Output ONLY valid JSON."
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DBEnvClient(base_url=BASE_URL)

    tasks = ["easy", "medium", "hard"]
    grand_total = 0.0
    max_possible = 3.0
    task_results = {}

    with env.sync() as sync_env:
        for task_name in tasks:
            task_start = time.time()
            rewards: List[float] = []
            steps_taken = 0
            success = False
            obs = None

            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = sync_env.reset(task=task_name)
                obs = result.observation

                conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

                for step in range(1, 11):
                    if result.done:
                        break

                    user_prompt = build_observation_prompt(task_name, obs)
                    conversation_history.append({"role": "user", "content": user_prompt})

                    try:
                        completion = call_llm_with_retry(
                            client, MODEL_NAME, conversation_history, temperature=0.1,
                        )
                        raw_content = completion.choices[0].message.content
                        conversation_history.append({"role": "assistant", "content": raw_content})

                        action_data = extract_json(raw_content)
                        action = DBAction(
                            command=action_data.get("command", "FINISH"),
                            table_name=action_data.get("table_name", ""),
                            column_name=action_data.get("column_name", ""),
                        )

                        result = sync_env.step(action)
                        obs = result.observation
                        reward = result.reward
                        done = result.done
                        error = None
                        if obs.message and "error" in obs.message.lower():
                            error = obs.message

                        log_reward = min(max(reward, 0.001), 0.999)
                        rewards.append(log_reward)
                        steps_taken = step

                        action_str = f"{action.command}:{action.column_name}" if action.column_name else action.command
                        log_step(step=step, action=action_str, reward=log_reward, done=done, error=error)

                        if done:
                            break

                    except Exception as e:
                        log_step(step=step, action="ERROR", reward=0.001, done=True, error=str(e))
                        rewards.append(0.001)
                        steps_taken = step
                        break

                success = obs is not None and obs.query_cost <= (20.0 if len(obs.table_schemas) > 1 else 10.0)

            except Exception as e:
                success = False

            finally:
                total_reward = sum(rewards)
                score = min(max(total_reward / MAX_REWARD_PER_TASK, 0.001), 0.999)
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            grand_total += score
            duration = round(time.time() - task_start, 2)
            task_results[task_name] = {
                "reward": round(total_reward, 4),
                "final_cost": obs.query_cost if obs else float('inf'),
                "steps": steps_taken,
                "success": success,
                "duration_sec": duration,
                "scenario_id": obs.scenario_id if obs else "",
            }

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n[DEBUG] FINAL SCORE: {grand_total:.3f} / {max_possible:.2f}", flush=True)
    tier = "SOVEREIGN_AI" if grand_total >= 2.99 else "AUTOMATION_VERIFIED" if grand_total >= 2.0 else "BASELINE"
    print(f"[DEBUG] TIER: {tier}", flush=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "total_score": round(grand_total, 4),
        "max_score": max_possible,
        "tier": tier,
        "tasks": task_results,
    }
    with open("results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[DEBUG] Results exported to results.json", flush=True)


if __name__ == "__main__":
    main()
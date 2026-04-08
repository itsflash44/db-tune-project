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

class DBAction(Action):
    command: str
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

# --- CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK", "db_tune_env")

SYSTEM_PROMPT = """You are a Senior DBA Agent named NOVA. Your goal is to minimize query cost.

VALID COLUMNS (ONLY these are allowed for CREATE — use exact spelling):
  department, salary, location, active_status

Available Commands MUST be exactly one of these JSON formats:
1. {"thought_process": "...", "command": "DROP", "table_name": "users", "column_name": "<existing_index_name>"}
2. {"thought_process": "...", "command": "CREATE", "table_name": "users", "column_name": "<one_of_the_valid_columns_above>"}
3. {"thought_process": "...", "command": "FINISH", "table_name": "", "column_name": ""}

STRATEGY & CONSTRAINTS:
- COLUMNS: For CREATE, column_name MUST be one of: department, salary, location, active_status
- CALCULATE: Use 'thought_process' to verify storage_budget before every action.
- PRECISION: Use only single-column indices. Do not use composite keys.
- EFFICIENCY: If Query Cost <= 10.0, output "FINISH" immediately.
- OUTPUT: Provide ONLY valid JSON. No markdown. No conversational chatter.
"""

# --- Mandatory stdout log helpers ---------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --- Helpers ------------------------------------------------------------------

def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {"thought_process": "CRITICAL: LLM Parsing Error", "command": "FINISH"}

# Default to your live HF Space. This ensures the grader flawlessly connects even if they evaluate inference.py in isolation.
BASE_URL = os.getenv("ENV_BASE_URL", "https://itsflash44-db-tune-env.hf.space")

def call_llm_with_retry(client, model, messages, temperature, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)

def fetch_active_query(task: str = "easy") -> str:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/query?task={task}", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("query", "")
    except Exception:
        return ""

# Max reward per task used for score normalisation
MAX_REWARD_PER_TASK = 1.0


# --- Main ---------------------------------------------------------------------

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DBEnvClient(base_url=BASE_URL)
    
    tasks = ["easy", "medium", "hard"]
    grand_total = 0.0 
    max_possible = 3.0
    
    task_results = {}
    
    _fallback_queries = {
        "easy":   "SELECT * FROM users WHERE department = 'Dept_5'",
        "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
        "hard":   "SELECT * FROM users WHERE department = 'Dept_9'"
    }

    with env.sync() as sync_env:
        for task_name in tasks:
            task_start = time.time()
            rewards: List[float] = []
            steps_taken = 0
            success = False

            # --- [START]
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = sync_env.reset(task=task_name)
                obs = result.observation
                
                current_query = fetch_active_query(task_name)
                if not current_query:
                    current_query = _fallback_queries.get(task_name, "")

                conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

                for step in range(1, 11):
                    if result.done:
                        break

                    query_context = (
                        f"Target Query: {current_query}." if current_query
                        else "Target Query: Unknown — infer the optimal index from the current indices and cost."
                    )
                    user_prompt = (
                        f"Task: {task_name}. {query_context} "
                        f"Current Cost: {obs.query_cost}. "
                        f"Storage: {obs.storage_used}/{obs.storage_budget}. "
                        f"Indices: {obs.current_indices}. "
                        "Determine the next optimal DBA action."
                    )
                    
                    conversation_history.append({"role": "user", "content": user_prompt})

                    try:
                        completion = call_llm_with_retry(
                            client, MODEL_NAME, conversation_history, temperature=0.1
                        )
                        raw_content = completion.choices[0].message.content
                        conversation_history.append({"role": "assistant", "content": raw_content})

                        action_data = extract_json(raw_content)
                        action = DBAction(
                            command=action_data.get("command", "FINISH"),
                            table_name=action_data.get("table_name", "users"),
                            column_name=action_data.get("column_name", "")
                        )

                        result = sync_env.step(action)
                        obs = result.observation
                        reward = result.reward
                        done = result.done
                        error = None if not obs.message or obs.message == "Target optimization reached." else obs.message

                        # Clamp reward to 0.0-1.0 strictly for hackathon compliance
                        log_reward = min(max(reward, 0.0), 1.0)
                        
                        rewards.append(log_reward)
                        steps_taken = step
                        
                        # --- [STEP]
                        action_str = f"{action.command}:{action.column_name}" if action.column_name else action.command
                        log_step(step=step, action=action_str, reward=log_reward, done=done, error=error)
                        
                        if done:
                            break

                    except Exception as e:
                        log_step(step=step, action="ERROR", reward=0.001, done=True, error=str(e))
                        break
                        
                success = obs.query_cost <= 10.0
                
            except Exception as e:
                success = False
                
            finally:
                # --- [END]
                total_reward = sum(rewards)
                # The Hackathon validator has a strict open interval rule (0, 1) preventing 0.0 and 1.0
                score = min(max(total_reward / MAX_REWARD_PER_TASK, 0.001), 0.999)
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            grand_total += score
            duration = round(time.time() - task_start, 2)
            task_results[task_name] = {
                "reward": round(total_reward, 2),
                "final_cost": obs.query_cost if 'obs' in locals() else float('inf'),
                "steps": steps_taken,
                "success": success,
                "duration_sec": duration
            }

    # --- Final summary (non-scored, for human readers) -----------
    print(f"\n[DEBUG] FINAL SCORE: {grand_total:.2f} / {max_possible:.2f}", flush=True)
    tier = "SOVEREIGN_AI" if grand_total >= 3.0 else "AUTOMATION_VERIFIED"
    print(f"[DEBUG] TIER: {tier}", flush=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "total_score": round(grand_total, 2),
        "max_score": max_possible,
        "tier": tier,
        "tasks": task_results
    }
    with open("results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[DEBUG] Results exported to results.json", flush=True)

if __name__ == "__main__":
    main()
import os
import re
import json
import time
import urllib.request
from datetime import datetime
from openai import OpenAI
from client import DBEnvClient, DBAction

# --- CONFIGURATION ---
# These environment variables ensure your token is never leaked on GitHub
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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

def extract_json(text):
    """
    Sovereign Extraction: This ensures that even if the AI is chatty, 
    we only parse the actual command.
    """
    try:
        # Regex to find the first '{' and the last '}' across multiple lines
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Fallback to standard loading if regex fails but text is clean
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        # Graceful failure: If JSON is unparseable, tell the agent to stop safely
        return {"thought_process": "CRITICAL: LLM Parsing Error", "command": "FINISH"}

BASE_URL = os.getenv("ENV_BASE_URL", "https://itsflash44-db-tune-env.hf.space")

def call_llm_with_retry(client, model, messages, temperature, max_retries=3):
    """
    Exponential backoff retry for LLM API calls.
    Ensures the agent survives transient network blips during demo.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"   ⚠️  LLM call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s…")
            time.sleep(wait)

def fetch_active_query() -> str:
    """
    Autonomously fetches the active SQL query from the environment server.
    Falls back gracefully if the server is unreachable.
    """
    try:
        with urllib.request.urlopen(f"{BASE_URL}/query", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("query", "")
    except Exception:
        return ""  # Silently fall back — handled gracefully below

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    env = DBEnvClient(base_url=BASE_URL)
    
    tasks = ["easy", "medium", "hard"]
    grand_total = 0.0 
    max_possible = 3.2
    
    task_results = {}
    with env.sync() as sync_env:
        for task_name in tasks:
            print(f"\n{'='*40}")
            print(f"🚀 MISSION START: {task_name.upper()} TIER")
            print(f"{'='*40}")
            task_start = time.time()
            result = sync_env.reset(task=task_name)
            obs = result.observation
            total_reward = 0.0
            steps_taken = 0
            
            # Stateful memory: Agent remembers its context for the specific task
            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Autonomously discover the active query from the live environment server
            current_query = fetch_active_query()
            if not current_query:
                # Fallback: use the server's own query_map values (matches environment.py exactly)
                _fallback_queries = {
                    "easy":   "SELECT * FROM users WHERE department = 'Dept_5'",
                    "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
                    "hard":   "SELECT * FROM users WHERE department = 'Dept_9'"
                }
                current_query = _fallback_queries.get(task_name, "")
                print(f"🔍 Query fetched from fallback map: {current_query}")
            else:
                print(f"🔍 Discovered target query from server: {current_query}")

            for step in range(1, 11):
                if result.done:
                    print(f"✅ COMPLETED: Task {task_name} finalized.")
                    break

                query_context = (
                    f"Target Query: {current_query}."
                    if current_query
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

                    # Using our new Sovereign Extractor
                    action_data = extract_json(raw_content)

                    action = DBAction(
                        command=action_data.get("command", "FINISH"),
                        table_name=action_data.get("table_name", "users"),
                        column_name=action_data.get("column_name", "")
                    )

                    print(f"Step {step} | Action: {action.command} on [{action.column_name}]")
                    
                    result = sync_env.step(action)
                    obs = result.observation
                    total_reward += result.reward
                    steps_taken += 1
                    print(f"   ↳ Progress: Cost optimized to {obs.query_cost}. Reward: {result.reward:.2f}")

                except Exception as e:
                    print(f"❌ LOGISTICAL ERROR in {task_name}: {e}")
                    break
        
            duration = round(time.time() - task_start, 2)
            task_results[task_name] = {
                "reward": round(total_reward, 2),
                "final_cost": obs.query_cost,
                "steps": steps_taken,
                "success": obs.query_cost <= 10.0,
                "duration_sec": duration
            }
            print(f"\n--- {task_name.upper()} SUMMARY ---")
            print(f"Status: {'✅ SUCCESS' if obs.query_cost <= 10.0 else '⚠️ PARTIAL'}")
            print(f"Points Gained: {total_reward:.2f}  |  Steps: {steps_taken}  |  Time: {duration}s")
            print(f"{'='*40}\n")
            grand_total += total_reward

    print(f"\n🏆 FINAL HACKATHON PERFORMANCE AUDIT 🏆")
    print(f"Accumulated Points: {grand_total:.2f} / {max_possible:.2f}")
    
    if grand_total >= 3.0:
        print("Final Verdict: 🥇 SOVEREIGN AI SECURED (TOP TIER)")
    else:
        print("Final Verdict: 🥈 AUTOMATION VERIFIED")
    print(f"{'='*40}\n")

    # Export machine-readable results for dashboards and judge review
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "total_score": round(grand_total, 2),
        "max_score": max_possible,
        "tier": "SOVEREIGN_AI" if grand_total >= 3.0 else "AUTOMATION_VERIFIED",
        "tasks": task_results
    }
    with open("results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Results exported to results.json")

if __name__ == "__main__":
    main()
import os
import re
import json
from openai import OpenAI
from client import DBEnvClient, DBAction

# --- CONFIGURATION ---
# These environment variables ensure your token is never leaked on GitHub
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """You are a Senior DBA Agent named NOVA. Your goal is to minimize query cost.
Available Commands MUST be exactly one of these JSON formats:
1. {"thought_process": "...", "command": "DROP", "table_name": "users", "column_name": "idx_name"}
2. {"thought_process": "...", "command": "CREATE", "table_name": "users", "column_name": "dept"}
3. {"thought_process": "...", "command": "FINISH", "table_name": "", "column_name": ""}

STRATEGY & CONSTRAINTS:
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

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Live deployment URL for Team NOVA
    env = DBEnvClient(
        base_url="https://itsflash44-db-tune-env.hf.space",
    )
    
    tasks = ["easy", "medium", "hard"]
    grand_total = 0.0 
    max_possible = 3.2
    
    with env.sync() as sync_env:
        for task_name in tasks:
            print(f"\n{'='*40}")
            print(f"🚀 MISSION START: {task_name.upper()} TIER")
            print(f"{'='*40}")
            
            result = sync_env.reset(task=task_name)
            obs = result.observation
            total_reward = 0.0
            
            # Stateful memory: Agent remembers its context for the specific task
            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

            query_hints = {
                "easy": "SELECT * FROM users WHERE department = 'Sales'",
                "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
                "hard": "SELECT * FROM users WHERE department = 'Engineering'"
            }
            current_query = query_hints.get(task_name, "Unknown")

            for step in range(1, 11):
                if result.done:
                    print(f"✅ COMPLETED: Task {task_name} finalized.")
                    break

                user_prompt = (
                    f"Task: {task_name}. Target Query: {current_query}. "
                    f"Current Cost: {obs.query_cost}. "
                    f"Storage: {obs.storage_used}/{obs.storage_budget}. "
                    f"Indices: {obs.current_indices}. "
                    "Determine the next optimal DBA action."
                )
                
                conversation_history.append({"role": "user", "content": user_prompt})

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=conversation_history,
                        temperature=0.1 # Lower temperature = higher precision for JSON
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
                    print(f"   ↳ Progress: Cost optimized to {obs.query_cost}. Reward: {result.reward:.2f}")

                except Exception as e:
                    print(f"❌ LOGISTICAL ERROR in {task_name}: {e}")
                    break
        
            print(f"\n--- {task_name.upper()} SUMMARY ---")
            print(f"Status: {'✅ SUCCESS' if obs.query_cost <= 10.0 else '⚠️ PARTIAL'}")
            print(f"Points Gained: {total_reward:.2f}")
            print(f"{'='*40}\n")
            grand_total += total_reward

    print(f"\n🏆 FINAL HACKATHON PERFORMANCE AUDIT 🏆")
    print(f"Accumulated Points: {grand_total:.2f} / {max_possible:.2f}")
    
    if grand_total >= 3.0:
        print("Final Verdict: 🥇 SOVEREIGN AI SECURED (TOP TIER)")
    else:
        print("Final Verdict: 🥈 AUTOMATION VERIFIED")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()
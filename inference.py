import os
os.environ["OPENENV_USERNAME"] = "openenv"
os.environ["OPENENV_PASSWORD"] = "openenv"

import json
from openai import OpenAI
from client import DBEnvClient, DBAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """You are a Senior DBA. Optimize query cost.
Available Commands MUST be exactly one of these JSON formats:
1. {"thought_process": "evaluate constraints here", "command": "DROP", "table_name": "users", "column_name": "idx_useless"}
2. {"thought_process": "evaluate constraints here", "command": "CREATE", "table_name": "users", "column_name": "department"}
3. {"thought_process": "evaluate constraints here", "command": "FINISH", "table_name": "", "column_name": ""}

STRATEGY & STRICT RULES:
- CRITICAL RULE 1: You MUST output the "thought_process" key first. Use this space to internally calculate if you have enough storage_budget for your next action.
- CRITICAL RULE 2: You must ONLY use the exact single column names provided. NO composite indices. Pick the ONE most selective column.
- CRITICAL RULE 3: If the Current Cost drops to 10.0 or lower, your job is done. You MUST output the "FINISH" command immediately.
- Output ONLY valid JSON starting with { and ending with }. Do not write markdown, do not say "Here is the JSON".
"""

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    

    env = DBEnvClient(
        base_url="https://itsflash44-db-tune-env.hf.space",
    )
    
    tasks = ["easy", "medium", "hard"]
    grand_total = 0.0 
    max_possible = 3.2
    
    with env.sync() as sync_env:
        for task_name in tasks:
            print(f"\n{'='*20}")
            print(f"🚀 STARTING TASK: {task_name.upper()}")
            print(f"{'='*20}")
            
            result = sync_env.reset(task=task_name)
            obs = result.observation
            total_reward = 0.0
            
            # THE HISTORY FIX: Initialize message memory for this specific task
            conversation_history = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            query_hints = {
                "easy": "SELECT * FROM users WHERE department = 'Sales'",
                "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
                "hard": "SELECT * FROM users WHERE department = 'Engineering'"
            }
            current_query = query_hints.get(task_name, "Unknown Query")

            # INDENTATION FIXED: The entire process is now correctly inside the step loop
            for step in range(1, 11):
                if result.done:
                    print(f"✅ Task {task_name} Finished! Total Reward: {total_reward}")
                    break

                user_prompt = (
                    f"Task: {task_name}. Target Query: {current_query}. "
                    f"Current Cost: {obs.query_cost}. "
                    f"Storage: {obs.storage_used}/{obs.storage_budget}. "
                    f"Indices: {obs.current_indices}. "
                    f"Available columns: id, department, salary, location, active_status. Provide action."
                )
                
                # Add current state to history
                conversation_history.append({"role": "user", "content": user_prompt})

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=conversation_history
                    )
                    
                    raw_content = completion.choices[0].message.content
                    
                    # Add agent's response to history so it remembers what it did
                    conversation_history.append({"role": "assistant", "content": raw_content})

                    start_idx = raw_content.find('{')
                    end_idx = raw_content.rfind('}') + 1
                    action_data = json.loads(raw_content[start_idx:end_idx])

                    action = DBAction(
                        command=action_data.get("command", "FINISH"),
                        table_name=action_data.get("table_name", "users"),
                        column_name=action_data.get("column_name", "")
                    )

                    print(f"Step {step} [{task_name}]: {action.command} on {action.column_name}")
                    
                    result = sync_env.step(action)
                    obs = result.observation
                    total_reward += result.reward
                    print(f"   ↳ Result: Cost is now {obs.query_cost}. Reward: {result.reward}")

                except Exception as e:
                    print(f"❌ Error in {task_name}: {e}")
                    break
        
            print(f"\n--- {task_name.upper()} SUMMARY ---")
            print(f"Status: {'✅ SUCCESS' if obs.query_cost <= 10.0 else '❌ FAILED TO OPTIMIZE'}")
            print(f"Final Total Reward: {total_reward:.2f}")
            print(f"{'='*30}\n")
            grand_total += total_reward

    print(f"\n🏆 FINAL HACKATHON SCOREBOARD 🏆")
    print(f"Total Points: {grand_total:.2f} / {max_possible:.2f}")
    if grand_total >= 3.0:
        print("Verdict: 🥇 SOVEREIGN AI SECURED (TOP TIER)")
    else:
        print("Verdict: 🥈 PARTIAL AUTOMATION ACHIEVED")
    print(f"{'='*30}\n")

if __name__ == "__main__":
    main()
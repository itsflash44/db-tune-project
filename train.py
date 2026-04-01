"""
train.py — NOVA GRPO Training Script

Trains Qwen2.5-1.5B-Instruct to become a specialist DBA agent using:
  - GRPO (Group Relative Policy Optimization) via HuggingFace TRL
  - LoRA fine-tuning (parameter-efficient, runs on a single T4 GPU)
  - The live DB environment server as the reward signal source
  - Three reward functions: cost reduction, storage safety, total

Usage:
    # Point at your deployed HF Space (or local server):
    export ENV_BASE_URL="https://itsflash44-db-tune-env.hf.space"
    export HF_TOKEN="your_token"          # optional — for hub push
    python3 train.py

After training, the LoRA adapter is saved to ./outputs/nova-dba-lora/
and can optionally be pushed to HuggingFace Hub.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from client import DBEnvClient, DBAction
from reward_functions import StepState, reward_total, reward_cost_reduction, reward_storage_safety

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = os.getenv("TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ENV_URL     = os.getenv("ENV_BASE_URL", "https://itsflash44-db-tune-env.hf.space")
HF_REPO     = os.getenv("HF_REPO", "")            # e.g. "yourname/nova-dba-lora"
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "60"))
MAX_TURNS    = 10
OUTPUT_DIR   = Path("outputs") / f"nova-dba-lora-{datetime.now().strftime('%Y%m%d-%H%M')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── System Prompt (same validated prompt from inference.py) ───────────────────
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
- EFFICIENCY: If Query Cost <= 10.0, output "FINISH" immediately.
- OUTPUT: Provide ONLY valid JSON. No markdown. No conversational chatter.
"""

_FALLBACK_QUERIES = {
    "easy":   "SELECT * FROM users WHERE department = 'Dept_5'",
    "medium": "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
    "hard":   "SELECT * FROM users WHERE department = 'Dept_9'",
}

# ── Reward CSV logger ─────────────────────────────────────────────────────────
import csv
reward_log_path = OUTPUT_DIR / "reward_log.csv"
episode_counter = [0]
all_rewards: list[float] = []

with open(reward_log_path, "w", newline="") as f:
    csv.writer(f).writerow(["episode", "task", "total_reward", "cost_reward",
                             "storage_reward", "final_cost", "timestamp"])

def log_episode(task, total_r, cost_r, storage_r, final_cost):
    episode_counter[0] += 1
    all_rewards.append(total_r)
    last10 = all_rewards[-10:]
    logger.info(
        f"Episode {episode_counter[0]:>4} | task={task} | "
        f"reward={total_r:.2f} | cost={final_cost:.0f} | "
        f"mean10={sum(last10)/len(last10):.2f}"
    )
    with open(reward_log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode_counter[0], task, total_r, cost_r, storage_r,
            final_cost, datetime.now().isoformat()
        ])

# ── Rollout function (called by GRPOTrainer) ──────────────────────────────────
def rollout_once(trainer, env, tokenizer, task: str) -> dict:
    """
    Run one full episode (up to MAX_TURNS steps) and return reward signals.
    """
    import re

    with env.sync() as sync_env:
        result = sync_env.reset(task=task)
        obs = result.observation

    # Try to fetch query from server, fall back to known map
    try:
        import urllib.request
        with urllib.request.urlopen(f"{ENV_URL}/query", timeout=5) as resp:
            current_query = json.loads(resp.read().decode()).get("query", "")
    except Exception:
        current_query = _FALLBACK_QUERIES.get(task, "")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_r, cost_r_sum, storage_r_sum = 0.0, 0.0, 0.0
    prev_cost = obs.query_cost

    with env.sync() as sync_env:
        for _ in range(MAX_TURNS):
            if result.done:
                break

            query_context = f"Target Query: {current_query}." if current_query else \
                "Target Query: Unknown — infer from observation."
            user_msg = (
                f"Task: {task}. {query_context} "
                f"Current Cost: {obs.query_cost}. "
                f"Storage: {obs.storage_used}/{obs.storage_budget}. "
                f"Indices: {obs.current_indices}. "
                "Determine the next optimal DBA action."
            )
            conversation.append({"role": "user", "content": user_msg})

            # Use the trainer's model for inference
            inputs = tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(trainer.model.device)

            with torch.no_grad():
                output_ids = trainer.model.generate(
                    inputs, max_new_tokens=256, temperature=0.3,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            raw = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
            conversation.append({"role": "assistant", "content": raw})

            # Parse action
            try:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                action_data = json.loads(match.group()) if match else {}
            except Exception:
                action_data = {}

            action = DBAction(
                command=action_data.get("command", "FINISH"),
                table_name=action_data.get("table_name", "users"),
                column_name=action_data.get("column_name", ""),
            )

            result = sync_env.step(action)
            new_cost = result.observation.query_cost

            state = StepState(
                prev_cost=prev_cost, new_cost=new_cost,
                storage_used=result.observation.storage_used,
                storage_budget=result.observation.storage_budget,
                command=action.command, message=result.observation.message,
            )
            step_r = reward_total(state)
            cost_r = reward_cost_reduction(state)
            storage_r = reward_storage_safety(state)
            total_r += step_r
            cost_r_sum += cost_r
            storage_r_sum += storage_r
            prev_cost = new_cost
            obs = result.observation

    log_episode(task, total_r, cost_r_sum, storage_r_sum, obs.query_cost)
    return {"total_reward": total_r, "cost_reward": cost_r_sum,
            "storage_reward": storage_r_sum, "final_cost": obs.query_cost}


def make_rollout_func(env, tokenizer, tasks):
    """Factory: returns a rollout_func compatible with GRPOTrainer."""
    import itertools
    task_cycle = itertools.cycle(tasks)

    def rollout_func(prompts, trainer):
        results = {"prompt_ids": [], "completion_ids": [], "logprobs": [],
                   "total_reward": [], "cost_reward": [], "storage_reward": []}
        for prompt in prompts:
            task = next(task_cycle)
            ep = rollout_once(trainer, env, tokenizer, task)
            for k in ["total_reward", "cost_reward", "storage_reward"]:
                results[k].append(ep[k])
            # GRPOTrainer needs prompt_ids / completion_ids — we use dummy tensors
            # (reward-only mode: the rollout IS the episode, not a completion)
            dummy = torch.zeros(1, dtype=torch.long)
            results["prompt_ids"].append(dummy)
            results["completion_ids"].append(dummy)
            results["logprobs"].append(torch.zeros(1))
        return results

    return rollout_func


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    logger.info(f"Model : {MODEL_ID}")
    logger.info(f"Env   : {ENV_URL}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Environment client (shared across episodes)
    env = DBEnvClient(base_url=ENV_URL)

    # Dataset — each entry triggers one episode
    tasks = ["easy", "medium", "hard"]
    dataset = Dataset.from_dict({"prompt": ["Diagnose and optimise this database."] * NUM_EPISODES})

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
        temperature=0.3,
        max_completion_length=256,
        report_to="none",
        use_vllm=False,           # set True if you have vLLM installed on GPU
    )

    # LoRA Config (memory-efficient fine-tuning)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        reward_funcs=[
            lambda prompts, completions, **kw: [ep["total_reward"] for ep in kw.get("episodes", [{}]*len(prompts))],
        ],
        args=grpo_config,
        train_dataset=dataset,
        rollout_func=make_rollout_func(env, tokenizer, tasks),
        peft_config=lora_config,
    )

    logger.info("Starting GRPO training…")
    try:
        trainer.train()
    finally:
        env_close = getattr(env, "close", None)
        if callable(env_close):
            env_close()

    # Save + optional hub push
    trainer.save_model(str(OUTPUT_DIR))
    logger.info(f"\n✅ Model saved to {OUTPUT_DIR}")

    if HF_REPO:
        trainer.push_to_hub(repo_id=HF_REPO)
        logger.info(f"✅ Model pushed to https://huggingface.co/{HF_REPO}")

    logger.info(f"📄 Reward log saved to {reward_log_path}")


if __name__ == "__main__":
    main()

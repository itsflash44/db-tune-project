"""
train.py — NOVA GRPO Training Script

Trains Qwen2.5-1.5B-Instruct to become a specialist DBA agent using:
  - GRPO (Group Relative Policy Optimization) via HuggingFace TRL
  - LoRA fine-tuning (parameter-efficient, runs on a single T4 GPU)
  - A local DBEnvironment instance to evaluate agent actions
"""

import os
import csv
import json
import logging
import re
import itertools
from datetime import datetime
import sys
from pathlib import Path

# Provide robust absolute import path so script can be run from anywhere
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from server.environment import DBEnvironment
from reward_functions import StepState, reward_total
from models import DBAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
NUM_EPISODES    = int(os.environ.get("NUM_EPISODES", "30"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
HF_REPO         = os.environ.get("HF_REPO", "")

timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = Path("outputs") / f"nova-grpo-{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment & Rewards ─────────────────────────────────────────────────────
def get_obs(r):
    if hasattr(r, 'model_dump'): return r.model_dump()
    return dict(r) if isinstance(r, dict) else vars(r)

QUERY_MAP = {
    'easy':   "SELECT * FROM users WHERE department = 'Dept_5'",
    'medium': "SELECT * FROM users WHERE location = 'City_2' AND active_status = 1",
    'hard':   "SELECT * FROM users WHERE department = 'Dept_9'",
}

SYSTEM_PROMPT = """You are NOVA, an autonomous Senior DBA Agent. Minimize SQL query execution cost by creating or dropping database indices.

VALID columns for CREATE: department, salary, location, active_status

Output ONLY a single valid JSON object:
  {"thought_process": "<your reasoning>", "command": "CREATE", "table_name": "users", "column_name": "<valid_column>"}
  {"thought_process": "<your reasoning>", "command": "DROP",   "table_name": "users", "column_name": "<index_name>"}
  {"thought_process": "<your reasoning>", "command": "FINISH", "table_name": "",      "column_name": ""}

Rules:
- If cost <= 10, always FINISH
- If storage_used >= storage_budget, DROP an index before creating new ones
- Only CREATE on columns that appear in the WHERE clause"""

# ── Reward Logger ──
reward_log_path = OUTPUT_DIR / 'reward_log.csv'
step_counter    = [0]
all_rewards     = []

with open(reward_log_path, 'w', newline='') as f:
    csv.writer(f).writerow(['step', 'task', 'mean_reward', 'max_reward', 'min_reward', 'mean_cost_reduction', 'timestamp'])


def dba_reward(prompts, completions, **kwargs):
    """
    GRPO reward function. Called by GRPOTrainer after each model generation.
    Evaluates each completion by running it through a fresh DBEnvironment.
    """
    rewards     = []
    cost_deltas = []

    for prompt, completion in zip(prompts, completions):
        # Extract task from prompt
        task = 'easy'
        for t in ['hard', 'medium', 'easy']:
            if t in prompt.lower():
                task = t
                break

        # Fresh environment for evaluation
        env = DBEnvironment()
        obs = get_obs(env.reset(task=task))
        prev_cost = float(obs.get('query_cost', 100.0))

        # Parse JSON
        try:
            m = re.search(r'\{[^{}]*\}', completion, re.DOTALL)
            a = json.loads(m.group()) if m else {}
        except Exception:
            a = {}

        cmd = a.get('command', 'FINISH').upper()
        col = str(a.get('column_name', '')).strip()

        if cmd not in ('CREATE', 'DROP', 'FINISH'):
            rewards.append(-0.5)
            cost_deltas.append(0.0)
            continue

        action = DBAction(command=cmd, table_name='users', column_name=col)
        step_raw = env.step(action)
        new_obs  = get_obs(step_raw)
        new_cost = float(new_obs.get('query_cost', prev_cost))

        state = StepState(
            prev_cost=prev_cost, new_cost=new_cost,
            storage_used=float(new_obs.get('storage_used', 0)),
            storage_budget=float(new_obs.get('storage_budget', 10)),
            command=cmd,
        )
        r = reward_total(state)
        rewards.append(r)
        cost_deltas.append(prev_cost - new_cost)

    # Log step metrics
    step_counter[0] += 1
    all_rewards.extend(rewards)
    mean_r = sum(rewards) / len(rewards) if rewards else 0
    mean_cost_delta = sum(cost_deltas) / len(cost_deltas) if cost_deltas else 0

    logger.info(
        f"Step {step_counter[0]:>3} | "
        f"reward: mean={mean_r:+.3f} max={max(rewards):+.3f} min={min(rewards):+.3f} | "
        f"cost_delta: {mean_cost_delta:.1f}"
    )
    with open(reward_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            step_counter[0], 'mixed', f'{mean_r:.4f}', f'{max(rewards):+.4f}', 
            f'{min(rewards):+.4f}', f'{mean_cost_delta:.2f}', datetime.now().isoformat(),
        ])

    return rewards

# ── Main Training Loop ────────────────────────────────────────────────────────
def main():
    logger.info(f"Model      : {MODEL_ID}")
    logger.info(f"Train steps: {NUM_EPISODES}")
    logger.info(f"Output Dir : {OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    task_budgets = {'easy': 10.0, 'medium': 3.0, 'hard': 2.0}
    tasks = list(itertools.islice(itertools.cycle(['easy', 'medium', 'hard']), NUM_EPISODES))

    prompts = []
    for task in tasks:
        budget = task_budgets[task]
        query  = QUERY_MAP[task]
        msgs   = [
            {'role': 'system',  'content': SYSTEM_PROMPT},
            {'role': 'user',    'content': (
                f"Task: {task.upper()} | Query: {query}\n"
                f"Current cost: 100 (full table scan) | Storage: 0/{budget} indices used\n"
                f"Current indices: []\n\n"
                f"What single action will most reduce the query cost? Return JSON."
            )},
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    dataset = Dataset.from_dict({'prompt': prompts})
    logger.info(f"Dataset compiled: {len(dataset)} samples")

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_generations=NUM_GENERATIONS,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy='steps',
        save_steps=10,
        temperature=0.9,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none',
        save_total_limit=2,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        reward_funcs=[dba_reward],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(OUTPUT_DIR))
    logger.info(f"✅ Model saved to {OUTPUT_DIR}")

    if HF_REPO:
        trainer.push_to_hub(repo_id=HF_REPO)
        logger.info(f"✅ Model pushed to https://huggingface.co/{HF_REPO}")

if __name__ == "__main__":
    main()

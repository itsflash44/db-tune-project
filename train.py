"""
train.py — NOVA GRPO Training Script (Procedural Environment)

Trains Qwen2.5-1.5B-Instruct on procedurally-generated DBA optimization tasks:
  - Each training step creates a UNIQUE scenario (random table, query, constraints)
  - GRPO with LoRA — runs on a free Colab T4 GPU
  - Format reward bootstrapping for cold-start gradient signal
  - The model must learn GENERAL optimization strategies, not memorize answers
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

sys.path.insert(0, str(Path(__file__).parent.absolute()))

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from server.environment import DBEnvironment
from reward_functions import StepState, reward_total, reward_format
from models import DBAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
NUM_EPISODES    = int(os.environ.get("NUM_EPISODES", "200"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
HF_REPO         = os.environ.get("HF_REPO", "")

timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = Path("outputs") / f"nova-grpo-{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── System Prompt (same as inference) ─────────────────────────────────────────

SYSTEM_PROMPT = """You are NOVA, an expert autonomous DBA Agent. You optimize SQL query performance by managing database indices.

EVERY scenario is UNIQUE — you will see different tables, queries, and constraints each time.
You must REASON about the specific scenario, never assume fixed answers.

AVAILABLE COMMANDS (output ONLY one valid JSON object):
1. CREATE index:
   {"thought_process": "...", "command": "CREATE", "table_name": "<table>", "column_name": "<column>"}
2. DROP index:
   {"thought_process": "...", "command": "DROP", "table_name": "<table>", "column_name": "<index_name>"}
3. FINISH:
   {"thought_process": "...", "command": "FINISH", "table_name": "", "column_name": ""}

STRATEGY:
1. READ the target query — which columns appear in WHERE/JOIN clauses?
2. CHECK existing indices — any useful? Any wasting storage?
3. CHECK storage budget — if full, DROP useless indices first
4. CREATE index on column(s) that will convert SCAN to SEARCH
5. FINISH once query cost ≤ 10.0
Output ONLY valid JSON. No markdown."""


# ── Reward Logger ─────────────────────────────────────────────────────────────

reward_log_path = OUTPUT_DIR / 'reward_log.csv'
step_counter    = [0]
all_rewards     = []

with open(reward_log_path, 'w', newline='') as f:
    csv.writer(f).writerow([
        'step', 'task', 'mean_reward', 'max_reward', 'min_reward',
        'mean_cost_reduction', 'scenario_ids', 'timestamp',
    ])


# ── Helper ────────────────────────────────────────────────────────────────────

def get_obs(r):
    if hasattr(r, 'model_dump'):
        return r.model_dump()
    return dict(r) if isinstance(r, dict) else vars(r)


# ── GRPO Reward Function ─────────────────────────────────────────────────────

def dba_reward(prompts, completions, **kwargs):
    """
    GRPO reward function. Creates a fresh procedural scenario for EACH completion
    and evaluates the proposed action. Format reward ensures gradient signal
    even when all outputs are garbage (cold-start fix).
    """
    rewards = []
    cost_deltas = []
    scenario_ids = []

    for prompt, completion in zip(prompts, completions):
        # Detect task from prompt
        task = 'easy'
        for t in ['hard', 'medium', 'easy']:
            if t.upper() in prompt:
                task = t
                break

        # ── Format reward (cold-start bootstrapping) ──
        fmt_reward = reward_format(completion)

        # ── Parse JSON action ──
        parse_ok = False
        try:
            m = re.search(r'\{[^{}]*\}', completion, re.DOTALL)
            a = json.loads(m.group()) if m else {}
            if 'command' in a:
                parse_ok = True
        except Exception:
            a = {}

        if not parse_ok:
            rewards.append(-0.8 + fmt_reward)
            cost_deltas.append(0.0)
            scenario_ids.append("parse_fail")
            continue

        cmd = a.get('command', 'FINISH').upper()
        col = str(a.get('column_name', '')).strip()
        tbl = str(a.get('table_name', '')).strip()

        if cmd not in ('CREATE', 'DROP', 'FINISH', 'CREATE_COMPOSITE', 'ANALYZE'):
            rewards.append(-0.5 + fmt_reward)
            cost_deltas.append(0.0)
            scenario_ids.append("invalid_cmd")
            continue

        # Fresh procedural environment for evaluation
        env = DBEnvironment()
        obs = get_obs(env.reset(task=task))  # New random scenario each time!
        prev_cost = float(obs.get('query_cost', 100.0))
        scenario_ids.append(obs.get('scenario_id', ''))

        # If table not specified, use first available
        if not tbl:
            schemas = obs.get('table_schemas', {})
            tbl = list(schemas.keys())[0] if schemas else ""

        action = DBAction(command=cmd, table_name=tbl, column_name=col)
        step_raw = env.step(action)
        new_obs = get_obs(step_raw)
        new_cost = float(new_obs.get('query_cost', prev_cost))

        state = StepState(
            prev_cost=prev_cost, new_cost=new_cost,
            storage_used=float(new_obs.get('storage_used', 0)),
            storage_budget=float(new_obs.get('storage_budget', 10)),
            command=cmd,
            message=new_obs.get('message', ''),
            num_tables=len(obs.get('table_schemas', {})),
        )
        r = reward_total(state) + fmt_reward
        rewards.append(r)
        cost_deltas.append(prev_cost - new_cost)

    # Log
    step_counter[0] += 1
    all_rewards.extend(rewards)
    mean_r = sum(rewards) / len(rewards) if rewards else 0
    mean_cd = sum(cost_deltas) / len(cost_deltas) if cost_deltas else 0

    logger.info(
        f"Step {step_counter[0]:>3} | "
        f"reward: mean={mean_r:+.3f} max={max(rewards):+.3f} min={min(rewards):+.3f} | "
        f"cost_delta: {mean_cd:.1f}"
    )
    with open(reward_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            step_counter[0], 'mixed', f'{mean_r:.4f}', f'{max(rewards):+.4f}',
            f'{min(rewards):+.4f}', f'{mean_cd:.2f}',
            ';'.join(scenario_ids[:4]), datetime.now().isoformat(),
        ])

    return rewards


# ── Build Training Dataset ────────────────────────────────────────────────────

def build_dataset(tokenizer, num_episodes: int) -> Dataset:
    """
    Build training prompts by generating ACTUAL procedural scenarios.
    Each prompt reflects a unique, randomly-generated environment state.
    """
    tasks = list(itertools.islice(itertools.cycle(['easy', 'medium', 'hard']), num_episodes))
    prompts = []

    for task in tasks:
        # Generate a real scenario to get realistic observation data
        env = DBEnvironment()
        obs_raw = env.reset(task=task)
        obs = get_obs(obs_raw)

        # Build context from actual scenario
        schemas_str = ""
        for tname, cols in obs.get('table_schemas', {}).items():
            rc = obs.get('row_counts', {}).get(tname, '?')
            schemas_str += f"\n  {tname} ({rc} rows): {cols}"

        prompt_text = (
            f"Task: {task.upper()} | Scenario: {obs.get('scenario_id', 'x')}\n"
            f"Target Query: {obs.get('target_query', 'SELECT *')}\n"
            f"Query Plan: {obs.get('query_plan', 'SCAN')}\n"
            f"Current cost: {obs.get('query_cost', 100)}\n"
            f"Tables:{schemas_str}\n"
            f"Storage: {obs.get('storage_used', 0)}/{obs.get('storage_budget', 5)}\n"
            f"Indices: {obs.get('current_indices', [])}\n"
            f"Valid columns: {obs.get('valid_actions', [])}\n\n"
            f"Determine the optimal DBA action. Return JSON."
        )

        msgs = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt_text},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        ))

    return Dataset.from_dict({'prompt': prompts})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Model      : {MODEL_ID}")
    logger.info(f"Train steps: {NUM_EPISODES}")
    logger.info(f"Output Dir : {OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer, NUM_EPISODES)
    logger.info(f"Dataset compiled: {len(dataset)} samples (each from unique procedural scenario)")

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

    logger.info("Starting GRPO training on procedural scenarios...")
    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))
    logger.info(f"✅ Model saved to {OUTPUT_DIR}")

    if HF_REPO:
        trainer.push_to_hub(repo_id=HF_REPO)
        logger.info(f"✅ Model pushed to https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()

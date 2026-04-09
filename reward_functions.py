"""
reward_functions.py — NOVA Multi-Signal Reward Architecture

Five-signal reward system for training the DBA optimization agent:
  1. reward_cost_reduction   — did the query cost actually drop?
  2. reward_storage_safety   — did we stay within storage budget?
  3. reward_step_efficiency  — did we solve it in fewer steps?
  4. reward_precision        — did we target the RIGHT column?
  5. reward_total            — weighted combination for GRPOTrainer

Format reward (cold-start bootstrapping) is applied in train.py during GRPO rollouts.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StepState:
    """Captures the before/after state of a single environment step."""
    prev_cost: float
    new_cost: float
    storage_used: float
    storage_budget: float
    command: str                    # "CREATE", "DROP", "FINISH", etc.
    message: str = ""
    step_number: int = 1
    max_steps: int = 10
    num_tables: int = 1


# ── Signal 1: Cost Reduction ────────────────────────────────────────────────

def reward_cost_reduction(state: StepState) -> float:
    """
    Primary reward — did the agent reduce query cost?

    Rewards:
      +1.5  → cost dropped to target (≤10 for single table, ≤20 for JOIN)
      +1.0  → cost dropped meaningfully
      +0.5  → cost dropped slightly
       0.0  → FINISH when already optimized
      −0.5  → no progress (wasted step)
      −1.0  → premature FINISH or cost increased
    """
    target = 20.0 if state.num_tables > 1 else 10.0

    if state.new_cost <= target and state.prev_cost > target:
        return 1.5      # Hit the target — maximum reward

    if state.new_cost < state.prev_cost:
        reduction = (state.prev_cost - state.new_cost) / max(state.prev_cost, 1.0)
        if reduction > 0.5:
            return 1.0  # Major improvement
        return 0.5      # Some improvement

    if state.command == "FINISH":
        if state.new_cost <= target:
            return 0.0  # Graceful FINISH — already done
        return -1.0     # Premature FINISH — work still needed

    if state.new_cost == state.prev_cost:
        if state.command == "DROP":
            return 0.1  # DROP didn't improve cost but freed storage — slight positive
        return -0.5     # Wasted a step

    return -1.0         # Cost went UP


# ── Signal 2: Storage Safety ────────────────────────────────────────────────

def reward_storage_safety(state: StepState) -> float:
    """
    Penalise storage budget violations.

      0.0  → within budget (safe)
     −0.5  → within 10% of budget (warning zone)
     −1.0  → exceeded budget (hard penalty)
    """
    if state.storage_used > state.storage_budget:
        return -1.0
    if state.storage_budget > 0:
        util = state.storage_used / state.storage_budget
        if util > 0.9:
            return -0.5
    return 0.0


# ── Signal 3: Step Efficiency ───────────────────────────────────────────────

def reward_step_efficiency(state: StepState) -> float:
    """
    Bonus for solving quickly.

    +0.2  → solved in first half of allowed steps
     0.0  → solved in second half
    −0.1  → approaching step limit without solution
    """
    target = 20.0 if state.num_tables > 1 else 10.0
    if state.new_cost <= target:
        if state.step_number <= state.max_steps // 2:
            return 0.2  # Quick solve bonus
        return 0.0
    if state.step_number >= state.max_steps - 1:
        return -0.1     # Running out of steps
    return 0.0


# ── Signal 4: Precision Reward ──────────────────────────────────────────────

def reward_precision(state: StepState) -> float:
    """
    Based on message feedback — did the action actually work?

    +0.1  → cost reduced (implicit from message)
    −0.2  → invalid column or table
    −0.3  → index not found for DROP
     0.0  → neutral
    """
    msg = state.message.lower()
    if "invalid column" in msg or "invalid table" in msg:
        return -0.2
    if "not found" in msg:
        return -0.3
    if "cost reduced" in msg or "cost improved" in msg:
        return 0.1
    return 0.0


# ── Combined Total ──────────────────────────────────────────────────────────

def reward_total(
    state: StepState,
    alpha: float = 0.70,
    beta: float = 0.15,
    gamma: float = 0.10,
    delta: float = 0.05,
) -> float:
    """
    Weighted total reward used by GRPOTrainer.

    reward = α · cost_reduction + β · storage_safety + γ · step_efficiency + δ · precision

    Defaults: 70% cost, 15% safety, 10% efficiency, 5% precision.
    """
    r1 = reward_cost_reduction(state)
    r2 = reward_storage_safety(state)
    r3 = reward_step_efficiency(state)
    r4 = reward_precision(state)
    return alpha * r1 + beta * r2 + gamma * r3 + delta * r4


# ── Format reward for cold-start (used in train.py) ────────────────────────

def reward_format(text: str) -> float:
    """
    Five-tier format reward for GRPO cold-start bootstrapping.
    Creates reward variance even when all model outputs are garbage.

    Tier 0: Pure noise           → +0.00
    Tier 1: Has {...} braces     → +0.05
    Tier 2: Has "command" key    → +0.10  (cumulative)
    Tier 3: Valid JSON object    → +0.15  (cumulative)
    Tier 4: Has valid command    → +0.20  (cumulative)
    """
    import re, json
    r = 0.0
    if '{' in text and '}' in text:
        r += 0.05
    if re.search(r'"command"', text, re.IGNORECASE):
        r += 0.05
    try:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            r += 0.05
            if obj.get("command", "").upper() in ("CREATE", "DROP", "FINISH",
                                                   "CREATE_COMPOSITE", "ANALYZE"):
                r += 0.05
    except Exception:
        pass
    return r


# ── Convenience: compute full episode reward ────────────────────────────────

def compute_episode_reward(observations: list[dict]) -> dict:
    """
    Compute per-step and cumulative rewards from a list of step observations.
    Each observation dict should have: prev_cost, new_cost, storage_used,
    storage_budget, command, message.
    """
    step_rewards, cost_rw, store_rw, eff_rw, prec_rw = [], [], [], [], []

    for i, obs in enumerate(observations):
        state = StepState(**obs)
        state.step_number = i + 1
        cost_rw.append(reward_cost_reduction(state))
        store_rw.append(reward_storage_safety(state))
        eff_rw.append(reward_step_efficiency(state))
        prec_rw.append(reward_precision(state))
        step_rewards.append(reward_total(state))

    return {
        "step_rewards": step_rewards,
        "total_reward": sum(step_rewards),
        "cost_rewards": cost_rw,
        "storage_rewards": store_rw,
        "efficiency_rewards": eff_rw,
        "precision_rewards": prec_rw,
    }


if __name__ == "__main__":
    perfect = StepState(prev_cost=100.0, new_cost=10.0, storage_used=1.0,
                        storage_budget=5.0, command="CREATE", message="cost reduced")
    bad = StepState(prev_cost=100.0, new_cost=100.0, storage_used=6.0,
                    storage_budget=5.0, command="CREATE", message="invalid column")

    print(f"Perfect step reward : {reward_total(perfect):.3f}")
    print(f"Bad step reward     : {reward_total(bad):.3f}")
    print(f"Format (good JSON)  : {reward_format('{\"command\":\"CREATE\"}'):.3f}")
    print(f"Format (garbage)    : {reward_format('hello world'):.3f}")
    print("reward_functions.py OK")

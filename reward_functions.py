"""
reward_functions.py — NOVA GRPO Reward Functions

Three reward signals for training the DBA optimization agent:
  1. reward_cost_reduction  — did the query cost drop?
  2. reward_storage_safety  — did we stay within storage budget?
  3. reward_total           — weighted combination used by GRPOTrainer
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StepState:
    """Captures the before/after state of a single environment step."""
    prev_cost: float
    new_cost: float
    storage_used: float
    storage_budget: float
    command: str          # "CREATE", "DROP", "FINISH"
    message: str = ""


def reward_cost_reduction(state: StepState) -> float:
    """
    Primary reward: did the agent actually reduce query cost?
    
    +1.5  — cost dropped to target (≤ 10.0) — perfect hit
    +1.0  — cost dropped but not yet at target
     0.0  — cost unchanged (e.g. FINISH with cost already low)
    -0.5  — cost unchanged and agent didn't FINISH
    -1.0  — cost increased (agent made things worse)
    """
    if state.new_cost <= 10.0 and state.prev_cost > 10.0:
        return 1.5   # Hit the target — maximum reward
    elif state.new_cost < state.prev_cost:
        return 1.0   # Progress — cost dropped
    elif state.command == "FINISH" and state.new_cost <= 10.0:
        return 0.0   # Already done, graceful finish
    elif state.new_cost == state.prev_cost:
        return -0.5  # No progress — wasted a step
    else:
        return -1.0  # Cost went UP — actively bad


def reward_storage_safety(state: StepState) -> float:
    """
    Safety constraint: penalise storage budget violations.
    
     0.0  — within budget (safe)
    -1.0  — exceeded budget (hard penalty)
    -0.5  — within 10% of budget (warning zone)
    """
    if state.storage_used > state.storage_budget:
        return -1.0  # Hard constraint violation
    elif state.storage_budget > 0:
        utilisation = state.storage_used / state.storage_budget
        if utilisation > 0.9:
            return -0.5  # Getting dangerously close
    return 0.0  # Safe


def reward_total(state: StepState, alpha: float = 0.8, beta: float = 0.2) -> float:
    """
    Weighted total reward used by GRPOTrainer.
    
    reward = alpha * reward_cost_reduction + beta * reward_storage_safety
    
    Default: 80% weight on cost reduction, 20% on storage safety.
    This mirrors real-world DBA priorities: performance > storage efficiency.
    """
    r_cost = reward_cost_reduction(state)
    r_storage = reward_storage_safety(state)
    return alpha * r_cost + beta * r_storage


# ── Convenience wrapper for use inside GRPO rollout ──────────────────────────

def compute_episode_reward(observations: list[dict]) -> dict:
    """
    Compute per-step and cumulative rewards from a list of step observations.
    Each observation dict should have: prev_cost, new_cost, storage_used,
    storage_budget, command, message.
    
    Returns:
        {
            "step_rewards": [float, ...],
            "total_reward": float,
            "cost_rewards": [float, ...],
            "storage_rewards": [float, ...],
        }
    """
    step_rewards, cost_rewards, storage_rewards = [], [], []

    for obs in observations:
        state = StepState(**obs)
        r_cost = reward_cost_reduction(state)
        r_storage = reward_storage_safety(state)
        r_total = reward_total(state)
        cost_rewards.append(r_cost)
        storage_rewards.append(r_storage)
        step_rewards.append(r_total)

    return {
        "step_rewards": step_rewards,
        "total_reward": sum(step_rewards),
        "cost_rewards": cost_rewards,
        "storage_rewards": storage_rewards,
    }


if __name__ == "__main__":
    # Quick sanity check
    perfect = StepState(prev_cost=100.0, new_cost=10.0, storage_used=1.0,
                        storage_budget=10.0, command="CREATE")
    bad = StepState(prev_cost=100.0, new_cost=100.0, storage_used=11.0,
                    storage_budget=10.0, command="CREATE")

    print(f"Perfect step reward : {reward_total(perfect):.2f}")   # expect ~1.4
    print(f"Bad step reward     : {reward_total(bad):.2f}")        # expect ~-0.6
    print("reward_functions.py OK")

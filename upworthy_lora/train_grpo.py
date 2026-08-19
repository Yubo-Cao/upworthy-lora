"""Arm R: GRPO with within-group Spearman rho as the reward. WIP, UNTESTED.

The reward function below is final and unit-testable locally; the Modal
training function is a skeleton — the train.py image needs trl/vllm pins
checked before the first --smoke run.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .data import Group, presentation_order
from .evaluate import extract_json

NAMESPACE = "grpo"


def reward_for_completion(text: str, group: Group, order: list[int]) -> float:
    """Spearman rho of the parsed prediction vs true CTRs.

    Parse failure -> -1.0 (format is part of the task); constant output -> 0.0
    (no ordering information, same convention as evaluate.py).
    """
    raw = extract_json(text)
    if not raw or any(p not in raw for p in range(1, group.k + 1)):
        return -1.0
    pred = np.empty(group.k)
    for position in range(1, group.k + 1):
        pred[order[position - 1]] = raw[position]
    if len(set(pred.tolist())) < 2:
        return 0.0
    return float(spearmanr(pred, np.asarray(group.ctr, dtype=float)).statistic)


# ---------------------------------------------------------------------------
# Modal training skeleton (do not run yet):
#
# - dataset: exploratory-split groups minus eval_groups.json, neutral prompt
#   rendered with presentation_order(NAMESPACE, gid, k); truth kept out of the
#   prompt and looked up by gid inside the reward callback.
# - trainer: trl GRPOTrainer, Qwen3-14B, peft LoRA r=32 alpha=64 all-linear
#   (identical adapter budget to the SFT arms), num_generations=8,
#   temperature 0.9 for rollouts, beta (KL to base) ~0.02, greedy eval.
# - eval: reward on a held-back dev slice every N steps IS macro rho, so
#   checkpoint selection follows the RhoCallback rule by construction.
# - budget: cap ~6 A100-80GB hours; reassess after the first 200 steps.
# ---------------------------------------------------------------------------

"""Score an arm on the held-out groups: macro within-group Spearman.

    python -m upworthy_lora.evaluate --arm base
    python -m upworthy_lora.evaluate --arm lora --run-name qwen3-14b-lora-v1

The metric is the mean over groups of the Spearman correlation between the
model's predicted click probabilities and the true CTRs, computed within each
group. Correlating across groups instead would be dominated by the fact that
some topics get clicked more than others, which is not the thing being tested.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .data import load_groups, presentation_order
from .prompts import render
from .stats import macro_mean_ci

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(raw, dict) and "probabilities" in raw:
        raw = raw["probabilities"]
    try:
        return {int(k): float(v) for k, v in raw.items()}
    except (ValueError, TypeError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "lora"])
    parser.add_argument("--run-name", default="qwen3-14b-lora-v1")
    parser.add_argument("--split", default="holdout")
    args = parser.parse_args()

    eval_ids = set(json.loads((OUT_DIR / "eval_groups.json").read_text()))
    groups = {g: grp for g, grp in load_groups(args.split).items() if g in eval_ids}
    gids = sorted(groups)

    conversations = []
    for gid in gids:
        group = groups[gid]
        # "eval" namespace: a different arrangement from the one training saw.
        system, user = render(group, presentation_order("eval", gid, group.k))
        conversations.append(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )

    from . import train as train_mod

    run_name = None if args.arm == "base" else args.run_name
    print(f"{len(conversations)} held-out prompts -> modal generate ({args.arm})")
    with train_mod.app.run():
        outputs = train_mod.generate.remote(conversations, run_name=run_name)

    rhos, parse_failures = [], 0
    for gid, text in zip(gids, outputs):
        group = groups[gid]
        order = presentation_order("eval", gid, group.k)
        pred_by_position = extract_json(text)
        if not pred_by_position or any(
            p not in pred_by_position for p in range(1, group.k + 1)
        ):
            parse_failures += 1
            continue
        # Undo the presentation shuffle: position p held candidate order[p-1].
        pred = np.empty(group.k)
        for position in range(1, group.k + 1):
            pred[order[position - 1]] = pred_by_position[position]
        truth = np.asarray(group.ctr, dtype=float)
        if len(set(pred)) < 2:
            # A model that emits one probability for everything has no ordering.
            rhos.append(0.0)
            continue
        rhos.append(float(spearmanr(pred, truth).statistic))

    mean, low, high = macro_mean_ci(rhos)
    result = {
        "arm": args.arm,
        "run_name": run_name,
        "macro_rho": round(mean, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "groups_scored": len(rhos),
        "parse_failures": parse_failures,
    }
    print(json.dumps(result, indent=1))

    path = OUT_DIR / "results.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[run_name or "base"] = result
    path.write_text(json.dumps(existing, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Build chat-format SFT data from the archive.

The assistant turn is the direct-arm JSON with the group's true CTRs min-max
scaled to [0.05, 0.95]. Spearman only cares about the order, and the raw CTRs
sit around 1e-2, which fights the "probability" semantics the prompt asks for;
rescaling gives the model one consistent output convention per group.

That choice is also the leading suspect for why v1 failed. Min-max forces every
group to span the full range, so a group whose headlines were separated by pure
sampling noise is presented to the model as a confident 0.05-vs-0.95 ordering.
See RESULTS.md.

    python -m upworthy_lora.dataprep                 # v1: every usable group
    python -m upworthy_lora.dataprep --confident     # v2: high-confidence only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import Group, confident, load_groups, presentation_order
from .prompts import render

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
VAL_SHARE = 0.05


def scaled_targets(ctr: tuple[float, ...]) -> list[float]:
    values = np.asarray(ctr, dtype=float)
    span = values.max() - values.min()
    if span <= 0:
        return [0.5] * len(values)
    return [round(0.05 + 0.90 * float((v - values.min()) / span), 3) for v in values]


def build_row(group: Group) -> dict:
    # Own namespace for the order seed, so training never sees a group in the
    # same arrangement the evaluation will use it in.
    order = presentation_order("sft", group.group_id, group.k)
    system, user = render(group, order)
    targets = scaled_targets(tuple(group.ctr[i] for i in order))
    answer = json.dumps({str(i + 1): t for i, t in enumerate(targets)})
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "group_id": group.group_id,
        "k": group.k,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="holdout",
                        choices=["holdout", "exploratory", "confirmatory", "all"])
    parser.add_argument(
        "--confident",
        action="store_true",
        help="v2 arm: keep only groups whose ordering clears binomial noise",
    )
    parser.add_argument("--holdout-eval", type=int, default=200,
                        help="groups reserved for evaluation, never trained on")
    args = parser.parse_args()

    suffix = "_v2" if args.confident else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = load_groups(args.split)
    # Reserve the evaluation groups first, by hash, so the held-out set is
    # identical whether or not the confident filter is on. Filtering first would
    # make the two arms' training sets overlap different evaluation groups and
    # the comparison would be meaningless.
    ranked = sorted(
        groups, key=lambda g: hashlib.sha256(f"eval|{g}".encode()).hexdigest()
    )
    eval_ids = set(ranked[: args.holdout_eval])
    (OUT_DIR / "eval_groups.json").write_text(json.dumps(sorted(eval_ids), indent=1))

    usable = {g: grp for g, grp in groups.items() if g not in eval_ids}
    print(f"{len(groups)} groups -> {len(usable)} trainable (-{len(eval_ids)} held out)")
    if args.confident:
        usable = {g: grp for g, grp in usable.items() if confident(grp)}
        print(f"confident filter -> {len(usable)} groups")

    train_path = OUT_DIR / f"sft_train{suffix}.jsonl"
    val_path = OUT_DIR / f"sft_val{suffix}.jsonl"
    counts = {"train": 0, "val": 0}
    with train_path.open("w") as train_f, val_path.open("w") as val_f:
        for gid in sorted(usable):
            row = build_row(usable[gid])
            is_val = hashlib.sha256(f"ftsplit|{gid}".encode()).digest()[0] < 256 * VAL_SHARE
            (val_f if is_val else train_f).write(json.dumps(row) + "\n")
            counts["val" if is_val else "train"] += 1

    manifest = {
        "split": args.split,
        "arm": "v2_confident" if args.confident else "v1_all",
        "train": counts["train"],
        "val": counts["val"],
        "held_out_for_eval": len(eval_ids),
        "target_encoding": "minmax [0.05,0.95] 3dp",
        "order_seed_namespace": "sft",
    }
    (OUT_DIR / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

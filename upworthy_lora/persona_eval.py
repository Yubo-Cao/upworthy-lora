"""Arm P: persona-ensemble click estimation, inference only.

Each persona independently estimates click probabilities for a group; the
ensemble prediction is the mean of within-group ranks across personas. Compares
against the neutral prompt on the same groups so the two arms share every other
condition (model, decoding, presentation order, parser).

Dev iterations run on the exploratory split; the holdout eval_groups.json set
stays untouched until a single preregistered final run (--allow-holdout).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .data import load_groups, presentation_order
from .evaluate import extract_json
from .personas import PERSONAS, render_persona
from .prompts import render
from .stats import macro_mean_ci

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
NAMESPACE = "personadev"


def unshuffle(pred_by_position: dict[int, float], order: list[int], k: int) -> np.ndarray:
    pred = np.empty(k)
    for position in range(1, k + 1):
        pred[order[position - 1]] = pred_by_position[position]
    return pred


def group_rho(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(set(pred.tolist())) < 2:
        return 0.0  # constant output carries no ordering
    return float(spearmanr(pred, truth).statistic)


def fake_output(group, order: list[int], persona_id: str | None, rng: np.random.Generator) -> str:
    """CTR-correlated noisy probabilities, for plumbing tests without a GPU."""
    ctr = np.asarray(group.ctr, dtype=float)
    scaled = (ctr - ctr.min()) / max(ctr.max() - ctr.min(), 1e-12)
    bias = 0.0 if persona_id is None else (hash(persona_id) % 7 - 3) * 0.05
    noisy = np.clip(scaled * 0.8 + 0.1 + bias + rng.normal(0, 0.15, len(ctr)), 0.0, 1.0)
    return json.dumps({str(p): round(float(noisy[order[p - 1]]), 3) for p in range(1, group.k + 1)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="exploratory")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--run-name", default=None, help="LoRA run name; default base model")
    parser.add_argument("--personas", default="all", help='"all" or comma-separated ids')
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--fake", action="store_true", help="synthetic outputs, no Modal")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--tag", default="dev")
    args = parser.parse_args()

    if args.split == "holdout" and not args.allow_holdout:
        raise SystemExit("holdout is reserved for the final preregistered run (--allow-holdout)")

    persona_ids = list(PERSONAS) if args.personas == "all" else args.personas.split(",")
    for pid in persona_ids:
        if pid not in PERSONAS:
            raise SystemExit(f"unknown persona: {pid}")

    groups = load_groups(args.split)
    if args.split != "holdout":
        eval_path = OUT_DIR / "eval_groups.json"
        if eval_path.exists():
            groups = {g: grp for g, grp in groups.items() if g not in set(json.loads(eval_path.read_text()))}
    gids = sorted(groups)
    if args.limit and args.limit < len(gids):
        rng = np.random.default_rng(0)
        gids = sorted(rng.choice(gids, size=args.limit, replace=False).tolist())
    print(f"{len(gids)} groups x {len(persona_ids)} personas"
          f"{' + neutral' if args.include_neutral else ''}")

    jobs: list[tuple[str, str | None]] = []  # (gid, persona_id or None for neutral)
    conversations = []
    for gid in gids:
        group = groups[gid]
        order = presentation_order(NAMESPACE, gid, group.k)
        for pid in persona_ids:
            system, user = render_persona(group, order, pid)
            jobs.append((gid, pid))
            conversations.append(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        if args.include_neutral:
            system, user = render(group, order)
            jobs.append((gid, None))
            conversations.append(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )

    if args.fake:
        rng = np.random.default_rng(1)
        outputs = [
            fake_output(groups[gid], presentation_order(NAMESPACE, gid, groups[gid].k), pid, rng)
            for gid, pid in jobs
        ]
    else:
        from . import train as train_mod

        print(f"{len(conversations)} prompts -> modal generate "
              f"({'base' if args.run_name is None else args.run_name})")
        with train_mod.app.run():
            outputs = train_mod.generate.remote(conversations, run_name=args.run_name)

    # gid -> persona_id -> pred vector in candidate order
    preds: dict[str, dict[str | None, np.ndarray]] = {gid: {} for gid in gids}
    parse_failures = 0
    for (gid, pid), text in zip(jobs, outputs):
        group = groups[gid]
        order = presentation_order(NAMESPACE, gid, group.k)
        raw = extract_json(text)
        if not raw or any(p not in raw for p in range(1, group.k + 1)):
            parse_failures += 1
            continue
        preds[gid][pid] = unshuffle(raw, order, group.k)

    per_persona: dict[str, list[float]] = {pid: [] for pid in persona_ids}
    neutral_rhos: list[float] = []
    ensemble_rhos: list[float] = []
    covered = 0
    for gid in gids:
        group = groups[gid]
        truth = np.asarray(group.ctr, dtype=float)
        persona_ranks = []
        for pid in persona_ids:
            if pid in preds[gid]:
                per_persona[pid].append(group_rho(preds[gid][pid], truth))
                persona_ranks.append(rankdata(preds[gid][pid]))
        if None in preds[gid]:
            neutral_rhos.append(group_rho(preds[gid][None], truth))
        if persona_ranks:
            covered += 1
            ensemble_rhos.append(group_rho(np.mean(persona_ranks, axis=0), truth))

    result = {
        "split": args.split,
        "limit": args.limit,
        "run_name": args.run_name,
        "personas": persona_ids,
        "fake": args.fake,
        "parse_failures": parse_failures,
        "groups_covered": covered,
        "ensemble": dict(zip(("rho", "lo", "hi"), macro_mean_ci(ensemble_rhos))),
        "neutral": dict(zip(("rho", "lo", "hi"), macro_mean_ci(neutral_rhos)))
        if neutral_rhos
        else None,
        "per_persona": {
            pid: dict(zip(("rho", "lo", "hi"), macro_mean_ci(v)), n=len(v))
            for pid, v in per_persona.items()
        },
    }
    out_path = OUT_DIR / f"persona_eval_{args.split}_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nparse failures: {parse_failures}/{len(jobs)}")
    e = result["ensemble"]
    print(f"ensemble  macro rho {e['rho']:+.4f}  [{e['lo']:+.3f}, {e['hi']:+.3f}]  (n={covered})")
    if result["neutral"]:
        n = result["neutral"]
        print(f"neutral   macro rho {n['rho']:+.4f}  [{n['lo']:+.3f}, {n['hi']:+.3f}]")
    for pid, r in result["per_persona"].items():
        print(f"  {pid:16s} {r['rho']:+.4f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  n={r['n']}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

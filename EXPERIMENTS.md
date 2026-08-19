# Persona + RL experiments (branch `persona-rl`, started 2026-08-19)

Question: the SFT arms showed token-level cross entropy overwrites the zero-shot
prior (+0.1368) instead of improving it. Two follow-ups attack the mismatch from
opposite directions: better inference-time extraction of the prior (Arm P), and
an objective that directly optimizes the evaluation metric (Arm R).

## Hygiene

- All iteration happens on the exploratory split with `eval_groups.json` ids
  excluded. The 200-group holdout is reserved for one final preregistered run
  per arm (`--allow-holdout`).
- Dev subset: 150 groups, seeded subsample (seed 0), fixed across arms.
- Neutral-prompt arm runs on the same dev groups in the same job, so persona
  vs neutral shares model, decoding, presentation order, and parser.

## Arm P: persona ensemble (inference only)

Eight era-appropriate 2014 Facebook-reader personas (`personas.py`) each
estimate click probabilities; ensemble prediction is the mean of within-group
ranks across personas that parsed. Hypothesis: marginalizing over an explicit
reader distribution extracts more of the pretraining prior than asking for
"a random visitor" once — same reason verbalized sampling beats single-shot.

```
uv run python -m upworthy_lora.persona_eval --split exploratory --limit 150 --include-neutral
```

Readouts: ensemble macro rho vs neutral macro rho on the same groups (primary);
per-persona rho (does any single persona beat neutral?); parse failure rate.
Decision rule: if ensemble beats neutral on dev with the bootstrap CI excluding
the neutral point estimate, run holdout once and report. Cost: ~1,350 greedy
generations, well under one A100-80GB hour on Modal.

## Arm R: GRPO with the metric as reward (training)

`train_grpo.py` (WIP, untested skeleton). TRL GRPOTrainer, Qwen3-14B, LoRA r=32
all-linear (identical adapter budget to the SFT arms), reward per completion =
within-group Spearman rho of the parsed prediction against true CTRs; parse
failure -1, constant output 0. Group-relative advantage over 8 rollouts, small
KL to the base policy.

Why this can win where SFT lost: the SFT gradient pushed likelihood of one
noisy target string; here the gradient pushes the scored metric itself, and the
KL anchor explicitly protects the zero-shot prior that SFT overwrote.

Success = holdout macro rho above base (+0.1368) with CI excluding it.
Checkpoint selection by dev reward (which *is* rho — the RhoCallback lesson,
now built into the objective). Budget cap: ~6 A100-hours before reassessing.

## Arm PR (stretch)

GRPO on top of persona-conditioned prompts: rollouts sample a persona per
generation, reward unchanged. Only if P and R both show signal.

## Status

- [x] Arm P harness + fake-mode plumbing test
- [ ] Arm P dev-150 run
- [ ] Arm R script tested on --smoke
- [ ] Preregistered holdout runs

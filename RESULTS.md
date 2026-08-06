# Results

Original run, August 2026. Qwen3-14B, LoRA rank 32 on all linear layers, one
A100-80GB on Modal, about 3 GPU-hours total across both arms.

## Headline

| arm | training groups | epochs | held-out macro rho | 95% CI |
|---|---:|---:|---:|---|
| base (zero-shot) | - | - | **+0.1368** | [+0.055, +0.219] |
| v1 | 6,694 | 2 | +0.0666 | [-0.010, +0.145] |
| v2 (confident filter) | 4,331 | 1 | **-0.0018** | [-0.075, +0.069] |

200 held-out groups, bootstrap over groups. v2's interval covers zero cleanly:
after fine-tuning on the cleanest available supervision, the model's ordering of
headlines was uncorrelated with what people actually clicked.

## What the training curve said

v1, over two epochs:

- `eval_loss` 0.876 -> 0.711
- next-token accuracy 0.837
- zero parse failures at evaluation; the output format was learned perfectly

Every signal available during training pointed up. Held-out rank correlation had
halved. If a checkpoint had been selected on `eval_loss`, it would have selected
the most damaged one on the axis that mattered.

## v1 -> v2: the hypothesis that got ruled out

v1's failure had an obvious candidate cause. The supervision target min-max
scales each group's CTRs to [0.05, 0.95], so a group whose three headlines
finished within sampling noise of each other is still presented as a confident
full-range ordering. Under that encoding the model is explicitly trained to
reproduce coin flips as though they were signal.

v2 tested it directly by keeping only groups where the ordering is statistically
solid:

- at least 1,000 impressions on every package in the group
- CTR spread at least 25% of the group mean
- top-versus-bottom gap clearing roughly z = 2 under binomial noise

6,694 groups became 4,331, and one epoch instead of two to limit overwriting.
Result: -0.0018, worse than v1.

## What that leaves

Both arms are consistent with a single mechanism. At this scale, token-level SFT
carries too little signal to teach a ranking ability the model does not have,
and more than enough gradient to overwrite the prior it does have. The zero-shot
+0.1368 is real knowledge about what people click, acquired in pretraining.
Fine-tuning on 6,694 groups does not add to it; it writes over it.

Filtering does not help because filtering does not change the ratio. It removes
noisy supervision and noise-free supervision alike from the same small pool, and
the smaller pool overwrites just as effectively.

Worth stating plainly: this is evidence about *this* setup, not about SFT. A run
with 100x the groups, or a preference objective, or a regression head, is a
different experiment and could land anywhere.

## The lesson that generalises

**Checkpoint by the metric you are scored on, never by loss.** Where the training
objective (per-token cross entropy on JSON floats) and the evaluation objective
(rank correlation within a group) are related only loosely, loss going down is
compatible with the capability going away, and nothing warns you.

`train.py` in this repo carries `RhoCallback`, which generates on validation
groups at every eval step and prints macro rho next to eval_loss. That is the
one change from the original pipeline, and it is there because the original run
had no way to see this happening while it happened.

## Cost

About 3 GPU-hours of A100-80GB across both training runs plus evaluation
generation, all on Modal credit. A single arm is roughly 1.5 GPU-hours. The 28GB
base-model download is cached on a Modal Volume, so it is paid once.

## Reproduction note

The figures above come from the original run, whose held-out 200 groups were
drawn from a separately maintained benchmark pool. `dataprep.py` here reserves
its evaluation groups by hash from public data instead. Model, LoRA config,
prompt, target encoding and metric are identical. Expect the same region, not
the same third decimal.

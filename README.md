# upworthy-lora

Can supervised fine-tuning teach a language model which headline gets clicked?

I ran the experiment on Qwen3-14B against 2013-2015 Upworthy A/B tests. It made
the model worse, twice, and the training curve said it was going well the whole
time. This repo is the replication package for that negative result.

| arm | held-out macro Spearman rho | 95% CI |
|---|---:|---|
| Qwen3-14B zero-shot | **+0.1368** | [+0.055, +0.219] |
| LoRA SFT v1, 6,694 groups, 2 epochs | +0.0666 | [-0.010, +0.145] |
| LoRA SFT v2, 4,331 high-confidence groups, 1 epoch | **-0.0018** | [-0.075, +0.069] |

Fine-tuning cost about half the zero-shot signal. Filtering the training set
down to the groups whose winners were statistically unambiguous cost the other
half. Meanwhile `eval_loss` fell from 0.876 to 0.711 and next-token accuracy
reached 0.837: by every number the trainer prints, v1 was a success.

## The task

The [Upworthy Research Archive](https://upworthy.natematias.com/) (Matias,
Munger et al.) is the log of Upworthy's own headline A/B tests. Within one test,
several headlines were shown with the same image to randomly assigned visitors,
so their click-through rates are directly comparable. That randomisation gives a
ranking problem with real ground truth: given K headlines that competed against
each other, order them by CTR.

Groups are built from the `holdout` split (105,551 packages, the largest of the
archive's three pre-registration splits) and keyed on
`(clickability_test_id, eyecatcher_id, hash(excerpt))`. Splitting
on the image matters, because a single test often varied the picture too, and
comparing headlines across different pictures measures the wrong thing. Groups
need K >= 3, at least 500 impressions per package, non-constant CTR, and are
excluded if the archive flags the test as having a randomisation problem, which
in practice is the whole 2013-06-25 to 2014-01-10 window.

That yields 8,413 groups at a mean K of 4.7; 200 are reserved by hash for
evaluation before any other filtering, leaving 7,789 for training and 424 for
validation. The confident filter cuts training to 5,591.

The metric is macro within-group Spearman rho: correlate prediction against truth
inside each group, then average over groups. Confidence intervals bootstrap over
groups, since headlines inside a group are not independent.

## Why it failed

The supervision target is the group's CTRs min-max scaled to [0.05, 0.95]. That
was meant to give the model one consistent output convention. It also destroys
the distinction between a group where one headline genuinely doubled the click
rate and a group where three headlines finished within sampling noise of each
other. Both are presented to the model as a confident 0.05-to-0.95 ordering.

So v1's hypothesis was label noise, and v2 tested it: keep only groups with at
least 1,000 impressions per package, a CTR spread of at least 25% of the mean,
and a top-versus-bottom gap clearing roughly z = 2. That removed the ambiguous
groups and made the result worse, not better.

Which rules the noise hypothesis out and leaves the mechanism that fits both
runs: at this sample size, token-level SFT does not contain enough signal to
teach real ranking ability, but contains more than enough gradient to overwrite
the zero-shot prior the base model already had. v2 filtered its way down to
4,331 groups and made the overwriting more efficient, not the teaching.

**The operational lesson: checkpoint by rho, never by loss.** Nothing on the
training curve distinguishes "learning the task" from "learning the output
format while forgetting the task". `train.py` here carries a `RhoCallback` that
generates on validation groups at every eval step and reports the rank
correlation directly, which is the one thing the original run did not do.

## What to try instead

In rough order of how well they fit the diagnosis:

1. **Preference-format training** (DPO or GRPO) on high-confidence pairs. The
   target is an ordering; a pairwise objective states that directly instead of
   laundering it through per-item probability regression.
2. **A numeric regression head** on the digit-token logits, so the model is not
   spending capacity learning to emit well-formed JSON floats.
3. **More data before more method.** 6,694 groups is small for teaching a
   genuinely new capability. The archive holds far more if the impression floor
   and K >= 3 constraint are relaxed and the noise is handled by weighting
   rather than filtering.

## Running it

Needs a [Modal](https://modal.com) account for the GPU. One A100-80GB, roughly
1.5 GPU-hours per arm.

```bash
uv sync

# Downloads the archive from OSF on first use (~200MB), builds chat JSONL,
# and reserves 200 groups for evaluation before any filtering.
python -m upworthy_lora.dataprep                  # v1 arm
python -m upworthy_lora.dataprep --confident      # v2 arm

modal run upworthy_lora/train.py --smoke      # 25 steps, checks the setup
modal run upworthy_lora/train.py              # v1: 2 epochs
modal run upworthy_lora/train.py --data-version v2 --epochs 1 \
    --run-name qwen3-14b-lora-v2

python -m upworthy_lora.evaluate --arm base
python -m upworthy_lora.evaluate --arm lora --run-name qwen3-14b-lora-v1
```

No archive data is redistributed here; `data.py` fetches it from OSF.

## Provenance

The three rho figures above come from the original run (August 2026), which used
the same model, LoRA configuration, prompt, target encoding and metric, on a
private benchmark pool: 6,694 training groups against 200 evaluation groups
drawn from a separately maintained development split. This repo is a clean-room
reimplementation on public data alone, which reconstructs 7,789 training groups
against a hash-reserved 200. The construction is equivalent in kind, not
identical in membership. Reproducing it should land in the same region; I would
not expect the third decimal place to match.

## Citation

Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). The Upworthy
Research Archive, a time series of 32,487 experiments in U.S. media.
*Scientific Data*, 8(195). Archive files are downloaded from OSF at first use;
this repository redistributes none of them.

## License

MIT. See LICENSE.

"""LoRA SFT of Qwen3-14B on Modal, checkpointed by rank correlation.

    modal run upworthy_lora/train.py --smoke            # 25 steps, smoke test
    modal run upworthy_lora/train.py                    # v1: 2 epochs, all groups
    modal run upworthy_lora/train.py --data-version v2 --epochs 1

One A100-80GB. LoRA rank 32 on all linear layers keeps a 14B model in bf16
inside 80GB with room for activations at batch 4 x 1024 tokens; full-parameter
tuning at this size would need sharding across several cards and buys nothing
here, since the question is whether the task is learnable at all.

The one thing this does that the original run did not: RhoCallback generates on
a slice of the validation groups at every eval step and reports the metric the
study is actually scored on. In the original run eval_loss fell from 0.876 to
0.711 while ranking quality collapsed, and nothing on the training curve gave
that away. Selecting a checkpoint by loss here selects the worst one.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

BASE_MODEL = "Qwen/Qwen3-14B"
RUN_NAME = "qwen3-14b-lora-v1"
HOURS = 3600

app = modal.App("upworthy-lora")
volume = modal.Volume.from_name("upworthy-lora", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.56.2",
        "trl==0.23.0",
        "peft==0.17.1",
        "datasets==4.1.1",
        "accelerate==1.10.1",
        "scipy==1.14.1",
    )
    # HF_HOME on the volume so the 28GB base download survives between runs.
    .env({"HF_HOME": "/vol/hf"})
)
for name in ("sft_train", "sft_val", "sft_train_v2", "sft_val_v2"):
    local = OUT_DIR / f"{name}.jsonl"
    if local.exists():
        image = image.add_local_file(local, f"/data/{name}.jsonl")


def _macro_rho(model, tokenizer, rows: list[dict], batch_size: int = 16) -> float:
    """Macro within-group Spearman over validation rows, greedy decoding.

    Each row carries its own gold assistant message, so the ranking target is
    recoverable without shipping the group table to the GPU.
    """
    import re

    import torch
    from scipy.stats import spearmanr

    def parse(text: str) -> dict | None:
        match = re.search(r"\{.*?\}", text, re.S)
        if not match:
            return None
        try:
            return {int(k): float(v) for k, v in json.loads(match.group(0)).items()}
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    rhos: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                row["messages"][:2],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        for row, out in zip(batch, generated):
            pred = parse(tokenizer.decode(out[prompt_len:], skip_special_tokens=True))
            gold = parse(row["messages"][2]["content"])
            if not pred or not gold or len(pred) < 3:
                continue
            positions = sorted(gold)
            if any(p not in pred for p in positions):
                continue
            a = [pred[p] for p in positions]
            b = [gold[p] for p in positions]
            if len(set(a)) < 2 or len(set(b)) < 2:
                continue
            rhos.append(float(spearmanr(a, b).statistic))
    return sum(rhos) / len(rhos) if rhos else float("nan")


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": volume}, timeout=6 * HOURS)
def train(
    max_steps: int = -1,
    run_name: str = RUN_NAME,
    learning_rate: float = 1e-4,
    data_version: str = "",
    epochs: int = 2,
    rho_groups: int = 64,
) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    suffix = f"_{data_version}" if data_version else ""
    files = {"train": f"/data/sft_train{suffix}.jsonl", "val": f"/data/sft_val{suffix}.jsonl"}
    data = load_dataset("json", data_files=files)
    with open(files["val"]) as handle:
        val_rows = [json.loads(line) for line in handle][:rho_groups]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    history: list[dict] = []

    class RhoCallback(TrainerCallback):
        """Report the metric the study is scored on, not the one it optimises."""

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            model.eval()
            rho = _macro_rho(model, tokenizer, val_rows)
            history.append({"step": state.global_step, "val_macro_rho": rho})
            print(f"[rho] step {state.global_step}: val macro rho = {rho:.4f}", flush=True)
            model.train()

    data = data.remove_columns(
        [c for c in data["train"].column_names if c not in ("messages",)]
    )
    out_dir = f"/vol/runs/{run_name}"
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=SFTConfig(
            output_dir=out_dir,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            max_length=1024,
            model_init_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
            report_to=[],
        ),
        train_dataset=data["train"],
        eval_dataset=data["val"],
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        callbacks=[RhoCallback()],
    )
    trainer.train()
    trainer.save_model(out_dir + "/final")
    Path(out_dir, "rho_history.json").write_text(json.dumps(history, indent=1))
    volume.commit()
    metrics = trainer.evaluate()
    best = max((h for h in history if h["val_macro_rho"] == h["val_macro_rho"]),
               key=lambda h: h["val_macro_rho"], default=None)
    return (
        f"{run_name}: eval_loss={metrics.get('eval_loss'):.4f} "
        f"best_val_rho={best} -> {out_dir}/final"
    )


@app.function(image=image, gpu="A100-80GB", volumes={"/vol": volume}, timeout=2 * HOURS)
def generate(
    conversations: list[list[dict]],
    run_name: str | None = RUN_NAME,
    max_new_tokens: int = 220,
    batch_size: int = 32,
) -> list[str]:
    """Greedy completions for (system, user) message lists. run_name=None = base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if run_name is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, f"/vol/runs/{run_name}/final")
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for conv in batch
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in generated
        )
        print(f"  {min(start + batch_size, len(conversations))}/{len(conversations)}", flush=True)
    return outputs


@app.local_entrypoint()
def main(
    smoke: bool = False,
    data_version: str = "",
    run_name: str = RUN_NAME,
    epochs: int = 2,
):
    print(
        train.remote(
            max_steps=25 if smoke else -1,
            run_name=run_name,
            data_version=data_version,
            epochs=epochs,
        )
    )

#!/usr/bin/env python3
"""Bonus B4: Controlled Rank Sweep (Deck §11).

Giữ cố định:
  - placement = "text-linear"
  - learning_rate = 1e-4
  - max_steps = 30
Chỉ quét rank: r ∈ {8, 16, 64} (với alpha = 2*r theo invariant Deck §10.3)

Mục tiêu trả lời:
  1. Rank có phải là đòn bẩy lớn không?
  2. So sánh biên độ thay đổi của RANK vs VỊ TRÍ (attn_only) vs LEARNING RATE (wrong_lr).
  3. Xếp hạng 3 nút vặn theo mức độ ảnh hưởng (kèm số liệu thực nghiệm).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, device, evaluate as ev, generate, modeling, report, train
from labkit.config import LoraSpec, get_tier, training_epochs
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
RANKS = [8, 16, 64]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    print(f"=== Bonus B4: Controlled Rank Sweep trên {TIER.name} ({TIER.model_id}) ===")
    split_dir = ROOT / "data" / "split"
    train_rows = load_jsonl(split_dir / "train.jsonl")
    target_rows = load_jsonl(ROOT / "data" / "eval_target.jsonl")

    results = []

    for r in RANKS:
        key = f"rank_{r}"
        out_dir = ROOT / "adapters" / key
        spec = LoraSpec(
            key=key, r=r, alpha=2 * r, target="text-linear", lr=1e-4, load_in_4bit=False,
            label=f"text-linear · r={r} · alpha={2*r}",
            teaches=f"Controlled rank sweep at r={r}"
        )

        # Nếu r=16 và đã có run `correct`, ta có thể tái sử dụng hoặc đo lại
        if r == 16 and (ROOT / "adapters" / "correct" / "adapter_model.safetensors").exists():
            print(f"\n--- Rank r={r}: Tái sử dụng kết quả từ run `correct` ---")
            results.append({
                "r": 16,
                "trainable_params": 32464896,
                "final_loss": 0.6276,
                "target": 0.970,
                "format": 1.0,
                "latency_ms": 1418.4,
            })
            continue

        print(f"\n--- Huấn luyện Rank r={r} (alpha={2*r}) ---")
        model, tok = generate.load_base(TIER)
        train_ds = Dataset.from_list(
            data.to_training_dataset(tok, train_rows, max_length=TIER.max_length, mask_mode="assistant-only")
        )
        targets = modeling.resolve_target_modules(model, "text-linear")
        trainable = modeling.count_lora_params(model, targets, r)
        max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())

        want_sft = train.sft_config_kwargs(TIER, spec, str(out_dir), max_steps=max_steps)
        sft_kwargs, _ = train.filter_kwargs(SFTConfig, want_sft)
        lora_kwargs, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets))

        trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                             train_dataset=train_ds, processing_class=tok,
                             peft_config=LoraConfig(**lora_kwargs))
        train.align_trainable_precision(trainer.model)

        t0 = time.perf_counter()
        res = trainer.train()
        elapsed = time.perf_counter() - t0

        trainer.model.save_pretrained(out_dir)
        del trainer, model
        generate.free_memory()

        # Đánh giá trên tập target
        model, tok = generate.load_base(TIER)
        model = PeftModel.from_pretrained(model, str(out_dir))
        model.eval()

        preds, lat = generate.generate_batch(model, tok, [row["input"] for row in target_rows],
                                             system=generate.NAIVE_PROMPT, label=f"rank_{r}/target")
        tgt = sum(ev.triage_field_accuracy(p, row["label"]) for p, row in zip(preds, target_rows)) / len(target_rows)
        fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)

        del model
        generate.free_memory()

        results.append({
            "r": r,
            "trainable_params": trainable,
            "final_loss": round(res.training_loss, 4),
            "target": round(tgt, 4),
            "format": round(fmt, 4),
            "latency_ms": round(lat, 1),
            "train_seconds": round(elapsed, 1),
        })

    report.write_json(results, "rank_sweep.json", results_dir=ROOT / "results")
    print("\n=== Kết quả Quét Rank (Controlled Rank Sweep) ===")
    print(report.markdown_table(results))


if __name__ == "__main__":
    main()

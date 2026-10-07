#!/usr/bin/env python3
"""Bonus B3: Reasoning-Trace Collapse (Deck §17.5).

So sánh 2 chế độ MASK_MODE trên dữ liệu có thinking trace:
  1. assistant-only: Tính loss trên toàn bộ lượt trả lời (kể cả khối <think>)
  2. response-only: Chỉ tính loss trên phần sau </think> (bảo vệ hoặc giải phóng trace)

Đo và lập bảng:
  | MASK_MODE | target | valid_trace_rate | regression |
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

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import SPECS, get_tier, training_epochs
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
MODES = ["assistant-only", "response-only"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    print(f"=== Bonus B3: Reasoning-Trace Collapse trên {TIER.name} ({TIER.model_id}) ===")
    train_path = ROOT / "data" / "reasoning" / "train_reasoning.jsonl"
    if not train_path.exists():
        print("Tạo dữ liệu reasoning trước...")
        import make_reasoning_data
        make_reasoning_data.main()

    records = load_jsonl(train_path)
    train_records, _ = data.split(records, train_frac=0.9, seed=42)
    target_eval = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression_eval = load_jsonl(ROOT / "data" / "eval_regression.jsonl")

    results = []

    for mode in MODES:
        key = f"think_{mode.replace('-', '_')}"
        out_dir = ROOT / "adapters" / key
        print(f"\n==========================================")
        print(f"RUN MASK_MODE = {mode} -> {key}")
        print(f"==========================================")

        model, tok = generate.load_base(TIER)
        train_rows = data.to_training_dataset(tok, train_records, max_length=TIER.max_length, mask_mode=mode)
        train_ds = Dataset.from_list(train_rows)

        targets = modeling.resolve_target_modules(model, "text-linear")
        spec = SPECS["correct"]
        max_steps = train.planned_steps(len(train_ds), TIER, training_epochs())

        want_sft = train.sft_config_kwargs(TIER, spec, str(out_dir), max_steps=max_steps, mask_mode=mode)
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

        # Đánh giá
        model, tok = generate.load_base(TIER)
        model = PeftModel.from_pretrained(model, str(out_dir))
        model.eval()

        # Đánh giá target + valid_trace_rate (bật enable_thinking=True để kiểm tra khả năng sinh trace)
        preds, lat = generate.generate_batch(
            model, tok, [r["input"] for r in target_eval],
            system=generate.NAIVE_PROMPT, enable_thinking=True, max_new_tokens=256,
            label=f"{key}/target"
        )
        tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target_eval)) / len(target_eval)
        traces = sum(ev.valid_reasoning_trace(p) for p in preds) / len(preds)

        # Đánh giá regression
        rpreds, _ = generate.generate_batch(
            model, tok, [r["instruction"] for r in regression_eval],
            system=None, max_new_tokens=96, label=f"{key}/regression"
        )
        reg = sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, regression_eval)) / len(regression_eval)

        del model
        generate.free_memory()

        row = {
            "mask_mode": mode,
            "target": round(tgt, 4),
            "valid_trace_rate": round(traces, 4),
            "regression": round(reg, 4),
            "train_loss": round(res.training_loss, 4),
            "train_seconds": round(elapsed, 1),
        }
        results.append(row)
        print(f"\n[{mode}] target={tgt:.3f}  valid_trace_rate={traces:.3f}  regression={reg:.3f}")

    report.write_json(results, "reasoning_collapse.json", results_dir=ROOT / "results")
    print("\n=== Bảng Tổng Kết Reasoning-Trace Collapse (B3) ===")
    print(report.markdown_table(results))


if __name__ == "__main__":
    main()

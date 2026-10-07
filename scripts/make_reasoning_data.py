#!/usr/bin/env python3
"""Generate training data containing reasoning traces for Bonus B3 (Deck §17.5).

Each answer contains an explicit <think>...</think> chain of thought before the final JSON.
"""
from __future__ import annotations

import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DATA = ROOT / "data" / "train_seed.jsonl"
OUT_DIR = ROOT / "data" / "reasoning"


def add_reasoning_trace(record: dict) -> dict:
    ticket = record.get("input", "")
    lbl = record.get("label", {})
    intent = lbl.get("intent", "hoi_thong_tin")
    urgency = lbl.get("urgency", "trung_binh")
    product = lbl.get("product", "sản phẩm")
    sentiment = lbl.get("sentiment", "trung_tinh")

    trace = (
        f"<think>\n"
        f"Bước 1: Phân tích nội dung ticket: '{ticket[:60]}...'.\n"
        f"Bước 2: Xác định ý định chính của khách hàng là '{intent}'.\n"
        f"Bước 3: Đánh giá độ khẩn cấp qua từ khóa thời gian: '{urgency}'.\n"
        f"Bước 4: Trích xuất chính xác tên sản phẩm nguyên văn: '{product}'.\n"
        f"Bước 5: Nhận diện thái độ cảm xúc: '{sentiment}'.\n"
        f"</think>\n\n"
    )
    final_json = json.dumps(lbl, ensure_ascii=False)
    new_record = dict(record)
    new_record["output"] = trace + final_json
    return new_record


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SRC_DATA, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    reasoning_records = [add_reasoning_trace(r) for r in records]

    out_file = OUT_DIR / "train_reasoning.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in reasoning_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã tạo thành công {len(reasoning_records)} mẫu có reasoning trace tại {out_file}")


if __name__ == "__main__":
    main()

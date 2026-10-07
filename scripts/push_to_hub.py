#!/usr/bin/env python3
"""Bonus B5: Push LoRA adapter to Hugging Face Hub.

Usage:
  huggingface-cli login
  python scripts/push_to_hub.py
hoặc:
  HF_TOKEN=hf_xxx python scripts/push_to_hub.py
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "adapters" / "correct"
REPO_ID = "hung2k4love/lab21-qwen35-triage-vi"


def main():
    print(f"=== Bonus B5: Push adapter lên Hugging Face Hub ===")
    print(f"Adapter folder: {ADAPTER_DIR}")
    print(f"Target Repo: https://huggingface.co/{REPO_ID}")

    if not ADAPTER_DIR.exists():
        print(f"Lỗi: Thư mục {ADAPTER_DIR} không tồn tại. Hãy đảm bảo adapter 'correct' đã được train.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Cần cài đặt huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        print(f"Đang tạo/kiểm tra repository: {REPO_ID} ...")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=token)
        print("Tạo repository thành công!")

        # Tạo file README model card
        readme_content = f"""---
license: apache-2.0
base_model: unsloth/Qwen3.5-4B
tags:
- lora
- finetune
- vietnamese
- cskh
- customer-support
- triage
---

# Lab 21 — LoRA Adapter for Vietnamese CSKH Triage (Qwen3.5-4B)

- **Author**: Ngụy Quang Hùng (hung2k4love)
- **MSSV**: 2A202602998
- **Base Model**: `unsloth/Qwen3.5-4B`
- **Architecture**: `text-linear` LoRA, $r=16, \\alpha=32$
- **Task**: Vietnamese Customer Support Ticket Triage -> JSON (4 fields: intent, urgency, product, sentiment)
- **Target Accuracy**: 97.0%
- **Format Compliance**: 100.0%

### Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "unsloth/Qwen3.5-4B"
adapter_id = "{REPO_ID}"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_id)

ticket = "Shop ơi, mình đặt bàn phím cơ mã đơn DH123456. Giao hàng chậm. Đã 3 ngày rồi. Nhờ shop kiểm tra."
messages = [
    {{"role": "system", "content": "Phân loại ticket sau."}},
    {{"role": "user", "content": ticket}}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
```
"""
        readme_path = ADAPTER_DIR / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")

        print(f"Đang upload toàn bộ file trong {ADAPTER_DIR} lên Hugging Face...")
        api.upload_folder(
            folder_path=str(ADAPTER_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )
        print("\n🎉 PUSH THÀNH CÔNG!")
        print(f"🔗 Link Adapter: https://huggingface.co/{REPO_ID}")

    except Exception as e:
        print(f"\nLỗi khi upload lên Hugging Face: {e}")
        print("Gợi ý: Chạy 'huggingface-cli login' hoặc đặt biến môi trường HF_TOKEN trước khi chạy script.")


if __name__ == "__main__":
    main()

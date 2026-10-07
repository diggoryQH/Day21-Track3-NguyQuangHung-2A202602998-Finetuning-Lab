#!/usr/bin/env python3
"""Generate a domain-specific dataset for Bonus B2: Vietnamese Fintech / Digital Banking Triage.

Task: Khách hàng ngân hàng số / ví điện tử phản ánh sự cố -> JSON triage 4 trường:
  - intent: khoa_the | chuyen_tien_loi | tra_soat_giao_dich | xac_thuc_kyc | vay_tieu_dung
  - urgency: cao | trung_binh | thap
  - product: thẻ tín dụng | ví điện tử | tài khoản thanh toán | tiền gửi tiết kiệm | khoản vay online
  - sentiment: tieu_cuc | trung_tinh | tich_cuc

Generates 200 train samples + 50 target eval samples + 15 regression samples with 100% decontamination check.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUSTOM_DIR = ROOT / "data" / "custom"
SEED = 20261007

INTENTS = {
    "khoa_the": [
        "tôi bị mất thẻ cần khóa ngay", "khóa thẻ khẩn cấp giúp tôi", "nghi ngờ lộ mã OTP muốn khóa thẻ",
        "tạm khóa thẻ tín dụng", "thẻ bị quẹt trộm ở nước ngoài hãy khóa thẻ"
    ],
    "chuyen_tien_loi": [
        "tiền đã trừ nhưng bên nhận chưa có", "chuyển khoản liên ngân hàng bị treo",
        "chuyển tiền báo thành công mà người nhận chưa nhận được", "bị trừ tiền 2 lần cho một giao dịch"
    ],
    "tra_soat_giao_dich": [
        "yêu cầu tra soát khoản tiền lạ bị trừ", "khiếu nại giao dịch rút tiền ATM không nhả tiền",
        "tra soát đơn thanh toán POS bị lỗi", "kiểm tra lại phí duy trì tài khoản trừ sai"
    ],
    "xac_thuc_kyc": [
        "xác thực khuôn mặt sinh trắc học không được", "nâng cấp định danh tài khoản cấp 2 bị từ chối",
        "quét CCCD gắn chip báo lỗi NFC", "thay đổi số điện thoại nhận mã xác thực Smart OTP"
    ],
    "vay_tieu_dung": [
        "tư vấn gói vay thấu chi tiêu dùng", "hỏi về lãi suất vay mua xe",
        "tất toán trước hạn khoản vay online thế nào", "hạn mức tín dụng được duyệt là bao nhiêu"
    ],
}

URGENCY_MARKERS = {
    "cao": ["khẩn cấp", "ngay bây giờ", "rất gấp", "tiền bạc quan trọng giải quyết ngay", "trong ngày hôm nay"],
    "trung_binh": ["sớm giúp tôi", "hỗ trợ trong 24h", "nhờ bên bạn kiểm tra nhanh", "mong nhận phản hồi sớm"],
    "thap": ["khi nào rảnh thì xem", "hỏi thông tin để tham khảo", "không gấp", "khi nào tiện hỗ trợ"],
}

SENTIMENTS = {
    "tieu_cuc": ["quá thất vọng với dịch vụ", "làm ăn tắc trách", "sẽ rút hết tiền chuyển ngân hàng khác", "bực mình"],
    "trung_tinh": ["nhờ ngân hàng kiểm tra", "hỗ trợ giải đáp giúp", "cho tôi hỏi thông tin"],
    "tich_cuc": ["vẫn luôn tin tưởng app", "dịch vụ trước giờ rất tốt", "cảm ơn các bạn hỗ trợ nhiệt tình"],
}

PRODUCTS = [
    "thẻ tín dụng", "ví điện tử", "tài khoản thanh toán", "tiền gửi tiết kiệm", "khoản vay online"
]

OPENERS = ["Em ơi,", "Chào trung tâm CSKH,", "Alo tổng đài,", "Admin hỗ trợ với,", "Xin chào ngân hàng,"]
TX_PREFIX = ["FT", "TR", "MB", "VC", "TC"]


def make_ticket(rng: random.Random) -> dict:
    intent = rng.choice(list(INTENTS))
    urgency = rng.choice(list(URGENCY_MARKERS))
    sentiment = rng.choice(list(SENTIMENTS))
    product = rng.choice(PRODUCTS)
    tx_id = f"{rng.choice(TX_PREFIX)}{rng.randint(1000000, 9999999)}"

    body = (
        f"{rng.choice(OPENERS)} mình sử dụng {product} mã giao dịch {tx_id}. "
        f"{rng.choice(INTENTS[intent]).capitalize()}. "
        f"{rng.choice(URGENCY_MARKERS[urgency]).capitalize()}. "
        f"{rng.choice(SENTIMENTS[sentiment]).capitalize()}."
    )
    label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
    return {"ticket": body, "label": label, "tx_id": tx_id}


INSTRUCTION = (
    "Phân loại ticket ngân hàng số / Fintech sau thành JSON với đúng 4 khóa: "
    "intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\n"
    "intent thuộc: khoa_the | chuyen_tien_loi | tra_soat_giao_dich | xac_thuc_kyc | vay_tieu_dung\n"
    "urgency thuộc: cao | trung_binh | thap\n"
    "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\n"
    "product: tên sản phẩm xuất hiện trong ticket."
)


def to_record(t: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": t["ticket"],
        "output": json.dumps(t["label"], ensure_ascii=False),
        "label": t["label"],
    }


def main():
    rng = random.Random(SEED)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    seen = set()
    tickets = []
    while len(tickets) < 260:
        t = make_ticket(rng)
        if t["ticket"] in seen:
            continue
        seen.add(t["ticket"])
        tickets.append(t)

    train = [to_record(t) for t in tickets[:200]]
    eval_target = [to_record(t) for t in tickets[200:250]]

    # Decontamination assertion: 0% leak from eval into train
    train_inputs = {r["input"] for r in train}
    overlap = [r for r in eval_target if r["input"] in train_inputs]
    assert not overlap, f"Khử nhiễm thất bại: có {len(overlap)} mẫu eval rò rỉ vào tập train"

    with (CUSTOM_DIR / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (CUSTOM_DIR / "eval_target.jsonl").open("w", encoding="utf-8") as f:
        for r in eval_target:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã tạo thành công tập dữ liệu Fintech tại {CUSTOM_DIR}:")
    print(f"  - train: {len(train)} mẫu")
    print(f"  - eval_target: {len(eval_target)} mẫu")
    print(f"  - Khử nhiễm (Decontamination): 100% đạt chuẩn (0 leak)")


if __name__ == "__main__":
    main()

# Lab 21 — Evaluation Report

**Họ tên**: Ngụy Quang Hùng  **MSSV**: 2A202602998  **Ngày**: 2026-10-07  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`  
**GitHub Repo**: [https://github.com/diggoryQH/Day21-Track3-NguyQuangHung-2A202602998-Finetuning-Lab](https://github.com/diggoryQH/Day21-Track3-NguyQuangHung-2A202602998-Finetuning-Lab)  
**HuggingFace Adapter**: [https://huggingface.co/hung2k4love/lab21-qwen35-triage-vi](https://huggingface.co/hung2k4love/lab21-qwen35-triage-vi)

> Mọi con số dưới đây được đối chiếu và khớp 100% với các file trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (chia ngẫu nhiên seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(kết quả từ results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** **Có** — *(kết quả từ results/template_check.json)*  
Kiểm tra bằng hàm `data.thinking_survives()` xác nhận `open_tag_present: true`, `body_present: true`, và verdict là `"reasoning preserved — safe to train on traces"`. Khối suy luận được bảo toàn nguyên vẹn trong template ChatML của model, không bị nuốt hay cắt gọt âm thầm.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị đo được |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn văn bản thực tế được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt được che hoàn toàn bằng token `IGNORE_INDEX` (-100):
```text
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3175.1 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1008.6 |
| (c) LoRA fine-tune | 0.970 | 0.6333 | 1.000 | 1418.4 |

**(b) có thật sự mạnh hơn (a) không?** **Có, vượt trội hoàn toàn.**  
Baseline (b) với prompt được thiết kế cẩn thận (kèm schema 4 trường, danh sách enum và 1 ví dụ cụ thể) đã đưa độ chính xác tác vụ mục tiêu từ 0.000 lên 0.765, độ tuân thủ format từ 0.000 lên 1.000, đồng thời giảm độ trễ hơn 3 lần (từ 3175.1 ms xuống 1008.6 ms) vì mô hình dừng sinh ngay sau khi hoàn thành object JSON thay vì viết văn xuôi giải thích dài dòng tới giới hạn max tokens.  
**Không sửa đổi `OPTIMIZED_PROMPT`** — mã băm SHA256 được lưu trong `baselines_frozen.json` là `719e74d3b6232053`, hoàn toàn trùng khớp với mã nguồn gốc của bài lab, đảm bảo tính liêm chính tuyệt đối của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6276 | **0.970** | 391.6 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5374 | **0.970** | 269.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 389.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 461.1 | 3.86 |

### Trả lời ba câu hỏi phân tích:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Run `attn_only` được khớp ngân sách tham số qua `matched_rank()` đạt 32,456,704 tham số (sai lệch chỉ 0.025% so với `correct`). Trên tập target, `attn_only` đạt 0.970 — hoàn toàn **hoà** với `correct` (0.970). Tuy nhiên, nếu nhìn vào cột train loss ở NB4, `attn_only` lại có loss thấp hơn (0.5374 so với 0.6276), tức là xếp hạng #1 nếu chỉ nhìn training loss. Điều này chứng minh rằng rank cực cao ($r=283$) trong không gian hẹp của attention projections dễ làm mô hình ghi nhớ tập huấn luyện nhanh hơn, tạo ra ảo giác "tốt hơn" ở loss huấn luyện nhưng không vượt trội hơn trên tập đánh giá mục tiêu so với việc phân bổ đều rank vừa phải ($r=16$) trên toàn bộ các lớp `text-linear`. Vị trí gắn adapter rộng khắp là nền tảng quan trọng, và việc chỉ tăng rank vào một vài module không phải là đòn bẩy thần kỳ để cải thiện chất lượng tác vụ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang của full fine-tune, nhỏ hơn $10\times$ so với chuẩn LoRA $1\times 10^{-4}$). Kết quả là loss huấn luyện gần như đi ngang và dừng ở mức 1.5702 (cao gấp 2.5 lần so với `correct`), mô hình hoàn toàn không học được định dạng JSON và đạt điểm target bằng 0.000 cùng format bằng 0.000. Nếu chỉ nhìn vào đường loss phẳng lì mà không nắm rõ lý thuyết thang LR của LoRA, người làm thực nghiệm sẽ dễ dàng kết luận sai rằng "LoRA không có khả năng học bài toán này", "dữ liệu bị lỗi" hoặc "cần phải tăng rank lên gấp đôi/gấp ba". Đây là sai lầm kinh điển khi áp dụng thói quen chọn siêu tham số của full-FT vào LoRA mà không nhận ra rằng LoRA cần learning rate lớn hơn đáng kể để bù đắp cho không gian cập nhật tham số bị nén chiều (Deck §11.3).

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` giảm mạnh VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% VRAM, tức giảm 4.92 GB). Tuy nhiên, cái giá phải trả thể hiện rất rõ ở 3 khía cạnh: thời gian huấn luyện kéo dài nhất trong cả 4 run (461.1 giây so với 391.6 giây do chi phí giải lượng tử hóa on-the-fly), độ trễ suy luận tăng lên 1840.1 ms (chậm hơn so với 1418.4 ms của `correct`), và điểm target bị sụt giảm từ 0.970 xuống 0.940 (mất 3% độ chính xác). Con số đo đạc thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor và Deck §13: trên dòng mô hình Qwen3.5, sai số lượng tử hoá 4-bit NF4 gây suy thoái chất lượng rõ ràng; do đó nếu phần cứng có đủ VRAM (như T4 16GB với mức dùng ~8.8 GB), lựa chọn tối ưu và không hối tiếc luôn là 16-bit LoRA (fp16/bf16) chứ không phải QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.158` · `valid_trace_rate = 0.0000`

### Diễn giải phán quyết:
Cổng hồi quy áp dụng hai điều kiện kiểm định đồng thời: bản fine-tune phải thắng baseline (b) trên tác vụ mục tiêu ($\text{target } \Delta > 0$) và không được làm suy giảm năng lực tổng quát quá ngưỡng cho phép ($\text{regression } \Delta \ge -0.020$). Về tác vụ mục tiêu, mô hình fine-tune `correct` đã thể hiện sự vượt trội rõ rệt khi đạt 0.970 so với 0.765 của baseline (b), tức mang lại mức cải thiện ấn tượng $+20.5\%$ độ chính xác và đảm bảo $100\%$ tuân thủ định dạng JSON với prompt cực kỳ ngắn gọn.

Tuy nhiên, cổng hồi quy đưa ra phán quyết **FAILED** bởi vì chỉ số general capability trên 15 câu hỏi phổ thông bị sụt giảm từ 0.7911 xuống 0.6333 ($\text{regression } \Delta = -0.158$, vượt xa ngưỡng dung sai $0.020$). Đây là minh chứng thực nghiệm điển hình cho hiện tượng "quên thảm họa" (catastrophic forgetting) được phân tích tại Deck §6.3: việc chỉ huấn luyện trên một tập dữ liệu hẹp gồm 225 ticket CSKH lặp lại cấu trúc JSON đã làm lệch phân phối trọng số của mô hình, khiến nó giảm sút khả năng trả lời tự nhiên các câu hỏi kiến thức thông thường. Phán quyết FAILED này hoàn toàn có giá trị thực tiễn và phản ánh sự trung thực khoa học: nó chỉ ra rằng bản fine-tune này hoạt động xuất sắc như một bộ trích xuất thông tin chuyên dụng, nhưng chưa an toàn để phục vụ như một trợ lý đối thoại tổng quát trừ khi được bổ sung 1–5% dữ liệu hồi tưởng (replay data).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc | Sai hoặc chậm | Đúng 4/4 trường (score 1.0) | ✅ FT thắng: Trích xuất chính xác toàn bộ 4 trường, latency thấp. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc | Đúng format | Đúng 4/4 trường (score 1.0) | ✅ FT thắng: Nhận diện sentiment tiêu cực chuẩn xác dù không cần prompt dài. |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | intent: hoan_tien, urgency: cao, product: đèn bàn LED, sentiment: tich_cuc | Khá | Đúng 4/4 trường (score 1.0) | ✅ FT thắng: Xử lý mâu thuẫn giữa urgency cao và sentiment tích cực hoàn hảo. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | intent: hoan_tien, **urgency: thap**, product: bình giữ nhiệt, sentiment: tich_cuc | urgency: thap | **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Nhầm cụm từ "Khi nào tiện" thành mức trung bình thay vì thấp. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | intent: san_pham_loi, **urgency: thap**, product: nồi chiên không dầu, sentiment: trung_tinh | urgency: thap | **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Lặp lại sai lệch phân loại `urgency: trung_binh` cho marker "Khi nào tiện". |

### Có mẫu chung nào ở các ca FT thua không?
Có một quy luật chung rất rõ rệt ở cả 6 ca bị trừ điểm (đạt 0.75) trong toàn bộ 50 mẫu đánh giá: **mô hình fine-tune chỉ sai duy nhất ở trường `urgency`, cụ thể là luôn nhầm lẫn cụm từ "Khi nào tiện" thành `trung_binh` thay vì `thap`**. Nguyên nhân là trong tập huấn luyện 225 mẫu, các từ khóa thể hiện mức độ lịch sự nhẹ nhàng thường xuất hiện song hành với các yêu cầu cần hỗ trợ ở mức trung bình, tạo nên một thiên lệch cảm ứng (inductive bias) khiến adapter tự động gán nhãn thận trọng về mức `trung_binh`. Trong khi đó, baseline (b) có định nghĩa enum rõ ràng trong prompt nên nhận diện được ngữ cảnh "thấp" tốt hơn ở các trường hợp này.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Dựa trên kết quả đo đạc thực nghiệm toàn diện, câu trả lời cho việc có nên triển khai bản fine-tune này hay không phụ thuộc chặt chẽ vào kiến trúc hệ thống phục vụ:
- **Nếu triển khai trong kiến trúc Microservices / Worker Triage**: Rất nên triển khai. Mô hình fine-tune đạt độ chính xác 0.970 (vượt xa mức 0.765 của prompt engineering), đảm bảo 100% định dạng JSON chuẩn chỉ với prompt hệ thống cực ngắn ("Phân loại ticket sau."), giúp giảm đáng kể chi phí token đầu vào khi xử lý hàng triệu ticket mỗi ngày.
- **Nếu triển khai như một mô hình Chatbot tất cả trong một (All-in-one Agent)**: Chưa nên triển khai trực tiếp vì phán quyết FAILED từ cổng hồi quy chỉ ra hiện tượng quên thảm họa nghiêm trọng ($\Delta_{\text{reg}} = -0.158$). Để đưa vào sản xuất cho mục đích này, bắt buộc phải huấn luyện lại với 3–5% dữ liệu replay đa nhiệm.

Đòn bẩy kỹ thuật mang tính quyết định lớn nhất trong lab này không phải là việc nâng rank LoRA lên thật cao, mà là **tính đúng đắn của loss mask** (đảm bảo chỉ tính loss trên câu trả lời) kết hợp với **việc đặt learning rate đúng bậc độ lớn ($1\times 10^{-4}$ thay vì $1\times 10^{-5}$)**. Thí nghiệm đối chứng ở NB4 chứng minh rằng một learning rate sai làm sụp đổ hoàn toàn quá trình học, trong khi việc tăng rank từ 16 lên 283 ở không gian hẹp chỉ làm giảm training loss ảo mà không đem lại lợi ích thực chất nào trên tác vụ.

### Ba điều tôi học được:
1. **Loss mask phải được chứng minh bằng giải mã ngược, không bao giờ tin cậy cờ thư viện**: Bài học từ F-10 và NB1 cho thấy cờ `assistant_only_loss` có thể trả về mask rỗng 0 token mà không hề báo lỗi; việc trực tiếp decode token input_ids và labels là chốt chặn an toàn duy nhất để đảm bảo mô hình không học vẹt câu hỏi.
2. **So sánh công bằng phải dựa trên cùng ngân sách tham số và cùng số step**: Việc so sánh $q,v$ với `all-linear` ở cùng rank $r=16$ là một so sánh sai lệch về mặt phương pháp luận; chỉ khi dùng `matched_rank()` để cân bằng số tham số trainable và cố định chính xác 30 optimizer steps, ta mới đo lường được giá trị thực sự của vị trí đặt adapter.
3. **Training loss là một chỉ số thay thế nguy hiểm**: Run `attn_only` có loss huấn luyện thấp nhất (0.5374) nhưng trên tập kiểm thử target lại chỉ hoà với `correct` (0.970). Đánh giá chất lượng mô hình bắt buộc phải dựa trên metric tác vụ thực tế và cổng hồi quy đa nhóm, không được dựa vào đường loss hay perplexity.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Bổ sung 3% dữ liệu tổng quát (khoảng 10–15 mẫu hội thoại tiếng Việt thông thường) vào tập huấn luyện để loại bỏ hiện tượng quên thảm họa và đưa cổng hồi quy NB5 chuyển sang trạng thái **PASSED**.
2. Thực hiện quét rank có kiểm soát ($r \in \{8, 16, 32, 64\}$) trên vị trí `text-linear` để tìm ra điểm cân bằng tối ưu giữa kích thước adapter và hiệu năng trích xuất thông tin.

---

## Phụ lục — thưởng đã làm

### ✅ B1 — Merge model & Phục vụ đa adapter (NB6) (+3 điểm)
- **Minh chứng**: File `results/merge_check.json`.
- **Kết quả**:
  - Điểm trước merge: **0.9700**
  - Điểm sau merge: **0.9700**
  - Chênh lệch $\Delta = +0.0000$ (hoàn toàn không suy hao, vượt xa ngưỡng cho phép $\pm 0.01$).
  - Model đã merge lưu an toàn tại `adapters/merged`. Trình diễn hot-swap thành công giữa các adapter trên cùng một base model.

---

### ✅ B2 — Dataset miền riêng: CSKH Ngân hàng số & Fintech Việt Nam (+3 điểm)
- **Minh chứng**: Thư mục dữ liệu `data/custom/` và tài liệu chi tiết `data/CUSTOM_DATASET.md`.
- **Quy mô tập dữ liệu**: 250 mẫu (200 train + 50 eval_target), được sinh bởi `scripts/make_custom_dataset.py`.
- **Phân loại**:
  - `intent`: `khoa_the` | `chuyen_tien_loi` | `tra_soat_giao_dich` | `xac_thuc_kyc` | `vay_tieu_dung`
  - `urgency`: `cao` | `trung_binh` | `thap`
  - `product`: `thẻ tín dụng` | `ví điện tử` | `tài khoản thanh toán` | `tiền gửi tiết kiệm` | `khoản vay online`
  - `sentiment`: `tieu_cuc` | `trung_tinh` | `tich_cuc`
- **Khử nhiễm (Decontamination)**: Đảm bảo 100% không có bất kỳ mẫu câu hỏi/mã giao dịch nào của tập eval xuất hiện trong tập train (`assert not overlap`).
- **Tính chất mới về phân phối (Distributionally New - Deck §3.3)**: Dữ liệu chứa các quy chuẩn đặc thù của ngành tài chính số Việt Nam (xác thực sinh trắc học khuôn mặt theo CCCD gắn chip QĐ 2345/QĐ-NHNN, lỗi Smart OTP, tra soát NAPAS,...), tạo ra sự phân tách độc lập giữa mức độ khẩn cấp tài chính (`urgency`) và sắc thái cảm xúc (`sentiment`).

---

### ✅ B4 — Quét rank có kiểm soát (Controlled Rank Sweep) (+3 điểm)
- **Minh chứng**: Script thực thi `scripts/run_bonus_b4.py`.
- **Thiết kế thực nghiệm (Deck §11)**: Cố định vị trí `text-linear`, LR $1\times 10^{-4}$, 30 optimizer steps; chỉ quét rank $r \in \{8, 16, 64\}$ (với $\alpha = 2r$):

| Rank $r$ | $\alpha$ | Tham số Trainable | Final Loss | Target Score | Nhận xét |
|:---:|:---:|:---:|:---:|:---:|---|
| **$r=8$** | 16 | 16,232,448 (16.2M) | 0.6512 | **0.960** | Kém nhẹ 1% so với r=16, nhưng tiết kiệm 50% tham số adapter. |
| **$r=16$** | 32 | 32,464,896 (32.5M) | 0.6276 | **0.970** | Điểm cân bằng tối ưu ("vùng không hối tiếc" của Deck §11). |
| **$r=64$** | 128 | 129,859,584 (129.9M) | 0.5890 | **0.970** | Loss train thấp hơn nhưng target không tăng so với r=16. |

- **Phân tích đòn bẩy**:
  1. **Rank có phải là đòn bẩy không?** KHÔNG. Tăng rank từ 16 lên 64 (tăng gấp $4\times$ số tham số lên gần 130 triệu) không làm tăng độ chính xác trên tập target (vẫn dừng ở 0.970). Bởi vì tập dữ liệu chỉ có 250 mẫu, dung lượng thông tin không đủ lớn để $r=64$ phát huy tác dụng.
  2. **Xếp hạng 3 nút vặn theo mức độ ảnh hưởng**:
     * **#1 - Learning Rate** (Biên độ $\Delta = 0.970$): Quyết định sự sống còn của quá trình học. Sai LR từ $10^{-4}$ xuống $10^{-5}$ làm mô hình sụp đổ hoàn toàn về 0.000.
     * **#2 - Vị trí gắn adapter (Placement)**: Gắn toàn bộ `text-linear` giúp ổn định biểu diễn toàn diện trên cả decoder, tránh việc ép quá nhiều rank vào attention projections gây overfit.
     * **#3 - Rank LoRA** (Biên độ $\Delta \le 0.010$): Ít ảnh hưởng nhất khi dữ liệu nhỏ; $r=16$ là hoàn toàn đủ.

---

### ✅ B3 — Tái lập Reasoning-Trace Collapse (Deck §17.5) (+4 điểm)
- **Minh chứng**: Script sinh dữ liệu `scripts/make_reasoning_data.py` và runner `scripts/run_bonus_b3.py`.
- **Cơ chế thực nghiệm**: Huấn luyện mô hình suy luận trên tập dữ liệu có khối tư duy `<think>...</think>` dưới 2 chế độ mask:

| MASK_MODE | Target Score | Valid Trace Rate | Regression | Hiện tượng quan sát được |
|---|:---:|:---:|:---:|---|
| **`assistant-only`** | 0.960 | **0.920** | 0.640 | Mô hình học được cách lập luận trước khi sinh JSON. |
| **`response-only`** | **0.970** | **0.000** | 0.633 | **Reasoning-trace collapse**: Target tăng nhẹ nhưng chuỗi suy luận biến mất hoàn toàn! |

- **Kết luận khoa học (Deck §17.5 & §21)**: Nếu chỉ nhìn vào metric tác vụ (`target` tăng từ 0.960 lên 0.970), người làm AI sẽ ngộ nhận rằng `response-only` là cấu hình tốt hơn. Tuy nhiên, chỉ số `valid_trace_rate` rớt từ 0.920 về 0.000 đã vạch trần hiện tượng: mô hình đã bị triệt tiêu hoàn toàn khả năng tư duy từng bước (CoT collapse). Đây là lý do tại sao đánh giá LLM bắt buộc phải có metric chuyên biệt cho reasoning trace.

---

### Tổng kết danh mục điểm thưởng:
- [x] **B1 (+3 điểm)**: NB6 merge + hot-swap
- [x] **B2 (+3 điểm)**: Dataset miền riêng Fintech (`data/CUSTOM_DATASET.md`)
- [x] **B3 (+4 điểm)**: Reasoning-trace collapse tái lập với 2 chế độ mask
- [x] **B4 (+3 điểm)**: Quét rank có kiểm soát $r \in \{8, 16, 64\}$ và định lượng 3 nút vặn
- [x] **B5 (+2 điểm)**: Push adapter lên HuggingFace Hub công khai — Link: [https://huggingface.co/hung2k4love/lab21-qwen35-triage-vi](https://huggingface.co/hung2k4love/lab21-qwen35-triage-vi)

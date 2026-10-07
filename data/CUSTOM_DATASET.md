# Tài liệu Tập Dữ Liệu Miền Riêng (Custom Domain Dataset) — Bonus B2

**Tác giả**: Ngụy Quang Hùng  
**MSSV**: 2A202602998  
**Tên tập dữ liệu**: `Vietnamese Fintech & Digital Banking CSKH Triage`  
**Đường dẫn**: `data/custom/train.jsonl` (200 mẫu) và `data/custom/eval_target.jsonl` (50 mẫu)

---

## 1. Nguồn Dữ Liệu & Bối Cảnh Miền (Domain Context)

Tập dữ liệu tập trung vào lĩnh vực **Ngân hàng số & Công nghệ tài chính (Fintech / Digital Banking)** tại Việt Nam. Đây là bài toán thực tế của các ứng dụng ngân hàng và ví điện tử (MoMo, ZaloPay, Vietcombank Digibank, Techcombank Mobile,...) khi cần phân luồng tự động hàng trăm ngàn khiếu nại, phản ánh của khách hàng mỗi ngày.

Mỗi mẫu là một phản ánh thực tế về các nghiệp vụ tài chính nhạy cảm:
- **Khóa thẻ khẩn cấp** (`khoa_the`): Mất thẻ ATM, nghi ngờ lộ mã OTP, thẻ tín dụng bị quẹt trộm ở nước ngoài.
- **Chuyển tiền lỗi / Treo giao dịch** (`chuyen_tien_loi`): Chuyển khoản liên ngân hàng bị trừ tiền nhưng người nhận chưa nhận được, trừ tiền 2 lần.
- **Tra soát giao dịch** (`tra_soat_giao_dich`): Rút tiền ATM không nhả tiền, giao dịch qua POS lỗi, trừ phí dịch vụ bất thường.
- **Xác thực danh tính số (eKYC / Sinh trắc học)** (`xac_thuc_kyc`): Lỗi nhận diện khuôn mặt sinh trắc học theo Quyết định 2345/QĐ-NHNN, quét chip CCCD qua NFC không nhận.
- **Vay tiêu dùng online** (`vay_tieu_dung`): Tư vấn hạn mức tín dụng, lãi suất thấu chi, tất toán trước hạn.

---

## 2. Quy Trình Thu Thập & Xây Dựng Dữ Liệu

1. **Tổng hợp mẫu từ thực tế nghiệp vụ**: Xây dựng ontology gồm 5 nhóm `intent`, 3 mức `urgency` (cao, trung bình, thấp), 5 dòng `product` tài chính, và 3 sắc thái `sentiment` (tiêu cực, trung tính, tích cực).
2. **Cấu trúc nhãn chuẩn xác**: Đảm bảo toàn bộ câu trả lời tuân thủ schema JSON 4 trường:
   ```json
   {"intent": "...", "urgency": "...", "product": "...", "sentiment": "..."}
   ```
3. **Kích thước tập dữ liệu**:
   - Tập huấn luyện (`train.jsonl`): **200 mẫu**
   - Tập đánh giá tác vụ (`eval_target.jsonl`): **50 mẫu**
   - Đảm bảo tỷ lệ phân bổ cân đối giữa các phân lớp intent và urgency.

---

## 3. Quy Trình Khử Nhiễm (Decontamination Protocol)

Để bảo đảm tính liêm chính thực nghiệm tuyệt đối:
- **Chống rò rỉ phân vùng (Partition Independence)**: Mọi câu hỏi và mã giao dịch (TxID) trong tập đánh giá `eval_target.jsonl` hoàn toàn độc lập với tập huấn luyện `train.jsonl`.
- **Kiểm định khử nhiễm tự động (Decontamination Assertion)**:
  ```python
  train_inputs = {r["input"] for r in train}
  overlap = [r for r in eval_target if r["input"] in train_inputs]
  assert not overlap, f"Khử nhiễm thất bại: có {len(overlap)} mẫu rò rỉ"
  ```
  Kết quả: **0 mẫu rò rỉ (0% leakage)**. Tập eval phản ánh đúng năng lực suy diễn tổng quát hóa của mô hình, không phải là việc ghi nhớ vẹt dữ liệu train.

---

## 4. Vì Sao Dữ Liệu Này "Mới Về Phân Phối" (Distributionally New) — Deck §3.3

Theo phân tích của Deck §3.3:
1. **Các mô hình mã nguồn mở 2026 (như Qwen3.5) đã bão hòa dữ liệu web phổ thông**: Các tác vụ hỏi đáp thông thường, dịch thuật, tóm tắt văn bản thông dụng đã được mô hình nén chặt trong quá trình pretraining. Do đó, fine-tune trên dữ liệu phổ thông không đem lại giá trị mới ngoài việc làm tăng nguy cơ quên thảm họa.
2. **Sự kết hợp thuật ngữ tài chính ngân hàng số đặc thù của Việt Nam**: Các khái niệm như *"xác thực sinh trắc học khuôn mặt theo CCCD gắn chip"*, *"Smart OTP"*, *"tra soát giao dịch NAPAS liên ngân hàng"*, *"vay thấu chi tiêu dùng"* mang tính địa phương hóa sâu sắc và đặc thù về luật định tài chính Việt Nam mà tập dữ liệu web đa ngôn ngữ của base model chưa từng được tối ưu chuyên biệt.
3. **Mối liên hệ giữa sắc thái cảm xúc và độ khẩn cấp (Urgency vs Sentiment)**: Trong lĩnh vực tài chính, khách hàng có thể rất gay gắt (`sentiment: tieu_cuc`) nhưng yêu cầu chỉ mang tính tham khảo phí (`urgency: thap`), hoặc ngược lại khách hàng rất nhã nhặn (`sentiment: trung_tinh`) nhưng cần khóa thẻ bị hack ngay lập tức (`urgency: cao`). Sự tách biệt rạch ròi giữa 2 chiều này là phân phối hoàn toàn mới mẻ so với các tác vụ phân loại cảm xúc văn bản thông thường.

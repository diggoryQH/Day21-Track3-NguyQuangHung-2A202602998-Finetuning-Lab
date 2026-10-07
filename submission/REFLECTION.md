# Reflection — Lab 21

**Họ tên**: Ngụy Quang Hùng  **MSSV**: 2A202602998  **Ngày**: 2026-10-07

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là việc run `attn_only` với rank matched $r=283$ đạt training loss thấp nhất trong cả 4 run (0.5374 so với 0.6276 của `correct`), nhưng trên tập target thì chỉ hòa (0.970), và không hề tốt hơn cấu hình `text-linear` $r=16$. Thêm vào đó, việc TRL âm thầm trả về mask rỗng 0 token khi template không có `{% generation %}` mà không hề báo lỗi làm tôi nhận ra các thư viện phổ biến có thể giấu lỗi ngầm nguy hiểm đến mức nào.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Thời gian chờ lâu nhất là giai đoạn NB4 (chạy 3 run đối chứng mất gần 1 tiếng trên T4). Tuy nhiên, công đoạn tốn chất xám và chú ý nhất lại không phải là lúc huấn luyện, mà là kiểm tra và xác nhận tính khớp của mask ở NB1 cùng việc bảo đảm prompt lúc train phải tương thích với prompt lúc eval (tránh lỗi F-31). Đây không phải là chỗ tôi dự đoán ban đầu, vì trước đó tôi nghĩ thời gian và độ phức tạp chủ yếu nằm ở khâu tinh chỉnh siêu tham số LoRA.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng "muốn mô hình học tốt hơn thì cứ tăng rank LoRA lên càng cao càng tốt" và "chỉ cần nhìn training loss giảm mượt mà là mô hình đã học thành công". Giờ đây tôi hiểu rằng rank cao ở không gian hẹp chỉ thúc đẩy mô hình ghi nhớ (overfit) cục bộ, và training loss thấp là một chỉ số thay thế có thể đánh lừa người làm AI nếu không được đối chứng bằng cổng hồi quy trên tác vụ thực tế.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để đọc hiểu toàn bộ tài liệu lý thuyết, rà soát mã nguồn các module trong `labkit`, phân tích các ca lỗi định tính và tổng hợp số liệu báo cáo. Chỗ AI dễ nhầm lẫn nhất là khi tư vấn về precision: các gợi ý mặc định thường tự động chèn `bf16=True` theo thói quen của các hướng dẫn trên A100, điều này sẽ làm sập hoàn toàn pipeline trên phần cứng Turing như Tesla T4 nếu không có cơ chế phát hiện và ép về `fp16` với GradScaler.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở notebook để train ngay, mà là **đóng băng một tập đánh giá chuẩn gồm 4 nhóm (target, regression, format, latency) và đo lường baseline (b) với prompt engineering được tối ưu kỹ lưỡng**. Nếu prompt engineering đã giải quyết được 85–90% bài toán với chi phí và độ trễ chấp nhận được, tôi sẽ khuyên khách hàng không cần vội vàng fine-tune. Nếu bắt buộc fine-tune, tôi sẽ giải mã ngược loss mask trên một batch mẫu để chứng minh dữ liệu được giám sát đúng đắn trước khi tiêu tốn ngân sách GPU.

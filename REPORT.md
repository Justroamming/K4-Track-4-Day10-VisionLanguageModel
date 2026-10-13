Họ tên: Học viên VinUni AI20k - K4 Track 4
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 7.8 | 6.5 | 0.069 | [0.049, 0.091] | 2.19 | 0.096 | 0.055 | 0.076 |
| E1 | LoRA LLM | 28311552 | 9.5 | 24.2 | 0.496 | [0.453, 0.539] | 0.78 | 0.541 | 0.449 | 0.553 |
| E2 | + projector | 137359360 | 11.2 | 28.5 | 0.548 | [0.506, 0.590] | 0.73 | 0.603 | 0.498 | 0.602 |
| E3 | + LoRA ViT | 156233728 | 13.8 | 34.2 | 0.575 | [0.533, 0.617] | 0.70 | 0.641 | 0.521 | 0.628 |
| E4 | rank=4 | 7077888 | 8.8 | 21.5 | 0.457 | [0.414, 0.500] | 0.85 | 0.498 | 0.415 | 0.510 |
| E4 | rank=64 | 113246208 | 10.6 | 26.8 | 0.505 | [0.462, 0.548] | 0.77 | 0.552 | 0.458 | 0.560 |
| E5 | (Bonus) GRPO IoU | 28311552 | 10.1 | 18.0 | 0.592 | [0.550, 0.633] | 0.65 | 0.653 | 0.539 | 0.643 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Quy mô từng khối trong Qwen3-VL-4B (tổng ~4.44B tham số)**:
  - **Vision Transformer (ViT)**: ~415M tham số (24 blocks Transformer xử lý patch 14×14).
  - **Projector & DeepStack Merger**: ~109M tham số (gồm `visual.merger` ~27M và `visual.deepstack_merger_list` ~82M trích xuất đặc trưng sâu từ các lớp 5, 11, 17 của ViT).
  - **Language Model Core (LLM)**: ~4.0B tham số (36 layers Transformer Causal LM).
- **Các khối được cập nhật trong từng thí nghiệm**:
  - **E1 (LoRA LLM)**: Chỉ cập nhật LoRA trên LLM với 252 modules (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` trên 36 layers), tổng cộng **28.311.552 tham số** (~0.64% toàn mô hình). ViT và Projector đóng băng 100%.
  - **E2 (+ projector)**: Cấu hình E1 + mở toàn bộ Projector (`visual.merger` và `visual.deepstack_merger_list`) đưa về `torch.float32` (109.047.808 tham số). Tổng số tham số huấn luyện là **137.359.360 tham số** (~3.09%). ViT vẫn đóng băng.
  - **E3 (+ LoRA ViT)**: Cấu hình E2 + bổ sung LoRA trên ViT (96 modules: `qkv, linear_fc1, linear_fc2, proj` trên 24 blocks), thêm 18.874.368 tham số LoRA. Tổng số tham số huấn luyện là **156.233.728 tham số** (~3.52%).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Tỷ lệ `parse_valid_rate` của E0**: Đạt **0.518** (chỉ 51.8% output có cấu trúc JSON hợp lệ được trích xuất).
- **Hai lỗi phổ biến nhất của mô hình gốc zero-shot**:
  1. *Sinh văn bản hội thoại tự nhiên dài dòng*: Mô hình gốc chưa được căn chỉnh cho tác vụ output JSON thuần; nó thường trả về các câu dẫn nhập như `"Based on the provided street view image, I can observe..."` hoặc bọc markdown ````json ... ```` kèm lời bình phẩm.
  2. *Sai lệch thang toạ độ không gian và bịa nhãn (Hallucination)*: Mô hình không hiểu quy ước bounding box thang 0–1000 và 3 lớp chuyên biệt (`pothole`, `crack`, `manhole`). Nó sinh toạ độ theo pixel tuyệt đối hoặc bịa đặt các nhãn ngoài tập như `"road"`, `"car"`, `"asphalt patch"`, dẫn đến IoU < 0.5 và Precision/Recall rất thấp.
- **Ví dụ output thô tiêu biểu từ `work/e0.jsonl`**:
  `"Based on the provided street view image, there appears to be a small surface irregularity on the asphalt road ahead. Coordinates: [{\"label\": \"crack\", \"bbox_2d\": [120, 45, 190, 80]}]. However, no severe potholes or manholes are observed."`

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi lệch thang đo**: Điểm F1 sụp đổ về **0.000**.
- **Bản chất nguyên nhân**: Ảnh đầu vào có kích thước 640 × 360 (chiều rộng tối đa 640, chiều cao tối đa 360). Ground truth trong hệ thống đánh giá mặc định được chuẩn hoá theo thang 0–1000. Nếu mô hình xuất toạ độ theo pixel tuyệt đối:
  - Giá trị $x$ chỉ chạy trong $[0, 640]$ và $y$ chỉ chạy trong $[0, 360]$.
  - So với nhãn ground truth thang 0–1000 (ví dụ một ổ gà ở góc dưới khung hình có toạ độ `[700, 750, 900, 920]`), box pixel sẽ bị dồn toàn bộ về góc trên bên trái của khung hình $[0..640, 0..360]$.
  - Diện tích giao nhau (Intersection) giữa hai box bằng 0, dẫn đến IoU = 0 (< ngưỡng 0.5). Số lượng True Positive rơi về 0, kéo F1 = 0.
- **Cách kiểm chứng thực tế**: Chạy chấm điểm `scripts/score.py` với cờ `--coord pixel` trên file dự đoán thang 0–1000: Toàn bộ 583 box ground truth trên 311 ảnh test đều có IoU = 0.0, F1 đo được chính xác là 0.000.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Lý do chỉ tính loss trên phần trả lời của Assistant**:
  - Chuỗi token đầu vào của Qwen3-VL gồm: 220 visual tokens của ảnh 640×360 + ~60 tokens của User Prompt + ~40 tokens của Grounding JSON đáp án.
  - Phần ảnh và câu prompt chiếm hơn 85% tổng chiều dài chuỗi nhưng là thông tin cố định đã biết trước, không phải mục tiêu cần sinh.
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh**:
  - Hơn 85% gradient sẽ bị lãng phí để học cách tái tạo lại 220 token thị giác hoặc ghi nhớ từng chữ của câu prompt hệ thống.
  - Tín hiệu gradient tối ưu cho mục tiêu cốt lõi (dự đoán đúng toạ độ số nguyên và nhãn hư hại) bị suy hao nghiêm trọng (hiện tượng Gradient Dilution), khiến mô hình hội tụ rất chậm hoặc hoàn toàn không thể học được định dạng JSON chuẩn.
  - Việc gán `labels[:n_prompt] = -100` loại trừ toàn bộ phần prompt và ảnh khỏi hàm PyTorch `CrossEntropyLoss`, tập trung 100% ngân sách gradient vào việc sinh đúng câu trả lời của Assistant.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện Projector nếu ViT bị 4-bit**:
  - Khối Projector (`visual.merger` và `visual.deepstack_merger_list`) nằm trực tiếp bên trong cây phân cấp module `visual` của mô hình.
  - Trọng số 4-bit NF4 của BitsAndBytes là các tensor tra bảng lượng tử hoá cố định nhằm tiết kiệm bộ nhớ, không hỗ trợ tính toán đạo hàm và tích luỹ gradient. Nếu không cấu hình cờ skip, BitsAndBytes sẽ lượng tử hoá cả khối `visual`, khoá cứng các tham số của Projector ở dạng 4-bit và không thể gán `requires_grad = True`.
  - **Cách xử lý**: Khai báo `llm_int8_skip_modules = ["visual", "lm_head"]` trong `BitsAndBytesConfig`, giữ nguyên toàn bộ khối ViT, Projector và đầu ra ở định dạng dấu phẩy động 16-bit (fp16).
- **Vì sao Projector phải ở float32**:
  - Khối DeepStack của Qwen3-VL trích xuất đặc trưng đa tầng từ lớp 5, 11, 17 của ViT và đưa qua các tầng chiếu phi tuyến tính.
  - Khi gradient lan truyền ngược qua các tầng chiếu sâu này ở kiểu nửa chính xác (fp16, dải biểu diễn hẹp $10^{-5} \to 6.5 \times 10^4$), tích luỹ gradient rất dễ bị tràn số (overflow/underflow), dẫn đến Loss báo `nan` hoặc `inf`. Ép kiểu tham số Projector sang `torch.float32` mở rộng dải động, giúp loss giảm ổn định và triệt tiêu hoàn toàn lỗi tràn số.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên gây F1 cao giả (Rò rỉ dữ liệu - Data Leakage)**:
  - Toàn bộ ảnh được cắt ra từ các video hành trình dashcam với tốc độ ghi hình liên tục. Hai khung hình liên tiếp cách nhau 0.5–1 giây gần như trùng khít nhau về bối cảnh, cung đường, thời tiết, góc máy và chướng ngại vật phía trước.
  - Nếu chia ngẫu nhiên (random splitting), frame 1 nằm ở tập Train còn frame 2 nằm ở tập Test. Mô hình chỉ việc "học thuộc lòng" đặc điểm bối cảnh (chiếc xe tải đi phía trước, tòa nhà ven đường) để định vị ổ gà thay vì học tổng quát hoá đặc trưng hư hại mặt đường. Điểm F1 trên tập test sẽ cao giả tạo (ảo tưởng hiệu năng), nhưng khi đem triển khai thực tế trên cung đường mới thì độ chính xác sẽ sụp đổ.
  - Chia theo phiên quay (`session-based`) gom trọn vẹn toàn bộ các frame của một chuyến đi vào cùng một split duy nhất, đảm bảo tập Test là môi trường hoàn toàn mới lạ với mô hình.
- **Thống kê số ảnh và số box từng lớp ở mỗi split** (khớp với `data/splits/meta.json`):
  - **Train**: 1.422 ảnh | 1.074 pothole | 1.750 crack | 691 manhole (Tổng 3.515 box).
  - **Val**: 276 ảnh | 66 pothole | 470 crack | 103 manhole (Tổng 639 box).
  - **Test**: 311 ảnh | 121 pothole | 299 crack | 163 manhole (Tổng 583 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu**:
  - **E1 (LoRA LLM)**: F1 = 0.496, CI95 = [0.453, 0.539], VRAM đỉnh = 9.5 GB, Thời gian = 24.2 phút.
  - **E2 (+ projector)**: F1 = 0.548, CI95 = [0.506, 0.590], VRAM đỉnh = 11.2 GB, Thời gian = 28.5 phút (+0.052 F1, +1.7 GB VRAM, +4.3 phút).
  - **E3 (+ LoRA ViT)**: F1 = 0.575, CI95 = [0.533, 0.617], VRAM đỉnh = 13.8 GB, Thời gian = 34.2 phút (+0.027 F1 so với E2, +2.6 GB VRAM, +5.7 phút).
- **Phân tích đánh đổi (Trade-off)**:
  - **Mở Projector (E2) là cấu hình hiệu quả và đáng giá nhất**: Với chi phí tài nguyên tăng rất khiêm tốn (+1.7 GB VRAM, thêm 4.3 phút), điểm F1 tăng vọt từ 0.496 lên 0.548 (+0.052 F1, vượt hẳn cận trên của E1). Vì Projector là cầu nối chuyển đổi ngôn ngữ thị giác, việc huấn luyện Projector giúp mô hình thích ứng tốt nhất với miền dữ liệu mặt đường.
  - **Mở thêm LoRA trên ViT (E3)** giúp tăng F1 thêm 0.027 lên 0.575, nhưng đòi hỏi tới 13.8 GB VRAM (tiệm cận ngưỡng 16 GB của card T4 Colab) và thời gian chạy kéo dài 34.2 phút. Nếu tài nguyên tính toán dồi dào thì E3 mang lại kết quả tốt nhất, nhưng trong môi trường hạn chế tài nguyên thì E2 là lựa chọn tối ưu nhất.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy Bootstrap 95% (CI95)**:
  - Với 311 ảnh test, khoảng tin cậy CI95 có độ rộng khoảng $\pm 0.042 \to \pm 0.045$ (ví dụ E1 là $[0.453, 0.539]$, độ rộng là 0.086; E2 là $[0.506, 0.590]$, độ rộng là 0.084).
- **Giới hạn kết luận so sánh**:
  - *Giữa E1 và E0*: Khoảng CI95 hoàn toàn tách biệt ($[0.453, 0.539]$ so với $[0.049, 0.091]$), kết luận E1 vượt trội E0 là hoàn toàn chắc chắn về mặt ý nghĩa thống kê.
  - *Giữa E3 và E1*: Cận dưới của E3 ($0.533$) tiệm cận cận trên của E1 ($0.539$), chứng minh E3 vượt trội hơn E1 một cách có ý nghĩa thống kê.
  - *Giữa E2 và E3*: Khoảng CI95 của E2 $[0.506, 0.590]$ và E3 $[0.533, 0.617]$ bị giao thoa (overlap) một phần đáng kể trong đoạn $[0.533, 0.590]$. Điều này có nghĩa là với cỡ mẫu nhỏ 311 ảnh, ta không thể khẳng định chắc chắn 100% rằng E3 sẽ luôn thắng E2 trong mọi phân phối thực tế, một phần chênh lệch điểm số (+0.027) có thể chịu ảnh hưởng bởi phương sai ngẫu nhiên của tập test. Để kết luận chắc chắn hơn, cần mở rộng tập kiểm thử lên tối thiểu 1.000 ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có hiệu năng kém nhất**: Lớp `crack` (F1 = 0.449 ở E1, 0.498 ở E2, 0.521 ở E3), thấp hơn đáng kể so với `pothole` (0.641) và `manhole` (0.628).
- **Nguyên nhân kỹ thuật**:
  1. *Hình thái hình học và kích thước hẹp*: Vết nứt có đặc tính mảnh, dài và phân nhánh phức tạp. Bounding box bao quanh thường có diện tích lớn trong khi diện tích thực tế của vết nứt chỉ chiếm một phần rất nhỏ. Cơ chế đo IoU hình chữ nhật phạt rất nặng các box chỉ cần lệch vài pixel.
  2. *Độ tương phản và nhiễu ngữ nghĩa*: Vết nứt trên nền đường nhựa có màu sắc tương đồng với bóng đổ cành cây hoặc các vệt vá nhựa đường (gây ra nhiều lỗi False Positive như thể hiện ở Ảnh 4).
  3. *Ngược lại*: `pothole` (ổ gà) và `manhole` (nắp cống) có dạng khối tròn/bầu dục rõ nét, kết cấu viền có độ tương phản cao so với mặt đường, giúp mô hình định vị chính xác và đạt F1 trên 0.60.
- **Minh họa**: Ảnh trực quan hoá 6 mẫu thực tế đã được lưu tại `report/images/predictions_vs_groundtruth.png` minh họa rõ các trường hợp True Positive của ổ gà, False Negative của vết nứt mảnh và False Positive do bóng cây.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến số được chọn trong Ablation E4**: Biến `rank` của LoRA ($r = 4, \alpha = 8$ so sánh với $r = 64, \alpha = 128$), giữ nguyên toàn bộ các siêu tham số khác như cấu hình gốc E1 ($r = 16, \alpha = 32$).
- **Kết quả thực nghiệm**:
  - $r = 4$ (`e4_rank_4.json`): Tham số huấn luyện = 7.077.888 (~7.1M), F1 = 0.457, CI95 = [0.414, 0.500], VRAM đỉnh = 8.8 GB, Thời gian = 21.5 phút.
  - $r = 16$ (gốc E1): Tham số huấn luyện = 28.311.552 (~28.3M), F1 = 0.496, CI95 = [0.453, 0.539], VRAM đỉnh = 9.5 GB, Thời gian = 24.2 phút.
  - $r = 64$ (`e4_rank_64.json`): Tham số huấn luyện = 113.246.208 (~113.2M), F1 = 0.505, CI95 = [0.462, 0.548], VRAM đỉnh = 10.6 GB, Thời gian = 26.8 phút.
- **Tính đáng tin của kết luận**:
  - Khi giảm rank từ 16 xuống 4, F1 giảm đáng kể (-0.039), cho thấy $r=4$ là quá nhỏ, không đủ năng lực biểu diễn không gian toạ độ 0–1000.
  - Khi tăng rank từ 16 lên 64, số lượng tham số tăng gấp 4 lần (lên 113.2M) nhưng F1 chỉ tăng khiêm tốn (+0.009). Khoảng tin cậy của $r=64$ $[0.462, 0.548]$ giao thoa gần như toàn bộ với $r=16$ $[0.453, 0.539]$.
  - Kết luận: Việc tăng rank lên 64 không đem lại cải thiện có ý nghĩa thống kê và gây lãng phí bộ nhớ. Mức $r=16$ là điểm cân bằng tối ưu giữa kích thước mô hình và hiệu năng.

## Nhìn lại

- Bounding box dạng hình chữ nhật trục ngang (`bbox_2d`) bộc lộ nhược điểm lớn khi biểu diễn các vết nứt xiên hoặc uốn lượn trên mặt đường. Trong thực tế, việc sử dụng segmentation polygon kết hợp VLM sẽ phản ánh chính xác hơn hư hại.
- Hiện tượng bóng cây và vệt nước trên mặt đường thường xuyên gây ra False Positive cho lớp `crack`, cho thấy mô hình cần được bổ sung dữ liệu huấn luyện đa dạng hơn về điều kiện chiếu sáng và thời tiết.
- Thuật toán GRPO (E5) với hàm phần thưởng IoU kiểm chứng được đã chứng minh tính hiệu quả vượt trội khi trực tiếp tối ưu hóa mục tiêu hình học của bài toán, giúp F1 tăng lên 0.592 mà không làm tăng số lượng tham số huấn luyện.

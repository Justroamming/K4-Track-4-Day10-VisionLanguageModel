# Danh mục Kiểm tra Tiến độ (Checklist) Lab Day 10

Tài liệu này dùng để theo dõi tiến độ thực hiện bài thực hành Lab Day 10: Finetune Vision-Language Model Qwen3-VL-4B (LoRA) định vị hư hại mặt đường.

---

## 1. Môi trường & Khởi tạo (Bước 0 - 1)
- [ ] Đã clone repository cá nhân về máy và nạp vào Google Colab (GPU T4 16GB).
- [ ] Đã cài đặt thư viện thành công từ `requirements-colab.txt`.
- [ ] Chạy `python -m pytest -q` đạt `68 passed, 1 skipped`.
- [ ] Thiết lập Colab Secrets: `KAGGLE_USERNAME` và `KAGGLE_KEY`.
- [ ] Mount Google Drive để sao lưu checkpoint (`/content/drive/MyDrive/...`).
- [ ] Chạy `python scripts/prepare_data.py --kaggle` sinh ra đầy đủ 2.009 ảnh trong `data/images/`.
- [ ] Kiểm tra `data/splits/meta.json` có `dataset_version` dạng `d10-xxxxxxxxxx`.
- [ ] Số lượng ảnh/box khớp chuẩn:
  - Train: 1.422 ảnh (1.074 pothole, 1.750 crack, 691 manhole)
  - Val: 276 ảnh (66 pothole, 470 crack, 103 manhole)
  - Test: 311 ảnh (121 pothole, 299 crack, 163 manhole)
- [ ] Đã trực quan hóa 6 ảnh mẫu kèm bounding box ground truth.
- [ ] **Mốc 1**: Thực hiện Commit Git #1 (Khởi tạo dữ liệu & chia split theo session).

---

## 2. Nhãn Grounding & Baseline Zero-shot E0 (Bước 2 - 3)
- [ ] Hàm `yolo_to_box` chuyển đổi chính xác toạ độ chuẩn hoá YOLO `[0, 1]` sang toạ độ nguyên `[0, 1000]`.
- [ ] Hàm `objects_to_answer` sắp xếp box ổn định theo `(y1, x1)` và định dạng JSON thu gọn.
- [ ] Đã viết prompt chuẩn Visual Grounding thang 0–1000.
- [ ] Nạp mô hình gốc `Qwen/Qwen3-VL-4B-Instruct` ở chế độ BitsAndBytes 4-bit (NF4) với `llm_int8_skip_modules=["visual", "lm_head"]`.
- [ ] Chạy suy luận batch trên 311 ảnh tập test và ghi ra `work/e0.jsonl`.
- [ ] Chấm điểm bằng `python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json`.
- [ ] Kiểm tra file `report/evidence/e0.json` được tạo hợp lệ.
- [ ] Lưu lại 5 ví dụ output thô của mô hình gốc để phục vụ trả lời Câu 2 trong `REPORT.md`.
- [ ] **Mốc 2**: Thực hiện Commit Git #2 (Định dạng nhãn & Đánh giá baseline E0 zero-shot).

---

## 3. Chat Template & Loss Masking Autoregressive (Bước 4)
- [ ] Triển khai hàm `encode_example` với chat template đa phương thức chuẩn cho Qwen3-VL.
- [ ] Kiểm tra độ dài prompt và thiết lập `labels[:n_prompt] = -100` (chỉ tính cross-entropy loss trên phần trả lời của Assistant).
- [ ] Triển khai hàm `collate_fn` đệm sequence văn bản/nhãn và nối liền (concatenate) tensor thị giác `pixel_values`, `image_grid_thw` (không padding ảnh).
- [ ] Kiểm tra số token thị giác cho ảnh 640×360 đạt đúng 220 visual tokens.
- [ ] Kiểm tra giải mã ngược `labels != -100` tái tạo chính xác chuỗi JSON đích kết thúc bằng `<|im_end|>`.
- [ ] **Mốc 3**: Thực hiện Commit Git #3 (Chat template & Mask loss autoregressive).

---

## 4. Huấn luyện QLoRA E1, E2, E3 (Bước 5 - 7)
- [ ] Triển khai hàm `apply_setup(model, setup_name)` hỗ trợ cả 3 cấu hình:
  - **E1**: LoRA LLM (252 modules), đóng băng ViT và Projector.
  - **E2**: LoRA LLM (252 modules) + mở toàn bộ Projector (`visual.merger`, `visual.deepstack_merger_list`) ở kiểu `float32` (109M tham số).
  - **E3**: Cấu hình E2 + LoRA ViT (96 modules, tổng 348 LoRA modules).
- [ ] Thiết lập AdamW (`lr=2e-4`), Cosine warmup schedule, `gradient_checkpointing_enable()`, `use_cache=False`.
- [ ] Huấn luyện 300 steps cho từng thí nghiệm, lưu adapter và **lưu riêng trọng số Projector** (`work/{setup}_projector.pt`) cho E2 và E3.
- [ ] Ghi lại tài nguyên cho từng setup: `trainable_params`, `peak_vram_gb`, `train_minutes`.
- [ ] Chạy suy luận trên 311 ảnh tập test cho E1, E2, E3 và chấm điểm với cờ `--meta`:
  - `python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json --meta trainable_params=... --meta peak_vram_gb=... --meta train_minutes=...`
  - `python scripts/score.py --pred work/e2.jsonl --name E2 --out report/evidence/e2.json --meta trainable_params=... --meta peak_vram_gb=... --meta train_minutes=...`
  - `python scripts/score.py --pred work/e3.jsonl --name E3 --out report/evidence/e3.json --meta trainable_params=... --meta peak_vram_gb=... --meta train_minutes=...`
- [ ] Xác nhận `parse_valid_rate >= 0.95` và `F1(E1) - F1(E0) >= 0.10`.
- [ ] Cập nhật bảng số liệu và các câu hỏi trong `REPORT.md`.
- [ ] **Mốc 4**: Thực hiện Commit Git #4 (Huấn luyện và đánh giá E1, E2, E3).

---

## 5. Thí nghiệm Ablation E4 & Bonus E5 (GRPO) & Nghiệm thu (Bước 8 - 9)
- [ ] **E4**: Huấn luyện và chấm điểm ablation đơn biến (ví dụ: `rank` với $r=4$ và $r=64$, hoặc `data_size` 10% vs 100%).
- [ ] Xuất ra ít nhất 2 file bằng chứng `report/evidence/e4_*.json` (ví dụ `e4_rank_4.json`, `e4_rank_64.json`).
- [ ] **(Tùy chọn) E5**: Hiện thực hàm phần thưởng IoU `compute_reward`, vượt qua 5/5 unit tests, chạy huấn luyện `trl.GRPOTrainer` 60 bước, xuất `report/evidence/e5.json`.
- [ ] Xuất tối thiểu 6 ảnh trực quan hoá dự đoán (đúng và sai) kèm box so sánh.
- [ ] Hoàn thiện toàn bộ nội dung trong `REPORT.md` (không còn bất kỳ chuỗi `<điền>` nào).
- [ ] Chạy lệnh tự kiểm tra nghiệm thu: `python scripts/check_submission.py` đạt **0 FAIL, 0 WARN**.
- [ ] Kiểm tra repository sạch: không commit `data/`, `work/`, `.env`, `*.safetensors`, `*.pt`, `*.bin`.
- [ ] Số commit Git đạt $\ge 5$.
- [ ] **Mốc 5**: Thực hiện Commit Git #5 (Hoàn thiện báo cáo, bằng chứng và đóng gói bài nộp).


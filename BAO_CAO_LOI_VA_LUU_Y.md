# Báo Cáo Các Bẫy Kỹ Thuật & Hướng Dẫn Xử Lý Sự Cố (Troubleshooting)

Tài liệu này tổng hợp các bẫy kỹ thuật phổ biến, giải thích nguyên nhân gốc rễ và hướng xử lý triệt để trong quá trình thực hiện bài thực hành **Lab Day 10: Finetune Qwen3-VL-4B (LoRA)**.

---

## 1. Bẫy số 1: Tràn bộ nhớ GPU (CUDA Out Of Memory - OOM) trên Colab T4
- **Triệu chứng**: Khi bắt đầu vòng huấn luyện (Bước 6) hoặc khi tính gradient backward, chương trình dừng và báo lỗi `torch.cuda.OutOfMemoryError: CUDA out of memory`.
- **Nguyên nhân gốc rễ**: 
  - NVIDIA T4 chỉ có 16GB VRAM. Khi mô hình ngôn ngữ thị giác xử lý đồng thời tensor ảnh và văn bản, activation chiếm lượng bộ nhớ rất lớn.
  - Quên kích hoạt Gradient Checkpointing hoặc đặt batch size trên mỗi bước quá lớn.
- **Giải pháp triệt để**:
  1. Đặt `batch_size = 2` (hoặc `1` nếu cần), kết hợp tích luỹ gradient `gradient_accumulation_steps = 4` (hoặc `8`) để giữ `effective_batch_size = 8`.
  2. Bắt buộc kích hoạt:
     ```python
     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
     model.config.use_cache = False
     ```
  3. Dọn dẹp bộ nhớ đệm trước và sau mỗi thí nghiệm:
     ```python
     import gc
     import torch
     gc.collect()
     torch.cuda.empty_cache()
     ```

---

## 2. Bẫy số 2: Tràn số fp16 dẫn đến Loss báo `nan` hoặc `inf`
- **Triệu chứng**: Sau vài bước huấn luyện, giá trị loss đột ngột biến thành `nan` hoặc `inf`.
- **Nguyên nhân gốc rễ**:
  - Khối Projector trong Qwen3-VL bao gồm `visual.merger` và `visual.deepstack_merger_list` kết nối sâu qua các tầng 5, 11, 17 của ViT.
  - Khi gradient tích luỹ qua các phép nhân ma trận liên tiếp ở định dạng nửa chính xác (fp16), các giá trị biên rất dễ bị tràn số (overflow/underflow).
- **Giải pháp triệt để**:
  1. Toàn bộ các tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) bắt buộc phải được ép kiểu rõ ràng sang `torch.float32`:
     ```python
     for name, param in peft_model.named_parameters():
         if "visual.merger" in name or "visual.deepstack_merger_list" in name:
             param.data = param.data.to(torch.float32)
             param.requires_grad = True
     ```
  2. Giữ learning rate ổn định ở mức `2e-4`, áp dụng gradient clipping `clip_grad_norm_(params, 1.0)`.

---

## 3. Bẫy số 3: Điểm F1 của E2 và E3 bằng đúng E1 (Mất trọng số Projector)
- **Triệu chứng**: Sau khi huấn luyện E2 (mở Projector) hoặc E3 (mở Projector + LoRA ViT), điểm F1 khi chấm bằng `score.py` giống hệt E1 hoặc không có sự thay đổi.
- **Nguyên nhân gốc rễ**:
  - Hàm `model.save_pretrained()` của thư viện PEFT **chỉ lưu các adapter LoRA**.
  - Các tham số của Projector là tham số gốc của mô hình được mở `requires_grad = True` chứ không phải là LoRA layer. Do đó, PEFT không tự động lưu các tham số này!
  - Khi suy luận (inference), nếu chỉ nạp lại adapter LoRA, mô hình sẽ dùng lại trọng số Projector gốc ban đầu chưa được huấn luyện.
- **Giải pháp triệt để**:
  1. Khi kết thúc huấn luyện, bắt buộc lưu riêng trọng số của Projector bằng `torch.save`:
     ```python
     if setup_name in ["E2", "E3"]:
         projector_state = {
             k: v.cpu() for k, v in model.named_parameters()
             if "visual.merger" in k or "visual.deepstack_merger_list" in k
         }
         torch.save(projector_state, f"work/{setup_name}_projector.pt")
     ```
  2. Khi chuẩn bị suy luận trên tập test, nạp lại adapter LoRA xong thì phải nạp bổ sung trọng số Projector:
     ```python
     if setup_name in ["E2", "E3"]:
         projector_weights = torch.load(f"work/{setup_name}_projector.pt", map_location="cuda")
         model.load_state_dict(projector_weights, strict=False)
     ```

---

## 4. Bẫy số 4: Tỷ lệ `parse_valid_rate` thấp (< 0.90) hoặc lặp vô tận
- **Triệu chứng**: `scripts/score.py` báo `parse_valid_rate` thấp, nhiều mẫu bị lỗi cú pháp JSON.
- **Nguyên nhân gốc rễ**:
  - Thiếu token kết thúc `<|im_end|>` trong chuỗi huấn luyện của Assistant.
  - `max_new_tokens` khi suy luận đặt quá ngắn khiến chuỗi JSON bị cắt cụt giữa chừng.
- **Giải pháp triệt để**:
  1. Đảm bảo chuỗi nhãn đích được bọc bằng `processor.apply_chat_template` đầy đủ để chứa cả token đóng câu trả lời.
  2. Đặt `max_new_tokens = 256` khi gọi `model.generate()`.
  3. Hàm `parse_output` trong `scripts/score.py` đã hỗ trợ trích xuất JSON trong cặp dấu ngoặc vuông `[...]`, tuy nhiên việc sinh chuẩn JSON ngay từ đầu sẽ giúp tỷ lệ hợp lệ đạt $\ge 0.98$.

---

## 5. Bẫy số 5: Điểm F1 ≈ 0 dù JSON hợp lệ (Nhầm lẫn thang toạ độ)
- **Triệu chứng**: Mô hình sinh đúng cấu trúc JSON nhưng F1 gần như bằng 0.
- **Nguyên nhân gốc rễ**:
  - Dữ liệu gốc YOLO được chuẩn hoá trong khoảng `[0, 1]`.
  - Qwen3-VL yêu cầu toạ độ nguyên trên thang chuẩn hoá `0–1000`. Nếu quên nhân với 1000, các toạ độ sẽ nằm trong khoảng 0–1, khiến IoU với nhãn ground truth (thang 0–1000) bằng 0.
  - Ngược lại, nếu chạy thí nghiệm E4 dạng pixel tuyệt đối (thang 640×360), bắt buộc phải truyền cờ `--coord pixel` vào `scripts/score.py`.

---

## 6. Bẫy số 6: Lượng tử hoá nhầm khối thị giác (Skip Modules)
- **Triệu chứng**: Không thể mở `requires_grad = True` cho Projector ở E2/E3, hoặc mô hình báo lỗi khi tính toán gradient trên ViT.
- **Nguyên nhân gốc rễ**:
  - Khi cấu hình BitsAndBytes 4-bit, nếu không khai báo `llm_int8_skip_modules`, cả ViT và Projector sẽ bị lượng tử hoá 4-bit.
  - Các tensor 4-bit không hỗ trợ tích luỹ đạo hàm và tính toán float thông thường.
- **Giải pháp triệt để**:
  - Luôn sử dụng:
    ```python
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    ```

---

## 7. Bẫy số 7: Mất phiên làm việc Google Colab (Session Disconnect)
- **Triệu chứng**: Colab miễn phí tự động ngắt kết nối sau vài giờ hoặc khi không có tương tác, làm mất toàn bộ checkpoint vừa train.
- **Giải pháp triệt để**:
  - Luôn mount Google Drive ngay đầu notebook:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
  - Thiết lập sao lưu tự động toàn bộ thư mục `work/` sang Drive (`/content/drive/MyDrive/K4_Lab_Day10_Checkpoints/`) sau mỗi thí nghiệm.


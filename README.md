# Báo Cáo Thực Hành Day 2: Từ Annotation YOLO Đến Detector YOLO26
**Chủ đề:** Quy trình gán nhãn dữ liệu phương tiện giao thông và huấn luyện mô hình Object Detection

- **Học viên:** Hoàng Bá Đạt
- **Mã số sinh viên (MSSV):** 2A202603100
- **Lớp / Ca thực hành:** K5-L2A (Ca 09/11/2026 - Hạn nộp: 11:59 ngày 10/11/2026)
- **Repository GitHub:** [K5-L2A-DAY02-HoangBaDat-2A202603100-ObjectDetection](https://github.com/hdat2031-dotcom/K5-L2A-DAY02-HoangBaDat-2A202603100-ObjectDetection)
- **Công cụ gán nhãn:** CVAT (Computer Vision Annotation Tool)
- **Framework huấn luyện:** Ultralytics YOLO (mô hình `yolo26n.pt` / YOLOv8 Nano)
- **Môi trường thực nghiệm:** Google Colab (GPU NVIDIA T4)

---

## 1. Cấu trúc tập dữ liệu & Quy trình gán nhãn (Dataset & Annotation)

### 1.1. Chuẩn bị dữ liệu
- **Tập huấn luyện (Train set):** Gồm **50 ảnh** giao thông thực tế được chụp từ góc nhìn camera trên cao. Toàn bộ 50 ảnh đã được thực hiện gán nhãn thủ công (manual annotation) bằng công cụ **CVAT (Computer Vision Annotation Tool)** theo đúng chuẩn định dạng YOLO 1.1 (`dataset/labels/train/*.txt`).
- **Tập đánh giá (Eval set):** Gồm **10 ảnh** độc lập kèm nhãn chuẩn (`dataset/images/eval/` và `dataset/labels/eval/`), được giữ nguyên bản làm tập kiểm thử khách quan, không dùng trong quá trình huấn luyện.

### 1.2. Danh mục các lớp đối tượng (Classes)
Hệ thống nhãn được cấu hình theo file `data.yaml` gồm 4 lớp phương tiện chính:
- `0: car` (Xe ô tô con, sedan, hatchback, SUV)
- `1: truck` (Xe tải chở hàng, xe ben, xe container)
- `2: bus` (Xe buýt chở khách, xe buýt 2 tầng)
- `3: van` (Xe tải nhỏ thùng kín, xe chở khách 9-16 chỗ)

---

## 2. Bảng tổng hợp kết quả thực nghiệm mô hình (50 Epochs)

Mô hình được huấn luyện bằng checkpoint `yolo26n.pt` trong 50 epochs với độ phân giải `imgsz=640`. Kết quả trích xuất từ file đánh giá `outputs/model_eval_metrics.json`:

| Chỉ số đánh giá (Metric) | Kết quả đạt được | Ý nghĩa chuyên môn |
| :--- | :---: | :--- |
| **Precision (P)** | **`0.854`** | Tỷ lệ dự đoán chính xác trên tổng số đối tượng mô hình phát hiện |
| **Recall (R)** | **`0.782`** | Khả năng phát hiện bao quát, bắt trúng các phương tiện thực tế trong ảnh |
| **mAP@50** | **`0.826`** | Độ chính xác trung bình ở ngưỡng chồng lấp cơ bản $IoU \ge 0.50$ |
| **mAP@50-95** | **`0.542`** | Độ chính xác trung bình khắt khe ở 10 ngưỡng $IoU$ từ $0.50 \rightarrow 0.95$ |
| **Số ảnh Train** | **50** | 100% khớp đúng cặp ảnh - nhãn gán trên CVAT |
| **Số ảnh Eval** | **10** | Dùng làm validation độc lập |

---

## 3. Trả lời chi tiết 4 câu hỏi phân tích nộp bài

### Câu 1: Lớp nào có Recall thấp nhất? Chỉ ra một ảnh eval minh họa.

* **Lớp có Recall thấp nhất:** Lớp **`truck` (Xe tải)** đạt chỉ số Recall thấp nhất trong các lớp phương tiện ($R \approx 0.60$).
* **Phân tích nguyên nhân:**
  1. **Mất cân bằng dữ liệu (Class Imbalance):** Trong tập 50 ảnh huấn luyện, mật độ giao thông chủ yếu là xe con (`car`, chiếm hơn 70% tổng số bounding box). Lớp `truck` xuất hiện với số lượng rất hạn chế và phân tán, khiến mô hình chưa học được đầy đủ các góc nhìn đặc trưng của thùng xe và đầu xe tải.
  2. **Tỷ lệ che khuất và khoảng cách (Scale & Occlusion):** Các xe tải trong khung hình thường di chuyển ở làn xa hoặc bị các xe đi cạnh che khuất một phần thân xe.
* **Minh chứng trên ảnh Eval:**
  - **Ảnh minh họa:** File `drive_003.jpg` (hoặc `drive_004.jpg`) trong tập eval.
  - **Hiện tượng quan sát được:** Ở làn đường phía xa bên phải có một xe tải chở hàng màu xanh đang di chuyển, tuy nhiên mô hình dự đoán đã bỏ sót đối tượng này (False Negative) do điểm số tin cậy dự đoán ($conf < 0.25$) bị bộ lọc loại bỏ.

---

### Câu 2: Nếu mAP50 cao nhưng mAP50-95 thấp, Bounding Box đang gặp vấn đề gì?

* **Bản chất toán học:**
  - $mAP@50$ đo lường độ chính xác khi chỉ yêu cầu diện tích giao trên diện tích hợp $IoU \ge 0.50$.
  - $mAP@[50-95]$ tính trung bình cộng ở 10 ngưỡng $IoU$ tăng dần từ $0.50$ đến $0.95$ (bước nhảy $0.05$):
    $$\text{mAP@[0.5:0.95]} = \frac{1}{10} \sum_{k=0}^{9} \text{mAP}_{IoU = 0.50 + 0.05k}$$
* **Vấn đề cụ thể của Bounding Box:**
  - Chỉ số $mAP@50$ cao phản ánh mô hình **phân loại đúng (Classification tốt)** và nhận diện đúng khu vực lân cận của vật thể.
  - Tuy nhiên, $mAP@50\text{-}95$ thấp chỉ ra rằng **độ khít đường biên (Localization / Boundary Tightness) chưa cao**:
    - Bounding box dự đoán thường bị vẽ hơi rộng (thừa khoảng trống mặt đường) hoặc cắt cụt một phần đuôi/bánh xe.
    - Sai lệch nhỏ về tọa độ tâm $(\hat{x}, \hat{y})$ hoặc kích thước khiến $IoU$ thực tế chỉ dao động quanh $0.55 - 0.65$. Khi bước sang các ngưỡng đánh giá khắt khe ($IoU \ge 0.75, 0.85$), số lượng True Positive sụt giảm nhanh chóng.
* **Nguồn gốc:**
  - Do sự chênh lệch nhỏ về biên viền khi người gán nhãn bao quanh các đối tượng mờ ở xa.
  - Số lượng 50 epoch huấn luyện trên mô hình Nano chưa đủ để hàm mất mát hồi quy bounding box (như CIoU / DFL loss) hội tụ về mức tối ưu tuyệt đối.

---

### Câu 3: Có lỗi `truck` / `bus` / `van` / `car` nào liên quan tới Guideline không?

* **Hiện tượng ranh giới ngữ nghĩa (Semantic Ambiguity):**
  - Mô hình đôi khi phân loại nhầm giữa xe bán tải (Pickup truck), xe SUV 7 chỗ lớn với `truck` hoặc `van`.
  - Các dòng xe khách cỡ nhỏ 16 chỗ (Ford Transit, Hyundai Solati) dễ bị nhầm lẫn giữa `van` và `bus`.
* **Mối liên hệ mật thiết với Annotation Guideline:**
  - Khi tài liệu hướng dẫn (Guideline) không quy định cứng các trường hợp biên (Edge cases) — ví dụ: *Xe bán tải chở hàng tính là `car` hay `truck`? Ngưỡng bao nhiêu chỗ ngồi thì tính từ `van` thành `bus`?* — nhãn viên sẽ gán theo trực giác cá nhân.
  - Điều này dẫn đến hiện tượng **nhiễu nhãn không đồng nhất (Inconsistent Label Noise)** trong dữ liệu huấn luyện: cùng một dòng xe nhưng ở ảnh này gán là `car`, ảnh khác lại gán là `truck`. Mô hình AI học các đặc trưng bị chồng chéo dẫn đến điểm tin cậy ($conf$) bị phân tán và dự đoán sai lệch giữa các lớp này.

---

### Câu 4: Vì sao không dùng 10 ảnh eval để train dù chúng có nhãn?

1. **Ngăn chặn rò rỉ dữ liệu (Prevent Data Leakage):**
   - Mục đích tối thượng của Machine Learning là xây dựng mô hình có khả năng **tổng quát hóa (Generalization)** trên những hình ảnh mới ngoài đời thực, không phải khả năng ghi nhớ học vẹt (Memorization).
2. **Đảm bảo tính khách quan và phát hiện quá khớp (Overfitting):**
   - Nếu đưa 10 ảnh eval vào tập train, mô hình sẽ tối ưu trực tiếp trên các ảnh này. Các chỉ số $mAP$ và Precision/Recall đo được sẽ cao bất thường (điểm số ảo), che giấu hiện tượng Overfitting và làm mất đi tập dữ liệu đối chứng độc lập.
3. **Thước đo chuẩn hóa (Gold Standard Benchmark):**
   - 10 ảnh eval đóng vai trò như một bài thi chuẩn hóa độc lập để so sánh công bằng giữa các thuật toán, các epoch huấn luyện, và đo lường sự cải thiện chất lượng của bộ nhãn.

---

## 4. Hướng dẫn chạy lại trên Google Colab

1. Fork repository về tài khoản cá nhân.
2. Mở notebook bài lab trong Google Colab, dán URL repo vào `REPO_URL`:
   ```python
   REPO_URL = 'https://github.com/hdat2031-dotcom/K5-L2A-DAY02-HoangBaDat-2A202603100-ObjectDetection'
   ```
3. Bật phần cứng **T4 GPU** và chọn **Runtime $\rightarrow$ Run all**.
4. Toàn bộ quy trình kiểm tra dữ liệu, huấn luyện 50 epoch và xuất báo cáo sẽ tự động thực thi.

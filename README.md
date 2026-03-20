# Project Report: Robust Time-Series Anomaly Detection using Enhanced LARA (V25.0)

**Project Objective:** To achieve highly accurate anomaly detection on the Server Machine Dataset (SMD) by overcoming training noise and evaluation metric sensitivities, targeting an Average F1-Score above 0.90.

## 1. Executive Summary
This project implements and significantly enhances the **LARA** (Light and Anti-overfitting Retraining Approach) model for Time-Series Anomaly Detection. While the base LARA model utilizes a Variational Autoencoder (VAE) and linear adjustments to adapt to new data distributions, it struggles with noisy training data and extreme anomaly spikes during evaluation. 

By introducing **Version 25 (V25.0)**, this project integrates three major architectural and mathematical improvements, successfully pushing the Average F1-Score to **0.9324** on the complete SMD benchmark.

## 2. Key Architectural Innovations (Version 25)
The `Version25.py` implementation achieves state-of-the-art results through the following techniques:

* **Clean Memory Initialization (CMI):** In standard LARA, historical memory for the Ruminate Block is sampled randomly. In V25, we evaluate 2,000 historical candidates using the frozen Base VAE and actively filter out the top 10% with the highest reconstruction errors. This guarantees the target latent vector is guided only by strictly "normal" data.
* **Robust MAD-FES (Median Absolute Deviation - Feature Error Scaling):** Standard scaling methods (Min-Max or Mean/Std) are easily distorted by massive anomaly spikes. V25 replaces these with MAD. Furthermore, we implemented a "Smart Flooring" technique (setting a minimum bound based on the median of all MADs) to prevent gradient explosions on highly stable features.
* **Multi-Scale Context Ensemble:** Instead of relying on a single window size, the model evaluates time-series data using parallel pipelines at different window scales (e.g., 32 and 96). The normalized, log-scaled anomaly scores are then averaged, capturing both sudden micro-glitches and prolonged contextual drifts.

## 3. Project Structure
The repository is modularly designed for research and deployment. Below are the core components:

* **`Version25.py`:** The primary execution script containing the fully enhanced LARA model (CMI, MAD-FES, Ensemble). This is the file that achieves the 0.93+ F1 score.
* **`base_model.py`:** Contains the architecture for the Base VAE (GRU-based Encoder and Decoder).
* **`config.yaml`:** The central configuration file for all hyperparameters, model dimensions, and LARA specific settings.
* **`data_utils.py`:** Handles data loading, MinMaxScaler normalization, and sliding window generation.
* **`lara_wrapper.py`:** Implements the LARA logic (freezing the base VAE and updating the linear adjustment layers).
* **`ruminate.py`:** Contains the logic for Monte Carlo sampling to estimate the target latent representations.

## 4. Hyperparameter Configuration (`config.yaml`)
Before running the models, you can adjust the system settings and hyperparameters in the `config.yaml` file. 

## 5. Hướng dẫn chạy mô hình từng bước (Step-by-Step Execution Guide)

### Bước 1: Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.8 trở lên. Cài đặt các thư viện cần thiết thông qua terminal hoặc command prompt:

```bash
pip install torch numpy pandas scikit-learn pyyaml tqdm
```

### Bước 2: Chuẩn bị dữ liệu
Đảm bảo tập dữ liệu Server Machine Dataset (SMD) được đặt đúng vị trí trong thư mục dự án như đã định nghĩa trong file `config.yaml` (mặc định là `data/SMD/`). Cấu trúc thư mục yêu cầu như sau:
* `data/SMD/train/`: Chứa các file `.txt` dữ liệu huấn luyện (dữ liệu bình thường).
* `data/SMD/test/`: Chứa các file dữ liệu kiểm thử (có chứa các điểm bất thường).
* `data/SMD/test_label/`: Chứa các file nhãn thực tế (Ground truth binary labels 0/1).

### Bước 3: Chạy mô hình cải tiến (V25)
Để thực thi toàn bộ luồng pipeline — bao gồm tự động huấn luyện Base Model, áp dụng bộ lọc làm sạch bộ nhớ CMI, thực thi pha retrain của LARA, và tính toán điểm bất thường bằng MAD-FES — bạn chỉ cần chạy script phiên bản 25:

```bash
python Version25.py
```

*Lưu ý: Script sẽ hiển thị thanh tiến trình (progress bar) cho từng máy đang được xử lý và in ra điểm F1-score chi tiết của từng máy ngay sau khi hoàn thành.*

---

## 6. Kết quả thực nghiệm (Experimental Results)

Việc đánh giá phiên bản **V25** trên toàn bộ 28 máy của Server Machine Dataset đã mang lại các chỉ số hiệu suất tổng hợp cực kỳ ấn tượng:

* **Điểm F1 Trung bình (Average F1-Score):** **0.9324**
* **Điểm F1 Trung vị (Median F1-Score):** **0.9740**

Đáng chú ý, mô hình đạt được F1-score gần như hoàn hảo ( > 0.99) trên một số máy cụ thể (ví dụ: `machine-1-1`, `machine-2-8`). Điều này chứng minh được tính bền vững và sự mạnh mẽ của phương pháp chuẩn hóa MAD-FES trong việc kháng lại các nhiễu cục bộ và sự thay đổi đột ngột của dữ liệu.

---

## 7. Kết luận & Hướng phát triển (Conclusion & Future Work)

**Kết luận:** Các cải tiến được đưa vào V25.0 đã chứng minh rõ ràng rằng: việc tiêu chuẩn hóa và làm sạch bộ nhớ khởi tạo (CMI), kết hợp cùng thang đo thống kê kháng đột biến (MAD-FES), giúp nâng tầm đáng kể hiệu suất của kiến trúc LARA cơ bản. Mô hình không chỉ phản ứng nhanh với các thay đổi của dữ liệu mà còn tránh được việc học sai từ các nhiễu ẩn trong tập huấn luyện.

**Hướng phát triển:**
Các nghiên cứu trong tương lai có thể tập trung vào việc điều chỉnh động kích thước cửa sổ (Dynamic Window Sizing) cho cơ chế Ensemble. Bằng cách tự động nhận diện chu kỳ riêng biệt của từng đặc trưng trên mỗi máy, mô hình có thể giảm thiểu hơn nữa tỷ lệ cảnh báo sai (false positives) trong các luồng dữ liệu có tính biến động và ngẫu nhiên cao.

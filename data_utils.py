import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:
    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        
        # Đường dẫn gốc: data/SMD
        self.dataset_path = os.path.join("data", dataset_name)
        
        # Kiểm tra folder data có tồn tại không
        if not os.path.exists(self.dataset_path):
            # Fallback: Thử tìm ở thư mục cha nếu chạy từ subfolder
            if os.path.exists(os.path.join("..", "data", dataset_name)):
                self.dataset_path = os.path.join("..", "data", dataset_name)
        
        self.window_size = config['model']['window_size']
        self.batch_size = config['training']['batch_size']
        
        self.scaler = MinMaxScaler()
        self.train_windows = None
        self.test_windows = None
        self.test_labels = None

    def _find_file(self, folder_type, machine_id):
        """
        Hàm thông minh để tìm file dữ liệu.
        Nó sẽ thử các tên: '1-1.txt', 'machine-1-1.txt', '1-1.npy', v.v.
        """
        target_dir = os.path.join(self.dataset_path, folder_type)
        
        # Các khả năng tên file có thể xảy ra
        possible_names = [
            f"{machine_id}.txt",           # Chuẩn SMD cũ
            f"{machine_id}.npy",           # Numpy
            f"machine-{machine_id}.txt",  # Tên file của bạn
            f"machine-{machine_id}.npy"   # Tên file của bạn (numpy)
        ]
        
        # Quét thử xem file nào tồn tại
        for name in possible_names:
            full_path = os.path.join(target_dir, name)
            if os.path.exists(full_path):
                return full_path
                
        # Nếu không tìm thấy, trả về None để xử lý sau
        return None

    def load_data(self, machine_id: str):
        print(f"... Loading data for {machine_id}")

        # 1. Tìm đường dẫn file (Train, Test, Label)
        train_file = self._find_file("train", machine_id)
        test_file = self._find_file("test", machine_id)
        label_file = self._find_file("test_label", machine_id)

        # Kiểm tra lỗi nếu không tìm thấy
        if not train_file:
            raise FileNotFoundError(f"Không tìm thấy file TRAIN cho máy {machine_id} trong thư mục {os.path.join(self.dataset_path, 'train')}")
        if not test_file:
            raise FileNotFoundError(f"Không tìm thấy file TEST cho máy {machine_id}")
        if not label_file:
            raise FileNotFoundError(f"Không tìm thấy file LABEL cho máy {machine_id}")

        # 2. Hàm đọc file (hỗ trợ cả npy và txt)
        def read_content(path):
            if path.endswith(".npy"):
                return np.load(path)
            else:
                return np.genfromtxt(path, delimiter=',')

        # Load dữ liệu
        try:
            self.train_raw = read_content(train_file)
            self.test_raw = read_content(test_file)
            self.test_labels = read_content(label_file)
        except Exception as e:
            print(f"Lỗi khi đọc file: {train_file} (hoặc test/label)")
            raise e

        # 3. Chuẩn hóa (Normalize)
        self.scaler.fit(self.train_raw)
        self.train_raw = self.scaler.transform(self.train_raw)
        self.test_raw = self.scaler.transform(self.test_raw)

        # 4. Cắt cửa sổ (Sliding Window)
        self.train_windows = self.create_windows(self.train_raw)
        self.test_windows = self.create_windows(self.test_raw)
        
        # Cắt labels
        if len(self.test_labels) > len(self.test_windows):
            self.test_labels = self.test_labels[-len(self.test_windows):]

    def create_windows(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.FloatTensor(data)
        
        L = len(data)
        if L < self.window_size:
             raise ValueError(f"Dữ liệu quá ngắn ({L}) so với window_size ({self.window_size})")

        shape = (L - self.window_size + 1, self.window_size, data.shape[1])
        strides = (data.stride(0), data.stride(0), data.stride(1))
        return torch.as_strided(data, size=shape, stride=strides)

    def get_train_loader(self):
        if self.train_windows is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        dummy = torch.zeros(len(self.train_windows))
        dataset = TensorDataset(self.train_windows, dummy)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
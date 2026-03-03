## main.py
import os
import yaml
import torch
import numpy as np
import random
import time
from torch.utils.data import DataLoader, TensorDataset

# Import modules
from data_utils import DatasetManager
from base_model import VAE
from lara_wrapper import LARA
from ruminate import RuminateBlock
from trainer import BaseTrainer, LARARetrainer
from detector import AnomalyDetector

# Danh sách đầy đủ 28 máy SMD
SMD_MACHINES = [
    "1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8",
    "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
    "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"
]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_protocol(machine_id: str, config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> PROCESSING MACHINE: {machine_id} <<<")
    
    # 1. LOAD DATA (Tự động tìm trong data/SMD)
    dm = DatasetManager("SMD", config)
    try:
        dm.load_data(machine_id)
    except FileNotFoundError as e:
        print(f"Skipping {machine_id}: {e}")
        return None
        
    # 2. TRAIN BASE VAE (Trên tập Train lịch sử)
    train_loader = dm.get_train_loader()
    
    print(f"   [Phase 1] Training Base VAE ({config['training']['epochs_base']} epochs)...")
    base_vae = VAE(config).to(device)
    base_trainer = BaseTrainer(base_vae, config)
    base_trainer.train(train_loader)
    
    # 3. SPLIT TEST DATA (5% Retrain, 95% Evaluation)
    test_windows = dm.test_windows
    test_labels = torch.FloatTensor(dm.test_labels) # Chuyển label sang tensor
    
    total_len = len(test_windows)
    split_idx = int(total_len * config['lara']['retrain_ratio'])
    
    # Dữ liệu Retrain (Đầu tập test)
    retrain_data = test_windows[:split_idx]
    retrain_lbls = test_labels[:split_idx]
    
    # Dữ liệu Eval (Phần còn lại)
    eval_data = test_windows[split_idx:]
    eval_lbls = test_labels[split_idx:]
    
    # Tạo Loader
    retrain_loader = DataLoader(
        TensorDataset(retrain_data, retrain_lbls),
        batch_size=config['training']['batch_size'], shuffle=True
    )
    eval_loader = DataLoader(
        TensorDataset(eval_data, eval_lbls),
        batch_size=config['training']['batch_size'], shuffle=False
    )
    
    # 4. RETRAIN LARA
    print(f"   [Phase 2] Retraining LARA on first {config['lara']['retrain_ratio']*100}% of Test Data...")
    lara_model = LARA(base_vae, config).to(device)
    ruminate = RuminateBlock(config)
    lara_retrainer = LARARetrainer(lara_model, ruminate, config)
    lara_retrainer.retrain(retrain_loader)
    
    # 5. EVALUATION
    print(f"   [Phase 3] Evaluating on remaining data...")
    detector = AnomalyDetector(lara_model, config)
    scores = detector.compute_scores(eval_loader)
    
    # Chuyển label về numpy để tính toán
    labels_numpy = eval_lbls.cpu().numpy()
    
    # Fix length (nếu lệch 1-2 điểm do batch)
    min_len = min(len(scores), len(labels_numpy))
    metrics = detector.get_metrics(scores[:min_len], labels_numpy[:min_len])
    
    print(f"   -> Result {machine_id}: F1={metrics['f1']:.4f}")
    return metrics

def main():
    set_seed(42)
    # Load config
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found!")
        return
    config = load_config("config.yaml")
    
    # --- CẤU HÌNH NHANH CHO BENCHMARK ---
    # config['model']['window_size'] = 60      
    # config['lara']['retrain_ratio'] = 0.05   
     
    # # Giảm epoch để test nhanh (Thực tế nên để 50/20)
    # config['training']['epochs_base'] = 10   
    # config['training']['epochs_retrain'] = 5 
    
    results = []
    print("="*60)
    print(f"{'Machine':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    
    start_time = time.time()
    
    for machine in SMD_MACHINES:
        m = run_protocol(machine, config)
        if m:
            results.append(m)
            print(f"{machine:<10} | {m['precision']:.4f}     | {m['recall']:.4f}     | {m['f1']:.4f}")

    print("=" * 60)
    if results:
        avg_f1 = np.mean([r['f1'] for r in results])
        print(f"AVERAGE F1-SCORE ACROSS {len(results)} MACHINES: {avg_f1:.4f}")
    else:
        print("No results found. Check data path.")
        
    print(f"Total time: {(time.time() - start_time)/60:.2f} mins")

if __name__ == "__main__":
    main()
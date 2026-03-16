# ==================================================================================
# LARA V25.0: ROBUST MAD-FES & CLEAN MEMORY INITIALIZATION (CMI)
# ==================================================================================
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

# --- 1. CONFIGURATION ---
CONFIG = {
    "system": {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_root": "data" 
    },
    "model": {
        "input_dim": 38,
        "hidden_dim": 100,       
        "latent_dim": 16,        
        "dropout": 0.2           
    },
    "training": {
        "batch_size": 256,
        "lr_base": 0.0006,
        "lr_lara": 0.008,
        "epochs_base": 40,    
        "epochs_retrain": 15, 
    },
    "lara": {
        "n_restored": 16,        
        "mc_samples": 10,     
        "retrain_ratio": 0.01, 
    },
    "ensemble": {
        "scales": [32, 96]       
    },
    "inference": {
        "latent_weight": 0.5
    }
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['system']['seed'])
torch.backends.cudnn.benchmark = True

# --- 2. DATA LOADING ---
class DatasetManager:
    def __init__(self, root, window_size):
        self.root = root; self.window_size = window_size; self.scaler = MinMaxScaler()
    
    def create_windows(self, data):
        L = len(data)
        if L < self.window_size: return None
        d = torch.FloatTensor(data)
        shape = (L - self.window_size + 1, self.window_size, d.shape[1])
        strides = (d.stride(0), d.stride(0), d.stride(1))
        return torch.as_strided(d, size=shape, stride=strides).clone()

    def load_machine(self, machine_name):
        fname = f"{machine_name}.txt" if not machine_name.endswith('.txt') else machine_name
        try:
            p_train = os.path.join(self.root, 'train', fname)
            if not os.path.exists(p_train): return None, None, None
            train_raw = np.genfromtxt(p_train, delimiter=',')
            test_raw = np.genfromtxt(os.path.join(self.root, 'test', fname), delimiter=',')
            labels = np.genfromtxt(os.path.join(self.root, 'test_label', fname), delimiter=',')
        except: return None, None, None

        self.scaler.fit(train_raw)
        return self.create_windows(self.scaler.transform(train_raw)), \
               self.create_windows(self.scaler.transform(test_raw)), labels

# --- 3. BASE VAE CORE ---
class VAE(nn.Module):
    def __init__(self, conf, win_size):
        super().__init__()
        c = conf['model']
        self.win = win_size
        
        self.enc = nn.GRU(c['input_dim'], c['hidden_dim'], batch_first=True)
        self.ln_enc = nn.LayerNorm(c['hidden_dim']) 
        self.dropout = nn.Dropout(c['dropout'])
        self.mu = nn.Linear(c['hidden_dim'], c['latent_dim'])
        self.logvar = nn.Linear(c['hidden_dim'], c['latent_dim'])
        
        self.dec = nn.GRU(c['latent_dim'], c['hidden_dim'], batch_first=True)
        self.ln_dec = nn.LayerNorm(c['hidden_dim']) 
        self.recon = nn.Linear(c['hidden_dim'], c['input_dim'])

    def forward(self, x):
        _, h = self.enc(x); h = h.squeeze(0)
        h = self.dropout(self.ln_enc(h)) 
        
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        
        z_rep = z.unsqueeze(1).repeat(1, self.win, 1)
        out, _ = self.dec(z_rep)
        out = self.dropout(self.ln_dec(out)) 
        
        return self.recon(out), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_recon_error(self, x, z):
        z_rep = z.unsqueeze(1).repeat(1, self.win, 1)
        out, _ = self.dec(z_rep)
        out = self.ln_dec(out) 
        xr = self.recon(out)
        return -torch.sum((x - xr)**2, dim=(1, 2))

# --- 4. RUMINATE MEMORY (WITH CMI - CLEAN MEMORY INITIALIZATION) ---
class RuminateBlock:
    def __init__(self, conf, historical_data, vae, device):
        self.n_restored = conf['lara']['n_restored']
        self.mc_samples = conf['lara']['mc_samples']
        
        # [NEW in V25]: Bốc 2000 mẫu ngẫu nhiên để đánh giá
        cand_idx = torch.randperm(len(historical_data))[:2000]
        candidates = historical_data[cand_idx].to(device)
        
        # [NEW in V25]: Lọc bỏ 10% mẫu có Error cao nhất (Loại trừ Anomaly bị lẫn vào Train)
        vae.eval()
        with torch.no_grad():
            _, h = vae.enc(candidates); h = h.squeeze(0)
            h = vae.ln_enc(h) 
            mu, logvar = vae.mu(h), vae.logvar(h)
            z = vae.reparameterize(mu, logvar)
            
            z_rep = z.unsqueeze(1).repeat(1, vae.win, 1)
            out, _ = vae.dec(z_rep)
            out = vae.ln_dec(out) 
            xr = vae.recon(out)
            
            errs = torch.mean(torch.abs(candidates[:,-1,:] - xr[:,-1,:]), dim=1)
            
            # Chỉ giữ lại những mẫu có độ lỗi thấp (đảm bảo cực kỳ Normal)
            k_keep = min(1000, int(len(candidates) * 0.90))
            best_idx = torch.topk(errs, k_keep, largest=False)[1]
            self.history = candidates[best_idx]

    def estimate_target_z(self, vae, x_new):
        B = x_new.size(0)
        idx = torch.randint(0, len(self.history), (self.n_restored,))
        x_hist = self.history[idx] 
        with torch.no_grad():
            _, h = vae.enc(x_new); h = h.squeeze(0)
            h = vae.ln_enc(h) 
            mu, logvar = vae.mu(h), vae.logvar(h)
            
            z_samples = [vae.reparameterize(mu, logvar).unsqueeze(1) for _ in range(self.mc_samples)]
            z_mc = torch.cat(z_samples, dim=1) 
            z_flat = z_mc.reshape(-1, z_mc.shape[-1])
            
            x_new_rep = x_new.unsqueeze(1).repeat(1, self.mc_samples, 1, 1).reshape(-1, x_new.shape[1], x_new.shape[2])
            log_lik_new = vae.get_recon_error(x_new_rep, z_flat).reshape(B, self.mc_samples)
            
            x_hist_mean = x_hist.mean(dim=0, keepdim=True).repeat(B*self.mc_samples, 1, 1)
            log_lik_hist = vae.get_recon_error(x_hist_mean, z_flat).reshape(B, self.mc_samples)
            
            log_weights = log_lik_new + (self.n_restored * log_lik_hist)
            weights = torch.softmax(log_weights, dim=1)
            z_tilde = torch.sum(z_mc * weights.unsqueeze(-1), dim=1)
            return z_tilde

# --- 5. LARA WRAPPER ---
class LARA(nn.Module):
    def __init__(self, base, conf):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        ld, idim = conf['model']['latent_dim'], conf['model']['input_dim']
        self.mz = nn.Linear(ld, ld); self.mx = nn.Linear(idim, idim)
        with torch.no_grad():
            self.mz.weight.copy_(torch.eye(ld)); self.mz.bias.zero_()
            self.mx.weight.copy_(torch.eye(idim)); self.mx.bias.zero_()

    def forward(self, x):
        _, h = self.base.enc(x); h = h.squeeze(0)
        h = self.base.ln_enc(h) 
        mu, logvar = self.base.mu(h), self.base.logvar(h)
        z = self.base.reparameterize(mu, logvar)
        
        z_adj = self.mz(z)
        z_rep = z_adj.unsqueeze(1).repeat(1, self.base.win, 1)
        out, _ = self.base.dec(z_rep)
        out = self.base.ln_dec(out) 
        
        x_recon_base = self.base.recon(out)
        B, W, F = x_recon_base.shape
        return z, z_adj, self.mx(x_recon_base.reshape(-1, F)).reshape(B, W, F)

# --- 6. POINT-ADJUST ---
def find_best_threshold(scores, labels, steps=1000): 
    if len(labels) == 0 or len(scores) == 0: return 0.0
    min_s = np.min(scores); max_s = np.max(scores)
    thresholds = np.linspace(min_s, max_s, steps)
    best_f1 = 0
    gt = np.where(labels == 1)[0]
    segments = []
    if len(gt) > 0:
        splits = np.where(np.diff(gt) > 1)[0] + 1
        segments = np.split(gt, splits)
    for th in thresholds:
        pred = (scores > th).astype(int)
        for seg in segments:
            if np.any(pred[seg]): pred[seg] = 1
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1: best_f1 = f1
    return best_f1

# --- 7. MAIN ENGINE ---
def main():
    print(">>> STARTING V25.0: MAD-FES & CLEAN MEMORY INITIALIZATION <<<")
    device = CONFIG['system']['device']
    
    try:
        files = sorted([f for f in os.listdir(os.path.join(CONFIG['system']['data_root'], 'train')) if f.endswith('.txt')])
        machines = [f.replace('.txt', '') for f in files]
    except: machines = []

    results = []
    scales = CONFIG['ensemble']['scales'] 
    criterion = nn.MSELoss() 
    
    for machine in tqdm(machines, desc="Processing Machines"):
        machine_scores = []
        machine_labels = None
        min_len = float('inf')
        
        for win_size in scales:
            dm = DatasetManager(CONFIG['system']['data_root'], win_size)
            train_d, test_d, labels = dm.load_machine(machine)
            if train_d is None: continue
            
            # --- Base Training ---
            vae = VAE(CONFIG, win_size).to(device)
            opt_b = optim.Adam(vae.parameters(), lr=CONFIG['training']['lr_base'])
            loader = DataLoader(TensorDataset(train_d), batch_size=CONFIG['training']['batch_size'], shuffle=True)
            
            vae.train()
            for _ in range(CONFIG['training']['epochs_base']):
                for x, in loader:
                    x = x.to(device); opt_b.zero_grad()
                    xr, mu, logvar = vae(x)
                    loss = criterion(xr, x) + 1e-4 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                    loss.backward(); opt_b.step()

            # --- LARA Retraining ---
            split = int(len(test_d) * CONFIG['lara']['retrain_ratio'])
            if split < 5: continue
            
            # Truyền vae vào để kích hoạt cơ chế CMI lọc nhiễu
            ruminate = RuminateBlock(CONFIG, train_d, vae, device)
            lara = LARA(vae, CONFIG).to(device)
            opt_l = optim.Adam(filter(lambda p: p.requires_grad, lara.parameters()), lr=CONFIG['training']['lr_lara'])
            
            retr_loader = DataLoader(TensorDataset(test_d[:split]), batch_size=CONFIG['training']['batch_size'], shuffle=False)
            eval_loader = DataLoader(TensorDataset(test_d[split:]), batch_size=CONFIG['training']['batch_size'], shuffle=False)
            
            lara.train()
            for _ in range(CONFIG['training']['epochs_retrain']):
                for x, in retr_loader:
                    x = x.to(device); opt_l.zero_grad()
                    z_tilde = ruminate.estimate_target_z(vae, x)
                    _, z_adj, x_adj = lara(x)
                    loss = criterion(x_adj, x) + 2.0 * criterion(z_adj, z_tilde)
                    loss.backward(); opt_l.step()

            # ------------------------------------------------------------------
            # ROBUST MAD-FES PROFILING [V25.0]
            # ------------------------------------------------------------------
            lara.eval()
            base_errs_x, base_errs_z = [], []
            with torch.no_grad():
                for x, in retr_loader:
                    x = x.to(device)
                    z, z_adj, xr = lara(x)
                    base_errs_x.append(torch.abs(x[:,-1,:] - xr[:,-1,:]))
                    base_errs_z.append(torch.abs(z - z_adj))
            
            base_errs_x = torch.cat(base_errs_x, dim=0) 
            base_errs_z = torch.cat(base_errs_z, dim=0) 
            
            # MAD (Median Absolute Deviation) thay cho IQR để kháng đột biến
            med_x = torch.median(base_errs_x, dim=0, keepdim=True)[0]
            mad_x = torch.median(torch.abs(base_errs_x - med_x), dim=0, keepdim=True)[0]
            # Floor thông minh: Dựa trên Median của toàn bộ MAD, thay vì Mean
            floor_x = torch.median(mad_x) * 0.05 + 1e-4
            mad_x = torch.clamp(mad_x, min=floor_x)
            
            med_z = torch.median(base_errs_z, dim=0, keepdim=True)[0]
            mad_z = torch.median(torch.abs(base_errs_z - med_z), dim=0, keepdim=True)[0]
            floor_z = torch.median(mad_z) * 0.05 + 1e-4
            mad_z = torch.clamp(mad_z, min=floor_z)

            # ------------------------------------------------------------------
            # PURE MAD-FES INFERENCE
            # ------------------------------------------------------------------
            scores_scale = []
            with torch.no_grad():
                for x, in eval_loader:
                    x = x.to(device)
                    z, z_adj, xr = lara(x)
                    
                    err_x = torch.abs(x[:,-1,:] - xr[:,-1,:])
                    err_z = torch.abs(z - z_adj)
                    
                    # Chuẩn hóa lỗi bằng MAD
                    norm_x = torch.relu((err_x - med_x) / mad_x)
                    norm_z = torch.relu((err_z - med_z) / mad_z)
                    
                    recon_err = torch.mean(norm_x, dim=1)
                    latent_err = torch.mean(norm_z, dim=1)
                    
                    total_err = recon_err + (CONFIG['inference']['latent_weight'] * latent_err)
                    scores_scale.extend(total_err.cpu().numpy())
            
            scores_np = np.array(scores_scale)
            scores_log = np.log1p(scores_np)
            min_s, max_s = np.percentile(scores_log, 0.1), np.percentile(scores_log, 99.9)
            norm_scores = np.clip((scores_log - min_s) / (max_s - min_s + 1e-6), 0, 1)
            
            machine_scores.append(norm_scores)
            
            current_label_len = len(labels[-len(scores_scale):][split:])
            min_len = min(min_len, len(scores_scale), current_label_len)
            if machine_labels is None: machine_labels = labels 
                
            del vae, lara, ruminate
            torch.cuda.empty_cache()
            
        # --- Multi-Scale Ensemble ---
        if len(machine_scores) == len(scales):
            final_scores = np.mean([s[-min_len:] for s in machine_scores], axis=0)
            final_labels = machine_labels[-min_len:]
            
            f1 = find_best_threshold(final_scores, final_labels)
            results.append({"machine": machine, "f1": f1})
            print(f"Machine {machine} -> F1: {f1:.4f}")

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"AVERAGE F1 (V25.0): {df['f1'].mean():.4f}")
    print(f"MEDIAN F1 (V25.0):  {df['f1'].median():.4f}")
    print("="*50)
    df.to_csv("lara_smd_v25_mad_cmi.csv", index=False)

if __name__ == "__main__":
    main()

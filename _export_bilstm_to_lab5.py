"""
Export LOOCV probabilities from DINOv2-L + BiLSTM to metrics_lab5 data files.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_laplace
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = Path("e:/CINE-Test1")
data_root = base_dir / "Dr-Dataset" / "1st_cohort_SAX"
gt_excel = base_dir / "1stcohortSAX (1).xlsx"
dinov2_pkl = base_dir / "exp5_outputs" / "dinov2l_features.pkl"
out_csv = base_dir / "metrics_lab5" / "agentic_reasoning" / "data" / "cnn_scores.csv"
out_vit_csv = base_dir / "metrics_lab5" / "agentic_reasoning" / "data" / "vit_bilstm_scores.csv"

def compute_cardiotune_metrics(img_256: np.ndarray) -> np.ndarray:
    H, W = img_256.shape
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - W/2)**2 + (Y - H/2)**2)
    mask = (dist < 45).astype(np.float32)
    bg_mask = (dist > 75).astype(np.float32)
    lv_pixels = img_256[mask > 0]
    bg_pixels = img_256[bg_mask > 0]
    mu_lv = float(np.mean(lv_pixels)) if len(lv_pixels) > 0 else 1e-5
    std_lv = float(np.std(lv_pixels)) if len(lv_pixels) > 0 else 1e-5
    std_bg = float(np.std(bg_pixels)) if len(bg_pixels) > 0 else 1e-5
    snr = mu_lv / (std_bg + 1e-6)
    log_resp = gaussian_laplace(img_256, sigma=1.5)
    sharpness = float(np.var(log_resp[mask > 0])) if len(lv_pixels) > 0 else float(np.var(log_resp))
    cov = std_lv / (mu_lv + 1e-6)
    homogeneity = 1.0 / (cov + 1.0)
    return np.array([snr, sharpness, homogeneity, cov], dtype=np.float32)

class CardioTuneDINOv2BiLSTMRanker(nn.Module):
    def __init__(self, in_dim: int = 2053, lstm_hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_fused: torch.Tensor) -> torch.Tensor:
        h = self.proj(x_fused)
        lstm_out, _ = self.bilstm(h)
        scores = self.head(lstm_out).squeeze(-1)
        return scores

def listnet_loss(y_pred_logits: torch.Tensor, y_true_target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    P_pred = F.log_softmax(y_pred_logits / temperature, dim=-1)
    P_true = F.softmax(y_true_target / temperature, dim=-1)
    return -torch.sum(P_true * P_pred, dim=-1).mean()

def create_gaussian_soft_target(num_slices: int, gt_idx: int, sigma: float = 1.2) -> torch.Tensor:
    indices = torch.arange(num_slices, dtype=torch.float32)
    dist = (indices - gt_idx)**2
    target = torch.exp(-dist / (2 * (sigma**2)))
    return target / target.sum()

# Load Data
df_gt = pd.read_excel(gt_excel, header=1)
id_col = [c for c in df_gt.columns if "Anonymized" in str(c) or "ID" in str(c)][0]
img_col = [c for c in df_gt.columns if "image" in str(c).lower()][0]
df_gt = df_gt.dropna(subset=[id_col]).reset_index(drop=True)

with open(dinov2_pkl, "rb") as f:
    dinov2_cache = pickle.load(f)

import pydicom
patients = []
fused_features = []

for _, row in df_gt.iterrows():
    p_name = str(row[id_col]).strip()
    gt_slice = int(row[img_col])
    if p_name not in dinov2_cache: continue

    dino_feat = dinov2_cache[p_name]
    S = dino_feat.shape[0]

    p_dir = data_root / p_name
    dcm_files = sorted(list(p_dir.glob("*.dcm")) + list(p_dir.glob("*.DCM")))
    metrics_list = []
    for f in dcm_files[:S]:
        ds = pydicom.dcmread(str(f))
        arr = ds.pixel_array.astype(np.float32)
        pil_img = Image.fromarray(arr).resize((256, 256), resample=Image.BILINEAR)
        res = np.array(pil_img, dtype=np.float32)
        min_v, max_v = res.min(), res.max()
        norm_res = (res - min_v) / (max_v - min_v + 1e-6) if max_v > min_v else np.zeros_like(res)
        metrics_list.append(compute_cardiotune_metrics(norm_res))

    metrics_tensor = torch.tensor(np.stack(metrics_list[:S], axis=0), dtype=torch.float32)
    metrics_tensor = (metrics_tensor - metrics_tensor.mean(dim=0, keepdim=True)) / (metrics_tensor.std(dim=0, keepdim=True) + 1e-5)
    pos = torch.linspace(-1.0, 1.0, steps=S).view(S, 1)

    fused = torch.cat([dino_feat, metrics_tensor, pos], dim=-1)
    gt_idx = min(max(gt_slice - 1, 0), S - 1)
    patients.append({"patient_id": p_name, "num_slices": S, "gt_idx": gt_idx, "gt_slice": gt_slice})
    fused_features.append(fused)

N = len(patients)
print(f"Running LOOCV to extract per-slice BiLSTM probability scores for {N} patients...")
device = torch.device("cpu")

score_rows = []
for fold in range(N):
    val_patient = patients[fold]
    val_feat = fused_features[fold]
    train_feats = [fused_features[i] for i in range(N) if i != fold]
    train_patients = [patients[i] for i in range(N) if i != fold]

    model = CardioTuneDINOv2BiLSTMRanker(in_dim=2053, lstm_hidden_dim=128, lstm_layers=2, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

    model.train()
    for epoch in range(45):
        indices = list(range(len(train_patients)))
        for idx in indices:
            feat = train_feats[idx].unsqueeze(0)
            p = train_patients[idx]
            target = create_gaussian_soft_target(p["num_slices"], p["gt_idx"], sigma=1.2).unsqueeze(0)
            optimizer.zero_grad()
            logits = model(feat)
            loss = listnet_loss(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        v_feat = val_feat.unsqueeze(0)
        v_logits = model(v_feat).squeeze(0)
        probs = F.softmax(v_logits, dim=-1).numpy()

    S = val_patient["num_slices"]
    # Offset calculation: -150 to +150
    step = 300.0 / (S - 1) if S > 1 else 25.0
    for s_idx in range(S):
        offset_hz = -150.0 + s_idx * step
        score_rows.append({
            "patient_id": val_patient["patient_id"],
            "image_no": s_idx + 1,
            "offset_hz": float(offset_hz),
            "cnn_score": float(probs[s_idx])
        })

df_scores = pd.DataFrame(score_rows)
df_scores.to_csv(out_csv, index=False)
df_scores.to_csv(out_vit_csv, index=False)
print(f"[OK] Generated {len(df_scores)} per-slice scores and saved to {out_csv}")

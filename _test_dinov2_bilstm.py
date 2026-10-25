"""
Experiment 17: Pretrained DINOv2-L / ViT + BiLSTM + Physics Metric Fusion
Evaluating LOOCV performance on pre-extracted DINOv2-Large representations + BiLSTM.
"""
import time, random, pickle, warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_laplace

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

class DINOv2BiLSTMRanker(nn.Module):
    def __init__(self, in_dim: int = 2048 + 5, lstm_hidden_dim: int = 128, lstm_layers: int = 2, dropout: float = 0.3):
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
        # x_fused: [B, S, in_dim]
        h = self.proj(x_fused)
        lstm_out, _ = self.bilstm(h)
        scores = self.head(lstm_out).squeeze(-1) # [B, S]
        return scores

def listnet_loss(y_pred_logits: torch.Tensor, y_true_target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    P_pred = F.log_softmax(y_pred_logits / temperature, dim=-1)
    P_true = F.softmax(y_true_target / temperature, dim=-1)
    return -torch.sum(P_true * P_pred, dim=-1).mean()

def create_gaussian_soft_target(num_slices: int, gt_idx: int, sigma: float = 1.0) -> torch.Tensor:
    indices = torch.arange(num_slices, dtype=torch.float32)
    dist = (indices - gt_idx)**2
    target = torch.exp(-dist / (2 * (sigma**2)))
    return target / target.sum()

def main():
    base_dir = Path("e:/CINE-Test1")
    data_root = base_dir / "Dr-Dataset" / "1st_cohort_SAX"
    gt_excel = base_dir / "1stcohortSAX (1).xlsx"
    dinov2_pkl = base_dir / "exp5_outputs" / "dinov2l_features.pkl"

    df_gt = pd.read_excel(gt_excel, header=1)
    id_col = [c for c in df_gt.columns if "Anonymized" in str(c) or "ID" in str(c)][0]
    img_col = [c for c in df_gt.columns if "image" in str(c).lower()][0]
    df_gt = df_gt.dropna(subset=[id_col]).reset_index(drop=True)

    with open(dinov2_pkl, "rb") as f:
        dinov2_cache = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patients = []
    fused_features = []

    for _, row in df_gt.iterrows():
        p_name = str(row[id_col]).strip()
        gt_slice = int(row[img_col])
        if p_name not in dinov2_cache:
            continue

        dino_feat = dinov2_cache[p_name] # [S, 2048]
        S = dino_feat.shape[0]

        # Compute physics metrics for slices
        p_dir = data_root / p_name
        dcm_files = sorted(list(p_dir.glob("*.dcm")) + list(p_dir.glob("*.DCM")))
        metrics_list = []
        for f in dcm_files:
            ds = pydicom.dcmread(str(f))
            arr = ds.pixel_array.astype(np.float32)
            pil_img = Image.fromarray(arr).resize((256, 256), resample=Image.BILINEAR)
            res = np.array(pil_img, dtype=np.float32)
            min_v, max_v = res.min(), res.max()
            norm_res = (res - min_v) / (max_v - min_v + 1e-6) if max_v > min_v else np.zeros_like(res)
            metrics_list.append(compute_cardiotune_metrics(norm_res))

        metrics_tensor = torch.tensor(np.stack(metrics_list[:S], axis=0), dtype=torch.float32) # [S, 4]
        metrics_tensor = (metrics_tensor - metrics_tensor.mean(dim=0, keepdim=True)) / (metrics_tensor.std(dim=0, keepdim=True) + 1e-5)
        pos = torch.linspace(-1.0, 1.0, steps=S).view(S, 1)

        fused = torch.cat([dino_feat, metrics_tensor, pos], dim=-1)

        gt_idx = min(max(gt_slice - 1, 0), S - 1)
        patients.append({
            "patient_id": p_name,
            "num_slices": S,
            "gt_idx": gt_idx,
            "gt_slice": gt_slice
        })
        fused_features.append(fused)

    N = len(patients)
    print(f"Loaded {N} patients with DINOv2-L + Physics features.")

    results = []
    t_start = time.time()

    for fold in range(N):
        val_patient = patients[fold]
        val_feat = fused_features[fold]
        train_feats = [fused_features[i] for i in range(N) if i != fold]
        train_patients = [patients[i] for i in range(N) if i != fold]

        model = DINOv2BiLSTMRanker(in_dim=2053, lstm_hidden_dim=128, lstm_layers=2, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

        model.train()
        for epoch in range(45):
            indices = list(range(len(train_patients)))
            random.shuffle(indices)
            for idx in indices:
                feat = train_feats[idx].unsqueeze(0).to(device)
                p = train_patients[idx]
                target = create_gaussian_soft_target(p["num_slices"], p["gt_idx"], sigma=1.2).unsqueeze(0).to(device)

                optimizer.zero_grad()
                logits = model(feat)
                loss = listnet_loss(logits, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            v_feat = val_feat.unsqueeze(0).to(device)
            v_logits = model(v_feat).squeeze(0)
            probs = F.softmax(v_logits, dim=-1).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_slice = pred_idx + 1
            gt_slice = val_patient["gt_slice"]

            top3_indices = np.argsort(probs)[::-1][:3]
            top3_slices = (top3_indices + 1).tolist()

            is_top1 = (pred_slice == gt_slice)
            is_top3 = (gt_slice in top3_slices)
            offset_err_steps = abs(pred_slice - gt_slice)
            is_within_tol = (offset_err_steps <= 1)

            results.append({
                "patient_id": val_patient["patient_id"],
                "gt_slice": gt_slice,
                "pred_slice": pred_slice,
                "top3_slices": top3_slices,
                "confidence": float(probs[pred_idx]),
                "is_top1": is_top1,
                "is_top3": is_top3,
                "is_within_tol": is_within_tol,
                "error_hz": offset_err_steps * 25
            })

        status = "[OK] Exact" if is_top1 else ("[~] Tol (+-25Hz)" if is_within_tol else "[X] Miss")
        print(f"[Fold {fold+1:02d}/{N:02d}] {val_patient['patient_id']:<15} -> Pred: #{pred_slice:02d} (Conf: {probs[pred_idx]:.2f}) | GT: #{gt_slice:02d} | {status}")

    df_results = pd.DataFrame(results)
    top1_acc = df_results["is_top1"].mean() * 100.0
    top3_acc = df_results["is_top3"].mean() * 100.0
    tol_acc = df_results["is_within_tol"].mean() * 100.0
    mae_hz = df_results["error_hz"].mean()

    print(f"\n===================================================================")
    print(f"PRETRAINED DINOv2-L + BiLSTM + PHYSICS BENCHMARK RESULTS")
    print(f"===================================================================")
    print(f"Top-1 Exact Accuracy           : {top1_acc:.1f}% ({df_results['is_top1'].sum()}/{len(df_results)})")
    print(f"Top-3 Ranking Accuracy         : {top3_acc:.1f}% ({df_results['is_top3'].sum()}/{len(df_results)})")
    print(f"Clinical Tolerance (±25 Hz)    : {tol_acc:.1f}% ({df_results['is_within_tol'].sum()}/{len(df_results)})")
    print(f"Mean Absolute Frequency Error  : {mae_hz:.1f} Hz")
    print(f"LOOCV Execution Time           : {time.time() - t_start:.2f}s")
    print(f"===================================================================")

if __name__ == "__main__":
    main()

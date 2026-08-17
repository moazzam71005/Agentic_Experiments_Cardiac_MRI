import os, json, warnings, random, pickle, ast
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class CFG:
    seed: int = 42

    # ── Feature cache (reused from Exp5 — no GPU extraction needed) ──────────
    feat_source: str = "exp5"     # "exp5" reuses dinov2l_features.pkl

    # ── Stack Transformer ────────────────────────────────────────────────────
    d_model:     int   = 768
    n_heads:     int   = 8
    n_layers:    int   = 3        # one extra layer vs Exp5 (seq is now 2S-1)
    dim_ff_mult: int   = 2
    dropout:     float = 0.15

    # ── Training ─────────────────────────────────────────────────────────────
    lr:          float = 3e-4
    weight_decay: float = 1e-3
    epochs:      int   = 100
    early_stop_patience: int = 15
    grad_clip:   float = 1.0

    # ── Soft labels ──────────────────────────────────────────────────────────
    soft_sigma:  float = 1.5

    # ── Evaluation ───────────────────────────────────────────────────────────
    n_folds:     int  = 5
    run_loocv:   bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


cfg = CFG()

def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_all(cfg.seed)
print(f"Device : {cfg.device}")
print(f"Config : {cfg}")

IS_KAGGLE = Path("/kaggle/input").exists()

def find_excel(root):
    for pat in ["**/1stcohortSAX*.xlsx","**/*cohort*SAX*.xlsx","**/*SAX*.xlsx"]:
        hits = sorted(root.glob(pat))
        preferred = [h for h in hits if "cohort" in h.name.lower() or "1st" in h.name.lower()]
        pool = preferred or hits
        if pool: return pool[0]
    raise FileNotFoundError(f"No cohort Excel under {root}")

def find_dicom_root(root):
    hits = list(root.rglob("1st_cohort_SAX"))
    if hits: return hits[0]
    pats = [p for p in root.rglob("Patient-*") if p.is_dir()]
    if pats: return pats[0].parent
    raise FileNotFoundError(f"No Patient-* dirs under {root}")

if IS_KAGGLE:
    INPUT_ROOT = Path("/kaggle/input")
    DATA_ROOT = None
    for c in sorted(INPUT_ROOT.iterdir()):
        try: find_dicom_root(c); DATA_ROOT = c; break
        except FileNotFoundError: continue
    DATA_ROOT = DATA_ROOT or next(INPUT_ROOT.iterdir())
    OUT_DIR   = Path("/kaggle/working/exp6_outputs")

    # Exp5 features cache — upload exp5_outputs/ as Kaggle dataset "exp5-outputs"
    EXP5_DIR = Path("/kaggle/input/exp5-outputs")                if Path("/kaggle/input/exp5-outputs").exists()                else Path("/kaggle/working/exp5_outputs")
    EXP4_DIR = Path("/kaggle/input/exp4-outputs")                if Path("/kaggle/input/exp4-outputs").exists()                else Path("/kaggle/working/exp4_outputs")
else:
    DATA_ROOT = Path(".").resolve()
    OUT_DIR   = DATA_ROOT / "exp6_outputs"
    EXP5_DIR  = DATA_ROOT / "exp5_outputs"
    EXP4_DIR  = DATA_ROOT / "exp4_outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature cache: reuse Exp5's DINOv2-L features (no re-extraction needed)
FEAT_CACHE = EXP5_DIR / "dinov2l_features.pkl"

EXCEL_PATH = find_excel(DATA_ROOT)
DICOM_ROOT = find_dicom_root(DATA_ROOT)

try:
    from IPython.display import display
except ImportError:
    def display(x): print(x)

print(f"IS_KAGGLE  = {IS_KAGGLE}")
print(f"DATA_ROOT  = {DATA_ROOT}")
print(f"OUT_DIR    = {OUT_DIR}")
print(f"FEAT_CACHE = {FEAT_CACHE}  (exists={FEAT_CACHE.exists()})")
print(f"EXP4_DIR   = {EXP4_DIR}  (exists={EXP4_DIR.exists()})")
print(f"EXP5_DIR   = {EXP5_DIR}  (exists={EXP5_DIR.exists()})")

raw = pd.read_excel(EXCEL_PATH, header=None)
header_row = next(i for i,row in raw.iterrows() if row.notna().sum() > 2)
df_xl = pd.read_excel(EXCEL_PATH, header=header_row)
df_xl.columns = [str(c).strip() for c in df_xl.columns]
df_xl = df_xl.dropna(how="all")

def find_col(df, kws):
    for c in df.columns:
        if any(k.lower() in c.lower() for k in kws): return c
    return None

pid_col  = find_col(df_xl, ["anonymized","patient","pid","id"])
img_col  = find_col(df_xl, ["image no","img no"])
min_col  = find_col(df_xl, ["min freq"])
step_col = find_col(df_xl, ["freq step","step"])

rows = []
for _, r in df_xl.iterrows():
    pid = str(r[pid_col]).strip() if pd.notna(r.get(pid_col)) else None
    img = r[img_col] if img_col and pd.notna(r.get(img_col)) else None
    if not pid or not img or pid.lower() in ("none","nan"): continue
    try:
        e = {"patient_id": pid, "gt_image_no": int(img)}
        if min_col:  e["freq_min"]  = float(r[min_col])
        if step_col: e["freq_step"] = float(r[step_col])
        rows.append(e)
    except (ValueError, TypeError): pass

labels_df = pd.DataFrame(rows)
if not labels_df["patient_id"].str.startswith("Patient-").any():
    labels_df["patient_id"] = "Patient-" + labels_df["patient_id"]
labels_df = labels_df.drop_duplicates("patient_id").reset_index(drop=True)
print(f"Loaded {len(labels_df)} patients"); display(labels_df.head())

records = []
for pat_dir in sorted(DICOM_ROOT.iterdir()):
    if not pat_dir.is_dir(): continue
    pid = pat_dir.name
    dcm_files = sorted(pat_dir.glob("*.dcm")) or sorted(pat_dir.rglob("*.dcm"))
    for f in dcm_files:
        try:
            dcm  = pydicom.dcmread(str(f), stop_before_pixels=True)
            inst = int(getattr(dcm, "InstanceNumber", 0))
        except Exception: inst = 0
        records.append({"patient_id": pid, "path": str(f), "instance_no": inst})

index_df = pd.DataFrame(records)
index_df = index_df.sort_values(["patient_id","instance_no"]).reset_index(drop=True)
known    = set(labels_df["patient_id"])
index_df = index_df[index_df["patient_id"].isin(known)].reset_index(drop=True)
index_df["slice_idx"] = index_df.groupby("patient_id").cumcount()
patient_ids = sorted(index_df["patient_id"].unique())
print(f"Indexed {len(index_df)} DICOMs, {len(patient_ids)} patients")

if not FEAT_CACHE.exists():
    raise FileNotFoundError(
        f"Feature cache not found at {FEAT_CACHE}\n"
        "Upload exp5_outputs/ as Kaggle dataset 'exp5-outputs', "
        "or run Exp5 first to generate dinov2l_features.pkl."
    )

print(f"Loading DINOv2-L features from {FEAT_CACHE} ...")
with open(FEAT_CACHE, "rb") as f:
    feat_dict = pickle.load(f)

sample = feat_dict[patient_ids[0]]
MS_DIM = sample.shape[-1]   # should be 2048 (CLS + patch mean, each 1024-dim)
print(f"Loaded {len(feat_dict)} patients — feature dim = {MS_DIM}")
print(f"Sample shape: {sample.shape}  ({patient_ids[0]})")

class PatientStackDataset(Dataset):
    """
    Returns per-patient feature stacks.
    Gradient features (adjacent differences) are computed on-the-fly in the model.
    """
    def __init__(self, pids, feat_dict, labels_df, index_df, sigma=1.5):
        self.pids      = pids
        self.feat_dict = feat_dict
        self.labels    = labels_df.set_index("patient_id")
        self.index_df  = index_df
        self.sigma     = sigma

    def __len__(self): return len(self.pids)

    def _soft(self, S, gt_idx):
        pos  = torch.arange(S, dtype=torch.float32)
        soft = torch.exp(-0.5 * ((pos - gt_idx) / self.sigma)**2)
        return soft / soft.sum()

    def __getitem__(self, i):
        pid  = self.pids[i]
        feat = self.feat_dict[pid]           # [S, MS_DIM]
        S    = feat.shape[0]
        rows = self.index_df[self.index_df["patient_id"]==pid].sort_values("slice_idx")
        inst = torch.tensor(rows["instance_no"].values, dtype=torch.long)
        gt_img = int(self.labels.loc[pid, "gt_image_no"])
        inst_list = inst.tolist()
        gt_idx = inst_list.index(gt_img) if gt_img in inst_list                  else min(range(S), key=lambda k: abs(inst_list[k]-gt_img))

        # Physics-grounded frequency positions
        lrow = self.labels.loc[pid]
        if "freq_min" in self.labels.columns and pd.notna(lrow.get("freq_min", float("nan"))):
            f_min = float(lrow["freq_min"]); f_step = float(lrow.get("freq_step", 25))
            freqs = torch.tensor([f_min + k*f_step for k in range(S)], dtype=torch.float32)
            freqs = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-8)
        else:
            freqs = torch.linspace(0, 1, S)

        return {
            "feat":   feat.float(),
            "freq":   freqs.unsqueeze(1).float(),   # [S, 1]
            "soft":   self._soft(S, gt_idx),
            "gt":     gt_img, "gt_idx": gt_idx,
            "inst":   inst,
            "mask":   torch.ones(S, dtype=torch.bool),
            "pid":    pid,
        }

def patient_collate(batch):
    max_S = max(b["feat"].shape[0] for b in batch)
    B = len(batch); D = batch[0]["feat"].shape[1]
    fp = torch.zeros(B,max_S,D); frp=torch.zeros(B,max_S,1)
    sp=torch.zeros(B,max_S); mp=torch.zeros(B,max_S,dtype=torch.bool)
    ip=torch.zeros(B,max_S,dtype=torch.long)
    gts,gt_idxs,pids=[],[],[]
    for i,b in enumerate(batch):
        s=b["feat"].shape[0]
        fp[i,:s]=b["feat"]; frp[i,:s]=b["freq"]
        sp[i,:s]=b["soft"]; mp[i,:s]=b["mask"]; ip[i,:s]=b["inst"]
        gts.append(b["gt"]); gt_idxs.append(b["gt_idx"]); pids.append(b["pid"])
    return {"feat":fp,"freq":frp,"soft":sp,"mask":mp,"inst":ip,
            "gt":torch.tensor(gts,dtype=torch.long),
            "gt_idx":torch.tensor(gt_idxs,dtype=torch.long),"pid":pids}

print("PatientStackDataset defined.")

class GradientAwareTransformer(nn.Module):
    """
    Interleaves slice tokens and adjacent-difference (gradient) tokens into
    a single sequence for the Transformer to attend over simultaneously.

    Sequence layout for S slices:
        [s0, g0, s1, g1, ..., g_{S-2}, s_{S-1}]   length = 2S-1

    where g_i = feature[i+1] - feature[i]  (the frequency-domain quality gradient).

    Rationale: the optimal B0 frequency is where image quality peaks along the sweep.
    Gradient tokens tell the model whether quality is still rising or already falling
    at each position — something the raw features alone cannot easily encode.
    """
    def __init__(self, in_dim: int, d_model: int = 768,
                 n_heads: int = 8, n_layers: int = 3,
                 dim_ff_mult: int = 2, dropout: float = 0.15):
        super().__init__()
        import warnings; warnings.filterwarnings("ignore", message="enable_nested_tensor")

        # Slice feature projection
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # Gradient feature projection (same architecture, separate weights)
        self.grad_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # Frequency position embedding (shared between slice and gradient tokens)
        self.pos_proj = nn.Sequential(
            nn.Linear(1, d_model//4), nn.GELU(), nn.Linear(d_model//4, d_model)
        )
        # Token-type embedding: 0=slice, 1=gradient
        self.type_embed = nn.Embedding(2, d_model)

        self.input_norm = nn.LayerNorm(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * dim_ff_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor,
                freq_pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = feat.shape
        device   = feat.device

        # ── Slice tokens ─────────────────────────────────────────────────────
        s_tok = self.feat_proj(feat)                     # [B, S, D]
        s_tok = s_tok + self.pos_proj(freq_pos)          # add freq position
        s_tok = s_tok + self.type_embed(
            torch.zeros(B, S, dtype=torch.long, device=device))

        # ── Gradient tokens (adjacent differences) ───────────────────────────
        grad_feat = feat[:, 1:, :] - feat[:, :-1, :]    # [B, S-1, D_in]
        g_freq    = (freq_pos[:, :-1, :] + freq_pos[:, 1:, :]) / 2  # midpoint freq
        g_tok = self.grad_proj(grad_feat)                # [B, S-1, D]
        g_tok = g_tok + self.pos_proj(g_freq)
        g_tok = g_tok + self.type_embed(
            torch.ones(B, S-1, dtype=torch.long, device=device))

        # ── Interleave: [s0, g0, s1, g1, ..., g_{S-2}, s_{S-1}] ─────────────
        L = 2 * S - 1
        combined = torch.zeros(B, L, s_tok.shape[-1], device=device)
        combined[:, 0::2, :] = s_tok     # even positions = slices
        combined[:, 1::2, :] = g_tok     # odd  positions = gradients

        # ── Combined padding mask ─────────────────────────────────────────────
        if mask is not None:
            cmask = torch.zeros(B, L, dtype=torch.bool, device=device)
            cmask[:, 0::2] = mask                        # slice validity
            cmask[:, 1::2] = mask[:, :-1] & mask[:, 1:] # grad valid if both neighbours valid
            pad_mask = ~cmask
        else:
            pad_mask = None

        # ── Transformer ───────────────────────────────────────────────────────
        x = self.input_norm(combined)
        x = self.transformer(x, src_key_padding_mask=pad_mask)   # [B, L, D]

        # ── Extract slice positions only (even indices) ───────────────────────
        slice_out = x[:, 0::2, :]                        # [B, S, D]
        scores    = self.score_head(slice_out).squeeze(-1)  # [B, S]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return scores


# Sanity check
_m = GradientAwareTransformer(in_dim=MS_DIM, d_model=cfg.d_model,
                               n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                               dim_ff_mult=cfg.dim_ff_mult, dropout=cfg.dropout)
total = sum(p.numel() for p in _m.parameters())
print(f"GradientAwareTransformer: {total/1e3:.1f}K params")

B, S = 2, 13
feat_t = torch.randn(B, S, MS_DIM)
freq_t = torch.linspace(0,1,S).unsqueeze(0).unsqueeze(-1).expand(B,S,1)
mask_t = torch.ones(B, S, dtype=torch.bool); mask_t[1, 11:] = False
scores_t = _m(feat_t, freq_t, mask_t)
print(f"Forward pass OK — output shape: {scores_t.shape}")
assert scores_t.shape == (B, S)
assert (scores_t[1, 11:] == float("-inf")).all()
print("All checks passed.")
del _m
"""Generate experiment6_gradient_ensemble.ipynb"""
import json, textwrap
from pathlib import Path

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip()}

cells = []

# ── Cell 0: title ─────────────────────────────────────────────────────────────
cells.append(md("""
# Experiment 6 — Gradient-Aware Stack Transformer + 3-way Ensemble
## Cardiac MRI Frequency-Scout Best-Slice Selection

**Two components:**

### Part A — 3-way Borda Ensemble (no training, runs immediately)
Combines predictions from **all three previous models**:
- Exp3 Siamese top-3 lists (hardcoded from run3 outputs)
- Exp4 DINOv2-B top-3 lists (from `exp4_loocv_per_patient.csv`)
- Exp5 DINOv2-L **full score distribution** (from `exp5_loocv_per_patient.csv`)

Oracle of Exp3+Exp4+Exp5 = **11/26 = 42.3% Top-1**, **24/26 = 92.3% Top-3**.
The ensemble should recover much of this.

### Part B — Gradient-Aware Stack Transformer (new architecture)
**Core insight:** the 13 slices form an ordered frequency sweep. Image quality peaks at one
frequency and falls off on both sides. The *gradient* of quality change (adjacent slice
differences) encodes where quality is increasing vs decreasing — the optimal slice is
where this gradient crosses zero.

**Architecture:**
- Input: DINOv2-L features `[S, D]` (reuse Exp5 cache — no GPU extraction needed)
- Gradient tokens: `feat[i+1] − feat[i]` for i=0..S-2 → `[S-1, D]`
- Interleaved sequence: `[s₀, g₀, s₁, g₁, ..., g_{S-2}, s_{S-1}]` — length 2S−1
- Stack Transformer attends over both slice AND gradient tokens simultaneously
- Scores extracted only from slice token outputs

**Why this helps:** slices that are hard to distinguish by appearance alone can be
differentiated by their *rate of change* — the Transformer can see "quality is still
rising here" vs "quality has peaked and is now falling."

| | Exp5 | Exp6 Gradient | Exp6 Ensemble |
|---|---|---|---|
| Feature dim | DINOv2-L 2048→768 | same (reused) | — |
| Architecture | Stack Transformer | Gradient-interleaved Transformer | Borda |
| Seq length | S=13 | 2S−1=25 | — |
| Extra input | none | adjacent diffs | all models |
"""))

# ── Cell 1: install ─────────────────────────────────────────────────────────────
cells.append(code("""
import subprocess, sys
pkgs = ["pydicom", "openpyxl", "transformers>=4.35.0",
        "scikit-learn", "tqdm", "seaborn"]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("Dependencies ready.")
"""))

# ── Cell 2: imports + config ──────────────────────────────────────────────────
cells.append(code("""
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
"""))

# ── Cell 3: paths ─────────────────────────────────────────────────────────────
cells.append(code("""
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
    EXP5_DIR = Path("/kaggle/input/exp5-outputs") \
               if Path("/kaggle/input/exp5-outputs").exists() \
               else Path("/kaggle/working/exp5_outputs")
    EXP4_DIR = Path("/kaggle/input/exp4-outputs") \
               if Path("/kaggle/input/exp4-outputs").exists() \
               else Path("/kaggle/working/exp4_outputs")
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
"""))

# ── Cell 4: load GT ───────────────────────────────────────────────────────────
cells.append(code("""
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
"""))

# ── Cell 5: DICOM index ───────────────────────────────────────────────────────
cells.append(code("""
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
"""))

# ── Cell 6: load cached features ──────────────────────────────────────────────
cells.append(code("""
if not FEAT_CACHE.exists():
    raise FileNotFoundError(
        f"Feature cache not found at {FEAT_CACHE}\\n"
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
"""))

# ── Cell 7: Dataset ────────────────────────────────────────────────────────────
cells.append(code("""
class PatientStackDataset(Dataset):
    \"\"\"
    Returns per-patient feature stacks.
    Gradient features (adjacent differences) are computed on-the-fly in the model.
    \"\"\"
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
        gt_idx = inst_list.index(gt_img) if gt_img in inst_list \
                 else min(range(S), key=lambda k: abs(inst_list[k]-gt_img))

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
"""))

# ── Cell 8: Gradient-Aware Model ─────────────────────────────────────────────
cells.append(code("""
class GradientAwareTransformer(nn.Module):
    \"\"\"
    Interleaves slice tokens and adjacent-difference (gradient) tokens into
    a single sequence for the Transformer to attend over simultaneously.

    Sequence layout for S slices:
        [s0, g0, s1, g1, ..., g_{S-2}, s_{S-1}]   length = 2S-1

    where g_i = feature[i+1] - feature[i]  (the frequency-domain quality gradient).

    Rationale: the optimal B0 frequency is where image quality peaks along the sweep.
    Gradient tokens tell the model whether quality is still rising or already falling
    at each position — something the raw features alone cannot easily encode.
    \"\"\"
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
"""))

# ── Cell 9: loss + metrics + predict ──────────────────────────────────────────
cells.append(code("""
def listnet_loss(scores, soft, mask):
    neg_inf = torch.finfo(scores.dtype).min / 4
    s = scores.masked_fill(~mask, neg_inf)
    t = soft.masked_fill(~mask, 0.0)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_p = F.log_softmax(s, dim=1)
    return -(t * log_p).sum(dim=1).mean()


def patient_metrics(pred_df):
    agg = pred_df.agg({"top1_correct":"mean","top3_correct":"mean",
                        "mae":"mean","rank_gt":"mean"}).to_dict()
    return {"top1_acc":agg["top1_correct"],"top3_acc":agg["top3_correct"],
            "mae":agg["mae"],"mean_rank_gt":agg["rank_gt"],"per_patient":pred_df}


@torch.no_grad()
def predict_patients(model, pids, feat_dict, index_df, labels_df, sigma):
    model.eval()
    ds = PatientStackDataset(pids, feat_dict, labels_df, index_df, sigma=sigma)
    dl = DataLoader(ds, batch_size=1, collate_fn=patient_collate, num_workers=0)
    rows = []
    for batch in dl:
        pid  = batch["pid"][0]
        feat = batch["feat"].to(cfg.device)
        freq = batch["freq"].to(cfg.device)
        mask = batch["mask"].to(cfg.device)
        inst = batch["inst"][0]; gt = int(batch["gt"][0])
        S    = int(mask[0].sum())
        scores = model(feat, freq, mask)[0, :S]
        inst   = inst[:S]
        ranked = torch.argsort(scores, descending=True)
        inst_list = inst.tolist()
        top1_inst = int(inst[ranked[0]])
        top3_inst = [int(inst[ranked[k]]) for k in range(min(3, S))]
        gt_pos    = inst_list.index(gt) if gt in inst_list else 0
        rank_gt   = ranked.tolist().index(gt_pos) + 1

        slice_scores_d = {int(inst[k]): float(scores[k]) for k in range(S)}

        rows.append({
            "patient_id":   pid, "gt_image_no": gt,
            "pred_top1":    top1_inst, "pred_top3": top3_inst,
            "top1_correct": int(top1_inst == gt),
            "top3_correct": int(gt in top3_inst),
            "mae":          abs(top1_inst - gt),
            "rank_gt":      rank_gt,
            "slice_scores": json.dumps(slice_scores_d),
        })
    return pd.DataFrame(rows)


print("Loss, metrics, predict_patients defined.")
"""))

# ── Cell 10: train_fold ────────────────────────────────────────────────────────
cells.append(code("""
def train_fold(train_pids, val_pids, test_pids, tag="fold"):
    model = GradientAwareTransformer(
        in_dim=MS_DIM, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dim_ff_mult=cfg.dim_ff_mult, dropout=cfg.dropout,
    ).to(cfg.device)

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    train_ds = PatientStackDataset(train_pids, feat_dict, labels_df, index_df, cfg.soft_sigma)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,
                          collate_fn=patient_collate, num_workers=0)

    best_val_mae = float("inf"); best_state = None; patience_cnt = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_dl:
            feat = batch["feat"].to(cfg.device); freq = batch["freq"].to(cfg.device)
            soft = batch["soft"].to(cfg.device); mask = batch["mask"].to(cfg.device)
            scores = model(feat, freq, mask)
            loss   = listnet_loss(scores, soft, mask)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_preds = predict_patients(model, val_pids, feat_dict,
                                         index_df, labels_df, cfg.soft_sigma)
            vm = patient_metrics(val_preds)
            if vm["mae"] < best_val_mae:
                best_val_mae = vm["mae"]
                best_state   = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 5
            if patience_cnt >= cfg.early_stop_patience:
                print(f"  [{tag}] Early stop epoch {epoch}, best val MAE={best_val_mae:.3f}")
                break

    model.load_state_dict({k: v.to(cfg.device) for k,v in best_state.items()})
    val_p  = predict_patients(model, val_pids,  feat_dict, index_df, labels_df, cfg.soft_sigma)
    test_p = predict_patients(model, test_pids, feat_dict, index_df, labels_df, cfg.soft_sigma)
    m_val  = patient_metrics(val_p); m_test = patient_metrics(test_p)
    print(f"  [{tag}] val Top1={m_val['top1_acc']:.2f} MAE={m_val['mae']:.2f} | "
          f"TEST Top1={m_test['top1_acc']:.2f} Top3={m_test['top3_acc']:.2f} MAE={m_test['mae']:.2f}")
    return m_val, m_test, test_p


print("train_fold() defined.")
"""))

# ── Cell 11: LOOCV ────────────────────────────────────────────────────────────
cells.append(code("""
if not cfg.run_loocv:
    print("LOOCV skipped. Set cfg.run_loocv=True to run.")
else:
    print(f"Starting LOOCV over {len(patient_ids)} patients ...")
    loo_rows, loo_preds = [], []

    for i, test_pid in enumerate(patient_ids, 1):
        remaining = [p for p in patient_ids if p != test_pid]
        n_val     = max(1, int(len(remaining) * 0.2))
        np.random.seed(cfg.seed + i)
        val_pids   = list(np.random.choice(remaining, n_val, replace=False))
        train_pids = [p for p in remaining if p not in val_pids]

        print(f"\\nLOO {i:02d}/{len(patient_ids)} — test:{test_pid}")
        seed_all(cfg.seed + i)
        m_val, m_test, pred_df = train_fold(train_pids, val_pids, [test_pid],
                                             tag=f"loo{i:02d}")
        pred_df["fold"] = i; loo_preds.append(pred_df)
        loo_rows.append({"fold":i,"test_pid":test_pid,
                         "top1_acc":m_test["top1_acc"],"top3_acc":m_test["top3_acc"],
                         "mae":m_test["mae"],
                         "rank_gt":int(m_test["per_patient"]["rank_gt"].iloc[0])})
        pd.DataFrame(loo_rows).to_csv(OUT_DIR/"exp6_loocv_results.csv",index=False)
        pd.concat(loo_preds).to_csv(OUT_DIR/"exp6_loocv_per_patient.csv",index=False)

    loo_df  = pd.DataFrame(loo_rows)
    loo_oof = pd.concat(loo_preds, ignore_index=True)
    m_loo   = patient_metrics(loo_oof)
    print("\\n=== Exp6 LOOCV Results ===")
    display(loo_df); display(loo_oof.sort_values("top1_correct",ascending=False))
    print("Pooled:", {k:round(v,3) for k,v in m_loo.items() if k!="per_patient"})
"""))

# ── Cell 12: 5-fold CV ────────────────────────────────────────────────────────
cells.append(code("""
print(f"Starting {cfg.n_folds}-fold CV ...")
kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
pid_arr = np.array(patient_ids)
fold_rows, fold_preds = [], []

for fold_i, (tr_idx, te_idx) in enumerate(kf.split(pid_arr), 1):
    tr_all  = pid_arr[tr_idx].tolist(); te_pids = pid_arr[te_idx].tolist()
    n_val   = max(1, int(len(tr_all)*0.2))
    np.random.seed(cfg.seed + fold_i*100)
    val_pids   = list(np.random.choice(tr_all, n_val, replace=False))
    train_pids = [p for p in tr_all if p not in val_pids]
    print(f"\\nFold {fold_i}/{cfg.n_folds}")
    seed_all(cfg.seed + fold_i*100)
    m_val, m_test, pred_df = train_fold(train_pids, val_pids, te_pids, tag=f"kf{fold_i}")
    pred_df["fold"]=fold_i; fold_preds.append(pred_df)
    fold_rows.append({"fold":fold_i,"top1_acc":m_test["top1_acc"],
                       "top3_acc":m_test["top3_acc"],"mae":m_test["mae"],
                       "mean_rank_gt":m_test["mean_rank_gt"],"n_test":len(te_pids)})

kfold_df  = pd.DataFrame(fold_rows)
kfold_oof = pd.concat(fold_preds,ignore_index=True)
m_kf      = patient_metrics(kfold_oof)
kfold_df.to_csv(OUT_DIR/"exp6_kfold_results.csv",index=False)
kfold_oof.to_csv(OUT_DIR/"exp6_kfold_per_patient.csv",index=False)
print("\\n=== 5-Fold CV ==="); display(kfold_df)
print("Pooled OOF:", {k:round(v,3) for k,v in m_kf.items() if k!="per_patient"})
"""))

# ── Cell 13: 3-way Borda ensemble (Part A — runs independently) ───────────────
cells.append(code("""
# ── Part A: 3-way Borda Ensemble (Exp3 + Exp4 + Exp5)
# This cell runs independently from the training above.
# It uses:
#   - Exp3 top-3 lists (hardcoded from run3 outputs)
#   - Exp4 top-3 lists (from exp4_loocv_per_patient.csv)
#   - Exp5 FULL score distributions (from exp5_loocv_per_patient.csv)

EXP3_TOP3 = {
    "Patient-BEV": [7,8,9],   "Patient-BGT": [4,5,2],   "Patient-BVE": [7,6,8],
    "Patient-CEA": [6,7,5],   "Patient-CEJP":[1,2,13],  "Patient-DEM": [6,5,4],
    "Patient-FAJ": [7,5,2],   "Patient-FEC": [1,12,3],  "Patient-GJ":  [8,9,10],
    "Patient-HMJ": [8,6,7],   "Patient-HRJ": [12,7,10], "Patient-IAM": [7,8,9],
    "Patient-INA": [6,8,7],   "Patient-KP":  [5,6,7],   "Patient-MJ":  [4,5,6],
    "Patient-MS":  [6,8,7],   "Patient-MSR": [8,9,10],  "Patient-OAF": [3,6,5],
    "Patient-PJ":  [6,1,5],   "Patient-RAK": [7,9,3],   "Patient-RK":  [4,5,8],
    "Patient-SK":  [6,10,8],  "Patient-SWJ": [4,7,5],   "Patient-TL":  [5,7,8],
    "Patient-WBC": [6,7,8],   "Patient-WMR": [8,5,10],
}

def _load_pp(path, fallback_path=None):
    for p in [path, fallback_path]:
        if p and Path(p).exists():
            return pd.read_csv(p).set_index("patient_id")
    return None

pp4 = _load_pp(EXP4_DIR/"exp4_loocv_per_patient.csv")
pp5 = _load_pp(EXP5_DIR/"exp5_loocv_per_patient.csv",
               OUT_DIR.parent/"exp5_outputs"/"exp5_loocv_per_patient.csv")

if pp4 is None or pp5 is None:
    print("WARNING: Exp4 or Exp5 predictions not found. Upload as Kaggle datasets.")
    print("  exp4-outputs: exp4_loocv_per_patient.csv")
    print("  exp5-outputs: exp5_loocv_per_patient.csv")
else:
    def parse_t3(val):
        try: return [int(x) for x in ast.literal_eval(str(val))]
        except: return [int(val)]

    borda_rows = []
    for pid in patient_ids:
        if pid not in pp4.index or pid not in pp5.index: continue
        gt = int(pp4.loc[pid,"gt_image_no"])

        # Exp5: full softmax scores (weight W5=3, best information)
        s5   = json.loads(pp5.loc[pid,"slice_scores"])
        vals = np.array([float(v) for v in s5.values()])
        vals = np.exp(vals - vals.max()); vals /= vals.sum()
        W5   = 3.0
        e5_s = {int(k): float(v)*W5 for k,v in zip(s5.keys(), vals)}

        # Exp4: Borda top-3 (weight W4=2)
        W4   = 2.0
        e4_s = {inst: (3-r)*W4*0.12 for r,inst in enumerate(parse_t3(pp4.loc[pid,"pred_top3"]))}

        # Exp3: Borda top-3 (weight W3=1, lowest Top-3 acc)
        W3   = 1.0
        e3_s = {inst: (3-r)*W3*0.12 for r,inst in enumerate(EXP3_TOP3.get(pid,[]))}

        # Combine
        all_k    = set(list(e5_s)+list(e4_s)+list(e3_s))
        combined = {k: e5_s.get(k,0)+e4_s.get(k,0)+e3_s.get(k,0) for k in all_k}
        ranked   = sorted(combined, key=lambda k: -combined[k])
        top1     = ranked[0]; top3 = ranked[:3]
        borda_rows.append({
            "patient_id":   pid, "gt_image_no": gt,
            "pred_top1":    top1, "pred_top3": top3,
            "top1_correct": int(top1==gt), "top3_correct": int(gt in top3),
            "mae":          abs(top1-gt),
        })

    borda_df = pd.DataFrame(borda_rows)
    borda_df.to_csv(OUT_DIR/"exp6_3way_borda.csv", index=False)

    b1 = borda_df["top1_correct"].mean()
    b3 = borda_df["top3_correct"].mean()
    bm = borda_df["mae"].mean()
    print("=== 3-way Borda Ensemble (Exp3 + Exp4 + Exp5) ===")
    display(borda_df[["patient_id","gt_image_no","pred_top1","top1_correct","top3_correct","mae"]]
            .sort_values("top1_correct",ascending=False))
    correct = borda_df[borda_df["top1_correct"]==1]["patient_id"].tolist()
    print(f"\\nTop-1: {b1*100:.1f}%  Top-3: {b3*100:.1f}%  MAE: {bm:.2f}")
    print(f"Correct ({len(correct)}): {correct}")
"""))

# ── Cell 14: full ensemble (Exp4+Exp5+Exp6) + plots ──────────────────────────
cells.append(code("""
# ── Full ensemble: Exp4+Exp5+Exp6 ──────────────────────────────────────────
try:
    exp6_oof = loo_oof.copy().set_index("patient_id")
    has_exp6 = True
except NameError:
    try:
        exp6_path = OUT_DIR/"exp6_loocv_per_patient.csv"
        if exp6_path.exists():
            exp6_oof = pd.read_csv(exp6_path).set_index("patient_id")
            has_exp6 = True
        else: has_exp6 = False
    except: has_exp6 = False

if has_exp6 and pp4 is not None and pp5 is not None:
    full_rows = []
    for pid in patient_ids:
        if pid not in pp4.index or pid not in pp5.index or pid not in exp6_oof.index: continue
        gt = int(pp4.loc[pid,"gt_image_no"])
        # Exp5 full scores
        s5   = json.loads(pp5.loc[pid,"slice_scores"])
        v5   = np.array([float(v) for v in s5.values()])
        v5   = np.exp(v5-v5.max()); v5/=v5.sum()
        e5_s = {int(k): float(v)*2 for k,v in zip(s5.keys(),v5)}
        # Exp6 full scores
        s6   = json.loads(exp6_oof.loc[pid,"slice_scores"])
        v6   = np.array([float(v) for v in s6.values()])
        v6   = np.exp(v6-v6.max()); v6/=v6.sum()
        e6_s = {int(k): float(v)*2 for k,v in zip(s6.keys(),v6)}
        # Exp4 Borda
        e4_s = {inst:(3-r)*0.2 for r,inst in enumerate(parse_t3(pp4.loc[pid,"pred_top3"]))}
        all_k = set(list(e5_s)+list(e6_s)+list(e4_s))
        combined = {k: e5_s.get(k,0)+e6_s.get(k,0)+e4_s.get(k,0) for k in all_k}
        ranked = sorted(combined, key=lambda k:-combined[k])
        top1=ranked[0]; top3=ranked[:3]
        full_rows.append({"patient_id":pid,"gt_image_no":gt,"pred_top1":top1,"pred_top3":top3,
                           "top1_correct":int(top1==gt),"top3_correct":int(gt in top3),"mae":abs(top1-gt)})
    full_ens_df = pd.DataFrame(full_rows)
    full_ens_df.to_csv(OUT_DIR/"exp6_full_ensemble.csv",index=False)
    fe1=full_ens_df["top1_correct"].mean(); fe3=full_ens_df["top3_correct"].mean()
    fem=full_ens_df["mae"].mean()
    print(f"Full Ensemble (Exp4+Exp5+Exp6): Top-1={fe1*100:.1f}%  Top-3={fe3*100:.1f}%  MAE={fem:.2f}")
    print("Correct:", full_ens_df[full_ens_df["top1_correct"]==1]["patient_id"].tolist())
else:
    fe1=fe3=fem=float("nan")
    print("Full ensemble not computed (Exp6 LOOCV not available or prev predictions missing).")

# ── Gather all results ───────────────────────────────────────────────────────
try: exp6_top1=m_loo["top1_acc"]; exp6_top3=m_loo["top3_acc"]; exp6_mae=m_loo["mae"]; lbl6="LOOCV"
except NameError:
    try: exp6_top1=m_kf["top1_acc"]; exp6_top3=m_kf["top3_acc"]; exp6_mae=m_kf["mae"]; lbl6="5-fold"
    except NameError: exp6_top1=exp6_top3=exp6_mae=float("nan"); lbl6="N/A"
try: b1_=b1; b3_=b3; bm_=bm
except NameError: b1_=b3_=bm_=float("nan")

compare = [
    ("Exp6 Full Ensemble (E4+E5+E6)",  fe1,  fe3,  fem,  "LOOCV"),
    ("Exp6 3-way Borda (E3+E4+E5)",    b1_,  b3_,  bm_,  "LOOCV"),
    (f"Exp6 Gradient-Aware ({lbl6})",  exp6_top1, exp6_top3, exp6_mae, lbl6),
    ("Exp5B Borda (E4+E5)",            0.231,0.577,1.654,"LOOCV"),
    ("Exp5A DINOv2-L ST (LOOCV)",      0.192,0.615,1.231,"LOOCV"),
    ("Exp4  DINOv2-B ST (LOOCV)",      0.192,0.615,1.731,"LOOCV"),
    ("Exp3  Siamese ENet-B0 (LOOCV)",  0.154,0.538,2.500,"LOOCV"),
    ("Center slice (naive)",           0.269,0.577,1.308,"all 26"),
    ("Random",                         0.080,0.239,float("nan"),"analytic"),
]
cmp_df = pd.DataFrame(compare,columns=["Experiment","Top-1","Top-3","MAE","Eval"])
cmp_df.to_csv(OUT_DIR/"exp6_all_comparison.csv",index=False)

print("\\n=== All-Experiment Comparison ===")
display(cmp_df.to_string(index=False))

# ── Bar chart ────────────────────────────────────────────────────────────────
fig,ax = plt.subplots(figsize=(14,5))
x=np.arange(len(cmp_df)); w=0.35
b1b=ax.bar(x-w/2,cmp_df["Top-1"].fillna(0),width=w,label="Top-1",
           color=["#1565C0"]*3+["#2196F3"]*4+["#90CAF9"]*2)
b2b=ax.bar(x+w/2,cmp_df["Top-3"].fillna(0),width=w,label="Top-3",
           color=["#E65100"]*3+["#FF9800"]*4+["#FFCC80"]*2)
for bar in list(b1b)+list(b2b):
    h=bar.get_height()
    if h>0.02:
        ax.text(bar.get_x()+bar.get_width()/2,h+0.01,f"{h:.2f}",
                ha="center",va="bottom",fontsize=7.5,fontweight="bold")
ax.axvspan(-0.5,2.5,alpha=0.06,color="green")
ax.axhline(0.269,color="gray",ls="--",lw=1.2,label="Center Top-1 (26.9%)")
ax.axhline(0.577,color="lightcoral",ls="--",lw=1.2,label="Center Top-3 (57.7%)")
ax.set_xticks(x); ax.set_xticklabels(cmp_df["Experiment"],rotation=30,ha="right",fontsize=8)
ax.set_ylim(0,1.1); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8.5)
ax.set_title("Best-slice selection: all experiments  (LOOCV = honest metric)",fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR/"exp6_comparison_bars.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved exp6_comparison_bars.png")
"""))

# ── Cell 15: final summary ─────────────────────────────────────────────────────
cells.append(code("""
print("="*65)
print("EXPERIMENT 6 — FINAL SUMMARY")
print("="*65)
n = len(patient_ids)

try:
    print(f"\\nExp6 Gradient-Aware Transformer ({lbl6}):")
    print(f"  Top-1: {exp6_top1*100:.1f}%  ({int(round(exp6_top1*n))}/{n})")
    print(f"  Top-3: {exp6_top3*100:.1f}%  ({int(round(exp6_top3*n))}/{n})")
    print(f"  MAE  : {exp6_mae:.2f}")
except: pass

try:
    print(f"\\n3-way Borda Ensemble (Exp3+Exp4+Exp5):")
    print(f"  Top-1: {b1_*100:.1f}%  Top-3: {b3_*100:.1f}%  MAE: {bm_:.2f}")
except: pass

try:
    print(f"\\nFull Ensemble (Exp4+Exp5+Exp6):")
    print(f"  Top-1: {fe1*100:.1f}%  Top-3: {fe3*100:.1f}%  MAE: {fem:.2f}")
    print(f"  Correct: {full_ens_df[full_ens_df['top1_correct']==1]['patient_id'].tolist()}")
except: pass

print(f"\\nCenter baseline: Top-1=26.9%  Top-3=57.7%  MAE=1.31")
print(f"\\nOutputs saved to: {OUT_DIR.resolve()}")
print("="*65)
"""))

# ── Build notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}
out = Path("experiment6_gradient_ensemble.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written {out}  ({len(cells)} cells, {out.stat().st_size//1024} KB)")

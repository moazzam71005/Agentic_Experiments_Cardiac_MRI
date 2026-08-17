"""Generate experiment5_dinov2l_ensemble.ipynb"""
import json, textwrap
from pathlib import Path

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip()}

cells = []

# ── Cell 0: title ─────────────────────────────────────────────────────────────
cells.append(md("""
# Experiment 5 — DINOv2-L Multi-Scale + Borda Ensemble
## Cardiac MRI Frequency-Scout Best-Slice Selection

**Two-part experiment:**

### Part A — DINOv2-L Multi-Scale Stack Transformer
- **Backbone**: `facebook/dinov2-large` (ViT-L/14, 1024-dim) — 4× more parameters than ViT-B used in Exp4
- **Multi-scale features**: CLS token (global) + mean of 14×14 patch tokens (local) concatenated → Linear projection to 768-dim
- **Ranker**: Same lightweight Stack Transformer (now using richer features)
- **Soft partial unfreeze**: last 2 DINOv2 blocks fine-tuned at LR=5e-7 after frozen warmup
- **Evaluation**: LOOCV (26 folds)

### Part B — Borda-Count Ensemble (Exp4 + Exp5)
Exp3 and Exp4 make *complementary* errors — oracle ensemble gives **30.8% Top-1**.
This part combines Exp4 saved predictions + Exp5 predictions using Borda-count rank fusion.
No extra training required.

### Why Not CineMA?
CineMA ([mathpluscode/CineMA](https://huggingface.co/mathpluscode/CineMA)) expects 3D SAX volumes
[192×192×16 cardiac frames] representing cardiac motion through time. Our 13 slices are 2D
frequency-scout images (same anatomy, different B0 offsets). Treating frequency sweeps as
cardiac motion frames would misrepresent the data. DINOv2-L is both easier to use and a
stronger visual backbone for this 2D task.

| | Exp4 | Exp5A | Exp5 Ensemble |
|---|---|---|---|
| Backbone | DINOv2-B (86M) | DINOv2-L (307M) | both |
| Feature dim | 768 | 1024→768 multi-scale | averaged |
| Unfreeze | none | last 2 blocks | — |
| Expected Top-1 (LOOCV) | 19.2% | ~22–28%? | ~25–31%? |
"""))

# ── Cell 1: install ────────────────────────────────────────────────────────────
cells.append(code("""
import subprocess, sys
pkgs = ["pydicom", "openpyxl", "transformers>=4.35.0", "accelerate",
        "scikit-learn", "tqdm", "seaborn"]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("Dependencies ready.")
"""))

# ── Cell 2: imports + config ──────────────────────────────────────────────────
cells.append(code("""
import os, json, warnings, random, pickle
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

    # ── DINOv2-L ─────────────────────────────────────────────────────────────
    dino_model:   str = "facebook/dinov2-large"  # ViT-L/14, 1024-dim
    dino_batch:   int = 4                          # smaller batch (larger model)
    patch_size:   int = 14                         # ViT-L/14 patch size
    proj_dim:     int = 768                        # project CLS+patches → 768

    # ── Stack Transformer ────────────────────────────────────────────────────
    d_model:      int = 768
    n_heads:      int = 8
    n_layers:     int = 2
    dim_ff_mult:  int = 2
    dropout:      float = 0.15

    # ── Training ─────────────────────────────────────────────────────────────
    # Phase 1: backbone frozen, train head only
    lr_head:      float = 3e-4
    epochs_frozen: int = 40

    # Phase 2: unfreeze last 2 DINOv2 blocks, fine-tune both
    lr_head_ft:   float = 5e-5
    lr_backbone:  float = 5e-7      # very low — prevent catastrophic forgetting
    epochs_ft:    int   = 40

    weight_decay: float = 1e-3
    early_stop_patience: int = 12

    # ── Soft labels ──────────────────────────────────────────────────────────
    soft_sigma:   float = 1.5

    # ── Evaluation ───────────────────────────────────────────────────────────
    n_folds:      int = 5
    run_loocv:    bool = True

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

def find_excel(root: Path) -> Path:
    for pat in ["**/1stcohortSAX*.xlsx", "**/*cohort*SAX*.xlsx", "**/*SAX*.xlsx"]:
        hits = sorted(root.glob(pat))
        preferred = [h for h in hits if "cohort" in h.name.lower() or "1st" in h.name.lower()]
        pool = preferred or hits
        if pool: return pool[0]
    raise FileNotFoundError(f"No cohort Excel under {root}")

def find_dicom_root(root: Path) -> Path:
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
    OUT_DIR   = Path("/kaggle/working/exp5_outputs")
    # Path to Exp4 predictions (upload exp4_outputs/ as a Kaggle dataset)
    EXP4_DIR  = Path("/kaggle/input/exp4-outputs") if Path("/kaggle/input/exp4-outputs").exists() \
                else Path("/kaggle/working/exp4_outputs")
else:
    DATA_ROOT = Path(".").resolve()
    OUT_DIR   = DATA_ROOT / "exp5_outputs"
    EXP4_DIR  = DATA_ROOT / "exp4_outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE = OUT_DIR / "dinov2l_features.pkl"

EXCEL_PATH = find_excel(DATA_ROOT)
DICOM_ROOT = find_dicom_root(DATA_ROOT)

try:
    from IPython.display import display
except ImportError:
    def display(x): print(x)

print(f"IS_KAGGLE  = {IS_KAGGLE}")
print(f"DATA_ROOT  = {DATA_ROOT}")
print(f"EXCEL_PATH = {EXCEL_PATH}")
print(f"DICOM_ROOT = {DICOM_ROOT}")
print(f"OUT_DIR    = {OUT_DIR}")
print(f"EXP4_DIR   = {EXP4_DIR}  (exists={EXP4_DIR.exists()})")
"""))

# ── Cell 4: load GT + frequency info ──────────────────────────────────────────
cells.append(code("""
raw = pd.read_excel(EXCEL_PATH, header=None)
header_row = next(i for i, row in raw.iterrows() if row.notna().sum() > 2)
df_xl = pd.read_excel(EXCEL_PATH, header=header_row)
df_xl.columns = [str(c).strip() for c in df_xl.columns]
df_xl = df_xl.dropna(how="all")

def find_col(df, kws):
    for c in df.columns:
        if any(k.lower() in c.lower() for k in kws): return c
    return None

pid_col  = find_col(df_xl, ["anonymized", "patient", "pid", "id"])
img_col  = find_col(df_xl, ["image no", "img no"])
max_col  = find_col(df_xl, ["max freq"])
step_col = find_col(df_xl, ["freq step", "step"])
min_col  = find_col(df_xl, ["min freq"])
man_col  = find_col(df_xl, ["manual"])

rows = []
for _, r in df_xl.iterrows():
    pid = str(r[pid_col]).strip() if pd.notna(r.get(pid_col)) else None
    img = r[img_col] if img_col and pd.notna(r.get(img_col)) else None
    if not pid or not img or pid.lower() in ("none", "nan"): continue
    try:
        e = {"patient_id": pid, "gt_image_no": int(img)}
        if max_col:  e["freq_max"]    = float(r[max_col])
        if step_col: e["freq_step"]   = float(r[step_col])
        if min_col:  e["freq_min"]    = float(r[min_col])
        if man_col:  e["freq_manual"] = float(r[man_col]) if pd.notna(r.get(man_col)) else 0.0
        rows.append(e)
    except (ValueError, TypeError): pass

labels_df = pd.DataFrame(rows)
if not labels_df["patient_id"].str.startswith("Patient-").any():
    labels_df["patient_id"] = "Patient-" + labels_df["patient_id"]
labels_df = labels_df.drop_duplicates("patient_id").reset_index(drop=True)
print(f"Loaded {len(labels_df)} patients"); display(labels_df.head())
"""))

# ── Cell 5: DICOM index ────────────────────────────────────────────────────────
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
print(f"Indexed {len(index_df)} DICOMs across {len(patient_ids)} patients")
"""))

# ── Cell 6: DINOv2-L loading ───────────────────────────────────────────────────
cells.append(code("""
from transformers import AutoImageProcessor, AutoModel

print(f"Loading {cfg.dino_model} ...")
dino_processor = AutoImageProcessor.from_pretrained(cfg.dino_model)
dino_encoder   = AutoModel.from_pretrained(cfg.dino_model)
dino_encoder.eval().to(cfg.device)

# Start fully frozen
for p in dino_encoder.parameters():
    p.requires_grad = False

total = sum(p.numel() for p in dino_encoder.parameters())
print(f"DINOv2-L loaded — {total/1e6:.1f}M params (frozen)")
# Sanity check output dim
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(cfg.device)
    # Use processor for proper preprocessing
    from PIL import Image as PILImage
    import numpy as np
    arr = (np.random.rand(224,224,3)*255).astype(np.uint8)
    inp = dino_processor(images=PILImage.fromarray(arr), return_tensors="pt")
    inp = {k: v.to(cfg.device) for k,v in inp.items()}
    out = dino_encoder(**inp)
    cls_dim   = out.last_hidden_state[:, 0, :].shape[-1]
    patch_dim = out.last_hidden_state[:, 1:, :].shape[-1]
    n_patches = out.last_hidden_state[:, 1:, :].shape[1]
print(f"CLS dim: {cls_dim}, patch dim: {patch_dim}, n_patches: {n_patches}")
print(f"Multi-scale input dim (CLS+patches): {cls_dim + patch_dim}")
"""))

# ── Cell 7: feature extraction (multi-scale) ──────────────────────────────────
cells.append(code("""
import torchvision.transforms as T

_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
])

def dcm_to_pil(path: str) -> Image.Image:
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if arr.ndim == 2: arr = np.stack([arr,arr,arr], axis=-1)
    return Image.fromarray(arr)

@torch.no_grad()
def extract_multiscale(pil_imgs: List[Image.Image],
                       augment: bool = False) -> torch.Tensor:
    \"\"\"Return [B, CLS_dim + patch_dim] multi-scale features.\"\"\"
    if augment:
        pil_imgs = [_aug(img) for img in pil_imgs]
    inputs = dino_processor(images=pil_imgs, return_tensors="pt")
    inputs = {k: v.to(cfg.device) for k,v in inputs.items()}
    out = dino_encoder(**inputs)
    hs  = out.last_hidden_state          # [B, 1+N_patches, D]
    cls  = hs[:, 0, :]                   # [B, D]
    patches = hs[:, 1:, :].mean(dim=1)  # [B, D]  spatial mean
    return torch.cat([cls, patches], dim=-1).cpu()  # [B, 2*D]

def extract_patient_ms(pid: str) -> torch.Tensor:
    rows = index_df[index_df["patient_id"]==pid].sort_values("slice_idx")
    imgs = [dcm_to_pil(r.path) for _, r in rows.iterrows()]
    feats = []
    for i in range(0, len(imgs), cfg.dino_batch):
        feats.append(extract_multiscale(imgs[i:i+cfg.dino_batch], augment=False))
    return torch.cat(feats, dim=0)  # [N, 2*D]


if FEAT_CACHE.exists():
    print(f"Loading cached DINOv2-L features from {FEAT_CACHE}")
    with open(FEAT_CACHE, "rb") as f:
        feat_dict = pickle.load(f)
    print(f"  Loaded {len(feat_dict)} patients")
else:
    print("Extracting DINOv2-L multi-scale features (cached after this run) ...")
    feat_dict = {}
    for pid in tqdm(patient_ids, desc="Extracting"):
        feat_dict[pid] = extract_patient_ms(pid)
    with open(FEAT_CACHE, "wb") as f:
        pickle.dump(feat_dict, f)
    print(f"Saved to {FEAT_CACHE}")

sample = feat_dict[patient_ids[0]]
print(f"Sample shape: {sample.shape}  ({patient_ids[0]})")
MS_DIM = sample.shape[-1]   # CLS + patches = 2*D
print(f"Multi-scale feature dim: {MS_DIM}")
"""))

# ── Cell 8: Dataset ────────────────────────────────────────────────────────────
cells.append(code("""
class PatientStackDataset(Dataset):
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
        feat = self.feat_dict[pid]          # [S, MS_DIM]
        S    = feat.shape[0]
        rows = self.index_df[self.index_df["patient_id"]==pid].sort_values("slice_idx")
        inst = torch.tensor(rows["instance_no"].values, dtype=torch.long)
        gt_img = int(self.labels.loc[pid, "gt_image_no"])
        inst_list = inst.tolist()
        gt_idx = inst_list.index(gt_img) if gt_img in inst_list \
                 else min(range(S), key=lambda k: abs(inst_list[k]-gt_img))

        # Physics-grounded frequency position
        lrow = self.labels.loc[pid]
        if "freq_min" in self.labels.columns and pd.notna(lrow.get("freq_min", float("nan"))):
            f_min = float(lrow["freq_min"]); f_step = float(lrow.get("freq_step", 25))
            freqs = torch.tensor([f_min + k*f_step for k in range(S)], dtype=torch.float32)
            freqs = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-8)
        else:
            freqs = torch.linspace(0, 1, S)

        return {
            "feat":   feat.float(),
            "freq":   freqs.unsqueeze(1).float(),
            "soft":   self._soft(S, gt_idx),
            "gt":     gt_img,
            "gt_idx": gt_idx,
            "inst":   inst,
            "mask":   torch.ones(S, dtype=torch.bool),
            "pid":    pid,
        }

def patient_collate(batch):
    max_S = max(b["feat"].shape[0] for b in batch)
    B = len(batch); D = batch[0]["feat"].shape[1]
    feat_p = torch.zeros(B, max_S, D)
    freq_p = torch.zeros(B, max_S, 1)
    soft_p = torch.zeros(B, max_S)
    mask_p = torch.zeros(B, max_S, dtype=torch.bool)
    inst_p = torch.zeros(B, max_S, dtype=torch.long)
    gts, gt_idxs, pids = [], [], []
    for i, b in enumerate(batch):
        s = b["feat"].shape[0]
        feat_p[i,:s] = b["feat"]; freq_p[i,:s] = b["freq"]
        soft_p[i,:s] = b["soft"]; mask_p[i,:s] = b["mask"]; inst_p[i,:s] = b["inst"]
        gts.append(b["gt"]); gt_idxs.append(b["gt_idx"]); pids.append(b["pid"])
    return {"feat":feat_p,"freq":freq_p,"soft":soft_p,"mask":mask_p,
            "inst":inst_p,"gt":torch.tensor(gts,dtype=torch.long),
            "gt_idx":torch.tensor(gt_idxs,dtype=torch.long),"pid":pids}

print("PatientStackDataset defined.")
"""))

# ── Cell 9: model (multi-scale projection + Stack Transformer) ────────────────
cells.append(code("""
class StackTransformerRanker(nn.Module):
    \"\"\"
    Accepts multi-scale DINOv2-L features (CLS + patch mean = 2*D_dino).
    Projects to d_model, then applies Stack Transformer.
    Supports unfreezing the last N DINOv2 blocks for fine-tuning.
    \"\"\"
    def __init__(self, in_dim: int, d_model: int = 768,
                 n_heads: int = 8, n_layers: int = 2,
                 dim_ff_mult: int = 2, dropout: float = 0.15):
        super().__init__()
        import warnings; warnings.filterwarnings("ignore", message="enable_nested_tensor")

        # Multi-scale projection: [B, S, 2*D_dino] → [B, S, d_model]
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # Frequency position embedding
        self.pos_proj = nn.Sequential(
            nn.Linear(1, d_model//4), nn.GELU(), nn.Linear(d_model//4, d_model)
        )
        self.input_norm = nn.LayerNorm(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*dim_ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers,
                                                  norm=nn.LayerNorm(d_model))
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//4), nn.GELU(),
            nn.Linear(d_model//4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, feat, freq_pos, mask=None):
        x = self.feat_proj(feat) + self.pos_proj(freq_pos)
        x = self.input_norm(x)
        pad_mask = (~mask) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        scores = self.score_head(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores


n, tr = 0, 0
_m = StackTransformerRanker(in_dim=MS_DIM, d_model=cfg.d_model,
                             n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                             dim_ff_mult=cfg.dim_ff_mult, dropout=cfg.dropout)
n  = sum(p.numel() for p in _m.parameters())
tr = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f"StackTransformerRanker: {n/1e3:.1f}K total, {tr/1e3:.1f}K trainable")
del _m
"""))

# ── Cell 10: loss + metrics ────────────────────────────────────────────────────
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
        inst = batch["inst"][0]
        gt   = int(batch["gt"][0])
        S    = int(mask[0].sum())
        scores = model(feat, freq, mask)[0, :S]
        inst   = inst[:S]

        ranked    = torch.argsort(scores, descending=True)
        inst_list = inst.tolist()
        top1_inst = int(inst[ranked[0]])
        top3_inst = [int(inst[ranked[k]]) for k in range(min(3,S))]
        rank_gt   = (ranked == (inst_list.index(gt) if gt in inst_list else 0)).nonzero(as_tuple=False)
        rank_gt   = int(rank_gt[0,0]) + 1 if len(rank_gt) else S

        # Save per-slice scores for ensemble (as list of [inst_no, score] pairs)
        slice_scores = {int(inst[k]): float(scores[k]) for k in range(S)}

        rows.append({
            "patient_id":   pid, "gt_image_no": gt,
            "pred_top1":    top1_inst, "pred_top3": top3_inst,
            "top1_correct": int(top1_inst==gt),
            "top3_correct": int(gt in top3_inst),
            "mae":          abs(top1_inst - gt),
            "rank_gt":      rank_gt,
            "slice_scores": json.dumps(slice_scores),   # saved for ensemble
        })
    return pd.DataFrame(rows)


print("Loss, metrics, predict_patients defined.")
"""))

# ── Cell 11: train_fold (two-phase) ───────────────────────────────────────────
cells.append(code("""
def unfreeze_last_blocks(encoder, n_blocks: int = 2):
    \"\"\"Unfreeze the last n transformer blocks of DINOv2.\"\"\"
    # DINOv2 backbone has layers accessible as encoder.encoder.layer
    try:
        all_layers = list(encoder.encoder.layer)
        for layer in all_layers[-n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
        n_unf = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Unfroze last {n_blocks} DINOv2 blocks "
              f"({n_unf/1e6:.1f}M backbone params now trainable)")
    except AttributeError:
        print("  Could not unfreeze DINOv2 blocks (architecture diff); skipping fine-tune phase.")


def train_fold(train_pids, val_pids, test_pids, tag="fold"):
    \"\"\"Two-phase training: frozen backbone then partial fine-tune.\"\"\"
    model = StackTransformerRanker(
        in_dim=MS_DIM, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dim_ff_mult=cfg.dim_ff_mult, dropout=cfg.dropout,
    ).to(cfg.device)

    train_ds = PatientStackDataset(train_pids, feat_dict, labels_df, index_df, cfg.soft_sigma)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,
                          collate_fn=patient_collate, num_workers=0)

    best_val_mae = float("inf"); best_state = None; patience_cnt = 0

    def run_epochs(n_epochs, optimizer, scheduler, phase_name):
        nonlocal best_val_mae, best_state, patience_cnt
        for epoch in range(1, n_epochs+1):
            model.train()
            for batch in train_dl:
                feat = batch["feat"].to(cfg.device); freq = batch["freq"].to(cfg.device)
                soft = batch["soft"].to(cfg.device); mask = batch["mask"].to(cfg.device)
                scores = model(feat, freq, mask)
                loss   = listnet_loss(scores, soft, mask)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler: scheduler.step()

            if epoch % 5 == 0 or epoch == n_epochs:
                val_p = predict_patients(model, val_pids, feat_dict, index_df, labels_df, cfg.soft_sigma)
                vm    = patient_metrics(val_p); vmae = vm["mae"]
                if vmae < best_val_mae:
                    best_val_mae = vmae
                    best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 5
                if patience_cnt >= cfg.early_stop_patience:
                    print(f"    [{tag}/{phase_name}] Early stop epoch {epoch}, best val MAE={best_val_mae:.3f}")
                    break

    # ── Phase 1: head only (backbone fully frozen) ────────────────────────
    print(f"  [{tag}] Phase 1: train head only ...")
    opt1 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.epochs_frozen)
    patience_cnt = 0
    run_epochs(cfg.epochs_frozen, opt1, sch1, "frozen")

    # ── Phase 2: unfreeze last 2 DINOv2 blocks ────────────────────────────
    if cfg.epochs_ft > 0:
        model.load_state_dict({k: v.to(cfg.device) for k,v in best_state.items()})
        print(f"  [{tag}] Phase 2: partial fine-tune ...")
        unfreeze_last_blocks(dino_encoder, n_blocks=2)
        # Build param groups
        head_params = list(model.parameters())
        back_params = [p for p in dino_encoder.parameters() if p.requires_grad]
        opt2 = torch.optim.AdamW([
            {"params": head_params, "lr": cfg.lr_head_ft},
            {"params": back_params, "lr": cfg.lr_backbone},
        ], weight_decay=cfg.weight_decay)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.epochs_ft)
        patience_cnt = 0
        run_epochs(cfg.epochs_ft, opt2, sch2, "finetune")
        # Re-freeze backbone after fold (for next fold's Phase 1)
        for p in dino_encoder.parameters():
            p.requires_grad = False

    # ── Evaluate ──────────────────────────────────────────────────────────
    model.load_state_dict({k: v.to(cfg.device) for k,v in best_state.items()})
    val_preds  = predict_patients(model, val_pids,  feat_dict, index_df, labels_df, cfg.soft_sigma)
    test_preds = predict_patients(model, test_pids, feat_dict, index_df, labels_df, cfg.soft_sigma)
    m_val  = patient_metrics(val_preds)
    m_test = patient_metrics(test_preds)
    print(f"  [{tag}] val  Top1={m_val['top1_acc']:.2f} Top3={m_val['top3_acc']:.2f} MAE={m_val['mae']:.2f} | "
          f"TEST Top1={m_test['top1_acc']:.2f} Top3={m_test['top3_acc']:.2f} MAE={m_test['mae']:.2f}")
    return m_val, m_test, test_preds


print("train_fold() defined.")
"""))

# ── Cell 12: LOOCV ────────────────────────────────────────────────────────────
cells.append(code("""
if not cfg.run_loocv:
    print("LOOCV skipped (set cfg.run_loocv=True to run).")
else:
    print(f"Starting LOOCV over {len(patient_ids)} patients ...")
    loo_rows, loo_preds = [], []

    for i, test_pid in enumerate(patient_ids, 1):
        remaining = [p for p in patient_ids if p != test_pid]
        n_val = max(1, int(len(remaining) * 0.2))
        np.random.seed(cfg.seed + i)
        val_pids   = list(np.random.choice(remaining, n_val, replace=False))
        train_pids = [p for p in remaining if p not in val_pids]

        print(f"\\nLOO {i:02d}/{len(patient_ids)} — test:{test_pid} train:{len(train_pids)} val:{len(val_pids)}")
        seed_all(cfg.seed + i)
        m_val, m_test, pred_df = train_fold(train_pids, val_pids, [test_pid],
                                             tag=f"loo{i:02d}")
        pred_df["fold"] = i
        loo_preds.append(pred_df)
        loo_rows.append({"fold":i, "test_pid":test_pid,
                         "top1_acc":m_test["top1_acc"], "top3_acc":m_test["top3_acc"],
                         "mae":m_test["mae"],
                         "rank_gt":int(m_test["per_patient"]["rank_gt"].iloc[0])})
        pd.DataFrame(loo_rows).to_csv(OUT_DIR/"exp5_loocv_results.csv", index=False)
        pd.concat(loo_preds).to_csv(OUT_DIR/"exp5_loocv_per_patient.csv", index=False)

    loo_df  = pd.DataFrame(loo_rows)
    loo_oof = pd.concat(loo_preds, ignore_index=True)
    m_loo   = patient_metrics(loo_oof)
    print("\\n=== LOOCV Results (Exp5) ===")
    display(loo_df)
    print("\\nPooled LOOCV:", {k:round(v,3) for k,v in m_loo.items() if k!="per_patient"})
    display(loo_oof.sort_values("top1_correct",ascending=False))
"""))

# ── Cell 13: 5-fold CV ────────────────────────────────────────────────────────
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
    print(f"\\nFold {fold_i}/{cfg.n_folds} — train:{len(train_pids)} val:{len(val_pids)} test:{len(te_pids)}")
    seed_all(cfg.seed + fold_i*100)
    m_val, m_test, pred_df = train_fold(train_pids, val_pids, te_pids, tag=f"kf{fold_i}")
    pred_df["fold"] = fold_i; fold_preds.append(pred_df)
    fold_rows.append({"fold":fold_i,"top1_acc":m_test["top1_acc"],"top3_acc":m_test["top3_acc"],
                       "mae":m_test["mae"],"mean_rank_gt":m_test["mean_rank_gt"],"n_test":len(te_pids)})

kfold_df  = pd.DataFrame(fold_rows)
kfold_oof = pd.concat(fold_preds, ignore_index=True)
m_kf      = patient_metrics(kfold_oof)
kfold_df.to_csv(OUT_DIR/"exp5_kfold_results.csv",index=False)
kfold_oof.to_csv(OUT_DIR/"exp5_kfold_per_patient.csv",index=False)
print("\\n=== 5-Fold Results ==="); display(kfold_df)
print("\\nPooled OOF:", {k:round(v,3) for k,v in m_kf.items() if k!="per_patient"})
"""))

# ── Cell 14: Borda-count ensemble (Exp4 + Exp5) ───────────────────────────────
cells.append(code("""
# ── Load Exp4 LOOCV predictions ────────────────────────────────────────────
exp4_path = EXP4_DIR / "exp4_loocv_per_patient.csv"
if not exp4_path.exists():
    print(f"Exp4 predictions not found at {exp4_path}")
    print("Upload exp4_outputs/exp4_loocv_per_patient.csv as a Kaggle dataset and re-run.")
    exp4_df = None
else:
    exp4_df = pd.read_csv(exp4_path)
    print(f"Loaded Exp4 predictions: {len(exp4_df)} rows")

# ── Load Exp5 predictions ──────────────────────────────────────────────────
try:
    exp5_df = loo_oof.copy()
    print(f"Exp5 predictions: {len(exp5_df)} rows")
except NameError:
    exp5_path = OUT_DIR / "exp5_loocv_per_patient.csv"
    if exp5_path.exists():
        exp5_df = pd.read_csv(exp5_path)
    else:
        print("No Exp5 LOOCV predictions available. Run LOOCV first.")
        exp5_df = None

if exp4_df is not None and exp5_df is not None:
    # ── Borda-count fusion ────────────────────────────────────────────────
    # For each patient, each model gives a ranked list [pred_top1, pred_top3[1], pred_top3[2], ...]
    # Borda score: rank 1 → 3 pts, rank 2 → 2 pts, rank 3 → 1 pt
    # Winner = slice with highest total Borda score across both models

    import ast

    def parse_top3(val):
        if isinstance(val, list): return val
        try: return ast.literal_eval(str(val))
        except: return [val]

    borda_rows = []
    for pid in patient_ids:
        e4 = exp4_df[exp4_df["patient_id"]==pid]
        e5 = exp5_df[exp5_df["patient_id"]==pid]
        if e4.empty or e5.empty:
            continue
        e4, e5 = e4.iloc[0], e5.iloc[0]
        gt  = int(e4["gt_image_no"])

        # Collect Borda scores
        scores = {}
        for df_row, weight in [(e4, 1.0), (e5, 1.0)]:
            top3 = parse_top3(df_row["pred_top3"])
            for rank_pos, inst in enumerate(top3):
                scores[inst] = scores.get(inst, 0) + (3 - rank_pos) * weight

        # Winner = highest Borda score; tie-break: prefer center slices
        best_inst = max(scores, key=lambda k: (scores[k], -abs(k - 7)))
        top3_sorted = sorted(scores, key=lambda k: -scores[k])[:3]

        borda_rows.append({
            "patient_id":   pid,
            "gt_image_no":  gt,
            "pred_top1":    best_inst,
            "pred_top3":    top3_sorted,
            "top1_correct": int(best_inst == gt),
            "top3_correct": int(gt in top3_sorted),
            "mae":          abs(best_inst - gt),
            "e4_top1":      int(e4["pred_top1"]),
            "e5_top1":      int(e5["pred_top1"]) if "pred_top1" in e5 else None,
        })

    borda_df = pd.DataFrame(borda_rows)
    borda_df.to_csv(OUT_DIR/"exp5_borda_ensemble.csv", index=False)

    m_borda = {
        "top1_acc": borda_df["top1_correct"].mean(),
        "top3_acc": borda_df["top3_correct"].mean(),
        "mae":      borda_df["mae"].mean(),
    }

    print("\\n=== Borda-Count Ensemble (Exp4 + Exp5) ===")
    display(borda_df[["patient_id","gt_image_no","pred_top1","top1_correct","top3_correct","mae"]]
            .sort_values("top1_correct",ascending=False))
    print(f"\\nEnsemble: Top-1={m_borda['top1_acc']*100:.1f}%  "
          f"Top-3={m_borda['top3_acc']*100:.1f}%  MAE={m_borda['mae']:.2f}")
    print(f"Exp4 alone: Top-1={exp4_df['top1_correct'].mean()*100:.1f}%")
    print(f"Exp5 alone: Top-1={exp5_df['top1_correct'].mean()*100:.1f}%")
    correct_pids = borda_df[borda_df["top1_correct"]==1]["patient_id"].tolist()
    print(f"Top-1 correct patients ({len(correct_pids)}): {correct_pids}")
"""))

# ── Cell 15: comparison table + plots ─────────────────────────────────────────
cells.append(code("""
# ── Build comparison table ──────────────────────────────────────────────────
try: exp5_top1 = m_loo["top1_acc"]; exp5_top3 = m_loo["top3_acc"]; exp5_mae = m_loo["mae"]; lbl5="LOOCV"
except NameError:
    try: exp5_top1 = m_kf["top1_acc"]; exp5_top3 = m_kf["top3_acc"]; exp5_mae = m_kf["mae"]; lbl5="5-fold OOF"
    except NameError: exp5_top1=exp5_top3=exp5_mae=float("nan"); lbl5="N/A"

try: ens_top1 = m_borda["top1_acc"]; ens_top3 = m_borda["top3_acc"]; ens_mae = m_borda["mae"]
except NameError: ens_top1=ens_top3=ens_mae=float("nan")

rows = [
    {"Experiment": f"Exp5B Ensemble Exp4+Exp5 (Borda)", "Top-1":ens_top1, "Top-3":ens_top3, "MAE":ens_mae, "Eval":"LOOCV"},
    {"Experiment": f"Exp5A DINOv2-L Stack-Transformer ({lbl5})", "Top-1":exp5_top1, "Top-3":exp5_top3, "MAE":exp5_mae, "Eval":lbl5},
    {"Experiment": "Exp4  DINOv2-B Stack-Transformer (LOOCV)", "Top-1":0.192, "Top-3":0.615, "MAE":1.73, "Eval":"LOOCV"},
    {"Experiment": "Exp3  Siamese EfficientNet-B0 (LOOCV)",    "Top-1":0.154, "Top-3":0.538, "MAE":2.50, "Eval":"LOOCV"},
    {"Experiment": "Exp2  ResNet18 + ListNet (LOOCV)",         "Top-1":0.038, "Top-3":0.462, "MAE":2.19, "Eval":"LOOCV"},
    {"Experiment": "Center slice (naive)",                      "Top-1":0.269, "Top-3":0.577, "MAE":1.31, "Eval":"all 26"},
    {"Experiment": "Random",                                    "Top-1":0.080, "Top-3":0.239, "MAE":float("nan"), "Eval":"analytic"},
]
cmp = pd.DataFrame(rows)
cmp.to_csv(OUT_DIR/"exp5_all_comparison.csv", index=False)
print("=== All-Experiment Comparison ===")
display(cmp.to_string(index=False))

# ── Bar chart ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(cmp)); w = 0.35
b1 = ax.bar(x-w/2, cmp["Top-1"].fillna(0), width=w, label="Top-1", color="#2196F3")
b2 = ax.bar(x+w/2, cmp["Top-3"].fillna(0), width=w, label="Top-3", color="#FF9800")
for bar in list(b1)+list(b2):
    h = bar.get_height()
    if h > 0.02:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.axvspan(-0.5, 1.5, alpha=0.06, color="green")
ax.axhline(0.269, color="gray", ls="--", lw=1, label="Center baseline Top-1 (26.9%)")
ax.set_xticks(x)
ax.set_xticklabels(cmp["Experiment"], rotation=28, ha="right", fontsize=8.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.legend(fontsize=9)
ax.set_title("Best-slice selection: all experiments (LOOCV = honest metric)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR/"exp5_comparison_bars.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved exp5_comparison_bars.png")

# ── Per-patient MAE ───────────────────────────────────────────────────────────
try: pp = loo_oof.copy()
except NameError: pp = kfold_oof.copy()

fig, ax = plt.subplots(figsize=(10,6))
mae_s  = pp.set_index("patient_id")["mae"].sort_values()
pal = {0:"#4CAF50",1:"#8BC34A",2:"#FFC107",3:"#FF9800",4:"#FF5722",5:"#F44336"}
cols = [pal.get(int(v),"#B71C1C") for v in mae_s.values]
ax.barh(mae_s.index, mae_s.values, color=cols, edgecolor="white", height=0.7)
ax.axvline(mae_s.mean(), color="black", ls="--", lw=1.5, label=f"mean={mae_s.mean():.2f}")
ax.set_title("Exp5 LOOCV MAE per patient"); ax.set_xlabel("MAE (slice units)"); ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"exp5_per_patient_mae.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved exp5_per_patient_mae.png")
"""))

# ── Cell 16: final summary ─────────────────────────────────────────────────────
cells.append(code("""
print("="*60)
print("EXPERIMENT 5 — FINAL SUMMARY")
print("="*60)
n = len(patient_ids)

try:
    print(f"\\nExp5A DINOv2-L Stack-Transformer ({lbl5}, n={n}):")
    print(f"  Top-1: {exp5_top1*100:.1f}%  ({int(round(exp5_top1*n))}/{n})")
    print(f"  Top-3: {exp5_top3*100:.1f}%  ({int(round(exp5_top3*n))}/{n})")
    print(f"  MAE  : {exp5_mae:.2f}")
except: pass

try:
    print(f"\\nExp5B Borda Ensemble (Exp4 + Exp5A):")
    print(f"  Top-1: {ens_top1*100:.1f}%  ({int(round(ens_top1*n))}/{n})")
    print(f"  Top-3: {ens_top3*100:.1f}%  ({int(round(ens_top3*n))}/{n})")
    print(f"  MAE  : {ens_mae:.2f}")
    print(f"  Correct: {borda_df[borda_df['top1_correct']==1]['patient_id'].tolist()}")
except: pass

print(f"\\nCenter baseline: Top-1=26.9% Top-3=57.7% MAE=1.31")
print(f"\\nOutputs saved to: {OUT_DIR.resolve()}")
print("="*60)
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
out = Path("experiment5_dinov2l_ensemble.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written {out}  ({len(cells)} cells, {out.stat().st_size//1024} KB)")

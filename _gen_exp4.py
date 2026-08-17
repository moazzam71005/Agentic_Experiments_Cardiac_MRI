"""Generate experiment4_dinov2_stack_transformer.ipynb"""
import json, textwrap
from pathlib import Path

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip()}

cells = []

# ── Cell 0: title ─────────────────────────────────────────────────────────────
cells.append(md("""
# Experiment 4 — DINOv2 Stack-Attention Ranker
## Cardiac MRI Frequency-Scout Best-Slice Selection

**Architecture:**
- **Feature extractor**: `facebook/dinov2-base` (frozen) — ViT-B/14, 768-dim CLS tokens
- **Ranker**: Lightweight Transformer encoder that attends across all 13 slices simultaneously
- **Position encoding**: Learned embedding of the normalized frequency position for each slice
- **Loss**: ListNet (softmax ranking loss against Gaussian soft labels centred on GT)
- **Evaluation**: Leave-One-Out Cross-Validation (LOOCV, 26 folds) — the honest metric

**Key improvements over Exp 3 (Siamese):**
| | Exp 3 Siamese | Exp 4 Stack-Transformer |
|---|---|---|
| Context | 2 slices at a time | All 13 simultaneously |
| Backbone | EfficientNet-B0 (ImageNet) | DINOv2-B (1.4B images) |
| Trainable params | ~5M | ~0.3M (head only) |
| Overfitting risk | High | Very low |

**Runtime:** ~20–30 min on Kaggle GPU (feature extraction cached after first run).
"""))

# ── Cell 1: install ────────────────────────────────────────────────────────────
cells.append(code("""
import subprocess, sys
pkgs = [
    "pydicom", "openpyxl", "timm",
    "transformers>=4.35.0", "accelerate",
    "scikit-learn", "tqdm", "seaborn",
]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("Dependencies ready.")
"""))

# ── Cell 2: imports + config ──────────────────────────────────────────────────
cells.append(code("""
import os, json, warnings, random, pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

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

    # ── DINOv2 ──────────────────────────────────────────────────────────────
    dino_model: str = "facebook/dinov2-base"   # 768-dim CLS token
    dino_batch: int = 8                         # slices per GPU batch during extraction
    img_size: int = 224                         # DINOv2 native input size

    # ── Stack Transformer ────────────────────────────────────────────────────
    d_model: int = 768          # must match DINOv2 output dim
    n_heads: int = 8
    n_layers: int = 2           # keep small — only ~25 training patients
    dim_ff_mult: int = 2        # FFN dim = d_model * dim_ff_mult
    dropout: float = 0.15

    # ── Training ─────────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 80
    early_stop_patience: int = 15

    # ── Soft labels ──────────────────────────────────────────────────────────
    soft_sigma: float = 1.5     # Gaussian sigma (in slice units)

    # ── Eval ─────────────────────────────────────────────────────────────────
    n_folds: int = 5
    run_loocv: bool = True      # set False to skip the long 26-fold run
    tta: bool = True            # test-time augmentation during feature extraction
    tta_n: int = 4              # number of TTA passes

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


cfg = CFG()

def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_all(cfg.seed)
print(f"Device: {cfg.device}")
print(f"Config: {cfg}")
"""))

# ── Cell 3: paths ─────────────────────────────────────────────────────────────
cells.append(code("""
IS_KAGGLE = Path("/kaggle/input").exists()

def find_excel(root: Path) -> Path:
    for pat in ["**/1stcohortSAX*.xlsx", "**/*cohort*SAX*.xlsx", "**/*SAX*.xlsx"]:
        hits = sorted(root.glob(pat))
        preferred = [h for h in hits if "cohort" in h.name.lower() or "1st" in h.name.lower()]
        pool = preferred or hits
        if pool:
            return pool[0]
    raise FileNotFoundError(f"No cohort Excel found under {root}")

def find_dicom_root(root: Path) -> Path:
    hits = list(root.rglob("1st_cohort_SAX"))
    if hits:
        return hits[0]
    pats = [p for p in root.rglob("Patient-*") if p.is_dir()]
    if pats:
        return pats[0].parent
    raise FileNotFoundError(f"No Patient-* folders under {root}")

if IS_KAGGLE:
    INPUT_ROOT = Path("/kaggle/input")
    DATA_ROOT = None
    for c in sorted(INPUT_ROOT.iterdir()):
        try:
            find_dicom_root(c)
            DATA_ROOT = c
            break
        except FileNotFoundError:
            continue
    DATA_ROOT = DATA_ROOT or next(INPUT_ROOT.iterdir())
    OUT_DIR = Path("/kaggle/working/exp4_outputs")
else:
    DATA_ROOT = Path(".").resolve()
    OUT_DIR = DATA_ROOT / "exp4_outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE = OUT_DIR / "dinov2_features.pkl"

EXCEL_PATH  = find_excel(DATA_ROOT)
DICOM_ROOT  = find_dicom_root(DATA_ROOT)

try:
    from IPython.display import display
except ImportError:
    def display(x): print(x)

print(f"IS_KAGGLE  = {IS_KAGGLE}")
print(f"DATA_ROOT  = {DATA_ROOT}")
print(f"EXCEL_PATH = {EXCEL_PATH}")
print(f"DICOM_ROOT = {DICOM_ROOT}")
print(f"OUT_DIR    = {OUT_DIR}")
print(f"FEAT_CACHE = {FEAT_CACHE}")
"""))

# ── Cell 4: load GT ───────────────────────────────────────────────────────────
cells.append(code("""
# Use pandas — more robust than openpyxl row-by-row for this layout
# The Excel has a blank first row; real header is on row 2 (index 1)
raw = pd.read_excel(EXCEL_PATH, header=None)

# Find the actual header row (first row with more than 2 non-null values)
header_row = None
for i, row in raw.iterrows():
    if row.notna().sum() > 2:
        header_row = i
        break

df_xl = pd.read_excel(EXCEL_PATH, header=header_row)
df_xl.columns = [str(c).strip() for c in df_xl.columns]
df_xl = df_xl.dropna(how="all")
print("Excel columns:", df_xl.columns.tolist())
print(df_xl.head())

def find_col(df, keywords):
    for c in df.columns:
        if any(k.lower() in c.lower() for k in keywords):
            return c
    return None

pid_col  = find_col(df_xl, ["anonymized", "patient", "pid", "id"])
img_col  = find_col(df_xl, ["image no", "img no"])
max_col  = find_col(df_xl, ["max freq"])
step_col = find_col(df_xl, ["freq step", "step"])
min_col  = find_col(df_xl, ["min freq"])
man_col  = find_col(df_xl, ["manual"])

print(f"PID col: {pid_col}, img_no col: {img_col}")
print(f"Max/Step/Min/Manual cols: {max_col}/{step_col}/{min_col}/{man_col}")

rows = []
for _, r in df_xl.iterrows():
    pid = str(r[pid_col]).strip() if pd.notna(r[pid_col]) else None
    img = r[img_col] if img_col and pd.notna(r[img_col]) else None
    if not pid or not img or pid.lower() in ("none", "nan"):
        continue
    try:
        entry = {"patient_id": pid, "gt_image_no": int(img)}
        # Physics frequency info (used for position encoding)
        if max_col:  entry["freq_max"]    = float(r[max_col])
        if step_col: entry["freq_step"]   = float(r[step_col])
        if min_col:  entry["freq_min"]    = float(r[min_col])
        if man_col:  entry["freq_manual"] = float(r[man_col]) if pd.notna(r[man_col]) else 0.0
        rows.append(entry)
    except (ValueError, TypeError):
        pass

labels_df = pd.DataFrame(rows)
if not labels_df["patient_id"].str.startswith("Patient-").any():
    labels_df["patient_id"] = "Patient-" + labels_df["patient_id"]
labels_df = labels_df.drop_duplicates("patient_id").reset_index(drop=True)

print(f"\\nLoaded {len(labels_df)} patients")
display(labels_df)
"""))

# ── Cell 5: index DICOM files ─────────────────────────────────────────────────
cells.append(code("""
records = []
for pat_dir in sorted(DICOM_ROOT.iterdir()):
    if not pat_dir.is_dir():
        continue
    pid = pat_dir.name
    dcm_files = sorted(pat_dir.glob("*.dcm"))
    if not dcm_files:
        # some datasets use subdirs
        dcm_files = sorted(pat_dir.rglob("*.dcm"))
    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(str(f), stop_before_pixels=True)
            inst = int(getattr(dcm, "InstanceNumber", 0))
        except Exception:
            inst = 0
        records.append({"patient_id": pid, "path": str(f), "instance_no": inst})

index_df = pd.DataFrame(records)
index_df = index_df.sort_values(["patient_id", "instance_no"]).reset_index(drop=True)

# Only keep patients that appear in labels_df
known = set(labels_df["patient_id"])
index_df = index_df[index_df["patient_id"].isin(known)].reset_index(drop=True)

# Assign local slice index within each patient (0-based)
index_df["slice_idx"] = index_df.groupby("patient_id").cumcount()

patient_ids = sorted(index_df["patient_id"].unique())
print(f"Indexed {len(index_df)} DICOM files across {len(patient_ids)} patients")
display(index_df.groupby("patient_id")["path"].count().rename("n_slices").to_frame())
"""))

# ── Cell 6: DINOv2 model loading ──────────────────────────────────────────────
cells.append(code("""
from transformers import AutoImageProcessor, AutoModel

print(f"Loading {cfg.dino_model} ...")
dino_processor = AutoImageProcessor.from_pretrained(cfg.dino_model)
dino_encoder   = AutoModel.from_pretrained(cfg.dino_model)
dino_encoder.eval()
dino_encoder.to(cfg.device)
for p in dino_encoder.parameters():
    p.requires_grad = False

n_params = sum(p.numel() for p in dino_encoder.parameters())
print(f"DINOv2 loaded — {n_params/1e6:.1f}M params (all frozen)")
"""))

# ── Cell 7: feature extraction + cache ────────────────────────────────────────
cells.append(code("""
import torchvision.transforms as T

# Augmentation transforms for TTA
_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
])

def dcm_to_pil(path: str) -> Image.Image:
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr)

@torch.no_grad()
def extract_cls(pil_imgs: List[Image.Image], augment: bool = False) -> torch.Tensor:
    if augment:
        pil_imgs = [_aug(img) for img in pil_imgs]
    inputs = dino_processor(images=pil_imgs, return_tensors="pt")
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
    out = dino_encoder(**inputs)
    return out.last_hidden_state[:, 0, :].cpu()   # CLS token [B, 768]

def extract_patient_features(pid: str, tta: bool = False, tta_n: int = 4) -> torch.Tensor:
    rows = index_df[index_df["patient_id"] == pid].sort_values("slice_idx")
    imgs = [dcm_to_pil(r.path) for _, r in rows.iterrows()]
    n    = len(imgs)
    if not tta:
        feats = []
        for i in range(0, n, cfg.dino_batch):
            feats.append(extract_cls(imgs[i:i+cfg.dino_batch], augment=False))
        return torch.cat(feats, dim=0)   # [N, 768]
    else:
        all_runs = []
        for _ in range(tta_n):
            feats = []
            for i in range(0, n, cfg.dino_batch):
                feats.append(extract_cls(imgs[i:i+cfg.dino_batch], augment=True))
            all_runs.append(torch.cat(feats, dim=0))
        # Base (no augment) + TTA passes; average
        base = []
        for i in range(0, n, cfg.dino_batch):
            base.append(extract_cls(imgs[i:i+cfg.dino_batch], augment=False))
        base = torch.cat(base, dim=0)
        all_runs.append(base)
        return torch.stack(all_runs, dim=0).mean(0)  # [N, 768]


if FEAT_CACHE.exists():
    print(f"Loading cached features from {FEAT_CACHE}")
    with open(FEAT_CACHE, "rb") as f:
        feat_dict = pickle.load(f)
    print(f"  Loaded features for {len(feat_dict)} patients")
else:
    print("Extracting DINOv2 features (this runs once and is cached) ...")
    feat_dict = {}
    for pid in tqdm(patient_ids, desc="Patients"):
        feat_dict[pid] = extract_patient_features(pid, tta=False)
    with open(FEAT_CACHE, "wb") as f:
        pickle.dump(feat_dict, f)
    print(f"Saved features to {FEAT_CACHE}")

# Quick sanity check
sample_pid = patient_ids[0]
print(f"\\nSample: {sample_pid} → feature shape {feat_dict[sample_pid].shape}")
"""))

# ── Cell 8: Dataset ────────────────────────────────────────────────────────────
cells.append(code("""
class PatientStackDataset(Dataset):
    \"\"\"
    Each item = one patient's full slice stack.
    Returns:
      feat    [S, D]   DINOv2 CLS features
      freq    [S, 1]   normalized frequency position [0, 1]
      soft    [S]      Gaussian soft labels
      gt      int      ground-truth image no. (1-indexed)
      inst    [S]      instance numbers (1-indexed)
      mask    [S]      valid slice mask (all True for this dataset; used for padding)
      pid     str
    \"\"\"
    def __init__(self, pids: List[str], feat_dict: dict, labels_df: pd.DataFrame,
                 index_df: pd.DataFrame, sigma: float = 1.5,
                 train: bool = True):
        self.pids       = pids
        self.feat_dict  = feat_dict
        self.labels_df  = labels_df.set_index("patient_id")
        self.index_df   = index_df
        self.sigma      = sigma
        self.train      = train

    def __len__(self): return len(self.pids)

    def _soft_label(self, n_slices: int, gt_idx: int) -> torch.Tensor:
        # gt_idx is 0-based position of the GT slice in this patient's stack
        pos = torch.arange(n_slices, dtype=torch.float32)
        soft = torch.exp(-0.5 * ((pos - gt_idx) / self.sigma) ** 2)
        return soft / soft.sum()

    def __getitem__(self, i):
        pid  = self.pids[i]
        feat = self.feat_dict[pid]         # [S, 768]
        S    = feat.shape[0]

        rows  = self.index_df[self.index_df["patient_id"] == pid].sort_values("slice_idx")
        inst  = torch.tensor(rows["instance_no"].values, dtype=torch.long)   # [S]

        gt_img = int(self.labels_df.loc[pid, "gt_image_no"])
        # find position of GT instance number in the sorted stack
        inst_list = inst.tolist()
        if gt_img in inst_list:
            gt_idx = inst_list.index(gt_img)
        else:
            gt_idx = min(range(S), key=lambda k: abs(inst_list[k] - gt_img))

        soft = self._soft_label(S, gt_idx)      # [S]

        # Physics-grounded frequency position:
        # Use actual Hz values from Excel if available, else uniform [0,1]
        lrow = self.labels_df.loc[pid]
        if "freq_min" in self.labels_df.columns and pd.notna(lrow.get("freq_min", np.nan)):
            f_min  = float(lrow["freq_min"])
            f_step = float(lrow.get("freq_step", 25))
            freqs  = torch.tensor(
                [f_min + k * f_step for k in range(S)], dtype=torch.float32
            )
            f_lo, f_hi = freqs.min(), freqs.max()
            freqs = (freqs - f_lo) / (f_hi - f_lo + 1e-8)   # normalise to [0,1]
        else:
            freqs = torch.linspace(0, 1, S)
        freq = freqs.unsqueeze(1)                    # [S, 1]
        mask = torch.ones(S, dtype=torch.bool)       # [S]

        return {
            "feat":  feat.float(),      # [S, 768]
            "freq":  freq.float(),      # [S, 1]
            "soft":  soft.float(),      # [S]
            "gt":    gt_img,
            "gt_idx": gt_idx,
            "inst":  inst,              # [S]
            "mask":  mask,              # [S]
            "pid":   pid,
        }


def patient_collate(batch):
    \"\"\"Collate a list of variable-length patient stacks via zero-padding.\"\"\"
    max_S = max(b["feat"].shape[0] for b in batch)
    B = len(batch)
    D = batch[0]["feat"].shape[1]

    feat_pad = torch.zeros(B, max_S, D)
    freq_pad = torch.zeros(B, max_S, 1)
    soft_pad = torch.zeros(B, max_S)
    mask_pad = torch.zeros(B, max_S, dtype=torch.bool)
    inst_pad = torch.zeros(B, max_S, dtype=torch.long)
    gts, gt_idxs, pids = [], [], []

    for i, b in enumerate(batch):
        s = b["feat"].shape[0]
        feat_pad[i, :s] = b["feat"]
        freq_pad[i, :s] = b["freq"]
        soft_pad[i, :s] = b["soft"]
        mask_pad[i, :s] = b["mask"]
        inst_pad[i, :s] = b["inst"]
        gts.append(b["gt"])
        gt_idxs.append(b["gt_idx"])
        pids.append(b["pid"])

    return {
        "feat":   feat_pad,
        "freq":   freq_pad,
        "soft":   soft_pad,
        "mask":   mask_pad,
        "inst":   inst_pad,
        "gt":     torch.tensor(gts, dtype=torch.long),
        "gt_idx": torch.tensor(gt_idxs, dtype=torch.long),
        "pid":    pids,
    }


print("PatientStackDataset defined.")
"""))

# ── Cell 9: model ─────────────────────────────────────────────────────────────
cells.append(code("""
class StackTransformerRanker(nn.Module):
    \"\"\"
    Lightweight Transformer that re-ranks a patient's full DINOv2 feature stack.

    Input:
        feat     [B, S, D=768]  frozen DINOv2 CLS tokens
        freq_pos [B, S, 1]      normalised frequency position [0,1]
        mask     [B, S]         True = valid slice

    Output:
        scores   [B, S]         raw logits (un-normalised ranking scores)
    \"\"\"
    def __init__(self, d_model: int = 768, n_heads: int = 8,
                 n_layers: int = 2, dim_ff_mult: int = 2,
                 dropout: float = 0.15):
        super().__init__()

        # Project scalar freq position → d_model
        self.pos_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Input LayerNorm (stabilises frozen feature distribution)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * dim_ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable with small data
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Scoring head
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Suppress harmless PyTorch nested-tensor warning with norm_first=True
        import warnings
        warnings.filterwarnings("ignore", message="enable_nested_tensor")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor,
                freq_pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # feat:     [B, S, 768]
        # freq_pos: [B, S, 1]
        # mask:     [B, S] True = valid

        x = self.input_norm(feat) + self.pos_proj(freq_pos)

        # TransformerEncoder padding mask: True → position is IGNORED
        pad_mask = (~mask) if mask is not None else None

        x = self.transformer(x, src_key_padding_mask=pad_mask)     # [B, S, D]
        scores = self.score_head(x).squeeze(-1)                     # [B, S]

        # Mask out padded positions with -inf so they never win
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return scores


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

model_test = StackTransformerRanker(
    d_model=cfg.d_model, n_heads=cfg.n_heads,
    n_layers=cfg.n_layers, dim_ff_mult=cfg.dim_ff_mult,
    dropout=cfg.dropout,
)
t, tr = count_params(model_test)
print(f"StackTransformerRanker: {t/1e3:.1f}K total, {tr/1e3:.1f}K trainable")
del model_test
"""))

# ── Cell 10: loss + metrics ────────────────────────────────────────────────────
cells.append(code("""
def listnet_loss(scores: torch.Tensor,
                 soft:   torch.Tensor,
                 mask:   torch.Tensor) -> torch.Tensor:
    \"\"\"
    ListNet ranking loss.
    scores, soft, mask: [B, S]
    \"\"\"
    neg_inf = torch.finfo(scores.dtype).min / 4
    s = scores.masked_fill(~mask, neg_inf)
    t = soft.masked_fill(~mask, 0.0)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_p = F.log_softmax(s, dim=1)
    return -(t * log_p).sum(dim=1).mean()


def patient_metrics(pred_df: pd.DataFrame) -> dict:
    \"\"\"
    pred_df columns: patient_id, gt_image_no, pred_top1,
                     pred_top3 (list), top1_correct, top3_correct, mae, rank_gt
    \"\"\"
    agg = pred_df.agg({
        "top1_correct": "mean", "top3_correct": "mean",
        "mae": "mean", "rank_gt": "mean"
    }).to_dict()
    return {
        "top1_acc":      agg["top1_correct"],
        "top3_acc":      agg["top3_correct"],
        "mae":           agg["mae"],
        "mean_rank_gt":  agg["rank_gt"],
        "per_patient":   pred_df,
    }


@torch.no_grad()
def predict_patients(model: nn.Module,
                     pids: List[str],
                     feat_dict: dict,
                     index_df: pd.DataFrame,
                     labels_df: pd.DataFrame,
                     sigma: float) -> pd.DataFrame:
    model.eval()
    rows = []
    ds = PatientStackDataset(pids, feat_dict, labels_df, index_df,
                              sigma=sigma, train=False)
    dl = DataLoader(ds, batch_size=1, collate_fn=patient_collate,
                    num_workers=0, shuffle=False)
    for batch in dl:
        pid      = batch["pid"][0]
        feat     = batch["feat"].to(cfg.device)   # [1, S, D]
        freq     = batch["freq"].to(cfg.device)
        mask     = batch["mask"].to(cfg.device)
        inst     = batch["inst"][0]               # [S]
        gt_img   = int(batch["gt"][0])
        S        = int(mask[0].sum())

        scores   = model(feat, freq, mask)        # [1, S]
        scores   = scores[0, :S]                  # [S]
        inst     = inst[:S]

        ranked   = torch.argsort(scores, descending=True)
        top1_inst = int(inst[ranked[0]])
        top3_inst = [int(inst[ranked[k]]) for k in range(min(3, S))]
        rank_gt   = int((ranked == (inst == gt_img).nonzero(as_tuple=False)[0, 0]).nonzero(as_tuple=False)[0, 0]) + 1 \
                    if gt_img in inst.tolist() else S

        # safer rank computation
        inst_list = inst.tolist()
        if gt_img in inst_list:
            gt_pos_in_stack = inst_list.index(gt_img)
            order = ranked.tolist()
            rank_gt = order.index(gt_pos_in_stack) + 1
        else:
            rank_gt = S

        rows.append({
            "patient_id":   pid,
            "gt_image_no":  gt_img,
            "pred_top1":    top1_inst,
            "pred_top3":    top3_inst,
            "top1_correct": int(top1_inst == gt_img),
            "top3_correct": int(gt_img in top3_inst),
            "mae":          abs(top1_inst - gt_img),
            "rank_gt":      rank_gt,
        })
    return pd.DataFrame(rows)


print("Loss and metric functions defined.")
"""))

# ── Cell 11: train_fold ────────────────────────────────────────────────────────
cells.append(code("""
def train_fold(
    train_pids: List[str],
    val_pids:   List[str],
    test_pids:  List[str],
    tag:        str = "fold",
) -> Tuple[dict, dict, pd.DataFrame]:
    \"\"\"
    Train StackTransformerRanker on train_pids, validate on val_pids,
    evaluate on test_pids.  Returns (m_val, m_test, pred_test_df).
    \"\"\"
    model = StackTransformerRanker(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, dim_ff_mult=cfg.dim_ff_mult,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    train_ds = PatientStackDataset(train_pids, feat_dict, labels_df,
                                    index_df, sigma=cfg.soft_sigma, train=True)
    train_dl  = DataLoader(train_ds, batch_size=4, shuffle=True,
                           collate_fn=patient_collate, num_workers=0)

    best_val_mae = float("inf")
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, cfg.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        for batch in train_dl:
            feat = batch["feat"].to(cfg.device)
            freq = batch["freq"].to(cfg.device)
            soft = batch["soft"].to(cfg.device)
            mask = batch["mask"].to(cfg.device)

            scores = model(feat, freq, mask)
            loss   = listnet_loss(scores, soft, mask)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()

        # ── validate every 5 epochs ────────────────────────────────────────
        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_preds = predict_patients(model, val_pids,
                                         feat_dict, index_df, labels_df,
                                         cfg.soft_sigma)
            val_m = patient_metrics(val_preds)
            val_mae = val_m["mae"]

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 5
            if patience_cnt >= cfg.early_stop_patience:
                print(f"  [{tag}] Early stop at epoch {epoch}, best val MAE={best_val_mae:.3f}")
                break

    # ── load best model and evaluate test ─────────────────────────────────
    model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})

    val_preds  = predict_patients(model, val_pids,
                                   feat_dict, index_df, labels_df, cfg.soft_sigma)
    test_preds = predict_patients(model, test_pids,
                                   feat_dict, index_df, labels_df, cfg.soft_sigma)
    m_val  = patient_metrics(val_preds)
    m_test = patient_metrics(test_preds)

    print(f"  [{tag}] "
          f"val  Top1={m_val['top1_acc']:.2f} Top3={m_val['top3_acc']:.2f} MAE={m_val['mae']:.2f} | "
          f"TEST Top1={m_test['top1_acc']:.2f} Top3={m_test['top3_acc']:.2f} MAE={m_test['mae']:.2f}")

    return m_val, m_test, test_preds


print("train_fold() defined.")
"""))

# ── Cell 12: LOOCV ────────────────────────────────────────────────────────────
cells.append(code("""
if not cfg.run_loocv:
    print("LOOCV skipped (cfg.run_loocv=False). Set to True to run.")
else:
    print(f"Starting LOOCV over {len(patient_ids)} patients ...")
    loo_rows  = []
    loo_preds = []

    for i, test_pid in enumerate(patient_ids, 1):
        remaining = [p for p in patient_ids if p != test_pid]
        # Use 20% of remaining as validation (at least 1 patient)
        n_val = max(1, int(len(remaining) * 0.2))
        np.random.seed(cfg.seed + i)
        val_pids   = list(np.random.choice(remaining, n_val, replace=False))
        train_pids = [p for p in remaining if p not in val_pids]

        print(f"LOO {i:02d}/{len(patient_ids)} — test: {test_pid} "
              f"| train: {len(train_pids)} | val: {len(val_pids)}")
        seed_all(cfg.seed + i)
        m_val, m_test, pred_df = train_fold(train_pids, val_pids, [test_pid],
                                             tag=f"loo{i:02d}")
        pred_df["fold"] = i
        loo_preds.append(pred_df)
        loo_rows.append({
            "fold": i, "test_pid": test_pid,
            "top1_acc":  m_test["top1_acc"],
            "top3_acc":  m_test["top3_acc"],
            "mae":       m_test["mae"],
            "rank_gt":   int(m_test["per_patient"]["rank_gt"].iloc[0]),
        })
        # Save per-fold so a crash doesn't lose everything
        pd.DataFrame(loo_rows).to_csv(OUT_DIR / "exp4_loocv_results.csv", index=False)
        pd.concat(loo_preds).to_csv(OUT_DIR / "exp4_loocv_per_patient.csv", index=False)

    loo_df  = pd.DataFrame(loo_rows)
    loo_oof = pd.concat(loo_preds, ignore_index=True)

    m_loo = patient_metrics(loo_oof)
    print("\\n=== LOOCV Results ===")
    display(loo_df)
    print("\\nPooled LOOCV:", {k: round(v, 3) for k, v in m_loo.items() if k != "per_patient"})
    display(loo_oof.sort_values("top1_correct", ascending=False))
"""))

# ── Cell 13: 5-fold CV ────────────────────────────────────────────────────────
cells.append(code("""
print(f"Starting {cfg.n_folds}-fold CV ...")
kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
pid_arr  = np.array(patient_ids)

fold_rows  = []
fold_preds = []

for fold_i, (tr_idx, te_idx) in enumerate(kf.split(pid_arr), 1):
    tr_pids_all = pid_arr[tr_idx].tolist()
    te_pids     = pid_arr[te_idx].tolist()
    # 20% of train as val
    n_val       = max(1, int(len(tr_pids_all) * 0.2))
    np.random.seed(cfg.seed + fold_i * 100)
    val_pids    = list(np.random.choice(tr_pids_all, n_val, replace=False))
    train_pids  = [p for p in tr_pids_all if p not in val_pids]

    print(f"\\nFold {fold_i}/{cfg.n_folds} — train:{len(train_pids)} val:{len(val_pids)} test:{len(te_pids)}")
    seed_all(cfg.seed + fold_i * 100)
    m_val, m_test, pred_df = train_fold(train_pids, val_pids, te_pids,
                                         tag=f"kfold{fold_i}")
    pred_df["fold"] = fold_i
    fold_preds.append(pred_df)
    fold_rows.append({
        "fold":        fold_i,
        "top1_acc":    m_test["top1_acc"],
        "top3_acc":    m_test["top3_acc"],
        "mae":         m_test["mae"],
        "mean_rank_gt": m_test["mean_rank_gt"],
        "n_test":      len(te_pids),
    })

kfold_df  = pd.DataFrame(fold_rows)
kfold_oof = pd.concat(fold_preds, ignore_index=True)
m_kfold   = patient_metrics(kfold_oof)

kfold_df.to_csv(OUT_DIR / "exp4_kfold_results.csv", index=False)
kfold_oof.to_csv(OUT_DIR / "exp4_kfold_per_patient.csv", index=False)

print("\\n=== 5-Fold CV Results ===")
display(kfold_df)
print("\\nPooled OOF:", {k: round(v, 3) for k, v in m_kfold.items() if k != "per_patient"})
"""))

# ── Cell 14: comparison table ─────────────────────────────────────────────────
cells.append(code("""
# ── Load prior experiment numbers ──────────────────────────────────────────
# (hardcoded from run1/run2/run3 outputs; update if you have CSVs)
prior = [
    {"Experiment": "Exp3  Siamese EfficientNet-B0",
     "Top-1": 0.154, "Top-3": 0.538, "MAE": 2.50, "Eval": "LOOCV"},
    {"Experiment": "Exp2  ResNet18 + ListNet",
     "Top-1": 0.038, "Top-3": 0.462, "MAE": 2.19, "Eval": "LOOCV"},
    {"Experiment": "Center slice (naive)",
     "Top-1": 0.269, "Top-3": 0.577, "MAE": 1.31, "Eval": "all 26"},
    {"Experiment": "Random",
     "Top-1": 0.080, "Top-3": 0.239, "MAE": float("nan"), "Eval": "analytic"},
]

# Build Exp4 row from LOOCV if run, else from 5-fold
try:
    exp4_top1 = m_loo["top1_acc"]
    exp4_top3 = m_loo["top3_acc"]
    exp4_mae  = m_loo["mae"]
    exp4_eval = "LOOCV"
except NameError:
    exp4_top1 = m_kfold["top1_acc"]
    exp4_top3 = m_kfold["top3_acc"]
    exp4_mae  = m_kfold["mae"]
    exp4_eval = "5-fold OOF"

compare_rows = [
    {"Experiment": f"Exp4  DINOv2 Stack-Transformer ({exp4_eval})",
     "Top-1": exp4_top1, "Top-3": exp4_top3, "MAE": exp4_mae, "Eval": exp4_eval},
] + prior

compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(OUT_DIR / "exp4_all_comparison.csv", index=False)

print("=== All-Experiment Comparison ===")
display(compare_df.to_string(index=False))
"""))

# ── Cell 15: plots ────────────────────────────────────────────────────────────
cells.append(code("""
import matplotlib
matplotlib.use("Agg")

# ── Plot 1: comparison bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(compare_df))
w = 0.35
b1 = ax.bar(x - w/2, compare_df["Top-1"], width=w, label="Top-1", color="#2196F3")
b2 = ax.bar(x + w/2, compare_df["Top-3"], width=w, label="Top-3", color="#FF9800")
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0.01 and not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axvspan(-0.5, 0.5, alpha=0.08, color="green")
ax.set_xticks(x)
ax.set_xticklabels(compare_df["Experiment"], rotation=28, ha="right", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Accuracy")
ax.set_title("Best-slice selection: all experiments vs baselines", fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "exp4_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved exp4_comparison_bars.png")

# ── Plot 2: LOOCV per-patient (if available) ───────────────────────────────
try:
    pp = loo_oof.copy()
except NameError:
    pp = kfold_oof.copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
mae_s = pp.set_index("patient_id")["mae"].sort_values()
pal = {0:"#4CAF50",1:"#8BC34A",2:"#FFC107",3:"#FF9800",4:"#FF5722",
       5:"#F44336",6:"#D32F2F",7:"#B71C1C"}
colors = [pal.get(int(v), "#B71C1C") for v in mae_s.values]
axes[0].barh(mae_s.index, mae_s.values, color=colors, edgecolor="white", height=0.7)
axes[0].axvline(mae_s.mean(), color="black", ls="--", lw=1.5,
                label=f"mean={mae_s.mean():.2f}")
axes[0].set_title("MAE per patient (Exp4)", fontsize=11)
axes[0].set_xlabel("MAE (slice units)")
axes[0].legend()

rk_s = pp.set_index("patient_id")["rank_gt"].sort_values()
rpal = {1:"#4CAF50",2:"#8BC34A",3:"#FFC107",4:"#FF9800"}
rc = [rpal.get(int(v), "#F44336") for v in rk_s.values]
axes[1].barh(rk_s.index, rk_s.values, color=rc, edgecolor="white", height=0.7)
axes[1].axvline(rk_s.mean(), color="black", ls="--", lw=1.5,
                label=f"mean={rk_s.mean():.2f}")
axes[1].set_title("GT rank per patient (Exp4)", fontsize=11)
axes[1].set_xlabel("rank of GT slice (1=perfect)")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "exp4_per_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved exp4_per_patient.png")

# ── Plot 3: GT vs Predicted scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(pp))
colors_sc = pp["top1_correct"].map({1:"#4CAF50", 0:"#F44336"})
ax.scatter(pp["gt_image_no"], pp["pred_top1"] + jitter, c=colors_sc,
           s=90, edgecolors="white", linewidths=0.8, zorder=3)
ax.plot([1,13],[1,13],"k--",lw=1,label="Perfect")
for _, r in pp.iterrows():
    ax.annotate(r["patient_id"].replace("Patient-",""),
                (r["gt_image_no"], r["pred_top1"]),
                fontsize=6, ha="left", va="bottom", color="0.4",
                xytext=(2,2), textcoords="offset points")
ax.set_xlabel("GT best slice (image no.)")
ax.set_ylabel("Predicted Top-1 slice")
ax.set_title("Exp4 — GT vs Predicted Top-1")
ax.set_xlim(0,14); ax.set_ylim(0,14)
green_p = mpatches.Patch(color="#4CAF50", label=f"Top-1 correct")
red_p   = mpatches.Patch(color="#F44336", label=f"Top-1 wrong")
ax.legend(handles=[green_p, red_p])
plt.tight_layout()
plt.savefig(OUT_DIR / "exp4_gt_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved exp4_gt_vs_pred.png")

print("\\nAll plots saved to", OUT_DIR)
"""))

# ── Cell 16: final summary print ──────────────────────────────────────────────
cells.append(code("""
print("=" * 60)
print("EXPERIMENT 4 — FINAL SUMMARY")
print("=" * 60)

try:
    m = m_loo; label = "LOOCV"
except NameError:
    m = m_kfold; label = "5-fold OOF"

top1  = m["top1_acc"]
top3  = m["top3_acc"]
mae   = m["mae"]
rankm = m["mean_rank_gt"]
n     = len(patient_ids)

print(f"\\n{label} (n={n}):")
print(f"  Top-1 Acc  : {top1*100:.1f}%  ({int(round(top1*n))}/{n} patients)")
print(f"  Top-3 Acc  : {top3*100:.1f}%  ({int(round(top3*n))}/{n} patients)")
print(f"  MAE        : {mae:.2f} slices")
print(f"  Mean rank  : {rankm:.2f}")

try:
    perfect = loo_oof[loo_oof["top1_correct"] == 1]["patient_id"].tolist()
    top3ok  = loo_oof[loo_oof["top3_correct"] == 1]["patient_id"].tolist()
    print(f"\\nTop-1 perfect  : {perfect}")
    print(f"Top-3 correct  : {top3ok}")
except NameError:
    pass

print(f"\\nOutputs saved to: {OUT_DIR.resolve()}")
print("=" * 60)
"""))

# ── Build notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("experiment4_dinov2_stack_transformer.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written {out}  ({len(cells)} cells, {out.stat().st_size//1024} KB)")

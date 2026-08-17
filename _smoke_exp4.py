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

# ---- CELL BREAK ----

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

# ---- CELL BREAK ----

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

print(f"\nLoaded {len(labels_df)} patients")
display(labels_df)

# ---- CELL BREAK ----

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
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
    EXP4_DIR  = Path("/kaggle/input/exp4-outputs") if Path("/kaggle/input/exp4-outputs").exists()                 else Path("/kaggle/working/exp4_outputs")
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
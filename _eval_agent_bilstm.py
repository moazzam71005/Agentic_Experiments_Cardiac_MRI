"""
Evaluate Agentic Pipeline with BiLSTM replacing CNN.
Combines BiLSTM sequence score + BFM (Banded Fraction) + Homogeneity + Center prior.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

base_dir = Path("e:/CINE-Test1")
gt_excel = base_dir / "1stcohortSAX (1).xlsx"
bilstm_csv = base_dir / "outputs_exp17_dinov2_bilstm" / "dinov2_bilstm_loocv_results.csv"
banding_csv = base_dir / "metrics_lab" / "outputs" / "banding_metrics.csv"
signal_csv = base_dir / "metrics_lab" / "outputs" / "signal_metrics.csv"

df_gt = pd.read_excel(gt_excel, header=1)
id_col = [c for c in df_gt.columns if "Anonymized" in str(c) or "ID" in str(c)][0]
img_col = [c for c in df_gt.columns if "image" in str(c).lower()][0]
df_gt = df_gt.dropna(subset=[id_col]).reset_index(drop=True)

df_banding = pd.read_csv(banding_csv)
df_signal = pd.read_csv(signal_csv)

print("Evaluating Hybrid BiLSTM + Agentic Pipeline...")

"""
Extract all Exp3 results directly from run3.ipynb cell outputs,
generate the comparison table + plots, save to exp3_outputs/.
No retraining required.
"""
import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: just saves files, no pop-ups
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="notebook")

OUT_DIR = Path("exp3_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Hardcode all results extracted from run3.ipynb outputs ─────────────────

# LOOCV per-patient (from cell 21 pooled display_data in run3.ipynb)
loocv_per_patient = [
    # patient_id       gt  pred1  pred_top3               top1  top3  mae  rank
    ("Patient-BEV",    7,   7,   [7, 8, 9],               1,    1,    0,   1),
    ("Patient-BGT",    7,   4,   [4, 5, 2],               0,    0,    3,   5),
    ("Patient-BVE",    6,   7,   [7, 6, 8],               0,    1,    1,   2),
    ("Patient-CEA",    7,   6,   [6, 7, 5],               0,    1,    1,   2),
    ("Patient-CEJP",   4,   1,   [1, 2, 13],              0,    0,    3,   4),
    ("Patient-DEM",    9,   6,   [6, 5, 4],               0,    0,    3,   7),
    ("Patient-FAJ",    9,   7,   [7, 5, 2],               0,    0,    2,   4),
    ("Patient-FEC",    8,   1,   [1, 12, 3],              0,    0,    7,   6),
    ("Patient-GJ",     9,   8,   [8, 9, 10],              0,    1,    1,   2),
    ("Patient-HMJ",    8,   8,   [8, 6, 7],               1,    1,    0,   1),
    ("Patient-HRJ",    7,  12,   [12, 7, 10],             0,    1,    5,   2),
    ("Patient-IAM",    8,   7,   [7, 8, 9],               0,    1,    1,   2),
    ("Patient-INA",   11,   6,   [6, 8, 7],               0,    0,    5,  10),
    ("Patient-KP",     6,   5,   [5, 6, 7],               0,    1,    1,   2),
    ("Patient-MJ",     7,   4,   [4, 5, 6],               0,    0,    3,   6),
    ("Patient-MS",     9,   6,   [6, 8, 7],               0,    0,    3,   7),
    ("Patient-MSR",   10,   8,   [8, 9, 10],              0,    1,    2,   3),
    ("Patient-OAF",    9,   3,   [3, 6, 5],               0,    0,    6,   5),
    ("Patient-PJ",     8,   6,   [6, 1, 5],               0,    0,    2,  11),
    ("Patient-RAK",    7,   7,   [7, 9, 3],               1,    1,    0,   1),
    ("Patient-RK",     9,   4,   [4, 5, 8],               0,    0,    5,  12),
    ("Patient-SK",     8,   6,   [6, 10, 8],              0,    1,    2,   3),
    ("Patient-SWJ",    7,   4,   [4, 7, 5],               0,    1,    3,   2),
    ("Patient-TL",     9,   5,   [5, 7, 8],               0,    0,    4,   7),
    ("Patient-WBC",    8,   6,   [6, 7, 8],               0,    1,    2,   3),
    ("Patient-WMR",    8,   8,   [8, 5, 10],              1,    1,    0,   1),
]
loo_pp = pd.DataFrame(loocv_per_patient,
    columns=["patient_id","gt_image_no","pred_top1","pred_top3",
             "top1_correct","top3_correct","mae","rank_gt"])

# LOOCV fold-level (from loo_df display in cell 21)
loocv_folds = [
    (1,"Patient-BEV",  0.0, 1.0, 1.0, 2),
    (2,"Patient-BGT",  0.0, 0.0, 1.0, 6),
    (3,"Patient-BVE",  0.0, 1.0, 1.0, 3),
    (4,"Patient-CEA",  0.0, 0.0, 4.0, 8),
    (5,"Patient-CEJP", 0.0, 1.0, 2.0, 3),
    (6,"Patient-DEM",  0.0, 1.0, 1.0, 2),
    (7,"Patient-FAJ",  0.0, 0.0, 2.0, 7),
    (8,"Patient-FEC",  0.0, 0.0, 3.0, 4),
    (9,"Patient-GJ",   0.0, 1.0, 2.0, 2),
    (10,"Patient-HMJ", 0.0, 0.0, 1.0, 4),
    (11,"Patient-HRJ", 0.0, 0.0, 5.0, 7),
    (12,"Patient-IAM", 0.0, 0.0, 3.0, 5),
    (13,"Patient-INA", 0.0, 0.0, 4.0, 5),
    (14,"Patient-KP",  0.0, 1.0, 1.0, 2),
    (15,"Patient-MJ",  0.0, 1.0, 2.0, 2),
    (16,"Patient-MS",  1.0, 1.0, 0.0, 1),
    (17,"Patient-MSR", 0.0, 0.0, 1.0, 4),
    (18,"Patient-OAF", 0.0, 1.0, 2.0, 2),
    (19,"Patient-PJ",  0.0, 1.0, 1.0, 2),
    (20,"Patient-RAK", 0.0, 0.0, 2.0, 4),
    (21,"Patient-RK",  0.0, 0.0, 4.0, 7),
    (22,"Patient-SK",  0.0, 0.0, 3.0, 4),
    (23,"Patient-SWJ", 0.0, 1.0, 3.0, 2),
    (24,"Patient-TL",  0.0, 0.0, 3.0,10),
    (25,"Patient-WBC", 0.0, 1.0, 1.0, 2),
    (26,"Patient-WMR", 0.0, 0.0, 4.0, 6),
]
loo_df = pd.DataFrame(loocv_folds,
    columns=["fold","test_pid","top1_acc","top3_acc","mae","rank_gt"])

# LOOCV pooled (from run3 cell 21)
loo_pooled = {"top1_acc": 0.154, "top3_acc": 0.538, "mae": 2.500, "mean_rank_gt": 4.269}

# 5-fold results (from run3 cell 19)
kfold_data = [
    (1, 0.000, 0.167, 4.667, 6.333, 6),
    (2, 0.000, 0.600, 2.800, 4.800, 5),
    (3, 0.000, 0.400, 3.200, 5.200, 5),
    (4, 0.200, 0.400, 2.800, 4.200, 5),
    (5, 0.000, 0.400, 2.400, 5.800, 5),
]
fold_df = pd.DataFrame(kfold_data,
    columns=["fold","top1_acc","top3_acc","mae","mean_rank_gt","n_test"])

# Prior experiments (from run1 / run2 outputs)
# Exp1 5-fold means from run1
exp1_kfold = {"top1_acc": 0.227, "top3_acc": 0.613, "mae": 1.350}
# Exp2 LOOCV from run2
exp2_loocv = {"top1_acc": 0.038, "top3_acc": 0.462, "mae": 2.192}
# Exp2 5-fold mean from run2
exp2_kfold = {"top1_acc": 0.227, "top3_acc": 0.613, "mae": 1.347}

# Baselines (computed analytically from label distribution)
# step25: 13 slices each (25 patients), atypical KP: 7 slices
# center slice = slice 7 for 13-slice, slice 4 for 7-slice
# GT values and n_slices per patient
gt_info = [
    ("Patient-BEV",7,13),("Patient-BGT",7,13),("Patient-BVE",6,12),
    ("Patient-CEA",7,13),("Patient-CEJP",4,13),("Patient-DEM",9,13),
    ("Patient-FAJ",9,13),("Patient-FEC",8,13),("Patient-GJ",9,13),
    ("Patient-HMJ",8,13),("Patient-HRJ",7,13),("Patient-IAM",8,13),
    ("Patient-INA",11,13),("Patient-KP",6,7),("Patient-MJ",7,13),
    ("Patient-MS",9,13),("Patient-MSR",10,13),("Patient-OAF",9,13),
    ("Patient-PJ",8,13),("Patient-RAK",7,13),("Patient-RK",9,13),
    ("Patient-SK",8,13),("Patient-SWJ",7,13),("Patient-TL",9,13),
    ("Patient-WBC",8,13),("Patient-WMR",8,13),
]
gt_df = pd.DataFrame(gt_info, columns=["patient_id","gt","n_slices"])

def center_baseline(gt_df):
    rows = []
    for _, r in gt_df.iterrows():
        inst = list(range(1, int(r["n_slices"]) + 1))
        pred = inst[len(inst) // 2]
        gt   = int(r["gt"])
        neigh = sorted(inst, key=lambda i: (abs(i-pred),i))[:3]
        rows.append({"top1": int(pred==gt), "top3": int(gt in neigh), "mae": abs(pred-gt)})
    d = pd.DataFrame(rows)
    return d.mean().to_dict()

def mode_baseline(gt_df):
    mode_gt = int(gt_df["gt"].mode().iloc[0])
    rows = []
    for _, r in gt_df.iterrows():
        inst = list(range(1, int(r["n_slices"]) + 1))
        gt   = int(r["gt"])
        pred = mode_gt if mode_gt in inst else inst[len(inst)//2]
        neigh = sorted(inst, key=lambda i: (abs(i-pred),i))[:3]
        rows.append({"top1": int(pred==gt), "top3": int(gt in neigh), "mae": abs(pred-gt)})
    d = pd.DataFrame(rows)
    return d.mean().to_dict()

bl_center = center_baseline(gt_df)
bl_mode   = mode_baseline(gt_df)
ns        = gt_df["n_slices"].values
bl_random = {"top1_acc": float(np.mean(1/ns)),
             "top3_acc": float(np.mean([min(3,n)/n for n in ns])),
             "mae": float("nan")}

print("Baselines computed:")
print(f"  Center: Top1={bl_center['top1']:.3f} Top3={bl_center['top3']:.3f} MAE={bl_center['mae']:.3f}")
print(f"  Mode:   Top1={bl_mode['top1']:.3f}   Top3={bl_mode['top3']:.3f} MAE={bl_mode['mae']:.3f}")

# ── 2. Comparison table ───────────────────────────────────────────────────────

compare_rows = [
    # From best (honest) to baseline
    {"Experiment": "Exp3  Siamese EfficientNet-B0 (LOOCV)",
     "Top-1 Acc": loo_pooled["top1_acc"],
     "Top-3 Acc": loo_pooled["top3_acc"],
     "MAE":       loo_pooled["mae"],
     "Eval":      "LOOCV n=26"},
    {"Experiment": "Exp3  Siamese EfficientNet-B0 (5-fold OOF)",
     "Top-1 Acc": 0.038, "Top-3 Acc": 0.385, "MAE": 3.231,
     "Eval":      "5-fold OOF"},
    {"Experiment": "Exp2  ResNet18  ListNet (LOOCV)",
     "Top-1 Acc": exp2_loocv["top1_acc"],
     "Top-3 Acc": exp2_loocv["top3_acc"],
     "MAE":       exp2_loocv["mae"],
     "Eval":      "LOOCV n=26"},
    {"Experiment": "Exp1  ResNet18  BCE  (5-fold mean)",
     "Top-1 Acc": exp1_kfold["top1_acc"],
     "Top-3 Acc": exp1_kfold["top3_acc"],
     "MAE":       exp1_kfold["mae"],
     "Eval":      "5-fold mean"},
    {"Experiment": "Center slice (naive)",
     "Top-1 Acc": bl_center["top1"], "Top-3 Acc": bl_center["top3"],
     "MAE": bl_center["mae"], "Eval": "all 26 patients"},
    {"Experiment": "Global-mode GT (naive)",
     "Top-1 Acc": bl_mode["top1"], "Top-3 Acc": bl_mode["top3"],
     "MAE": bl_mode["mae"], "Eval": "all 26 patients"},
    {"Experiment": "Random (expected)",
     "Top-1 Acc": bl_random["top1_acc"], "Top-3 Acc": bl_random["top3_acc"],
     "MAE": float("nan"), "Eval": "analytic"},
]
compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(OUT_DIR / "exp3_all_experiment_comparison.csv", index=False)
print("\n=== All-experiment comparison ===")
print(compare_df.to_string(index=False))

# Save per-patient to CSV
loo_pp.to_csv(OUT_DIR / "exp3_loocv_per_patient.csv", index=False)
loo_df.to_csv(OUT_DIR / "exp3_loocv_results.csv", index=False)
fold_df.to_csv(OUT_DIR / "exp3_kfold_results.csv", index=False)

# ── 3. Plot 1: All-experiment bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(compare_df))
w = 0.35

def bar_label(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

b1 = ax.bar(x - w/2, compare_df["Top-1 Acc"].fillna(0), width=w, label="Top-1 Acc",
            color=["#2196F3"]*4 + ["#90CAF9"]*3)
b2 = ax.bar(x + w/2, compare_df["Top-3 Acc"].fillna(0), width=w, label="Top-3 Acc",
            color=["#FF9800"]*4 + ["#FFCC80"]*3)
bar_label(ax, b1)
bar_label(ax, b2)

# Highlight Exp3 LOOCV (best model) with a box
ax.axvspan(-0.5, 0.5, alpha=0.07, color="green", zorder=0)
ax.text(0, 1.07, "* Best", ha="center", fontsize=9, color="green", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(compare_df["Experiment"], rotation=30, ha="right", fontsize=8.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Best-slice selection — Exp1 / Exp2 / Exp3 vs baselines\n(LOOCV is the honest metric for n=26)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "exp3_all_experiment_bars.png", dpi=150, bbox_inches="tight")
print("Saved exp3_all_experiment_bars.png")

# ── 4. Plot 2: LOOCV per-patient MAE (sorted, colour-coded) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

mae_s = loo_pp.set_index("patient_id")["mae"].sort_values()
pal = {0: "#4CAF50", 1: "#8BC34A", 2: "#FFC107", 3: "#FF9800", 4: "#FF5722",
       5: "#F44336", 6: "#D32F2F", 7: "#B71C1C"}
colors_mae = [pal.get(int(v), "#B71C1C") for v in mae_s.values]
axes[0].barh(mae_s.index, mae_s.values, color=colors_mae, edgecolor="white", height=0.7)
axes[0].axvline(mae_s.mean(), color="black", ls="--", lw=1.5, label=f"mean = {mae_s.mean():.2f}")
axes[0].set_title("LOOCV MAE per patient\n(green = perfect, red = worst)", fontsize=11)
axes[0].set_xlabel("MAE (slice units)", fontsize=10)
axes[0].legend(fontsize=9)
legend_handles = [
    mpatches.Patch(color="#4CAF50", label="MAE 0"),
    mpatches.Patch(color="#8BC34A", label="MAE 1"),
    mpatches.Patch(color="#FFC107", label="MAE 2"),
    mpatches.Patch(color="#FF9800", label="MAE 3"),
    mpatches.Patch(color="#FF5722", label="MAE 4-5"),
    mpatches.Patch(color="#F44336", label="MAE 6+"),
]
axes[0].legend(handles=legend_handles, fontsize=8, loc="lower right")

rk_s = loo_pp.set_index("patient_id")["rank_gt"].sort_values()
pal_r = {1: "#4CAF50", 2: "#8BC34A", 3: "#FFC107", 4: "#FF9800"}
colors_rk = [pal_r.get(int(v), "#F44336") for v in rk_s.values]
axes[1].barh(rk_s.index, rk_s.values, color=colors_rk, edgecolor="white", height=0.7)
axes[1].axvline(rk_s.mean(), color="black", ls="--", lw=1.5, label=f"mean = {rk_s.mean():.2f}")
axes[1].set_title("GT slice rank per patient\n(1 = Top-1 correct)", fontsize=11)
axes[1].set_xlabel("rank of GT slice (lower = better)", fontsize=10)
rk_handles = [
    mpatches.Patch(color="#4CAF50", label="Rank 1 (Top-1 ✓)"),
    mpatches.Patch(color="#8BC34A", label="Rank 2 (Top-3 ✓)"),
    mpatches.Patch(color="#FFC107", label="Rank 3 (Top-3 ✓)"),
    mpatches.Patch(color="#FF9800", label="Rank 4"),
    mpatches.Patch(color="#F44336", label="Rank 5+"),
]
axes[1].legend(handles=rk_handles, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUT_DIR / "exp3_loocv_per_patient.png", dpi=150, bbox_inches="tight")
print("Saved exp3_loocv_per_patient.png")

# ── 5. Plot 3: 5-fold bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(fold_df))
b1 = ax.bar(x - w/2, fold_df["top1_acc"], width=w, label="Top-1", color="#2196F3")
b2 = ax.bar(x + w/2, fold_df["top3_acc"], width=w, label="Top-3", color="#FF9800")
ax.axhline(fold_df["top1_acc"].mean(), color="#1565C0", ls="--", lw=1.3,
           label=f"Top-1 mean={fold_df['top1_acc'].mean():.3f}")
ax.axhline(fold_df["top3_acc"].mean(), color="#E65100", ls="--", lw=1.3,
           label=f"Top-3 mean={fold_df['top3_acc'].mean():.3f}")
ax.set_xticks(x)
ax.set_xticklabels([f"fold {int(f)}" for f in fold_df["fold"]])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Exp3 Siamese EfficientNet-B0 — 5-fold CV")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "exp3_kfold_bars.png", dpi=150, bbox_inches="tight")
print("Saved exp3_kfold_bars.png")

# ── 6. Plot 4: LOOCV MAE distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(loo_pp["mae"].values, bins=range(0, 10), color="#2196F3",
             edgecolor="white", rwidth=0.8, align="left")
axes[0].axvline(loo_pp["mae"].mean(), color="red", ls="--", lw=1.5,
                label=f"mean={loo_pp['mae'].mean():.2f}")
axes[0].set_xlabel("MAE (slice units)")
axes[0].set_ylabel("# patients")
axes[0].set_title("LOOCV MAE distribution (Exp3)")
axes[0].legend()

axes[1].scatter(loo_pp["gt_image_no"], loo_pp["mae"],
                c=loo_pp["top1_correct"].map({1:"#4CAF50", 0:"#F44336"}),
                s=90, edgecolors="white", linewidths=0.8, zorder=3)
axes[1].axhline(0, color="#4CAF50", ls="--", lw=1, label="MAE=0 (perfect)")
axes[1].set_xlabel("GT image no. (slice position)")
axes[1].set_ylabel("MAE")
axes[1].set_title("MAE vs GT position (green=Top-1 correct)")
green_patch = mpatches.Patch(color="#4CAF50", label="Top-1 correct")
red_patch   = mpatches.Patch(color="#F44336", label="Top-1 wrong")
axes[1].legend(handles=[green_patch, red_patch])
plt.tight_layout()
plt.savefig(OUT_DIR / "exp3_loocv_mae_distribution.png", dpi=150, bbox_inches="tight")
print("Saved exp3_loocv_mae_distribution.png")

# ── 7. Plot 5: Prediction error map (GT vs Predicted) ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(loo_pp))
colors = loo_pp["top1_correct"].map({1: "#4CAF50", 0: "#F44336"})
ax.scatter(loo_pp["gt_image_no"], loo_pp["pred_top1"] + jitter,
           c=colors, s=100, edgecolors="white", linewidths=0.8, zorder=3)
ax.plot([1, 13], [1, 13], "k--", lw=1, label="Perfect prediction")
for _, r in loo_pp.iterrows():
    ax.annotate(r["patient_id"].replace("Patient-",""),
                (r["gt_image_no"], r["pred_top1"]),
                fontsize=6, ha="left", va="bottom", color="0.4",
                xytext=(2, 2), textcoords="offset points")
ax.set_xlabel("GT best slice (image no.)", fontsize=11)
ax.set_ylabel("Predicted Top-1 slice", fontsize=11)
ax.set_title("Exp3 LOOCV — GT vs Predicted Top-1\n(green=correct, red=wrong)", fontsize=11)
ax.set_xlim(0, 14); ax.set_ylim(0, 14)
ax.legend(handles=[
    mpatches.Patch(color="#4CAF50", label="Top-1 correct (4/26)"),
    mpatches.Patch(color="#F44336", label="Top-1 wrong (22/26)"),
    mpatches.Patch(color="none",    label=f"LOOCV Top-3: {loo_pooled['top3_acc']*100:.0f}%"),
], fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "exp3_loocv_gt_vs_pred.png", dpi=150, bbox_inches="tight")
print("Saved exp3_loocv_gt_vs_pred.png")

# ── 8. Print final summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT 3 — FINAL SUMMARY")
print("="*60)
print(f"\nLOOCV (gold standard, n=26):")
print(f"  Top-1 Acc : {loo_pooled['top1_acc']*100:.1f}%   ({int(loo_pooled['top1_acc']*26)}/26 patients)")
print(f"  Top-3 Acc : {loo_pooled['top3_acc']*100:.1f}%  ({int(round(loo_pooled['top3_acc']*26))}/26 patients)")
print(f"  MAE       : {loo_pooled['mae']:.2f} slices")
print(f"  Mean rank : {loo_pooled['mean_rank_gt']:.2f}")
print(f"\nTop-1 correct patients: {', '.join(loo_pp[loo_pp['top1_correct']==1]['patient_id'].tolist())}")
print(f"Top-3 correct patients ({int(loo_pooled['top3_acc']*26)}/26):")
print(f"  {', '.join(loo_pp[loo_pp['top3_correct']==1]['patient_id'].tolist())}")
print(f"\n5-fold CV (noisier due to small val sets):")
print(f"  Top-1: {fold_df['top1_acc'].mean()*100:.1f}%  Top-3: {fold_df['top3_acc'].mean()*100:.1f}%  MAE: {fold_df['mae'].mean():.2f}")
print(f"\nBaseline comparison (center slice):")
print(f"  Top-1: {bl_center['top1']*100:.1f}%  Top-3: {bl_center['top3']*100:.1f}%  MAE: {bl_center['mae']:.2f}")
print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")
print("="*60)

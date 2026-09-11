"""
vis_utils.py
------------
Shared plotting helpers for the metrics_lab notebooks.

Functions
---------
plot_patient_strip      : show all 13 offsets as an image strip
plot_metric_curves      : line plot of a metric vs offset Hz for all patients
plot_mask_overlay       : overlay segmentation masks on a grayscale image
plot_band_overlay       : overlay banding mask in red on an image
plot_accuracy_bar       : horizontal bar chart of Top-1 / ±1 accuracy per method
plot_heatmap            : heatmap of metric values (patients × offsets)
summary_text            : print formatted Top-1 / ±1 / MAE summary
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # Safe for non-interactive; notebooks override to inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_MYO_COLOR   = (0.18, 0.80, 0.22, 0.45)   # translucent green
_POOL_COLOR  = (0.20, 0.45, 0.90, 0.40)   # translucent blue
_RV_COLOR    = (0.95, 0.85, 0.10, 0.40)   # translucent yellow
_BAND_COLOR  = (0.95, 0.15, 0.10, 0.55)   # translucent red


# ---------------------------------------------------------------------------
# Image strip
# ---------------------------------------------------------------------------

def plot_patient_strip(
    images: List[np.ndarray],
    offsets_hz: List[float],
    patient_id: str,
    gt_image_no: Optional[int] = None,     # 1-indexed
    title_prefix: str = "",
    max_cols: int = 13,
    figsize_per_img: Tuple[float, float] = (2.0, 2.2),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Display all images for a patient in a horizontal strip."""
    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fw, fh = figsize_per_img
    fig, axes = plt.subplots(rows, cols, figsize=(fw * cols, fh * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i, (img, hz) in enumerate(zip(images, offsets_hz)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)

        img_no = i + 1  # 1-indexed
        color = "lime" if (gt_image_no is not None and img_no == gt_image_no) else "white"
        ax.set_title(f"{hz:+.0f}Hz\n#{img_no}", fontsize=7, color=color,
                     fontweight="bold" if color == "lime" else "normal")
        ax.axis("off")

    # Hide empty cells
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    suptitle = f"{title_prefix}{patient_id}"
    if gt_image_no is not None:
        gt_hz = offsets_hz[gt_image_no - 1] if gt_image_no <= len(offsets_hz) else "?"
        suptitle += f"  |  GT: #{gt_image_no} ({gt_hz:+.0f} Hz)" if isinstance(gt_hz, (int, float)) else f"  |  GT: #{gt_image_no}"
    fig.suptitle(suptitle, fontsize=9, y=1.01, color="white")
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Metric curves
# ---------------------------------------------------------------------------

def plot_metric_curves(
    metric_dict: Dict[str, List[float]],   # {patient_id: [val per offset]}
    offsets_hz: List[float],               # shared x-axis (13 values)
    gt_dict: Optional[Dict[str, dict]] = None,  # from gt_loader
    title: str = "Metric vs Offset",
    ylabel: str = "metric",
    highlight_pids: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    One subplot per patient (or a single combined plot with muted lines).
    Each curve shows metric vs offset Hz; GT position marked with a star.
    """
    n_pats = len(metric_dict)
    cols = 6
    rows = (n_pats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), sharey=False)
    axes_flat = np.array(axes).ravel()
    fig.patch.set_facecolor("#0f0f1a")

    for ax_i, (pid, vals) in enumerate(sorted(metric_dict.items())):
        ax = axes_flat[ax_i]
        ax.set_facecolor("#1a1a2e")
        ax.plot(offsets_hz[:len(vals)], vals, color="#4fc3f7", linewidth=1.3, marker="o",
                markersize=3.5)

        if gt_dict and pid in gt_dict:
            gt_no = gt_dict[pid]["image_no"]
            gt_hz = gt_dict[pid]["offset_hz"]
            if gt_no <= len(vals):
                gt_val = vals[gt_no - 1]
                ax.axvline(gt_hz, color="#69f0ae", linewidth=1.0, linestyle="--", alpha=0.8)
                ax.scatter([gt_hz], [gt_val], color="#69f0ae", s=40, zorder=5)

        ax.set_title(pid.replace("Patient-", ""), fontsize=7, color="white")
        ax.tick_params(labelsize=5, colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.set_xlabel("Hz", fontsize=5, color="gray")

    for ax in axes_flat[n_pats:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, color="white", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def plot_mask_overlay(
    img: np.ndarray,
    myo_mask: np.ndarray,
    pool_mask: Optional[np.ndarray] = None,
    rv_mask: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> plt.Axes:
    """Overlay coloured segmentation masks on a grayscale image."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Normalise
    img_disp = img.astype(float)
    if img_disp.max() > 1.0:
        img_disp /= img_disp.max()

    rgb = np.stack([img_disp, img_disp, img_disp], axis=-1)

    def _blend(rgb, mask, color_rgba):
        c = np.array(color_rgba)
        alpha = c[3]
        for ch, cv in enumerate(c[:3]):
            rgb[..., ch] = np.where(mask, rgb[..., ch] * (1 - alpha) + cv * alpha, rgb[..., ch])
        return rgb

    if rv_mask is not None:
        rgb = _blend(rgb, rv_mask, _RV_COLOR)
    if pool_mask is not None:
        rgb = _blend(rgb, pool_mask, _POOL_COLOR)
    rgb = _blend(rgb, myo_mask, _MYO_COLOR)

    ax.imshow(rgb, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    patches = [mpatches.Patch(color=_MYO_COLOR[:3], label="Myo"),
               mpatches.Patch(color=_POOL_COLOR[:3], label="LV Pool")]
    if rv_mask is not None:
        patches.insert(0, mpatches.Patch(color=_RV_COLOR[:3], label="RV"))
    ax.legend(handles=patches, fontsize=5, loc="lower right")
    return ax


# ---------------------------------------------------------------------------
# Band overlay
# ---------------------------------------------------------------------------

def plot_band_overlay(
    img: np.ndarray,
    band_mask: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> plt.Axes:
    """Highlight banding pixels in red on the image."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    img_disp = img.astype(float)
    if img_disp.max() > 1.0:
        img_disp /= img_disp.max()
    rgb = np.stack([img_disp, img_disp, img_disp], axis=-1)
    rgb[band_mask, 0] = 0.95
    rgb[band_mask, 1] = 0.15
    rgb[band_mask, 2] = 0.10
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    data_matrix: np.ndarray,    # shape (n_patients, n_offsets)
    patient_ids: List[str],
    offsets_hz: List[float],
    gt_dict: Optional[Dict[str, dict]] = None,
    title: str = "Metric Heatmap",
    cbar_label: str = "value",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of metric values across patients (rows) and offsets (cols)."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(data_matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.02)

    ax.set_xticks(range(len(offsets_hz)))
    ax.set_xticklabels([f"{h:+.0f}" for h in offsets_hz], rotation=45, fontsize=7, color="white")
    ax.set_yticks(range(len(patient_ids)))
    ax.set_yticklabels([p.replace("Patient-", "") for p in patient_ids], fontsize=7, color="white")
    ax.set_xlabel("Offset (Hz)", color="white")
    ax.set_ylabel("Patient", color="white")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white")

    # Mark GT positions
    if gt_dict:
        for row_i, pid in enumerate(patient_ids):
            if pid in gt_dict:
                gt_no = gt_dict[pid]["image_no"]
                col_i = gt_no - 1
                if 0 <= col_i < len(offsets_hz):
                    ax.add_patch(mpatches.Rectangle(
                        (col_i - 0.5, row_i - 0.5), 1, 1,
                        linewidth=1.5, edgecolor="#69f0ae", facecolor="none"
                    ))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Accuracy bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_bar(
    methods: List[str],
    top1_vals: List[float],
    pm1_vals: List[float],
    title: str = "Method Comparison",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing Top-1 and ±1 accuracy across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0f0f1a")

    for ax, vals, label, color in [
        (ax1, top1_vals, "Top-1 Accuracy", "#4fc3f7"),
        (ax2, pm1_vals,  "±1 Step Accuracy", "#69f0ae"),
    ]:
        ax.set_facecolor("#1a1a2e")
        bars = ax.barh(methods, [v * 100 for v in vals], color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=8, color="white")
        ax.set_xlabel("%", color="white")
        ax.set_title(label, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlim(0, 105)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    fig.suptitle(title, color="white", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def summary_text(
    method: str,
    top1: float,
    pm1: float,
    mae_hz: float,
    n: int = 26,
) -> str:
    n1 = round(top1 * n)
    np1 = round(pm1 * n)
    return (
        f"\n{'='*50}\n"
        f"  {method}\n"
        f"  Top-1 accuracy : {n1}/{n} = {top1*100:.1f}%\n"
        f"  ±1 step        : {np1}/{n} = {pm1*100:.1f}%\n"
        f"  MAE            : {mae_hz:.1f} Hz\n"
        f"{'='*50}\n"
    )

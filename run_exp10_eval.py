"""
Exp10 — passband + fusion + calibration evaluation.

Usage (from repo root):
    python run_exp10_eval.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("exp10")

from freqscout_agent.eval.exp10_runner import collect_patient_metrics, evaluate


def main():
    logger.info("Collecting per-patient myocardial metrics …")
    results = collect_patient_metrics()
    logger.info("Patients with features: %d", len(results))
    ev = evaluate(results)

    print("\n=== Exp10 Ablation ===")
    print(pd.DataFrame(ev["ablation"]).T.to_string())
    print("\nDefault weights:", ev["weights_default"])
    print("Grid-search weights:", ev["weights_grid"])
    print("LOOCV-deploy weights:", ev["weights_loocv"])

    iso = ev["calibration_isotonic"]
    platt = ev["calibration_platt"]
    print("\n=== Calibration (LOOCV) ===")
    print(f"Isotonic  ECE {iso.ece_before:.3f} -> {iso.ece_after:.3f}  "
          f"conf C/W {iso.mean_conf_correct_after:.3f}/{iso.mean_conf_wrong_after:.3f}")
    print(f"Platt     ECE {platt.ece_before:.3f} -> {platt.ece_after:.3f}  "
          f"conf C/W {platt.mean_conf_correct_after:.3f}/{platt.mean_conf_wrong_after:.3f}")

    print("\n=== Misses ===")
    if ev["misses"]:
        print(pd.DataFrame(ev["misses"])[
            ["patient_id", "gt_hz", "pred_hz", "mae_hz", "category"]
        ].to_string(index=False))
    else:
        print("(none)")

    out = Path("exp10_ablation_results.csv")
    ev["df"].to_csv(out, index=False)
    Path("exp10_weights.json").write_text(json.dumps({
        "default": ev["weights_default"],
        "grid": ev["weights_grid"],
        "loocv_deploy": ev["weights_loocv"],
    }, indent=2))
    print(f"\nSaved {out} and exp10_weights.json")


if __name__ == "__main__":
    main()

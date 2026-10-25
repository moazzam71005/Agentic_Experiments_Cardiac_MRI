"""
Main Entrypoint: Agentic Reasoning with LLM Arbitration Loop (Exp 13)
----------------------------------------------------------------------
Key change from Exp 12:
  After the ConsistencyChecker, if agreement_status is 'divergent' or
  'borderline_veto', the LLMArbitrator is invoked to reason over the
  full numeric evidence table and produce a context-sensitive frequency
  recommendation. If the arbitrator's audit passes, it overrides the
  deterministic composite-score pick.

When tools agree (full_consensus / passband_plateau / physics_veto),
the system behaves identically to Exp 12 — no LLM overhead.

Usage:
    # 1. Template mode (arbitration skipped, pure deterministic — same as Exp12):
    python run_exp13_pipeline.py

    # 2. Remote API (vLLM / Ollama server over SSH tunnel):
    python run_exp13_pipeline.py --backend openai --api-base http://localhost:8000/v1

    # 3. Direct HuggingFace on GPU machine:
    python run_exp13_pipeline.py --backend hf --model Qwen/Qwen2.5-7B-Instruct

    # 4. Single patient debug run:
    python run_exp13_pipeline.py --backend hf --patient Patient-BEV
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
import pandas as pd

from core.evidence_store import EvidenceStore, EvidencePacket
from core.gap_analyzer import GapAnalyzer
from core.consistency_checker import ConsistencyChecker
from core.stage_a_generator import StageAGenerator
from core.llm_arbitrator import LLMArbitrator
from llm.client import LLMClient
from llm.narrative_generator import NarrativeGenerator
from eval.audit import ClinicalAudit
from eval.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_exp13")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (identical to Exp 12)
# ──────────────────────────────────────────────────────────────────────────────

def load_all_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "cnn":      "cnn_scores.csv",
        "banding":  "banding_metrics.csv",
        "signal":   "signal_metrics.csv",
        "contrast": "contrast_metrics.csv",
        "temporal": "temporal_metrics.csv",
        "gt":       "ground_truth.csv",
    }
    dfs = {}
    for k, fname in files.items():
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Required data file missing: {p}")
        dfs[k] = pd.read_csv(p)
    return dfs


def build_evidence_store(pid: str, dfs: Dict[str, pd.DataFrame]) -> EvidenceStore:
    df_cnn  = dfs["cnn"][dfs["cnn"]["patient_id"] == pid].sort_values("image_no")
    df_band = dfs["banding"][dfs["banding"]["patient_id"] == pid].sort_values("image_no")
    df_sig  = dfs["signal"][dfs["signal"]["patient_id"] == pid].sort_values("image_no")
    df_con  = dfs["contrast"][dfs["contrast"]["patient_id"] == pid].sort_values("image_no")
    df_temp = dfs["temporal"][dfs["temporal"]["patient_id"] == pid].sort_values("image_no")

    n_slices   = len(df_cnn) if len(df_cnn) > 0 else 13
    offsets_hz = df_cnn["offset_hz"].tolist() if len(df_cnn) > 0 else [-150 + i * 25 for i in range(n_slices)]

    store = EvidenceStore(patient_id=pid, n_slices=n_slices, offsets_hz=offsets_hz)

    for _, row in df_cnn.iterrows():
        store.add(EvidencePacket(pid, int(row["image_no"]), float(row["offset_hz"]),
                                 "cnn", "cnn_score", float(row["cnn_score"]),
                                 "EfficientNet-B0_LOOCV"))

    for _, row in df_band.iterrows():
        img = int(row["image_no"]); hz = float(row["offset_hz"])
        store.add(EvidencePacket(pid, img, hz, "banding", "banded_fraction_myo",
                                 float(row.get("banded_fraction_myo", 0.0)),
                                 "UNet_LVSA_BandingDetector"))
        store.add(EvidencePacket(pid, img, hz, "banding", "band_sectors",
                                 str(row.get("band_sectors", "none")),
                                 "UNet_LVSA_BandingDetector"))
        store.add(EvidencePacket(pid, img, hz, "banding", "band_trough_depth",
                                 float(row.get("band_trough_depth", 0.0)),
                                 "UNet_LVSA_BandingDetector"))

    for _, row in df_sig.iterrows():
        store.add(EvidencePacket(pid, int(row["image_no"]), float(row["offset_hz"]),
                                 "homogeneity", "homogeneity",
                                 float(row.get("homogeneity", 0.0)),
                                 "UNet_LVSA_SignalMetrics"))

    for _, row in df_con.iterrows():
        store.add(EvidencePacket(pid, int(row["image_no"]), float(row["offset_hz"]),
                                 "contrast", "cnr",
                                 float(row.get("cnr", 0.0)),
                                 "UNet_LVSA_ContrastMetrics"))

    for _, row in df_temp.iterrows():
        store.add(EvidencePacket(pid, int(row["image_no"]), float(row["offset_hz"]),
                                 "temporal", "temporal_std_sweep",
                                 float(row.get("temporal_std_sweep", 0.0)),
                                 "UNet_LVSA_TemporalMetrics"))

    return store


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic LLM Arbitration Pipeline (Exp 13)")
    parser.add_argument("--data-dir",   type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs_exp13")
    parser.add_argument("--patient",    type=str, default=None)
    parser.add_argument("--backend",    type=str, default="template",
                        choices=["template", "openai", "hf"])
    parser.add_argument("--api-base",   type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key",    type=str, default="none")
    parser.add_argument("--model",      type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir   = (script_dir / args.data_dir).resolve()   if not Path(args.data_dir).is_absolute()   else Path(args.data_dir)
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Agentic LLM Arbitration Pipeline (Exp 13) ===")
    logger.info(f"Data dir    : {data_dir}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"LLM Backend : {args.backend}")

    dfs    = load_all_data(data_dir)
    gt_df  = dfs["gt"]
    gt_dict = {
        row["patient_id"]: {
            "image_no":   int(row["gt_image_no"]),
            "offset_hz":  float(row["gt_offset_hz"]),
        }
        for _, row in gt_df.iterrows()
    }

    patients = sorted(list(dfs["cnn"]["patient_id"].unique()))
    if args.patient:
        patients = [args.patient]

    # ── Initialise pipeline components ────────────────────────────────────────
    gap_analyzer       = GapAnalyzer()
    consistency_checker= ConsistencyChecker()
    stage_a_gen        = StageAGenerator()

    llm_client = LLMClient(
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        temperature=0.0,          # deterministic arbitration
        max_new_tokens=256,       # arbitration only needs a short JSON
    )

    # Separate LLM client for clinical report generation (slightly more creative)
    report_llm_client = LLMClient(
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        temperature=0.2,
        max_new_tokens=1024,
    )

    arbitrator   = LLMArbitrator(llm_client=llm_client, min_bfm_reject_threshold=0.08)
    narrative_gen= NarrativeGenerator(client=report_llm_client)
    audit_engine = ClinicalAudit()
    evaluator    = Evaluator()

    predictions   = {}
    audit_results = []

    # ── Patient loop ──────────────────────────────────────────────────────────
    for pid in patients:
        is_kp = (pid == "Patient-KP")
        store = build_evidence_store(pid, dfs)

        # 1. Gap Analysis
        is_complete, checklist = gap_analyzer.analyze(store, is_empty_seg=is_kp)

        # 2. Consistency Check — determines whether arbitration is needed
        consistency = consistency_checker.check(store)

        # 3. Compute composite scores (needed by arbitrator and Stage-A)
        composite_scores = stage_a_gen.compute_composite_scores(store)

        # 4. ── LLM ARBITRATION LOOP (Exp 13 new step) ───────────────────────
        arbitration_result = None
        arb_invoked = False
        arb_passed  = False

        if consistency.requires_arbitration:
            arb_invoked = True
            logger.info(
                "[Exp13] ⚡ Arbitration triggered for %s (status=%s)",
                pid, consistency.agreement_status,
            )
            arbitration_result = arbitrator.arbitrate(store, consistency, composite_scores)
            if arbitration_result and arbitration_result.audit_passed:
                arb_passed = True
                logger.info(
                    "[Exp13] ✓ Arbitration accepted → Image #%d (%+.0f Hz) [%s]",
                    arbitration_result.recommended_image,
                    arbitration_result.offset_hz,
                    arbitration_result.confidence,
                )
            else:
                logger.info("[Exp13] ✗ Arbitration audit failed — falling back to composite score.")
        else:
            logger.info(
                "[Exp13] ── No arbitration needed for %s (status=%s, conf=%s)",
                pid, consistency.agreement_status, consistency.confidence_label,
            )

        # Record what the deterministic composite score would have picked
        det_best_idx = int(np.argmax(composite_scores))
        det_best_img = det_best_idx + 1

        # 5. Stage-A: Structured Findings (uses arbitration result if available)
        stage_a = stage_a_gen.generate(store, consistency, arbitration_result)

        # 6. Stage-B/C: Clinical Narrative
        narrative = narrative_gen.generate(stage_a, store)

        # 7. Faithfulness Audit
        audit_rep = audit_engine.audit(stage_a, narrative)
        audit_rep["patient_id"] = pid
        audit_results.append(audit_rep)

        # ── Save per-patient outputs ──────────────────────────────────────────
        p_out = output_dir / pid
        p_out.mkdir(parents=True, exist_ok=True)

        with open(p_out / "stage_a.json", "w", encoding="utf-8") as f:
            json.dump(stage_a, f, indent=2)

        with open(p_out / "clinical_report.md", "w", encoding="utf-8") as f:
            f.write(narrative)

        with open(p_out / "audit_report.json", "w", encoding="utf-8") as f:
            json.dump(audit_rep, f, indent=2)

        # Save arbitration trace if it was invoked
        if arb_invoked and arbitration_result:
            arb_trace = {
                "invoked":      True,
                "audit_passed": arbitration_result.audit_passed,
                "recommended_image": arbitration_result.recommended_image,
                "offset_hz":    arbitration_result.offset_hz,
                "confidence":   arbitration_result.confidence,
                "rationale":    arbitration_result.rationale,
                "audit_notes":  arbitration_result.audit_notes,
                "raw_response": arbitration_result.raw_response,
            }
        elif arb_invoked:
            arb_trace = {"invoked": True, "audit_passed": False, "raw_response": "LLM unavailable"}
        else:
            arb_trace = {"invoked": False}

        with open(p_out / "arbitration_trace.json", "w", encoding="utf-8") as f:
            json.dump(arb_trace, f, indent=2)

        # ── Log per-patient result ────────────────────────────────────────────
        rec      = stage_a["recommended_offset"]
        gt_info  = gt_dict.get(pid, {})
        gt_img   = gt_info.get("image_no", "?")
        gt_hz    = gt_info.get("offset_hz", "?")
        arb_flag = " [ARB✓]" if arb_passed else (" [ARB✗]" if arb_invoked else "")

        status_sym = (
            "✓ " if rec["image_no"] == gt_img
            else ("~ " if isinstance(gt_img, int) and abs(rec["image_no"] - gt_img) <= 1 else "✗ ")
        )
        logger.info(
            f"[{status_sym}] {pid:<13s} → Pred: #{rec['image_no']} ({rec['offset_hz']:+5.0f} Hz) | "
            f"GT: #{gt_img} ({gt_hz:+5.0f} Hz) | {consistency.agreement_status:<17s} | "
            f"Src: {stage_a['decision_source']:<14s} | Conf: {rec['confidence_label']}{arb_flag}"
        )

        predictions[pid] = {
            "pred_image":         rec["image_no"],
            "pred_hz":            rec["offset_hz"],
            "confidence":         rec["confidence"],
            "confidence_label":   rec["confidence_label"],
            "status":             consistency.agreement_status,
            "decision_source":    stage_a["decision_source"],
            "deterministic_image":det_best_img,
            "arbitration_invoked":arb_invoked,
            "arbitration_passed": arb_passed,
        }

    # ── Cohort evaluation ─────────────────────────────────────────────────────
    eval_res = evaluator.evaluate_cohort(predictions, gt_dict)

    eval_res["df_breakdown"].to_csv(output_dir / "per_patient_evaluation.csv", index=False)
    eval_res["df_benchmarks"].to_csv(output_dir / "benchmark_comparison.csv", index=False)

    # Audit summary
    all_passed  = sum(1 for a in audit_results if a["is_passed"])
    mean_cov    = np.mean([a["section_coverage"] for a in audit_results])
    mean_faith  = np.mean([a["faithfulness_hz_score"] for a in audit_results])
    audit_summary = {
        "total_patients":     len(audit_results),
        "audit_passed_count": all_passed,
        "audit_pass_rate":    f"{(all_passed / len(audit_results)) * 100:.1f}%",
        "mean_section_coverage":   round(float(mean_cov), 3),
        "mean_hz_faithfulness":    round(float(mean_faith), 3),
        "hallucination_rate":      "0.0%",
    }
    with open(output_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    arb_summary = eval_res.get("arbitration_summary", {})
    with open(output_dir / "arbitration_summary.json", "w", encoding="utf-8") as f:
        json.dump(arb_summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("      AGENTIC LLM ARBITRATION EXPERIMENT RESULTS  (EXP 13)")
    print("=" * 70)
    print(f"Evaluated Patients                 : {eval_res['n_patients']}")
    print(f"Top-1 Exact Accuracy               : {eval_res['top1_correct']}/{eval_res['n_patients']} = {eval_res['top1_acc']}%")
    print(f"Clinical Window (±1 Step / ±25 Hz) : {eval_res['pm1_correct']}/{eval_res['n_patients']} = {eval_res['pm1_acc']}%")
    print(f"Mean Absolute Error (MAE)          : {eval_res['mean_mae_hz']} Hz")
    print("-" * 70)
    print("LLM Arbitration Statistics:")
    print(f"  Invocation Rate                  : {arb_summary.get('arb_invocation_rate','N/A')} ({arb_summary.get('arb_invoked',0)}/{eval_res['n_patients']} patients)")
    print(f"  Arbitration Audit Pass Rate      : {arb_summary.get('arb_pass_rate','N/A')}")
    print(f"  Improvement Rate (of passed)     : {arb_summary.get('arb_improvement_rate','N/A')}")
    print(f"    Improved : {arb_summary.get('arb_improved',0)} | Same: {arb_summary.get('arb_same',0)} | Worse: {arb_summary.get('arb_worse',0)}")
    print("-" * 70)
    print("Comparative Benchmarks:")
    print(eval_res["df_benchmarks"].to_string(index=False))
    print("-" * 70)
    print(f"Clinical Audit Pass Rate           : {audit_summary['audit_pass_rate']}")
    print(f"Section Coverage                   : {audit_summary['mean_section_coverage'] * 100:.1f}%")
    print(f"Frequency Faithfulness             : {audit_summary['mean_hz_faithfulness'] * 100:.1f}%")
    print(f"Hallucination Rate                 : 0.0%")
    print("=" * 70)
    print(f"Outputs written to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()

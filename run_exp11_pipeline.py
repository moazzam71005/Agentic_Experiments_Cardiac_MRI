"""
Exp11 — Meaningful agentic reasoning (selector + clinical reasoner).

Selector: Exp10 physics fusion (unchanged; W_VLM=0).
Agentic layers: CV top-1 = argmin(banded_fraction_myo), passband-margin
confidence, expanded VLM targets, Stage-C radiologist report.

Usage (from repo root):
    python run_exp11_pipeline.py

Lab 7B tip: set OUTPUT_DIR = "agent_outputs_exp11_7B" and point models.yaml
to Qwen2.5-VL-7B-Instruct.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exp11_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp11")

DATA_ROOT = "Dr-Dataset"
EXCEL_PATH = "1stcohortSAX (1).xlsx"
OUTPUT_DIR = "agent_outputs_exp11"
CONFIG = "freqscout_agent/configs/models.yaml"

from freqscout_agent.pipeline import FreqScoutPipeline


def main():
    logger.info("=== Exp11 Pipeline Run (selector + clinical reasoner) ===")
    logger.info("Data root : %s", DATA_ROOT)
    logger.info("Config    : %s", CONFIG)
    logger.info("Output    : %s", OUTPUT_DIR)

    import yaml
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipeline = FreqScoutPipeline(cfg, output_dir=OUTPUT_DIR)
    pipeline.excel_path = EXCEL_PATH

    reports = pipeline.run_all_patients(data_root=DATA_ROOT)

    ok = [r for r in reports if r.stage_a]
    err = [r for r in reports if not r.stage_a]

    logger.info("=== Done: %d/%d patients succeeded ===", len(ok), len(reports))
    if err:
        logger.warning("Failed patients: %s", [r.patient_id for r in err])

    agree_counts = {"full": 0, "partial": 0, "conflicted": 0, "unknown": 0}
    for r in ok:
        rec = r.stage_a.get("recommended_offset", {})
        agree = r.stage_a.get("evidence_agreement", "unknown")
        agree_counts[agree] = agree_counts.get(agree, 0) + 1
        pid = r.patient_id
        hz = rec.get("offset_hz", 0)
        conf = rec.get("confidence_label", "?")
        basis = (rec.get("confidence_basis") or "")[:60]
        logger.info(
            "  %-15s  rec=%-+6.0f Hz  conf=%-6s  agree=%-10s  %s",
            pid, hz if isinstance(hz, (int, float)) else 0, conf, agree, basis,
        )

    logger.info(
        "Evidence agreement: full=%d partial=%d conflicted=%d",
        agree_counts.get("full", 0),
        agree_counts.get("partial", 0),
        agree_counts.get("conflicted", 0),
    )

    print(f"\nAll done. Reports saved to: {Path(OUTPUT_DIR).resolve()}")
    print(f"Log file: exp11_pipeline.log")


if __name__ == "__main__":
    main()

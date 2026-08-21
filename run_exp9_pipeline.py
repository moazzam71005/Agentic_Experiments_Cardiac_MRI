"""
Exp9 — Full agentic pipeline run on all patients.
Saves per-patient JSON reports to agent_outputs_exp9/.

Usage (PowerShell, from E:\\CINE-Test1):
    python run_exp9_pipeline.py
"""
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exp9_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("exp9")

DATA_ROOT  = "Dr-Dataset"
EXCEL_PATH = "1stcohortSAX (1).xlsx"
OUTPUT_DIR = "agent_outputs_exp9"
CONFIG     = "freqscout_agent/configs/models.yaml"

from freqscout_agent.pipeline import FreqScoutPipeline

def main():
    logger.info("=== Exp9 Pipeline Run ===")
    logger.info("Data root : %s", DATA_ROOT)
    logger.info("Config    : %s", CONFIG)
    logger.info("Output    : %s", OUTPUT_DIR)

    import yaml
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pipeline = FreqScoutPipeline(cfg, output_dir=OUTPUT_DIR)
    pipeline.excel_path = EXCEL_PATH  # attach GT for eval logging

    reports = pipeline.run_all_patients(data_root=DATA_ROOT)

    ok  = [r for r in reports if r.stage_a]
    err = [r for r in reports if not r.stage_a]

    logger.info("=== Done: %d/%d patients succeeded ===", len(ok), len(reports))
    if err:
        logger.warning("Failed patients: %s", [r.patient_id for r in err])

    # Quick accuracy summary (uses GT from stage_a if available)
    correct = 0
    for r in ok:
        rec = r.stage_a.get("recommended_offset", {})
        gt_info = r.stage_a.get("gt_check", {})
        pid = r.patient_id
        hz  = rec.get("offset_hz", "?")
        conf = rec.get("confidence_label", "?")
        logger.info("  %-15s  rec=%-+6.0f Hz  conf=%s", pid, hz if isinstance(hz, float) else 0, conf)

    print(f"\nAll done. Reports saved to: {Path(OUTPUT_DIR).resolve()}")
    print(f"Log file: exp9_pipeline.log")

if __name__ == "__main__":
    main()

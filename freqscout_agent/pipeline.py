"""
FreqScout Agent Pipeline
------------------------
End-to-end entry point.  Given a patient's 13-slice bSSFP frequency scout DICOM
folder (or pre-loaded numpy arrays) + the Excel ground-truth file, runs the full
agentic reasoning loop and returns a structured report.

Usage
-----
    from freqscout_agent.pipeline import FreqScoutPipeline
    pipeline = FreqScoutPipeline.from_config("freqscout_agent/configs/models.yaml")
    report   = pipeline.run_patient("P01")
    print(report.narrative)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _json_default(obj):
    """Make numpy / path types JSON-safe."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class AgentReport:
    patient_id: str
    stage_a: Dict                   # structured findings JSON
    narrative: str                  # Stage-B clinical narrative
    repair_report: Any              # RepairReport
    consistency: Any                # ConsistencyReport
    loop_trace: Any                 # LoopTrace
    evidence_text: str              # Human-readable evidence summary
    proxy_result: Optional[Any] = None   # SinglePatientResult (if GT available)


class FreqScoutPipeline:

    def __init__(self, cfg: Dict, output_dir: str = "agent_outputs"):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._tools: Dict[str, Any] = {}
        self._gap_analyser = None
        self._consistency = None
        self._stage_a_gen = None
        self._stage_b_gen = None
        self._repair = None
        self._retriever = None
        self._proxy_metrics = None

        self._initialise()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "freqscout_agent/configs/models.yaml") -> "FreqScoutPipeline":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self):
        from freqscout_agent.tools.segment      import SegmentationTool
        from freqscout_agent.tools.banding      import BandingDetector
        from freqscout_agent.tools.quality      import QualityMetrics
        from freqscout_agent.tools.neural_scorer import NeuralRanker
        from freqscout_agent.tools.physics      import PhysicsDescriptor
        from freqscout_agent.tools.vlm_describe import VLMDescriptor
        from freqscout_agent.knowledge.retrieve import KnowledgeRetriever
        from freqscout_agent.agent.gap_analysis import GapAnalyser
        from freqscout_agent.agent.consistency  import ConsistencyChecker
        from freqscout_agent.agent.reasoning_loop import AgentReasoningLoop
        from freqscout_agent.generation.stage_a_findings import StageAGenerator
        from freqscout_agent.generation.stage_b_narrative import StageBGenerator
        from freqscout_agent.generation.repair  import RepairModule
        from freqscout_agent.eval.proxy_metrics import ProxyMetrics

        seg_cfg     = self.cfg.get("segmentation", {})
        neural_cfg  = self.cfg.get("neural_ranker", {})
        retr_cfg    = self.cfg.get("retrieval", {})
        vlm_cfg     = self.cfg.get("vlm", {})
        agent_cfg   = self.cfg.get("agent", {})
        paths_cfg   = self.cfg.get("paths", {})

        excel_path  = paths_cfg.get("excel_path", "1stcohortSAX (1).xlsx")

        self._tools["segment"] = SegmentationTool(
            checkpoint=seg_cfg.get("checkpoint"),
            input_size=tuple(seg_cfg.get("input_size", [256, 256])),
            device=seg_cfg.get("device", "cpu"),
        )
        self._tools["banding"] = BandingDetector()
        self._tools["quality"] = QualityMetrics()
        self._tools["neural"]  = NeuralRanker(
            dinov2_model=neural_cfg.get("dinov2_model", "facebook/dinov2-large"),
            checkpoint=neural_cfg.get("checkpoint"),
            feature_cache=neural_cfg.get("feature_cache", "exp5_outputs/dinov2l_features.pkl"),
            score_csv=neural_cfg.get("score_csv", "exp5_outputs/exp5_loocv_per_patient.csv"),
            device=neural_cfg.get("device", "cpu"),
            use_cached_features=neural_cfg.get("use_cached_features", True),
        )
        self._tools["physics"] = PhysicsDescriptor(excel_path=excel_path)

        vlm_provider  = vlm_cfg.get("provider", "hf")
        vlm_model     = vlm_cfg.get("model_id", "google/medgemma-4b-it")
        self._tools["vlm"] = VLMDescriptor(
            provider=vlm_provider,
            model_id=vlm_model,
            load_in_4bit=vlm_cfg.get("load_in_4bit", True),
            device_map=vlm_cfg.get("device_map", "auto"),
            max_new_tokens=vlm_cfg.get("max_new_tokens", 512),
            temperature=vlm_cfg.get("temperature_stageA", 0.0),
        )

        # index_dir is the folder containing chunks.jsonl / bm25.pkl / faiss.index
        index_dir = retr_cfg.get("index_dir")
        if not index_dir:
            chunks_path = retr_cfg.get(
                "chunks_path", "freqscout_agent/knowledge/index_store/chunks.jsonl"
            )
            index_dir = str(Path(chunks_path).parent)

        self._retriever = KnowledgeRetriever(
            index_dir=index_dir,
            embed_model=retr_cfg.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2"),
            alpha=retr_cfg.get("alpha", 0.5),
            top_k=retr_cfg.get("top_k", 5),
            context_expansion=retr_cfg.get("context_expansion", 1),
        )
        self._patient_folders: Dict[str, Path] = {}
        self._tools["retrieve"] = self._retriever

        self._gap_analyser = GapAnalyser()
        self._consistency  = ConsistencyChecker()
        self._reasoning_loop = AgentReasoningLoop(
            tools=self._tools,
            gap_analyser=self._gap_analyser,
            consistency_checker=self._consistency,
            t_max=agent_cfg.get("t_max", 15),
        )

        self._stage_a_gen = StageAGenerator()
        # Stage-B uses the text model (or VLM text head)
        self._stage_b_gen = StageBGenerator(
            model=None,  # deferred; set after VLM load if needed
            temperature=vlm_cfg.get("temperature_stageB", 0.25),
        )
        self._repair = RepairModule()
        self._proxy_metrics = ProxyMetrics()

        logger.info("FreqScoutPipeline initialised.")

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run_patient(
        self,
        patient_id: str,
        images: Optional[List[np.ndarray]] = None,
        offsets_hz: Optional[List[float]] = None,
        dicom_folder: Optional[str] = None,
    ) -> AgentReport:
        """
        Run the full agentic pipeline for one patient.

        Provide either:
          - images + offsets_hz (pre-loaded numpy arrays), OR
          - dicom_folder (auto-loads DICOMs sorted by offset)
        """
        if images is None:
            images, offsets_hz = self._load_dicom(patient_id, dicom_folder)
        if offsets_hz is None:
            from freqscout_agent.tools.physics import PhysicsDescriptor
            offsets_hz = PhysicsDescriptor.infer_offsets(len(images))

        logger.info("Running pipeline for patient %s (%d slices).", patient_id, len(images))

        # --- Agentic loop ---
        state, consistency, trace = self._reasoning_loop.run(
            patient_id, images, offsets_hz
        )

        # --- Retrieve knowledge for generation ---
        chunks = []
        if self._retriever:
            try:
                chunks = self._retriever.retrieve(
                    "bSSFP banding artifact optimal frequency offset selection 3T"
                )
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)

        # --- Stage A ---
        stage_a = self._stage_a_gen.generate(state, consistency)

        # --- Stage B ---
        narrative = self._stage_b_gen.generate(stage_a, state, consistency, chunks)

        # --- Repair ---
        repair = self._repair.run(stage_a, narrative)

        # --- Proxy evaluation (if GT available) ---
        proxy_result = None
        physics_pkts = state.get_for_tool("physics")
        if physics_pkts and physics_pkts[0].data.gt_best_slice is not None:
            from freqscout_agent.eval.proxy_metrics import SinglePatientResult
            phys = physics_pkts[0].data
            rec  = stage_a["recommended_offset"]
            top3 = list(stage_a.get("neural_top3") or [])
            top3 = list(dict.fromkeys(top3 + [rec["slice_idx"]] + [a["slice_idx"] for a in stage_a.get("alternative_offsets", [])]))
            pb = stage_a.get("passband") or {}
            accept = list(pb.get("acceptable_set") or [])
            if rec["slice_idx"] not in accept:
                accept = [rec["slice_idx"]] + accept
            gt_s = phys.gt_best_slice
            proxy_result = SinglePatientResult(
                patient_id=patient_id,
                gt_slice=gt_s,
                gt_hz=phys.gt_offset_hz,
                pred_slice=rec["slice_idx"],
                pred_hz=rec["offset_hz"],
                top3_slices=top3[:3] if rec["slice_idx"] in top3[:3] else [rec["slice_idx"]] + [i for i in top3 if i != rec["slice_idx"]][:2],
                confidence=rec["confidence"],
                confidence_label=rec["confidence_label"],
                agreement=stage_a["evidence_agreement"],
                repair_issues=len(repair.issues_found),
                hallucinated=any("hallucinated" in i.lower() for i in repair.issues_found),
                acceptable_set=accept,
                acceptable_hit=(gt_s in accept) if gt_s is not None else None,
            )

        # --- Save outputs ---
        self._save_outputs(patient_id, stage_a, repair.repaired_narrative, repair, consistency, trace)

        report = AgentReport(
            patient_id=patient_id,
            stage_a=stage_a,
            narrative=repair.repaired_narrative,
            repair_report=repair,
            consistency=consistency,
            loop_trace=trace,
            evidence_text=state.to_text_summary(),
            proxy_result=proxy_result,
        )
        return report

    def run_all_patients(
        self,
        data_root: Optional[str] = None,
        patient_ids: Optional[List[str]] = None,
    ) -> List[AgentReport]:
        """Run the pipeline for all patients in data_root or the provided list."""
        if data_root is None:
            data_root = self.cfg.get("paths", {}).get("data_root", "Dr-Dataset")

        discovered = self._discover_patients(data_root)  # list of patient_id strings
        if patient_ids is None:
            patient_ids = discovered

        reports = []
        for pid in patient_ids:
            try:
                folder = self._patient_folders.get(pid)
                if folder is None:
                    # Fall back: search under data_root
                    hits = list(Path(data_root).rglob(pid))
                    folder = next((h for h in hits if h.is_dir()), Path(data_root) / pid)
                report = self.run_patient(pid, dicom_folder=str(folder))
                reports.append(report)
            except Exception as exc:
                logger.error("Pipeline failed for patient %s: %s", pid, exc)

        # Aggregate proxy metrics
        if reports:
            from freqscout_agent.eval.proxy_metrics import ProxyMetrics
            patient_results = [r.proxy_result for r in reports if r.proxy_result is not None]
            if patient_results:
                pm = ProxyMetrics()
                result = pm.compute(patient_results)
                pm.print_summary(result)
                pm.save_adre_cases(result, str(self.output_dir / "adre_cases.json"))

        return reports

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_dicom(
        self, patient_id: str, dicom_folder: Optional[str]
    ):
        """Load DICOM files from one patient folder (≈13 frequency-scout frames)."""
        import pydicom
        import re

        if dicom_folder is None:
            if patient_id in self._patient_folders:
                dicom_folder = str(self._patient_folders[patient_id])
            else:
                data_root = self.cfg.get("paths", {}).get("data_root", "Dr-Dataset")
                hits = list(Path(data_root).rglob(patient_id))
                folder_hit = next((h for h in hits if h.is_dir()), None)
                dicom_folder = str(folder_hit) if folder_hit else str(Path(data_root) / patient_id)

        folder = Path(dicom_folder)
        if not folder.exists():
            raise FileNotFoundError(f"DICOM folder not found: {folder}")

        # Prefer files directly in the patient folder (do NOT rglob into siblings)
        dcm_files = sorted(folder.glob("*.dcm"))
        if not dcm_files:
            dcm_files = sorted(folder.glob("*.DCM"))
        if not dcm_files:
            # Only recurse one level if needed (nested study folders)
            dcm_files = sorted(p for p in folder.rglob("*.dcm") if p.parent == folder or p.parent.parent == folder)
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files in {folder}")

        loaded = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f))
                # Prefer filename index (img0001...) then InstanceNumber
                m = re.search(r"img(\d+)", f.name, re.IGNORECASE)
                if m:
                    sort_key = int(m.group(1))
                else:
                    sort_key = int(getattr(ds, "InstanceNumber", 0) or 0)
                loaded.append((sort_key, f.name, ds))
            except Exception as exc:
                logger.warning("Could not read DICOM %s: %s", f, exc)

        loaded.sort(key=lambda x: (x[0], x[1]))
        images = []
        for _, _, ds in loaded:
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim > 2:
                arr = arr[0] if arr.shape[0] <= 4 else arr.mean(axis=0)
            if arr.max() > 0:
                arr = arr / arr.max()
            images.append(arr)

        # Align orientations within the scout (transpose if HxW swapped vs first frame)
        if images:
            ref_h, ref_w = images[0].shape[:2]
            aligned = []
            for arr in images:
                h, w = arr.shape[:2]
                if (h, w) == (ref_h, ref_w):
                    aligned.append(arr)
                elif (h, w) == (ref_w, ref_h):
                    aligned.append(arr.T)
                else:
                    # Resize to reference shape
                    from PIL import Image as PILImage
                    pil = PILImage.fromarray((arr * 255).astype(np.uint8))
                    pil = pil.resize((ref_w, ref_h), PILImage.BILINEAR)
                    aligned.append(np.asarray(pil, dtype=np.float32) / 255.0)
            images = aligned

        # Typical scout is ~13 frames; warn if wildly off
        if len(images) > 20:
            logger.warning(
                "Patient %s has %d frames (expected ~13). Check that dicom_folder "
                "points at a single Patient-* directory, not the cohort root.",
                patient_id, len(images),
            )

        from freqscout_agent.tools.physics import PhysicsDescriptor
        offsets_hz = PhysicsDescriptor.infer_offsets(len(images))

        logger.info("Loaded %d DICOM frames for patient %s from %s.", len(images), patient_id, folder)
        return images, offsets_hz

    def _save_outputs(self, patient_id, stage_a, narrative, repair, consistency, trace):
        out = self.output_dir / patient_id
        out.mkdir(parents=True, exist_ok=True)

        (out / "stage_a.json").write_text(json.dumps(stage_a, indent=2, default=_json_default))
        (out / "narrative.txt").write_text(narrative)
        (out / "repair_issues.json").write_text(json.dumps(repair.issues_found, indent=2, default=_json_default))
        (out / "consistency.json").write_text(json.dumps({
            "neural_top1": consistency.neural_top1,
            "cv_top1": consistency.cv_top1,
            "vlm_top1": consistency.vlm_top1,
            "confidence": consistency.confidence,
            "confidence_label": consistency.confidence_label,
            "disagreements": consistency.disagreements,
        }, indent=2, default=_json_default))
        (out / "loop_trace.json").write_text(json.dumps({
            "iterations": trace.iterations,
            "terminated_at": trace.terminated_at,
            "termination_reason": trace.termination_reason,
        }, indent=2, default=_json_default))
        logger.info("Outputs saved to %s.", out)

    def _discover_patients(self, data_root: str) -> List[str]:
        """
        Find Patient-* folders under data_root (supports Dr-Dataset/1st_cohort_SAX/Patient-*).
        Populates self._patient_folders for later DICOM loading.
        """
        root = Path(data_root)
        if not root.exists():
            logger.warning("data_root does not exist: %s", root)
            return []

        # Prefer explicit Patient-* directories (recursive under Dr-Dataset/1st_cohort_SAX)
        patient_dirs = sorted(
            p for p in root.rglob("Patient-*")
            if p.is_dir() and (any(p.glob("*.dcm")) or any(p.glob("*.DCM")))
        )

        if not patient_dirs:
            # Fallback: immediate children that contain DICOMs
            patient_dirs = sorted(
                p for p in root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
                and (any(p.glob("*.dcm")) or any(p.glob("*.DCM")))
            )

        self._patient_folders = {p.name: p for p in patient_dirs}
        logger.info(
            "Discovered %d patients under %s: %s",
            len(patient_dirs), root, [p.name for p in patient_dirs[:5]] + (["…"] if len(patient_dirs) > 5 else []),
        )
        return [p.name for p in patient_dirs]

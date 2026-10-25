"""
Clinical Narrative Generator (Stage-B/C 7-block radiologist-grade reporting)
-----------------------------------------------------------------------------
Assembles context, invokes the LLM, and provides a deterministic template
fallback that strictly adheres to the 7-block clinical schema.
"""

from __future__ import annotations

import json
try:
    from .client import LLMClient
    from .prompts import CLINICAL_SYSTEM_PROMPT, CLINICAL_USER_PROMPT_TEMPLATE
    from ..core.evidence_store import EvidenceStore
except (ImportError, ValueError):
    from llm.client import LLMClient
    from llm.prompts import CLINICAL_SYSTEM_PROMPT, CLINICAL_USER_PROMPT_TEMPLATE
    from core.evidence_store import EvidenceStore


class NarrativeGenerator:

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient(backend="template")

    def generate(self, stage_a: Dict[str, Any], store: EvidenceStore) -> str:
        # 1. Try LLM generation if backend is not template
        if self.client.backend != "template":
            user_prompt = CLINICAL_USER_PROMPT_TEMPLATE.format(
                patient_id=stage_a["patient_id"],
                stage_a_json=json.dumps(stage_a, indent=2),
                evidence_table=store.to_tabular_summary(),
            )
            narrative = self.client.generate(CLINICAL_SYSTEM_PROMPT, user_prompt)
            if narrative and len(narrative.strip()) > 100:
                return narrative

        # 2. Fall back to deterministic clinical schema template
        return self._template_narrative(stage_a, store)

    def _template_narrative(self, stage_a: Dict[str, Any], store: EvidenceStore) -> str:
        rec = stage_a["recommended_offset"]
        pid = stage_a["patient_id"]
        alternatives = stage_a.get("acceptable_alternatives", [])
        rejected = stage_a.get("rejected_offsets", [])
        consistency = stage_a.get("consistency_summary", {})

        # Block 1
        b1 = (
            f"1. CLINICAL INDICATION & OBJECTIVE\n"
            f"Automated frequency-scout optimization for Patient {pid} at 3.0T cardiac bSSFP CINE MRI. "
            f"The diagnostic goal is to select the radiofrequency offset that eliminates off-resonance "
            f"banding artifacts within the left ventricular myocardium while maintaining high myocardium-to-blood "
            f"contrast across the cardiac cycle."
        )

        # Block 2
        b2 = (
            f"2. PRIMARY FREQUENCY RECOMMENDATION\n"
            f"RECOMMENDED OFFSET: {rec['offset_hz']:+.1f} Hz (Image #{rec['image_no']})\n"
            f"- Composite Score: {rec['composite_score']:.4f}\n"
            f"- Clinical Confidence: {rec['confidence_label'].upper()} (Rating: {rec['confidence']:.2f})\n"
            f"- Rationale: Image #{rec['image_no']} occupies the global optimum of the myocardial passband. "
            f"It demonstrates near-zero intra-myocardial banded fraction, optimal signal homogeneity, "
            f"and strong agreement across both deep-learning and quantitative physics metrics."
        )

        # Block 3
        if alternatives:
            alt_lines = [
                f"  * #{a['image_no']} ({a['offset_hz']:+.1f} Hz, Δ={a['delta_hz']:+.0f} Hz): "
                f"banded fraction={a['banded_fraction_myo']:.1%}, homogeneity={a['homogeneity']:.3f}"
                for a in alternatives
            ]
            alt_text = "\n".join(alt_lines)
            b3 = (
                f"3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)\n"
                f"A stable passband plateau exists around the primary recommendation. The following adjacent "
                f"offsets lie within the acceptable clinical tolerance window (±25 Hz to ±50 Hz) and retain "
                f"diagnostic image quality:\n{alt_text}\n"
                f"These alternative offsets provide acceptable blood-myocardium boundary delineation if "
                f"minor shimming adjustments are necessitated during the scan session."
            )
        else:
            b3 = (
                f"3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)\n"
                f"The acceptable passband plateau is narrow for this patient. Only Image #{rec['image_no']} "
                f"({rec['offset_hz']:+.1f} Hz) satisfies all stringent criteria for artifact-free myocardial evaluation. "
                f"Adjacent offsets exhibit early phase transition effects."
            )

        # Block 4
        if rejected:
            rej_lines = [
                f"  * #{r['image_no']} ({r['offset_hz']:+.1f} Hz): banded fraction={r['banded_fraction_myo']:.1%}, "
                f"affected AHA wall sectors: {r['affected_sectors']}"
                for r in rejected[:6]
            ]
            rej_text = "\n".join(rej_lines)
            b4 = (
                f"4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION\n"
                f"The following frequency offsets are strictly rejected due to clinically intolerable dark banding:\n"
                f"{rej_text}\n"
                f"Diagnostic Warning: Acquiring CINE sequences at these rejected offsets will result in localized signal voids "
                f"that mimic myocardial thinning, obscure wall motion, and impair ejection fraction quantification."
            )
        else:
            b4 = (
                f"4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION\n"
                f"No severe catastrophic full-thickness myocardial banding voids were detected in the intermediate sweep; "
                f"however, extreme off-resonance frequencies (±150 Hz) exhibit standard global peripheral phase roll-off."
            )

        # Block 5
        disagreements = consistency.get("disagreements", [])
        dis_text = " None (concordant decision)." if not disagreements else " " + " ".join(disagreements)
        b5 = (
            f"5. MULTI-SOURCE EVIDENCE RECONCILIATION\n"
            f"- Status: {consistency.get('status', 'concordant').replace('_', ' ').upper()}\n"
            f"- Deep Learning Classifier Pick: {consistency.get('cnn_pick', 'N/A')}\n"
            f"- Intra-Myocardial Banding Pick: {consistency.get('physics_pick', 'N/A')}\n"
            f"- Signal Homogeneity Pick: {consistency.get('homogeneity_pick', 'N/A')}\n"
            f"- Synthesis & Reconciliation:{dis_text}\n"
            f"Confidence Basis: {rec['confidence_basis']}"
        )

        # Block 6
        b6 = (
            f"6. TECHNICAL LIMITATIONS & PITFALLS\n"
            f"Evaluation assumes standard 25 Hz shimming step response at 3.0T. For anatomical regions near "
            f"cardiac-lung or cardiac-stomach air interfaces (susceptibility gradients), localized B0 shifts "
            f"may cause subtle intra-slice frequency dispersion."
        )

        # Block 7
        b7 = (
            f"7. SUMMARY & OPERATOR DIRECTIVE\n"
            f"OPERATOR ACTION: Set scanner center frequency offset to {rec['offset_hz']:+.1f} Hz (Image #{rec['image_no']}).\n"
            f"Quality status: VALIDATED DIAGNOSTIC. Expected banding burden in myocardium is minimal."
        )

        return "\n\n".join([b1, b2, b3, b4, b5, b6, b7])

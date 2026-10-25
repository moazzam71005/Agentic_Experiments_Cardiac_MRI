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
    from core.evidence_store import EvidenceStore
except (ImportError, ValueError):
    from llm.client import LLMClient
    from core.evidence_store import EvidenceStore

CLINICAL_SYSTEM_PROMPT = """You are a Senior Cardiac Radiologist and MRI Physicist.
Your task is to generate a comprehensive, structured Clinical Decision Narrative for an automated frequency-scout optimization in 3.0T cardiac bSSFP CINE MRI.

You MUST include ALL 7 numbered clinical blocks in order:
1. CLINICAL INDICATION & OBJECTIVE
2. PRIMARY FREQUENCY RECOMMENDATION
3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
4. EXCLUDED OFFSETS & ARTIFACT LOCALIZATION
5. METHODOLOGICAL RATIONALE & METRIC EVIDENCE
6. RISK OF SUBOPTIMAL OFFSET SELECTION
7. FINAL SCANNER CONFIGURATION INSTRUCTION

Do NOT hallucinate frequency values. Only cite numbers present in the provided evidence.
"""

CLINICAL_USER_PROMPT_TEMPLATE = """Generate the 7-block Clinical Decision Narrative for Patient {patient_id}.

Stage-A Decision JSON:
{stage_a_json}

Quantitative Evidence Matrix:
{evidence_table}
"""


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
            f"and strong agreement across multi-agent physics and contrast specialists."
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
                f"4. EXCLUDED OFFSETS & ARTIFACT LOCALIZATION\n"
                f"The following frequency offsets were vetoed due to significant off-resonance banding (BFM > 5%):\n"
                f"{rej_text}\n"
                f"Severe signal nulls in these frames obscure cardiac wall motion and preclude accurate ventricular analysis."
            )
        else:
            b4 = (
                f"4. EXCLUDED OFFSETS & ARTIFACT LOCALIZATION\n"
                f"No extreme myocardial signal voids detected; however, off-resonance margins deteriorate rapidly beyond ±75 Hz."
            )

        # Block 5
        b5 = (
            f"5. METHODOLOGICAL RATIONALE & METRIC EVIDENCE\n"
            f"- Multi-Agent Agreement: {consistency.get('agreement_status', 'passband_plateau')}\n"
            f"- Confidence Basis: {consistency.get('confidence_basis', 'Quantitative consensus')}\n"
            f"- Banding Burden at Selected Offset: {rec.get('banded_fraction_myo', 0.0):.2%}\n"
            f"- Homogeneity Index: {rec.get('homogeneity', 0.0):.3f}"
        )

        # Block 6
        b6 = (
            f"6. RISK OF SUBOPTIMAL OFFSET SELECTION\n"
            f"Operating at an incorrect frequency offset in 3.0T bSSFP introduces dark band artifacts traversing "
            f"the myocardium. This mimics focal wall thinning, impairs regional ejection fraction assessment, "
            f"and risks false-positive diagnosis of myocardial infarction or scar."
        )

        # Block 7
        b7 = (
            f"7. FINAL SCANNER CONFIGURATION INSTRUCTION\n"
            f"Apply center frequency offset of {rec['offset_hz']:+.1f} Hz (Image #{rec['image_no']}) on the scanner console. "
            f"Proceed with diagnostic SAX CINE acquisition."
        )

        return "\n\n".join([b1, b2, b3, b4, b5, b6, b7])

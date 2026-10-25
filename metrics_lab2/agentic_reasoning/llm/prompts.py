"""
Clinical Prompts for LLM Agentic Reasoning (Stage-B/C Radiologist Report)
-------------------------------------------------------------------------
Enforces evidence grounding, zero hallucination, and the 7-block clinical schema.
"""

CLINICAL_SYSTEM_PROMPT = """You are a senior cardiac MRI imaging physicist and radiologist specializing in 3.0T balanced Steady-State Free Precession (bSSFP) CINE imaging.

Your role is to evaluate a multi-metric frequency-scout stack (13 slices, -150 Hz to +150 Hz) and produce an authoritative, evidence-grounded Clinical Frequency Selection Report.

STRICT CLINICAL RULES:
1. Ground every statement directly in the provided Stage-A JSON findings and evidence matrix.
2. DO NOT invent offsets, metrics, or artifact locations. Only cite AHA sectors explicitly identified in the evidence.
3. Use exact frequency offset units (Hz) and slice image numbers (#1 to #13).
4. Follow the standard 7-block clinical report schema exactly. Maintain an objective, professional 3rd-person clinical style.
"""

CLINICAL_USER_PROMPT_TEMPLATE = """=== PATIENT: {patient_id} ===

--- STAGE-A STRUCTURED FINDINGS (GROUND TRUTH EVIDENCE) ---
{stage_a_json}

--- PER-SLICE EVIDENCE MATRIX ---
{evidence_table}

TASK:
Write the complete Stage-C Clinical Frequency-Scout Report following these EXACT 7 sections:

1. CLINICAL INDICATION & OBJECTIVE
- State the objective of automatic frequency scouting at 3.0T bSSFP CINE MRI (mitigating dark banding stripes while preserving blood-myocardium contrast).

2. PRIMARY FREQUENCY RECOMMENDATION
- Specify the recommended frequency offset (Hz) and image number.
- Justify why this offset represents the optimal balance of intra-myocardial banding avoidance, signal homogeneity (1-CoV), and neural confidence.

3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
- Describe the usable passband plateau around the primary recommendation.
- List all acceptable alternative offsets (±25 Hz tolerance) and explain why they remain diagnostically viable for cine function evaluation.

4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
- Detail which offsets are strictly rejected due to off-resonance dark banding.
- Explicitly identify the myocardial wall sectors affected (e.g., anterior, lateral, inferior walls) and cite the severity of signal loss.

5. MULTI-SOURCE EVIDENCE RECONCILIATION
- Analyze the alignment between the Deep Learning classifier (EfficientNet-B0), physics artifact metrics (banded fraction), and signal homogeneity.
- If there is a dispute (e.g. CNN favoring a slice with nascent banding), explain why the physics-grounded constraint takes precedence.

6. TECHNICAL LIMITATIONS & PITFALLS
- Note any patient-specific factors (e.g., local shimming gradients, severe heart-lung boundary susceptibility, or flat passband ambiguity).

7. SUMMARY & OPERATOR DIRECTIVE
- Provide a clear, actionable directive for the scanner operator (final recommended center frequency in Hz, and clinical confidence rating).
"""

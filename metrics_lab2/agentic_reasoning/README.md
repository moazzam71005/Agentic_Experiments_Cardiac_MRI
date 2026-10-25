# Experiment 13 — LLM Arbitration Loop (metrics_lab2)

## What's new vs Exp 12 (metrics_lab)

| Component | Exp 12 | Exp 13 |
|---|---|---|
| `consistency_checker.py` | 4 states | 5 states + `requires_arbitration` flag + `borderline_veto` |
| `stage_a_generator.py` | Always deterministic composite | Accepts `ArbitrationResult`, logs `decision_source` |
| `llm_arbitrator.py` | ❌ Did not exist | ✅ New: routes divergent cases to LLM |
| `run_exp13_pipeline.py` | ❌ | ✅ New: full orchestration with arbitration loop |
| `eval/evaluator.py` | Basic metrics | + arbitration tracking (rate, pass rate, improvement rate) |

## Architecture (Exp 13)

```
13 bSSFP Frames → [Multi-Tool Extraction] → [Evidence Store]
                                                    │
                                          [Consistency Checker]
                                        ┌─────────────┴────────────────┐
                              tools agree?                       tools disagree?
                     (full_consensus /                  (divergent / borderline_veto)
                      passband_plateau /                           │
                      physics_veto)                    [LLM ARBITRATOR]
                              │                   reason over numeric evidence table
                              │                        │ audit pass?
                              │                   ┌────┴────┐
                              │                  yes        no
                              │                   │         │
                              └──────────────────►[Stage-A]◄┘ (fallback to composite)
                                                    │
                                              [Stage-B Report]
                                                    │
                                          [Faithfulness Audit]
```

## Running

### Template mode (no LLM, identical to Exp 12):
```bash
cd metrics_lab2/agentic_reasoning
python run_exp13_pipeline.py
```

### On GPU machine (direct HuggingFace):
```bash
python run_exp13_pipeline.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
```

### Via SSH tunnel to GPU:
```bash
# On GPU machine: serve with vLLM or Ollama on port 8000
python run_exp13_pipeline.py --backend openai --api-base http://localhost:8000/v1
```

### Single patient debug:
```bash
python run_exp13_pipeline.py --backend hf --patient Patient-BEV
```

## Key Design Decisions

1. **LLM temperature = 0.0** for arbitration: deterministic, reproducible.
2. **Arbitration only on divergent / borderline_veto**: no overhead when tools agree.
3. **Hard safety audit**: arbitrator cannot recommend a frame with BFM > 0.08.
4. **Hz correction**: if LLM states wrong Hz for a given image number, the image number is authoritative and Hz is silently corrected.
5. **Fallback on any failure**: if LLM is unavailable or audit fails, the composite score is used as in Exp 12. The pipeline never crashes.
6. **Full provenance**: every output JSON includes `decision_source` and `arbitration_trace.json`.

## Output files per patient

```
outputs_exp13/
  Patient-XYZ/
    stage_a.json              # structured findings (includes decision_source)
    clinical_report.md        # 7-block radiologist report
    audit_report.json         # faithfulness audit
    arbitration_trace.json    # full LLM trace (invoked/passed/rationale/raw)
```

## Cohort-level outputs

```
outputs_exp13/
  per_patient_evaluation.csv   # per-row breakdown with arb_outcome column
  benchmark_comparison.csv     # Exp13 vs all prior baselines
  audit_summary.json           # hallucination / faithfulness stats
  arbitration_summary.json     # arbitration invocation + improvement rates
```

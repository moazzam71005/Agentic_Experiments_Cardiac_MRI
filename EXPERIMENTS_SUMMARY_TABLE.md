# FreqScout Agentic Pipeline — Experiments Summary & Pipeline Architecture

**Project:** FreqScout — Agentic Optimal Frequency Offset Selection in bSSFP Cardiac MRI  
**Dataset:** 26 Patients, Short-Axis (SAX) bSSFP Frequency Scout Series (13 frequency offsets per patient: −150 Hz to +150 Hz, step 25 Hz)  
**Evaluation Protocol:** Leave-One-Out Cross-Validation (LOOCV, $n=26$) & Ground-Truth Radiographer Selection Comparison  

---

## 1. Complete Experiments Summary Table (Exp 1 – Exp 11)

| Exp # | Experiment Name | Primary Models / Methodology | Features / Inputs | Top-1 Acc | Clinical Acc (±1 Step / ±25Hz) | MAE (Hz) | Evaluation Protocol | Key Findings & Impact |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | **Direct CNN Slice Scoring** | ResNet18, EfficientNet-B0, SimpleCNN with Soft Gaussian Labels ($\sigma=1.0$) | Single DICOM Slice (Independent scoring) | 22.7% | 61.3% (Top-3) | 1.35 steps (~33.8 Hz) | 5-Fold CV ($n=26$, optimistic) | First baseline. Scored slices independently without comparing within stack. 5-fold CV suffered from patient-level leakage across small folds. |
| **Exp 2** | **Ranking Losses & Honest LOOCV** | ResNet18 + ListNet Ranking Loss & Pairwise Margin Loss | Single DICOM Slice + Soft Labels ($\sigma=1.5$) | 3.8% | 46.2% (Top-3) | 2.19 steps (~54.8 Hz) | **LOOCV ($n=26$, honest)** | Performance collapsed from 22.7% (5-fold) to 3.8% (LOOCV), proving independent models overfit to fold quirks. Naive center-slice baseline (26.9%) beat independent scoring. |
| **Exp 3** | **Siamese Pairwise Network** | Shared EfficientNet-B0 backbone + Comparison MLP ($C(13,2)=78$ pairs/pt) | Slice Pairs `[fA, fB, fA-fB, fA*fB]` | 15.4% | 53.8% (Top-3) | 2.50 steps (~62.5 Hz) | LOOCV ($n=26$) | Relative pairwise comparison jumped Top-1 from 3.8% to 15.4%, proving relative comparisons generalize far better than scalar slice scoring. |
| **Exp 4** | **DINOv2 Feature Extraction** | DINOv2-Base (768-dim) feature cache + Stack Transformer | DINOv2-B frozen ViT embeddings across all 338 slices | — | — | — | Feature Cache | Established DINOv2 foundation embeddings as superior visual feature representations for cardiac MRI slice comparison. |
| **Exp 5** | **DINOv2-Large Neural Ranker** | DINOv2-Large (1024/2048-dim) + Stack Transformer | DINOv2-L frozen embeddings | — | — | — | LOOCV ($n=26$) | Produced primary neural score distribution (`exp5_loocv_per_patient.csv`) used as the neural prior across all subsequent experiments. |
| **Exp 6** | **Gradient Stack Transformer & Ensemble** | Gradient-interleaved Stack Transformer ($g_i = f_{i+1}-f_i$) + 3-Way Borda Ensemble | DINOv2-L features + adjacent slice difference tokens | 42.3% (Oracle) | 92.3% (Oracle Top-3) | — | LOOCV & Borda Oracle | Introduced adjacent difference tokens ($2S-1$ seq length) to capture quality peak crossover ($dq/df = 0$). Borda oracle showed high upper bound. |
| **Exp 7** | **Neural-Only Agent Baseline** | Pure Neural Ranker prior (Exp5 DINOv2-L) | Exp5 Neural Scores | 23.1% (6/26) | 38.5% (10/26) | 64.4 Hz | LOOCV ($n=26$) | **Reference Anchor.** Neural ranker alone picks extreme sweep frequencies (±150 Hz) where background looks uniform, failing on subtle intra-cardiac banding. |
| **Exp 8** | **Myocardium-Grounded CV Fusion** | LVSA U-Net Consensus Mask + Intra-Myocardial Banding Metrics | LVSA U-Net segmentation + `banded_fraction_myo` + Homogeneity | 30.8% (8/26) | 42.3% (11/26) | 63.5 Hz | LOOCV ($n=26$) | Integrated LVSA U-Net segmentation. Discovered standalone `argmin(burden)` fails (3.8% Top-1) because extreme offsets have zero myocardial burden; burden must be a soft penalty. |
| **Exp 9** | **Full Pipeline + Center Physics Prior** | Fused Agent Pipeline + Gaussian Center Prior $\exp(-0.5(Hz/100)^2)$ + Qwen2-VL | Neural + CV + Center Prior + Qwen2-VL (2B/7B) | 34.6% (9/26) | 65.4% (18/26) | 26.9 Hz | LOOCV ($n=26$) | **Major Milestone.** Gaussian center prior reduced MAE by **58%** (64.4 Hz → 26.9 Hz). **Critical Negative Result:** General VLMs suffer from 0 Hz center-bias (0% VLM Top-1); set $W_{\text{VLM}}=0$. |
| **Exp 10** | **Physics Passband & Acceptable Set Fusion** | Passband Peak Detector + Contiguous Acceptable Set ($\pm 1$ step) + Simplex Weight Rebalance | 1-BFM (0.70) + Neural (0.20) + Center Prior (0.10) | **42.3% (11/26)** | **76.9% (20/26)** | **22.1 Hz** | LOOCV ($n=26$) | **Best Deploy Selector.** Identified `1-banded_fraction_myo` as single strongest selector. Achieved **73.1% Acceptable-Set Hit** and reduced MAE by **65.7%** vs baseline. |
| **Exp 11** | **Full Agentic Clinical Reasoner** | Reconciled Multi-Tool Evidence + Passband Margin Confidence + 7-Block Stage-C Report | 7 Agentic Tools + Qwen2.5-7B-Instruct Clinical Reasoner | **42.3% (11/26)** | **76.9% (20/26)** | **22.1 Hz** | Full Agentic Pipeline Run | **Current Best System.** Frozen Exp10 selector + active agentic evidence reconciliation, passband-margin confidence calibration, zero-hallucination, and 7-block radiologist reporting. |

---

## 2. Comprehensive Experiment Descriptions

### Exp 1: Direct CNN Slice Scoring
- **Objective:** Evaluate baseline deep learning CNNs on individual DICOM slice quality scoring.
- **Architecture:** SimpleCNN, ResNet18, and EfficientNet-B0 trained to predict scalar quality scores using target Gaussian distributions ($\sigma=1.0$) centered on the ground-truth slice index.
- **Key Outcome:** ResNet18 achieved 22.7% Top-1 and 61.3% Top-3 on 5-Fold CV. However, because slices were evaluated independently without cross-slice stack context, the model failed to capture relative quality ranking and overfit to fold split quirks.

### Exp 2: Ranking Losses & Honest LOOCV Protocol
- **Objective:** Transition from scalar regression to stack-level probability ranking losses (ListNet + Pairwise Margin Loss) and establish a rigorous Leave-One-Out Cross-Validation (LOOCV) benchmark.
- **Key Outcome:** Under honest LOOCV ($n=26$), performance collapsed to 3.8% Top-1 (below naive center-slice baseline of 26.9%). Proved that independent slice scoring models fail to generalize across unseen patient stacks.

### Exp 3: Siamese Pairwise Comparison Network
- **Objective:** Formulate best-slice selection as a pairwise comparison problem ($C(13,2)=78$ pairs per patient) using a shared EfficientNet-B0 backbone with feature difference/product MLPs and tournament win-count inference.
- **Key Outcome:** Top-1 accuracy increased significantly from 3.8% to 15.4% on LOOCV, confirming that relative pairwise comparisons are vastly superior to single-image scoring for frequency scout ranking.

### Exp 4: DINOv2 Foundation Feature Extraction
- **Objective:** Leverage self-supervised vision foundation models to extract rich anatomical and artifact representations without requiring task-specific fine-tuning on small datasets.
- **Key Outcome:** Cached DINOv2-Base (768-dim) embeddings across all 338 images. Demonstrated superior representation stability over standard ImageNet-pretrained CNN backbones.

### Exp 5: DINOv2-Large Neural Ranker
- **Objective:** Build a LOOCV Stack Transformer utilizing DINOv2-Large (1024/2048-dim) embeddings to generate per-patient neural slice quality scores.
- **Key Outcome:** Outperformed prior deep models. The resulting score matrix (`exp5_loocv_per_patient.csv`) was established as the permanent **Neural Ranker prior** ($W_{\text{NEURAL}}$) for all downstream hybrid agent pipelines.

### Exp 6: Gradient-Aware Stack Transformer & Ensemble
- **Objective:** Incorporate spatial-frequency quality gradients ($g_i = f_{i+1} - f_i$) directly into the Transformer attention mechanism to detect where image quality peaks along the offset sweep ($dq/df = 0$).
- **Key Outcome:** Sequence length expanded to $2S-1=25$ tokens. The 3-way Borda oracle (Exp3+4+5) reached 42.3% Top-1 and 92.3% Top-3, demonstrating that frequency gradient cues strongly complement static visual features.

### Exp 7: Neural-Only Agent Baseline (Reference Anchor)
- **Objective:** Establish the performance of an unconstrained neural agent relying purely on DINOv2-L LOOCV scores without computer vision or physics constraints.
- **Key Outcome:** Achieved 23.1% Top-1, 38.5% Clinical Accuracy ($\pm 1$ step), and MAE of 64.4 Hz. Unconstrained neural rankers frequently fell into "extreme offset traps" ($\pm 150$ Hz) where background uniformity is high despite dark banding corrupting adjacent tissue.

### Exp 8: Myocardium-Grounded CV & Banding Burden Ablation
- **Objective:** Ground the agent in cardiac anatomy by embedding an LVSA U-Net consensus segmentation tool to extract myocardium masks (Label 2) and compute intra-myocardial banding metrics (`banded_fraction_myo`, `band_trough_depth`, `banding_burden`).
- **Key Outcome:** Standalone `argmin(burden)` failed catastrophically (3.8% Top-1) because at extreme offsets ($\pm 150$ Hz), banding has not yet entered the myocardium (burden $\approx 0$). Established that myocardial banding metrics must operate as soft penalties within a composite score rather than standalone selectors.

### Exp 9: Full Pipeline + Physics Center Gaussian Prior
- **Objective:** Integrate a physics-grounded Gaussian center prior $W_{\text{CENTER}} = 0.10 \times \exp(-0.5 (Hz/100)^2)$ into the composite score, rebalance weightings, and integrate local/lab VLMs (Qwen2-VL 2B / Qwen2.5-VL 7B) for visual artifact validation.
- **Key Outcome:** MAE dropped by **58%** (64.4 Hz → 26.9 Hz) and Clinical Accuracy jumped to 65.4%. **Critical Negative Result:** General VLMs exhibited severe center-frequency bias ("0 Hz is always optimal", 0% Top-1 accuracy). VLMs were strictly relegated to visual QA verification ($W_{\text{VLM}} = 0$).

### Exp 10: Physics Passband & Acceptable-Region Fusion (Deploy SOTA)
- **Objective:** Introduce `PassbandTool` to model bSSFP myocardial passband response, partition sweep into contiguous acceptable ($\pm 1$ step around peak) vs rejected sets, and optimize weight simplex.
- **Key Outcome:** Achieved **42.3% Top-1**, **76.9% Clinical Accuracy ($\pm 1$ step)**, **73.1% Acceptable-Set Hit**, and MAE of **22.1 Hz** (-65.7% reduction vs baseline). Identified `1 - banded_fraction_myo` (weight = 0.70) as the single strongest selector.

### Exp 11: Full Agentic Clinical Reasoner (Current Best Pipeline)
- **Objective:** Decouple the numerical selector layer from the clinical reasoning layer. Update `ConsistencyChecker` to reconcile multi-tool evidence, calibrate passband-margin confidence, target VLM calls to critical offsets, and generate 7-block structured clinical radiologist reports (Stage-C).
- **Key Outcome:** Preserves SOTA selector performance (42.3% Top-1, 76.9% $\pm 1$ step, 22.1 Hz MAE) while achieving complete cross-tool evidence alignment, calibrated confidence labels, 0% hallucination rate, and automated radiologist-ready reports.

---

## 3. Architecture Diagram of the Latest Best Agentic Pipeline (Exp 11 / Exp 10)

```mermaid
flowchart TD
    %% Styling
    classDef inputStyle fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc;
    classDef toolStyle fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef coreStyle fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef stageStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef outputStyle fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#f8fafc;

    subgraph INPUT ["1. DICOM Acquisition"]
        A["13-Slice bSSFP Frequency Scout Stack<br/>(-150 Hz to +150 Hz, step 25 Hz)"]:::inputStyle
    end

    subgraph TOOLS ["2. FreqScout Multi-Tool Analysis Suite"]
        T1["T1: NeuralScorer<br/>(Exp5 LOOCV DINOv2-Large Ranker)"]:::toolStyle
        T2["T2: SegmentationTool<br/>(LVSA U-Net Consensus Mask)"]:::toolStyle
        T3["T3: BandingDetector<br/>(Intra-Myocardial Banding Nulls & BFM)"]:::toolStyle
        T4["T4: QualityMetrics<br/>(Myocardial Homogeneity & CNR)"]:::toolStyle
        T5["T5: PhysicsDescriptor & PassbandTool<br/>(Offset Sweep & Center Gaussian Prior)"]:::toolStyle
        T6["T6: VLMDescriptor<br/>(Qwen2.5-VL-7B Visual QA)"]:::toolStyle
        T7["T7: KnowledgeRetriever<br/>(bSSFP Physics Guidelines & Rules)"]:::toolStyle
    end

    subgraph AGENT ["3. Agentic Reasoning & Reconciliation Engine"]
        GA["GapAnalyser<br/>(Identifies Missing Evidence & Tool Gaps)"]:::coreStyle
        CC["ConsistencyChecker<br/>(Multi-Tool Agreement & Passband Margin Confidence)"]:::coreStyle
    end

    subgraph COMPOSITE ["4. Stage-A Composite Score Generator"]
        SA["Composite Physics Fusion Formula:<br/>Score = 0.70 × (1 - BFM) + 0.20 × Neural + 0.10 × CenterPrior<br/>(W_VLM = 0.00 — Explain Only)"]:::stageStyle
    end

    subgraph DECISION ["5. Clinical Decision & Passband Partitioning"]
        PB["Passband Curve & Sets:<br/>• Primary Offset Pick (Highest Score)<br/>• Contiguous Acceptable Set (±1 Step around Peak)<br/>• Rejected Offsets (High BFM / Severe Banding)"]:::stageStyle
    end

    subgraph REPORT ["6. Stage-C Automated Clinical Report (LLM Reasoner)"]
        SB["Qwen2.5-7B-Instruct Clinical Reasoner<br/>Outputs 7-Block Radiologist Documentation:<br/>1. Primary Offset & Frequency<br/>2. Acceptable Windows<br/>3. Rejected Offsets & Physics Rationale<br/>4. Multi-Source Evidence Agreement<br/>5. Calibrated Confidence Basis<br/>6. VLM Visual QA Validation<br/>7. Actionable Clinical Recommendation"]:::outputStyle
    end

    %% Connections
    A --> T1 & T2 & T5
    T2 --> T3 & T4
    T1 & T2 & T3 & T4 & T5 --> GA
    GA --> T6 & T7
    T1 & T3 & T5 & T6 & T7 --> CC
    CC --> SA
    SA --> PB
    PB --> SB
```

---

## 4. Summary of Key Pipeline Components & Tool Specifications

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FREQSCOUT AGENT TOOL SPECIFICATIONS                               │
├───────────────────────┬─────────────────────────────────────────────────┬────────────────────────┤
│ Tool Identifier       │ Primary Function / Output                       │ Contribution Weight    │
├───────────────────────┼─────────────────────────────────────────────────┼────────────────────────┤
│ T1: NeuralScorer      │ DINOv2-Large Stack Transformer LOOCV score      │ W_NEURAL = 0.20        │
│ T2: SegmentationTool  │ LVSA U-Net consensus mask (RV, Myo, LV Pool)    │ Mask prerequisite      │
│ T3: BandingDetector   │ Banded Fraction of Myocardium (BFM) & null depth│ W_NO_BFM = 0.70        │
│ T4: QualityMetrics    │ Myocardial signal intensity, homogeneity, CNR  │ Diagnostic QA          │
│ T5: PhysicsDescriptor │ Gaussian Center Prior exp(-0.5*(Hz/100)^2)      │ W_CENTER = 0.10        │
│ T6: VLMDescriptor     │ Qwen2.5-VL-7B qualitative artifact inspection    │ W_VLM = 0.00 (QA only) │
│ T7: KnowledgeRetriever│ bSSFP physics rules & off-resonance lookup      │ Reasoning Context      │
└───────────────────────┴─────────────────────────────────────────────────┴────────────────────────┘
```

### Key Performance Highlights of the Latest Best Pipeline (Exp 11 / Exp 10):
- **Top-1 Accuracy:** **42.3%** (11/26 exact radiographer match)
- **Clinical Accuracy ($\pm 1$ step / $\pm 25$ Hz):** **76.9%** (20/26 clinically equivalent match)
- **Acceptable-Set Hit Rate:** **73.1%** (19/26 selected offset falls inside valid passband window)
- **Mean Absolute Error (MAE):** **22.1 Hz** (reduced from 64.4 Hz baseline — **65.7% improvement**)
- **Hallucination Rate:** **0.0%** (physically grounded evidence reconciliation)

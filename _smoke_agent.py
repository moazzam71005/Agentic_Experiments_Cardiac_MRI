"""Smoke test for the freqscout_agent package using synthetic data."""
import sys, numpy as np
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)

print('=== Smoke test with synthetic data ===')

# Generate 13 synthetic 128x128 frequency scout images
np.random.seed(42)
images = []
for i in range(13):
    img = np.random.rand(128, 128).astype(np.float32)
    cx, cy, r = 64, 64, 20
    # LV blood pool
    ys, xs = np.ogrid[:128, :128]
    circle = (xs - cx)**2 + (ys - cy)**2
    img[circle < r**2] = 0.85 + 0.1 * np.random.rand(*(img[circle < r**2].shape))
    # Myocardium ring
    ring = (circle >= r**2) & (circle < 32**2)
    img[ring] = 0.45 + 0.05 * np.random.rand(*(img[ring].shape))
    # Banding artifact on extreme slices
    if i < 3 or i > 10:
        img[50:55, :] = 0.02
    images.append(img)

offsets_hz = list(range(-150, 175, 25))[:13]
print(f'Images: {len(images)}, Offsets: {offsets_hz}')

# T1: Segmentation
from freqscout_agent.tools.segment import SegmentationTool
seg_tool = SegmentationTool(checkpoint=None, device='cpu')
seg_result = seg_tool.run(images, offsets_hz, 'SMOKE_TEST', source_slice_idx=6)
print(f'Segmentation: myo_pixels={seg_result.myocardium_mask.sum()}, pool_pixels={seg_result.lv_pool_mask.sum()}')
print(f'  LV centroid: {seg_result.lv_centroid}')

# T2: Banding
from freqscout_agent.tools.banding import BandingDetector
band_results = BandingDetector().run(images, offsets_hz, seg_result)
print(f'Banding: {sum(1 for b in band_results if b.band_present)}/{len(band_results)} slices have bands')

# T3: Quality
from freqscout_agent.tools.quality import QualityMetrics
qual_results = QualityMetrics().run(images, offsets_hz, seg_result)
covs = [q.cov for q in qual_results]
print(f'Quality: CoV range [{min(covs):.3f}, {max(covs):.3f}]')

# Evidence State
from freqscout_agent.agent.evidence_state import EvidenceState, EvidencePacket
state = EvidenceState('SMOKE_TEST', 13, offsets_hz)

# Neural mock
from freqscout_agent.tools.neural_scorer import NeuralResult
scores = np.random.rand(13).tolist()
softmax = np.exp(scores) / np.exp(scores).sum()
neural_res = NeuralResult(
    patient_id='SMOKE_TEST', slice_scores=scores,
    softmax_confidence=softmax.tolist(),
    ranking=list(np.argsort(scores)[::-1]),
    top1_idx=int(np.argmax(scores)),
    top3_idx=list(np.argsort(scores)[::-1][:3]),
    method='feature_norm', provenance={'tool': 'neural'},
)
state.add(EvidencePacket('SMOKE_TEST', -1, 0.0, 'neural', 0, neural_res, f'top1={neural_res.top1_idx}'))

for b in band_results:
    state.add(EvidencePacket('SMOKE_TEST', b.slice_idx, b.offset_hz, 'banding', 0, b,
        f'severity={b.severity:.2f} myo={b.bands_in_myocardium}'))

for q in qual_results:
    state.add(EvidencePacket('SMOKE_TEST', q.slice_idx, q.offset_hz, 'quality', 0, q,
        f'CoV={q.cov:.3f}'))

state.add(EvidencePacket('SMOKE_TEST', 6, offsets_hz[6], 'segment', 0, seg_result, 'seg done'))

from freqscout_agent.tools.physics import PhysicsResult
phys = PhysicsResult('SMOKE_TEST', offsets_hz, None, gt_best_slice=6, gt_offset_hz=0.0,
                     per_slice=[], provenance={})
state.add(EvidencePacket('SMOKE_TEST', -1, 0.0, 'physics', 0, phys, 'physics attached'))

from freqscout_agent.tools.vlm_describe import VLMSliceResult
vlm_r = VLMSliceResult('SMOKE_TEST', 6, 0.0, False, 'none', 'none', False, False,
    'excellent', 'bright', 'optimal', 'No banding', 'high', '{}', {'tool': 'vlm'})
state.add(EvidencePacket('SMOKE_TEST', 6, 0.0, 'vlm', 0, vlm_r, 'optimal'))

state.add(EvidencePacket('SMOKE_TEST', -1, 0.0, 'retrieve', 0, [], 'retrieved 3 chunks'))

print(f'Evidence state: {len(state)} packets, tools={state.tools_present()}')

# Gap analysis
from freqscout_agent.agent.gap_analysis import GapAnalyser
gap_report = GapAnalyser().analyse(state)
print(f'Gap: complete={gap_report.is_complete}, unsatisfied={[i.id for i in gap_report.unsatisfied]}')

# Consistency
from freqscout_agent.agent.consistency import ConsistencyChecker
cons = ConsistencyChecker().run(state)
print(f'Consistency: neural={cons.neural_top1} cv={cons.cv_top1} conf={cons.confidence:.3f} ({cons.confidence_label})')

# Stage A
from freqscout_agent.generation.stage_a_findings import StageAGenerator
stage_a = StageAGenerator().generate(state, cons)
rec = stage_a['recommended_offset']
print(f'Stage-A: recommend slice {rec["slice_idx"]} ({rec["offset_hz"]:+.0f} Hz), conf={rec["confidence"]:.3f}')
print(f'  Alternatives: {len(stage_a["alternative_offsets"])}, agreement={stage_a["evidence_agreement"]}')

# Stage B
from freqscout_agent.generation.stage_b_narrative import StageBGenerator
narrative = StageBGenerator().generate(stage_a, state, cons, retrieved_chunks=[])
print(f'Stage-B: {len(narrative)} chars')
print(narrative[:300])

# Repair
from freqscout_agent.generation.repair import RepairModule
repair = RepairModule().run(stage_a, narrative)
print(f'Repair: was_repaired={repair.was_repaired}, issues={len(repair.issues_found)}')

# Proxy metrics
from freqscout_agent.eval.proxy_metrics import ProxyMetrics, SinglePatientResult
pr = SinglePatientResult(
    patient_id='SMOKE_TEST', gt_slice=rec['slice_idx'], gt_hz=rec['offset_hz'],
    pred_slice=rec['slice_idx'], pred_hz=rec['offset_hz'],
    top3_slices=[a['slice_idx'] for a in stage_a['alternative_offsets']] + [rec['slice_idx']],
    confidence=rec['confidence'], confidence_label=rec['confidence_label'],
    agreement=stage_a['evidence_agreement'], repair_issues=len(repair.issues_found), hallucinated=False,
)
pm_result = ProxyMetrics().compute([pr])
pm_result_display = ProxyMetrics()
pm_result_display.print_summary(pm_result)

# Reasoning metrics
from freqscout_agent.eval.reasoning_metrics import ReasoningMetrics
rm_res = ReasoningMetrics().evaluate(narrative, stage_a, patient_id='SMOKE_TEST')
print(f'Reasoning: faithfulness={rm_res.faithfulness_score:.3f} coverage={rm_res.checklist_coverage:.3f}')

# KB fallback
from freqscout_agent.knowledge.parse_pdfs import _build_fallback_chunks
chunks = _build_fallback_chunks()
print(f'Fallback KB chunks: {len(chunks)}')

print()
print('=== ALL TESTS PASSED ===')

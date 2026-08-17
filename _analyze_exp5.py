
import pandas as pd, numpy as np, json, ast, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_theme(style='whitegrid', context='notebook')

pp4 = pd.read_csv('exp4_outputs/exp4_loocv_per_patient.csv').set_index('patient_id')
pp5 = pd.read_csv('exp5_outputs/exp5_loocv_per_patient.csv').set_index('patient_id')

def parse_top3(val):
    if isinstance(val, list): return [int(x) for x in val]
    try: return [int(x) for x in ast.literal_eval(str(val))]
    except: return [int(val)]

# ── 1. Borda ensemble (Exp4 top-3 + Exp5 full scores) ────────────────────────
rows = []
for pid in sorted(pp4.index):
    e4 = pp4.loc[pid]
    e5 = pp5.loc[pid]
    gt = int(e4['gt_image_no'])

    # Exp5: softmax-normalised scores
    s5 = json.loads(e5['slice_scores'])
    keys5 = [int(k) for k in s5.keys()]
    vals5 = np.array([float(v) for v in s5.values()])
    vals5 = np.exp(vals5 - vals5.max())
    vals5 = vals5 / vals5.sum()
    e5_norm = {k: float(v) for k, v in zip(keys5, vals5)}

    # Exp4: Borda from top-3 (3pts, 2pts, 1pt), scale so each ~0.2
    e4_top3 = parse_top3(e4['pred_top3'])
    borda = {inst: (3 - r) * 0.2 for r, inst in enumerate(e4_top3)}

    # Fuse
    all_insts = sorted(set(list(e5_norm.keys()) + list(borda.keys())))
    combined = {k: e5_norm.get(k, 0) + borda.get(k, 0) for k in all_insts}

    ranked = sorted(combined, key=lambda k: -combined[k])
    top1 = ranked[0]
    top3 = ranked[:3]

    rows.append({
        'patient_id':   pid,
        'gt_image_no':  gt,
        'pred_top1':    top1,
        'pred_top3':    top3,
        'top1_correct': int(top1 == gt),
        'top3_correct': int(gt in top3),
        'mae':          abs(top1 - gt),
    })

borda_df = pd.DataFrame(rows)
borda_df.to_csv('exp5_outputs/exp5_borda_ensemble.csv', index=False)

b_top1 = borda_df['top1_correct'].mean()
b_top3 = borda_df['top3_correct'].mean()
b_mae  = borda_df['mae'].mean()
print(f'Borda Ensemble (Exp4+Exp5): Top-1={b_top1*100:.1f}%  Top-3={b_top3*100:.1f}%  MAE={b_mae:.2f}')
print('Correct:', borda_df[borda_df['top1_correct']==1]['patient_id'].tolist())

# ── 2. Full progression table ─────────────────────────────────────────────────
exp3_pp = {
    'Patient-BEV':  (1,1,0),  'Patient-BGT':  (0,0,3),  'Patient-BVE':  (0,1,1),
    'Patient-CEA':  (0,1,1),  'Patient-CEJP': (0,0,3),  'Patient-DEM':  (0,0,3),
    'Patient-FAJ':  (0,0,2),  'Patient-FEC':  (0,0,7),  'Patient-GJ':   (0,1,1),
    'Patient-HMJ':  (1,1,0),  'Patient-HRJ':  (0,1,5),  'Patient-IAM':  (0,1,1),
    'Patient-INA':  (0,0,5),  'Patient-KP':   (0,1,1),  'Patient-MJ':   (0,0,3),
    'Patient-MS':   (0,0,3),  'Patient-MSR':  (0,1,2),  'Patient-OAF':  (0,0,6),
    'Patient-PJ':   (0,0,2),  'Patient-RAK':  (1,1,0),  'Patient-RK':   (0,0,5),
    'Patient-SK':   (0,1,2),  'Patient-SWJ':  (0,1,3),  'Patient-TL':   (0,0,4),
    'Patient-WBC':  (0,1,2),  'Patient-WMR':  (1,1,0),
}

print()
print('=== Progression: Exp3 -> Exp4 -> Exp5 -> Ensemble ===')
header = f"{'Patient':<14} E3t1 E4t1 E5t1 Ens1  E3mae E4mae E5mae Ensmae"
print(header)
for pid in sorted(pp4.index):
    e3t1, e3t3, e3mae = exp3_pp.get(pid, (0, 0, 0))
    e4 = pp4.loc[pid]
    e5 = pp5.loc[pid]
    ens = borda_df[borda_df['patient_id'] == pid].iloc[0]
    short = pid.replace('Patient-', '')
    print(f"{short:<14} {e3t1}    {int(e4['top1_correct'])}    {int(e5['top1_correct'])}    "
          f"{int(ens['top1_correct'])}      {e3mae:<5} {int(e4['mae']):<5} {int(e5['mae']):<5} {int(ens['mae'])}")

# ── 3. Oracle analysis ─────────────────────────────────────────────────────────
print()
oracle45  = sum(1 for pid in pp4.index if pp4.loc[pid,'top1_correct'] or pp5.loc[pid,'top1_correct'])
oracle_all= sum(1 for pid in pp4.index
                if exp3_pp.get(pid,(0,0,0))[0] or pp4.loc[pid,'top1_correct'] or pp5.loc[pid,'top1_correct'])
print(f'Oracle E4+E5:      {oracle45}/26 = {oracle45/26*100:.1f}%')
print(f'Oracle E3+E4+E5:   {oracle_all}/26 = {oracle_all/26*100:.1f}%')

# ── 4. All-experiment comparison table ─────────────────────────────────────────
print()
compare = [
    ('Borda Ensemble E4+E5 (LOOCV)',   b_top1,  b_top3,  b_mae),
    ('Exp5A DINOv2-L ST (LOOCV)',       0.192, 0.615, 1.231),
    ('Exp4  DINOv2-B ST (LOOCV)',       0.192, 0.615, 1.731),
    ('Exp3  Siamese ENet-B0 (LOOCV)',   0.154, 0.538, 2.500),
    ('Exp2  ResNet18+ListNet (LOOCV)',  0.038, 0.462, 2.192),
    ('Center slice (naive)',            0.269, 0.577, 1.308),
    ('Random',                          0.080, 0.239, float('nan')),
]
print(f"{'Experiment':<40} {'Top-1':>6} {'Top-3':>6} {'MAE':>6}")
for name, t1, t3, mae in compare:
    mae_s = f"{mae:.3f}" if not (isinstance(mae, float) and np.isnan(mae)) else ' — '
    print(f"{name:<40} {t1*100:>5.1f}% {t3*100:>5.1f}% {mae_s:>6}")

# ── 5. Generate updated comparison bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
labels = [c[0] for c in compare]
top1s  = [c[1] for c in compare]
top3s  = [c[2] for c in compare]
x = np.arange(len(labels)); w = 0.35

b1 = ax.bar(x - w/2, top1s, width=w, label='Top-1', color=['#1565C0']*2 + ['#2196F3']*3 + ['#90CAF9']*2)
b2 = ax.bar(x + w/2, top3s, width=w, label='Top-3', color=['#E65100']*2 + ['#FF9800']*3 + ['#FFCC80']*2)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0.02:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax.axvspan(-0.5, 1.5, alpha=0.07, color='green')
ax.axhline(0.269, color='gray', ls='--', lw=1.2, label='Center baseline Top-1 (26.9%)')
ax.axhline(0.577, color='lightcoral', ls='--', lw=1.2, label='Center baseline Top-3 (57.7%)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8.5)
ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy')
ax.set_title('Best-slice selection: all experiments (LOOCV = honest metric)', fontsize=11)
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig('exp5_outputs/exp5_full_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved exp5_full_comparison.png')

# ── 6. Per-patient heatmap across all experiments ─────────────────────────────
patients = sorted(pp4.index)
data = {'Exp3': [], 'Exp4': [], 'Exp5': [], 'Ensemble': []}
for pid in patients:
    data['Exp3'].append(exp3_pp.get(pid,(0,0,0))[0])
    data['Exp4'].append(int(pp4.loc[pid,'top1_correct']))
    data['Exp5'].append(int(pp5.loc[pid,'top1_correct']))
    data['Ensemble'].append(int(borda_df[borda_df['patient_id']==pid].iloc[0]['top1_correct']))

mat = np.array([data['Exp3'], data['Exp4'], data['Exp5'], data['Ensemble']])
fig, ax = plt.subplots(figsize=(14, 3.5))
im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_yticks(range(4)); ax.set_yticklabels(['Exp3','Exp4','Exp5','Ensemble'])
ax.set_xticks(range(len(patients)))
ax.set_xticklabels([p.replace('Patient-','') for p in patients], rotation=45, ha='right', fontsize=8)
ax.set_title('Top-1 Correct per Patient across Experiments (green=correct, red=wrong)')
for i in range(4):
    for j in range(len(patients)):
        ax.text(j, i, 'Y' if mat[i,j] else '', ha='center', va='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('exp5_outputs/exp5_per_patient_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved exp5_per_patient_heatmap.png')

# ── 7. MAE progression ────────────────────────────────────────────────────────
mae4 = [int(pp4.loc[pid,'mae']) for pid in patients]
mae5 = [int(pp5.loc[pid,'mae']) for pid in patients]
mae_e = [int(borda_df[borda_df['patient_id']==pid].iloc[0]['mae']) for pid in patients]

fig, ax = plt.subplots(figsize=(14, 4))
x = np.arange(len(patients))
ax.bar(x - 0.25, mae4, width=0.25, label=f'Exp4 (mean={np.mean(mae4):.2f})', color='#2196F3', alpha=0.85)
ax.bar(x,        mae5, width=0.25, label=f'Exp5 (mean={np.mean(mae5):.2f})', color='#4CAF50', alpha=0.85)
ax.bar(x + 0.25, mae_e, width=0.25, label=f'Ensemble (mean={np.mean(mae_e):.2f})', color='#FF9800', alpha=0.85)
ax.axhline(np.mean(mae4), color='#1565C0', ls='--', lw=1)
ax.axhline(np.mean(mae5), color='#2E7D32', ls='--', lw=1)
ax.axhline(1.308, color='gray', ls=':', lw=1.2, label='Center baseline MAE=1.31')
ax.set_xticks(x)
ax.set_xticklabels([p.replace('Patient-','') for p in patients], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('MAE (slice units)')
ax.set_title('Per-patient MAE: Exp4 vs Exp5 vs Ensemble')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('exp5_outputs/exp5_mae_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved exp5_mae_progression.png')

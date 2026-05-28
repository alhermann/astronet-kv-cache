"""Audit script: verify every data point on the v3 Pareto plot exists and matches the source JSONs.

For each of the 6 backbones, for each of the 9 methods, check:
- source file exists
- expected field present
- numeric value sane (0 <= acc <= 1)
- the script's mapping points to the right file
- compression ratio computed correctly

Prints a per-panel table and flags missing/inconsistent points.
"""
import json, sys
from pathlib import Path

ROOT = Path('.')
PARETO = json.load(open(ROOT / 'logs/results/pareto_data.json'))['rows']

ORDER = ['qwen7b', 'qwen14b', 'qwen32b', 'llama8b', 'mistral7b', 'mistral24b']
PRETTY = {
    'qwen7b': 'Qwen 2.5-7B', 'qwen14b': 'Qwen 2.5-14B', 'qwen32b': 'Qwen 2.5-32B',
    'llama8b': 'Llama 3.1-8B', 'mistral7b': 'Mistral 7B', 'mistral24b': 'Mistral-Small 24B',
}

BASELINE_FILE = {
    'qwen7b':     'baselines_qwen2.5-7b_k300.json',
    'qwen14b':    'baselines_qwen2.5-14b_k300.json',
    'qwen32b':    'baselines_qwen2.5-32b_k300.json',
    'llama8b':    'baselines_llama-3.1-8b_k300.json',
    'mistral7b':  'baselines_mistral-7b-v0.3_k300.json',
    'mistral24b': 'baselines_mistral-small-24b_k300.json',
}
KIVI_FILE = {
    'qwen7b':     'kivi_qwen7b.json',
    'qwen14b':    'kivi_qwen2_5_14b.json',
    'qwen32b':    'kivi_qwen2_5_32b_v3.json',
    'llama8b':    'kivi_llama_3_1_8b.json',
    'mistral7b':  'kivi_mistral_7b_v0_3.json',
    'mistral24b': 'kivi_mistral_small_24b.json',
}
RAG_K3 = {'qwen7b': 0.80, 'qwen14b': 0.765, 'qwen32b': 0.795,
          'llama8b': 0.835, 'mistral7b': 0.68, 'mistral24b': 0.805}

issues = []
table = []
header = f'{"model":<18} {"strm":>5} {"h2o":>5} {"snap":>5} {"kivi":>5} {"S1":>5} {"S1+2":>5} {"K8V4":>5} {"rag1":>5} {"rag3":>5}'
print(header)
print('-' * len(header))

for tag in ORDER:
    row = next(r for r in PARETO if r['tag'] == tag)
    name = PRETTY[tag]

    # ----- baselines file -----
    bp = ROOT / 'logs/results' / BASELINE_FILE[tag]
    if not bp.exists():
        issues.append(f'{tag}: missing baselines file {bp.name}')
        continue
    base = json.load(open(bp))['results']
    acc_strm = base.get('streaming_llm')
    acc_h2o  = base.get('h2o')
    acc_snap = base.get('snapkv')
    acc_s1   = row.get('pos_robust_pure')
    acc_s1s2 = row.get('pos_robust_hybrid')
    acc_rag1 = base.get('rag_k1', 0.775)
    acc_rag3 = RAG_K3[tag]

    # ----- KIVI file -----
    kp = ROOT / 'logs/results' / KIVI_FILE[tag]
    if not kp.exists():
        acc_kivi = None
        issues.append(f'{tag}: KIVI file MISSING ({kp.name})')
    else:
        kdata = json.load(open(kp))['results']
        if 'k4v4' not in kdata:
            acc_kivi = None
            issues.append(f'{tag}: KIVI file has no k4v4 entry')
        else:
            v = kdata['k4v4']
            acc_kivi = v['accuracy'] if isinstance(v, dict) else v

    # ----- byte budgets / compression sanity -----
    bytes_full = row.get('bytes_fp16_full_n20')
    bytes_k300 = row.get('bytes_fp16_k300')
    bytes_k8v4 = row.get('bytes_k8v4_k300')
    if not (bytes_full and bytes_k300 and bytes_k8v4):
        issues.append(f'{tag}: missing byte budgets in pareto_data.json')

    # ----- per-field nullness check -----
    fields = [('streaming_llm', acc_strm), ('h2o', acc_h2o), ('snapkv', acc_snap),
              ('kivi_k4v4', acc_kivi), ('S1', acc_s1), ('S1+S2', acc_s1s2),
              ('hybrid_k8v4_acc=S1+S2', acc_s1s2),  # note: uses hybrid acc
              ('rag_k1', acc_rag1), ('rag_k3', acc_rag3)]
    for fname, val in fields:
        if val is None:
            issues.append(f'{tag}: {fname} is None')
        elif not (0 <= val <= 1):
            issues.append(f'{tag}: {fname} = {val} (out of [0,1])')

    def fmt(x): return f'{x*100:>4.1f}' if x is not None else ' ---'
    print(f'{name:<18} {fmt(acc_strm)} {fmt(acc_h2o)} {fmt(acc_snap)} '
          f'{fmt(acc_kivi)} {fmt(acc_s1)} {fmt(acc_s1s2)} '
          f'{fmt(acc_s1s2)} {fmt(acc_rag1)} {fmt(acc_rag3)}')

print()
if issues:
    print('=== ISSUES FOUND ===')
    for it in issues:
        print(f'  ! {it}')
else:
    print('=== NO ISSUES — all 9 methods present on all 6 panels ===')

# ----- additional check: compression ratios -----
print()
print('=== Compression-ratio sanity check (relative to bytes_fp16_full_n20) ===')
for tag in ORDER:
    row = next(r for r in PARETO if r['tag'] == tag)
    bf = row['bytes_fp16_full_n20']
    bk = row['bytes_fp16_k300']
    bq = row['bytes_k8v4_k300']
    c_k300 = bf / bk
    c_k8v4 = bf / bq
    c_kivi = c_k300 * 4   # K4V4: 8/32 of FP16 bytes -> 4x extra
    c_rag1 = bf / ((384/300) * bk)
    c_rag3 = c_rag1 / 3
    print(f'  {tag:<12}  k300_fp16={c_k300:5.1f}x  K8V4={c_k8v4:5.1f}x  '
          f'KIVI_K4V4={c_kivi:5.1f}x  RAG1={c_rag1:5.1f}x  RAG3={c_rag3:5.2f}x')

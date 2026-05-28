"""Wrapper smoke test on all six backbones.

Runs the AstroNet wrapper end-to-end on each released checkpoint:
- loads the frozen 4-bit quantised backbone + AstroNet checkpoint
- runs `wrapper.answer(...)` on a fixed multi-window factoid prompt
- verifies the answer is non-empty and contains expected token (where checkable)
- reports KV-cache bytes vs full cache

Output: JSON with one entry per model.
"""
import sys, os, json, traceback, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from astronet.wrapper import AstroNetWrapper

MODELS = [
    {'tag': 'qwen2.5-7b', 'path': './models/qwen2.5-7b',
     'ckpt': './checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt',
     'attn_dim': 512, 'multi_gpu': False},
    {'tag': 'qwen2.5-14b', 'path': './models/qwen2.5-14b',
     'ckpt': './checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt',
     'attn_dim': 256, 'multi_gpu': False},
    {'tag': 'qwen2.5-32b', 'path': './models/qwen2.5-32b',
     'ckpt': './checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt',
     'attn_dim': 256, 'multi_gpu': True},
    {'tag': 'llama-3.1-8b', 'path': './models/llama-3.1-8b',
     'ckpt': './checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt',
     'attn_dim': 256, 'multi_gpu': False},
    {'tag': 'mistral-7b-v0.3', 'path': './models/mistral-7b-v0.3',
     'ckpt': './checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt',
     'attn_dim': 256, 'multi_gpu': False},
    {'tag': 'mistral-small-24b', 'path': './models/mistral-small-24b',
     'ckpt': './checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt',
     'attn_dim': 256, 'multi_gpu': True},
]

# Fixed prompt: 5 fact windows + question. Answer is buried in window 2.
WINDOWS = [
    "Marine biology is the scientific study of organisms in the sea, including "
    "their behaviour, growth, life cycles, and interactions with the environment.",
    "The secret access code for the vault is ALPHA-7829-OMEGA. It must be entered exactly "
    "as shown, with hyphens and capital letters, within thirty seconds.",
    "Renewable energy sources have grown rapidly. Solar capacity doubled between 2020 "
    "and 2024 due to falling panel costs and policy incentives.",
    "Classical music spans several centuries and includes composers such as Bach, "
    "Mozart, Beethoven, and Stravinsky.",
    "Modern architecture emphasises clean lines, open spaces, and the integration of "
    "natural light through large windows and glass facades.",
]
QUESTION = "What is the secret access code for the vault?"
EXPECTED_TOKEN = 'ALPHA-7829'

results = {}
for spec in MODELS:
    print(f"\n=== {spec['tag']} ===", flush=True)
    rec = {'tag': spec['tag'], 'ckpt': spec['ckpt'], 'attn_dim': spec['attn_dim']}
    try:
        device_map = 'auto' if spec['multi_gpu'] else {'': 'cuda:0'}
        t0 = time.time()
        wrapper = AstroNetWrapper.from_pretrained(
            spec['path'], astro_ckpt=spec['ckpt'],
            attn_dim=spec['attn_dim'], device_map=device_map,
        )
        rec['load_time_sec'] = round(time.time() - t0, 1)

        t0 = time.time()
        ans_hybrid = wrapper.answer(WINDOWS, QUESTION, k=300, method='hybrid', max_new_tokens=32)
        rec['answer_hybrid'] = ans_hybrid
        rec['hybrid_time_sec'] = round(time.time() - t0, 2)
        rec['hybrid_contains_answer'] = EXPECTED_TOKEN in ans_hybrid

        t0 = time.time()
        ans_mult = wrapper.answer(WINDOWS, QUESTION, k=300, method='mult', max_new_tokens=32)
        rec['answer_mult'] = ans_mult
        rec['mult_time_sec'] = round(time.time() - t0, 2)
        rec['mult_contains_answer'] = EXPECTED_TOKEN in ans_mult

        rec['cache_bytes_fp16_k300'] = wrapper.cache_bytes(k=300, dtype='fp16')
        rec['cache_bytes_k8v4_k300'] = wrapper.cache_bytes(k=300, dtype='k8v4')
        rec['cache_bytes_fp16_full_n20'] = wrapper.cache_bytes(k=20 * 384, dtype='fp16')

        rec['status'] = 'ok'
        print(f"  hybrid: {ans_hybrid!r}", flush=True)
        print(f"  mult:   {ans_mult!r}", flush=True)
        print(f"  contains '{EXPECTED_TOKEN}': hybrid={rec['hybrid_contains_answer']}, mult={rec['mult_contains_answer']}", flush=True)

        # Free GPU before next model
        import torch
        del wrapper
        torch.cuda.empty_cache()
    except Exception as e:
        rec['status'] = 'error'
        rec['error'] = str(e)
        rec['traceback'] = traceback.format_exc()
        print(f"  FAILED: {e}", flush=True)
    results[spec['tag']] = rec

os.makedirs('logs/results', exist_ok=True)
with open('logs/results/wrapper_smoke_v5.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
for tag, rec in results.items():
    status = rec.get('status', '?')
    if status == 'ok':
        h_ok = '✓' if rec.get('hybrid_contains_answer') else '✗'
        m_ok = '✓' if rec.get('mult_contains_answer') else '✗'
        print(f"  {tag:25s} {status} hybrid={h_ok} mult={m_ok} t={rec.get('hybrid_time_sec','?')}s")
    else:
        print(f"  {tag:25s} {status}: {rec.get('error','?')[:60]}")

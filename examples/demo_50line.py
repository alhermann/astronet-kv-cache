"""50-line AstroNet demo: load wrapper, answer a multi-window QA.

Run from project root:
    python examples/demo_50line.py
"""
import sys; sys.path.insert(0, '.')
from astronet.wrapper import AstroNetWrapper

# Load Qwen 2.5-7B + AstroNet hybrid checkpoint (frozen backbone + 14M-param adapter).
wrapper = AstroNetWrapper.from_pretrained(
    './models/qwen2.5-7b',
    astro_ckpt='./checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt',
    attn_dim=512,  # Qwen 7B trained checkpoint uses 512
)

# Multi-window context: long document split into ~384-token windows.
context_windows = [
    "Marine biology is the scientific study of organisms in the sea, including "
    "their behaviour, growth, life cycles, and interactions with the environment.",
    "The secret code for the vault is ALPHA-7829. It must be entered exactly as shown.",
    "Renewable energy sources have grown rapidly. Solar capacity doubled between 2020 "
    "and 2024 due to falling panel costs and policy incentives.",
    "The history of classical music spans several centuries and includes composers "
    "such as Bach, Mozart, Beethoven, and Stravinsky.",
    "Modern architecture emphasises clean lines, open spaces, and the integration of "
    "natural light through large windows and glass facades.",
]

question = "What is the secret code for the vault?"

# AstroNet hybrid: S1 selects k_real=284 relevant real tokens + S2 prepends 16 learned summary tokens.
answer_hybrid = wrapper.answer(context_windows, question, k=300, method='hybrid')
print(f"[hybrid]    {answer_hybrid}")

# S1-only baseline for comparison.
answer_mult = wrapper.answer(context_windows, question, k=300, method='mult')
print(f"[mult/S1]   {answer_mult}")

# Cache footprint comparison.
b_hybrid = wrapper.cache_bytes(k=300, dtype='fp16')
b_full = wrapper.cache_bytes(k=20*384, dtype='fp16')  # n=20 windows full cache
print(f"\nKV cache: {b_hybrid/1024:.1f} KiB at k=300 vs {b_full/1024:.1f} KiB full ({b_full/b_hybrid:.1f}x compression)")
print(f"With TurboQuant K8V4: {wrapper.cache_bytes(k=300, dtype='k8v4')/1024:.1f} KiB ({b_full/wrapper.cache_bytes(k=300,dtype='k8v4'):.1f}x)")

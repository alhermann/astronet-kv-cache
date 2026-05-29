"""Minimal AstroNet quickstart.

Loads a frozen 4-bit-quantised backbone with a pre-trained Stage 2 summary
module, processes a multi-window context, and answers a question against it.

Usage:
    python examples/quickstart.py \\
        --model_path Qwen/Qwen2.5-7B \\
        --checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astronet.wrapper import AstroNetWrapper


WINDOWS = [
    "Alice arrived in Vienna on Tuesday morning. She checked into the Hotel "
    "Sacher and went straight to the State Opera to pick up tickets.",
    "Bob, meanwhile, was already in Salzburg attending the music festival.",
    "On Wednesday Alice walked through the Belvedere gardens before her "
    "evening performance.",
    "Later that week Alice flew to Prague to meet her sister.",
]
QUESTION = "Which hotel did Alice stay at in Vienna?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True,
                   help='HuggingFace identifier or local path')
    p.add_argument('--checkpoint', required=True,
                   help='Path to the pre-trained Stage 2 module')
    p.add_argument('--k', type=int, default=300,
                   help='Total cache budget')
    p.add_argument('--method', default='hybrid',
                   choices=['hybrid', 'multiplicative', 'snapkv', 'h2o',
                            'streaming_llm'])
    args = p.parse_args()

    wrapper = AstroNetWrapper.from_pretrained(
        model_path_or_hf_id=args.model_path,
        astronet_checkpoint=args.checkpoint,
    )

    answer = wrapper.answer(
        windows=WINDOWS,
        question=QUESTION,
        k=args.k,
        method=args.method,
        max_new_tokens=32,
    )
    print(f'Question: {QUESTION}')
    print(f'Answer:   {answer}')


if __name__ == '__main__':
    main()

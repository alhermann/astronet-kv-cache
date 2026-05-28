"""Diverse multi-window QA dataset for AstroNet S2 training.

Combines SQuAD, HotpotQA, and NaturalQuestions into a unified multi-window
format. Each dataset contributes different reasoning patterns:
  - SQuAD: single-hop extractive QA over Wikipedia paragraphs
  - HotpotQA: multi-hop reasoning requiring information from multiple passages
  - NaturalQuestions: real user queries with Wikipedia answers

All samples are converted to CrossContextSample format compatible with
existing training loops.
"""
import random
from typing import List
from collections import defaultdict

from datasets import load_dataset as hf_load_dataset
from data.synthetic import CrossContextSample


# ---------------------------------------------------------------------------
# HotpotQA loading
# ---------------------------------------------------------------------------

def _load_hotpotqa(n_samples: int, seed: int = 42):
    """Load HotpotQA distractor setting (train split).

    Each sample has a question, answer, and supporting_facts pointing to
    specific paragraphs across multiple documents.
    """
    ds = hf_load_dataset("hotpot_qa", "distractor", split="train")
    all_data = list(ds)
    random.seed(seed)
    random.shuffle(all_data)

    good = []
    for ex in all_data:
        if not ex['answer'] or ex['answer'] == 'yes' or ex['answer'] == 'no':
            continue  # skip yes/no questions
        ans_words = len(ex['answer'].split())
        if ans_words > 5:
            continue  # skip long answers
        good.append(ex)
        if len(good) >= n_samples * 3:  # collect extra for distractors
            break
    return good


def _hotpotqa_to_samples(examples: List[dict], n_samples: int, n_windows: int = 5,
                          seed: int = 42, include_answer: bool = False):
    """Convert HotpotQA examples to multi-window CrossContextSample format.

    For HotpotQA, we place the supporting paragraphs in early windows and
    distractor paragraphs in between, then pose the question in the final window.
    """
    random.seed(seed + 1000)  # different seed from main
    samples = []

    for i, ex in enumerate(examples[:n_samples]):
        question = ex['question']
        answer = ex['answer']

        # Collect all paragraphs: titles and sentences
        contexts = []
        for title, sents in zip(ex['context']['title'], ex['context']['sentences']):
            para = ' '.join(sents)
            if len(para.split()) >= 20:  # skip very short
                contexts.append(para)

        if len(contexts) < 2:
            continue

        # Supporting facts tell us which paragraphs matter
        # Place fact paragraphs first, then distractors
        sup_titles = set(ex['supporting_facts']['title'])
        fact_paras = []
        dist_paras = []
        for title, sents in zip(ex['context']['title'], ex['context']['sentences']):
            para = ' '.join(sents)
            if len(para.split()) < 20:
                continue
            if title in sup_titles:
                fact_paras.append(para)
            else:
                dist_paras.append(para)

        if not fact_paras:
            continue

        # Build windows: fact paragraphs first, then fill with distractors
        windows = []
        for fp in fact_paras[:2]:  # max 2 fact paragraphs
            windows.append(fp)

        # Fill remaining intermediate windows with distractors
        while len(windows) < n_windows - 1:
            if dist_paras:
                windows.append(dist_paras.pop(0))
            else:
                break

        # Query window
        if include_answer:
            query = f"Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer: {answer}"
        else:
            query = f"Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer:"
        windows.append(query)

        samples.append(CrossContextSample(
            windows=windows,
            fact=fact_paras[0],
            question=question,
            answer=answer,
            fact_window=0,
            query_window=len(windows) - 1,
            distance=len(windows) - 1,
            template_idx=-2,  # HotpotQA marker
        ))

    return samples


# ---------------------------------------------------------------------------
# NaturalQuestions loading
# ---------------------------------------------------------------------------

def _load_nq(n_samples: int, seed: int = 42):
    """Load NaturalQuestions (simplified, train split).

    Uses the simplified version which has short_answers extracted.
    """
    ds = hf_load_dataset("natural_questions", "default", split="train",
                          streaming=True)

    random.seed(seed + 2000)
    good = []

    for ex in ds:
        # NQ has complex annotation structure
        annotations = ex.get('annotations', {})
        if not annotations:
            continue

        short_answers = annotations.get('short_answers', [])
        if not short_answers or not short_answers[0].get('text', []):
            continue

        answer_text = short_answers[0]['text'][0] if short_answers[0]['text'] else ''
        if not answer_text or len(answer_text.split()) > 5:
            continue

        # Get document text
        doc_text = ex.get('document', {}).get('tokens', {}).get('token', [])
        if not doc_text:
            continue

        doc_str = ' '.join(doc_text[:500])  # first 500 tokens
        question = ex.get('question', {}).get('text', '')

        if len(doc_str.split()) < 50 or not question:
            continue

        good.append({
            'context': doc_str,
            'question': question,
            'answer': answer_text,
        })

        if len(good) >= n_samples * 2:
            break

    return good


# ---------------------------------------------------------------------------
# Unified diverse dataset
# ---------------------------------------------------------------------------

def generate_diverse_dataset(
    n_samples: int = 5000,
    n_windows: int = 5,
    seed: int = 42,
    include_answer: bool = False,
    sources: str = 'squad+hotpotqa',
) -> List[CrossContextSample]:
    """Generate diverse multi-window QA training samples.

    Args:
        n_samples: Total samples to generate.
        n_windows: Windows per sample.
        seed: Random seed.
        include_answer: If True, include answer in query window (for training).
        sources: Comma-separated or '+'-separated list of sources.
                 Options: 'squad', 'hotpotqa', 'nq'

    Returns:
        List of CrossContextSample.
    """
    source_list = [s.strip() for s in sources.replace('+', ',').split(',')]
    n_per_source = n_samples // len(source_list)
    remainder = n_samples - n_per_source * len(source_list)

    all_samples = []

    if 'squad' in source_list:
        from data.real_qa import generate_squad_dataset
        n_squad = n_per_source + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        print(f"Loading SQuAD train: {n_squad} samples...", flush=True)
        squad_samples = generate_squad_dataset(
            n_samples=n_squad, n_windows=n_windows, vary_distance=True,
            seed=seed, split='train', include_answer=include_answer
        )
        print(f"  Got {len(squad_samples)} SQuAD samples", flush=True)
        all_samples.extend(squad_samples)

    if 'hotpotqa' in source_list:
        n_hotpot = n_per_source + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        print(f"Loading HotpotQA train: {n_hotpot} samples...", flush=True)
        hotpot_raw = _load_hotpotqa(n_hotpot, seed=seed)
        hotpot_samples = _hotpotqa_to_samples(
            hotpot_raw, n_hotpot, n_windows=n_windows,
            seed=seed, include_answer=include_answer
        )
        print(f"  Got {len(hotpot_samples)} HotpotQA samples", flush=True)
        all_samples.extend(hotpot_samples)

    if 'nq' in source_list:
        n_nq = n_per_source + (1 if remainder > 0 else 0)
        print(f"Loading NaturalQuestions train: {n_nq} samples...", flush=True)
        nq_raw = _load_nq(n_nq, seed=seed)
        # Convert NQ to multi-window format (similar to SQuAD)
        from data.real_qa import generate_squad_dataset, _filter_squad, _load_squad
        # For NQ, we just use the loaded examples as fact windows with SQuAD distractors
        nq_ds = _load_squad('train')
        nq_filtered = _filter_squad(nq_ds)
        random.seed(seed + 3000)
        random.shuffle(nq_filtered)

        nq_samples = []
        for j, nq_ex in enumerate(nq_raw[:n_nq]):
            # Build windows: NQ context as fact, SQuAD contexts as distractors
            windows = [nq_ex['context']]
            for d in range(n_windows - 2):
                idx = (j * (n_windows - 2) + d) % len(nq_filtered)
                windows.append(nq_filtered[idx]['context'])

            if include_answer:
                query = f"Based on what you read earlier, answer the following question.\nQuestion: {nq_ex['question']}\nAnswer: {nq_ex['answer']}"
            else:
                query = f"Based on what you read earlier, answer the following question.\nQuestion: {nq_ex['question']}\nAnswer:"
            windows.append(query)

            nq_samples.append(CrossContextSample(
                windows=windows,
                fact=nq_ex['context'],
                question=nq_ex['question'],
                answer=nq_ex['answer'],
                fact_window=0,
                query_window=len(windows) - 1,
                distance=len(windows) - 1,
                template_idx=-3,  # NQ marker
            ))

        print(f"  Got {len(nq_samples)} NQ samples", flush=True)
        all_samples.extend(nq_samples)

    # Shuffle all samples together
    random.seed(seed)
    random.shuffle(all_samples)

    print(f"\nTotal diverse dataset: {len(all_samples)} samples "
          f"({', '.join(source_list)})", flush=True)

    return all_samples[:n_samples]

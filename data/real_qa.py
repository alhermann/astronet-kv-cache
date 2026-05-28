"""Real-data QA benchmark for cross-context memory evaluation.

Uses SQuAD v1.1 (extractive QA over Wikipedia paragraphs) to create
multi-window samples with real text.  Each sample places the answer-bearing
paragraph in an early window, fills intermediate windows with unrelated
Wikipedia paragraphs, and poses the question in the final window.

This is the strongest test of AstroNet's memory: the model must recall a
specific fact from a real paragraph seen several context windows ago.

Returns List[CrossContextSample] compatible with all existing eval/training
loops.

Usage:
    from data.real_qa import generate_squad_dataset
    samples = generate_squad_dataset(n_samples=200, n_windows=5, seed=42)
"""
import random
from typing import List, Optional
from collections import defaultdict

from datasets import load_dataset as hf_load_dataset
from data.synthetic import CrossContextSample


# ---------------------------------------------------------------------------
# SQuAD loading and filtering
# ---------------------------------------------------------------------------

_SQUAD_CACHE = None


def _load_squad(split: str = "validation"):
    """Load SQuAD and cache it.  Use validation split for eval (unseen)."""
    global _SQUAD_CACHE
    if _SQUAD_CACHE is not None and _SQUAD_CACHE[0] == split:
        return _SQUAD_CACHE[1]
    ds = hf_load_dataset("squad", split=split)
    _SQUAD_CACHE = (split, ds)
    return ds


def _filter_squad(ds, min_context_words=40, max_context_words=250,
                  max_answer_words=5):
    """Filter to samples with appropriately-sized contexts and short answers.

    Short answers are critical: our eval checks `answer.lower() in gen.lower()`
    so multi-word answers must be concise enough to be generated verbatim.
    """
    # Convert to list of dicts once (much faster than row-by-row iteration)
    all_data = ds.to_list() if hasattr(ds, 'to_list') else list(ds)
    good = []
    for ex in all_data:
        answers = ex["answers"]["text"]
        if not answers:
            continue
        answer = answers[0]
        ctx_words = len(ex["context"].split())
        ans_words = len(answer.split())
        if min_context_words <= ctx_words <= max_context_words and ans_words <= max_answer_words:
            good.append(ex)
    return good


# ---------------------------------------------------------------------------
# Multi-window sample construction
# ---------------------------------------------------------------------------

def _make_squad_sample(
    fact_example: dict,
    distractor_examples: List[dict],
    distance: int,
    n_windows: int,
    include_answer: bool = False,
) -> CrossContextSample:
    """Build one CrossContextSample from SQuAD examples.

    Layout (n_windows=5, distance=3):
        [fact_ctx] [distractor1] [distractor2] [query]
        window 0    window 1      window 2      window 3

    The fact window contains the full SQuAD context paragraph.
    Distractor windows contain unrelated SQuAD context paragraphs.
    The query window poses the question with "Answer:" marker.
    """
    answer = fact_example["answers"]["text"][0]
    question = fact_example["question"]
    fact_context = fact_example["context"]

    # Build windows
    windows = []

    # Window 0: fact context
    windows.append(fact_context)

    # Intermediate windows: distractor contexts
    n_distractors = distance - 1
    for i in range(n_distractors):
        if i < len(distractor_examples):
            windows.append(distractor_examples[i]["context"])
        else:
            # Fallback: repeat a distractor
            windows.append(distractor_examples[i % max(len(distractor_examples), 1)]["context"])

    # Query window
    if include_answer:
        query_text = (
            f"Based on what you read earlier, answer the following question.\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )
    else:
        query_text = (
            f"Based on what you read earlier, answer the following question.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    windows.append(query_text)

    # Pad to n_windows if needed (add more distractors before query)
    while len(windows) < n_windows:
        if not distractor_examples:
            break
        idx = (len(windows) - 2) % len(distractor_examples)
        windows.insert(-1, distractor_examples[idx]["context"])

    return CrossContextSample(
        windows=windows,
        fact=fact_context,
        question=question,
        answer=answer,
        fact_window=0,
        query_window=len(windows) - 1,
        distance=distance,
        template_idx=-1,  # not template-based
    )


def generate_squad_dataset(
    n_samples: int = 200,
    n_windows: int = 5,
    vary_distance: bool = True,
    seed: int = 42,
    split: str = "validation",
    min_context_words: int = 40,
    max_context_words: int = 250,
    max_answer_words: int = 5,
    include_answer: bool = False,
) -> List[CrossContextSample]:
    """Generate multi-window QA samples from SQuAD.

    Args:
        n_samples: Total number of samples to generate.
        n_windows: Maximum windows per sample.
        vary_distance: If True, balance distances 1..max_dist evenly.
        seed: Random seed.
        split: SQuAD split to use ('train' or 'validation').
        min_context_words: Minimum context paragraph length.
        max_context_words: Maximum context paragraph length.
        max_answer_words: Maximum answer length in words.

    Returns:
        List of CrossContextSample with real Wikipedia text.
    """
    random.seed(seed)

    ds = _load_squad(split)
    filtered = _filter_squad(ds, min_context_words, max_context_words,
                              max_answer_words)

    if len(filtered) < n_samples * 2:
        print(f"Warning: only {len(filtered)} filtered SQuAD examples "
              f"(need ~{n_samples * 2} for fact+distractors)")

    random.shuffle(filtered)

    max_dist = n_windows - 1

    # Balanced distances
    if vary_distance:
        distances = []
        per_dist = n_samples // max_dist
        for d in range(1, max_dist + 1):
            distances.extend([d] * per_dist)
        while len(distances) < n_samples:
            distances.append(random.randint(1, max_dist))
        random.shuffle(distances)
    else:
        distances = [max_dist] * n_samples

    # Group filtered examples by title to avoid same-article distractors
    by_title = defaultdict(list)
    for ex in filtered:
        by_title[ex["title"]].append(ex)
    titles = list(by_title.keys())

    samples = []
    used_idx = 0

    for i in range(min(n_samples, len(filtered))):
        fact_ex = filtered[i]
        fact_title = fact_ex["title"]
        distance = distances[i]

        # Gather distractor examples from different articles
        # Need enough to fill all intermediate windows (max n_windows - 2)
        n_distractors_needed = max(distance - 1, n_windows - 2)
        distractors = []
        attempts = 0
        while len(distractors) < n_distractors_needed and attempts < 100:
            t = random.choice(titles)
            if t != fact_title and by_title[t]:
                distractors.append(random.choice(by_title[t]))
            attempts += 1

        sample = _make_squad_sample(fact_ex, distractors, distance, n_windows,
                                    include_answer=include_answer)
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Convenience: generate train + eval splits
# ---------------------------------------------------------------------------

def generate_squad_train_eval(
    n_train: int = 500,
    n_eval: int = 100,
    n_windows: int = 5,
    seed: int = 42,
) -> tuple:
    """Generate separate train and eval sets from different SQuAD splits.

    Train uses SQuAD train split, eval uses validation split.
    This ensures zero overlap.
    """
    train_samples = generate_squad_dataset(
        n_samples=n_train, n_windows=n_windows,
        vary_distance=True, seed=seed, split="train",
        include_answer=True,
    )
    eval_samples = generate_squad_dataset(
        n_samples=n_eval, n_windows=n_windows,
        vary_distance=True, seed=seed + 1000, split="validation",
    )
    return train_samples, eval_samples


if __name__ == "__main__":
    # Quick test
    samples = generate_squad_dataset(n_samples=10, n_windows=5, seed=42)
    print(f"Generated {len(samples)} samples")
    for s in samples[:3]:
        print(f"\n  distance={s.distance}, answer='{s.answer}'")
        print(f"  question='{s.question}'")
        print(f"  n_windows={len(s.windows)}")
        for wi, w in enumerate(s.windows):
            print(f"    window {wi}: {len(w.split())} words — {w[:80]}...")

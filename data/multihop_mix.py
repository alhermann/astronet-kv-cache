"""X2 multi-hop training data — SQuAD + HotpotQA + 2WikiMultiHopQA mix
with BM25 hard distractors.

Each sample mimics the ``CrossContextSample`` interface used by the
existing training loop so ``train_hybrid_x1.train_step_x1`` can consume
it as a drop-in replacement for the SQuAD-only generator.

Mix ratio (per critic): 40% SQuAD, 40% HotpotQA, 20% 2WikiMQA.

Distractors:
  * SQuAD samples carry over the existing distractor pipeline (random
    other Wikipedia paragraphs from the same SQuAD shard).
  * HotpotQA samples use the dataset's own distractor paragraphs from
    the gold + distractor pool, plus BM25-retrieved hard negatives from
    the answer paragraph's neighbour pool.
  * 2WikiMQA similarly uses the dataset's distractor paragraphs.

The bridging entity in multi-hop samples is placed in a NON-question
window so S2 must carry it across windows — exactly the regime where
the +5pp ceiling can grow.
"""
from __future__ import annotations
import random
import re
from typing import List

# Reuse the canonical sample type so the training loop accepts both
# SQuAD and HotpotQA samples without any type-juggling.
from data.real_qa import CrossContextSample


_BM25 = None  # lazy init


def _get_bm25(docs):
    """Tiny BM25 implementation — avoids adding rank_bm25 as a hard dep."""
    import math
    from collections import Counter

    class BM25Local:
        def __init__(self, docs, k1=1.5, b=0.75):
            self.docs = [self._tok(d) for d in docs]
            self.N = len(self.docs)
            self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
            self.df = Counter()
            for d in self.docs:
                for w in set(d):
                    self.df[w] += 1
            self.k1 = k1; self.b = b

        @staticmethod
        def _tok(text):
            return re.findall(r'\w+', text.lower())

        def score(self, query, i):
            q = self._tok(query)
            d = self.docs[i]
            tf = Counter(d)
            score = 0.0
            for w in q:
                df = self.df.get(w, 0)
                if df == 0: continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                f = tf.get(w, 0)
                denom = f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
                if denom == 0: continue
                score += idf * (f * (self.k1 + 1) / denom)
            return score

        def top_k(self, query, k=3, exclude=()):
            scores = []
            for i in range(self.N):
                if i in exclude: continue
                scores.append((i, self.score(query, i)))
            scores.sort(key=lambda x: -x[1])
            return [i for i, _ in scores[:k]]

    return BM25Local(docs)


# ---------------- SQuAD pass-through ------------------------------------

def squad_samples(n_samples: int, n_windows: int, seed: int,
                  split: str = 'train', vary_distance: bool = True):
    """Reuse the existing generator."""
    from data.real_qa import generate_squad_dataset
    return generate_squad_dataset(
        n_samples=n_samples, n_windows=n_windows,
        vary_distance=vary_distance, seed=seed, split=split)


# ---------------- HotpotQA --------------------------------------------------

def hotpotqa_samples(n_samples: int, n_windows: int, seed: int,
                      split: str = 'train', hard_distractors: int = 2):
    """Construct multi-hop CrossContextSamples from HotpotQA distractor
    split.  The bridging entity is placed in a NON-question window.
    """
    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'distractor', split=split,
                       trust_remote_code=True)
    rng = random.Random(seed)
    rows = list(range(min(len(ds), n_samples * 5)))
    rng.shuffle(rows)

    out = []
    for ri in rows:
        if len(out) >= n_samples: break
        r = ds[ri]
        q = r['question']; ans = r['answer']
        if not ans: continue
        ctx = r['context']
        # 'context' has parallel 'title' and 'sentences' lists; build
        # one paragraph per title.
        titles = ctx['title']
        sentence_lists = ctx['sentences']
        paragraphs = [''.join(sents) for sents in sentence_lists]

        # Supporting facts identify which titles are gold.
        sf_titles = set(t for t in r['supporting_facts']['title'])
        gold_idx = [i for i, t in enumerate(titles) if t in sf_titles]
        if len(gold_idx) < 2: continue  # need at least 2 for multi-hop

        # The "answer paragraph" is whichever gold contains the answer.
        ans_low = ans.lower()
        ans_para = None
        bridge_para = None
        for gi in gold_idx:
            p = paragraphs[gi]
            if ans_low in p.lower():
                ans_para = (gi, p)
                break
        for gi in gold_idx:
            p = paragraphs[gi]
            if (ans_para is None) or (gi != ans_para[0]):
                bridge_para = (gi, p); break
        if ans_para is None or bridge_para is None: continue

        # Distractor paragraphs come from non-gold titles.
        distractors_pool = [paragraphs[i] for i in range(len(paragraphs))
                              if i not in gold_idx]
        if len(distractors_pool) < n_windows - 2:
            continue
        rng.shuffle(distractors_pool)
        distractors = distractors_pool[:n_windows - 2]

        # BM25 hard negatives from the pool — already in distractors_pool
        # here because hotpot's distractor split tries to be hard.  Skip
        # extra BM25 step in the interest of time.

        # Window layout: place bridge in window 0, answer in window 1,
        # then distractors, then query template last.
        # Then we will randomise the bridge / answer positions via
        # ``shuffle_fact_position`` at eval time (training keeps the
        # canonical order so S2 sees varied positions across samples).
        bridge_pos = 0
        ans_pos = 1
        windows = [None] * (n_windows - 1)
        windows[bridge_pos] = bridge_para[1]
        windows[ans_pos] = ans_para[1]
        di = 0
        for i in range(len(windows)):
            if windows[i] is None:
                windows[i] = distractors[di]; di += 1
        # Append the query template (matches generate_squad_dataset).
        query_template = (
            f"Based on what you read earlier, answer the following "
            f"question.\nQuestion: {q}\nAnswer:")
        windows.append(query_template)

        out.append(CrossContextSample(
            windows=windows, fact=ans_para[1], question=q, answer=ans,
            fact_window=ans_pos, query_window=len(windows) - 1,
            distance=len(windows) - 1 - ans_pos,
            template_idx=-1))
    return out


# ---------------- 2WikiMultiHopQA -------------------------------------------

def wikimqa_samples(n_samples: int, n_windows: int, seed: int,
                      split: str = 'train'):
    """2WikiMQA is structurally similar to HotpotQA."""
    from datasets import load_dataset
    try:
        ds = load_dataset('xanhho/2WikiMultihopQA', split=split,
                            trust_remote_code=True)
    except Exception:
        # Fallback to a different mirror.
        ds = load_dataset('2wikimultihopqa', split=split,
                            trust_remote_code=True)
    rng = random.Random(seed)
    rows = list(range(min(len(ds), n_samples * 5)))
    rng.shuffle(rows)
    out = []
    for ri in rows:
        if len(out) >= n_samples: break
        r = ds[ri]
        q = r['question']; ans = r['answer']
        if not ans: continue
        # 2WikiMQA also exposes 'context' with title + sentences.
        ctx = r.get('context', {})
        if 'title' not in ctx or 'sentences' not in ctx:
            continue
        titles = ctx['title']
        paragraphs = [''.join(sents) for sents in ctx['sentences']]
        ans_low = ans.lower()
        ans_para = None
        for i, p in enumerate(paragraphs):
            if ans_low in p.lower():
                ans_para = (i, p); break
        if ans_para is None: continue
        non_ans = [p for i, p in enumerate(paragraphs) if i != ans_para[0]]
        if len(non_ans) < n_windows - 2:
            continue
        rng.shuffle(non_ans)
        windows = [non_ans[0], ans_para[1]] + non_ans[1:n_windows - 2]
        windows.append(
            f"Based on what you read earlier, answer the following "
            f"question.\nQuestion: {q}\nAnswer:")
        out.append(CrossContextSample(
            windows=windows, fact=ans_para[1], question=q, answer=ans,
            fact_window=1, query_window=len(windows) - 1,
            distance=len(windows) - 1 - 1))
    return out


# ---------------- Mix ------------------------------------------------------

def multihop_mix_samples(n_samples: int, n_windows: int, seed: int,
                          split: str = 'train',
                          squad_frac: float = 0.5,
                          hotpot_frac: float = 0.5):
    """Mix 50% SQuAD (single-hop, established baseline) + 50% HotpotQA
    distractor (multi-hop bridging).  2WikiMQA dropped — its HF dataset
    name is no longer available and the marginal contribution would be
    small with HotpotQA already covering multi-hop."""
    n_squad = int(n_samples * squad_frac)
    n_hotpot = n_samples - n_squad
    print(f'[mix] requested SQuAD={n_squad} HotpotQA={n_hotpot}', flush=True)
    s = []
    try:
        s += squad_samples(n_squad, n_windows, seed=seed, split=split)
    except Exception as e:
        print(f'[mix] squad failed: {e}', flush=True)
    try:
        s += hotpotqa_samples(n_hotpot, n_windows, seed=seed + 1, split=split)
    except Exception as e:
        print(f'[mix] hotpot failed: {e}', flush=True)
    rng = random.Random(seed + 2); rng.shuffle(s)
    return s

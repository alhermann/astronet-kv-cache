"""Long-context QA training data that matches LongBench evaluation format.

Creates training samples by:
1. Taking Wikipedia articles (from SQuAD source)
2. Chunking into 384-token windows (matching LongBench eval)
3. Using 10-15 windows per sample (matching LongBench document length)
4. Placing extractive QA at varying distances

This ensures training and eval distributions match.
"""
import random
from typing import List
from collections import defaultdict

from datasets import load_dataset as hf_load_dataset
from data.synthetic import CrossContextSample


def _load_squad_articles(split='train', min_article_tokens=2000):
    """Load SQuAD and group by article title to get longer documents.

    SQuAD has multiple QA pairs per Wikipedia article. We group by title
    to reconstruct longer article texts.
    """
    ds = hf_load_dataset("squad", split=split)

    # Group paragraphs by title
    articles = defaultdict(lambda: {'paragraphs': [], 'qas': []})
    seen_contexts = set()

    for ex in ds:
        title = ex['title']
        ctx = ex['context']
        # Deduplicate paragraphs within same article
        ctx_key = (title, ctx[:100])
        if ctx_key not in seen_contexts:
            seen_contexts.add(ctx_key)
            articles[title]['paragraphs'].append(ctx)

        # Keep QA pairs with short answers
        answers = ex['answers']['text']
        if answers and len(answers[0].split()) <= 5:
            articles[title]['qas'].append({
                'question': ex['question'],
                'answer': answers[0],
                'context': ctx,
            })

    # Filter to articles with enough text
    good_articles = {}
    for title, data in articles.items():
        full_text = ' '.join(data['paragraphs'])
        if len(full_text.split()) >= min_article_tokens // 1.3 and len(data['qas']) > 0:
            good_articles[title] = data

    return good_articles


def _chunk_text_by_tokens(text, tokenizer, chunk_size=384):
    """Chunk text into windows of chunk_size tokens, matching LongBench eval."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i:i + chunk_size]
        if len(chunk_ids) >= chunk_size // 4:  # skip very short trailing chunks
            chunks.append(tokenizer.decode(chunk_ids))
    return chunks


def generate_longctx_dataset(
    tokenizer,
    n_samples: int = 5000,
    n_windows: int = 12,
    chunk_size: int = 384,
    seed: int = 42,
    split: str = 'train',
    include_answer: bool = False,
) -> List[CrossContextSample]:
    """Generate long-context QA samples with variable window counts.

    When n_windows='variable' or a range like '5-15', randomly varies the
    number of windows per sample. This trains the memory module to handle
    any context length — principled, not benchmark-specific.

    Args:
        tokenizer: The model's tokenizer (for consistent chunking)
        n_samples: Number of samples to generate
        n_windows: Target windows per sample. Can be int, 'variable' (=5-15),
                   or 'min-max' string (e.g. '5-15')
        chunk_size: Tokens per window (384 = LongBench default)
        seed: Random seed
        split: SQuAD split
        include_answer: Include answer in query window (for training)
    """
    random.seed(seed)

    # Parse n_windows: int, 'variable', or 'min-max'
    if isinstance(n_windows, str):
        if n_windows == 'variable':
            win_min, win_max = 5, 15
        else:
            parts = n_windows.split('-')
            win_min, win_max = int(parts[0]), int(parts[1])
        variable_windows = True
    else:
        win_min = win_max = n_windows
        variable_windows = False

    articles = _load_squad_articles(split=split)
    titles = list(articles.keys())
    random.shuffle(titles)

    print(f"  Found {len(articles)} articles with sufficient length", flush=True)
    if variable_windows:
        print(f"  Variable windows: {win_min}-{win_max} per sample", flush=True)

    samples = []

    for title in titles:
        if len(samples) >= n_samples:
            break

        data = articles[title]
        full_text = ' '.join(data['paragraphs'])

        # Chunk into 384-token windows
        chunks = _chunk_text_by_tokens(full_text, tokenizer, chunk_size)

        if len(chunks) < 3:
            continue

        # For each QA pair in this article, create a sample
        for qa in data['qas']:
            if len(samples) >= n_samples:
                break

            answer = qa['answer']
            question = qa['question']
            fact_context = qa['context']

            # Find which chunk contains the answer
            answer_chunk_idx = None
            for ci, chunk in enumerate(chunks):
                if answer.lower() in chunk.lower():
                    answer_chunk_idx = ci
                    break

            if answer_chunk_idx is None:
                continue

            # Build windows: use chunks from this article
            # Place answer chunk, then fill with other chunks as context/distractors
            # Sample target window count for this sample
            sample_n_windows = random.randint(win_min, win_max) if variable_windows else win_min
            max_fact_windows = min(len(chunks), sample_n_windows - 1)

            if max_fact_windows < 3:
                continue

            # Select which chunks to use
            # Always include the answer chunk, then sample others
            available = list(range(len(chunks)))
            available.remove(answer_chunk_idx)
            random.shuffle(available)

            # Pick chunks: answer chunk at a random position, others around it
            n_fact = min(max_fact_windows, sample_n_windows - 1)
            other_chunks = available[:n_fact - 1]
            selected = [answer_chunk_idx] + other_chunks
            selected.sort()  # keep document order

            # Vary the distance: answer chunk position determines distance to query
            answer_pos_in_selected = selected.index(answer_chunk_idx)
            distance = n_fact - answer_pos_in_selected

            windows = [chunks[ci] for ci in selected]

            # Add distractor chunks from OTHER articles if needed
            if len(windows) < sample_n_windows - 1:
                other_titles = [t for t in titles if t != title]
                for ot in other_titles:
                    if len(windows) >= sample_n_windows - 1:
                        break
                    other_art = articles.get(ot)
                    if other_art:
                        other_text = ' '.join(other_art['paragraphs'])
                        other_chunks_list = _chunk_text_by_tokens(other_text, tokenizer, chunk_size)
                        if other_chunks_list:
                            windows.append(random.choice(other_chunks_list))

            # Query window
            if include_answer:
                query = (f"Based on what you read earlier, answer the following question.\n"
                        f"Question: {question}\nAnswer: {answer}")
            else:
                query = (f"Based on what you read earlier, answer the following question.\n"
                        f"Question: {question}\nAnswer:")
            windows.append(query)

            samples.append(CrossContextSample(
                windows=windows,
                fact=fact_context,
                question=question,
                answer=answer,
                fact_window=answer_pos_in_selected,
                query_window=len(windows) - 1,
                distance=distance,
                template_idx=-4,  # long-context marker
            ))

    random.shuffle(samples)
    print(f"  Generated {len(samples)} long-context samples "
          f"(avg {sum(len(s.windows) for s in samples)/max(len(samples),1):.1f} windows/sample)",
          flush=True)

    return samples[:n_samples]

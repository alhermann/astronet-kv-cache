"""LongBench full-context upper bound evaluation.

Processes each document in a single forward pass (no windowing, no KV selection).
This establishes the ceiling F1 for each model on LongBench, contextualizing
the multi-window results where all methods operate under aggressive compression.
"""
import sys; sys.path.insert(0, '.')
import os, json, argparse, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


TASKS = {
    'hotpotqa': {'max_gen': 32, 'metric': 'f1'},
    'multifieldqa_en': {'max_gen': 64, 'metric': 'f1'},
}

PROMPTS = {
    'hotpotqa': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    'multifieldqa_en': "Read the following text and answer briefly.\n\n{context}\n\nQuestion: {input}\nAnswer:",
}


def f1_score(prediction, ground_truths):
    """Token-level F1 between prediction and best ground truth."""
    def _f1(pred_tokens, truth_tokens):
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(truth_tokens)
        return 2 * prec * rec / (prec + rec)

    pred_tokens = prediction.lower().split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        gt_tokens = gt.lower().split()
        if gt_tokens:
            best = max(best, _f1(pred_tokens, gt_tokens))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--max_input_tokens', type=int, default=4096,
                        help='Max input tokens (truncate context to fit)')
    parser.add_argument('--tasks', nargs='+', default=['hotpotqa', 'multifieldqa_en'])
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    embed_device = model.get_input_embeddings().weight.device
    print(f"Model loaded on {embed_device}", flush=True)

    all_results = {}

    for task_name in args.tasks:
        print(f"\n{'='*50}\nTask: {task_name}\n{'='*50}", flush=True)

        ds = load_dataset('THUDM/LongBench', task_name, split='test')
        data = list(ds)[:args.max_samples]
        print(f"  {len(data)} samples", flush=True)

        task_cfg = TASKS[task_name]
        prompt_template = PROMPTS[task_name]
        scores = []

        for si, sample in enumerate(data):
            context = sample['context']
            question = sample['input']
            answers = json.loads(sample['answers']) if isinstance(sample['answers'], str) else sample['answers']

            # Build full prompt with context
            prompt = prompt_template.format(context=context, input=question)

            # Tokenize and truncate to max_input_tokens
            ids = tokenizer(prompt, return_tensors='pt', max_length=args.max_input_tokens,
                            truncation=True).to(embed_device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=ids['input_ids'],
                    max_new_tokens=task_cfg['max_gen'],
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                                    skip_special_tokens=True).strip()
            score = f1_score(gen, answers)
            scores.append(score)

            if (si + 1) % 25 == 0:
                mean_f1 = 100 * sum(scores) / len(scores)
                print(f"    {si+1}/{len(data)}: F1={mean_f1:.1f}", flush=True)

        mean_f1 = 100 * sum(scores) / len(scores)
        all_results[task_name] = mean_f1
        print(f"  full_context: F1={mean_f1:.1f}", flush=True)

    # Save
    save_path = f'logs/results/longbench_fullctx_{model_name}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'model': model_name,
            'method': 'full_context',
            'max_input_tokens': args.max_input_tokens,
            'max_samples': args.max_samples,
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)

    print(f"\n{'='*50}\nSUMMARY: {model_name} (full context, max {args.max_input_tokens} tokens)\n{'='*50}", flush=True)
    for task, f1 in all_results.items():
        print(f"  {task:20s}: {f1:.1f}", flush=True)


if __name__ == '__main__':
    main()

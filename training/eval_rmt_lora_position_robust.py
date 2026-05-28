"""Position-robust evaluation of the trained RMT-LoRA baseline against the
same multi-segment SQuAD protocol used for AstroNet S1+S2 in tab:squad_main.

Mirrors training/eval_hybrid_position_robust.py but uses the RMT-LoRA
recurrence: each non-query segment is forwarded as
    [read_mem, content_embeds, write_mem]
and the write-slot output becomes the next segment's read_mem. The query
segment is forwarded as [read_mem_final, content_query_no_answer] and the
answer is greedy-generated token-by-token from the final position.

Per the pre-flight methodology audit:
  - bnb 4-bit + bfloat16 compute + double-quant (matches training)
  - PEFT wrapper, model.eval(), gradient checkpointing OFF
  - read_mem = write_mem = mem_init.to(bf16) reset per sample
  - substring containment scoring: s.answer.lower() in gen.lower()
  - max_seg_tokens=256, max_new_tokens=20, greedy, stop on EOS
  - one seed per invocation; loop the script with --seed for 5-seed CIs
"""
import sys, os, json, argparse, torch, torch.nn as nn
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from data.real_qa import generate_squad_dataset, CrossContextSample
from training.eval_hybrid_position_robust import shuffle_fact_position


def forward_segment(model, embed_layer, content_ids, read_mem, write_mem):
    """RMT-lite forward of one non-query segment.

    Returns the write-slot hidden states (last n_mem positions of the
    final decoder layer), to be passed as read_mem of the next segment.
    """
    n_mem = read_mem.shape[1]
    content_embeds = embed_layer(content_ids)
    inputs_embeds = torch.cat([read_mem, content_embeds, write_mem], dim=1)
    out = model(inputs_embeds=inputs_embeds,
                output_hidden_states=True, use_cache=False)
    new_state = out.hidden_states[-1][:, -n_mem:, :]
    return new_state


def greedy_generate(model, tokenizer, read_mem, content_ids,
                    max_new_tokens=20):
    """Greedy decode after consuming [read_mem, query_content_no_answer].

    No write_mem on the query segment: write_mem positions in training
    sat AFTER the answer span, so at eval the autoregressive generation
    replaces them naturally.
    """
    embed_layer = model.get_input_embeddings()
    content_embeds = embed_layer(content_ids)
    inputs_embeds = torch.cat([read_mem, content_embeds], dim=1)
    out = model(inputs_embeds=inputs_embeds, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    generated = [next_tok.item()]
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos:
            break
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(next_tok.item())
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_rmt_lora(model, tokenizer, samples, mem_init, n_mem, device,
                      max_seg_tokens=256, max_new_tokens=20):
    embed_layer = model.get_input_embeddings()
    correct = 0
    for s in samples:
        # Reset recurrence state per sample.
        read_mem = mem_init.clone().to(device)
        write_mem = mem_init.clone().to(device)

        # Build query content (no answer) — the LAST window is the query.
        query_text = (
            f'Based on what you read earlier, answer the following '
            f'question.\nQuestion: {s.question}\nAnswer:'
        )

        # Forward each NON-query window with full segment recurrence.
        for wi, content in enumerate(s.windows):
            if wi == s.query_window:
                continue
            content_ids = tokenizer(
                content, return_tensors='pt',
                max_length=max_seg_tokens, truncation=True,
            )['input_ids'].to(device)
            with torch.no_grad():
                new_state = forward_segment(
                    model, embed_layer, content_ids, read_mem, write_mem,
                )
            read_mem = new_state
            write_mem = mem_init.clone().to(device)

        # Greedy-generate the answer from the query segment.
        query_ids = tokenizer(
            query_text, return_tensors='pt',
            max_length=max_seg_tokens, truncation=True,
        )['input_ids'].to(device)
        with torch.no_grad():
            gen = greedy_generate(
                model, tokenizer, read_mem, query_ids,
                max_new_tokens=max_new_tokens,
            )
        if s.answer.lower() in gen.lower():
            correct += 1
    return correct / max(len(samples), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--lora_dir', required=True)
    p.add_argument('--mem_ckpt', required=True,
                    help='*.pt containing {"mem_init": tensor(1,n_mem,d)}')
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--n_windows', type=int, default=5)
    p.add_argument('--max_seg_tokens', type=int, default=256)
    p.add_argument('--max_new_tokens', type=int, default=20)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', default=None)
    args = p.parse_args()

    model_name = os.path.basename(args.model_path)
    torch.manual_seed(args.seed)

    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match the training-time quantisation EXACTLY: bf16 compute + double-quant.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb,
        device_map={'': args.device},
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # Load mem_init (fp32 in checkpoint) and cast to bf16 at the boundary.
    state = torch.load(args.mem_ckpt, map_location=args.device,
                       weights_only=False)
    mem_init = state['mem_init'].to(torch.bfloat16)
    assert mem_init.shape[1] == args.n_mem, (
        f'mem_init n_mem mismatch: {mem_init.shape[1]} vs {args.n_mem}'
    )
    print(f'Loaded LoRA: {args.lora_dir}', flush=True)
    print(f'Loaded mem_init: {tuple(mem_init.shape)} dtype={mem_init.dtype}',
          flush=True)

    base_samples = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=args.n_windows,
        vary_distance=True, seed=args.seed, split='validation',
    )

    results = {}
    for pos in args.positions:
        samples = shuffle_fact_position(base_samples, pos)
        acc = evaluate_rmt_lora(
            model, tokenizer, samples, mem_init,
            n_mem=args.n_mem, device=args.device,
            max_seg_tokens=args.max_seg_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        results[f'pos_{pos}'] = {'acc': acc}
        print(f'  pos={pos}: acc={acc:.3f}', flush=True)

    avg = sum(v['acc'] for v in results.values()) / len(results)
    print(f'\nAverage across positions: acc={avg:.3f}', flush=True)

    save_path = args.save_path or (
        f'logs/results/'
        f'rmt_lora_pos_robust_{model_name.replace(".", "_")}_s{args.seed}.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'model': model_name, 'lora_dir': args.lora_dir,
            'mem_ckpt': args.mem_ckpt,
            'n_eval': args.n_eval, 'seed': args.seed,
            'n_mem': args.n_mem, 'n_windows': args.n_windows,
            'max_seg_tokens': args.max_seg_tokens,
            'max_new_tokens': args.max_new_tokens,
            'results': results,
            'average': {'acc': avg},
        }, f, indent=2)
    print(f'Saved to {save_path}', flush=True)


if __name__ == '__main__':
    main()

"""Evaluate multiplicative KV selection + TurboQuant compression.
Uses the actual TurboQuant algorithm (ICLR 2026): random rotation + Lloyd-Max quantization."""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset
from astronet.turboquant import TurboQuantMSE


def turboquant_compress(tensor, quantizer):
    """Apply TurboQuant MSE compression to a KV tensor.
    tensor: [batch, heads, seq, head_dim]
    Returns: dequantized tensor of same shape.
    """
    B, H, S, D = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # Normalize to unit norm (TurboQuant assumes unit-norm vectors)
    flat = tensor.reshape(-1, D).float()  # [B*H*S, D]
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B*H*S, 1]
    normalized = flat / norms

    # Quantize + dequantize
    quantizer_device = quantizer.Pi.device
    normalized = normalized.to(quantizer_device)
    reconstructed, _ = quantizer(normalized)
    reconstructed = reconstructed.to(device)

    # Denormalize
    result = (reconstructed * norms.to(device)).to(dtype)
    return result.reshape(B, H, S, D)


def multiplicative_turboquant_eval(model, tokenizer, samples, k_total=300, bits=3, n_sink=4):
    """Multiplicative KV selection + TurboQuant compression."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = model.config.hidden_size // nq
    qpk = nq // nkv
    device = model.get_input_embeddings().weight.device
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    # Create TurboQuant quantizer for head_dim
    print(f"  Creating TurboQuant MSE quantizer: d={hd}, bits={bits}", flush=True)
    t0 = time.time()
    quantizer = TurboQuantMSE(hd, bits, seed=42, device='cpu')
    print(f"  Codebook computed in {time.time()-t0:.1f}s", flush=True)

    correct = 0
    for si, s in enumerate(samples):
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384, truncation=True).to(device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0]
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:n_sink] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

        cross = torch.zeros(total, device=device)
        for li in inject:
            layer_device = all_kv[li][0][0].device
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(
                    Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                    K[hi].float().T) / math.sqrt(hd)
                cross += sc.sum(dim=(0, 1)).to(device)
        cross[:n_sink] = -1e9

        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:n_sink] = -1e9
        k = min(k_total, total)
        _, idx = mult.topk(k)
        idx = idx.sort().values

        # Build cache with TurboQuant-compressed KV pairs
        cache = DynamicCache()
        for li in range(nl):
            K = torch.cat(all_kv[li][0], dim=2)[:, :, idx, :]
            V = torch.cat(all_kv[li][1], dim=2)[:, :, idx, :]
            K_q = turboquant_compress(K, quantizer)
            V_q = turboquant_compress(V, quantizer)
            cache.update(K_q, V_q, li)

        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        input_ids = fq['input_ids']
        pos = torch.arange(k, k + input_ids.shape[1], device=device).unsqueeze(0)
        cur = input_ids
        cc = cache
        gen = []
        with torch.no_grad():
            for _ in range(20):
                o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item())
                cur = nxt
                pos = torch.tensor([[k + input_ids.shape[1] + len(gen) - 1]], device=device)
                if nxt[0, 0].item() == tokenizer.eos_token_id:
                    break
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if s.answer.lower() in text.lower():
            correct += 1

        if (si + 1) % 50 == 0:
            print(f'  TQ-{bits}bit {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)

    return correct / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--bits', nargs='+', type=int, default=[4, 3, 2],
                        help='TurboQuant bit widths to test')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    save_path = f'./logs/turboquant_{model_name}_k{args.k}.json'

    print(f"Loading: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Loaded {len(samples)} samples", flush=True)

    results = {}
    for bits in args.bits:
        print(f"\n{'='*50}\nMultiplicative k={args.k} + TurboQuant {bits}-bit\n{'='*50}", flush=True)
        t0 = time.time()
        acc = multiplicative_turboquant_eval(model, tokenizer, samples, k_total=args.k, bits=bits)
        elapsed = time.time() - t0
        print(f"  => {acc:.1%} ({elapsed:.0f}s)", flush=True)
        results[f'turboquant_{bits}bit'] = acc

    # Add baseline reference
    results['reference_fp16'] = 0.745  # from definitive eval
    results['reference_naive_8bit'] = 0.750
    results['reference_naive_4bit'] = 0.455

    save_data = {
        'model': model_name, 'n_samples': len(samples), 'seed': args.seed,
        'k': args.k, 'method': 'multiplicative + TurboQuant (ICLR 2026)',
        'results': results,
    }
    os.makedirs('logs', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)
    print(f"\n{'='*50}\nSUMMARY: {model_name} k={args.k}\n{'='*50}", flush=True)
    for key, acc in results.items():
        print(f"  {key:30s}: {acc:.1%}", flush=True)


if __name__ == '__main__':
    main()

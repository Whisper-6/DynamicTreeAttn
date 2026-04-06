"""Single-file benchmark runner for dense/tree forward/backward.

Usage examples:
1) Dense backward with original per-sequence path:
   python run.py --model /path/to/model --data /path/to/call.pt --run dense_backward --mb-tokens -1

2) Dense backward with FFD packed micro-batches ([1, T_total] + cu_seqlens):
   python run.py --model /path/to/model --data /path/to/call.pt --run dense_backward --mb-tokens 32768

3) Tree backward baseline:
   python run.py --model /path/to/model --data /path/to/call.pt --run tree_backward
"""

import torch
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine
from dense import (
    forward as _dense_forward,
    backward as _dense_backward,
    backward_packed as _dense_backward_packed,
)
import time

def get_time():
    torch.cuda.synchronize()
    return time.time()

def dense_forward(model, input_ids, use_tqdm: bool):

    forward_time = get_time()
    logprobs_list = _dense_forward(model, input_ids, use_tqdm=use_tqdm)
    loss = sum(-lp.mean().item() for lp in logprobs_list)
    forward_time = get_time() - forward_time

    stats = {
        "loss": loss,
        "time": forward_time,
        "n_sequences": len(input_ids),
        "n_tokens": sum(len(ids) for ids in input_ids)
    }

    return stats


def tree_forward(model, engine, input_ids, args):
    
    if engine is None:
        max_seq_len = max(len(ids) for ids in input_ids)
        engine = TreeTrainingEngine(
            model_config=model.config,
            device=model.device,
            dtype=args.dtype,
            max_seq_len=max_seq_len,
            forward_only=True
        )
    
    forward_time = get_time()

    trie = TokenTrie(input_ids)
    if args.permute == "random":
        trie.random_permute()
    elif args.permute == "idx":
        pass
    elif args.permute == "ours":
        trie.forward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {args.permute}")

    logprobs_list = engine.forward(model=model, token_trie=trie)

    loss = sum(-lp.mean().item() for lp in logprobs_list)
    forward_time = get_time() - forward_time

    stats = trie.get_stats(mode="forward")
    stats["loss"] = loss
    stats["time"] = forward_time

    return stats
    

def dense_backward(
    model,
    input_ids,
    attachs,
    loss_fn,
    act_ckpt: bool,
    use_tqdm,
    mb_tokens: int = -1,
):

    backward_time = get_time()
    if mb_tokens == -1:
        loss = _dense_backward(
            model, input_ids, attachs, loss_fn, act_ckpt, use_tqdm=use_tqdm
        )
    else:
        loss = _dense_backward_packed(
            model,
            input_ids,
            attachs,
            loss_fn,
            act_ckpt,
            use_tqdm=use_tqdm,
            mb_tokens=mb_tokens,
        )
    backward_time = get_time() - backward_time

    stats = {
        "loss": loss,
        "time": backward_time,
        "n_sequences": len(input_ids),
        "n_tokens": sum(len(ids) for ids in input_ids)
    }

    return stats

def tree_backward(model, engine, input_ids, attachs, loss_fn, args):
    
    if engine is None:
        max_seq_len = max(len(ids) for ids in input_ids)
        engine = TreeTrainingEngine(
            model_config=model.config,
            device=model.device,
            dtype=args.dtype,
            max_seq_len=max_seq_len
        )

    backward_time = get_time()
    trie = TokenTrie(input_ids, attachs)
    if args.permute == "random":
        trie.random_permute()
    elif args.permute == "idx":
        pass
    elif args.permute == "ours":
        trie.backward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {args.permute}")

    loss = engine.backward(
        model=model,
        token_trie=trie,
        loss_fn=loss_fn,
        block_size=args.block_size,
        cut_f1_tail=args.cut_f1_tail,
        profile=args.profile_tree_backward,
        profile_cuda_sync=not args.profile_no_cuda_sync,
    )
    backward_time = get_time() - backward_time

    stats = trie.get_stats(mode="backward", block_size=args.block_size)
    stats["loss"] = loss
    stats["time"] = backward_time
    if args.profile_tree_backward:
        stats["breakdown"] = engine.last_profile

    return stats


def _summarize_tree_breakdown(breakdown: dict, fallback_total: float) -> dict:
    total = breakdown.get("tree_backward_total_time", fallback_total)
    forward_graph = breakdown.get("pop_forward_graph_time", 0.0)
    autograd_backward = breakdown.get("pop_autograd_backward_time", 0.0)
    kv_cache_fill = breakdown.get("build_cache_time", 0.0)
    build_cache_calls = int(breakdown.get("build_cache_calls", 0))
    build_cache_tokens = int(breakdown.get("build_cache_tokens", 0))
    build_cache_lengths = breakdown.get("build_cache_lengths", [])
    build_cache_lens_count = len(build_cache_lengths)
    avg_build_cache_len = (
        float(build_cache_tokens) / build_cache_calls if build_cache_calls > 0 else 0.0
    )
    other = max(0.0, total - (forward_graph + autograd_backward + kv_cache_fill))
    return {
        "total": float(total),
        "forward_graph": float(forward_graph),
        "autograd_backward": float(autograd_backward),
        "kv_cache_fill": float(kv_cache_fill),
        "build_cache_calls": build_cache_calls,
        "build_cache_tokens": build_cache_tokens,
        "build_cache_lengths": build_cache_lengths,
        "build_cache_lens_count": build_cache_lens_count,
        "avg_build_cache_len": avg_build_cache_len,
        "other": float(other),
    }

# ---------------- Test ----------------

import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_DICT = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

ATTN_IMP_DICT = {
    "bf16": "flash_attention_3",
    "fp16": "flash_attention_3",
    "fp32": "sdpa",
}

def load_data(data_path: str, model_path: str):
    if data_path.endswith(".pt"):
        data = torch.load(data_path, map_location="cpu")
        input_ids = data
    elif data_path.endswith(".txt"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = []
        with open(data_path, "r") as f:
            for line in f:
                ids = tokenizer.encode(line.strip(), return_tensors="pt").squeeze(0)
                input_ids.append(ids)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")
    return input_ids

def loss_fn(logprob: torch.Tensor, entropy: torch.Tensor, attachment: dict):
    w_logprobs = attachment["w_logprobs"]
    w_entropy = attachment["w_entropy"]
    return w_logprobs * logprob.mean() + w_entropy * entropy.mean()

def save_gradients(model, path: str):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu()
        else:
            grads[name] = None
    torch.save(grads, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-imp", type=str, default="flash_attention_3",
                        choices=["flash_attention_3", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--run", type=str, required=True,
                        choices=["dense_forward", "tree_forward", "dense_backward", "tree_backward"])
    parser.add_argument("--grad-out", type=str, default=None)

    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--act-ckpt", type=bool, default=False, help="enable activation checkpointing")
    parser.add_argument("--mb-tokens", type=int, default=-1, help="dense backward micro-batch token cap; -1 keeps original per-sequence path")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", type=bool, default=True, help="enable cutting f1 tail")
    parser.add_argument("--leafization", type=bool, default=False, help="enable leafization")
    parser.add_argument("--warmup", action="store_true", help="run one warmup iteration before timed run")
    parser.add_argument("--warmup-nseq", type=int, default=16, help="sequence cap used by warmup")
    parser.add_argument(
        "--profile-tree-backward",
        action="store_true",
        help="collect detailed tree_backward time breakdown (adds measurement overhead)",
    )
    parser.add_argument(
        "--profile-no-cuda-sync",
        action="store_true",
        help="disable cuda synchronize around profile timers (less accurate, lower overhead)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="apply torch.compile to the model (reduce-overhead mode with dynamic shapes)",
    )

    args = parser.parse_args()
    if args.attn_imp is None:
        args.attn_imp = ATTN_IMP_DICT[args.dtype]
    args.dtype = DTYPE_DICT[args.dtype]
    run_name = args.run.replace('_', ' ').title()

    # -------- load data --------
    input_ids = load_data(args.data, args.model)

    if args.leafization:
        token_trie = TokenTrie(input_ids)
        input_ids = token_trie.inputs
    
    if args.run.endswith("backward"):
        attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1} for ids in input_ids]

    # -------- load model --------
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        attn_implementation=args.attn_imp,
        device_map="cuda",
    )

    if args.run.endswith("forward"):
        model.eval()
    else:
        model.train()

    if args.compile:
        model = torch.compile(model, dynamic=True)

    # -------- warmup --------
    if args.warmup:
        if args.run == "dense_forward":
            warmup_inputs = input_ids[: min(args.warmup_nseq, len(input_ids))]
            dense_forward(model, warmup_inputs, use_tqdm=False)
        elif args.run == "dense_backward":
            warmup_inputs = input_ids[: min(args.warmup_nseq, len(input_ids))]
            warmup_attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1} for _ in warmup_inputs]
            dense_backward(
                model,
                warmup_inputs,
                warmup_attachs,
                loss_fn,
                args.act_ckpt,
                use_tqdm=False,
                mb_tokens=args.mb_tokens,
            )
            model.zero_grad(set_to_none=True)
        elif args.run == "tree_forward":
            warmup_inputs = input_ids[: min(args.warmup_nseq, len(input_ids))]
            tree_forward(model, None, warmup_inputs, args)
        elif args.run == "tree_backward":
            warmup_inputs = input_ids[: min(args.warmup_nseq, len(input_ids))]
            warmup_attachs = attachs[: len(warmup_inputs)]
            tree_backward(model, None, warmup_inputs, warmup_attachs, loss_fn, args)
            model.zero_grad(set_to_none=True)

    # -------- run --------
    torch.cuda.reset_peak_memory_stats()

    if args.run == "dense_forward":
        stats = dense_forward(model, input_ids, use_tqdm=True)

    elif args.run == "dense_backward":
        stats = dense_backward(
            model,
            input_ids,
            attachs,
            loss_fn,
            args.act_ckpt,
            use_tqdm=True,
            mb_tokens=args.mb_tokens,
        )

    elif args.run == "tree_forward":
        stats = tree_forward(model, None, input_ids, args)

    elif args.run == "tree_backward":
        stats = tree_backward(model, None, input_ids, attachs, loss_fn, args)

    print(f"[{run_name}] Loss: {stats['loss']:.6f}")
    print(f"[{run_name}] Time: {stats['time']:.2f} s")
    print(f"[{run_name}] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    if args.run == "tree_backward" and args.profile_tree_backward:
        br = stats.get("breakdown", {})
        sm = _summarize_tree_breakdown(br, stats["time"])
        total = sm["total"] if sm["total"] > 0 else 1e-12
        print("[Tree Backward Breakdown]")
        print(f"  forward_graph:     {sm['forward_graph']:.6f} s ({sm['forward_graph'] / total * 100:.1f}%)")
        print(f"  autograd_backward: {sm['autograd_backward']:.6f} s ({sm['autograd_backward'] / total * 100:.1f}%)")
        print(f"  kv_cache_fill:     {sm['kv_cache_fill']:.6f} s ({sm['kv_cache_fill'] / total * 100:.1f}%)")
        print(f"  build_cache_calls: {sm['build_cache_calls']}")
        print(f"  build_cache_tokens:{sm['build_cache_tokens']}")
        print(f"  avg_cache_len:     {sm['avg_build_cache_len']:.2f}")
        print(f"  build_cache_lens_count: {sm['build_cache_lens_count']}")
        print(f"  build_cache_lens:  {sm['build_cache_lengths']}")
        print(f"  other:             {sm['other']:.6f} s ({sm['other'] / total * 100:.1f}%)")
        print(f"  total(profile):    {sm['total']:.6f} s")

    # -------- save gradients --------
    if args.run.endswith("backward") and args.grad_out is not None:
        if args.grad_out == "bash":
            # 输出前 10 个 model 参数的梯度模长
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"Param: {name}, Grad Norm: {grad_norm:.6f}")
                else:
                    print(f"Param: {name}, Grad is None")
                if i >= 9:
                    break
        else:
            save_gradients(model, args.grad_out)

"""
python run.py \
  --model /data/jiarui/dta/models/Qwen2.5-0.5B \
  --data /tmp/areal/amo_bench_rollout_dump/call_2.pt \
  --run tree_backward \
  --profile-tree-backward \
  --warmup

python run.py \
  --model /data/jiarui/dta/models/Qwen2.5-0.5B \
  --data /tmp/areal/amo_bench_rollout_dump/call_2.pt \
  --run dense_backward \
  --grad-out /tmp/tmp_dense.pt \
  --warmup \
  --mb-tokens 4096

python compare_grads.py \
    --baseline-grad grad/Qwen3-0.6B-DB-bf16.pt \
    --exp-grad grad/Qwen3-0.6B-TB-bf16.pt \
    --out grad/Qwen3-0.6B-TB-vs-DB-bf16.txt
"""
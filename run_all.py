import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tqdm
import json
import sys
import time

from run import dense_forward, tree_forward, dense_backward, tree_backward
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine

ATTACH = {
    "w_logprobs": -1.0,
    "w_entropy": 0.1
}

def loss_fn(logprob: torch.Tensor, entropy: torch.Tensor, attachment: dict):
    w_logprobs = attachment["w_logprobs"]
    w_entropy = attachment["w_entropy"]
    return w_logprobs * logprob.mean() + w_entropy * entropy.mean()

def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def cuda_peak_memory_gb():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)

def with_sample_peak_memory(fn):
    reset_cuda_peak_memory()
    stats = fn()
    stats["peak_memory_gb"] = cuda_peak_memory_gb()
    return stats

def run_one_sample(name, input_ids, fn, clear_grad=None):
    reset_cuda_peak_memory()
    start = time.monotonic()
    try:
        stats = fn()
        stats["status"] = "success"
    except Exception as exc:
        message = str(exc)
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower()
        stats = {
            "status": "oom" if is_oom else "error",
            "error_type": type(exc).__name__,
            "error_message": message[:1000],
            "elapsed_before_failure_s": time.monotonic() - start,
            "n_sequences": len(input_ids),
            "n_tokens": sum(len(ids) for ids in input_ids),
        }
    stats["peak_memory_gb"] = cuda_peak_memory_gb()
    stats["name"] = name
    if clear_grad is not None:
        try:
            clear_grad()
        except Exception as exc:
            stats["clear_grad_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats

def run_warmup(fn, clear_grad=None):
    try:
        fn()
    except Exception as exc:
        print(f"[warmup skipped] {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
    finally:
        if clear_grad is not None:
            try:
                clear_grad()
            except Exception as exc:
                print(f"[warmup clear_grad failed] {type(exc).__name__}: {str(exc)[:500]}", file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

def load_data(data_folder: str, num_shards: int=1, shard_rank: int=0):
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"shard_rank must be in [0, {num_shards}), got {shard_rank}")

    data_files = [os.path.join(data_folder, f)
                  for f in os.listdir(data_folder) if f.endswith(".pt")]

    datas = []
    for idx, file in enumerate(sorted(data_files)):
        if idx % num_shards != shard_rank:
            continue
        data = torch.load(file, map_location="cpu")
        name = os.path.basename(file)[:-3]
        datas.append((name, data))

    return datas


def run_dense_forward(model, datas, warmup: bool=True):

    if warmup:
        inputs = datas[0][1][:16]
        run_warmup(lambda: dense_forward(model, inputs, use_tqdm=False))

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        stats = run_one_sample(
            name,
            input_ids,
            lambda: dense_forward(model, input_ids, use_tqdm=False),
        )
        results.append(stats)
    
    return results

def run_tree_forward(model, datas, args, warmup: bool=True):
    
    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384, forward_only=True)

    if warmup:
        inputs = datas[0][1]
        run_warmup(lambda: tree_forward(model, engine, inputs, args))

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        stats = run_one_sample(
            name,
            input_ids,
            lambda: tree_forward(model, engine, input_ids, args),
        )
        results.append(stats)

    return results

def run_dense_backward(model, datas, loss_fn, act_ckpt, warmup: bool=True):

    if warmup:
        inputs = datas[0][1][:16]
        attachs = [ATTACH] * len(inputs)
        run_warmup(
            lambda: dense_backward(model, inputs, attachs, loss_fn, act_ckpt, use_tqdm=False),
            clear_grad=model.zero_grad,
        )

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = run_one_sample(
            name,
            input_ids,
            lambda: dense_backward(model, input_ids, attachs, loss_fn, act_ckpt, use_tqdm=False),
            clear_grad=model.zero_grad,
        )
        results.append(stats)

    return results

def run_tree_backward(model, datas, loss_fn, args, warmup: bool=True):

    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384)

    if warmup:
        inputs = datas[0][1]
        attachs = [ATTACH] * len(inputs)
        run_warmup(
            lambda: tree_backward(model, engine, inputs, attachs, loss_fn, args),
            clear_grad=model.zero_grad,
        )

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = run_one_sample(
            name,
            input_ids,
            lambda: tree_backward(model, engine, input_ids, attachs, loss_fn, args),
            clear_grad=model.zero_grad,
        )
        results.append(stats)

    return results

def run_archon_dense_forward(engine, datas, args, warmup: bool=True):
    from archon_bench import archon_dense_forward

    if warmup:
        run_warmup(
            lambda: archon_dense_forward(
                engine,
                datas[0][1][:16],
                max_tokens_per_mb=args.max_tokens_per_mb,
            )
        )

    results = []
    for name, input_ids in tqdm.tqdm(datas):
        stats = run_one_sample(
            name,
            input_ids,
            lambda: archon_dense_forward(
                engine,
                input_ids,
                max_tokens_per_mb=args.max_tokens_per_mb,
            ),
        )
        results.append(stats)
    return results

def run_archon_dense_backward(engine, datas, loss_fn, args, warmup: bool=True):
    from archon_bench import archon_dense_backward, zero_grad

    if warmup:
        inputs = datas[0][1][:16]
        attachs = [ATTACH] * len(inputs)
        run_warmup(
            lambda: archon_dense_backward(
                engine,
                inputs,
                attachs,
                loss_fn,
                max_tokens_per_mb=args.max_tokens_per_mb,
            ),
            clear_grad=lambda: zero_grad(engine),
        )

    results = []
    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = run_one_sample(
            name,
            input_ids,
            lambda: archon_dense_backward(
                engine,
                input_ids,
                attachs,
                loss_fn,
                max_tokens_per_mb=args.max_tokens_per_mb,
            ),
            clear_grad=lambda: zero_grad(engine),
        )
        results.append(stats)
    return results

def run_archon_sparse_forward(engine, datas, args, warmup: bool=True):
    from archon_bench import archon_sparse_forward

    if warmup:
        run_warmup(
            lambda: archon_sparse_forward(
                engine,
                datas[0][1],
                max_tokens_per_mb=args.max_tokens_per_mb,
                flex_block_size=args.flex_block_size,
            )
        )

    results = []
    for name, input_ids in tqdm.tqdm(datas):
        stats = run_one_sample(
            name,
            input_ids,
            lambda: archon_sparse_forward(
                engine,
                input_ids,
                max_tokens_per_mb=args.max_tokens_per_mb,
                flex_block_size=args.flex_block_size,
            ),
        )
        results.append(stats)
    return results

def run_archon_sparse_backward(engine, datas, loss_fn, args, warmup: bool=True):
    from archon_bench import archon_sparse_backward, zero_grad

    if warmup:
        inputs = datas[0][1]
        attachs = [ATTACH] * len(inputs)
        run_warmup(
            lambda: archon_sparse_backward(
                engine,
                inputs,
                attachs,
                loss_fn,
                max_tokens_per_mb=args.max_tokens_per_mb,
                block_size=args.block_size,
                flex_block_size=args.flex_block_size,
            ),
            clear_grad=lambda: zero_grad(engine),
        )

    results = []
    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = run_one_sample(
            name,
            input_ids,
            lambda: archon_sparse_backward(
                engine,
                input_ids,
                attachs,
                loss_fn,
                max_tokens_per_mb=args.max_tokens_per_mb,
                block_size=args.block_size,
                flex_block_size=args.flex_block_size,
            ),
            clear_grad=lambda: zero_grad(engine),
        )
        results.append(stats)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run", type=str, required=True,
                        choices=[
                            "dense_forward",
                            "tree_forward",
                            "dense_backward",
                            "tree_backward",
                            "archon_dense_forward",
                            "archon_dense_backward",
                            "archon_sparse_forward",
                            "archon_sparse_backward",
                        ])
    parser.add_argument("--stats-out", type=str, default=None)

    parser.add_argument("--attn-imp", type=str, default="flash_attention_3",
                        choices=["flash_attention_3", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-mb", type=int, default=16384)
    parser.add_argument("--act-ckpt", type=str2bool, default=False, help="enable activation checkpointing")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", type=str2bool, default=True, help="enable cutting f1 tail")
    parser.add_argument("--leafization", type=str2bool, default=False, help="enable leafization")
    parser.add_argument("--num-shards", type=int, default=1, help="number of data shards")
    parser.add_argument("--shard-rank", type=int, default=0, help="rank of this data shard")
    
    args = parser.parse_args()
    args.dtype = torch.bfloat16
    run_name = args.run.replace('_', ' ').title()

    # -------- load data --------
    datas = load_data(args.data, args.num_shards, args.shard_rank)

    if args.leafization:
        leafed_datas = []
        for name, input_ids in datas:
            token_trie = TokenTrie(input_ids)
            leafed_datas.append((name, token_trie.inputs))
        datas = leafed_datas

    if not datas:
        print(f"[{run_name}] shard {args.shard_rank}/{args.num_shards} has no samples")
        if args.stats_out is not None:
            with open(args.stats_out, "w"):
                pass
        sys.exit(0)

    # -------- load model --------
    archon_engine = None
    model = None
    if args.run.startswith("archon_"):
        from archon_bench import initialize_archon_engine, zero_grad

        archon_engine = initialize_archon_engine(
            args.model,
            dtype="bfloat16",
            sparse=args.run.startswith("archon_sparse"),
            max_tokens_per_mb=args.max_tokens_per_mb,
            flex_block_size=args.flex_block_size,
            act_ckpt=args.act_ckpt,
        )
        if args.run.endswith("forward"):
            archon_engine.eval()
        else:
            archon_engine.train()
            zero_grad(archon_engine)
    else:
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
    
    # -------- run --------
    if args.run == "dense_forward":
        results = run_dense_forward(model, datas)

    elif args.run == "dense_backward":
        results = run_dense_backward(model, datas, loss_fn, args.act_ckpt)

    elif args.run == "tree_forward":
        results = run_tree_forward(model, datas, args)

    elif args.run == "tree_backward":
        results = run_tree_backward(model, datas, loss_fn, args)

    elif args.run == "archon_dense_forward":
        results = run_archon_dense_forward(archon_engine, datas, args)

    elif args.run == "archon_dense_backward":
        results = run_archon_dense_backward(archon_engine, datas, loss_fn, args)

    elif args.run == "archon_sparse_forward":
        results = run_archon_sparse_forward(archon_engine, datas, args)

    elif args.run == "archon_sparse_backward":
        results = run_archon_sparse_backward(archon_engine, datas, loss_fn, args)

    successful_results = [stat for stat in results if stat.get("status") == "success"]
    total_tokens = sum(stat["n_tokens"] for stat in successful_results)
    total_time = sum(stat.get("time", 0.0) for stat in successful_results)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    peak_memory_gb = max(
        (stat.get("peak_memory_gb") for stat in results if stat.get("peak_memory_gb") is not None),
        default=cuda_peak_memory_gb(),
    )
    print(f"[{run_name}] Throughput: {throughput:.2f} tokens/s")
    if peak_memory_gb is not None:
        print(f"[{run_name}] Peak memory: {peak_memory_gb:.2f} GB")

    if archon_engine is not None:
        archon_engine.destroy()

    if args.stats_out is not None:
        with open(args.stats_out, "w") as f:
            for stat in results:
                f.write(json.dumps(stat) + "\n")

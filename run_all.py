"""Folder-level benchmark runner over multiple .pt files.

Usage examples:
1) Dense backward with original per-sequence path:
   python run_all.py --model /path/to/model --data /path/to/data_folder --run dense_backward --mb-tokens -1

2) Dense backward with FFD packed micro-batches ([1, T_total] + cu_seqlens):
   python run_all.py --model /path/to/model --data /path/to/data_folder --run dense_backward --mb-tokens 32768

3) Dense/Tree forward throughput sweep:
   python run_all.py --model /path/to/model --data /path/to/data_folder --run dense_forward
   python run_all.py --model /path/to/model --data /path/to/data_folder --run tree_forward
"""

import argparse
import fcntl
import torch
from transformers import AutoModelForCausalLM
import os
import tqdm
import json

from run import dense_forward, tree_forward, dense_backward, tree_backward
from tree_training_engine import TreeTrainingEngine
from token_trie import TokenTrie

ATTACH = {
    "w_logprobs": -1.0,
    "w_entropy": 0.1
}


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _recover_from_failure(model) -> None:
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_error_stat(name: str, exc: Exception) -> dict:
    return {
        "name": name,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "oom": _is_oom_error(exc),
        "n_tokens": 0,
        "time": 0.0,
    }


def _handle_warmup_failure(run_name: str, exc: Exception) -> None:
    print(f"[{run_name}] Warmup failed: {type(exc).__name__}: {exc}")


def loss_fn(logprob: torch.Tensor, entropy: torch.Tensor, attachment: dict):
    w_logprobs = attachment["w_logprobs"]
    w_entropy = attachment["w_entropy"]
    return w_logprobs * logprob.mean() + w_entropy * entropy.mean()

def load_data(data_folder: str):    
    data_files = [os.path.join(data_folder, f)
                  for f in os.listdir(data_folder) if f.endswith(".pt")]

    datas = []
    for file in sorted(data_files):
        data = torch.load(file, map_location="cpu")
        name = os.path.basename(file)[:-3]
        datas.append((name, data))

    return datas


def run_dense_forward(model, datas, warmup: bool=True, on_stat=None):
    if not datas:
        return []

    if warmup:
        inputs = datas[0][1][:16]
        try:
            dense_forward(model, inputs, use_tqdm=False)
        except Exception as exc:
            _handle_warmup_failure("Dense Forward", exc)
            _recover_from_failure(model)

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        try:
            stats = dense_forward(model, input_ids, use_tqdm=False)
            stats["name"] = name
            stats["status"] = "ok"
        except Exception as exc:
            stats = _make_error_stat(name, exc)
            _recover_from_failure(model)
        results.append(stats)
        if on_stat is not None:
            on_stat(stats)
    
    return results

def run_tree_forward(model, datas, args, warmup: bool=True, on_stat=None):
    if not datas:
        return []
    
    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384, forward_only=True)

    if warmup:
        inputs = datas[0][1]
        try:
            tree_forward(model, engine, inputs, args)
        except Exception as exc:
            _handle_warmup_failure("Tree Forward", exc)
            _recover_from_failure(model)

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        try:
            stats = tree_forward(model, engine, input_ids, args)
            stats["name"] = name
            stats["status"] = "ok"
        except Exception as exc:
            stats = _make_error_stat(name, exc)
            _recover_from_failure(model)
        results.append(stats)
        if on_stat is not None:
            on_stat(stats)

    return results

def run_dense_backward(
    model,
    datas,
    loss_fn,
    act_ckpt,
    mb_tokens: int = -1,
    act_ckpt_long_seq: bool = False,
    warmup: bool = True,
    on_stat=None,
):
    if not datas:
        return []

    if warmup:
        inputs = datas[0][1][:16]
        attachs = [ATTACH] * len(inputs)
        try:
            dense_backward(
                model,
                inputs,
                attachs,
                loss_fn,
                act_ckpt,
                use_tqdm=False,
                mb_tokens=mb_tokens,
                act_ckpt_long_seq=act_ckpt_long_seq,
            )
            model.zero_grad()
        except Exception as exc:
            _handle_warmup_failure("Dense Backward", exc)
            _recover_from_failure(model)

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        try:
            stats = dense_backward(
                model,
                input_ids,
                attachs,
                loss_fn,
                act_ckpt,
                use_tqdm=False,
                mb_tokens=mb_tokens,
                act_ckpt_long_seq=act_ckpt_long_seq,
            )
            stats["name"] = name
            stats["status"] = "ok"
        except Exception as exc:
            stats = _make_error_stat(name, exc)
            _recover_from_failure(model)
        results.append(stats)
        if on_stat is not None:
            on_stat(stats)

    return results

def run_tree_backward(model, datas, loss_fn, args, warmup: bool=True, on_stat=None):
    if not datas:
        return []

    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384)

    if warmup:
        inputs = datas[0][1]
        attachs = [ATTACH] * len(inputs)
        try:
            tree_backward(model, engine, inputs, attachs, loss_fn, args)
            model.zero_grad()
        except Exception as exc:
            _handle_warmup_failure("Tree Backward", exc)
            _recover_from_failure(model)

    results = []

    for name, input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        try:
            stats = tree_backward(model, engine, input_ids, attachs, loss_fn, args)
            stats["name"] = name
            stats["status"] = "ok"
        except Exception as exc:
            stats = _make_error_stat(name, exc)
            _recover_from_failure(model)
        results.append(stats)
        if on_stat is not None:
            on_stat(stats)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run", type=str, required=True,
                        choices=["dense_forward", "tree_forward", "dense_backward", "tree_backward"])
    parser.add_argument("--stats-out", type=str, default=None)

    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--act-ckpt", type=bool, default=False, help="enable activation checkpointing")
    parser.add_argument(
        "--act-ckpt-long-seq",
        action="store_true",
        help=(
            "dense backward only: when mb_tokens > 0, apply checkpointing only to "
            "sequences with length > mb_tokens; sequences <= mb_tokens are packed and run "
            "without checkpointing"
        ),
    )
    parser.add_argument("--mb-tokens", type=int, default=-1, help="dense backward micro-batch token cap; -1 keeps original per-sequence path")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", type=bool, default=True, help="enable cutting f1 tail")
    parser.add_argument("--leafization", type=bool, default=False, help="enable leafization")
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
        "--torchrun",
        action="store_true",
        help="enable torchrun sharding using RANK/WORLD_SIZE/LOCAL_RANK",
    )
    
    args = parser.parse_args()
    args.dtype = torch.bfloat16
    run_name = args.run.replace('_', ' ').title()
    if args.act_ckpt and args.act_ckpt_long_seq:
        parser.error("--act-ckpt and --act-ckpt-long-seq are mutually exclusive.")
    if args.act_ckpt_long_seq and args.run != "dense_backward":
        parser.error("--act-ckpt-long-seq is only valid with --run dense_backward.")
    if args.act_ckpt_long_seq and args.mb_tokens <= 0:
        parser.error("--act-ckpt-long-seq requires --mb-tokens > 0.")

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_torchrun = args.torchrun or env_world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = env_world_size if use_torchrun else 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    dist = None
    dist_initialized = False
    if use_torchrun:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun mode requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            import torch.distributed as dist  # type: ignore[no-redef]

            dist.init_process_group(backend="nccl", init_method="env://")
            dist_initialized = True

    # -------- load data --------
    datas = load_data(args.data)
    if use_torchrun:
        datas = datas[rank::world_size]

    if args.leafization:
        for _, input_ids in datas:
            token_trie = TokenTrie(input_ids)
            input_ids = token_trie.inputs

    # -------- load model --------
    if use_torchrun:
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        attn_implementation="flash_attention_3",
        device_map=device_map,
    )

    if args.run.endswith("forward"):
        model.eval()
    else:
        model.train()

    stats_out_file = None
    stats_out_path = None
    if args.stats_out is not None:
        stats_out_path = args.stats_out
        # In distributed runs, all ranks append to the same file.
        # Rank 0 truncates once, then everyone opens with append mode.
        if dist_initialized and rank == 0:
            with open(stats_out_path, "w"):
                pass
        if dist_initialized:
            dist.barrier()
        stats_out_file = open(stats_out_path, "a")

    def emit_stat(stat: dict) -> None:
        if stats_out_file is not None:
            fcntl.flock(stats_out_file.fileno(), fcntl.LOCK_EX)
            try:
                stats_out_file.write(json.dumps(stat) + "\n")
                stats_out_file.flush()
            finally:
                fcntl.flock(stats_out_file.fileno(), fcntl.LOCK_UN)
    
    # -------- run --------
    if args.run == "dense_forward":
        results = run_dense_forward(model, datas, on_stat=emit_stat)

    elif args.run == "dense_backward":
        results = run_dense_backward(
            model,
            datas,
            loss_fn,
            args.act_ckpt,
            mb_tokens=args.mb_tokens,
            act_ckpt_long_seq=args.act_ckpt_long_seq,
            on_stat=emit_stat,
        )

    elif args.run == "tree_forward":
        results = run_tree_forward(model, datas, args, on_stat=emit_stat)

    elif args.run == "tree_backward":
        results = run_tree_backward(model, datas, loss_fn, args, on_stat=emit_stat)

    total_tokens = sum(stat["n_tokens"] for stat in results)
    total_time = sum(stat["time"] for stat in results)
    local_throughput = total_tokens / total_time if total_time > 0 else 0.0
    local_peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    if dist_initialized:
        total_tokens_tensor = torch.tensor(float(total_tokens), device=model.device)
        total_time_tensor = torch.tensor(float(total_time), device=model.device)
        peak_mem_tensor = torch.tensor(float(local_peak_mem), device=model.device)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_time_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(peak_mem_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            throughput = (
                total_tokens_tensor.item() / total_time_tensor.item()
                if total_time_tensor.item() > 0
                else 0.0
            )
            print(f"[{run_name}] Throughput: {throughput:.2f} tokens/s")
            print(f"[{run_name}] Peak memory (max across ranks): {peak_mem_tensor.item():.2f} GB")
    else:
        print(f"[{run_name}] Throughput: {local_throughput:.2f} tokens/s")
        print(f"[{run_name}] Peak memory: {local_peak_mem:.2f} GB")

    if stats_out_file is not None:
        stats_out_file.close()
    if dist_initialized:
        dist.destroy_process_group()
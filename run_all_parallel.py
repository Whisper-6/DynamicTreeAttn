import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


RUN_CHOICES = [
    "dense_forward",
    "tree_forward",
    "dense_backward",
    "tree_backward",
    "archon_dense_forward",
    "archon_dense_backward",
    "archon_sparse_forward",
    "archon_sparse_backward",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def visible_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or value.strip() == "":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set. "
            "Set it explicitly, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3."
        )
    return [device.strip() for device in value.split(",") if device.strip()]


def shard_stats_path(stats_out: str, rank: int) -> str:
    path = Path(stats_out)
    suffix = path.suffix
    if suffix:
        return str(path.with_name(f"{path.stem}.rank{rank}{suffix}"))
    return str(path.with_name(f"{path.name}.rank{rank}"))


def shard_log_path(stats_out: str, rank: int) -> str:
    path = Path(stats_out)
    suffix = path.suffix
    if suffix:
        return str(path.with_name(f"{path.stem}.rank{rank}.log"))
    return str(path.with_name(f"{path.name}.rank{rank}.log"))


def data_name_order(data_folder: str):
    names = [
        Path(file).stem
        for file in os.listdir(data_folder)
        if file.endswith(".pt")
    ]
    return {name: idx for idx, name in enumerate(sorted(names))}


def build_worker_cmd(args, num_shards: int, shard_rank: int, stats_path: str):
    worker_script = Path(__file__).resolve().with_name("run_all.py")
    cmd = [
        sys.executable,
        str(worker_script),
        "--model",
        args.model,
        "--data",
        args.data,
        "--run",
        args.run,
        "--stats-out",
        stats_path,
        "--attn-imp",
        args.attn_imp,
        "--block-size",
        str(args.block_size),
        "--flex-block-size",
        str(args.flex_block_size),
        "--max-tokens-per-mb",
        str(args.max_tokens_per_mb),
        "--act-ckpt",
        str(args.act_ckpt),
        "--permute",
        args.permute,
        "--cut-f1-tail",
        str(args.cut_f1_tail),
        "--leafization",
        str(args.leafization),
        "--num-shards",
        str(num_shards),
        "--shard-rank",
        str(shard_rank),
    ]
    return cmd


def load_shard_stats(stats_paths, data_folder: str):
    order = data_name_order(data_folder)
    stats = []
    seen = set()

    for rank, path in enumerate(stats_paths):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stat = json.loads(line)
                name = stat.get("name")
                if name in seen:
                    raise RuntimeError(f"duplicate sample in shard outputs: {name}")
                seen.add(name)
                stat["rank"] = rank
                stats.append(stat)

    stats.sort(key=lambda stat: order.get(stat.get("name"), len(order)))
    return stats

def brief_stats_path(stats_out: str) -> str:
    path = Path(stats_out)
    suffix = path.suffix
    if suffix:
        return str(path.with_name(f"{path.stem}.brief{suffix}"))
    return str(path.with_name(f"{path.name}.brief"))


def weighted_ratio(stats, tree_tokens_key: str) -> float | None:
    successful_stats = [stat for stat in stats if stat.get("status", "success") == "success"]
    total_tokens = sum(stat.get("n_tokens", 0) for stat in successful_stats)
    total_tree_tokens = sum(stat.get(tree_tokens_key, 0) for stat in successful_stats)
    if total_tokens <= 0 or total_tree_tokens <= 0:
        return None
    return total_tokens / total_tree_tokens


def normalize_sample_stat(stat):
    stat = dict(stat)
    stat["record_type"] = "sample"
    if "prepare_time" in stat:
        stat["prepare_time_s"] = stat.pop("prepare_time")
    if "compute_time" in stat:
        stat["compute_time_s"] = stat.pop("compute_time")
    if "time" in stat:
        stat["total_time_s"] = stat.pop("time")
    if "tree_token_ratio" in stat:
        stat["tree_compressed_ratio_mb"] = stat.pop("tree_token_ratio")
    if "compressed_ratio" in stat:
        stat["tree_compressed_ratio_mb"] = stat.pop("compressed_ratio")
    if "n_tokens" in stat and stat.get("total_time_s", 0.0) > 0:
        stat["throughput_tokens_s"] = stat["n_tokens"] / stat["total_time_s"]
    return stat


def write_stats_files(stats, stats_out: str, brief_out: str, args, devices, wall_time: float):
    failures = [stat for stat in stats if stat.get("status", "success") != "success"]
    successful_stats = [stat for stat in stats if stat.get("status", "success") == "success"]
    total_tokens = sum(stat["n_tokens"] for stat in successful_stats)
    prepare_total = sum(stat.get("prepare_time", 0.0) for stat in successful_stats)
    compute_total = sum(stat.get("compute_time", stat.get("time", 0.0)) for stat in successful_stats)
    successful_total_time = prepare_total + compute_total
    total_time = None if failures else successful_total_time
    peak_memory_gb = max(
        (stat.get("peak_memory_gb") for stat in stats if stat.get("peak_memory_gb") is not None),
        default=None,
    )

    run_config = {
        "record_type": "run_config",
        "run": args.run,
        "model": args.model,
        "data": args.data,
        "devices": devices,
        "attn_imp": args.attn_imp,
        "block_size": args.block_size,
        "flex_block_size": args.flex_block_size,
        "max_tokens_per_mb": args.max_tokens_per_mb,
        "act_ckpt": args.act_ckpt,
        "permute": args.permute,
        "cut_f1_tail": args.cut_f1_tail,
        "leafization": args.leafization,
    }
    run_summary = {
        "record_type": "run_summary",
        "run": args.run,
        "status": "partial_failure" if failures else "success",
        "samples": len(stats),
        "successful_samples": len(successful_stats),
        "failed_samples": len(failures),
        "n_tokens": total_tokens,
        "n_tree_tokens_original": sum(stat.get("n_tree_tokens_original", 0) for stat in successful_stats) or None,
        "n_tree_tokens": sum(stat.get("n_tree_tokens", 0) for stat in successful_stats) or None,
        "n_padded_tokens": sum(stat.get("n_padded_tokens", 0) for stat in successful_stats) or None,
        "n_micro_batches": sum(stat.get("n_micro_batches", 0) for stat in successful_stats) or None,
        "prepare_total_s": prepare_total,
        "compute_total_s": compute_total,
        "total_time_s": total_time,
        "throughput_tokens_s": (total_tokens / total_time) if total_time else None,
        "parallel_wall_time_s": wall_time,
        "parallel_throughput_tokens_s": (total_tokens / wall_time) if not failures and wall_time > 0 else None,
        "peak_memory_gb": peak_memory_gb,
        "tree_compressed_ratio_original": weighted_ratio(stats, "n_tree_tokens_original"),
        "tree_compressed_ratio_mb": weighted_ratio(stats, "n_tree_tokens"),
        "truncated": any(stat.get("truncated", False) for stat in stats),
        "n_truncated_samples": sum(int(stat.get("truncated", False)) for stat in stats),
        "n_truncated_sequences": sum(stat.get("n_truncated_sequences", 0) for stat in stats),
        "n_tokens_before_truncate": sum(stat.get("n_tokens_before_truncate", stat.get("n_tokens", 0)) for stat in stats),
        "n_tokens_after_truncate": sum(stat.get("n_tokens_after_truncate", stat.get("n_tokens", 0)) for stat in stats),
    }
    if run_summary["truncated"]:
        run_summary["warning"] = (
            f"truncated {run_summary['n_truncated_samples']} samples/"
            f"{run_summary['n_truncated_sequences']} sequences to max_tokens_per_mb="
            f"{args.max_tokens_per_mb}; tokens {run_summary['n_tokens_before_truncate']}"
            f"->{run_summary['n_tokens_after_truncate']}"
        )

    detailed_samples = [normalize_sample_stat(stat) for stat in stats]

    out_path = Path(stats_out)
    brief_path = Path(brief_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(json.dumps(run_config) + "\n")
        f.write(json.dumps(run_summary) + "\n")
        for stat in detailed_samples:
            f.write(json.dumps(stat) + "\n")

    brief_keys = [
        "record_type",
        "run",
        "status",
        "name",
        "rank",
        "samples",
        "successful_samples",
        "failed_samples",
        "n_tokens",
        "prepare_total_s",
        "compute_total_s",
        "prepare_time_s",
        "compute_time_s",
        "total_time_s",
        "throughput_tokens_s",
        "peak_memory_gb",
        "tree_compressed_ratio_original",
        "tree_compressed_ratio_mb",
        "warning",
        "error_type",
        "error_message",
    ]
    with open(brief_path, "w") as f:
        f.write(json.dumps({
            "record_type": "run_config",
            "run": args.run,
            "attn_imp": args.attn_imp,
            "max_tokens_per_mb": args.max_tokens_per_mb,
            "act_ckpt": args.act_ckpt,
            "block_size": args.block_size,
            "flex_block_size": args.flex_block_size,
        }) + "\n")
        for record in [run_summary, *detailed_samples]:
            f.write(json.dumps({key: record[key] for key in brief_keys if key in record}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run", type=str, required=True, choices=RUN_CHOICES)
    parser.add_argument("--stats-out", type=str, required=True)
    parser.add_argument("--brief-stats-out", type=str, default=None)

    parser.add_argument("--attn-imp", type=str, default="flash_attention_3",
                        choices=["flash_attention_3", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--max-tokens-per-mb", type=int, default=16384)
    parser.add_argument("--act-ckpt", type=str2bool, default=False, help="enable activation checkpointing")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", type=str2bool, default=True, help="enable cutting f1 tail")
    parser.add_argument("--leafization", type=str2bool, default=False, help="enable leafization")
    parser.add_argument("--keep-shard-stats", action="store_true", help="keep per-rank jsonl files after merge")

    args = parser.parse_args()

    devices = visible_devices()
    if not devices:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not contain any devices")

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    procs = []
    stats_paths = []
    log_files = []
    start = time.monotonic()

    for rank, device in enumerate(devices):
        stats_path = shard_stats_path(args.stats_out, rank)
        log_path = shard_log_path(args.stats_out, rank)
        stats_paths.append(stats_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["WORLD_SIZE"] = "1"
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = str(29500 + rank)

        cmd = build_worker_cmd(args, len(devices), rank, stats_path)
        log_file = open(log_path, "w")
        log_files.append(log_file)
        print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={device} log={log_path}")
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)))

    failed = []
    for rank, proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failed.append((rank, returncode, shard_log_path(args.stats_out, rank)))

    for log_file in log_files:
        log_file.close()

    if failed:
        for rank, returncode, log_path in failed:
            print(f"[rank {rank}] failed with exit code {returncode}; see {log_path}", file=sys.stderr)
        sys.exit(1)

    wall_time = time.monotonic() - start
    stats = load_shard_stats(stats_paths, args.data)
    brief_out = args.brief_stats_out or brief_stats_path(args.stats_out)
    write_stats_files(stats, args.stats_out, brief_out, args, devices, wall_time)
    total_tokens = sum(stat["n_tokens"] for stat in stats)
    print(f"[Parallel {args.run}] Workers: {len(devices)}")
    print(f"[Parallel {args.run}] Samples: {len(stats)}")
    print(f"[Parallel {args.run}] Wall time: {wall_time:.2f} s")
    print(f"[Parallel {args.run}] Throughput: {total_tokens / wall_time:.2f} tokens/s")
    print(f"[Parallel {args.run}] Stats written to {args.stats_out}")

    if not args.keep_shard_stats:
        for path in stats_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

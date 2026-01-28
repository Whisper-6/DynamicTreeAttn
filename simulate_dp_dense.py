import os
# 设置环境变量以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import torch
import torch.multiprocessing as mp
from interface import worker_process
import os
import glob


def partition_ffd(input_ids, world_size):
    """
    LPT / FFD-style greedy load balancing.
    Return: list[list[Tensor]]  (sequences per GPU)
    """
    lengths = [seq.shape[0] for seq in input_ids]
    sorted_seqs = [seq for _, seq in sorted(
        zip(lengths, input_ids),
        key=lambda x: x[0],
        reverse=True
    )]

    buckets = [[] for _ in range(world_size)]
    loads = [0] * world_size

    for seq in sorted_seqs:
        seq_len = seq.shape[0]
        gpu = loads.index(min(loads))
        buckets[gpu].append(seq)
        loads[gpu] += seq_len

    return buckets



def load_merged_data(data_path):
    files = sorted(glob.glob(os.path.join(data_path, "merged*.pt")))
    datas = []
    for f in files:
        datas.append(torch.load(f, map_location="cpu"))
    return datas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="output file for (length, time) pairs")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    datas = load_merged_data(args.data_path)
    print(f"Loaded {len(datas)} merged files\n")

    mp.set_start_method("spawn", force=True)

    # 每个 GPU 一个 task list
    gpu_task_lists = [[] for _ in range(args.world_size)]

    print("Building sequence-level tasks...")
    task_len_map = {}   # task_id -> seq_len

    for file_idx, input_ids in enumerate(datas):
        print(f"File {file_idx}: {len(input_ids)} sequences")

        buckets = partition_ffd(input_ids, args.world_size)

        for gpu_id, seqs in enumerate(buckets):
            for local_id, seq in enumerate(seqs):
                seq_len = seq.shape[0]   # ← 自行计算

                task_id = f"file{file_idx}_gpu{gpu_id}_seq{local_id}"

                task_kwargs = {}
                if args.mode.endswith("backward"):
                    task_kwargs["gradient_checkpointing"] = True
                
                task_len_map[task_id] = seq_len

                task = (
                    task_id,
                    args.mode,
                    [seq],      
                    task_kwargs,
                )

                gpu_task_lists[gpu_id].append(task)


    print("\nTask assignment:")
    for i, tasks in enumerate(gpu_task_lists):
        print(f"  GPU {i}: {len(tasks)} sequences")

    result_queue = mp.Queue()
    processes = []

    print("\nStarting workers...")
    for gpu_id in range(args.world_size):
        if not gpu_task_lists[gpu_id]:
            continue

        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                gpu_task_lists[gpu_id],
                args.model_path,
                "bf16",
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # 收集结果 (必须在 join 之前进行，否则如果 queue 数据量大可能导致死锁)
    print("Collecting results...")
    length_time_pairs = []
    
    # 每个启动的 process 都会返回一次结果
    for _ in range(len(processes)):
        gpu_id, results = result_queue.get()
        for r in results:
            tid = r["task_id"]
            t = r["time"]
            seq_len = task_len_map[tid]   # ← 从主进程表里拿
            length_time_pairs.append((seq_len, t))

    print("Waiting for processes to finish...")
    for p in processes:
        p.join()

    print("All processes finished")

    print(f"\nCollected {len(length_time_pairs)} sequence measurements")

    torch.save(length_time_pairs, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

"""
python simulate_dp_dense.py \
    --data_path /data/tree/DynamicTreeAttn/data \
    --world_size 2 \
    --mode tree_forward \
    --model_path /data/tree/models/Qwen3-0.6B \
    --output dense_forward.pt
""" 
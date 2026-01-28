import argparse
import time
import torch
import torch.multiprocessing as mp
from interface import worker_process
from token_trie import TokenTrie
import os
import glob


def dp(input_ids, K, pattern, mem_bound):
    """
    将 input_ids 分配到 K 个 bucket
    
    Args:
        input_ids: list of tensors
        K: bucket 数量
        pattern: 'ffd' (First Fit Decreasing) 或 'trie'
        mem_bound: memory bound (仅 trie 模式使用)
    
    Returns:
        list[list[int]]: K 个 bucket，每个 bucket 是序列索引列表
    """
    if pattern == "ffd":
        # First Fit Decreasing
        lst = [input_ids[i].shape[0] for i in range(len(input_ids))]
        id_list = [idx for idx, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)]
        
        bucket = [[] for _ in range(K)]
        total = [0] * K
        
        for id in id_list:
            at = total.index(min(total))
            bucket[at].append(id)
            total[at] += input_ids[id].shape[0]
        
        return bucket
    elif pattern == "trie_forward":
        # Trie-based partitioning
        trie = TokenTrie(input_ids)
        trie.forward_permute()
        return trie.divide(K, mem_bound)

    elif pattern == "trie_backward":
        # Trie-based partitioning
        trie = TokenTrie(input_ids)
        trie.backward_permute(reversed=True)
        return trie.divide(K, mem_bound)


def load_merged_data(data_path):
    """从 data_path 读取所有 merged*.pt 文件"""
    files = sorted(glob.glob(os.path.join(data_path, "merged*.pt")))
    datas = []
    for f in files:
        data = torch.load(f, map_location="cpu")
        datas.append(data)
    return datas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, help="DP模拟份数")
    parser.add_argument("--world_size", type=int, required=True, help="物理GPU数量")
    parser.add_argument("--pattern", type=str, required=True, help="ffd 或 trie")
    parser.add_argument("--mem_bound", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        help="dense_forward, tree_forward, dense_backward, tree_backward")
    args = parser.parse_args()
    
    # 读取所有数据
    print(f"Loading data from {args.data_path}...")
    datas = load_merged_data(args.data_path)
    print(f"Loaded {len(datas)} merged files\n")
    
    # 设置 multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # 对所有 pt 文件进行 dp 分配，生成所有任务
    all_tasks = []
    file_info = {}  # file_idx -> {tokens, task_ids}
    
    for file_idx, input_ids in enumerate(datas):
        original_tokens = sum(seq.shape[0] for seq in input_ids)
        
        print(f"File {file_idx}: {len(input_ids)} sequences, {original_tokens} tokens")
        
        # 用 dp 分成 K 份
        buckets = dp(input_ids, args.K, args.pattern, args.mem_bound)
        
        task_ids = []
        for k in range(args.K):
            bucket_seqs = [input_ids[i] for i in buckets[k]]
            if len(bucket_seqs) == 0:
                continue
            
            task_kwargs = {}
            if args.mode == "dense_backward":
                task_kwargs["gradient_checkpointing"] = True
            
            task_id = f"file{file_idx}_dp{k}"
            task = (task_id, args.mode, bucket_seqs, task_kwargs)
            all_tasks.append(task)
            task_ids.append(task_id)
        
        file_info[file_idx] = {
            "tokens": original_tokens,
            "task_ids": task_ids
        }
    
    print(f"\nTotal tasks: {len(all_tasks)}")
    
    # Round-robin 分配到 world_size 个物理 GPU
    gpu_task_lists = [[] for _ in range(args.world_size)]
    for i, task in enumerate(all_tasks):
        gpu_id = i % args.world_size
        gpu_task_lists[gpu_id].append(task)
    
    print("\nTask assignment:")
    for gpu_id in range(args.world_size):
        print(f"  GPU {gpu_id}: {len(gpu_task_lists[gpu_id])} tasks")
    
    # 启动 world_size 个 worker_process
    result_queue = mp.Queue()
    processes = []
    
    print("\nStarting workers...")
    for gpu_id in range(args.world_size):
        if len(gpu_task_lists[gpu_id]) == 0:
            continue
        
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_task_lists[gpu_id], args.model_path, "bf16", result_queue)
        )
        p.start()
        processes.append(p)
    
    # 收集结果 (必须在 join 之前，防止死锁)
    all_results = []
    # 每个启动的 process 都会返回一次结果
    for _ in range(len(processes)):
        gpu_id, results = result_queue.get()
        all_results.extend(results)

    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 按 task_id 建立索引
    task_time_map = {}
    for result in all_results:
        task_time_map[result['task_id']] = result['time']
    
    # 计算每个文件的最大时间和吞吐
    file_results = {}
    print("\n" + "=" * 60)
    print("RESULTS PER FILE")
    print("=" * 60)
    
    for file_idx in sorted(file_info.keys()):
        info = file_info[file_idx]
        tokens = info["tokens"]
        task_ids = info["task_ids"]
        
        # 找出这个文件的 K 个子任务的最大时间
        times = [task_time_map[tid] for tid in task_ids]
        max_time = max(times)
        avg_time = sum(times) / len(times)
        throughput = tokens / max_time
        
        file_results[f"file{file_idx}"] = {
            "tokens": tokens,
            "max_time": max_time,
            "avg_time": avg_time,
            "throughput": throughput
        }
        
        print(f"File {file_idx}:")
        print(f"  Tokens: {tokens}")
        print(f"  Max time: {max_time:.4f}s")
        print(f"  Avg time: {avg_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} tokens/s")
    
    # 总结
    total_tokens = sum(info["tokens"] for info in file_results.values())
    total_time = sum(info["max_time"] for info in file_results.values())
    avg_throughput = total_tokens / total_time
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(file_results)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average throughput: {avg_throughput:.2f} tokens/s")


if __name__ == "__main__":
    main()

"""
python simulate_dp.py \
    --data_path /data/tree/DynamicTreeAttn/data \
    --world_size 2 \
    --mode tree_forward \
    --K 8 \
    --pattern trie_forward \
    --mem_bound 1024 \
    --model_path /data/tree/models/Qwen3-4B 

"""
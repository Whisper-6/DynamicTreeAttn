import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine
import os
import tqdm


def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def loss_fn(logprob: torch.Tensor, entropy: torch.Tensor, attachment: dict):
    length = logprob.size(0)
    w_logprobs = attachment["w_logprobs"] / (length - 1)
    w_entropy = attachment["w_entropy"] / length
    return w_logprobs * logprob.sum() + w_entropy * entropy.sum()


def load_data(data_folder: str):
    data_files = [os.path.join(data_folder, f)
                  for f in os.listdir(data_folder) if f.endswith(".pt")]

    datas = []
    for file in sorted(data_files):
        data = torch.load(file, map_location="cpu")
        datas.append(data)

    return datas


def run_dense_forward_single_rank(model, input_ids):
    """单个 rank 运行 dense forward"""
    from dense_forward import forward
    
    model.eval()
    
    torch.cuda.synchronize()
    forward_time = time.time()
    
    token_trie = TokenTrie(input_ids)
    forward(model, token_trie.inputs)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_time
    
    n_tokens = sum(len(ids) for ids in input_ids)
    
    return {
        'time': forward_time,
        'n_tokens': n_tokens,
        'n_sequences': len(input_ids),
        'throughput': n_tokens / forward_time if forward_time > 0 else 0
    }


def run_tree_forward_single_rank(model, engine, input_ids, permute):
    """单个 rank 运行 tree forward"""
    model.eval()
    
    torch.cuda.synchronize()
    forward_time = time.time()

    trie = TokenTrie(input_ids, attachs=None)
    if permute == "random":
        trie.random_permute()
    elif permute == "idx":
        pass  # 保持原始顺序
    elif permute == "ours":
        trie.forward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {permute}")

    engine.forward(model=model, token_trie=trie)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_time

    return {
        'time': forward_time,
        'n_tokens': trie.n_tokens,
        'n_sequences': trie.n_sequences,
        'n_tree_tokens': trie.n_tree_tokens,
        'overlap_ratio': trie.n_tokens / trie.n_tree_tokens if trie.n_tree_tokens > 0 else 0,
        'throughput': trie.n_tokens / forward_time if forward_time > 0 else 0
    }


def run_dense_backward_single_rank(model, input_ids, loss_fn, gradient_checkpointing_enabled):
    """单个 rank 运行 dense backward"""
    from dense_backward import backward
    
    model.train()
    
    torch.cuda.synchronize()
    backward_time = time.time()
    
    token_trie = TokenTrie(input_ids)
    inputs = token_trie.inputs
    attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(inputs)
    backward(model, inputs, attachs, loss_fn, gradient_checkpointing_enabled)
    
    torch.cuda.synchronize()
    backward_time = time.time() - backward_time
    
    n_tokens = sum(len(ids) for ids in input_ids)
    
    return {
        'time': backward_time,
        'n_tokens': n_tokens,
        'n_sequences': len(input_ids),
        'throughput': n_tokens / backward_time if backward_time > 0 else 0
    }


def run_tree_backward_single_rank(model, engine, input_ids, loss_fn, block_size, permute, cut_f1_tail):
    """单个 rank 运行 tree backward"""
    model.train()
    
    torch.cuda.synchronize()
    backward_time = time.time()

    attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(input_ids)
    trie = TokenTrie(input_ids, attachs=attachs)
    if permute == "random":
        trie.random_permute()
    elif permute == "idx":
        pass  # 保持原始顺序
    elif permute == "ours":
        trie.backward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {permute}")
    
    engine.backward(model=model, token_trie=trie, block_size=block_size, loss_fn=loss_fn, cut_f1_tail=cut_f1_tail)
    
    torch.cuda.synchronize()
    backward_time = time.time() - backward_time

    return {
        'time': backward_time,
        'n_tokens': trie.n_tokens,
        'n_sequences': trie.n_sequences,
        'n_tree_tokens': trie.n_tree_tokens,
        'overlap_ratio': trie.n_tokens / trie.n_tree_tokens if trie.n_tree_tokens > 0 else 0,
        'throughput': trie.n_tokens / backward_time if backward_time > 0 else 0
    }


def worker_process(gpu_id, task_list, model_path, dtype_str, result_queue):
    """
    Worker 进程：在指定 GPU 上执行分配的任务列表
    
    Args:
        gpu_id: GPU ID
        task_list: 任务列表，每个任务是 (task_id, mode, input_ids, kwargs)
            - task_id: 任务唯一标识
            - mode: 'dense_forward', 'tree_forward', 'dense_backward', 'tree_backward'
            - input_ids: list of token sequences
            - kwargs: 额外参数 dict
                - gradient_checkpointing: bool (dense_backward 需要，默认 False)
        model_path: 模型路径
        dtype_str: 数据类型字符串 ('bf16', 'fp16', 'fp32')
        result_queue: 用于返回结果的队列
    
    写死的参数:
        - attn_implementation: "flash_attention_3"
        - block_size: 4096
        - cut_f1_tail: True
        - permute: "ours"
        - max_seq_len: 16384
    
    Returns (通过 result_queue):
        (gpu_id, results) 其中 results 是结果列表，每个结果包含:
            - task_id: str, 任务 ID
            - mode: str, 执行模式
            - gpu_id: int, GPU ID
            - time: float, 执行时间（秒）
            - n_tokens: int, token 数量
            - n_sequences: int, 序列数量
            - throughput: float, tokens/s
            - start_time: float, 开始时间戳
            - end_time: float, 结束时间戳
            - (tree 模式额外字段):
                - n_tree_tokens: int
                - overlap_ratio: float
    
    Example:
        >>> import torch.multiprocessing as mp
        >>> from run_dp import worker_process
        >>> 
        >>> # 准备任务
        >>> tasks = [
        ...     ("task_0", "tree_forward", [seq1, seq2], {}),
        ...     ("task_1", "dense_backward", [seq3, seq4], {"gradient_checkpointing": True}),
        ... ]
        >>> 
        >>> # 启动 worker
        >>> result_queue = mp.Queue()
        >>> p = mp.Process(
        ...     target=worker_process,
        ...     args=(0, tasks, "/path/to/model", "bf16", result_queue)
        ... )
        >>> p.start()
        >>> p.join()
        >>> 
        >>> # 获取结果
        >>> gpu_id, results = result_queue.get()
        >>> for r in results:
        ...     print(f"{r['task_id']}: {r['throughput']:.2f} tokens/s, time={r['time']:.4f}s")
    """
    # 设置 GPU - 直接使用指定的 GPU ID
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    dtype = parse_dtype(dtype_str)
    
    # 写死的参数
    ATTN_IMP = "flash_attention_3"
    BLOCK_SIZE = 4096
    CUT_F1_TAIL = True
    PERMUTE = "ours"
    
    # 加载模型到指定 GPU - 使用显式的设备映射
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=ATTN_IMP,
        device_map={"": gpu_id},
    )
    
    # 准备 engines（按需创建）
    engines = {}  # mode -> engine
    
    # 执行任务
    results = []
    for task_id, mode, input_ids, kwargs in task_list:
        # 创建 engine（如果需要且未创建）
        if mode.startswith('tree') and mode not in engines:
            forward_only = (mode == 'tree_forward')
            engines[mode] = TreeTrainingEngine(
                model_config=model.config,
                device=device,
                dtype=dtype,
                max_seq_len=16384,
                forward_only=forward_only
            )
        
        # 执行任务
        torch.cuda.synchronize(device=gpu_id)
        start_time = time.time()
        
        if mode == 'dense_forward':
            result = run_dense_forward_single_rank(model, input_ids)
        elif mode == 'tree_forward':
            result = run_tree_forward_single_rank(model, engines[mode], input_ids, PERMUTE)
        elif mode == 'dense_backward':
            gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
            result = run_dense_backward_single_rank(model, input_ids, loss_fn, gradient_checkpointing)
        elif mode == 'tree_backward':
            result = run_tree_backward_single_rank(
                model, engines[mode], input_ids, loss_fn,
                BLOCK_SIZE, PERMUTE, CUT_F1_TAIL
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        torch.cuda.synchronize(device=gpu_id)
        end_time = time.time()
        
        result['task_id'] = task_id
        result['mode'] = mode
        result['gpu_id'] = gpu_id
        result['start_time'] = start_time
        result['end_time'] = end_time
        
        results.append(result)
    
    # 返回结果
    result_queue.put((gpu_id, results))


if __name__ == "__main__":
    # 简单的单元测试
    import torch.multiprocessing as mp
    
    def test_worker_process():
        """测试 worker_process 函数"""
        print("Running unit test for worker_process...")
        
        # 创建测试数据
        test_tasks = [
            (
                "task_1",
                "tree_forward",
                [torch.randint(0, 1000, (100,)) for _ in range(2)],
                {}  # tree_forward 不需要参数（permute 写死为 ours）
            ),
            (
                "task_2",
                "tree_backward",
                [torch.randint(0, 1000, (150,)) for _ in range(3)],
                {}  # tree_backward 不需要参数（permute 写死为 ours）
            ),
            (
                "task_3",
                "dense_backward",
                [torch.randint(0, 1000, (80,)) for _ in range(2)],
                {"gradient_checkpointing": True}  # 唯一需要的参数
            ),
            (
                "task_4",
                "dense_forward",
                [torch.randint(0, 1000, (60,)) for _ in range(2)],
                {}  # dense_forward 不需要参数
            ),
        ]
        
        # 设置测试参数
        gpu_id = 0
        model_path = "/data/tree/models/Qwen3-4B"
        dtype_str = "bf16"
        
        # 创建结果队列
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        
        # 启动 worker 进程
        print(f"Launching worker on GPU {gpu_id} with {len(test_tasks)} tasks...")
        print(f"写死的参数:")
        print(f"  - attn_implementation: flash_attention_3")
        print(f"  - block_size: 4096")
        print(f"  - cut_f1_tail: True")
        print(f"  - permute: ours")
        print(f"  - max_seq_len: 16384")
        print(f"\n用户可控参数: gradient_checkpointing (仅 dense_backward)")
        
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, test_tasks, model_path, dtype_str, result_queue)
        )
        p.start()
        p.join()
        
        # 获取结果
        gpu_id_result, results = result_queue.get()
        
        print(f"\nResults from GPU {gpu_id_result}:")
        for result in results:
            print(f"  Task {result['task_id']}: {result['n_tokens']} tokens in {result['time']:.4f}s")
            print(f"    Throughput: {result['throughput']:.2f} tokens/s")
            if 'overlap_ratio' in result:
                print(f"    Overlap ratio: {result['overlap_ratio']:.4f}")
        
        print("\nTest passed!")
    
    def test_multi_gpu_round_robin():
        """测试多 GPU round-robin 调度"""
        print("\n" + "=" * 60)
        print("测试多 GPU Round-Robin 调度")
        print("=" * 60)
        
        # 设置 multiprocessing（只调用一次）
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # 已经设置过了
        
        # 读取数据
        data_folder = "/data/tree/DynamicTreeAttn/data"
        print(f"\nLoading data from {data_folder}...")
        datas = load_data(data_folder)
        print(f"Loaded {len(datas)} data files")
        
        # 参数设置
        num_ranks = 16  # DP 分成 16 个任务（更多任务 = 每个任务更小）
        num_gpus = 2   # 使用 2 个 GPU
        model_path = "/data/tree/models/Qwen2.5-1.5B-Instruct"
        dtype_str = "bf16"
        
        print(f"\nSettings:")
        print(f"  - Num ranks (DP tasks): {num_ranks}")
        print(f"  - Num GPUs: {num_gpus}")
        print(f"  - Mode: dense_backward with gradient_checkpointing=True")
        
        # 对每个 data batch 进行处理
        all_results = []
        
        for batch_idx, input_ids in enumerate(datas[:2]):  # 只测试前 2 个 batch
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_idx}...")
            print(f"  Sequences: {len(input_ids)}")
            print(f"  Total tokens: {sum(len(ids) for ids in input_ids)}")
            
            # 使用 TokenTrie.divide 分割任务
            token_trie = TokenTrie(input_ids)
            parts = token_trie.divide(n_parts=num_ranks, mem_bound=100)
            
            # 为每个 rank 创建任务
            all_tasks = []
            for rank_id in range(num_ranks):
                rank_input_ids = [input_ids[i] for i in parts[rank_id]]
                if len(rank_input_ids) == 0:
                    continue
                
                # 统计这个 rank 的数据量
                rank_n_seqs = len(rank_input_ids)
                rank_n_tokens = sum(len(ids) for ids in rank_input_ids)
                rank_max_len = max(len(ids) for ids in rank_input_ids) if rank_n_seqs > 0 else 0
                
                print(f"    Rank {rank_id}: {rank_n_seqs} seqs, {rank_n_tokens} tokens, max_len={rank_max_len}")
                
                task = (
                    f"batch{batch_idx}_rank{rank_id}",
                    "dense_backward",
                    rank_input_ids,
                    {"gradient_checkpointing": True}
                )
                all_tasks.append((rank_id, task))
            
            print(f"  Created {len(all_tasks)} tasks")
            
            # Round-robin 分配到 GPU
            gpu_task_lists = [[] for _ in range(num_gpus)]
            for rank_id, task in all_tasks:
                gpu_id = rank_id % num_gpus
                gpu_task_lists[gpu_id].append(task)
            
            print(f"\nTask assignment:")
            for gpu_id in range(num_gpus):
                print(f"  GPU {gpu_id}: {len(gpu_task_lists[gpu_id])} tasks")
                if len(gpu_task_lists[gpu_id]) > 0:
                    # 打印每个 GPU 分配到的任务 ID
                    task_ids = [t[0] for t in gpu_task_lists[gpu_id]]
                    print(f"    Task IDs: {task_ids}")
            
            # 启动多个 worker 进程
            result_queue = mp.Queue()
            processes = []
            
            overall_start = time.time()
            
            for gpu_id in range(num_gpus):
                if len(gpu_task_lists[gpu_id]) == 0:
                    continue
                
                p = mp.Process(
                    target=worker_process,
                    args=(gpu_id, gpu_task_lists[gpu_id], model_path, dtype_str, result_queue)
                )
                p.start()
                processes.append((gpu_id, p))
            
            # 等待所有进程完成
            for gpu_id, p in processes:
                p.join()
            
            overall_time = time.time() - overall_start
            
            # 收集结果
            batch_results = []
            while not result_queue.empty():
                gpu_id, results = result_queue.get()
                batch_results.extend(results)
            
            # 计算统计
            total_tokens = sum(r['n_tokens'] for r in batch_results)
            total_sequences = sum(r['n_sequences'] for r in batch_results)
            max_time = max(r['end_time'] - r['start_time'] for r in batch_results)
            
            # 按 GPU 分组统计
            gpu_stats = {}
            for r in batch_results:
                gpu_id = r['gpu_id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {'tokens': 0, 'sequences': 0, 'times': []}
                gpu_stats[gpu_id]['tokens'] += r['n_tokens']
                gpu_stats[gpu_id]['sequences'] += r['n_sequences']
                gpu_stats[gpu_id]['times'].append(r['end_time'] - r['start_time'])
            
            print(f"\nBatch {batch_idx} Results:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Total sequences: {total_sequences}")
            print(f"  Overall time (multi-GPU): {overall_time:.4f}s")
            print(f"  Max task time (simulated DP): {max_time:.4f}s")
            print(f"  Actual throughput: {total_tokens/overall_time:.2f} tokens/s")
            print(f"  Simulated DP throughput: {total_tokens/max_time:.2f} tokens/s")
            
            print(f"\n  Per-GPU Statistics:")
            for gpu_id in sorted(gpu_stats.keys()):
                stats = gpu_stats[gpu_id]
                print(f"    GPU {gpu_id}:")
                print(f"      - Tasks: {len(stats['times'])}")
                print(f"      - Tokens: {stats['tokens']}")
                print(f"      - Max task time: {max(stats['times']):.4f}s")
            
            all_results.append({
                'batch_idx': batch_idx,
                'overall_time': overall_time,
                'max_time': max_time,
                'total_tokens': total_tokens,
                'results': batch_results
            })
        
        # 总结
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total_tokens = sum(r['total_tokens'] for r in all_results)
        total_overall_time = sum(r['overall_time'] for r in all_results)
        total_max_time = sum(r['max_time'] for r in all_results)
        
        print(f"Total batches: {len(all_results)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total overall time: {total_overall_time:.2f}s")
        print(f"Total max time: {total_max_time:.2f}s")
        print(f"Avg actual throughput: {total_tokens/total_overall_time:.2f} tokens/s")
        print(f"Avg simulated DP throughput: {total_tokens/total_max_time:.2f} tokens/s")
        print("\nTest passed!")
    
    # 运行测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-gpu":
        test_multi_gpu_round_robin()
    else:
        test_worker_process()
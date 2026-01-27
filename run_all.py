import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tqdm
from tree_training_engine import TreeTrainingEngine
from token_trie import TokenTrie

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
    # return torch.tensor(0.0, device=logprob.device).requires_grad_(True)  # Placeholder


def load_data(data_folder: str):    
    data_files = [os.path.join(data_folder, f)
                  for f in os.listdir(data_folder) if f.endswith(".pt")]

    datas = []
    for file in sorted(data_files):
        data = torch.load(file, map_location="cpu")
        datas.append(data)

    return datas


def run_dense_forward(model, datas):
    from token_trie import TokenTrie
    from dense_forward import forward

    model.eval()

    n_tokens = sum(len(ids) for input_ids in datas for ids in input_ids)

    print(f"total files:{len(datas)}, total tokens:{n_tokens}")

    # warmup
    forward(model, datas[0])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    forward_time = time.time()

    for input_ids in datas:
        token_trie = TokenTrie(input_ids)
        forward(model, token_trie.inputs)

    torch.cuda.synchronize()
    forward_time = time.time() - forward_time
    
    throughput = n_tokens / forward_time

    print(f"[Dense Inference] Throughput : {throughput:.2f} tokens/s")
    print(f"[Dense Inference] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

def run_tree_forward(model, dtype, datas, permute):
    
    model.eval()
    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=dtype, max_seq_len=16384, forward_only=True)

    results = []

    # warmup
    trie = TokenTrie(datas[0], attachs=None)
    engine.forward(model=model, token_trie=trie)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_time = time.time()

    # 使用 tqdm 进度条
    for input_ids in tqdm.tqdm(datas):
        torch.cuda.synchronize()
        forward_time = time.time()
        token_trie = TokenTrie(input_ids, attachs=None)
        if permute == "random":
            token_trie.random_permute()
        elif permute == "idx":
            pass  # 保持原始顺序
        elif permute == "ours":
            token_trie.forward_permute()
        else:
            raise ValueError(f"Unsupported permute method: {permute}")
        engine.forward(model=model, token_trie=token_trie)
        torch.cuda.synchronize()
        forward_time = time.time() - forward_time

        overlap_ratio = trie.n_tokens / trie.n_tree_tokens
        throughput = trie.n_tokens / forward_time
        results.append((overlap_ratio, throughput))

    torch.cuda.synchronize()
    total_time = time.time() - total_time

    total_n_tokens = sum(len(ids) for input_ids in datas for ids in input_ids)
    total_throughput = total_n_tokens / total_time
    print(f"[Tree Inference] Total Throughput : {total_throughput:.2f} tokens/s")
    print(f"[Tree Inference] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"[Tree Inference] Results: ", results)

def run_dense_backward(model, datas, loss_fn, gradient_checkpointing_enabled: bool):
    from token_trie import TokenTrie
    from dense_backward import backward

    model.train()

    n_tokens = sum(len(ids) for input_ids in datas for ids in input_ids)

    # warmup
    attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(datas[0])
    backward(model, datas[0], attachs, loss_fn, gradient_checkpointing_enabled)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_time = time.time()

    for input_ids in datas:
        token_trie = TokenTrie(input_ids)
        inputs = token_trie.inputs
        attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(inputs)
        backward(model, inputs, attachs, loss_fn, gradient_checkpointing_enabled=True)

        # 清理显存
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    total_time = time.time() - total_time

    throughput = n_tokens / total_time
    print(f"[Dense Training] Throughput : {throughput:.2f} tokens/s")
    print(f"[Dense Training] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

def run_tree_backward(model, dtype, datas, loss_fn, block_size, permute):
    
    model.train()

    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=dtype, max_seq_len=16384)

    n_tokens = sum(len(ids) for input_ids in datas for ids in input_ids)

    # warmup
    attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(datas[0])
    token_trie = TokenTrie(datas[0], attachs=attachs)
    engine.backward(model=model, token_trie=token_trie, block_size=block_size, loss_fn=loss_fn)

    results = []

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_time = time.time()

    for input_ids in tqdm.tqdm(datas):
        torch.cuda.synchronize()
        backward_time = time.time()
        attachs = [{"w_logprobs": -1.0, "w_entropy": 0.1}] * len(input_ids)
        token_trie = TokenTrie(input_ids, attachs=attachs)
        if permute == "random":
            token_trie.random_permute()
        elif permute == "idx":
            pass  # 保持原始顺序
        elif permute == "ours":
            token_trie.backward_permute()
        else:
            raise ValueError(f"Unsupported permute method: {permute}")
        engine.backward(model=model, token_trie=token_trie, block_size=block_size, loss_fn=loss_fn)
        torch.cuda.synchronize()
        backward_time = time.time() - backward_time

        overlap_ratio = token_trie.n_tokens / token_trie.n_tree_tokens
        throughput = token_trie.n_tokens / backward_time
        results.append((overlap_ratio, throughput))

    torch.cuda.synchronize()
    total_time = time.time() - total_time

    throughput = n_tokens / total_time
    print(f"[Tree Training] Throughput: {throughput:.2f} tokens/s")
    print(f"[Tree Training] Peak Memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"[Tree Training] Results: ", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)

    parser.add_argument("--run", type=str, required=True,
                        choices=["dense_forward", "tree_forward", "dense_backward", "tree_backward"])

    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--act-ckpt", action="store_true", help="enable activation checkpointing")
    parser.add_argument("--permute", type=str, default=None, choices=["random", "idx", "ours"])

    args = parser.parse_args()
    dtype = torch.bfloat16
    model_path = f"{args.model_folder}/{args.model}"

    # -------- load data --------
    datas = load_data(args.data_folder)

    # -------- load model --------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )

    if args.run == "dense_forward":
        run_dense_forward(model, datas)

    elif args.run == "dense_backward":
        run_dense_backward(model, datas, loss_fn, gradient_checkpointing_enabled=args.act_ckpt)

    elif args.run == "tree_forward":
        run_tree_forward(model, dtype, datas, args.permute)

    elif args.run == "tree_backward":
        run_tree_backward(model, dtype, datas, loss_fn, args.block_size, args.permute)
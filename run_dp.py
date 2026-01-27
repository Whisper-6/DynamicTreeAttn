import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine


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
    return -logprob.sum()


def run_tree_forward(model, engine, input_ids):
    
    model.eval()
    
    forward_time = time.time()

    trie = TokenTrie(input_ids, attachs=None, dtype=dtype)
    print(f"n_seq: {trie.n_sequences}, n_tokens: {trie.n_tokens}, n_leafed_tokens: {trie.n_leafed_tokens}, n_tree_tokens: {trie.n_tree_tokens}")
    print(f"avg_seq_len: {trie.n_tokens / trie.n_sequences:.2f}, overlap_ratio: {trie.n_tokens / trie.n_tree_tokens:.4f}")
    trie.forward_permute()

    logprobs_list = engine.forward(model=model, token_trie=trie)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_time

    return forward_time


def run_tree_backward(model, engine, input_ids, loss_fn):
    
    backward_time = time.time()

    trie = TokenTrie(input_ids, attachs=None, dtype=dtype)
    trie.backward_permute()
    
    loss = engine.backward(model=model, token_trie=trie, block_size=args.block_size, loss_fn=loss_fn)

    torch.cuda.synchronize()
    backward_time = time.time() - backward_time

    return backward_time

def run(model, input_ids, engine, mem_bound):
    # forward
    token_trie = TokenTrie(input_ids, sorted=False, dtype=dtype)
    token_trie.forward_permute()
    parts = token_trie.divide(n_parts=args.num_ranks, mem_bound=mem_bound)
    print(parts)
    forward_times = []
    for rank_id in range(args.num_ranks):
        rank_input_ids = [input_ids[i] for i in parts[rank_id]]
        forward_time = run_tree_forward(model, engine, rank_input_ids)
        forward_times.append(forward_time)

    print(f"[Tree Inference] Forward Times : {forward_times}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-imp", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager"])

    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--num-ranks", type=int, default=None)
    parser.add_argument("--mem-bound", type=int, default=None)

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    model_path = f"{args.model_folder}/{args.model}"

    # -------- load data --------
    input_ids = torch.load(args.data, map_location="cpu")

    # -------- load model --------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=args.attn_imp,
        device_map="cuda",
    )

    # 清空显存统计
    torch.cuda.reset_peak_memory_stats()

    max_seq_len = max(len(ids) for ids in input_ids)

    engine = TreeTrainingEngine(
        model_config=model.config,
        device="cuda",
        dtype=dtype,
        max_seq_len=max_seq_len
    )

    run(model, input_ids, engine, args.mem_bound)
    run(model, input_ids, engine, args.mem_bound)

    

    print(f"[Tree Inference] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
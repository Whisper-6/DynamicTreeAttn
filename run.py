import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine


def strip_padding(ids: torch.Tensor, padding_token: int = 0) -> torch.Tensor:
    assert ids.dim() == 1, "ids must be a 1D tensor"
    mask = ids != padding_token
    last = mask.nonzero(as_tuple=False)[-1].item()
    return ids[: last + 1]

def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def save_gradients(model, path: str):
    """
    Save all parameter gradients to a file.
    Stored as a dict: {param_name: grad_tensor (cpu)}
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu()
        else:
            grads[name] = None
    torch.save(grads, path)

def loss_fn(logprob: torch.Tensor, entropy: torch.Tensor, attachment: dict):
    return -logprob.sum() # / length
    
    # w_logprobs = attachment["w_logprobs"] / (length - 1)
    # w_entropy = attachment["w_entropy"] / length
    # return w_logprobs * logprob.mean() + w_entropy * entropy.sum()
    # return torch.tensor(0.0, device=logprob.device).requires_grad_(True)  # Placeholder


def load_data(data_path: str, model_path: str):
    if data_path.endswith(".pt"):
        data = torch.load(data_path, map_location="cpu")
        input_ids = data
        # input_ids = data["input_data"]["input_ids"]
        # input_ids = [strip_padding(ids.squeeze(0)) for ids in input_ids]
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


def run_dense_forward(model, input_ids):
    from dense_forward import forward

    model.eval()

    forward_time = time.time()

    logprobs_list = forward(model, input_ids)

    torch.cuda.synchronize()
    forward_time = time.time() - forward_time

    n_tokens = sum(len(ids) for ids in input_ids)
    throughput = n_tokens / forward_time

    loss = sum(-lp.mean().item() for lp in logprobs_list)

    print(f"[Dense Inference]")
    print(f"Loss          : {loss:.6f}")
    print(f"Time (s)      : {forward_time:.3f}")
    print(f"Throughput    : {throughput:.2f} tokens/s")

def run_tree_forward(model, dtype, input_ids, permute):
    
    model.eval()
    
    forward_time = time.time()

    max_seq_len = max(len(ids) for ids in input_ids)
    engine = TreeTrainingEngine(
        model_config=model.config,
        device=model.device,
        dtype=dtype,
        max_seq_len=max_seq_len,
        forward_only=True
    )
    trie = TokenTrie(input_ids, attachs=None, dtype=dtype)

    if permute == "random":
        trie.random_permute()
    elif permute == "idx":
        pass  # 保持原始顺序
    elif permute == "ours":
        trie.forward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {permute}")


    logprobs_list = engine.forward(model=model, token_trie=trie)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_time

    throughput = trie.n_tokens / forward_time
    throughput_leafed = trie.n_leafed_tokens / forward_time
    throughput_tree = trie.n_tree_tokens / forward_time

    loss = sum(-lp.mean().item() for lp in logprobs_list)

    print(f"[Tree Inference]")
    print(f"Loss          : {loss:.6f}")
    print(f"Time (s)      : {forward_time:.3f}")
    print(f"Throughput    : {throughput:.2f} tokens/s, {throughput_leafed:.2f} leafed-tokens/s, {throughput_tree:.2f} tree-tokens/s")
    print(f"n_tokens = {trie.n_tokens}, n_leafed_tokens = {trie.n_leafed_tokens}, n_tree_tokens = {trie.n_tree_tokens}")
    print(f"Overlap Ratio = {trie.n_tokens / trie.n_tree_tokens:.4f}x, {trie.n_leafed_tokens / trie.n_tree_tokens:.4f}x (Leafed)")

def run_dense_backward(model, input_ids, attachs, loss_fn, gradient_checkpointing_enabled: bool):
    from dense_backward import backward

    model.train()

    start_time = time.time()
    n_tokens = sum(len(ids) for ids in input_ids)

    loss = backward(model, input_ids, attachs, loss_fn, gradient_checkpointing_enabled)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    throughput = n_tokens / elapsed

    print(f"[Dense Training]")
    print(f"Loss       : {loss:.6f}")
    print(f"Time (s)   : {elapsed:.3f}")
    print(f"Throughput : {throughput:.2f} tokens/s")

    return loss

def run_tree_backward(model, input_ids, attachs, loss_fn, permute):
    
    backward_time = time.time()

    max_seq_len = max(len(ids) for ids in input_ids)
    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=dtype, max_seq_len=max_seq_len)

    trie = TokenTrie(input_ids, attachs, dtype=dtype)
    if permute == "random":
        trie.random_permute()
    elif permute == "idx":
        pass  # 保持原始顺序
    elif permute == "ours":
        trie.backward_permute()
    else:
        raise ValueError(f"Unsupported permute method: {permute}")

    loss = engine.backward(model=model, token_trie=trie, block_size=args.block_size, loss_fn=loss_fn)

    torch.cuda.synchronize()
    backward_time = time.time() - backward_time

    throughput = trie.n_tokens / backward_time
    throughput_leafed = trie.n_leafed_tokens / backward_time
    throughput_tree = trie.n_tree_tokens / backward_time

    print(f"[Tree Training]")
    print(f"Loss          : {loss:.6f}")
    print(f"Time (s)      : {backward_time:.3f}")
    print(f"Throughput    : {throughput:.2f} tokens/s, {throughput_leafed:.2f} leafed-tokens/s, {throughput_tree:.2f} tree-tokens/s")
    print(f"n_tokens = {trie.n_tokens}, n_leafed_tokens = {trie.n_leafed_tokens}, n_tree_tokens = {trie.n_tree_tokens}")
    print(f"Overlap Ratio = {trie.n_tokens / trie.n_tree_tokens:.4f}x, {trie.n_leafed_tokens / trie.n_tree_tokens:.4f}x (Leafed)")
    print(f"Leaf Sequences: {len(trie.inputs)}")
    print(f"Avg Segment Length = {trie.n_tree_tokens / len(trie.inputs):.2f} tokens/sequence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-imp", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--run", type=str, required=True,
                        choices=["dense_forward", "tree_forward", "dense_backward", "tree_backward"])

    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--act-ckpt", action="store_true", help="enable activation checkpointing")

    parser.add_argument("--throw-prefix", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--grad-out", type=str, default=None)
    parser.add_argument("--permute", type=str, default=None, choices=["random", "idx", "ours"])

    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    model_path = f"{args.model_folder}/{args.model}"

    # -------- load data --------
    input_ids = load_data(args.data, model_path)
    if args.throw_prefix is not None:
        input_ids = [ids[args.throw_prefix:] for ids in input_ids if ids.numel() > args.throw_prefix]
    if args.max_seq_len is not None:
        input_ids = [ids[: args.max_seq_len] for ids in input_ids]

    # -------- load model --------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=args.attn_imp,
        device_map="cuda",
    )

    # 清空显存统计
    torch.cuda.reset_peak_memory_stats()

    if args.run.endswith("backward"):
        attachs = [{"w_logprobs": -1.0, "w_entropy": 0.0} for ids in input_ids]

    if args.run == "dense_forward":
        run_dense_forward(model, input_ids)

    elif args.run == "dense_backward":
        run_dense_backward(model, input_ids, attachs, loss_fn, gradient_checkpointing_enabled=args.act_ckpt)

    elif args.run == "tree_forward":
        run_tree_forward(model, dtype, input_ids, args.permute)

    elif args.run == "tree_backward":
        run_tree_backward(model, input_ids, attachs, loss_fn, args.permute)
        run_tree_backward(model, input_ids, attachs, loss_fn, args.permute)

    print(f"[Tree Inference] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

    # -------- save gradients --------
    if args.run.endswith("backward"):
        if args.grad_out is not None:
            save_gradients(model, args.grad_out)
        else:
            # 输出前 10 个 model 参数的梯度模长
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"Param: {name}, Grad Norm: {grad_norm:.6f}")
                else:
                    print(f"Param: {name}, Grad is None")
                if i >= 9:
                    break
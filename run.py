import torch
from token_trie import TokenTrie
from tree_training_engine import TreeTrainingEngine
from dense import forward as _dense_forward, backward as _dense_backward
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
    

def dense_backward(model, input_ids, attachs, loss_fn, act_ckpt: bool, use_tqdm):

    backward_time = get_time()
    loss = _dense_backward(model, input_ids, attachs, loss_fn, act_ckpt, use_tqdm=use_tqdm)
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
        cut_f1_tail=args.cut_f1_tail
    )
    backward_time = get_time() - backward_time

    stats = trie.get_stats(mode="backward", block_size=args.block_size)
    stats["loss"] = loss
    stats["time"] = backward_time

    return stats

# ---------------- Test ----------------

import argparse
import os
from transformers import AutoModelForCausalLM

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

    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--act-ckpt", type=bool, default=False, help="enable activation checkpointing")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", type=bool, default=True, help="enable cutting f1 tail")
    parser.add_argument("--leafization", type=bool, default=False, help="enable leafization")

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

    # -------- run --------
    torch.cuda.reset_peak_memory_stats()

    if args.run == "dense_forward":
        stats = dense_forward(model, input_ids, use_tqdm=True)

    elif args.run == "dense_backward":
        stats = dense_backward(model, input_ids, attachs, loss_fn, args.act_ckpt, use_tqdm=True)

    elif args.run == "tree_forward":
        stats = tree_forward(model, None, input_ids, args)

    elif args.run == "tree_backward":
        stats = tree_backward(model, None, input_ids, attachs, loss_fn, args)

    print(f"[{run_name}] Loss: {stats['loss']:.6f}")
    print(f"[{run_name}] Time: {stats['time']:.2f} s")
    print(f"[{run_name}] Peak Memory : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

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
  --model /data/tree/models/Qwen3-0.6B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --grad-out grad/Qwen3-0.6B-TB-bf16.pt

python run.py \
  --model /data/tree/models/Qwen3-0.6B \
  --data data/tau2-16k-merged/call1.pt \
  --run dense_backward \
  --grad-out grad/Qwen3-0.6B-DB-bf16.pt

python compare_grads.py \
    --baseline-grad grad/Qwen3-0.6B-DB-bf16.pt \
    --exp-grad grad/Qwen3-0.6B-TB-bf16.pt \
    --out grad/Qwen3-0.6B-TB-vs-DB-bf16.txt
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tqdm
import json

from run import dense_forward, tree_forward, dense_backward, tree_backward
from tree_training_engine import TreeTrainingEngine

ATTACH = {
    "w_logprobs": -1.0,
    "w_entropy": 0.1
}

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
        datas.append(data)

    return datas


def run_dense_forward(model, datas, warmup: bool=True):

    if warmup:
        dense_forward(model, datas[0][:16], use_tqdm=False)

    results = []

    for input_ids in tqdm.tqdm(datas):
        stats = dense_forward(model, input_ids, use_tqdm=False)
        results.append(stats)
    
    return results

def run_tree_forward(model, datas, args, warmup: bool=True):
    
    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384, forward_only=True)

    if warmup:
        tree_forward(model, engine, datas[0], args)

    results = []

    for input_ids in tqdm.tqdm(datas):
        stats = tree_forward(model, engine, input_ids, args)
        results.append(stats)

    return results

def run_dense_backward(model, datas, loss_fn, act_ckpt, warmup: bool=True):

    if warmup:
        inputs = datas[0][:16]
        attachs = [ATTACH] * len(inputs)
        dense_backward(model, inputs, attachs, loss_fn, act_ckpt, use_tqdm=False)
        model.zero_grad()

    results = []

    for input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = dense_backward(model, input_ids, attachs, loss_fn, act_ckpt, use_tqdm=False)
        results.append(stats)

    return results

def run_tree_backward(model, datas, loss_fn, args, warmup: bool=True):

    engine = TreeTrainingEngine(model_config=model.config, device=model.device, dtype=args.dtype, max_seq_len=16384)

    if warmup:
        attachs = [ATTACH] * len(datas[0])
        tree_backward(model, engine, datas[0], attachs, loss_fn, args)
        model.zero_grad()

    results = []

    for input_ids in tqdm.tqdm(datas):
        attachs = [ATTACH] * len(input_ids)
        stats = tree_backward(model, engine, input_ids, attachs, loss_fn, args)
        results.append(stats)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run", type=str, required=True,
                        choices=["dense_forward", "tree_forward", "dense_backward", "tree_backward"])
    parser.add_argument("--stats-out", type=str, default=None)

    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--act-ckpt", action="store_true", help="enable activation checkpointing")
    parser.add_argument("--permute", type=str, default="ours", choices=["random", "idx", "ours"])
    parser.add_argument("--cut-f1-tail", action="store_true", help="enable cutting f1 tail")
    parser.add_argument("--leafization", action="store_true", help="enable leafization")
    
    args = parser.parse_args()
    args.dtype = torch.bfloat16
    run_name = args.run.replace('_', ' ').title()

    # -------- load data --------
    datas = load_data(args.data)

    if args.leafization:
        for input_ids in datas:
            token_trie = TokenTrie(input_ids)
            input_ids = token_trie.inputs

    # -------- load model --------
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        attn_implementation="flash_attention_3",
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

    total_tokens = sum(stat["n_tokens"] for stat in results)
    total_time = sum(stat["time"] for stat in results)
    throughput = total_tokens / total_time
    print(f"[{run_name}] Throughput: {throughput:.2f} tokens/s")

    if args.stats_out is not None:
        with open(args.stats_out, "w") as f:
            for stat in results:
                f.write(json.dumps(stat) + "\n")

from typing import Set, List, Optional

from token_trie import TokenTrie
from trie import CompressedTrie, _get_subtrie, _get_stats

import time

def LB_by_n_tokens(token_seqs, K):
    bins = [[] for _ in range(K)]
    bin_lens = [0] * K
    seq_indices = sorted(range(len(token_seqs)), key=lambda i: -len(token_seqs[i]))
    for i in seq_indices:
        min_bin = min(range(K), key=lambda j: bin_lens[j])
        bins[min_bin].append(i)
        bin_lens[min_bin] += len(token_seqs[i])
    return bins

def pred_time(compressed_trie, permute: str, tree_time_model):
    if permute not in {"forward", "backward"}:
        raise ValueError(f"Unsupported permute method: {permute}")
    
    if permute == "forward":
        _, lens, lcp_lens = compressed_trie.get_order_forward()
    elif permute == "backward":
        _, lens, lcp_lens = compressed_trie.get_order_backward()
        lens = lens[::-1]
        lcp_lens = lcp_lens[::-1]
    
    stats = _get_stats(lens, lcp_lens)
    return tree_time_model.pred(stats)

def get_original_bins(token_trie: TokenTrie, leaf_bins: List[List[int]]) -> List[List[int]]:
    bins = [[] for _ in range(len(leaf_bins))]
    for bucket_idx, leaf_bucket in enumerate(leaf_bins):
        for leaf_idx in leaf_bucket:
            attach_lists = token_trie.attach_lists[leaf_idx]
            for attach, _ in attach_lists:
                original_seq_idx = attach['_sequence_batch_id']
                bins[bucket_idx].append(original_seq_idx)
    return bins

def LB_by_tree_time_model(token_seqs, tree_time_model, permute: str, K):

    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)

    leaf_bins = [[] for _ in range(K)]
    bin_times = [0.0] * K

    for i in range(n_leaf_seqs):
        min_bin = min(range(K), key=lambda j: bin_times[j])
        leaf_bins[min_bin].append(i)
        bin_compressed_trie = _get_subtrie(compressed_trie, leaf_bins[min_bin])
        bin_times[min_bin] = pred_time(bin_compressed_trie, permute, tree_time_model)

    bins = get_original_bins(token_trie, leaf_bins)
    return bins

def try_devide(compressed_trie, n_seqs, permute: str, cost_limit: float) -> List[List[int]] | None:

    bins = []

    start = 0
    while start < n_seqs:
        L = start - 1
        R = n_seqs - 1
        while L < R:
            mid = (L + R + 1) // 2
            cur_subtrie = _get_subtrie(compressed_trie, set(range(start, mid + 1)))
            est_time = pred_time(cur_subtrie, permute, tree_time_model)
            if est_time <= cost_limit:
                L = mid
            else:
                R = mid - 1
        end = L
        if end < start:
            return None
        bins.append(list(range(start, end + 1)))
        start = end + 1

    return bins

def LB_by_DFS_and_tree_time_model(token_seqs, tree_time_model, permute: str, K):

    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)
    
    R = float(pred_time(compressed_trie, permute, tree_time_model))
    L = R / K
    eps = R * 1e-4
    while R - L > eps:
        mid = (L + R) / 2.0
        bins = try_devide(compressed_trie, n_leaf_seqs, permute, mid)
        if bins is not None and len(bins) <= K:
            R = mid
        else:
            L = mid + eps

    leaf_bins = try_devide(compressed_trie, n_leaf_seqs, permute, R)
    bins = get_original_bins(token_trie, leaf_bins)

    return bins

import argparse
import torch
from tree_time_model import TreeTimeModel
import os
import json

def load_data(data_folder: str):
    datas = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pt"):
            name = filename[:-3]
            file_path = os.path.join(data_folder, filename)
            data = torch.load(file_path, map_location="cpu")
            datas.append((name, data))
    return datas

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--permute", type=str, required=True, choices=["forward", "backward"])
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--stats-file", type=str, required=True)
    parser.add_argument("--out-folder", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    print("Loading data...")

    datas = load_data(args.data_folder)

    print("Data loaded.")

    if args.method in {"LB_by_tree_time_model", "LB_by_DFS_and_tree_time_model"}:
        tree_time_model = TreeTimeModel()
        if args.stats_file is not None:
            stats_data = []
            with open(args.stats_file, "r") as f:
                for line in f:
                    stats = json.loads(line)
                    stats_data.append(stats)
            tree_time_model.add_data(stats_data)

    dp_time = 0.0

    for name, inputs in datas:
        dp_time -= time.time()
        if args.method == "LB_by_n_tokens":
            bins = LB_by_n_tokens(inputs, args.K)
        elif args.method == "LB_by_tree_time_model":
            bins = LB_by_tree_time_model(inputs, tree_time_model, args.permute, args.K)
        elif args.method == "LB_by_DFS_and_tree_time_model":
            bins = LB_by_DFS_and_tree_time_model(inputs, tree_time_model, args.permute, args.K)
        else:
            raise ValueError(f"Unsupported method: {args.method}")
        dp_time += time.time()
        
        for bucket_idx, bucket in enumerate(bins):
            out_path = os.path.join(args.out_folder, f"{name}_bin{bucket_idx}.pt")
            bucket_inputs = [inputs[i] for i in bucket]
            torch.save(bucket_inputs, out_path)

    print(f"Data parallel time: {dp_time:.6f} seconds")

"""
python data_parallel.py --data data/tau2-16k-merged/call3.pt --method LB_by_n_tokens --K 8
python data_parallel.py --data data/tau2-16k-merged/call3.pt --method LB_by_tree_time_model --K 8
python data_parallel.py \
    --data-folder data/tau2-16k-merged \
    --out-folder data/tau2-16k-K8-DFS-forward-TM \
    --stats-file stats/Qwen3-1.7B-K8-DFS-forward.jsonl \
    --method LB_by_DFS_and_tree_time_model \
    --K 8 \
    --permute forward

python data_parallel.py \
    --data-folder data/tau2-16k-merged \
    --out-folder data/tau2-16k-K8-DFS-backward-TM-1.7B \
    --stats-file stats/Qwen3-1.7B-K8-DFS-backward.jsonl \
    --method LB_by_DFS_and_tree_time_model \
    --K 8 \
    --permute backward
"""
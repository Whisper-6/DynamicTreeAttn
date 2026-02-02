from typing import Set, List, Optional

from token_trie import TokenTrie
from trie import CompressedTrie, _get_subtrie, _get_stats

def LB_by_n_tokens(token_seqs, K):
    bins = [[] for _ in range(K)]
    bin_lens = [0] * K
    seq_indices = sorted(range(len(token_seqs)), key=lambda i: -len(token_seqs[i]))
    for i in seq_indices:
        min_bin = min(range(K), key=lambda j: bin_lens[j])
        bins[min_bin].append(i)
        bin_lens[min_bin] += len(token_seqs[i])
    return bins

def pred_time(compressed_trie, time_model, permute: str):
    if permute not in {"forward", "backward"}:
        raise ValueError(f"Unsupported permute method: {permute}")
    
    if permute == "forward":
        _, lens, lcp_lens = compressed_trie.get_order_forward()
    elif permute == "backward":
        _, lens, lcp_lens = compressed_trie.get_order_backward()
        lens = lens[::-1]
        lcp_lens = lcp_lens[::-1]
    
    stats = _get_stats(lens, lcp_lens)
    return time_model.pred(stats)

def get_original_bins(token_trie: TokenTrie, leaf_bins: List[List[int]]) -> List[List[int]]:
    bins = [[] for _ in range(len(leaf_bins))]
    for bucket_idx, leaf_bucket in enumerate(leaf_bins):
        for leaf_idx in leaf_bucket:
            attach_lists = token_trie.attach_lists[leaf_idx]
            for attach, _ in attach_lists:
                original_seq_idx = attach['_sequence_batch_id']
                bins[bucket_idx].append(original_seq_idx)
    return bins

def LB_by_TM(token_seqs, time_model, permute: str, K):

    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)

    leaf_bins = [[] for _ in range(K)]
    bin_times = [0.0] * K

    for i in range(n_leaf_seqs):
        min_bin = min(range(K), key=lambda j: bin_times[j])
        leaf_bins[min_bin].append(i)
        bin_compressed_trie = _get_subtrie(compressed_trie, leaf_bins[min_bin])
        bin_times[min_bin] = pred_time(bin_compressed_trie, time_model, permute)

    bins = get_original_bins(token_trie, leaf_bins)
    return bins

def try_devide(compressed_trie, n_seqs, K, divL, divR, time_model, permute: str, cost_limit: float) -> List[List[int]] | None:

    divs = []

    start = 0
    while start < n_seqs:
        divs.append(start)
        if len(divs) > K:
            break
        L = max(divL[len(divs)] - 1, start)
        R = divR[len(divs)] - 1
        while L < R:
            mid = (L + R + 1) // 2
            cur_subtrie = _get_subtrie(compressed_trie, set(range(start, mid + 1)))
            est_time = pred_time(cur_subtrie, time_model, permute)
            if est_time <= cost_limit:
                L = mid
            else:
                R = mid - 1
        start = L + 1

    return divs

def LB_by_DFS_and_TM(token_seqs, time_model, permute: str, K):

    token_trie = TokenTrie(token_seqs)
    n_leaf_seqs = len(token_trie.inputs)
    compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)
    
    R = float(pred_time(compressed_trie, time_model, permute))
    L = R / K
    eps = R * 1e-4

    divL = [0] * (K+1)
    divR = [n_leaf_seqs] * (K+1)

    while R - L > eps:
        mid = (L + R) / 2.0
        divs = try_devide(compressed_trie, n_leaf_seqs, K, divL, divR, time_model, permute, mid)
        if len(divs) <= K:
            R = mid
            divR[:len(divs)] = divs
        else:
            L = mid + eps
            divL = divs[:K+1]

    leaf_bins = [list(range(divR[i], divR[i + 1])) for i in range(K)]
    bins = get_original_bins(token_trie, leaf_bins)
    return bins


# -------- Test --------

def eval(token_seqs, bins, time_model, permute: str):
    total_time = 0.0
    max_time = 0.0
    for bucket in bins:
        token_trie = TokenTrie([token_seqs[i] for i in bucket])
        compressed_trie = CompressedTrie(token_trie.lens, token_trie.lcp_lens)
        bucket_pred_time = pred_time(compressed_trie, time_model, permute)
        total_time += bucket_pred_time
        max_time = max(max_time, bucket_pred_time)
    return total_time, max_time

import argparse
import torch
from tree_time_model import TreeTimeModel
import os
import json
import time

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
    parser.add_argument("--permute", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--stats-file", type=str, default=None)
    parser.add_argument("--out-folder", type=str, default=None)
    parser.add_argument("--disable-TM", action="store_true")
    args = parser.parse_args()

    if args.out_folder is not None and not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    datas = load_data(args.data_folder)

    time_model_eval = TreeTimeModel()
    if args.stats_file is not None:
        stats_data = []
        with open(args.stats_file, "r") as f:
            for line in f:
                stats = json.loads(line)
                stats_data.append(stats)
        time_model_eval.add_data(stats_data)

    time_model = TreeTimeModel() if args.disable_TM else time_model_eval

    dp_time = 0.0
    total_time_sum = 0.0
    max_time_sum = 0.0

    for name, inputs in datas:
        dp_time -= time.time()
        if args.method == "LB_by_n_tokens":
            bins = LB_by_n_tokens(inputs, args.K)
        elif args.method == "LB_by_TM":
            bins = LB_by_TM(inputs, time_model, args.permute, args.K)
        elif args.method == "LB_by_DFS_and_TM":
            bins = LB_by_DFS_and_TM(inputs, time_model, args.permute, args.K)
        else:
            raise ValueError(f"Unsupported method: {args.method}")
        dp_time += time.time()

        if args.stats_file is not None:
            total_time, max_time = eval(inputs, bins, time_model_eval, args.permute)
            total_time_sum += total_time
            max_time_sum += max_time
            # print(f"Dataset: {name},  Best time: {total_time/args.K:.4f} s, Max time: {max_time:.4f} s")
        
        if args.out_folder is not None:
            for bucket_idx, bucket in enumerate(bins):
                out_path = os.path.join(args.out_folder, f"{name}_bin{bucket_idx}.pt")
                bucket_inputs = [inputs[i] for i in bucket]
                torch.save(bucket_inputs, out_path)

    print(f"Data parallel time: {dp_time:.4f} seconds")
    if args.stats_file is not None:
        print(f"Total best time: {total_time_sum/args.K:.4f} seconds")
        print(f"Total max time: {max_time_sum:.4f} seconds")

"""
python data_parallel.py \
    --data-folder data/tau2-16k-merged \
    --out-folder data/tau2-16k-K8-DFS-forward-TM \
    --stats-file stats/Qwen3-1.7B-K8-DFS-forward.jsonl \
    --method LB_by_DFS_and_TM \
    --K 8 \
    --permute forward

python data_parallel.py \
    --data-folder data/tau2-16k-merged \
    --method LB_by_DFS_and_TM \
    --K 8 \
    --permute backward
"""
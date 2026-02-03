import json
import os
import torch

from token_trie import TokenTrie

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["forward", "backward"], required=True)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    stats_list = []
    with open(args.stats_file, "r") as f:
        for line in f:
            stats = json.loads(line)
            stats_list.append(stats)

    stats_list_new = []
    for stats in stats_list:
        data_path = os.path.join(args.data, f"{stats['name']}.pt")
        data = torch.load(data_path, map_location="cpu")
        token_trie = TokenTrie(data)
        if args.mode == "forward":
            token_trie.forward_permute()
        else:
            token_trie.backward_permute()
        stats_new = token_trie.get_stats(mode=args.mode, block_size=args.block_size)
        stats_new["name"] = stats["name"]
        stats_new["time"] = stats["time"]
        stats_new["loss"] = stats["loss"]
        stats_list_new.append(stats_new)

    # 覆写原文件
    with open(args.stats_file, "w") as f:
        for stats in stats_list_new:
            f.write(json.dumps(stats) + "\n")

"""
python remark.py \
    --stats-file stats/Qwen3-4B-K8-DFS-tree-backward.jsonl \
    --data data/tau2-16k-dp/K8-DFS-tree \
    --mode backward \
    --block-size 2048
"""
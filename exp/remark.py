MODEL_FOLDER = "/data/tree/models"
DATA_FOLDER = "data/tau2-16k-merged"
DP_FOLDER = "data/tau2-16k-dp"

models = [
  "Qwen3-1.7B",
  "Qwen3-4B"
]

K_set = [2, 4, 8]
DP_methods = ["FFD-token", "FFD-tree", "DFS-tree"]

import os
import time

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

for model in models:
    for K in K_set:
        for method in DP_methods:
            print(f"Processing model={model}, K={K}, method={method}, task=backward")
            data = f"{DP_FOLDER}/K{K}-{method}"
            stats_file = f"stats/{model}-K{K}-{method}-backward.jsonl"
            if os.path.exists(stats_file):
                run(f"python remark.py \
                    --stats-file {stats_file} \
                    --data {data} \
                    --mode backward \
                    --block-size 2048")
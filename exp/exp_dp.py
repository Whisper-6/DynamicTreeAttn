MODEL_FOLDER = "/data/tree/models"
DATA_FOLDER = "data/tau2-16k-merged"
DP_FOLDER = "data/tau2-16k-dp"

models = [
  "Qwen3-1.7B",
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B"
]

K_set = [2, 4, 8]
tasks = ["forward", "backward"]
DP_methods = ["FFD-token", "FFD-tree", "DFS-tree"]
DP_methods_call = {
    "FFD-token" : "LB_by_n_tokens",
    "FFD-tree" : "LB_by_TM",
    "DFS-tree" : "LB_by_DFS_and_TM"
}

import os
import time

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

for model in models:
    for K in K_set:
        for method in DP_methods:
            for task in tasks:

                print(f"Processing model={model}, K={K}, method={method}, task={task}")

                data = f"{DP_FOLDER}/K{K}-{method}"
                if not os.path.exists(data):
                    run(f"python data_parallel.py \
                        --data-folder {DATA_FOLDER} \
                        --out-folder {data} \
                        --method {DP_methods_call[method]} \
                        --K {K}")

                stats_file = f"stats/{model}-K{K}-{method}-{task}.jsonl"
                if not os.path.exists(stats_file):
                    run(f"python run_all.py \
                        --model {MODEL_FOLDER}/{model} \
                        --data {data} \
                        --run tree_{task} \
                        --stats-out {stats_file}")
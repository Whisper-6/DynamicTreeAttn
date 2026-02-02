# 以下是一段 Python 脚本

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

import os

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

for model in models:
    for K in K_set:
        for method in DP_methods:
            for task in tasks:
                run(f"python run_all.py \
                    --model {MODEL_FOLDER}/{model} \
                    --data {DP_FOLDER}/K{K}-{method}-{model}-{task} \
                    --run tree_{task} \
                    --stats-out stats/{model}-K{K}-{method}-{task}.jsonl")
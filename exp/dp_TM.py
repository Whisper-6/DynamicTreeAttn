DATA_FOLDER = "data/tau2-16k-merged"
DP_FOLDER = "data/tau2-16k-dp"

models = [
  "Qwen3-1.7B",
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B"
]

K_set = [2, 4, 8]
DP_methods = ["FFD", "DFS"]

import os

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

# DP with TimeModel
for K in K_set:
    for method in DP_methods:
        for model in models:
            for task in tasks:
                run(f"python data_parallel.py \
                    --data-folder {DATA_FOLDER} \
                    --out-folder {DP_FOLDER}/K{K}-{method}-TM-{model}-{task} \
                    --stats-file stats/{model}-K{K}-{method}-tree-{task}.jsonl \
                    --method LB_by_TM \
                    --K {K}")
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
DP_methods_old = ["FFD-token", "FFD-tree", "DFS-tree"]
DP_methods = ["FFD-TM", "DFS-TM"]
DP_methods_call = {
    "FFD-TM" : "LB_by_TM",
    "DFS-TM" : "LB_by_DFS_and_TM"
}
block_size = 2048

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

                stats_woTM_file = f"stats/{model}-K{K}-woTM-{task}.jsonl"
                if not os.path.exists(stats_woTM_file):
                    for old_method in DP_methods_old:
                        old_stats = f"stats/{model}-K{K}-{old_method}-{task}.jsonl"
                        with open(old_stats, "r") as f_in, open(stats_woTM_file, "a") as f_out:
                            for line in f_in:
                                f_out.write(line)

                data = f"{DP_FOLDER}/K{K}-{method}-{model}-{task}"
                if not os.path.exists(data):
                    run(f"python data_parallel.py \
                        --data-folder {DATA_FOLDER} \
                        --out-folder {data} \
                        --stats-file {stats_woTM_file} \
                        --method {DP_methods_call[method]} \
                        --mode {task} \
                        --block-size {block_size} \
                        --K {K}")

                stats_file = f"stats/{model}-K{K}-{method}-{task}.jsonl"
                if not os.path.exists(stats_file):
                    run(f"python run_all.py \
                        --model {MODEL_FOLDER}/{model} \
                        --data {data} \
                        --run tree_{task} \
                        --stats-out {stats_file}")
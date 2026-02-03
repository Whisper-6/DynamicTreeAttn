MODEL_FOLDER = "/data/tree/models"
DATA_FOLDER = "data/tau2-16k-merged"

models = [
  "Qwen3-1.7B",
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B"
]

import os
import time

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

for model in models:
    os.makedirs(f"stats/{model}", exist_ok=True)

    model_path = os.path.join(MODEL_FOLDER, model)
    prefix = f"python run_all.py --model {model_path} --data {DATA_FOLDER}"

    print(f"Running experiments for model: {model}")
    
    print("  Tree Forward")
    stats_file = f"stats/{model}/TF.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_forward \
            --stats-out {stats_file}")

    print("  Tree Backward")
    stats_file = f"stats/{model}/TB.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_backward \
            --stats-out {stats_file}")

    print("  Tree Forward (random permute)")
    stats_file = f"stats/{model}/TF_ran.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_forward \
            --permute random \
            --stats-out {stats_file}")

    print("  Tree Backward (random permute, tailed)")
    stats_file = f"stats/{model}/TB_ran_tailed.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_backward \
            --permute random \
            --cut-f1-tail False \
            --stats-out {stats_file}")

    print("  Tree Backward (random permute)")
    stats_file = f"stats/{model}/TB_ran.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_backward \
            --permute random \
            --stats-out {stats_file}")

    print("  Tree Backward (Larger Block Size)")
    stats_file = f"stats/{model}/TB_LB.jsonl"
    if not os.path.exists(stats_file):
        run(f"{prefix} \
            --run tree_backward \
            --block-size 4096 \
            --stats-out {stats_file}")
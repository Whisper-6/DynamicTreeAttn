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

    print("  Tree Forward (random permute)")
    run(f"{prefix} \
        --run tree_forward \
        --permute random \
        --stats-out stats/{model}/TF_ran.jsonl")

    print("  Tree Forward")
    run(f"{prefix} \
        --run tree_forward \
        --stats-out stats/{model}/TF.jsonl")


    print("  Tree Backward (random permute, tailed)")
    run(f"{prefix} \
        --run tree_backward \
        --permute random \
        --cut-f1-tail False \
        --stats-out stats/{model}/TB_ran_tailed.jsonl")

    print("  Tree Backward (tailed)")
    run(f"{prefix} \
        --run tree_backward \
        --permute random \
        --stats-out stats/{model}/TB_ran.jsonl")

    print("  Tree Backward")
    run(f"{prefix} \
        --run tree_backward \
        --stats-out stats/{model}/TB.jsonl")
    
    print("  Tree Backward (Larger Block Size)")
    run(f"{prefix} \
        --run tree_backward \
        --block-size 4096 \
        --stats-out stats/{model}/TB_LB.jsonl")
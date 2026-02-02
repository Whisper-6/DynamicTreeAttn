MODEL_FOLDER = "/data/tree/models"
DATA_FOLDER = "data/tau2-16k-merged"

models = [
  "Qwen3-1.7B",
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B"
]

models_OOM = [
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B"
]

import os

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

for model in models:
    os.makedirs(f"stats/{model}", exist_ok=True)

    model_path = os.path.join(MODEL_FOLDER, model)
    prefix = f"python run_all.py --model {model_path} --data {DATA_FOLDER}"

    print(f"Running experiments for model: {model}")

    print("  Dense Forward")
    run(f"{prefix} \
        --run dense_forward \
        --stats-out stats/{model}/DF.jsonl")

    print("  Dense Forward with Leafization")
    run(f"{prefix} \
        --run dense_forward \
        --leafization True \
        --stats-out stats/{model}/DF_leafed.jsonl")


    if model not in models_OOM:
        print("  Dense Backward")
        run(f"{prefix} \
            --run dense_backward \
            --stats-out stats/{model}/DB.jsonl")

        print("  Dense Backward with Leafization")
        run(f"{prefix} \
            --run dense_backward \
            --leafization True \
            --stats-out stats/{model}/DB_leafed.jsonl")

    print("  Dense Backward with Activation Checkpointing")
    run(f"{prefix} \
        --run dense_backward \
        --act-ckpt True \
        --stats-out stats/{model}/DB_ckpt.jsonl")

    print("  Dense Backward with Leafization and Activation Checkpointing")
    run(f"{prefix} \
        --run dense_backward \
        --leafization True \
        --act-ckpt True \
        --stats-out stats/{model}/DB_leafed_ckpt.jsonl")
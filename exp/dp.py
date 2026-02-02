DATA_FOLDER = "data/tau2-16k-merged"
DP_FOLDER = "data/tau2-16k-dp"

K_set = [2, 4, 8]
DP_methods = ["FFD-token", "FFD-tree", "DFS-tree"]
DP_methods_call = ["LB_by_n_tokens", "LB_by_TM", "LB_by_DFS_and_TM"]

import os

def run(cmd: str):
    os.system(cmd)
    time.sleep(1)

# DP without TimeModel
for K in K_set:
    for method, method_call in zip(DP_methods, DP_methods_call):
        run(f"python data_parallel.py \
            --data-folder {DATA_FOLDER} \
            --out-folder {DP_FOLDER}/K{K}-{method} \
            --method {method_call} \
            --K {K}")
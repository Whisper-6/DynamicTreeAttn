# DynamicTreeAttn
Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models


python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run dense_forward

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run dense_backward

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_forward

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --block-size 2048

python run_all.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data-folder data/tau2-16k-merged \
  --run dense_forward

python run_all.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data-folder data/tau2-16k-merged \
  --run dense_backward

python run_all.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data-folder data/tau2-16k-merged \
  --run tree_forward

python run_all.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data-folder data/tau2-16k-merged \
  --run tree_backward \
  --block-size 2048

python run_dp.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --num-ranks 2

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --block-size 4096 \
  --permute ours
  --cut-tail

python run.py \
  --model /data/tree/models/Qwen3-0.6B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward

python run_all.py \
  --model /data/tree/models/Qwen3-0.6B \
  --data data/tau2-16k-merged \
  --run tree_forward \
  --stats-out stats/Qwen3-0.6B-tree-forward.jsonl

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-0.6B \
  --data-folder stats/data/tau2-16k-merged \
  --run tree_forward \
  --permute ours \
  --stats-out Qwen3-0.6B-forward.jsonl

python run_all.py \
  --model /data/tree/models/Qwen3-0.6B \
  --data data/tau2-16k-K8-DFS \
  --run tree_forward \
  --stats-out stats/Qwen3-0.6B-K8-DFS-forward.jsonl

python run_all.py \
  --model /data/tree/models/Qwen3-1.7B \
  --data data/tau2-16k-K8-DFS-TM-forward \
  --run tree_forward \
  --stats-out stats/Qwen3-1.7B-K8-DFS-TM-forward.jsonl


python run_all.py   --model /data/tree/models/Qwen3-4B   --data data/tau2-16k-merged   --run tree_backward   --stats-out stats/Qwen3-4B-tree-backward.jsonl

python tree_time_model.py --stats-file stats/Qwen3-0.6B-K8-DFS-forward.jsonl
python tree_time_model.py --stats-file stats/Qwen3-0.6B-forward.jsonl
python tree_time_model.py --stats-file stats/Qwen3-4B-backward.jsonl

docker exec -it areal bash
cd data/tree/DynamicTreeAttn

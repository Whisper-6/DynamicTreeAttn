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
  --run dense_backward

python run_all.py \
  --model /data/tree/models/Qwen3-1.7B \
  --data data/tau2-16k-merged \
  --run dense_forward \
  --stats-out stats/Qwen3-1.7B-forward.jsonl

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-0.6B \
  --data-folder stats/data/tau2-16k-merged \
  --run tree_forward \
  --permute ours \
  --stats-out Qwen3-0.6B-forward.jsonl



python data_stats.py --data-folder data/tau2-16k-merged

docker exec -it areal bash
cd data/tree/DynamicTreeAttn

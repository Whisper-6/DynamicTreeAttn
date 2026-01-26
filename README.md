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

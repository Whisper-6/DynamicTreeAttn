
python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_forward \
  --permute random

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_forward \
  --permute idx

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_forward \
  --permute ours


python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --block-size 4096 \
  --permute random

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --block-size 4096 \
  --permute idx

python run.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data data/tau2-16k-merged/call1.pt \
  --run tree_backward \
  --block-size 4096 \
  --permute ours

python run_all.py \
  --model-folder /data/tree/models \
  --model Qwen3-1.7B \
  --data-folder data/tau2-16k-small \
  --run dense_backward \
  --act-ckpt
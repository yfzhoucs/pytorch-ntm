export CUDA_VISIBLE_DEVICES=1
python train.py \
  --task add-one-onehot \
  -psequence_min_len=1 \
  -psequence_max_len=9 \
  -pmemory_n=512 \
  -pmemory_m=512 \
  -pname=add-one-onehot-expand-memory \
  --checkpoint-interval=50000

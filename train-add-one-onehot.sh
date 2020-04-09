export CUDA_VISIBLE_DEVICES=1
python train.py \
  --task add-one-onehot \
  -psequence_min_len=1 \
  -psequence_max_len=9 \
  --checkpoint-interval=50000

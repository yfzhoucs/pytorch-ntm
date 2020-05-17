export CUDA_VISIBLE_DEVICES=3
python train.py \
  --task add-bcd \
  -psequence_min_len=1 \
  -psequence_max_len=9 \
  -pnum_heads=2 \
  --checkpoint-interval=50000 \
  --checkpoint-path=./ckpts/

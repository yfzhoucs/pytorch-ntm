python test.py \
  --task add-one-onehot \
  -psequence_min_len=1 \
  -psequence_max_len=9 \
  -pmemory_n=512 \
  -pmemory_m=512 \
  --checkpoint_file='./ckpts/add-one-onehot-expand-memory-1000-batch-250000.model'

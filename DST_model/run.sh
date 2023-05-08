
CUDA_VISIBLE_DEVICES=2  python main.py \
    --save_prefix debugging \
    --do_train 1 \
    --do_short 1 \
    --seed 1 \
    --g 1\
    --max_epoch 1 \
    --train_path '../../woz_data/train_data.json' \
    --dev_path '../../woz_data/dev_data.json' \
    --test_path '../../woz_data/test_data.json' \
    --max_length 512 \
    --batch_size 8 \
    --test_batch_size 16 \




python train.py \
    --data_folder 'data/sample100_L160(test)' \
    --output_folder 'output/sample(test)' \
    --img_height 160 \
    --img_width 160 \
    --batch_size 32 \
    --val_batch_size 100 \
    --max_epoch 50 \
    --learning_rate 0.01 \
    --decay_epoch 10 \
    --device gpu
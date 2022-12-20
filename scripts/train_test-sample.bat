python train.py ^
    --data_folder data/sample10_L96(test) ^
    --output_folder output/sample(test) ^
    --img_height 96 ^
    --img_width 96 ^
    --batch_size 6 ^
    --val_batch_size 100 ^
    --max_epoch 5 ^
    --learning_rate 0.01 ^
    --decay_epoch 2 ^
    --device gpu ^
    --pin_memory_train True

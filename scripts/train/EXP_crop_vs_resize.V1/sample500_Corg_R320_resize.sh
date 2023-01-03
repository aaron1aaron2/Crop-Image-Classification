# run on colab
expname='EXP_crop_vs_resize/sample500_Lorg_resize'
python train.py \
    --data_folder "data/${expname}" \
    --output_folder "output/${expname}" \
    --img_height 320 \
    --img_width 320 \
    --batch_size 16 \
    --val_batch_size 64 \
    --max_epoch 10 \
    --learning_rate 0.01 \
    --decay_epoch 3 \
    --decay_rate 0.5 \
    --device gpu
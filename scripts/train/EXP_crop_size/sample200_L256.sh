# run on colab
expname='EXP_crop_size/sample200_L256'
python train.py \
    --data_folder "data/${expname}" \
    --output_folder "output/${expname}" \
    --img_height 256 \
    --img_width 256 \
    --batch_size 5 \
    --val_batch_size 64 \
    --max_epoch 10 \
    --learning_rate 0.01 \
    --decay_epoch 3 \
    --device gpu
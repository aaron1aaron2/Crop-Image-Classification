# run on colab
expname='sample200_L480'
python train.py \
    --data_folder "data/sample_nums_experiment/${expname}" \
    --output_folder "output/sample_nums_experiment/${expnames}" \
    --img_height 480 \
    --img_width 480 \
    --batch_size 5 \
    --val_batch_size 64 \
    --max_epoch 10 \
    --learning_rate 0.01 \
    --decay_epoch 10 \
    --device gpu
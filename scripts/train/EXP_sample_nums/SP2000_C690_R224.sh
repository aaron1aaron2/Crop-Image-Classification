# run on kaggle P100
expname='EXP_sample_nums/SP2000_C690_R224'
python train.py \
    --data_folder "data/${expname}" \
    --output_folder "output/${expname}" \
    --img_height 224 \
    --img_width 224 \
    --batch_size 40 \
    --val_batch_size 100 \
    --max_epoch 25 \
    --learning_rate 0.01 \
    --decay_epoch 5 \
    --decay_rate 0.5 \
    --patience 5 \
    --device gpu

python scripts/image_process.py ^
    --file_folder data/predata ^
    --class_list_path data/class_ls.txt ^
    --img_coordinate_path data/tag_locCoor.csv ^
    --output_folder data/data-all_L480 ^
    --sample False ^
    --crop_length 480 ^
    --train_ratio 0.7 ^
    --val_ratio 0.1 ^
    --test_ratio 0.2
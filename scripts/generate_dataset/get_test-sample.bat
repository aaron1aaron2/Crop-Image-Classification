python scripts/image_process.py ^
    --file_folder data/predata ^
    --class_list_path data/class_ls.txt ^
    --img_coordinate_path data/tag_locCoor.csv ^
    --output_folder data/sample10_L160(test) ^
    --sample True ^
    --sample_num_per_class 10 ^
    --crop_length 160 ^
    --train_ratio 0.7 ^
    --val_ratio 0.1 ^
    --test_ratio 0.2

python scripts/image_process.py ^
    --file_folder data/predata ^
    --class_list_path data/class_ls.txt ^
    --img_coordinate_path data/tag_locCoor.csv ^
    --output_folder data/sample10_L160_center(test) ^
    --sample_file data/sample10_L160(test)/image_info.csv ^
    --crop_length 160 ^
    --crop_center True ^
    --train_ratio 0.7 ^
    --val_ratio 0.1 ^
    --test_ratio 0.2
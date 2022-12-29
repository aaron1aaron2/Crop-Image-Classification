python scripts/image_process.py ^
    --file_folder data/predata ^
    --class_list_path data/class_ls.txt ^
    --img_coordinate_path data/tag_locCoor.csv ^
    --output_folder data/EXP_crop_vs_resize/sample500_L690 ^
    --sample_file data/EXP_crop_vs_resize/sample500_Lorg_resize/image_info.csv ^
    --crop_length 690 ^
    --train_ratio 0.7 ^
    --val_ratio 0.1 ^
    --test_ratio 0.2
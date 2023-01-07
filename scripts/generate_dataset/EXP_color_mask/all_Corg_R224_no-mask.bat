@REM same as: scripts\generate_dataset\EXP_crop_vs_resize.V2\all_Corg_R224_resize.bat
python scripts/image_process.py ^
    --file_folder data/predata ^
    --class_list_path data/class_ls.txt ^
    --img_coordinate_path data/tag_locCoor.csv ^
    --output_folder data/EXP_color_mask/all_Corg_R224_no-mask ^
    --sample_file data/EXP_crop_vs_resize.V2/all_Corg_R224/image_info.csv ^
    --crop_image False ^
    --resize_image True ^
    --resize_length 224 ^
    --train_ratio 0.7 ^
    --val_ratio 0.1 ^
    --test_ratio 0.2
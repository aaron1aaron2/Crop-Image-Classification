"""
Author: 何彥南 (yen-nan ho)
Github: https://github.com/aaron1aaron2
Email: aaron1aaron2@gmail.com
Create Date: 2022.12.01
Last Update: 2022.12.01
Describe: 資料前處理、產生訓練資料(train、val、test)
"""
import os
import cv2
import glob
import tqdm
import imghdr
import random
import argparse


import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # input
    parser.add_argument('--file_folder', type=str, default='data/rawdata')
    parser.add_argument('--class_list_path', type=str, default='data/class_ls.txt')

    parser.add_argument('--img_coordinate_path', type=str, default='data/tag_locCoor.csv')

    # output
    parser.add_argument('--output_folder', type=str, default='data/sample_baseline')

    # parameters
    parser.add_argument('--sample_num_per_class', type=int, default=200)


    return parser.parse_args()

def crop_img_target(img, plant_class, img_name, img_shape, crop_length):
    """
    get cropped img with raw img & target info
    """
    img_tag_Y_plant_df = img_tag_Y_df_dict[plant_class]

    ## targey info(準心資訊)
    target_df = img_tag_Y_plant_df[img_tag_Y_plant_df["Img"] == img_name] 
    target_y = target_df["target_y"]
    target_x = target_df["target_x"]
    
    ## img shape info
    img_h = img_shape[0]  # height
    img_w = img_shape[1]  # width
    orig_y = img_h/2  # origin of coordinates in each img(原點座標)
    orig_x = img_w/2
    
    ## target location(準心位置)
    aim_y = int(orig_x + target_y)
    aim_x = int(orig_y + target_x)
    
    ## crop image with target
    crop_h_lower, crop_h_upper = int(aim_y - crop_length), int(aim_y + crop_length)
    crop_w_lower, crop_w_upper = int(aim_x - crop_length), int(aim_x + crop_length)
    
    crop_img = img[crop_w_lower:crop_w_upper, crop_h_lower:crop_h_upper]   
    return img_name, crop_img

def crop_img():
    pass

def main():
    args = get_args()

    # 資料讀取 >>>>>>>>>>>
    os.makedirs(args.output_folder, exist_ok=True)
    class_detected_ls = os.listdir(args.file_folder)
    class_folder_ls = [(i, os.path.join(args.file_folder, i)) for i in class_detected_ls]

    coor_df = pd.read_csv('data/tag_locCoor.csv', encoding='big5', usecols=['TARGET_FID', 'Img', 'target_x', 'target_y'])
    with open(args.class_list_path, encoding='utf8') as f:
        CLASS_ls = [i.strip() for i in f.readlines()]

    # 檢查各類別資料夾存在
    absent_data = list(set(CLASS_ls) - set(class_detected_ls))
    assert len(absent_data) == 0, f"Absent data folders under '{args.file_folder}': {absent_data}"
    
    # 沒用到但是存在的資料夾
    not_used_data = list(set(class_detected_ls) - set(CLASS_ls))
    if len(not_used_data) != 0: 
        print('Category folders not used: ', not_used_data)
    # <<<<<<<<<<<<<<<<<<<


    # 整理路徑、類別資訊到表上
    print('searching image..')
    image_info = []
    for label, class_folder in tqdm.tqdm(class_folder_ls):
        img_path_ls = glob.glob(f'{class_folder}/**/*.jpg', recursive=True)
        img_ls = [os.path.basename(i) for i in img_path_ls]
        label_ls = [label]*len(img_ls)
        image_info.extend(tuple(zip(label_ls, img_ls, img_path_ls)))
    
    coor_df = coor_df.merge(
        pd.DataFrame(image_info, columns=['label', 'Img', 'path']),
        how='left', on='Img'
        )
    
    coor_df['no_coor'] = (coor_df[['target_x', 'target_y']] == 0).all(axis=1)

    no_coor_ct = coor_df['no_coor'].value_counts()
    print('image have coordinate: {}/{} ({:.2f}%)'.format(
        no_coor_ct[False], 
        no_coor_ct[True],
        no_coor_ct[False]/no_coor_ct[True]
        ))

    coor_df.to_csv(os.path.join(args.output_folder, 'image_list.csv'), index=False)



if __name__ == '__main__':
    main()
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
import imghdr
import random
import argparse

from IPython import embed

CLASS_ls = []

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # input
    parser.add_argument('--file_folder', type=str, default='data/rawdata')
    parser.add_argument('--class_list_path', type=str, default='data/class_ls.txt')

    parser.add_argument('--img_coordinate_path', type=str, default='data/tag_locCoor.csv')
    parser.add_argument('--coordinate_X', type=str, default='target_x', 
                            help='The coordinate name of X in the coordinate file')
    parser.add_argument('--coordinate_Y', type=str, default='target_y', 
                            help='The coordinate name of Y in the coordinate file')

    # output
    parser.add_argument('--output_folder', type=str, default='data/sample200')

    # parameters
    parser.add_argument('--sample_num_per_class', type=int, default=200)


    return parser.parse_args()

def get_class(label_code):    
    return CLASS_ls[label_code]

# def read_img(path, image_ls):
#     images_sample = random.shuffle(images)[:sample_num_per_class]


def img_tag_Y_list(plant_class):
    """
    get img_name list w/ tag in eacg plant_class
    """
    img_tag_Y_df_dict = {}
    img_tag_Y_list_dict = {}
    img_tag_Y_list_df = img_tag_Y[img_tag_Y["Img"].isin(img_name_plant_class[plant_class])]
    img_tag_Y_df_dict[plant_class] = img_tag_Y_list_df
    img_tag_Y_list_dict[plant_class] = img_tag_Y_list_df["Img"].tolist()

def crop_img_target(img_data, plant_class, img_name, img_shape, crop_length):
    """
    get cropped img with raw img & target info
    """
    img = img_data
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


def main():
    args = get_args()

    class_detected_ls = os.listdir(args.file_folder)
    class_folder_ls = [os.path.join(args.file_folder, i) for i in class_detected_ls]

    with open(args.class_list_path, encoding='utf8') as f:
        CLASS_ls = [i.strip() for i in f.readlines()]
    embed()
    exit()

    # 檢查各類別資料夾存在
    absent_data = list(set(CLASS_ls) - set(class_detected_ls))
    assert len(absent_data) == 0, f"Absent data folders under '{args.file_folder}': {absent_data}"
    
    # 沒用到但是存在的資料夾
    not_used_data = list(set(class_detected_ls) - set(CLASS_ls))
    if len(not_used_data) != 0: 
        print('Category folders not used: ', not_used_data)


    for plant_class in plant_class_list:
        img_tag_Y_list(plant_class)
        
    for k, v in img_tag_Y_list_dict.items():
        print("{}:  Numbers of crop_img = {}".format(k, len(v)))

    Error_list = []
    for plant_class, img_list in progress_bar(img_tag_Y_list_dict.items()):
        for img_name in progress_bar(img_list):
            img_data, img_shape = get_img_shape(plant_class, img_name)
            try:
                img_name_AA, crop_img_AA = crop_img_target(img_data, plant_class, img_name, img_shape_AA, 95)
                save_crop_img(img_name_AA, crop_img_AA)
            except Exception as ee:
                Error_list.append(ee)

if __name__ == '__main__':
    main()
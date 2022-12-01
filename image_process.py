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

CLASS_ls = [
    'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower',
    'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 'greenhouse',
    'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah',
    'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
    'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 
    'taro', 'tea', 'waterbamboo'
    ]

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--file_folder', type=str, default='data/rawdata')
    parser.add_argument('--output_folder', type=str, default='data/sample200')
    parser.add_argument('--class_list', type=str, default='data/class_ls.txt')


    parser.add_argument('--sample_num_per_class', type=int, default=200)


    return parser.parse_args()

def get_class(label_code):    
    return CLASS_ls[label_code]

# def read_img(path, image_ls):
#     images_sample = random.shuffle(images)[:sample_num_per_class]

img_tag_Y_df_dict = {}
img_tag_Y_list_dict = {}

def img_tag_Y_list(plant_class):
    """
    get img_name list w/ tag in eacg plant_class
    """
    img_tag_Y_list_df = img_tag_Y[img_tag_Y["Img"].isin(img_name_plant_class[plant_class])]
    img_tag_Y_df_dict[plant_class] = img_tag_Y_list_df
    img_tag_Y_list_dict[plant_class] = img_tag_Y_list_df["Img"].tolist()

for plant_class in plant_class_list:
    img_tag_Y_list(plant_class)
    
for k, v in img_tag_Y_list_dict.items():
    print("{}:  Numbers of crop_img = {}".format(k, len(v)))

def main():
    args = get_args()
    file_ls = [os.path.join(root_path, i) for i in os.listdir(root_path)]


if __name__ == '__main__':
    main()
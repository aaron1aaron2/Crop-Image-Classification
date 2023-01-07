"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.12.01
Last Update: 2023.1.04
Describe: 資料前處理、產生訓練資料(train、val、test)
    - 抽樣方法 -> panda DataFrameGroupBy.sample
    - 圖片剪裁、縮放方法(inter)、色彩 mask(HSV) -> openCV
"""
import os
import cv2
import glob
import tqdm
import json
import shutil
import argparse

import pandas as pd
import numpy as np

from typing import Tuple

def get_args():
    parser = argparse.ArgumentParser(add_help=False)

    # input
    parser.add_argument('--file_folder', type=str, default='data/predata')
    parser.add_argument('--class_list_path', type=str, default='data/class_ls.txt')

    parser.add_argument('--img_coordinate_path', type=str, default='data/tag_locCoor.csv')

    # output
    parser.add_argument('--output_folder', type=str, default='data/sample')

    # sampling parameters
    parser.add_argument('--sample', type=str2bool, default=True)
    parser.add_argument('--sample_num_per_class', type=int, default=100)
    parser.add_argument('--sample_file', type=str, default=None, help='使用已有的 sample(image_info.csv)，確保 sample 是一樣的')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # process parameters
    parser.add_argument('--crop_image', type=str2bool, default=True)
    parser.add_argument('--crop_length', type=int, default=200)
    parser.add_argument('--crop_center', type=str2bool, default=False)

    parser.add_argument('--resize_image', type=str2bool, default=False)
    parser.add_argument('--resize_length', type=int, default=200)

    parser.add_argument('--mask_image', type=str2bool, default=False)
    parser.add_argument('--mask_lower', type=int, nargs='+', default=(10, 0, 0), 
                            help='hsv lowerBound')
    parser.add_argument('--mask_upper', type=int, nargs='+', default=(80, 255, 255), 
                            help='hsv upperBound')

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def crop_img_target(img:np.ndarray, crop_length:int, target_x:int, target_y:int) -> np.ndarray:
    ## img shape info
    img_h, img_w, _ = img.shape
    crop_length_half = int(crop_length/2)

    ## target location(準心位置) + 防止超出照片
    border_h = img_h/2 - crop_length_half
    border_w = img_w/2 - crop_length_half

    aim_y = int(img_h - crop_length_half if (target_y > border_h) else (
                crop_length_half if (abs(target_y) > border_h) else img_h/2 + target_y
            )) # 上界 -> 下界 -> 正常狀況
    aim_x = int(img_w - crop_length_half if (target_x > border_w) else (
                crop_length_half if (abs(target_x) > border_w) else img_w/2 + target_x
            )) # 上界 -> 下界 -> 正常狀況

    ## crop image with target
    crop_h_lower, crop_h_upper = int(aim_y - crop_length_half), int(aim_y + crop_length_half)
    crop_w_lower, crop_w_upper = int(aim_x - crop_length_half), int(aim_x + crop_length_half)

    crop_img = img[crop_h_lower:crop_h_upper, crop_w_lower:crop_w_upper]  

    return crop_img

def mask_img(img:np.ndarray, lower:Tuple[int, ...], upper:Tuple[int, ...]) -> np.ndarray:
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    mask = cv2.inRange(hsv, tuple(lower), tuple(upper))

    ## slice the green
    imask = mask > 0
    result = np.zeros_like(img, np.uint8)
    result[imask] = img[imask]

    return result

def copy_img(in_path:str, out_path:str) -> None:
    shutil.copyfile(in_path, out_path)

def main():
    args = get_args()
    print('\n', args, '\n')

    # 資料讀取 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    class_detected_ls = os.listdir(args.file_folder)
    class_folder_ls = [(i, os.path.join(args.file_folder, i)) for i in class_detected_ls]

    os.makedirs(os.path.join(args.output_folder, 'error'), exist_ok=True)
    for split in ['train', 'val', 'test']:
        for class_ in class_detected_ls:
            os.makedirs(os.path.join(args.output_folder, split, class_), exist_ok=True)

    coor_df = pd.read_csv(args.img_coordinate_path, encoding='big5', usecols=['TARGET_FID', 'Img', 'target_x', 'target_y'])
    
    with open(args.class_list_path, encoding='utf8') as f:
        CLASS_ls = [i.strip() for i in f.readlines()]

    # 檢查各類別資料夾存在
    absent_data = list(set(CLASS_ls) - set(class_detected_ls))
    assert len(absent_data) == 0, f"Absent data folders under '{args.file_folder}': {absent_data}"

    # 沒用到但是存在的資料夾
    not_used_data = list(set(class_detected_ls) - set(CLASS_ls))
    if len(not_used_data) != 0: 
        print('Category folders not used: ', not_used_data)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 整理路徑、類別資訊到表上 >>>>>>>>>>>>>>>>>>>>>
    print('Searching image..')
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

    print('image have coordinate: {}/{} ({:.2f}%)\n'.format(
        no_coor_ct[False], 
        len(coor_df),
        no_coor_ct[False]/len(coor_df)
        ))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 抽樣 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.sample_file:
        print(f'Load sample from: {args.sample_file}\n')
        coor_df = pd.read_csv(args.sample_file)
    else:
        # 各類別數量上限
        if args.sample:
            print('sampling data...')
            coor_df = coor_df.groupby('label').sample(n=args.sample_num_per_class)
        else:
            print('Use all data...')

        #  train、test、val
        train = coor_df.groupby('label').sample(frac=args.train_ratio) # frac 用比例抽
        tmp = coor_df.drop(train.index) # val & test

        new_val_rate = args.val_ratio/(args.test_ratio + args.val_ratio)
        val = tmp.groupby('label').sample(frac=new_val_rate)
        test = tmp.drop(val.index)

        n = len(coor_df)
        train_n, val_n, test_n = train.shape[0], val.shape[0], test.shape[0]
        print(
            'Train: {} ({:.2f}%)'.format(train_n, train_n/n),
            '\nVal: {} ({:.2f}%)'.format(val_n, val_n/n),
            '\nTest: {} ({:.2f}%)\n'.format(test_n, test_n/n)
        )

        coor_df.loc[train.index, 'split'] = 'train'
        coor_df.loc[val.index, 'split'] = 'val'
        coor_df.loc[test.index, 'split'] = 'test'
        del train; del val; del test

        coor_df.sort_values('TARGET_FID', inplace=True)
    
    coor_df.to_csv(os.path.join(args.output_folder, 'image_info.csv'), index=False)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 準心中心裁切 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    img_dt = coor_df.to_dict(orient='records')
    if args.crop_image | args.resize_image | args.mask_image:
        print('Image processing...')
        print(f'Croping({args.crop_image}) -> Resizing({args.resize_image}) -> Masking({args.mask_image})\n')
    else:
        print('Copying image..\n')

    error_ls = []
    for i in tqdm.tqdm(img_dt):
        try:
            if (args.crop_image) | (args.resize_image):
                # read image
                img = cv2.imread(i['path'])
                
                # crop image
                if args.crop_image:
                    x, y = (0, 0) if args.crop_center else (i['target_x'], i['target_y'])
                    img = crop_img_target(img, args.crop_length, x, y)

                # resize image
                if args.resize_image:
                    img = cv2.resize(
                            src=img, 
                            dsize=(args.resize_length, args.resize_length), 
                            interpolation=cv2.INTER_AREA
                            )
                
                # mask image
                if args.mask_image:
                    img = mask_img(img, args.mask_lower, args.mask_upper)

                # output
                cv2.imwrite(os.path.join(args.output_folder, i['split'], i['label'], i['Img']), img)
            else:
                copy_img(i['path'], os.path.join(args.output_folder, i['split'], i['label'], i['Img']))

        except Exception as e:
            i['error_msg'] = str(e)
            error_ls.append(i)

    with open(os.path.join(args.output_folder, 'error.json'), 'w', encoding='utf-8') as outfile:  
        json.dump(error_ls, outfile, indent=2, ensure_ascii=False)

    if len(error_ls) == 0:
        shutil.rmtree(os.path.join(args.output_folder, 'error'), ignore_errors=True)

    print('Error num:', len(error_ls))
    print('\nFinish!!')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == '__main__':
    main()
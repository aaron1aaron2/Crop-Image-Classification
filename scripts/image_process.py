"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.12.01
Last Update: 2022.12.03
Describe: 資料前處理、產生訓練資料(train、val、test)
    - 抽樣方法 -> panda DataFrameGroupBy.sample
    - 圖片剪裁方法 -> openCV
"""
import os
import cv2
import glob
import tqdm
import random
import argparse


import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # input
    parser.add_argument('--file_folder', type=str, default='data/predata')
    parser.add_argument('--class_list_path', type=str, default='data/class_ls.txt')

    parser.add_argument('--img_coordinate_path', type=str, default='data/tag_locCoor.csv')

    # output
    parser.add_argument('--output_folder', type=str, default='data/sample100')

    # parameters
    parser.add_argument('--sample', type=bool, default=True)
    parser.add_argument('--sample_num_per_class', type=int, default=100)

    parser.add_argument('--train_ratio', type=int, default=0.7)
    parser.add_argument('--val_ratio', type=int, default=0.1)
    parser.add_argument('--test_ratio', type=int, default=0.2)


    return parser.parse_args()

def crop_img_target(img_path, img_name, label, crop_length, target_x, target_y, output):
    img = cv2.imread(img_path)

    ## img shape info
    img_h, img_w, _ = img.shape

    
    crop_length_half = int(crop_length/2)

    ## target location(準心位置) + 防止超出照片
    aim_y = int(img_h/2 + target_y) if (target_y < img_h/2) else int(img_h-crop_length_half)
    aim_x = int(img_w/2 + target_x) if (target_x < img_w/2) else int(img_w-crop_length_half)

    ## crop image with target
    crop_h_lower, crop_h_upper = int(aim_y - crop_length_half), int(aim_y + crop_length_half)
    crop_w_lower, crop_w_upper = int(aim_x - crop_length_half), int(aim_x + crop_length_half)

    crop_img = img[crop_h_lower:crop_h_upper, crop_w_lower:crop_w_upper]  

    ## output
    cv2.imwrite(output, crop_img)
    # cv2.imwrite(output.replace('.jpg', '_org.jpg'), img)

    return img_name, crop_img

def main():
    args = get_args()

    # 資料讀取 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # os.makedirs(args.output_folder, exist_ok=True)
    for i in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_folder, i), exist_ok=True)
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
        no_coor_ct[True],
        no_coor_ct[False]/no_coor_ct[True]
        ))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 抽樣 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 各類別數量上限
    if args.sample:
        coor_df = coor_df.groupby('label').sample(n=args.sample_num_per_class)

    #  train、test、val
    train = coor_df.groupby('label').sample(frac=args.train_ratio) # frac 用比例抽
    tmp = coor_df.drop(train.index) # val & test

    new_val_rate = args.val_ratio/(args.test_ratio + args.val_ratio)
    val = tmp.groupby('label').sample(frac=new_val_rate)
    test = tmp.drop(val.index)

    n = len(coor_df)
    train_n, val_n, test_n = train.shape[0], val.shape[0], test.shape[0]
    print(
        'Train: {} ({:.2f}%)\n'.format(train_n, train_n/n),
        'Val: {} ({:.2f}%)\n'.format(val_n, val_n/n),
        'Test: {} ({:.2f}%)\n'.format(test_n, test_n/n)
    )

    coor_df.loc[train.index, 'split'] = 'train'
    coor_df.loc[val.index, 'split'] = 'val'
    coor_df.loc[test.index, 'split'] = 'test'
    del train; del val; del test

    coor_df.sort_values('TARGET_FID', inplace=True)
    coor_df.to_csv(os.path.join(args.output_folder, 'image_info.csv'), index=False)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 準心圍中心裁切 >>>>>>>>>>>>>>>>>>>>>
    print('Croping image..\n')
    img_dt = coor_df.to_dict(orient='records')

    error_ls = []
    for i in tqdm.tqdm(img_dt):
        try:
            crop_img_target(
                i['path'], i['Img'], i['label'], 200,   
                i['target_x'], i['target_y'], 
                os.path.join(args.output_folder, i['split'], i['Img'])
                )
        except:
            error_ls.append(i['TARGET_FID'])
            exit()
    
    with open(os.path.join(args.output_folder, 'error.txt')) as f:
        f.writelines(error_ls)

    print('Error num:', len(error_ls))
    print('\n\n Finish!!')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



if __name__ == '__main__':
    main()
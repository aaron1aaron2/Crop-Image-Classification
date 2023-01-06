"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2023.1.05
Last Update: 2023.1.05
Describe: 抽取圖片中的 rgb 資訊
"""
import os
import json
import argparse
import extcolors

import pandas as pd

from tqdm.auto import tqdm
# from colormap import rgb2hex
from multiprocessing import Pool, cpu_count

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='data/sample10_L96(test)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    path = os.path.join(args.data_folder, 'image_info.csv')

    print(f'Load data from {path}\n')
    df = pd.read_csv(path)

    print(f'number of cpus available: {cpu_count()}')
    print(f'Extracting color...\n')

    result = pd.DataFrame()
    color_embed_dt = {}
    for path, imgname in tqdm(df[['path', 'Img']].values):

        colors_x, _ = extcolors.extract_from_path(path, tolerance = 12, limit = 12)
        df_color = pd.DataFrame(colors_x, columns=['rgb', 'count'])
        df_color['rate'] = df_color['count'] / df_color['count'].sum()
        df_color['img'] = imgname

        color_embed = [list(i[0]) + [i[1]] for i in df_color[['rgb', 'rate']].values]
        color_embed_dt[imgname] = color_embed

        result = pd.concat([result, df_color])

    # TODO 多進程
    # lists = range(100)

    # pool = Pool(8)

    # pool.map(test, lists)

    # pool.close()

    # pool.join()

    data_path =os.path.join(args.data_folder, 'img_color.json')
    print(f'data has been stored as {data_path}')
    with open(data_path, 'w', encoding='utf-8') as outfile:  
        json.dump(color_embed, outfile, indent=2, ensure_ascii=False)

    result.to_csv(os.path.join(args.data_folder, 'img_color.csv'), index=False)
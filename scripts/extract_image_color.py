"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2023.1.05
Last Update: 2023.1.05
Describe: 資料前處理、產生訓練資料(train、val、test)
    - 抽樣方法 -> panda DataFrameGroupBy.sample
    - 圖片剪裁 & 縮放方法 -> openCV
"""
import json
import argparse
import extcolors

import pandas as pd

from colormap import rgb2hex

img_url = "data/predata/asparagus/0a4b5fe3-c9f9-4175-a863-53759a8b7cd7.jpg"
colors_x = extcolors.extract_from_path(img_url, tolerance = 12, limit = 12)

def color_to_df(input):
    colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
    #convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                            int(i.split(", ")[1]),
                            int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df

df_color = color_to_df(colors_x)
df_color


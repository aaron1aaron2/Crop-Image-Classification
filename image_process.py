import os
import math
import zipfile
import cv2
import imghdr


import numpy as np
import matplotlib.pyplot as plot

from random import randint
from datetime import datetime

root_path = 'rawdata'
file_ls = [os.path.join(root_path, i) for i in os.listdir(root_path) if i.find('.zip') != -1]

class_ls = [
    'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower',
    'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 'greenhouse',
    'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah',
    'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
    'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 
    'taro', 'tea', 'waterbamboo'
    ]


# code_name_dt = {i:name for i, name in enumerate(class_ls)} 

def get_class(label_code):    
    return class_ls[label_code]

def output_zip(path, image_ls):
    # 建立 ZIP 壓縮檔
    with zipfile.ZipFile(path, mode='w') as zf:
        # 加入要壓縮的檔案
        for n, i in image_ls:
            zf.writestr(n, i)

for f in file_ls:
    with zipfile.ZipFile(f, 'r') as zf:
        images = zf.namelist()
"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.12.29
Last Update: 2022.12.29
Describe: 檢查圖片大小
"""
import os
import argparse

from PIL import Image 

folder_images = "dat/predata"
size_images = {}

for dirpath, _, filenames in os.walk(folder_images):
    for path_image in filenames:
        image = os.path.abspath(os.path.join(dirpath, path_image))
        with Image.open(image) as img:
            width, heigth = img.size
            size_images[path_image] = {'width': width, 'heigth': heigth}

print(size_images)
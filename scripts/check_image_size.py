"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.12.29
Last Update: 2022.12.29
Describe: 檢查圖片大小
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image 

folder_images = "data/predata"
size_images = {}

for dirpath, _, filenames in os.walk(folder_images):
    for path_image in filenames:
        image = os.path.abspath(os.path.join(dirpath, path_image))
        with Image.open(image) as img:
            width, heigth = img.size
            size_images[path_image] = {'width': width, 'heigth': heigth}

df = pd.DataFrame(size_images).T

print('width:', df['width'].min()) # 690
print('heigth:', df['heigth'].min()) # 690


df['width'].plot.hist()
df['heigth'].plot.hist()

plt.show()
print(size_images)
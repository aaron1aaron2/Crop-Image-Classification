"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.09.05
Last Update: 2022.09.05
Describe: 工具箱
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.pyplot import MultipleLocator

from pathlib import Path

def build_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def saveJson(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:  
        json.dump(data, outfile, indent=2, ensure_ascii=False)

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.clf()
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')

    # gca() 獲取當前的 axis
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # 不要小數點
    plt.gca().xaxis.set_major_locator(MultipleLocator(1)) # 間隔1


    plt.savefig(file_path)


def to_prob(nums):
    nums = np.array(nums)
    nums = nums + abs(nums.min())
    nums = nums / nums.sum()
    
    return nums.tolist()
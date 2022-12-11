"""
Author: 何彥南 (yen-nan ho)
Email: aaron1aaron2@gmail.com
Create Date: 2022.09.05
Last Update: 2022.09.05
Describe: 工具箱
"""
import json
import time

import matplotlib.pyplot as plt

from pathlib import Path
from functools import wraps

def build_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def saveJson(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:  
        json.dump(data, outfile, indent=2, ensure_ascii=False)

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def timer(func):
    @wraps(func) # 讓屬性的 name 維持原本
    def wrap(*args, **kargs):
        t_start = time.time()
        value = func(*args, **kargs)
        t_end = time.time()
        t_count = t_end - t_start
        print(f'[timeuse] {round(t_count, 5)} seconds')
        return value
    return wrap

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)
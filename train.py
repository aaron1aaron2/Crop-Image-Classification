"""
[Mender]
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2022.12.10
Last Update: 2022.12.19

[Original]
Author: 林政委
GitHub: https://github.com/VincLee8188/GMAN-PyTorch

Describe: train pipeline

[ISSUE]
1. pin_memory under torch.utils.data.DataLoader' 使用問題
    -> 在我的電腦上使用 GPU 測，記憶體用量差不多，但是 False 的時候速度比較快
2. coatnet 模型本身限制
    -> img_height、img_width 必須是 32 倍數。(預測 5 層就要需要整除 2^5 次的長寬。)
3. 追蹤 dataloader 後對應的 image path
    -> 自定義 datasets.ImageFolder 的 __getitem__ 方法。(https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d)
"""
import os
import time
import psutil
import platform
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms

from utils.utils import *
from utils.eval_metrics import get_evaluation, plot_confusion_matrix
from model.coatnet import CoAtNet


def get_args():
    parser = argparse.ArgumentParser()

    # 輸入
    parser.add_argument('--data_folder', type=str, default='data/test_sample10_L96')

    # 輸出
    parser.add_argument('--output_folder', type=str, default='output/test_sample')
    parser.add_argument('--use_tracedmodule', type=str2bool, default=True)
    parser.add_argument('--auto_save_model', type=str2bool, default=True, help='save model when improve')

    # 前處理
    parser.add_argument('--img_height', type=int, default=128) # 32 倍數
    parser.add_argument('--img_width', type=int, default=128) # 32 倍數
    parser.add_argument('--img_nor_mean', type=float, nargs='+', default=(0.4914, 0.4822, 0.4465), help='mean in torchvision.transforms.Normalize')
    parser.add_argument('--img_nor_std', type=float, nargs='+', default=(0.2023, 0.1994, 0.2010), help='std in torchvision.transforms.Normalize')

    # coatnet 參數
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 2, 12, 28, 2], help='Set num_blocks')  # python arg.py -l 2 2 12 28 2
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 64, 128, 256, 512], help='Set channels') 
    parser.add_argument('--in_channels', type=int, default=3, help='Set in_channels')
    parser.add_argument('--train_prob', type=str2bool, default=False, help='The output of the last layer is converted into a probability')  # python arg.py -l 2 2 12 28 2
    parser.add_argument('--prob', type=str2bool, default=True, help='The output of the last layer is converted into a probability')  # python arg.py -l 2 2 12 28 2

    # 超參數
    parser.add_argument('--batch_size', type=int, default=6,
                        help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=100,
                        help='val batch size')
    parser.add_argument('--max_epoch', type=int, default=5, # 50
                        help='epoch to run')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='decay epoch')

    # 其他
    parser.add_argument('--device', default='gpu', 
                        help='cpu or cuda')
    parser.add_argument('--pin_memory_train', type=str2bool, default=False, 
                        help='argument under torch.utils.data.DataLoader')

    args = parser.parse_args()

    return args


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def check_args(args):
    # 新增參數
    args.device = 'cuda' if torch.cuda.is_available() and args.device in ['gpu', 'cuda'] else 'cpu'
    args.train_folder = os.path.join(args.data_folder, 'train')
    args.test_folder = os.path.join(args.data_folder, 'test')
    args.val_folder = os.path.join(args.data_folder, 'val')

    args.model_file = os.path.join(args.output_folder, 
        'model.pth' if args.use_tracedmodule else 'model.pkl'
    )

    # 檢查
    for path in [args.train_folder, args.test_folder, args.val_folder]:
        assert os.path.exists(path), f'Data not found at {path}'

    assert len(args.num_blocks) == len(args.channels), '--num_blocks & --channels arguments must be of the same length'

    # 路徑
    args.fig_folder = os.path.join(args.output_folder, 'figure')

    build_folder(args.output_folder)
    # build_folder(fig_folder)

    # config
    saveJson(args.__dict__, os.path.join(args.output_folder, 'configures.json'))

    return args


def log_system_info(args, log):
    message = f"Computer network name: {platform.node()}\n"+ \
                f"Machine type: {platform.machine()}\n" + \
                f"Processor type: {platform.processor()}\n" + \
                f"Platform type: {platform.platform()}\n" + \
                f"Number of physical cores: {psutil.cpu_count(logical=False)}\n" + \
                f"Number of logical cores: {psutil.cpu_count(logical=True)}\n" + \
                f"Max CPU frequency: {psutil.cpu_freq().max}\n"

    cuda_divice = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    message += f'Train with the {args.device}({cuda_divice})\n'
    log_string(log, '\n[System Info]\n' + message + '='*20)


def load_data(args, log, eval_stage=False):
    transform_train = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(args.img_nor_mean, args.img_nor_std),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.img_nor_mean, args.img_nor_std),
    ])

    train_folder = datasets.ImageFolder(args.train_folder, transform=transform_train)
    val_folder = datasets.ImageFolder(args.val_folder, transform=transform_eval)
    test_folder = datasets.ImageFolder(args.test_folder, transform=transform_eval)

    # log_string(log, f'\nclass idx:\n{train_folder.class_to_idx}\n')
    saveJson(train_folder.class_to_idx, os.path.join(args.output_folder, 'class_idx.json'))
    
    if eval_stage:
        train_loader = torch.utils.data.DataLoader(train_folder, batch_size=args.val_batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory_train)
    else:
        train_loader = torch.utils.data.DataLoader(train_folder, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=args.pin_memory_train)

    val_loader = torch.utils.data.DataLoader(val_folder, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_folder, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader, 'test': test_loader}

    log_string(log, f'images numbers: train({len(train_folder)}) | val({len(val_folder)}) | test({len(test_folder)})')

    return dataloaders_dict


def save_model(args, model):
    if args.use_tracedmodule:
        traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, args.img_height, args.img_width))
        traced.save(args.model_file)
    else:
        torch.save(model, args.model_file)


def train_model(args, log, model, dataloaders_dict, criterion, optimizer, scheduler):
    num_epochs, patience = args.max_epoch, args.patience

    # best_acc = 0.0
    best_loss = float('inf')
    wait = 0
    reuslt_ls = []
    for epoch in range(num_epochs):
        if wait >= patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        
        epoch_result_dt = {}
        for phase in ['train', 'val']:
            start = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                images = item[0].to(args.device).float()
                classes = item[1].to(args.device).long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)
                    loss = criterion(output, classes)
                    _, preds = torch.max(output, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * len(output)
                    epoch_acc += torch.sum(preds == classes.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = (epoch_acc.double() / data_size).tolist()
            epoch_result_dt.update({phase:{'loss':epoch_loss, 'acc':epoch_acc, 'timeuse':time.time() - start}})

        log_string(log, '\n{} | epoch: {}/{}, train loss: {:.4f}, val_loss: {:.4f} | training time: {:.1f}s, inference time: {:.1f}s'.format(
                                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                    epoch + 1,
                                    args.max_epoch, 
                                    epoch_result_dt['train']['loss'],
                                    epoch_result_dt['val']['loss'],
                                    epoch_result_dt['train']['timeuse'], 
                                    epoch_result_dt['val']['timeuse']
                                )
                )

        reuslt_ls.append(epoch_result_dt)
        scheduler.step()

        if epoch_loss < best_loss:
            best_model_wts = model.state_dict()
            log_string(log, f'-> Val Accuracy improve from {best_acc:.4f} to {epoch_acc:.4f}, saving model')

            if args.auto_save_model:
                a = time.time()
                save_model(args, model)
                print(time.time() - a)

            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1

    # 儲存模型
    model.load_state_dict(best_model_wts)
    save_model(args, model)


    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')

    return reuslt_ls


def test_model(args, log, dataloaders_dict, criterion):
    model = torch.jit.load(args.model_file)
    model.eval()
    model = model.to(args.device)

    class_idx_dt = json.load(open(os.path.join(args.output_folder, 'class_idx.json'), 'r'))
    idx_class_dt = {v:k for k,v in class_idx_dt.items()}

    evaluation_dt = {}
    df_pred = pd.DataFrame()
    df_model_out = pd.DataFrame()
    with torch.no_grad():
        for phase in ['train', 'val', 'test']:
            start = time.time()

            epoch_loss = 0.0
            epoch_acc = 0

            Output_list = []
            Pred_list = []
            Label_list = []

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                images = item[0].to(args.device).float()
                classes = item[1].to(args.device).long()

                output = model(images)
                loss = criterion(output, classes)
                _, preds = torch.max(output, 1)

                epoch_loss += loss.item() * len(output)
                epoch_acc += torch.sum(preds == classes.data)

                Pred_list.extend(preds.tolist())
                Label_list.extend(classes.tolist())
                Output_list.extend(output.tolist())

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = (epoch_acc.double() / data_size).tolist()

            df_pred = df_pred.append(pd.DataFrame(
                {'Label_idx':Label_list, 'Predict_idx': Pred_list, 'Phase': [phase]*len(Pred_list)}
            ))
            df_model_out = df_model_out.append(pd.DataFrame(output.cpu(), columns=class_idx_dt.keys()))
            
            Output_list = list(map(to_prob, Output_list)) if args.prob else None

            phase_eval_dt = {'loss':epoch_loss, 'acc':epoch_acc, 'timeuse':time.time() - start} 
            phase_eval_dt.update(get_evaluation(Pred_list, Label_list, Output_list))
            evaluation_dt.update({phase:phase_eval_dt})

        df_pred['Label'] = df_pred['Label_idx'].apply(lambda x: idx_class_dt[x])
        df_pred['Predict'] = df_pred['Predict_idx'].apply(lambda x: idx_class_dt[x])

        df_pred.to_csv(os.path.join(args.output_folder, 'pred_and_label.csv'), index=False)
        df_model_out.to_csv(os.path.join(args.output_folder, 'model_output.csv'), index=False)
        saveJson(evaluation_dt, os.path.join(args.output_folder, 'evaluation.json'))

if __name__ == '__main__':
    # 參數
    args = get_args()
    args = check_args(args)

    # log
    log = open(os.path.join(args.output_folder, 'log.txt'), 'w')
    log_string(log, '='*20 + '\n[arguments]\n' + f'{str(args)[10: -1]}')
    log_system_info(args, log)

    # load data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'loading data...')
    dataloaders_dict = load_data(args, log)
    log_string(log, 'data loaded!\n' + '='*20)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # build model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'compiling model...')
    
    model = CoAtNet(
                image_size=(args.img_height, args.img_width), 
                in_channels=args.in_channels, 
                num_blocks=args.num_blocks, 
                channels=args.channels, 
                num_classes=33,
                prob=args.train_prob
                )

    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=args.decay_epoch,
                                        gamma=0.9)
    
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))
    log_string(log, 'model loaded!\n' + '='*20)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'training model...')
    result_ls = train_model(args, log, model, dataloaders_dict, criterion, optimizer, scheduler)
    saveJson(result_ls, os.path.join(args.output_folder, 'epoch_result.json'))

    plot_train_val_loss(
        train_total_loss=[i['train']['loss'] for i in result_ls], 
        val_total_loss=[i['val']['loss'] for i in result_ls],
        file_path=os.path.join(args.output_folder, 'train_val_loss.png')
        )

    log_string(log, 'training finish\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # test model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'calculating evaluation...')
    dataloaders_dict = load_data(args, log, eval_stage=True)
    test_model(args, log, dataloaders_dict, criterion)
    log_string(log, 'finished!!!\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
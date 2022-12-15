"""
[Mender]
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2022.12.10
Last Update: 2022.12.11

[Original]
Author: 林政委
GitHub: https://github.com/VincLee8188/GMAN-PyTorch

Describe: train pipeline
"""
import os
import psutil
import platform
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

#  import torchvision.transforms as transforms
from torchvision import datasets, transforms

from utils import *

from model.coatnet import CoAtNet

def get_args():
    parser = argparse.ArgumentParser()

    # 輸入
    parser.add_argument('--data_folder', type=str, default='data/sample200_L200')

    # 輸出
    parser.add_argument('--output_folder', type=str, default='./output/sample')
    parser.add_argument('--use_Tracedmodule', type=str2bool, default=True)

    # 前處理
    parser.add_argument('--img_height', type=int, default=224) # 32 倍數
    parser.add_argument('--img_width', type=int, default=224) # 32 倍數
    parser.add_argument('--img_nor_mean', type=float, nargs='+', default=(0.4914, 0.4822, 0.4465), help='mean in torchvision.transforms.Normalize')
    parser.add_argument('--img_nor_std', type=float, nargs='+', default=(0.2023, 0.1994, 0.2010), help='std in torchvision.transforms.Normalize')

    # coatnet 參數
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 2, 12, 28, 2], help='Set num_blocks') # python arg.py -l 1234 2345 3456 4567
    parser.add_argument('--channels', type=int, nargs='+', default=[64, 64, 128, 256, 512], help='Set channels') 
    parser.add_argument('--in_channels', type=int, default=3, help='Set in_channels') 

    # 超參數
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=100,
                        help='val batch size')
    parser.add_argument('--max_epoch', type=int, default=50,
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

    args = parser.parse_args()

    return args

def check_args(args):
    # 新增參數
    args.device = 'cuda' if torch.cuda.is_available() and args.device in ['gpu', 'cuda'] else 'cpu'
    args.train_folder = os.path.join(args.data_folder, 'train')
    args.test_folder = os.path.join(args.data_folder, 'test')
    args.val_folder = os.path.join(args.data_folder, 'val')

    args.model_file = os.path.join(args.data_folder, 
        'model.pt' if args.use_Tracedmodule else 'model.pkl'
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
    log_string(log, '='*20 + '\n' + message + '='*20)


def load_data(args, log):
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

    log_string(log, f'\nclass idx:\n{train_folder.class_to_idx}\n')

    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_folder, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_folder, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader, 'test': test_loader}

    log_string(log, f'images numbers: train({len(train_folder)}) | val({len(val_folder)}) |test({len(test_folder)})')

    return dataloaders_dict

def train_model(log, model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, patience):
    best_acc = 0.0
    wait = 0
    for epoch in range(num_epochs):
        if wait >= patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break

        for phase in ['train', 'val']:
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
            epoch_acc = epoch_acc.double() / data_size
            log_string(log, f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        scheduler.step()

        if epoch_acc > best_acc:
            best_model_wts = model.state_dict()
            log_string(log, f'val loss decrease from {best_acc:.4f} to {epoch_acc:.4f}, saving model to {args.model_file}')

            best_acc = epoch_acc
            wait = 0
        else:
            wait += 1

    model.load_state_dict(best_model_wts)

    # 儲存模型
    if args.use_Tracedmodule:
        traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 224, 224))
        traced.save(args.model_file)
    else:
        torch.save(model, args.model_file)

    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')


if __name__ == '__main__':
    # 參數
    args = get_args()
    args = check_args(args)

    # log
    log = open(os.path.join(args.output_folder, 'log.txt'), 'w')
    log_string(log, f'{str(args)[10: -1]}\n')
    log_string(log, f'main output folder: {args.output_folder}')
    
    log_system_info(args, log)

    # load data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'loading data...')
    dataloaders_dict = load_data(args, log)
    log_string(log, 'data loaded!\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # build model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'compiling model...')
    
    model = CoAtNet((args.img_height, args.img_width), args.in_channels, args.num_blocks, args.channels, num_classes=33)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=args.decay_epoch,
                                        gamma=0.9)
    
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))
    log_string(log, 'model loaded!\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'training model...')
    train_model(log, model, dataloaders_dict, criterion, optimizer, scheduler, args.max_epoch, args.patience)
    log_string(log, 'training finish\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # test model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plot_train_val_loss(loss_train, loss_val, 
    #             os.path.join(fig_folder, 'train_val_loss.png'))
    # trainPred, valPred, testPred, eval_dt = test(args, log)
    # saveJson(eval_dt, os.path.join(output_folder, 'evaluation.json'))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
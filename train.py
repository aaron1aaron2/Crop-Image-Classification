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
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

#  import torchvision.transforms as transforms
from torchvision import datasets, transforms

from utils import *

from model import CoAtNet

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='data/sample100_200x200')
    parser.add_argument('--img_height', type=int, default=200)
    parser.add_argument('--img_width', type=int, default=200)

    parser.add_argument('-l','--list', nargs='+', help='Set flag') # python arg.py -l 1234 2345 3456 4567


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

    parser.add_argument('--output_folder', type=str, default='./output')
    parser.add_argument('--device', default='gpu', 
                        help='cpu or cuda')

    args = parser.parse_args()

    return args

def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, patience):
    best_acc = 0.0
    wait = 0
    for epoch in range(num_epochs):
        if wait >= patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        model.cuda()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                images = item[0].cuda().float()
                classes = item[1].cuda().long()
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
            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        scheduler.step()
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 224, 224))
            traced.save(args.model_file)
            # best_model_wts = model.state_dict()
            log_string(log, f'val loss decrease from {best_acc:.4f} to {epoch_acc:.4f}, saving model to {args.model_file}')

            best_acc = epoch_acc
            wait = 0
        else:
            wait += 1
    # model.load_state_dict(best_model_wts)
    # torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')

if __name__ == '__main__':
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() and args.device in ['gpu', 'cuda'] else 'cpu'

    fig_folder = os.path.join(args.output_folder, 'figure')

    build_folder(args.output_folder)
    build_folder(fig_folder)

    log = open(os.path.join(args.output_folder, 'log.txt'), 'w')
    log_string(log, str(args)[10: -1])
    log_string(log, f'main output folder{args.output_folder}')

    saveJson(args.__dict__, os.path.join(args.output_folder, 'configures.json'))

    # load data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'loading data...')

    transform_train = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.ImageFolder('C:/pre', transform=transform_train)
    log_string(log, f'{train_data.class_to_idx}')
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    trainset, testset = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    # log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
    # log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    # log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
    log_string(log, 'data loaded!')

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # build model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'compiling model...')

    num_blocks = [2, 2, 12, 28, 2]
    channels = [64, 64, 128, 256, 512]

    # image_size, in_channels, num_blocks, channels, num_classes, block_types{'C': MBConv, 'T': Transformer}
    model = CoAtNet((args.img_height, args.img_width), 3, num_blocks, channels, num_classes=33).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=args.decay_epoch,
                                        gamma=0.9)
    
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # train model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(log, 'training model...')
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, args.max_epoch, args.patience)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # test model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plot_train_val_loss(loss_train, loss_val, 
    #             os.path.join(fig_folder, 'train_val_loss.png'))
    # trainPred, valPred, testPred, eval_dt = test(args, log)
    # saveJson(eval_dt, os.path.join(output_folder, 'evaluation.json'))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

from tqdm import tqdm

#  import torchvision.transforms as transforms
from torchvision import datasets, transforms

from utils import saveJson, build_folder, log_string

def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--L', type=int, default=1,
    #                     help='number of STAtt Blocks')
    # parser.add_argument('--K', type=int, default=8,
    #                     help='number of attention heads')
    # parser.add_argument('--d', type=int, default=8,
    #                     help='dims of each head attention outputs')

    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='training set [default : 0.7]')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='validation set [default : 0.1]')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='testing set [default : 0.2]')

    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='epoch to run')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='decay epoch')

    parser.add_argument('--image_folder', default='./model/data/pems-bay.h5',
                        help='traffic file')
    parser.add_argument('--model_file', default='./output/GMAN.pkl',
                        help='save the model to disk')
    parser.add_argument('--log_file', default='./output/log.txt',
                        help='log file')

    parser.add_argument('--output_folder', type=str, default='./output')
    parser.add_argument('--device', default='gpu', 
                        help='cpu or cuda')

    args = parser.parse_args()

    return args

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
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
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 3, 224, 224))
            traced.save('model4.pth')
            best_acc = epoch_acc


if __name__ == '__main__':
    args = get_args()
    output_folder = os.path.dirname(args.log_file)
    args.device = 'cuda' if torch.cuda.is_available() and args.device in ['gpu', 'cuda'] else 'cpu'
    fig_folder = os.path.join(args.output_folder, 'figure')
    build_folder(args.output_folder)
    build_folder(fig_folder)

    log = open(args.log_file, 'w')
    log_string(log, str(args)[10: -1])
    log_string(log, f'main output folder{output_folder}')

    device = torch.device("cuda")

    num_blocks = [2, 2, 12, 28, 2]
    channels = [64, 64, 128, 256, 512]

    model = CoAtNet((224, 224), 3, num_blocks, channels, num_classes=33).to(device)
    BATCH_SIZE = 32



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
    print(train_data.class_to_idx)
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    trainset, testset = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_model(model, dataloaders_dict, criterion, optimizer, 25)

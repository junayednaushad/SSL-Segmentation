import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
from tqdm import tqdm
import argparse
from models import UNet
from dataset import CarvanaDataset
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Hyper-parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
HEIGHT = 256
WIDTH = 256
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Dice(preds, labels):
    '''Computes Dice Coefficient'''

    smooth = 1e-8
    dice_score = (2*torch.sum(preds*labels) + smooth) / ( torch.sum(preds**2) + torch.sum(labels**2) + smooth )
    return dice_score

def train(train_loader, model, optimizer, criterion):
    '''Trains one epoch of the model and computes avg loss for that epoch'''

    model.train()
    loop = tqdm(train_loader)
    avg_loss = 0

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        preds = model(images)
        optimizer.zero_grad()
        loss = criterion(preds, labels)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    
    avg_loss /= len(train_loader)
    return avg_loss

def eval(val_loader, model, criterion):
    '''Evaluates one epoch of the model and computes avg loss, avg dice, std dev dice, and median dice score for that epoch'''

    model.eval()
    loop = tqdm(val_loader)
    avg_loss = 0
    dice_scores = []
    std_dice = []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            preds = model(images)
            loss = criterion(preds, labels)
            preds = sigmoid(preds)
            preds = (preds > 0.5).float()
            dice_score = Dice(preds, labels).item()
            dice_scores.append(dice_score)
            avg_loss += loss.item()

            loop.set_postfix(loss=loss.item(), dice=dice_score)

    avg_loss /= len(val_loader)
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    median_dice = np.median(dice_scores)
    print('Avg Dice Score: {:.4f}'.format(avg_dice))
    print('Dice std dev: {:.4f}'.format(std_dice))
    print('Median Dice: {:.4f}'.format(median_dice))
    return avg_loss, avg_dice, std_dice, median_dice


def main(args):
    '''Trains and evaluates the model and plots loss and dice curves'''

    train_image_dir = os.path.join(args.root_dir, 'train')
    train_mask_dir = os.path.join(args.root_dir, 'train_masks')
    train_images = os.listdir(train_image_dir)
    num_cars = len(set([filename.split('_')[0] for filename in train_images]))

    if args.train_amount < 1.0:
        train_ids = train_images[:int(num_cars * args.train_amount)*16]
        if args.use_rotnet:
            if args.freeze_weights:
                save_loc = './saved_models/best_reduced_unet_rotnet_frozen.pt'
            else:
                save_loc = './saved_models/best_reduced_unet_rotnet.pt'
        elif args.use_resnet:
            save_loc = './saved_models/best_reduced_unet_resnet.pt'
        else:
            save_loc = './saved_models/best_reduced_unet.pt'
    else:
        train_ids = train_images[:int(num_cars * (1-args.val_split))*16]
        if args.use_rotnet:
            if args.freeze_weights:
                save_loc = './saved_models/best_unet_rotnet_frozen.pt'
            else:
                save_loc = './saved_models/best_unet_rotnet.pt'
        elif args.use_resnet:
            save_loc = './saved_models/best_unet_resnet.pt'
        else:
            save_loc = './saved_models/best_unet.pt'

    val_ids = train_images[int(num_cars * (1-args.val_split))*16:]

    print('Total number of cars:', num_cars)
    print('Total number of images:', len(train_images))
    print('Number of training images:', len(train_ids))
    print('Number of validation images:', len(val_ids))

    train_transform = A.Compose([
            A.Resize(height=HEIGHT, width=WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            ToTensorV2(),
        ])
    val_transform = A.Compose([
            A.Resize(height=HEIGHT, width=WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            ToTensorV2(),
        ])
    train_dataset = CarvanaDataset(train_image_dir, train_mask_dir, train_ids, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dataset = CarvanaDataset(train_image_dir, train_mask_dir, val_ids, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    if args.use_resnet:
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(DEVICE)
    else:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    starting_epoch = 0

    if args.use_rotnet:
        rotnet_dict = torch.load('./saved_models/best_rotnet.pt')
        unet_dict = model.state_dict()
        rotnet_dict = {k: v for k,v in rotnet_dict.items() if k in unet_dict}
        unet_dict.update(rotnet_dict)
        model.load_state_dict(unet_dict)
        if args.freeze_weights:
            for name, param in model.named_parameters():
                if name in rotnet_dict:
                    param.requires_grad = False

    if args.load_model:
        print('Loading model from: ' + save_loc)
        checkpoint = torch.load(save_loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
    
    train_losses = []
    val_losses = []
    val_dice = []
    std_dice_scores = []
    median_dice_scores = []
    best_val_loss = 1e8

    epochs = list(range(starting_epoch, NUM_EPOCHS))
    for epoch in epochs:
        avg_train_loss = train(train_loader, model, optimizer, criterion)
        train_losses.append(avg_train_loss)

        avg_val_loss, avg_dice, std_dice, median_dice = eval(val_loader, model, criterion)
        val_losses.append(avg_val_loss)
        val_dice.append(avg_dice)
        std_dice_scores.append(std_dice)
        median_dice_scores.append(median_dice)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('Found better model at epoch {}'.format(epoch))
            print('Saving model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_loc)

    
    plt.plot(epochs, train_losses, 'go-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, val_dice, 'bo-')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.show()

    plt.plot(epochs, std_dice_scores, 'bo-')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Standard Deviation')
    plt.show()

    plt.plot(epochs, median_dice_scores, 'bo-')
    plt.xlabel('Epoch')
    plt.ylabel('Median Dice Scores')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.', help='Directory that contains images')
    parser.add_argument('--val_split', type=float, default=0.25, help='Proportion of training data to use for validation')
    parser.add_argument('--load_model', action='store_true', default=False, help='Load model to resume training')
    parser.add_argument('--train_amount', type=float, default=1.0, help='Proportion of training data to use')
    parser.add_argument('--use_rotnet', action='store_true', default=False, help='Use RotNet weights for the UNet encoder')
    parser.add_argument('--freeze_weights', action='store_true', default=False, help='Freeze the RotNet weights during training')
    parser.add_argument('--use_resnet', action='store_true', default=False, help='Use Unet with ResNet34 encoder')
    args = parser.parse_args()
    main(args)
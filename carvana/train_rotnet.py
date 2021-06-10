import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
from models import RotNet
from dataset import RotDataset
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Hyper-parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
HEIGHT = 256
WIDTH = 256
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, optimizer, criterion):
    '''Trains one epoch of the model and computes avg loss for that epoch'''

    model.train()
    loop = tqdm(train_loader)
    avg_loss = 0

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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
    '''Evaluates one epoch of the model and computes avg loss and avg accuracy for that epoch'''

    model.eval()
    loop = tqdm(val_loader)
    avg_loss = 0
    num_images = 0
    num_correct = 0

    with torch.no_grad():
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, labels)
            avg_loss += loss.item()

            num_images += len(images)
            _, pred_class = torch.max(preds, 1)
            num_correct += pred_class.eq(labels).sum()

            loop.set_postfix(loss=loss.item(), acc=(num_correct/num_images).item())
    
    avg_loss /= len(val_loader)
    avg_acc = (num_correct / num_images).item()
    print('Avg Accuracy: {:.4f}'.format(avg_acc))
    return avg_loss, avg_acc

def main(args):
    '''Trains and evaluates the model and plots loss and accuracy curves'''

    train_image_dir = os.path.join(args.root_dir, 'train')
    train_images = os.listdir(train_image_dir)
    num_cars = len(set([filename.split('_')[0] for filename in train_images]))
    train_ids = train_images[:int(num_cars * (1-args.val_split))*16]
    save_loc = './saved_models/best_rotnet.pt'
    val_ids = train_images[int(num_cars * (1-args.val_split))*16:]

    print('Total number of cars:', num_cars)
    print('Total number of images:', len(train_images))
    print('Number of training images:', len(train_ids))
    print('Number of validation images:', len(val_ids))

    transform = A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            ToTensorV2(),
        ])
    
    train_dataset = RotDataset(train_image_dir, train_ids, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dataset = RotDataset(train_image_dir, val_ids, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = RotNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_acc = []
    best_val_loss = 1e8

    epochs = list(range(NUM_EPOCHS))
    for epoch in epochs:
        avg_train_loss = train(train_loader, model, optimizer, criterion)
        train_losses.append(avg_train_loss)

        avg_val_loss, avg_acc = eval(val_loader, model, criterion)
        val_losses.append(avg_val_loss)
        val_acc.append(avg_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('Found better model at epoch {}'.format(epoch))
            print('Saving model')
            torch.save(model.state_dict(), save_loc)

    plt.plot(epochs, train_losses, 'go-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, val_acc, 'bo-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--val_split', type=float, default=0.25)
    args = parser.parse_args()
    main(args)
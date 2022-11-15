import argparse
import glob
import os
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.attentions import AttentionSelector
from models.vit import ViT


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


def get_dataloaders(batch_size):
    os.makedirs('data', exist_ok=True)

    train_dir = 'data/train'
    test_dir = 'data/test'

    with zipfile.ZipFile('train.zip') as train_zip:
        train_zip.extractall('data')

    with zipfile.ZipFile('test.zip') as test_zip:
        test_zip.extractall('data')

    train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    labels = [path.split('/')[-1].split('.')[0] for path in train_list]
    train_list, valid_list = train_test_split(train_list,
                                              test_size=0.2,
                                              stratify=labels, )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
    test_data = CatsDogsDataset(test_list, transform=test_transforms)
    train_loader_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    test_loader_ = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_loader_, valid_loader_, test_loader_


def train(model, train_loader_, valid_loader_, device, lr, epochs):
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader_):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader_)
            epoch_loss += loss / len(train_loader_)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader_:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader_)
                epoch_val_loss += val_loss / len(valid_loader_)

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss :"
            f" {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda', help='What device to use')
    parser.add_argument('-b', '--batch_size', default=64)
    parser.add_argument('-e', '--epochs', default=20)
    parser.add_argument('-lr', '--lr', default=3e-5)
    parser.add_argument('-g', '--gamma', default=0.7)
    parser.add_argument('-a', '--attention', default='BASELINE', help='Attention type. Options are BASELINE for normal'
                                                                      'transformer, MAX for the pointwise maximum,'
                                                                      'and MESSAGE for message passing')
    args = parser.parse_args()

    vit = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=2,
        channels=3,
        depth=12,
        heads=8,
        mlp_dim=64,
        attention=AttentionSelector[args.attention].value).to(args.device)
    train_loader, valid_loader, test_loader = get_dataloaders(args.batch_size)
    train(vit, train_loader, valid_loader, args.device, args.lr, args.epochs)

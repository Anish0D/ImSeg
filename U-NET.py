import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Dataset --------------------
class CarSegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            image, mask = self.augment(image, mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

# -------------------- Augmentations --------------------
TARGET_SIZE = [1024, 768]  # [height, width]

def augmentation_1(image, mask):
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    angle = random.uniform(-15, 15)
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    image = TF.resize(image, TARGET_SIZE)
    mask = TF.resize(mask, TARGET_SIZE)
    return image, mask

def augmentation_2(image, mask):
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(896, 672))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)
    image = TF.resize(image, TARGET_SIZE)
    mask = TF.resize(mask, TARGET_SIZE)
    return image, mask

def augmentation_3(image, mask):
    image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
    image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    image = TF.resize(image, TARGET_SIZE)
    mask = TF.resize(mask, TARGET_SIZE)
    return image, mask

def no_aug(image, mask):
    image = TF.resize(image, TARGET_SIZE)
    mask = TF.resize(mask, TARGET_SIZE)
    return image, mask

# -------------------- Path Loader --------------------
def load_paths(root_dir):
    img_dir = os.path.join(root_dir, 'img')
    mask_dir = os.path.join(root_dir, 'masks')
    img_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))
    img_paths = [os.path.join(img_dir, name) for name in img_names]
    mask_paths = [os.path.join(mask_dir, name) for name in mask_names]
    return list(zip(img_paths, mask_paths))

# -------------------- Data Split --------------------
all_pairs = all_pairs = load_paths('/Users/sangeetadegalmadikar/Desktop/SDSU_Internship/ImSeg/cars')


random.seed(42)
random.shuffle(all_pairs)

total = len(all_pairs)
train_split = int(0.7 * total)
val_split = int(0.85 * total)

train_pairs = all_pairs[:train_split]
val_pairs = all_pairs[train_split:val_split]
test_pairs = all_pairs[val_split:]

train_imgs, train_masks = zip(*train_pairs)
val_imgs, val_masks = zip(*val_pairs)
test_imgs, test_masks = zip(*test_pairs)

train_dataset = ConcatDataset([
    CarSegmentationDataset(train_imgs, train_masks, augment=augmentation_1),
    CarSegmentationDataset(train_imgs, train_masks, augment=augmentation_2),
    CarSegmentationDataset(train_imgs, train_masks, augment=augmentation_3)
])

validation_dataset = CarSegmentationDataset(val_imgs, val_masks, augment=no_aug)
test_dataset = CarSegmentationDataset(test_imgs, test_masks, augment=no_aug)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# U-Net Model
def doubleConv(input_channels, output_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = doubleConv(1, 64)
        self.down_conv_2 = doubleConv(64, 128)
        self.down_conv_3 = doubleConv(128, 256)
        self.down_conv_4 = doubleConv(256, 512) 
        self.down_conv_5 = doubleConv(512, 1024)

    def forward(self, image):
        x1 = self.down_conv_1(image)
        print(f"x1: {x1.shape}")
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print(f"x9: {x9.shape}")
        return x9


if __name__ == "__main__":
    image = torch.rand(1, 1, 1024, 768).to(device)  # Updated input size
    model = Unet().to(device)
    print(model(image))

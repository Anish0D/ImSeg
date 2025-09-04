import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image

# -------------------- Config --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = [1024, 768]  # [height, width]
MODEL_SAVE_PATH = "unet_segmentation_model.pth"

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

# -------------------- Path Loading --------------------
def load_paths(root_dir):
    img_dir = os.path.join(root_dir, 'img')
    mask_dir = os.path.join(root_dir, 'masks')
    img_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))
    img_paths = [os.path.join(img_dir, name) for name in img_names]
    mask_paths = [os.path.join(mask_dir, name) for name in mask_names]
    return list(zip(img_paths, mask_paths))

all_pairs = load_paths('/Users/sangeetadegalmadikar/Desktop/SDSU_Internship/ImSeg/cars')
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

#  U-Net Model 
def doubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down_conv_1 = doubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_2 = doubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_3 = doubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_4 = doubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = doubleConv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = doubleConv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = doubleConv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = doubleConv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = doubleConv(128, 64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def crop_and_concat(self, enc_feat, x):
        if enc_feat.size()[2:] != x.size()[2:]:
            diffY = enc_feat.size()[2] - x.size()[2]
            diffX = enc_feat.size()[3] - x.size()[3]
            enc_feat = enc_feat[:, :, diffY//2:-diffY//2, diffX//2:-diffX//2]
        return torch.cat([enc_feat, x], dim=1)

    def forward(self, x):
        x1 = self.down_conv_1(x)
        x2 = self.pool1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.pool2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.pool3(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.pool4(x7)

        x9 = self.bottleneck(x8)

        x = self.up_trans_1(x9)
        x = self.crop_and_concat(x7, x)
        x = self.up_conv_1(x)

        x = self.up_trans_2(x)
        x = self.crop_and_concat(x5, x)
        x = self.up_conv_2(x)

        x = self.up_trans_3(x)
        x = self.crop_and_concat(x3, x)
        x = self.up_conv_3(x)

        x = self.up_trans_4(x)
        x = self.crop_and_concat(x1, x)
        x = self.up_conv_4(x)

        return self.output_layer(x)


def dice_score(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

def iou_score(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)

def visualize_prediction(image, pred_mask, true_mask):
    image = image.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    true_mask = true_mask.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title('Predicted Mask')
    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title('Ground Truth')

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


model = Unet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Evaluate on validation set
    model.eval()
    dice_total, iou_total, count = 0.0, 0.0, 0
    val_images, val_masks = None, None
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_preds = model(val_images)
            val_probs = torch.sigmoid(val_preds)
            val_bin = (val_probs > 0.5).float()
            for i in range(val_images.size(0)):
                dice_total += dice_score(val_bin[i], val_masks[i])
                iou_total += iou_score(val_bin[i], val_masks[i])
                count += 1

    avg_dice = dice_total / count
    avg_iou = iou_total / count

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")

    # Visualize prediction from one random validation image
    rand_idx = random.randint(0, val_images.size(0) - 1)
    visualize_prediction(val_images[rand_idx], val_bin[rand_idx], val_masks[rand_idx])


# Final save
torch.save(model.state_dict(), MODEL_SAVE_PATH)
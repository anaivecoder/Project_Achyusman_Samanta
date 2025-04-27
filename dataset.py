import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from config import resize_x, resize_y, batchsize

# Transformation (augmentation for training)
transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

DATA_DIR = "my_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

class My_DataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(root_dir))  # sorted to ensure consistent indexing
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def My_DataLoader(train=True):
    dataset = My_DataSet(TRAIN_DIR if train else TEST_DIR, transform=transform)

    if train:
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(50))
        train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batchsize, shuffle=False)
        return train_loader, val_loader
    else:
        return DataLoader(dataset, batch_size=batchsize, shuffle=False)



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from model import MyCustomModel
from dataset import My_DataLoader, My_DataSet
import config
from torchvision import datasets
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

def Model_Training(*args, **kwargs):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # full_dataset = My_DataLoader(train=True)
    # val_size = int(0.2 * len(full_dataset))
    # train_size = len(full_dataset) - val_size
    # train_subset, val_subset = random_split(
    #     full_dataset, [train_size, val_size],
    #     generator=torch.Generator().manual_seed(10)
    # )

    # train_loader = DataLoader(train_subset, batch_size=config.batchsize, shuffle=True)
    # val_loader = DataLoader(val_subset, batch_size=config.batchsize, shuffle=False)

    train_loader, val_loader = My_DataLoader(train=True)

    dataset = My_DataSet("my_dataset/train")
    class_names = dataset.classes
    print("Classes: ", class_names)

    model = MyCustomModel(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0

    for epoch in range(config.epochs):
        model.train()
        tr_correct, tr_total = 0, 0
        scaler = GradScaler()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)

        train_acc = tr_correct / tr_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.checkpoint_path)
            print("__Saved new best model__")
 
Model_Training()
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# PyTorch Image Models (timm) is the standard for pre-trained vision models
import timm 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

KAGGLE_DATASET_PATH = "/kaggle/input/competitions/dlp-26t2-week9-assignment/" # UPDATE THIS
TRAIN_DIR = os.path.join(KAGGLE_DATASET_PATH, "train")
TEST_DIR = os.path.join(KAGGLE_DATASET_PATH, "test")

IMAGE_SIZE = 300  # EfficientNet-B3 natively uses 300x300
BATCH_SIZE = 32   # Slightly smaller batch size for larger image resolution
EPOCHS = 15       # Pre-trained models converge extremely fast

# ==========================================
# 2. DATASET CLASSES & AUGMENTATION
# ==========================================
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.transform = transform

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        img_id = os.path.splitext(os.path.basename(path))[0]
        if self.transform: image = self.transform(image)
        return image, img_id

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self): return len(self.subset)

# Standard augmentation; the pre-trained weights are already highly robust
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = ImageFolder(root=TRAIN_DIR)
train_indices, val_indices = train_test_split(
    np.arange(len(full_dataset)), test_size=0.10, stratify=full_dataset.targets, random_state=42
)

train_dataset = TransformSubset(Subset(full_dataset, train_indices), train_transforms)
val_dataset = TransformSubset(Subset(full_dataset, val_indices), val_test_transforms)
test_dataset = TestDataset(img_dir=TEST_DIR, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. PRE-TRAINED MODEL SETUP (TRANSFER LEARNING)
# ==========================================
print("Downloading Pre-Trained EfficientNet-B3...")
# Create model with ImageNet weights
model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)

# Swap the final classification layer from 1000 classes to our 10 classes
num_in_features = model.classifier.in_features
model.classifier = nn.Linear(num_in_features, 10)

model = model.to(device)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Lower learning rate because the base model already has excellent weights
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

best_val_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    train_loss = running_loss / len(train_dataset)

    # Validation Phase
    model.eval()
    val_preds, val_targets, val_loss = [], [], 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())
            
    val_loss = val_loss / len(val_dataset)
    val_micro_f1 = f1_score(val_targets, val_preds, average='micro')
    scheduler.step(val_micro_f1)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Micro F1: {val_micro_f1:.4f}")
    
    if val_micro_f1 > best_val_f1:
        best_val_f1 = val_micro_f1
        torch.save(model.state_dict(), "best_pretrained_model.pth")
        print(f"--> Best model saved! (F1: {best_val_f1:.4f})")

# ==========================================
# 5. INFERENCE
# ==========================================
print("\nRunning Test Set Inference...")
model.load_state_dict(torch.load("best_pretrained_model.pth"))
model.eval()

image_ids, final_predictions = [], []

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        image_ids.extend(ids)
        final_predictions.extend(preds)

submission_df = pd.DataFrame({"Image_ID": image_ids, "Label": final_predictions})
submission_df.to_csv("submission.csv", index=False)
print("\nSubmission Complete! File saved as 'submission.csv'")

# In[ ]:




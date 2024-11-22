import kagglehub
import numpy as np
import os
import pandas as pd
import shutil
import sklearn
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, random_split, Subset


class HAM10000Dataset(Subset):
    def __init__(self, dataset, indices, metadata=None):
        super().__init__(dataset, indices)
        self.metadata = metadata

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        row = self.metadata.iloc[self.indices[idx]]
        label = row['class_idx']
        return image, label


path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

metadata = pd.read_csv(f"{path}/HAM10000_metadata.csv")
benign_classes = ["nv", "bkl", "df", "vasc"]
malignant_classes = ["mel", "bcc", "akiec"]

class_mapping = {cls: 0 if cls in benign_classes else 1 for cls in benign_classes + malignant_classes}
metadata['class_idx'] = metadata['dx'].map(class_mapping)

data_path = path + '/data'
path_part1 = path + '/ham10000_images_part_1'
path_part2 = path + '/ham10000_images_part_2'

source_dirs = [path_part1, path_part2]
if not os.path.exists(data_path):
    os.makedirs(data_path)
    os.makedirs(f"{data_path}/0")
    os.makedirs(f"{data_path}/1")
    for _, row in metadata.iterrows():
        image_id = row['image_id'] + ".jpg"
        class_dir = f"{data_path}/{row['class_idx']}"  # 0 or 1

        # Locate the image in the source directories
        for src_dir in source_dirs:
            src_path = os.path.join(src_dir, image_id)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(class_dir, image_id))
                break

path = data_path

train_transform = v2.Compose([
    v2.Resize((224, 224)),  # Resize to 224x224
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # ToTensor()
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

val_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # ToTensor()
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

test_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # ToTensor()
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

train_dataset = datasets.ImageFolder(root=path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=path, transform=val_transform)
test_dataset = datasets.ImageFolder(root=path, transform=test_transform)

indices = [i for i in range(len(train_dataset))]

train_indices, indices_ = sklearn.model_selection.train_test_split(indices, test_size=0.30, random_state=42)
val_indices, test_indices = sklearn.model_selection.train_test_split(indices_, test_size=0.50, random_state=42)

# train_set = HAM10000Dataset(train_dataset, train_indices, metadata)
# val_set = HAM10000Dataset(val_dataset, val_indices, metadata)
# test_set = HAM10000Dataset(test_dataset, test_indices, metadata)
train_set = Subset(train_dataset, train_indices)
val_set = Subset(val_dataset, val_indices)
test_set = Subset(test_dataset, test_indices)

# train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])
# train_set.dataset.transform = train_transform
# val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

print("Path to dataset files:", path)
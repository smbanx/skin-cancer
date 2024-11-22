import kagglehub
import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, Subset

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

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
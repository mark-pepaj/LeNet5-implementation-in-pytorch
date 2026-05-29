import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from convolutional_neural_network.model import CNN


# augmentation
random_rotate = transforms.RandomRotation(10) # 10 degrees
random_affine = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5) # prob that it flips
vertical_flip = transforms.RandomVerticalFlip(p=0.5) # prob that it flips
augment_shape = transforms.RandomResizedCrop((28, 28), scale=(0.5, 1), ratio=(0.5, 2))
augment_color = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)

train_augments = transforms.Compose([
    random_affine,
    transforms.Pad(2),
    transforms.ToTensor()
    ])
val_augments = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])


train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=train_augments
)

# Load test data
val_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=val_augments
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 100
batch_size = 128

criterion_config = {"name":"cross_entropy"}
optimizer_config = {"name": "AdamW", "lr":1e-3}


channels = [1, 6, 16, 120]

layer_configs = [
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "avg_pooling", "kernel_size": 2, "stride": 2},
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "avg_pooling", "kernel_size": 2, "stride": 2},
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "flatten"},
        {"type": "linear", "in_features": 120, "out_features": 84},
        {"type": "nonlinearity", "name": "relu"},
        {"type": "linear", "in_features": 84, "out_features": 10},
        {"type": "nonlinearity", "name": "softmax"},
]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = CNN(channels=channels, layer_configs=layer_configs).to(device)

history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        layer_configs=layer_configs,
        criterion_config=criterion_config,
        optimizer_config=optimizer_config,
        )


#plt.plot(history["train_loss"], label="train_loss")
#plt.plot(history["val_loss"], label="val_loss")
#plt.legend()
#plt.show()

#plt.plot(history["val_acc"], label="val_acc")
#plt.legend()
#plt.show()

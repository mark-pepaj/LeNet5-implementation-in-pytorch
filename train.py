import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from convolutional_neural_network.model import *
from data_augmentation_for_deep_learning.augment import *



augmentor = Augmentor(
        pad=2,
        augment_configs=[RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),]
        )


train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=augmentor.get_train_transforms()
)

# Load test data
val_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=augmentor.get_val_transforms()
)



num_epochs = 100
batch_size = 256
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")



layer_configs = [
        ConvolutionalLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        AveragePooling(kernel_size=2, stride=2),
        ConvolutionalLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        AveragePooling(kernel_size=2, stride=2),
        ConvolutionalLayer(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        Flatten(),
        LinearLayer(in_features=120, out_features=100),
        Nonlinearity(name="tanh"),
        LinearLayer(in_features=100, out_features=10),
]



model = CNN(layer_configs=layer_configs).to(device)
criterion_config = Criterion(name="cross_entropy")
optimizer_config = Optimizer(name="AdamW", lr=1e-3, weight_decay=0.01)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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

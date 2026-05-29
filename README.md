# LeNet-5

<p align="center">
  ***Images coming soon...***
</p>


An implementation of the LeNet-5 architecture for image classification, in Python using PyTorch and Torchvision.

## Install

```sh
pip install -r requirements.txt
```

## Quick Start

Since this is a reproduction of LeNet-5 the model is already configured and can easily be trained and tested.


```sh
python train.py
```

Out of the box, `train.py` trains the model on the MNIST dataset.
During training, the weights which produce an output with the lowest validation loss are saved to `trained_weights.pth`

The trained weights can be loaded and tested on by running:

```sh
python test.py
```

The output of this script shows the accuracy of the model when tested on data that hasn't been used for training.
It also produces a confusion matrix showing classes which were incorrectly classified as other classes.

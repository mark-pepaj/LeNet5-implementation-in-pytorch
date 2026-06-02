# LeNet-5


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



## LeNet-5 Configuration
<div align="center">
  <img width="895" height="192" alt="LeNet5_config" src="https://github.com/user-attachments/assets/32b3c648-1ab3-40f5-a9d5-eac5225b3fd5" />
</div>


## Training Results

<table>
  <tr>
    <td><img width="620" height="461" alt="train_val_loss" src="https://github.com/user-attachments/assets/c40fbb0b-aca7-451a-8705-4662b58c98a5"/></td>
    <td><img width="620" height="461" alt="val_acc" src="https://github.com/user-attachments/assets/b7dd25c3-1981-45c5-a925-b781aed0309d"/></td>
  </tr>
</table>

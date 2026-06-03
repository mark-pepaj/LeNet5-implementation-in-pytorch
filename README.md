# LeNet-5

An implementation of the LeNet-5 architecture for image classification, in Python using PyTorch and Torchvision.

## Install

```sh
pip install -r requirements.txt
```
## Quick Start
#### Train the Model
Since this is a reproduction of LeNet-5 the model is already configured and can easily be trained and tested.

```sh
python train.py
```
Out of the box, `train.py` trains the model on the MNIST dataset.
During training, the weights which produce an output with the lowest validation loss are saved to `trained_weights.pth`

#### Inference
The trained weights can be loaded and tested on by running:

```sh
python test.py
```
The output of this script shows the accuracy of the model when tested on data that hasn't been used for training.
It also produces a confusion matrix showing classes which were incorrectly classified as other classes.

## LeNet-5 Configuration
Using the CNN configurator that I built in a previous project, I was able to easily replicate the configuration of the LeNet-5 model declaring each layer in a list. Since LeNet-5 only contains a few layers, I manually added each layer to the layer_configs list. 
<div align="center">
  <img width="895" height="192" alt="LeNet5_config" src="https://github.com/user-attachments/assets/32b3c648-1ab3-40f5-a9d5-eac5225b3fd5" />
</div>

## Data Augmentation
Although the original LeNet-5 paper doesn't mention any data augmentation, I wanted to see if it would help improve the accuracy of the model.
It should be noted, however, that here augmentation is limited, since the MNIST dataset is grayscale and numbers should not be flipped or inverted
<div align="center">
  <img width="906" height="36" alt="augmentation" src="https://github.com/user-attachments/assets/bc30b8db-66f1-41e7-ab43-d22fe9c47ca1" />
</div>

I used the Augmentor class that I also built in a previous project to configure the augmentation.<br>
Here I used Random Affine, which randomly applies a combination of the transformations passed to it, and kept it simple with a rotation of up to 10 degrees, a translation (shift) of up to 10% of the image's height and width, a zoom between 90%-110%, and a shear of up to 5 degrees.

## Training Results
The lowest validation loss the model acheived was `5.7%` after `87` epochs.
<div align = "center">
  <img width="454" height="61" alt="best_validation_loss" src="https://github.com/user-attachments/assets/96085f87-a543-4772-bd32-dc831a1c0790" />
</div>

<table>
  <tr>
    <td><img width="620" height="461" alt="train_val_loss" src="https://github.com/user-attachments/assets/c40fbb0b-aca7-451a-8705-4662b58c98a5" e/></td>
    <td><img width="620" height="461" alt="val_acc" src="https://github.com/user-attachments/assets/b7dd25c3-1981-45c5-a925-b781aed0309d" /></td>
  </tr>
</table>



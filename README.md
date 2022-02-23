# Self-Supervised Learning for Image Segmentation

Self-Supervised pretraining for the Carvana Image Masking Challenge (https://www.kaggle.com/c/carvana-image-masking-challenge)

Reimplemented RotNet as described in this paper: https://arxiv.org/abs/1803.07728

- dataset.py
    - CarvanaDataset: dataset class for preparing the data to be trained with the U-Net
    - RotDataset: dataset class for preparing the data to be trained with the RotNet
    
- models.py
    - DoubleConv: module used in the U-Net that performs convolution, followed by batch normalization, followed by ReLU activation and repeats that sequence twice
    - UNet: class that implements U-Net
    - RotNet: class that implements RotNet, which is the same as the contracting path of the U-Net with a fully connected layer at the end
    
- train\_rotnet.py
    - main: trains and evaluates the RotNet and plots loss and accuracy curves 
    - train: trains one epoch of the model and computes the average loss for that epoch 
    - eval: evaluates one epoch of the model and computes the average loss and accuracy for that epoch 
    
- train.py
    - Dice: computes Dice Score 
    - main: trains and evaluates the U-Net and plots loss and Dice curves 
    - train: trains one epoch of the model and computes the average loss for that epoch 
    - eval: evaluates one epoch of the model and computes the average loss, average Dice, Dice standard deviation, and median Dice for that epoch
    

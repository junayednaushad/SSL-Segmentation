# Self-Supervised Learning for Image Segmentation


- dataset.py
    - CarvanaDataset: dataset class for preparing the data to be trained with the U-Net (written from scratch)
    - RotDataset: dataset class for preparing the data to be trained with the RotNet (written from scratch)
    
- models.py
    - DoubleConv: module used in the U-Net that performs convolution, followed by batch normalization, followed by ReLU activation and repeats that sequence twice (obtained from GitHub)
    - UNet: class that implements the U-Net architecture described in this project (obtained from GitHub)
    - RotNet: class that implements the RotNet, which is the same as the contracting path of the U-Net with a fully connected layer at the end (refactored U-Net code with some modifications)
    - I did not implement the U-Net with ResNet34 encoder, instead I just imported it form the segmentation\_models\_pytorch package in the train.py file
    
- train\_rotnet.py
    - main: trains and evaluates the RotNet and plots loss and accuracy curves (written from scratch)
    - train: trains one epoch of the model and computes the average loss for that epoch (written from scratch)
    - eval: evaluates one epoch of the model and computes the average loss and accuracy for that epoch (written from scratch)
    
- train.py
    - Dice: computes Dice Score (written from scratch)
    - main: trains and evaluates the U-Net and plots loss and Dice curves (written from scratch)
    - train: trains one epoch of the model and computes the average loss for that epoch (written from scratch)
    - eval: evaluates one epoch of the model and computes the average loss, average Dice, Dice standard deviation, and median Dice for that epoch (written from scratch)
    

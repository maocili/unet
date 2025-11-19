# unet

## Project Structure

```
unet-reconstruction/
├── model.py                  # Defines the U-Net architecture (contracting & expansive paths)
├── train.py                  # Main script for training the model and saving weights
├── predict.py                # Script to load the model and visualize predictions
└── utils/
    ├── dataset.py            # Handles loading image/mask pairs (TiffDataset class)
    ├── loss_function.py      # Implements Dice Loss and coefficients
    ├── transformers.py       # Data preprocessing and normalization logic
    ├── data_augmentation.py  # Tools for augmenting training data
    └── weights.py            # Weight initialization function (Kaiming init) 
```

Thanks to:

https://www.kaggle.com/code/hamzamohiuddin/u-net-implementation-part-4/notebook#Create-Augmented-Dataset

https://github.com/milesial/Pytorch-UNet/blob/master/train.py#L197

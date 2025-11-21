# unet

## Project Structure

```
unet/
├── model.py                  # Defines the U-Net architecture (contracting & expansive paths)
├── train.py                  # Main script for training the model and saving weights
├── predict.py                # Script to load the model and visualize predictions
├── script/
│   └── data_process.py       # Data preprocessing and offline augmentation script (Micro-CT image processing pipeline)
└── utils/
    ├── dataset.py            # Handles loading image/mask pairs (TiffDataset class)
    ├── transformers.py       # Data preprocessing, normalization, and online augmentation logic (MicroTransformers class)
    ├── weights.py            # Weight initialization function (Kaiming init)
    └── loss_function/        # Collection of loss function implementations
        ├── combo.py          # Combo loss function implementation (e.g., Combo Loss, Focal+Dice)
        ├── dice.py           # Dice coefficient and Dice Loss implementation
        └── iou.py            # IoU coefficient calculation and Loss implementation
```

Thanks to:

https://www.kaggle.com/code/hamzamohiuddin/u-net-implementation-part-4/notebook#Create-Augmented-Dataset

https://github.com/milesial/Pytorch-UNet/blob/master/train.py#L197

import os
import shutil
import kagglehub

import numpy as np
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v2 as iio
from scipy import ndimage
from scipy.ndimage import zoom, rotate, shift


download_path = "./isbi-2012-chanllenge"

if not os.path.exists(download_path):
    path = kagglehub.dataset_download("hamzamohiuddin/isbi-2012-challenge")
    os.makedirs(download_path, exist_ok=True)
    dest = os.path.join(download_path, os.path.basename(path))
    shutil.move(path, dest)

print("Path to dataset files:", download_path)

img_root = os.path.join(download_path,"/train/imgs")
lab_root = os.path.join(download_path,"/train/labels")


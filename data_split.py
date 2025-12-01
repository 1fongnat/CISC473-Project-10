import tensorflow_datasets as tfds
import os
from PIL import Image
import random
import glob
import shutil
from pathlib import Path

"""
Split dataset into train, validation, and test directories
"""

SAVE_DIR = "./HiFiC/data/clic_all"

# Split data 80:10:10
random.seed(0)
files = glob.glob("*.png", root_dir=SAVE_DIR)
random.shuffle(files)
train_thres = round(0.8 * len(files))
val_thres = round(0.9 * len(files))

train_files = files[:train_thres]
val_files = files[train_thres:val_thres]
test_files = files[val_thres:]

for i in train_files:
    shutil.move(Path(SAVE_DIR) / i, Path(SAVE_DIR) / "train" / i)

for i in val_files:
    shutil.move(Path(SAVE_DIR) / i, Path(SAVE_DIR) / "validation" / i)

for i in test_files:
    shutil.move(Path(SAVE_DIR) / i, Path(SAVE_DIR) / "test" / i)

print(len(glob.glob("*.png", root_dir=os.path.join(SAVE_DIR, 'train'))))
print(len(glob.glob("*.png", root_dir=os.path.join(SAVE_DIR, 'validation'))))
print(len(glob.glob("*.png", root_dir=os.path.join(SAVE_DIR, 'test'))))
import tensorflow_datasets as tfds
import os
from PIL import Image

SAVE_DIR = "./HiFiC/data/clic-small"

ds = tfds.load("clic", split="train", shuffle_files=False)
os.makedirs(SAVE_DIR, exist_ok=True)

N = 200   # <-- small subset size (change as you like)

for i, example in enumerate(ds.take(N)):
    img = example["image"].numpy()
    Image.fromarray(img).save(os.path.join(SAVE_DIR, f"{i:05d}.png"))
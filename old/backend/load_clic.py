import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load CLIC dataset
ds = tfds.load('clic', split='train', shuffle_files=True)

# Show a random image
example = next(iter(ds.shuffle(1000).take(1)))
image = example['image'].numpy()
plt.imshow(image)
plt.axis("off")
plt.show()
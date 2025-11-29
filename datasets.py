import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CLICDataset(Dataset):
    """CLIC dataset for training and validation."""
    def __init__(self, root, patch_size=256, transform=None):
        self.root = root
        self.image_files = sorted(glob.glob(os.path.join(self.root, "*.png")))

        if transform:
            self.transform = transform
        elif patch_size:
            self.transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        """Loads an image and applies transformations."""
        try:
            img = Image.open(self.image_files[index]).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {self.image_files[index]}: {e}")
            return None

    def __len__(self):
        return len(self.image_files)
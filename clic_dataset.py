from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CLICDataset(Dataset):
    """
    Using CLIC dataset for training
    """
    def __init__(self, root):
        self.root = root
        self.files = [
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg"))
        ]
        self.files.sort()

        self.transform = transforms.Compose([
            transforms.RandomCrop(256),  # HiFiC supports up to 256x256
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = os.path.join(self.root, self.files[idx])
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        return {"image": img}

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CLICDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted(os.listdir(root))
        self.transform = transforms.Compose([
            transforms.ToTensor(),   # float in [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = os.path.join(self.root, self.files[idx])
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        return {"image": img}


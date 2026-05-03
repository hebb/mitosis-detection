from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatchDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = Path(root_dir)

        self.samples = []
        for label_name, label in [("false", 0), ("true", 1)]:
            folder = self.root_dir / label_name
            for path in sorted(folder.glob("*.png")):
                self.samples.append((path, label))

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation([0, 270]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

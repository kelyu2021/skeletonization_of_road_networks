import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RoadDataset(Dataset):
    def __init__(self, image_dir, label_dir, geojson_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.geojson_dir = geojson_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

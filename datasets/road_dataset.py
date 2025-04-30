import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RoadDataset(Dataset):
    def __init__(self, image_dir, label_dir, geojson_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.geojson_dir = geojson_dir

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.label_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')


        image = self.img_transform(image)
        label = self.label_transform(label)

        return image, label

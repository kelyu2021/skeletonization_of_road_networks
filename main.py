import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from datetime import datetime
import config
from datasets.road_dataset import RoadDataset
from models.unet import UNet
from trainers.trainer import Trainer
from evaluators.evaluator import Evaluator
import config

timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# 自动选择 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = RoadDataset(config.image_dir, config.label_dir, config.geojson_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 创建模型
model = UNet(1, 1).to(device)

# 训练
trainer = Trainer(model, config.model_id, train_loader, device)
trainer.train()

# 加载最优模型
model.load_state_dict(torch.load(config.model_save_path, weights_only=True))

# 评估
evaluator = Evaluator(model, test_loader, device)
evaluator.evaluate()

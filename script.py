import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from datetime import datetime
from datasets.road_dataset import RoadDataset
from models.unet import UNet
from trainers.trainer import Trainer
from evaluators.evaluator import Evaluator
import config

timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = RoadDataset(config.image_dir, config.label_dir, config.geojson_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# init model
model = UNet(1, 1).to(device)

# train
trainer = Trainer(model, train_loader, device)
trainer.train()

# choose best model
model.load_state_dict(torch.load(f"{config.model_save_path}/model_{config.loss_fn}_epoch{config.num_epochs}_dtw{config.distance_transform_weight}.pth", weights_only=True))

# evaluate
evaluator = Evaluator(model, test_loader, device)
evaluator.evaluate()

import torch
import config
from tools import visualize_batch
from skimage.morphology import binary_erosion, disk
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

class Trainer:

    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.num_epochs = config.num_epochs
        self.best_loss = float('inf')
        self.max_thin_iters = config.max_thin_iters

    # Apply Progressive Morphological Thinning to Labels
    # 1.Preprocess labels in each epoch with increasing degrees of morphological thinning.
    # 2.This simulates the de-thickening process and teaches the model to predict thinner and more refined outputs.
    def progressively_thin(self, label, epoch, total_epochs):

        if epoch < int(0.3 * total_epochs):
            return label

        label_np = label.squeeze().cpu().numpy() > 0.5  # 确保是二值图

        iterations = int((epoch / total_epochs) * self.max_thin_iters)

        # 使用 erosion 模拟 de-thickening
        for _ in range(iterations):
            label_np = binary_erosion(label_np, disk(1))  # 每次腐蚀1像素

            # 如果腐蚀过度，保留上一次
            if label_np.sum() < 0.1:
                break

        thinned = torch.tensor(label_np.astype(np.float32), device=label.device).unsqueeze(0)
        return thinned

    def compute_distance_transform_loss(self, prediction, label):
        pred = torch.sigmoid(prediction).squeeze(1)
        loss = 0.0
        for i in range(pred.size(0)):
            label_np = label[i].squeeze().cpu().numpy() > 0.5
            dist_map = distance_transform_edt(label_np)
            dist_map = torch.tensor(dist_map, dtype=torch.float32, device=label.device)
            dist_map = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min() + 1e-8)  # normalize
            loss += torch.mean((1 - pred[i]) * dist_map)
        return loss / pred.size(0)


    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Apply progressive thinning to labels
                labels_thin = torch.stack([self.progressively_thin(lbl, epoch, self.num_epochs) for lbl in labels])

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                predictions = torch.sigmoid(outputs) > 0.5
                loss = self.criterion(outputs, labels_thin)
                loss += self.compute_distance_transform_loss(outputs, labels_thin.detach())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    visualize_batch(inputs, labels, outputs, 'train', save_dir=config.outputs,epoch=epoch, batch_id=i)

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            # 保存最好模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save(self.model.state_dict(), config.model_save_path)
                print(f"Best model saved at epoch {epoch+1} with loss {avg_loss:.4f}")

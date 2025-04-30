import torch
import config
from tools import visualize_batch
from skimage.morphology import thin
import numpy as np

class Trainer:

    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.num_epochs = config.num_epochs
        self.best_loss = float('inf')

    # Apply Progressive Morphological Thinning to Labels
    # 1.Preprocess labels in each epoch with increasing degrees of morphological thinning.
    # 2.This simulates the de-thickening process and teaches the model to predict thinner and more refined outputs.
    def progressively_thin(label, epoch, total_epochs):
        thinning_fraction = epoch / total_epochs
        # Convert tensor to numpy and apply thinning based on epoch
        label_np = label.squeeze().cpu().numpy()
        thinned = thin(label_np)
        # Optionally apply thinning iteratively based on epoch
        for _ in range(int(thinning_fraction * 10)):  # adjust factor as needed
            thinned = thin(thinned)
        return torch.tensor(thinned, device=label.device).unsqueeze(0).float()

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Apply progressive thinning to labels
                labels = torch.stack([self.progressively_thin(lbl, epoch, self.num_epochs) for lbl in labels])

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                predictions = torch.sigmoid(outputs) > 0.5
                loss = self.criterion(outputs, labels)
                # loss += dice_loss(F.sigmoid(outputs).squeeze(1), labels.squeeze(1), multiclass=False)
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

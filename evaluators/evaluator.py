import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tools import visualize_batch
import config

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_mse = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs) > 0.5

                total_mse += torch.mean((preds.float() - labels) ** 2).item()
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

                # Visualize the first image in the batch
                # visualize_batch(inputs, labels, preds, 'test')
                visualize_batch(inputs, labels, outputs, 'test', save_dir=config.outputs, epoch=None, batch_id=i)

        avg_mse = total_mse / len(self.test_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Test MSE: {avg_mse:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

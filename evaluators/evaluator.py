import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tools import visualize_batch
import config
import cv2
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


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
        # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        self.evaluate_node_precision_recall()

    def evaluate_node_precision_recall(self):
        def get_valent_nodes(skeleton):
            kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
            val_map = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
            coords = np.argwhere(skeleton)
            val_dict = {}
            for y, x in coords:
                neighbors = val_map[y, x] - 10
                if neighbors in [1, 2, 3, 4]:
                    val_dict.setdefault(neighbors, []).append((y, x))
            return val_dict

        match_radius = 3
        # tolerance = lambda a, b: cdist(np.array(a), np.array(b)) <= match_radius

        self.model.eval()
        node_stats = {v: {'tp': 0, 'gt': 0, 'pred': 0} for v in [1, 2, 3, 4]}

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                best_thresh = 0.5
                best_f1 = 0

                for t in np.arange(0.1, 0.9, 0.05):
                    pred_mask = torch.sigmoid(outputs) > t
                    f1 = f1_score(labels.cpu().numpy().flatten(), pred_mask.cpu().numpy().flatten())
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t

                #print(f"Best threshold: {best_thresh}, Best F1: {best_f1}")

                predictions = torch.sigmoid(outputs) > best_thresh


                for i in range(inputs.size(0)):
                    gt = labels[i, 0].cpu().numpy() > 0.5
                    pred = predictions[i, 0].cpu().numpy()

                    gt_skel = skeletonize(gt)
                    pred_skel = skeletonize(pred)

                    gt_nodes = get_valent_nodes(gt_skel)
                    pred_nodes = get_valent_nodes(pred_skel)

                    # print(f'gt_skel: {gt_skel}')
                    # print(f'pred_skel: {pred_skel}')
                    # print(f'gt_nodes: {gt_nodes}')
                    # print(f'pred_nodes: {pred_nodes}')

                    for v in [1, 2, 3, 4]:
                        gt_pts = gt_nodes.get(v, [])
                        pred_pts = pred_nodes.get(v, [])

                        node_stats[v]['gt'] += len(gt_pts)
                        node_stats[v]['pred'] += len(pred_pts)

                        if gt_pts and pred_pts:
                            dists = cdist(gt_pts, pred_pts)
                            matched = (dists <= match_radius)
                            gt_matched = set()
                            pred_matched = set()
                            for gi, row in enumerate(matched):
                                for pi, m in enumerate(row):
                                    if m and gi not in gt_matched and pi not in pred_matched:
                                        gt_matched.add(gi)
                                        pred_matched.add(pi)
                                        node_stats[v]['tp'] += 1
        print("\nNode Precision & Recall:")
        for v in [1, 2, 3, 4]:
            stats = node_stats[v]
            prec = stats['tp'] / stats['pred'] if stats['pred'] > 0 else 0
            rec = stats['tp'] / stats['gt'] if stats['gt'] > 0 else 0
            print(f"{v}-valent → Precision: {prec:.3f}, Recall: {rec:.3f}")

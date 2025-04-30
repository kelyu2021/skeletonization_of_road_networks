import os
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_batch(inputs, labels, predictions, phase='train', save_dir=None, epoch=None, batch_id=None):
    """
    Visualizes a batch of inputs, ground truth labels, and predictions.
    Optionally saves the figure to a directory.

    Args:
        inputs (tensor): A batch of input images.
        labels (tensor): A batch of ground truth labels.
        predictions (tensor): A batch of model predictions.
        phase (str): 'train' or 'test' (used in the figure title or filename).
        save_dir (str, optional): If provided, save the figure into this directory.
        filename (str, optional): Name of the saved figure file.
    """
    # num_samples = inputs.shape[0] if phase == 'test' else 1
    num_samples = inputs.shape[0]
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    # fig.suptitle(f'{phase.upper()}, EPOCH_{epoch}_BATCH_{batch_id}', fontsize=16)

    if num_samples == 1:
        axes = [axes]  # make iterable for consistency

    for i in range(num_samples):
        axes[i][0].imshow(inputs[i].cpu().squeeze(), cmap='gray')
        axes[i][0].set_title('Input Image')
        axes[i][0].axis('off')

        axes[i][1].imshow(labels[i].cpu().squeeze(), cmap='gray')
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')

        axes[i][2].imshow(predictions[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i][2].set_title('Model Prediction')
        axes[i][2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'{phase.upper()}, EPOCH_{epoch}_BATCH_{batch_id}', fontsize=16)

    filename = f"{phase}_{epoch}_{batch_id}.png"

    # If save_dir is given, save the figure
    if save_dir:
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # save_dir = os.path.join(save_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        # print(f"Saved figure at {save_path}")

    plt.show()
    plt.close(fig)  # Very important to release memory if you save many figures

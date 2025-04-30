import os
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{phase.upper()}, EPOCH_{epoch}_BATCH_{batch_id}', fontsize=16)

    # Show the first sample in the batch
    axes[0].imshow(inputs[0].cpu().squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(labels[0].cpu().squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(predictions[0].cpu().detach().numpy().squeeze(), cmap='gray')
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')

    plt.tight_layout()

    filename = f"{phase}_{epoch}_{batch_id}.png"

    # If save_dir is given, save the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        # print(f"Saved figure at {save_path}")

    plt.show()
    plt.close(fig)  # Very important to release memory if you save many figures

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import random

from sklearn.metrics import confusion_matrix
from .utils import denormalize_image, collect_predictions


def plot_config() -> None:
    """
    Show plot and apply tight layout.
    """
    plt.tight_layout()
    plt.show()


def class_distribution_barplot(df: pd.DataFrame, x: str, y: str, title: str, x_rotation=30) -> None:
    """
    Plot bar plot distribution for classes.
    Parameters:
        df: Data dataframe.
        x:  Target feature.
        y:  Target class.
        title: Title of plot (one word).
        x_rotation: Rotation of x-axis default = 30 degrees.
    """
    ax = sns.barplot(df, x=x, y=y, edgecolor="black")

    for i in ax.containers:
        ax.bar_label(i, )

    plt.xlabel(f"{title.capitalize()} Class")
    plt.ylabel("Count")
    plt.title(f"{title.capitalize()} Class Distribution")
    plt.xticks(rotation=x_rotation)
    plot_config()


def horizontal_hist_box_plot(df: pd.DataFrame, x: str, hue: str = None, bins: int = 30) -> None:
    """
    Plot horizontal kde plot.
    Parameters:
        df: Data dataframe.
        x:  Target feature.
        hue:  Target feature.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(data=df, bins=bins, edgecolor="black", ax=ax1)
    ax1.set_title(f"{x} Histogram Over {hue}" if hue else f"{x} Histogram")

    sns.boxplot(data=df, x=x, hue=hue, ax=ax2)
    ax2.set_title(f"{x} Distribution Over {hue}" if hue else f"{x} Distribution")

    ax1.set_xlabel(f"{x}")

    plot_config()


def plot_regplot(
        x: pd.Series, y: pd.Series,
        title: str, alpha: int = 0.5,
        xlabel: str = None, ylabel: str = None) -> None:
    """
    Plot a regression plot using Seaborn.
    Parameters:
        x: X-axis data
        y: Y-axis data
        title: Title of plot.
        alpha: Alpha parameter.
        xlabel: Label of x-axis.
        ylabel: Label of y-axis.
    """
    sns.regplot(x=x, y=y, scatter_kws={'alpha': alpha})
    plt.title(title)
    plt.xlabel(xlabel if xlabel else "X-axis")
    plt.ylabel(ylabel if ylabel else "Y-axis")

    plot_config()


def plot_learning_curve(loss_history: dict) -> None:
    """
    Plot training and validation loss over epochs.

    Parameters:
        loss_history: Loss history.
    """
    min_epochs = min(len(loss_history["train_loss"]), len(loss_history["val_loss"]))
    epochs = range(1, min_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history["train_loss"][:min_epochs], label="Training Loss")
    plt.plot(epochs, loss_history["val_loss"][:min_epochs], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)

    # Set xticks to be whole numbers only
    plt.xticks(ticks=range(1, min_epochs + 1))

    plot_config()


def cm_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list, model_name: str, ax: plt.Axes,
              xticks_rotation: int = 0) -> None:
    """
    Calculates and plots a confusion matrix heatmap.

    Parameters:
        labels: True labels (1D array-like).
        preds: Predicted labels (1D array-like).
        class_names: List of class names for display.
        model_name: Name of the model for the title.
        ax: Axis object where the heatmap will be plotted.
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Plot confusion matrix heatmap
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".4g", ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.tick_params(axis='x', rotation=xticks_rotation)


def visualize_predictions(model, test_loader, class_names, num_images: int = 9, device=None) -> None:
    """
    Visualize predictions on a few test images.

    Parameters:
        model (torch.nn.Module): The trained model for visualization.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list[str]): List of class names for human-readable labels.
        num_images (int, optional): Number of images to visualize. Defaults to 9.
        device (torch.device, optional): Device (CPU or GPU) to run the visualization. Defaults to None.

    Returns:
        None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Fetch a batch of test images and labels
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    images = images * std[:, None, None] + mean[:, None, None]
    images = images.cpu()

    # Plot the images with predictions
    fig, axes = plt.subplots(1, min(num_images, len(images)), figsize=(15, 5))
    for img, label, pred, ax in zip(images[:num_images], labels[:num_images], preds[:num_images], axes):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"True: {class_names[label.item()]}\nPred: {class_names[pred.item()]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, class_names, num_images: int = 9) -> None:
    """
    Visualize random predictions from different classes.
    Parameters:
        model (torch.nn.Module): The trained model for visualization.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list[str]): List of class names for human-readable labels.
        num_images (int, optional): Number of images to visualize. Defaults to 9.
    """
    device = next(model.parameters()).device

    # Collect predictions
    images, labels, preds = collect_predictions(model, test_loader, device)

    # Group by class for diversity
    class_samples = {cls: [] for cls in range(len(class_names))}
    for img, lbl, pred in zip(images, labels, preds):
        class_samples[lbl.item()].append((img, lbl, pred))

    # Select random samples
    selected_samples = [
        random.choice(class_samples[cls])
        for cls in range(len(class_names))
        if class_samples[cls]
    ]
    while len(selected_samples) < num_images:
        cls = random.choice(list(class_samples.keys()))
        if class_samples[cls]:
            selected_samples.append(random.choice(class_samples[cls]))

    # Plot selected samples
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    fig, axes = plt.subplots(1, len(selected_samples), figsize=(15, 5))
    for (img, true, pred), ax in zip(selected_samples, axes):
        img = denormalize_image(img, mean, std).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"True: {class_names[true.item()]}\nPred: {class_names[pred.item()]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

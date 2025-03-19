import os
from PIL import Image
import numpy as np
from pathlib import Path

from sklearn.metrics import classification_report
from tqdm import tqdm
import shutil
from collections import defaultdict
import pandas as pd
from PIL import ImageEnhance
import random
from sklearn.model_selection import train_test_split
import torch


def denormalize_image(img, mean, std):
    """
    Denormalize an image tensor.

    Parameters:
        img: Image to be denormalized.
        mean: Mean
        std: Standard deviation
    """
    mean = torch.tensor(mean)[:, None, None]
    std = torch.tensor(std)[:, None, None]
    return img * std + mean


def collect_predictions(model, loader, device = None):
    """
    Run model inference and collect predictions, labels, and images.

    Parameters:
        model: Model to be used for inference.
        loader: Loader to be used for inference.
        device: Device to be used for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            predictions = outputs.argmax(dim=1)
            images.append(imgs.cpu())
            labels.append(lbls.cpu())
            preds.append(predictions.cpu())

    return torch.cat(images), torch.cat(labels), torch.cat(preds)


def is_image_file(file_path):
    """
    Check if the file is a valid image.

    Parameters:
        file_path: Path of the file.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_file_size(file_path):
    """Get file size in KB."""
    return os.path.getsize(file_path) / 1024


def resize_image(image, target_size=(224, 224)):
    """Resize an image.

    Parameters:
        image: Image to be resized.
        target_size: Target size of the resized image.
    """
    return image.resize(target_size)


def is_black_or_white(file_path, threshold=0.99):
    """
    Check if the image is mostly black or white.

    Parameters:
        file_path: Path of the file.
        threshold: Threshold for black or white.
    """
    try:
        with Image.open(file_path) as img:
            img = img.convert("L")  # Convert to grayscale
            pixels = np.array(img) / 255.0  # Normalize pixel values
            black_ratio = np.mean(pixels < (1 - threshold))  # Black
            white_ratio = np.mean(pixels > threshold)  # White
            return black_ratio > 0.95 or white_ratio > 0.95
    except Exception:
        return True


def clean_dataset(dataset_path):
    """Clean the dataset by validating images."""
    invalid_files = []
    size_issues = []
    black_or_white_images = []

    for folder in tqdm(Path(dataset_path).iterdir(), desc="Scanning folders"):
        if folder.is_dir():
            for file in folder.iterdir():
                if file.is_file():
                    # Check if it is valid image
                    if not is_image_file(file):
                        invalid_files.append(file)
                        continue

                    # Check if size of file is less than 1kb
                    file_size = get_file_size(file)
                    if file_size < 1:
                        size_issues.append((file, file_size))
                        continue

                    # Check if image is not all black or white
                    if is_black_or_white(file):
                        black_or_white_images.append(file)

    return invalid_files, size_issues, black_or_white_images


def move_bad_images(bad_images, base_bad_path, dataset_path):
    """
    Move bad images to the 'bad' folder, preserving their folder structure.
    If the file already exists in the target folder, delete it from the original location.

    Parameters:
    bad_images (list of Path): List of bad image file paths to move.
    base_bad_path (Path): Path to the base 'bad' folder.
    dataset_path (Path): Path to the original dataset folder.
    """

    bad_count = 0  # Initialize counter for bad images

    for file_path in bad_images:
        # Get the path to the files from dataset root
        relative_path = file_path.relative_to(dataset_path)

        # Create new folder if needed for bad files
        target_folder = base_bad_path / relative_path.parent
        target_folder.mkdir(parents=True, exist_ok=True)

        # Construct the target file path
        target_file_path = target_folder / file_path.name

        if target_file_path.exists():
            # If the file already exists in the target folder, delete it from the original location
            file_path.unlink()
            print(f"File already exists in target folder. Deleted {file_path}.")
        else:
            # Move the file to the target folder
            shutil.move(str(file_path), str(target_file_path))
            print(f"Moved {file_path} to {target_folder}.")

        bad_count += 1

    print(f"Moved {bad_count} bad images to {base_bad_path}.")


def analyze_dataset(dataset_path):
    """Analyze the dataset for class distribution and image properties."""
    class_distribution = defaultdict(int)
    image_sizes_kb = []
    image_dimensions = []

    for folder in tqdm(Path(dataset_path).iterdir(), desc="Analyzing folders"):
        if folder.is_dir():
            for file in folder.iterdir():
                if file.is_file() and is_image_file(file):
                    # Update class distribution
                    class_distribution[folder.name] += 1

                    # Update image size in MB
                    file_size = get_file_size(file)
                    image_sizes_kb.append({"File": str(file), "Size (KB)": file_size})

                    # Get image dimensions
                    try:
                        with Image.open(file) as img:
                            image_dimensions.append(img.size)  # (width, height)
                    except Exception:
                        continue

    # Convert image_sizes_kb to DataFrame
    image_sizes_df = pd.DataFrame(image_sizes_kb)

    return class_distribution, image_sizes_df, image_dimensions


def augment_image(image, crop_size = (224, 224)):
    """
    Augment the image with mirroring, rotating and adjusting brightness.

    Parameters:
        image (PIL.Image.Image): The input image.
        crop_size (tuple): The size of the crop (width, height).

    Returns:
    list: A list of augmented images.
    """

    width, height = image.size
    transforms = []

    # Mirroring
    transforms.append(image.transpose(Image.FLIP_LEFT_RIGHT))

    # Random rotation
    transforms.append(image.rotate(random.choice([-90, 90])))

    # Brightness adjustment
    transforms.append(ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2)))

    # Random crop
    if width > crop_size[0] and height > crop_size[1]:
        left = random.randint(0, width - crop_size[0])
        top = random.randint(0, height - crop_size[1])
        right = left + crop_size[0]
        bottom = top + crop_size[1]
        crop = image.crop((left, top, right, bottom))
        transforms.append(crop)

    return transforms


def split_data(input_path, output_path, train_ratio=0.7, val_ratio=0.15, stratify=True):
    """
    Splits images into train, validation, and test sets and copies them to respective folders.

    Parameters:
    input_path (str): Path to the input dataset organized by class.
    output_path (str): Path to the output directory for train, validation, and test folders.
    train_ratio (float): Proportion of images to include in the training set.
    val_ratio (float): Proportion of images to include in the validation set.
    stratify (bool): Whether to stratify splits by class labels.

    Returns:
    None
    """
    print("Splitting dataset into train, validation, and test...")

    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")

    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # List all valid image files
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                  if img.endswith(('.png', '.jpg', '.jpeg'))]

        total_images = len(images)
        print(f"Processing class '{class_name}' with {total_images} images.")

        # Skip classes with less than 3 images
        if total_images < 3:
            print(f"  Skipping class '{class_name}' as it has less than 3 images.")
            continue

        # Create a label list for stratification
        labels = [class_name] * len(images) if stratify else None

        # Split into train, validation, and test
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            images,
            labels,
            test_size=1 - train_ratio,
            random_state=42,
            stratify=labels if stratify else None
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files,
            temp_labels,
            test_size=val_ratio / (1 - train_ratio),
            random_state=42,
            stratify=temp_labels if stratify else None
        )

        # Function to copy files to destination
        def copy_files(files, dest_dir):
            dest_dir_class = os.path.join(dest_dir, class_name)
            os.makedirs(dest_dir_class, exist_ok=True)
            for file in files:
                shutil.copy(file, os.path.join(dest_dir_class, os.path.basename(file)))

        # Copy to respective folders
        copy_files(train_files, train_path)
        copy_files(val_files, val_path)
        copy_files(test_files, test_path)

        print(f"  Class '{class_name}' split as follows:")
        print(f"    Train: {len(train_files)}")
        print(f"    Validation: {len(val_files)}")
        print(f"    Test: {len(test_files)}")

    print("Dataset splitting complete.")


def preprocess_images(input_path: str, output_path: str, target_size = (224, 224), augment: bool = True) -> None:
    """
    Adjust images using augmentations and resizing.

    Parameters:
        input_path: Path to the directory containing class-wise images.
        output_path: Path to save processed images.
        target_size: Desired size for resizing (width, height).
        augment: Apply augmentation or not.

    Returns:
    """
    print("Processing images...")

    # Loop for all class folders in main folder
    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        processed_class_dir = os.path.join(output_path, class_name)
        os.makedirs(processed_class_dir, exist_ok=True)

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if not file_name.endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load and process image
            with Image.open(file_path) as img:
                img = img.convert("RGB")  # Convert all images are RGB set

                # Augmentation step if True
                if augment:
                    augmented_images = augment_image(img)
                else:
                    augmented_images = [img]

                # Resize each augmented image and save
                for idx, aug_img in enumerate(augmented_images):
                    resized_img = resize_image(aug_img, target_size)

                    # Save original or augmented versions
                    if idx == 0 and not augment:
                        save_name = file_name  # Original name if no augment
                    else:
                        save_name = f"{os.path.splitext(file_name)[0]}_aug{idx}.jpg"

                    resized_img.save(os.path.join(processed_class_dir, save_name))

    print("Processing complete.")


def count_files_in_split(split_path):
    counts = {}
    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)
        if os.path.isdir(class_dir):
            counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    return counts


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate the model and generate a classification report.

    Parameters:
        model (torch.nn.Module): The trained model for evaluation.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list[str]): List of class names for human-readable labels.
        device (torch.device, optional): Device (CPU or GPU) to run the evaluation. Defaults to None.

    Returns:
        tuple: (all_labels, all_preds) as numpy arrays for further analysis.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []

    # Perform evaluation and collect predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Generate and print the classification report
    report = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=class_names)
    print("Classification Report:")
    print(report)

    return all_labels.numpy(), all_preds.numpy()


from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_split_v2"
RESULTS_DIR = ROOT / "results_v2"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 5
SEED = 42

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EXPERIMENT 7: RESNET50 ON DATASET V2")
print("=" * 60)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 2. CHECK DATASET FOLDERS
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for class_name in ["adenoma", "hyperplastic"]:
        class_folder = folder / class_name

        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {class_folder}"
            )

# Training augmentation only.
# Validation and test images are not augmented.

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


# ============================================================
# 3. LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

# Load the test dataset for folder verification only.
# Do not run test evaluation until model selection is finished.
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

class_names = train_dataset.classes

if class_names != ["adenoma", "hyperplastic"]:
    raise ValueError(
        f"Unexpected class mapping: {class_names}"
    )

if val_dataset.classes != class_names:
    raise ValueError("Validation classes do not match training.")

if test_dataset.classes != class_names:
    raise ValueError("Test classes do not match training.")

if min(
    len(train_dataset),
    len(val_dataset),
    len(test_dataset)
) == 0:
    raise ValueError("At least one dataset split is empty.")

print("\nClass mapping:", train_dataset.class_to_idx)
print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

print("\nTraining class counts:")
for class_name in class_names:
    count = sum(
        1 for _, label in train_dataset.samples
        if train_dataset.classes[label] == class_name
    )
    print(f"  {class_name}: {count}")

print("\nValidation class counts:")
for class_name in class_names:
    count = sum(
        1 for _, label in val_dataset.samples
        if val_dataset.classes[label] == class_name
    )
    print(f"  {class_name}: {count}")

print("\nTest class counts:")
for class_name in class_names:
    count = sum(
        1 for _, label in test_dataset.samples
        if test_dataset.classes[label] == class_name
    )
    print(f"  {class_name}: {count}")


# num_workers=0 is a safe starting point on Windows.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 4. LOAD PRETRAINED RESNET50
# ============================================================

print("\nLoading pretrained ResNet50...")

try:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

except Exception as error:
    raise RuntimeError(
        "Could not load pretrained ResNet50 weights. "
        "Check your internet connection and torchvision installation. "
        "The script stops rather than silently using random weights."
    ) from error

# Replace the final classification layer for two classes.
number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, len(class_names))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 5. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def run_training_epoch(model, loader):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


@torch.no_grad()
def run_validation(model, loader):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

history = []

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

checkpoint_path = RESULTS_DIR / "best_resnet50_v2.pth"

for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy = run_training_epoch(
        model, train_loader
    )

    val_loss, val_accuracy = run_validation(
        model, val_loader
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    # Select the checkpoint using validation loss only.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": best_state,
            "class_to_idx": train_dataset.class_to_idx,
            "best_val_loss": best_val_loss,
            "image_size": IMAGE_SIZE,
        }, checkpoint_path)

        print("  -> Best validation checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping after epoch {epoch}. "
            "Validation loss did not improve."
        )
        break


# Save training history.
history_path = RESULTS_DIR / "training_history_v2.csv"
pd.DataFrame(history).to_csv(history_path, index=False)


# ============================================================
# 7. REPORT BEST VALIDATION RESULT
# ============================================================

if best_state is None:
    raise RuntimeError("No best model checkpoint was created.")

model.load_state_dict(best_state)

best_val_loss, best_val_accuracy = run_validation(
    model, val_loader
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Checkpoint: {checkpoint_path}")
print(f"Training history: {history_path}")

print("\nThe test dataset has NOT been evaluated.")
print("Keep it untouched until model selection is complete.")


from pathlib import Path
import csv
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

DATA_DIR = PROJECT_DIR / "data_split"
RESULTS_DIR = PROJECT_DIR / "results" / "experiment3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
PATIENCE = 5

CLASSES = ["adenoma", "hyperplastic"]

# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("EXPERIMENT 3: FROZEN RESNET50 FEATURE EXTRACTOR")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 3. PREPROCESSING
# ============================================================

# Keep augmentation mild because endoscopic color and vascular
# patterns can be diagnostically important.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(
        brightness=0.05,
        contrast=0.05,
        saturation=0.0,
        hue=0.0,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation images receive no random augmentation.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ImageFolder normally converts images to RGB.
train_dataset = datasets.ImageFolder(
    str(DATA_DIR / "train"),
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    str(DATA_DIR / "val"),
    transform=val_transform,
)

if train_dataset.classes != CLASSES:
    raise ValueError(
        f"Unexpected training classes: {train_dataset.classes}"
    )

if val_dataset.class_to_idx != train_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings differ.")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")

print("\nClass mapping:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

# ============================================================
# 4. LOAD PRETRAINED RESNET50 AND FREEZE BACKBONE
# ============================================================

print("\nLoading pretrained ResNet50...")

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all pretrained parameters.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the final classification layer.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(CLASSES))

# The new classification layer remains trainable.
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
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
# 5. TRAINING AND VALIDATION
# ============================================================

def run_epoch(loader, training):
    # Keep the pretrained backbone in evaluation mode so its
    # BatchNorm running statistics do not change.
    model.eval()

    if training:
        model.fc.train()

    total_loss = 0.0
    labels_all = []
    predictions_all = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(labels_all, predictions_all)

    return average_loss, accuracy


# ============================================================
# 6. TRAIN WITH EARLY STOPPING
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"LR: {current_lr:.6f} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": best_state,
            "classes": CLASSES,
            "class_to_idx": train_dataset.class_to_idx,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "image_size": IMAGE_SIZE,
        }, RESULTS_DIR / "best_resnet50_experiment3.pth")

        print("  -> Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}"
        )
        break

# Save history separately from Experiments 1 and 2.
with (RESULTS_DIR / "training_history.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# ============================================================
# 7. EVALUATE BEST CHECKPOINT
# ============================================================

model.load_state_dict(best_state)
model.eval()

actual_all = []
predicted_all = []
probabilities_all = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        actual_all.extend(labels.tolist())
        predicted_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

accuracy = accuracy_score(actual_all, predicted_all)

report = classification_report(
    actual_all,
    predicted_all,
    labels=[0, 1],
    target_names=CLASSES,
    zero_division=0,
)

matrix = confusion_matrix(
    actual_all,
    predicted_all,
    labels=[0, 1],
)

with (RESULTS_DIR / "evaluation_report.txt").open(
    "w", encoding="utf-8"
) as file:
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Best validation loss: {best_val_loss:.4f}\n")
    file.write(f"Validation accuracy: {accuracy:.4f}\n\n")
    file.write(report)
    file.write("\n\nConfusion matrix:\n")
    file.write(f"Class order: {CLASSES}\n")
    file.write(str(matrix))
    file.write("\n")

with (RESULTS_DIR / "validation_predictions.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "filename",
        "actual_class",
        "predicted_class",
        "prob_adenoma",
        "prob_hyperplastic",
        "correct",
    ])

    for index, (actual, predicted, probs) in enumerate(
        zip(actual_all, predicted_all, probabilities_all)
    ):
        filename = Path(val_dataset.samples[index][0]).name

        writer.writerow([
            filename,
            CLASSES[actual],
            CLASSES[predicted],
            f"{probs[0]:.6f}",
            f"{probs[1]:.6f}",
            actual == predicted,
        ])

print("\n" + "=" * 60)
print("EXPERIMENT 3 COMPLETE")
print("=" * 60)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)
print("\nConfusion matrix:")
print(matrix)
print("\nResults folder:", RESULTS_DIR)

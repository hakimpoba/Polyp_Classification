
from pathlib import Path
import copy
import random
import csv

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

DATA_DIR = PROJECT_DIR / "data_augmented"
RESULTS_DIR = PROJECT_DIR / "results" / "experiment4"

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 25
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
PATIENCE = 5
SEED = 42

CLASSES = ["adenoma", "hyperplastic"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EXPERIMENT 4: RESNET50 WITH OFFLINE DATA AUGMENTATION")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATA TRANSFORMS
# No random augmentation here: the training images have
# already been augmented and saved to data_augmented.
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(
        f"Could not find train/val folders inside {DATA_DIR}"
    )

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=image_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=image_transform
)

if train_dataset.classes != CLASSES:
    raise ValueError(
        f"Unexpected training classes: {train_dataset.classes}"
    )

if val_dataset.class_to_idx != train_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings differ.")

print("\nClass mapping:", train_dataset.class_to_idx)
print("Training files:", len(train_dataset))
print("Validation files:", len(val_dataset))

# Show the class distribution
for class_name in CLASSES:
    class_idx = train_dataset.class_to_idx[class_name]
    count = sum(
        1 for _, label in train_dataset.samples
        if label == class_idx
    )
    print(f"  Train {class_name}: {count}")

for class_name in CLASSES:
    class_idx = val_dataset.class_to_idx[class_name]
    count = sum(
        1 for _, label in val_dataset.samples
        if label == class_idx
    )
    print(f"  Val {class_name}: {count}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# MODEL: FULL RESNET50 FINE-TUNING
# ============================================================

print("\nLoading pretrained ResNet50...")

model = models.resnet50(
    weights=ResNet50_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ============================================================
# TRAINING / VALIDATION FUNCTION
# ============================================================

def run_epoch(loader, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images
    )

# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_acc = run_epoch(
        train_loader, training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader, training=False
    )

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "learning_rate": current_lr,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
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
            "image_size": IMAGE_SIZE
        }, RESULTS_DIR / "best_resnet50_experiment4.pth")

        print("  -> Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

# Save training history
pd.DataFrame(history).to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False
)

# ============================================================
# EVALUATE THE BEST CHECKPOINT
# ============================================================

if best_state is None:
    raise RuntimeError("No best model checkpoint was saved.")

model.load_state_dict(best_state)
model.eval()

y_true = []
y_pred = []
prediction_rows = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        probs = probabilities.cpu().numpy()
        labels_np = labels.numpy()

        for i in range(len(labels_np)):
            actual_idx = int(labels_np[i])
            predicted_idx = int(predictions[i])

            image_path, _ = val_dataset.samples[
                len(y_true)
            ]

            y_true.append(actual_idx)
            y_pred.append(predicted_idx)

            prediction_rows.append({
                "filename": Path(image_path).name,
                "actual_class": CLASSES[actual_idx],
                "predicted_class": CLASSES[predicted_idx],
                "prob_adenoma": float(probs[i][0]),
                "prob_hyperplastic": float(probs[i][1]),
                "correct": actual_idx == predicted_idx
            })

accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

matrix = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

pd.DataFrame(prediction_rows).to_csv(
    RESULTS_DIR / "validation_predictions.csv",
    index=False
)

report_text = (
    "EXPERIMENT 4: RESNET50 WITH OFFLINE DATA AUGMENTATION\n"
    f"Best epoch: {best_epoch}\n"
    f"Best validation loss: {best_val_loss:.4f}\n"
    f"Validation accuracy: {accuracy:.4f}\n\n"
    "Classification report:\n"
    f"{report}\n"
    "Confusion matrix (actual rows, predicted columns):\n"
    f"{matrix}\n"
)

(RESULTS_DIR / "evaluation_report.txt").write_text(
    report_text,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("EXPERIMENT 4 COMPLETE")
print("=" * 60)
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)
print("Confusion matrix:")
print(matrix)
print(f"\nResults saved to: {RESULTS_DIR}")

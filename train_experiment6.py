
from pathlib import Path
import copy
import csv
import json
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
    f1_score,
    recall_score,
)

# ============================================================
# EXPERIMENT 6 CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

DATA_DIR = PROJECT_DIR / "data_augmented"
RESULTS_DIR = PROJECT_DIR / "results" / "experiment6"

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 25
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
PATIENCE = 5
SEED = 42

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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 6: CLASS-WEIGHTED FROZEN RESNET50")
print("=" * 60)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATA TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=preprocess.mean,
        std=preprocess.std
    )
])

# ============================================================
# LOAD DATA
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(
        f"Dataset folders not found under {DATA_DIR}"
    )

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=transform
)

if train_dataset.classes != val_dataset.classes:
    raise ValueError(
        "Training and validation class mappings differ."
    )

class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx
num_classes = len(class_names)

if num_classes != 2:
    raise ValueError(
        f"Expected 2 classes, found {num_classes}: {class_names}"
    )

train_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
)

val_counts = np.bincount(
    val_dataset.targets,
    minlength=num_classes
)

print(f"\nClass mapping: {class_to_idx}")
print(f"Training files: {len(train_dataset)}")
print(f"Validation files: {len(val_dataset)}")

for index, name in enumerate(class_names):
    print(
        f"  {name}: "
        f"train={train_counts[index]}, "
        f"validation={val_counts[index]}"
    )

# ============================================================
# CLASS WEIGHTS
# ============================================================

# Inverse-frequency weighting:
# weight = total training samples / (classes * samples in class)
#
# Weights are normalized to have a mean of 1.
# This changes the relative penalty for class errors.

if np.any(train_counts == 0):
    raise ValueError("A training class has zero images.")

class_weights = len(train_dataset) / (
    num_classes * train_counts.astype(np.float64)
)

class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")
for index, name in enumerate(class_names):
    print(f"  {name}: {class_weights[index]:.4f}")

# ============================================================
# DATA LOADERS
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# BUILD FROZEN RESNET50
# ============================================================

model = models.resnet50(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    num_classes
)

# Only the final classification layer is trainable.
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

print("\nModel: ResNet50 with ImageNet pretrained weights")
print("Backbone: Frozen")
print("Final fully connected layer: Trainable")

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)

print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(loader, training):
    # Keep the whole model in evaluation mode to freeze the
    # backbone's batch-normalization running statistics.
    model.eval()

    # Only the final layer is being optimized.
    if training:
        model.fc.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_accuracy = run_epoch(
        val_loader,
        training=False
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            best_state,
            RESULTS_DIR / "best_model.pth"
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

if best_state is None:
    raise RuntimeError("No best model checkpoint was produced.")

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    RESULTS_DIR / "training_history.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ]
    )
    writer.writeheader()
    writer.writerows(history)

# ============================================================
# EVALUATE BEST CHECKPOINT
# ============================================================

model.load_state_dict(best_state)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )
        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

print("\n" + "=" * 60)
print("EXPERIMENT 6 RESULTS")
print("=" * 60)
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(matrix)

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("EXPERIMENT 6: CLASS-WEIGHTED FROZEN RESNET50\n\n")
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Best validation loss: {best_val_loss:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro recall: {macro_recall:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(np.array2string(matrix))

# ============================================================
# SAVE PER-IMAGE PREDICTIONS
# ============================================================

prediction_path = RESULTS_DIR / "validation_predictions.csv"

with open(
    prediction_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "actual_class",
        "predicted_class",
        "prob_adenoma",
        "prob_hyperplastic",
        "correct"
    ])

    for index, (actual, predicted, probs) in enumerate(
        zip(all_labels, all_predictions, all_probabilities)
    ):
        filename = Path(
            val_dataset.samples[index][0]
        ).name

        writer.writerow([
            filename,
            class_names[actual],
            class_names[predicted],
            float(probs[class_to_idx["adenoma"]]),
            float(probs[class_to_idx["hyperplastic"]]),
            actual == predicted
        ])

# ============================================================
# SAVE EXPERIMENT SUMMARY
# ============================================================

summary = {
    "experiment": 6,
    "model": "ResNet50 pretrained on ImageNet",
    "backbone_frozen": True,
    "trainable_layer": "fc",
    "loss": "class_weighted_cross_entropy",
    "class_mapping": class_to_idx,
    "class_weights": {
        name: float(class_weights[class_to_idx[name]])
        for name in class_names
    },
    "training_files": len(train_dataset),
    "validation_files": len(val_dataset),
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "validation_accuracy": accuracy,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "confusion_matrix": matrix.tolist()
}

with open(
    RESULTS_DIR / "summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nSaved files:")
print("- best_model.pth")
print("- training_history.csv")
print("- classification_report.txt")
print("- validation_predictions.csv")
print("- summary.json")

print("\nExperiment 6 completed.")

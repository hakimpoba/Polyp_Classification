
from pathlib import Path
import copy
import random
import csv
import json

import numpy as np
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
RESULTS_DIR = PROJECT_DIR / "results" / "experiment5"

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 25
LEARNING_RATE = 0.0001
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("EXPERIMENT 5: FROZEN RESNET50")
print("=" * 55)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATA TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ============================================================
# LOAD DATASET
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(
        "Dataset folders not found. Check DATA_DIR."
    )

train_dataset = datasets.ImageFolder(
    train_dir, transform=transform
)

val_dataset = datasets.ImageFolder(
    val_dir, transform=transform
)

if train_dataset.classes != val_dataset.classes:
    raise ValueError(
        "Training and validation class mappings differ."
    )

class_names = train_dataset.classes

print(f"\nClass mapping: {train_dataset.class_to_idx}")
print(f"Training files: {len(train_dataset)}")
print(f"Validation files: {len(val_dataset)}")

for class_name, count in zip(
    train_dataset.classes,
    np.bincount(train_dataset.targets)
):
    print(f"  Train {class_name}: {count}")

for class_name, count in zip(
    val_dataset.classes,
    np.bincount(val_dataset.targets)
):
    print(f"  Validation {class_name}: {count}")

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
# BUILD PRETRAINED RESNET50
# ============================================================

model = models.resnet50(weights=weights)

# Freeze all pretrained parameters.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the final classification layer.
input_features = model.fc.in_features
number_of_classes = len(class_names)

model.fc = nn.Linear(
    input_features,
    number_of_classes
)

# The replacement layer is trainable.
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

trainable_parameters = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nModel: ResNet50 with pretrained ImageNet weights")
print("Backbone: Frozen")
print("Classification layer: Trainable")
print(f"Trainable parameters: {trainable_parameters:,}")
print(f"Total parameters: {total_parameters:,}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.001
)

# ============================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def run_epoch(loader, training):
    model.train(training)

    if not training:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

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
        total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images
    )


# ============================================================
# TRAIN MODEL
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_acc = run_epoch(
        train_loader, training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader, training=False
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
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

        torch.save(
            best_state,
            RESULTS_DIR / "best_model.pth"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = RESULTS_DIR / "training_history.csv"

with open(
    history_path, "w", newline="", encoding="utf-8"
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

if best_state is None:
    raise RuntimeError("Training did not produce a checkpoint.")

# ============================================================
# EVALUATE THE BEST CHECKPOINT
# ============================================================

model.load_state_dict(best_state)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []
all_filenames = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

# ImageFolder orders samples consistently with val_dataset.samples.
all_filenames = [
    Path(path).name for path, _ in val_dataset.samples
]

accuracy = accuracy_score(
    all_labels, all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

print("\n" + "=" * 55)
print("EXPERIMENT 5 RESULTS")
print("=" * 55)
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best-checkpoint validation accuracy: {accuracy:.4f}")
print(f"Best-checkpoint validation accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(matrix)

# ============================================================
# SAVE RESULTS
# ============================================================

report_path = RESULTS_DIR / "classification_report.txt"

with open(
    report_path, "w", encoding="utf-8"
) as file:
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Best validation loss: {best_val_loss:.4f}\n")
    file.write(f"Validation accuracy: {accuracy:.4f}\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(np.array2string(matrix))

predictions_path = RESULTS_DIR / "validation_predictions.csv"

with open(
    predictions_path, "w", newline="", encoding="utf-8"
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

    for i, (actual, predicted, probabilities) in enumerate(
        zip(all_labels, all_predictions, all_probabilities)
    ):
        writer.writerow([
            all_filenames[i],
            class_names[actual],
            class_names[predicted],
            float(probabilities[0]),
            float(probabilities[1]),
            actual == predicted
        ])

summary = {
    "experiment": 5,
    "model": "ResNet50 pretrained on ImageNet",
    "backbone_frozen": True,
    "trainable_layer": "fc",
    "class_mapping": train_dataset.class_to_idx,
    "training_files": len(train_dataset),
    "validation_files": len(val_dataset),
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "validation_accuracy": accuracy,
    "confusion_matrix": matrix.tolist()
}

with open(
    RESULTS_DIR / "summary.json", "w", encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nSaved results to:")
print(RESULTS_DIR)
print("\nFiles created:")
print("- best_model.pth")
print("- training_history.csv")
print("- classification_report.txt")
print("- validation_predictions.csv")
print("- summary.json")

print("\nExperiment 5 completed.")

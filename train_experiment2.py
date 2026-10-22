
from pathlib import Path
import csv
import random
import copy

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
RESULTS_DIR = PROJECT_DIR / "results" / "experiment2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 5

EXPECTED_CLASSES = ["adenoma", "hyperplastic"]

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
print("EXPERIMENT 2: RESNET50 FINE-TUNING")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 3. IMAGE TRANSFORMS
# ============================================================

# Augmentation is applied only to training images.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation images remain unaugmented.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# 4. LOAD DATASET
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

if not train_dir.is_dir() or not val_dir.is_dir():
    raise FileNotFoundError(
        "Expected data_split/train and data_split/val. "
        "Check your dataset folders."
    )

train_dataset = datasets.ImageFolder(
    root=str(train_dir),
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    root=str(val_dir),
    transform=val_transform,
)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings differ.")

if train_dataset.classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Unexpected class order: {train_dataset.classes}. "
        f"Expected: {EXPECTED_CLASSES}"
    )

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")

print("\nClasses:", train_dataset.class_to_idx)
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
# 5. LOAD PRETRAINED RESNET50
# ============================================================

print("\nLoading pretrained ResNet50...")

weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Replace the final layer for our two classes.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(EXPECTED_CLASSES))

# All layers are trainable in this experiment.
for parameter in model.parameters():
    parameter.requires_grad = True

model = model.to(device)

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
# 6. TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def run_epoch(loader, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

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

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


# ============================================================
# 7. TRAIN WITH EARLY STOPPING
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(train_loader, training=True)
    val_loss, val_accuracy = run_epoch(val_loader, training=False)

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
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"LR: {current_lr:.6f} | "
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

        checkpoint_path = RESULTS_DIR / "best_resnet50_experiment2.pth"

        torch.save({
            "model_state_dict": best_state,
            "classes": EXPECTED_CLASSES,
            "class_to_idx": train_dataset.class_to_idx,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "image_size": IMAGE_SIZE,
        }, checkpoint_path)

        print("  -> Best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch was {best_epoch}."
        )
        break

# Save the training history without overwriting Experiment 1.
history_path = RESULTS_DIR / "training_history.csv"

with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# ============================================================
# 8. EVALUATE THE BEST CHECKPOINT
# ============================================================

if best_state is None:
    raise RuntimeError("Training did not produce a best checkpoint.")

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
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

final_accuracy = accuracy_score(all_labels, all_predictions)

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=EXPECTED_CLASSES,
    zero_division=0,
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1],
)

report_path = RESULTS_DIR / "evaluation_report.txt"

with report_path.open("w", encoding="utf-8") as file:
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Best validation loss: {best_val_loss:.4f}\n")
    file.write(f"Validation accuracy: {final_accuracy:.4f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\n\nConfusion matrix\n")
    file.write("Rows = actual classes; columns = predicted classes\n")
    file.write(f"Class order: {EXPECTED_CLASSES}\n")
    file.write(str(matrix))
    file.write("\n")

predictions_path = RESULTS_DIR / "validation_predictions.csv"

with predictions_path.open(
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
        zip(all_labels, all_predictions, all_probabilities)
    ):
        image_path, _ = val_dataset.samples[index]

        writer.writerow([
            Path(image_path).name,
            EXPECTED_CLASSES[actual],
            EXPECTED_CLASSES[predicted],
            f"{probs[0]:.6f}",
            f"{probs[1]:.6f}",
            actual == predicted,
        ])

print("\n" + "=" * 60)
print("EXPERIMENT 2 COMPLETE")
print("=" * 60)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy: {final_accuracy:.4f}")
print("\nClassification report:")
print(report)
print("\nConfusion matrix:")
print(matrix)
print("\nResults saved to:", RESULTS_DIR)
print("- best_resnet50_experiment2.pth")
print("- training_history.csv")
print("- evaluation_report.txt")
print("- validation_predictions.csv")

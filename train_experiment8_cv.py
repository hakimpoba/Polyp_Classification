
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "dataset_v2_manifest_with_splits.csv"
RESULTS_DIR = ROOT / "results_v2_cv"

N_SPLITS = 5
EPOCHS = 20
PATIENCE = 5
IMAGE_SIZE = 224
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
SEED = 42

CLASS_NAMES = ["adenoma", "hyperplastic"]
CLASS_TO_IDX = {"adenoma": 0, "hyperplastic": 1}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 65)
print("EXPERIMENT 8: 5-FOLD PATIENT-GROUPED CROSS-VALIDATION")
print("=" * 65)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 3. LOAD THE MANIFEST
# ============================================================

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "relative_path",
    "class_label",
    "patient_group_id",
    "split",
]

missing_columns = [
    col for col in required_columns
    if col not in manifest.columns
]

if missing_columns:
    raise ValueError(
        f"Manifest is missing columns: {missing_columns}"
    )

if manifest["patient_group_id"].isna().any():
    raise ValueError("Some images have missing patient group IDs.")

if manifest["relative_path"].duplicated().any():
    raise ValueError("Duplicate relative paths found in manifest.")

if not set(manifest["class_label"].unique()).issubset(CLASS_TO_IDX):
    raise ValueError("Unexpected class label in manifest.")

# Use only the previous training and validation data.
# The previous test set is deliberately excluded.
development = manifest[
    manifest["split"].isin(["train", "val"])
].copy().reset_index(drop=True)

excluded_test = manifest[
    manifest["split"] == "test"
].copy()

if len(development) == 0:
    raise ValueError("No development images were found.")

if len(excluded_test) == 0:
    raise ValueError(
        "No previously held-out test rows were found. "
        "Check the manifest before proceeding."
    )

# Confirm all image files exist.
missing_files = []

for relative_path in development["relative_path"]:
    path = ROOT / str(relative_path)

    if not path.is_file():
        missing_files.append(str(path))

if missing_files:
    raise FileNotFoundError(
        "Some development images are missing:\n"
        + "\n".join(missing_files[:10])
    )

print(f"\nDevelopment images: {len(development)}")
print(
    "Development patient groups:",
    development["patient_group_id"].nunique()
)
print(f"Excluded test images: {len(excluded_test)}")

print("\nDevelopment class counts:")
print(development["class_label"].value_counts().to_string())

print("\nTest set remains excluded from cross-validation.")


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

# Augmentation is applied only to training images.
# Validation images use deterministic preprocessing.

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
# 5. CUSTOM DATASET
# ============================================================

class PolypDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = ROOT / str(row["relative_path"])
        label = CLASS_TO_IDX[row["class_label"]]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 6. MODEL AND EVALUATION FUNCTIONS
# ============================================================

def create_model():
    # Pretrained weights are used for every fresh fold model.
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    true_labels = []
    predictions = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)

        true_labels.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(true_labels, predictions),
    )


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    true_labels = []
    predictions = []
    probabilities = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        true_labels.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        probabilities.extend(probs[:, 1].cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            true_labels, predictions
        ),
        "macro_precision": precision_score(
            true_labels, predictions,
            average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            true_labels, predictions,
            average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            true_labels, predictions,
            average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            true_labels, predictions,
            average="weighted", zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(true_labels, probabilities)
            if len(set(true_labels)) == 2
            else float("nan")
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=[0, 1]
        ),
    }


# ============================================================
# 7. CREATE PATIENT-GROUPED FOLDS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

X = development["relative_path"].to_numpy()
y = development["class_label"].map(CLASS_TO_IDX).to_numpy()
groups = development["patient_group_id"].to_numpy()

fold_results = []
history_rows = []

# Each row belongs to one validation fold exactly once.
validation_appearances = np.zeros(len(development), dtype=int)

# ============================================================
# 8. TRAIN EACH FOLD
# ============================================================

for fold_number, (train_indices, val_indices) in enumerate(
    cv.split(X, y, groups), start=1
):
    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 65)

    train_df = development.iloc[train_indices].copy()
    val_df = development.iloc[val_indices].copy()

    train_groups = set(train_df["patient_group_id"])
    val_groups = set(val_df["patient_group_id"])

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected in fold {fold_number}: "
            f"{overlap}"
        )

    validation_appearances[val_indices] += 1

    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Training patient groups: {len(train_groups)}")
    print(f"Validation patient groups: {len(val_groups)}")

    print("\nTraining class counts:")
    print(train_df["class_label"].value_counts().to_string())

    print("\nValidation class counts:")
    print(val_df["class_label"].value_counts().to_string())

    train_dataset = PolypDataset(
        train_df, transform=train_transform
    )
    val_dataset = PolypDataset(
        val_df, transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(SEED + fold_number)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    set_seed(SEED + fold_number)

    model = create_model()
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

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_metrics = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append({
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
        })

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping for this fold.")
            break

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint found for fold {fold_number}."
        )

    # Evaluate this fold using its best validation-loss checkpoint.
    model.load_state_dict(best_state)

    final_metrics = validate(model, val_loader, criterion)

    result = {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "train_images": len(train_df),
        "val_images": len(val_df),
        "train_patient_groups": len(train_groups),
        "val_patient_groups": len(val_groups),
        "best_val_loss": best_val_loss,
        "accuracy": final_metrics["accuracy"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "macro_precision": final_metrics["macro_precision"],
        "macro_recall": final_metrics["macro_recall"],
        "macro_f1": final_metrics["macro_f1"],
        "weighted_f1": final_metrics["weighted_f1"],
        "roc_auc": final_metrics["roc_auc"],
        "tn": int(final_metrics["confusion_matrix"][0, 0]),
        "fp": int(final_metrics["confusion_matrix"][0, 1]),
        "fn": int(final_metrics["confusion_matrix"][1, 0]),
        "tp": int(final_metrics["confusion_matrix"][1, 1]),
    }

    fold_results.append(result)

    print(f"\nFold {fold_number} best epoch: {best_epoch}")
    print(f"Fold {fold_number} accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Fold {fold_number} macro F1: {result['macro_f1']:.4f}")

    # Free GPU memory before creating the next fold's model.
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 9. VERIFY CROSS-VALIDATION COVERAGE
# ============================================================

if not np.all(validation_appearances == 1):
    raise RuntimeError(
        "Each development image must appear in validation "
        "exactly once across the five folds."
    )

print("\nVerified: each development image was validated once.")
print("Verified: no patient group crosses train/validation boundaries.")


# ============================================================
# 10. SAVE FOLD RESULTS AND SUMMARY
# ============================================================

results_df = pd.DataFrame(fold_results)
history_df = pd.DataFrame(history_rows)

results_df.to_csv(
    RESULTS_DIR / "experiment8_cv_fold_results.csv",
    index=False,
)

history_df.to_csv(
    RESULTS_DIR / "experiment8_cv_training_history.csv",
    index=False,
)

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
]

summary_rows = []

for metric in metric_columns:
    values = results_df[metric].dropna()

    summary_rows.append({
        "metric": metric,
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "folds": len(values),
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULTS_DIR / "experiment8_cv_summary.csv",
    index=False,
)


# ============================================================
# 11. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT 8: CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(
    results_df[
        ["fold", "best_epoch", "accuracy", "balanced_accuracy",
         "macro_f1", "roc_auc"]
    ].to_string(index=False)
)

print("\nMean and standard deviation across folds:")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nResults saved in:")
print(RESULTS_DIR)
print("\nThe previously held-out test set was not used.")

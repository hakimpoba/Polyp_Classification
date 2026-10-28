
from pathlib import Path

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
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

TEST_DIR = ROOT / "data_split_v2" / "test"
CHECKPOINT_PATH = (
    ROOT / "results_v2" / "best_resnet50_v2.pth"
)
RESULTS_DIR = ROOT / "results_v2"

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["adenoma", "hyperplastic"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 7: FINAL TEST EVALUATION")
print("=" * 60)
print(f"Device: {DEVICE}")


# ============================================================
# 2. CHECK INPUTS
# ============================================================

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


# ============================================================
# 3. LOAD TEST IMAGES
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform,
)

if test_dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"Unexpected test class mapping: "
        f"{test_dataset.class_to_idx}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(f"Test images: {len(test_dataset)}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# ============================================================
# 4. LOAD THE SELECTED CHECKPOINT
# ============================================================

# This checkpoint contains weights created by our own
# training script. weights_only=False is needed to load
# its complete checkpoint dictionary in newer PyTorch versions.

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
except TypeError:
    # Compatibility with older PyTorch versions.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

if checkpoint["class_to_idx"] != test_dataset.class_to_idx:
    raise ValueError("Checkpoint and test class mappings differ.")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print(f"Selected checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")


# ============================================================
# 5. PREDICT TEST IMAGES
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

all_probabilities = pd.DataFrame(
    all_probabilities,
    columns=[
        f"probability_{name}"
        for name in CLASS_NAMES
    ],
)

true_labels = [CLASS_NAMES[i] for i in all_labels]
predicted_labels = [
    CLASS_NAMES[i] for i in all_predictions
]


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(all_labels, all_predictions)

balanced_accuracy = balanced_accuracy_score(
    all_labels, all_predictions
)

precision_macro = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

f1_weighted = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1],
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4,
)

# ROC-AUC requires both classes to be present in the test set.
if len(set(all_labels)) == 2:
    auc = roc_auc_score(
        all_labels,
        all_probabilities["probability_hyperplastic"].tolist(),
    )
else:
    auc = None


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro precision:   {precision_macro:.4f}")
print(f"Macro recall:      {recall_macro:.4f}")
print(f"Macro F1-score:    {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

if auc is not None:
    print(f"ROC-AUC:           {auc:.4f}")
else:
    print("ROC-AUC: unavailable; test set contains only one class.")

print("\nClassification report:")
print(report)

print("Confusion matrix")
print("Rows = actual class; columns = predicted class")
print("Class order: adenoma, hyperplastic")
print(cm)


# ============================================================
# 8. SAVE REPORTS AND IMAGE-LEVEL PREDICTIONS
# ============================================================

metrics = {
    "checkpoint_epoch": checkpoint["epoch"],
    "best_validation_loss": checkpoint["best_val_loss"],
    "test_images": len(test_dataset),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_precision": precision_macro,
    "macro_recall": recall_macro,
    "macro_f1": f1_macro,
    "weighted_f1": f1_weighted,
    "roc_auc": auc,
}

pd.DataFrame([metrics]).to_csv(
    RESULTS_DIR / "experiment7_test_metrics.csv",
    index=False,
)

pd.DataFrame(
    cm,
    index=["actual_adenoma", "actual_hyperplastic"],
    columns=["predicted_adenoma", "predicted_hyperplastic"],
).to_csv(
    RESULTS_DIR / "experiment7_test_confusion_matrix.csv"
)

report_path = RESULTS_DIR / "experiment7_test_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("EXPERIMENT 7: TEST EVALUATION\n\n")
    file.write(f"Checkpoint epoch: {checkpoint['epoch']}\n")
    file.write(f"Test images: {len(test_dataset)}\n\n")

    for name, value in metrics.items():
        file.write(f"{name}: {value}\n")

    file.write("\nClassification report:\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(cm))

prediction_rows = []

for (image_path, _), true_label, predicted_label in zip(
    test_dataset.samples,
    true_labels,
    predicted_labels,
):
    prediction_rows.append({
        "filename": Path(image_path).name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": true_label == predicted_label,
    })

predictions_df = pd.DataFrame(prediction_rows)

probabilities_df = all_probabilities.reset_index(drop=True)
predictions_df = pd.concat(
    [predictions_df, probabilities_df],
    axis=1,
)

predictions_df.to_csv(
    RESULTS_DIR / "experiment7_test_predictions.csv",
    index=False,
)

print("\nFiles saved:")
print(RESULTS_DIR / "experiment7_test_metrics.csv")
print(RESULTS_DIR / "experiment7_test_confusion_matrix.csv")
print(RESULTS_DIR / "experiment7_test_report.txt")
print(RESULTS_DIR / "experiment7_test_predictions.csv")

print("\nTest evaluation completed.")
print("Do not use these test results to tune this same model.")

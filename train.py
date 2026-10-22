
from pathlib import Path
import csv
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

DATA_DIR = PROJECT_DIR / "data_split"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("POLYP CLASSIFICATION: RESNET50 BASELINE")
print("=" * 55)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# 2. Check dataset folders
# --------------------------------------------------

for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

weights = ResNet50_Weights.DEFAULT

# Training: mild augmentation only
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation: standard preprocessing for pretrained weights
val_transform = weights.transforms()

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform,
)

print("\nClass mapping:", train_dataset.class_to_idx)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Training and validation class mappings differ.")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("\nTraining class counts:")
for class_name, count in sorted(
    train_dataset.targets.count(i), key=lambda x: x
) if False else []:
    print(class_name, count)

for class_name in train_dataset.classes:
    count = sum(
        1 for _, label in train_dataset.samples
        if train_dataset.classes[label] == class_name
    )
    print(f"  {class_name}: {count}")

print("\nValidation class counts:")
for class_name in val_dataset.classes:
    count = sum(
        1 for _, label in val_dataset.samples
        if val_dataset.classes[label] == class_name
    )
    print(f"  {class_name}: {count}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# 4. Load pretrained ResNet50
# --------------------------------------------------

print("\nLoading pretrained ResNet50...")
print("The first run may download ImageNet weights.")

model = resnet50(weights=weights)

# Freeze the pretrained feature-extraction layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the original 1000-class output with 2 classes
input_features = model.fc.in_features
model.fc = nn.Linear(input_features, len(train_dataset.classes))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)

print("Model ready.")
print("Trainable parameters: classification layer only.")

# --------------------------------------------------
# 5. Training and validation functions
# --------------------------------------------------

def run_training_epoch():
    model.train()

    # Keep frozen backbone in evaluation mode
    model.eval()
    model.fc.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # No gradients are needed for the frozen backbone
        with torch.no_grad():
            features = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(
                                model.maxpool(
                                    model.relu(model.bn1(model.conv1(images)))
                                )
                            )
                        )
                    )
                )
            )
            features = torch.flatten(features, 1)

        outputs = model.fc(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate():
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(val_dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy, all_labels, all_predictions


# --------------------------------------------------
# 6. Train for 10 epochs
# --------------------------------------------------

history = []
best_val_loss = float("inf")
best_state = None

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    train_loss, train_accuracy = run_training_epoch()

    (
        val_loss,
        val_accuracy,
        _,
        _,
    ) = evaluate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "class_to_idx": train_dataset.class_to_idx,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            },
            RESULTS_DIR / "best_resnet50.pth",
        )

# Save epoch history
with (RESULTS_DIR / "training_history.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# --------------------------------------------------
# 7. Evaluate the best checkpoint
# --------------------------------------------------

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_state)

(
    final_val_loss,
    final_val_accuracy,
    true_labels,
    predicted_labels,
) = evaluate()

class_names = train_dataset.classes
label_ids = list(range(len(class_names)))

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=label_ids,
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=label_ids,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 55)
print("FINAL VALIDATION RESULTS")
print("=" * 55)
print(f"Best checkpoint validation loss: {final_val_loss:.4f}")
print(f"Validation accuracy: {final_val_accuracy:.4f}")
print(f"Validation accuracy (%): {final_val_accuracy * 100:.2f}%")

print("\nClassification report:")
print(report)

print("Confusion matrix")
print("Rows = actual classes; columns = predicted classes")
print("Class order:", class_names)
print(cm)

with (RESULTS_DIR / "evaluation_report.txt").open(
    "w", encoding="utf-8"
) as file:
    file.write("ResNet50 baseline evaluation\n")
    file.write(f"Validation loss: {final_val_loss:.4f}\n")
    file.write(f"Validation accuracy: {final_val_accuracy:.4f}\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(cm))
    file.write("\n")

print("\nFiles saved in:", RESULTS_DIR)
print(" - best_resnet50.pth")
print(" - training_history.csv")
print(" - evaluation_report.txt")
print("\nExperiment completed.")

from pathlib import Path
import copy, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "dataset_v2_manifest_with_splits.csv"
OUT = ROOT / "results_v2_aug_cv"  # Separate from Experiment 10 output
OUT.mkdir(parents=True, exist_ok=True)

N_SPLITS, MAX_EPOCHS, WARMUP_EPOCHS, PATIENCE = 5, 20, 3, 5
IMAGE_SIZE, BATCH_SIZE, SEED = 224, 8, 42
HEAD_LR, LAYER4_LR, WEIGHT_DECAY = 1e-4, 1e-5, 1e-4
CLASS_NAMES = ["adenoma", "hyperplastic"]
CLASS_TO_IDX = {"adenoma": 0, "hyperplastic": 1}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolypDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(ROOT / str(row["relative_path"])) as im:
            image = im.convert("RGB")
        return self.transform(image), CLASS_TO_IDX[row["class_label"]]


def create_model():
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except (AttributeError, TypeError):
        model = models.resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def unfreeze_layer4(model):
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def build_optimizer(model):
    return torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": LAYER4_LR},
        {"params": model.fc.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for name in ["conv1", "bn1", "layer1", "layer2", "layer3"]:
        getattr(model, name).eval()
    if not any(p.requires_grad for p in model.layer4.parameters()):
        model.layer4.eval()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        n = y.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == y).sum().item()
        total += n
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, ys, preds, probs = 0, [], [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        loss = criterion(logits, y.to(DEVICE))
        loss_sum += loss.item() * len(y)
        ys.extend(y.numpy().tolist())
        preds.extend(logits.argmax(1).cpu().numpy().tolist())
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_precision": precision_score(ys, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(ys, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(ys, probs) if len(set(ys)) == 2 else float("nan"),
        "cm": cm,
    }


def main():
    set_seed(SEED)
    print("=" * 68)
    print("EXPERIMENT 11: DYNAMIC AUGMENTATION + PARTIAL RESNET50 FINE-TUNING")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    manifest = pd.read_csv(MANIFEST)
    required = ["relative_path", "class_label", "patient_group_id", "split"]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if manifest["patient_group_id"].isna().any():
        raise ValueError("Missing patient group IDs.")
    if manifest["relative_path"].duplicated().any():
        raise ValueError("Duplicate image paths.")
    if not set(manifest["class_label"].unique()).issubset(CLASS_TO_IDX):
        raise ValueError("Unexpected class label.")

    # Same development rows as Experiment 10; original test split excluded.
    dev = manifest[manifest["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    test = manifest[manifest["split"] == "test"]
    if dev.empty or test.empty:
        raise ValueError("Expected development train/val rows and excluded test rows.")
    missing_files = [str(ROOT / str(p)) for p in dev["relative_path"] if not (ROOT / str(p)).is_file()]
    if missing_files:
        raise FileNotFoundError("Missing images:\n" + "\n".join(missing_files[:10]))
    print(f"Development images: {len(dev)} | patient groups: {dev['patient_group_id'].nunique()}")
    print(f"Excluded test images: {len(test)}")
    print("Class counts:\n", dev["class_label"].value_counts().to_string())

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Experiment 11: mild brightness/contrast/saturation jitter during training only.
    # Hue is unchanged to avoid unnecessary shifts in clinically relevant colours.
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.90, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y = dev["class_label"].map(CLASS_TO_IDX).to_numpy()
    groups = dev["patient_group_id"].to_numpy()
    x = dev["relative_path"].to_numpy()
    fold_rows, history_rows = [], []
    validation_count = np.zeros(len(dev), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x, y, groups), 1):
        print("\n" + "=" * 68 + f"\nFOLD {fold}/{N_SPLITS}\n" + "=" * 68)
        tr, va = dev.iloc[tr_idx].copy(), dev.iloc[va_idx].copy()
        tr_groups, va_groups = set(tr.patient_group_id), set(va.patient_group_id)
        if tr_groups & va_groups:
            raise RuntimeError(f"Patient leakage in fold {fold}")
        validation_count[va_idx] += 1
        print(f"Train: {len(tr)} images / {len(tr_groups)} patients; validation: {len(va)} images / {len(va_groups)} patients")

        train_loader = DataLoader(
            PolypDataset(tr, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(SEED + fold))
        val_loader = DataLoader(
            PolypDataset(va, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available())

        model = create_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        scheduler = None
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            if epoch == WARMUP_EPOCHS + 1:
                print("Unfreezing layer4; earlier backbone layers remain frozen.")
                unfreeze_layer4(model)
                optimizer = build_optimizer(model)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2)
                best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0

            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_m = evaluate(model, val_loader, criterion)
            if scheduler is not None:
                scheduler.step(val_m["loss"])
            stage = "head_warmup" if epoch <= WARMUP_EPOCHS else "partial_finetune"
            lrs = [g["lr"] for g in optimizer.param_groups]
            history_rows.append({
                "fold": fold, "epoch": epoch, "stage": stage,
                "train_loss": tr_loss, "train_accuracy": tr_acc,
                "val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"],
                "val_balanced_accuracy": val_m["balanced_accuracy"],
                "layer4_lr": lrs[0] if stage == "partial_finetune" else 0,
                "head_lr": lrs[-1],
            })
            print(f"Epoch {epoch:02d}/{MAX_EPOCHS} [{stage}] | train loss {tr_loss:.4f} acc {tr_acc:.2%} | val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.2%}")

            if epoch <= WARMUP_EPOCHS:
                continue
            if val_m["loss"] < best_loss:
                best_loss, best_epoch = val_m["loss"], epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                print("Early stopping.")
                break

        if best_state is None:
            raise RuntimeError(f"No best checkpoint for fold {fold}")
        model.load_state_dict(best_state)
        final = evaluate(model, val_loader, criterion)
        cm = final["cm"]
        row = {
            "fold": fold, "best_epoch": best_epoch, "train_images": len(tr), "val_images": len(va),
            "train_patient_groups": len(tr_groups), "val_patient_groups": len(va_groups),
            "best_val_loss": best_loss,
            **{k: final[k] for k in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc"]},
            "actual_adenoma_pred_adenoma": int(cm[0, 0]),
            "actual_adenoma_pred_hyperplastic": int(cm[0, 1]),
            "actual_hyperplastic_pred_adenoma": int(cm[1, 0]),
            "actual_hyperplastic_pred_hyperplastic": int(cm[1, 1]),
        }
        fold_rows.append(row)
        print(f"Best epoch {best_epoch}; accuracy {row['accuracy']:.4f}; macro F1 {row['macro_f1']:.4f}; ROC-AUC {row['roc_auc']:.4f}")
        print("Confusion matrix (rows=actual, columns=predicted; adenoma, hyperplastic):\n", cm)
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not np.all(validation_count == 1):
        raise RuntimeError("Each development image must be validated exactly once.")
    results, history = pd.DataFrame(fold_rows), pd.DataFrame(history_rows)
    results.to_csv(OUT / "experiment11_cv_fold_results.csv", index=False)
    history.to_csv(OUT / "experiment11_cv_training_history.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc"]
    summary = pd.DataFrame([{
        "metric": m, "mean": results[m].dropna().mean(),
        "std": results[m].dropna().std(ddof=1), "folds": results[m].notna().sum()
    } for m in metrics])
    summary.to_csv(OUT / "experiment11_cv_summary.csv", index=False)
    print("\nEXPERIMENT 11 SUMMARY")
    print(results[["fold", "best_epoch", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]].to_string(index=False))
    print("\nMean ± SD across folds:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nVerified: each development image validated once; no patient group crosses fold boundaries.")
    print("Previously evaluated test split was excluded from this CV.")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()

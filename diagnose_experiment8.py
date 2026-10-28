
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST = Path("dataset_v2_manifest_with_splits.csv")
RESULTS = Path("results_v2_cv/experiment8_cv_fold_results.csv")

df = pd.read_csv(MANIFEST)

# Match Experiment 8: use development data only.
dev = df[df["split"].isin(["train", "val"])].copy()
excluded_test = df[df["split"] == "test"].copy()

# Match the class mapping used in training.
class_map = {"adenoma": 0, "hyperplastic": 1}
dev["label"] = dev["class_label"].map(class_map)

if dev["label"].isna().any():
    raise ValueError("Unexpected class label found in the manifest.")

# Reconstruct the same five patient-grouped folds.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(
    cv.split(
        X=dev,
        y=dev["label"],
        groups=dev["patient_group_id"]
    )
)

results = pd.read_csv(RESULTS)

print(f"Development images: {len(dev)}")
print(f"Development patients: {dev['patient_group_id'].nunique()}")
print(f"Excluded test images: {len(excluded_test)}")

for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
    val = dev.iloc[val_idx].copy()
    train = dev.iloc[train_idx].copy()

    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}")
    print("=" * 65)

    print("\nValidation class counts:")
    print(val["class_label"].value_counts().to_string())
    print("\nTRAINING class by source:")
    print(
    pd.crosstab(
        train["source_dataset"].fillna("missing"),
        train["class_label"]
    ).to_string()
    )

    print("\nVALIDATION class by source:")
    print(
    pd.crosstab(
        val["source_dataset"].fillna("missing"),
        val["class_label"]
    ).to_string()
    )

    print("\nValidation source counts:")
    print(val["source_dataset"].fillna("missing").value_counts().to_string())

    print("\nValidation modality counts:")
    print(val["modality"].fillna("missing").value_counts().to_string())

    print("\nValidation class by source:")
    print(
        pd.crosstab(
            val["source_dataset"].fillna("missing"),
            val["class_label"]
        ).to_string()
    )

    print("\nValidation class by modality:")
    print(
        pd.crosstab(
            val["modality"].fillna("missing"),
            val["class_label"]
        ).to_string()
    )

    overlap = set(train["patient_group_id"]) & set(val["patient_group_id"])
    print(f"\nPatient leakage check: {len(overlap)} overlapping groups")

    row = results.loc[results["fold"] == fold_number]
    if not row.empty:
        print("\nRecorded model metrics:")
        print(row.to_string(index=False))

    

from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "dataset_v2_manifest.csv"
OUTPUT_MANIFEST = ROOT / "dataset_v2_manifest_with_splits.csv"
OUTPUT_DIR = ROOT / "data_split_v2"

SEED = 42

# Desired proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Candidate search settings
TEST_CANDIDATES = 200
VAL_CANDIDATES_PER_TEST = 60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_split_counts(df, indices):
    """Return image counts by class for a proposed split."""
    subset = df.iloc[list(indices)]

    return {
        "images": len(subset),
        "adenoma": int((subset["class_label"] == "adenoma").sum()),
        "hyperplastic": int(
            (subset["class_label"] == "hyperplastic").sum()
        ),
        "patients": subset["patient_group_id"].nunique(),
    }


def calculate_score(df, train_idx, val_idx, test_idx):
    """
    Lower scores indicate splits closer to the requested
    image proportions and class distributions.
    """
    targets = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    partitions = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    total_images = len(df)
    class_totals = df["class_label"].value_counts().to_dict()

    score = 0.0

    for split_name, indices in partitions.items():
        actual = get_split_counts(df, indices)
        ratio = targets[split_name]

        # Match the overall number of images.
        expected_images = total_images * ratio
        score += (
            (actual["images"] - expected_images)
            / max(expected_images, 1)
        ) ** 2

        # Match the number of images in each class.
        for class_name in ["adenoma", "hyperplastic"]:
            expected_class = class_totals.get(class_name, 0) * ratio

            score += (
                (actual[class_name] - expected_class)
                / max(expected_class, 1)
            ) ** 2

    return score


# ============================================================
# 1. CHECK THE INPUT MANIFEST
# ============================================================

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Cannot find the manifest:\n{MANIFEST_PATH}\n"
        "Run prepare_dataset_v2.py first."
    )

df = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "image_id",
    "relative_path",
    "class_label",
    "patient_group_id",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Manifest is missing columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("The manifest contains no images.")

if df["image_id"].duplicated().any():
    duplicates = df.loc[
        df["image_id"].duplicated(), "image_id"
    ].tolist()
    raise ValueError(f"Duplicate image IDs found: {duplicates}")

if df["patient_group_id"].isna().any():
    raise ValueError("Some images have missing patient group IDs.")

allowed_classes = {"adenoma", "hyperplastic"}

if set(df["class_label"].unique()) != allowed_classes:
    raise ValueError(
        "Expected exactly these classes: "
        "'adenoma' and 'hyperplastic'."
    )

# Confirm that every source image exists before splitting.
missing_files = []

for relative_path in df["relative_path"]:
    image_path = ROOT / str(relative_path)

    if not image_path.is_file():
        missing_files.append(str(image_path))

if missing_files:
    preview = "\n".join(missing_files[:10])
    raise FileNotFoundError(
        f"{len(missing_files)} source image(s) are missing.\n"
        f"First missing paths:\n{preview}"
    )

print("=" * 60)
print("DATASET V2 PATIENT-LEVEL SPLITTING")
print("=" * 60)
print(f"Total images: {len(df)}")
print(f"Distinct patient groups: {df['patient_group_id'].nunique()}")
print("\nOriginal class counts:")
print(df["class_label"].value_counts().to_string())


# ============================================================
# 2. PREVENT ACCIDENTAL OVERWRITING
# ============================================================

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*")):
    raise FileExistsError(
        f"\nOutput folder already contains files:\n{OUTPUT_DIR}\n"
        "Rename or move that folder before running this script again. "
        "Do not delete it until you have checked its contents."
    )


# ============================================================
# 3. SEARCH FOR A GOOD GROUP-AWARE SPLIT
# ============================================================

# GroupShuffleSplit keeps every patient's images together.
# We search multiple candidates because patients may have
# different numbers of images.

groups = df["patient_group_id"].to_numpy()

test_splitter = GroupShuffleSplit(
    n_splits=TEST_CANDIDATES,
    test_size=TEST_RATIO,
    random_state=SEED,
)

best_score = float("inf")
best_indices = None

for trainval_idx, test_idx in test_splitter.split(
    df, groups=groups
):
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    trainval_groups = trainval_df["patient_group_id"].to_numpy()

    # Validation should be about 15% of the entire dataset.
    # Within the remaining 85%, this is approximately 17.65%.
    val_fraction_of_remaining = VAL_RATIO / (
        TRAIN_RATIO + VAL_RATIO
    )

    val_splitter = GroupShuffleSplit(
        n_splits=VAL_CANDIDATES_PER_TEST,
        test_size=val_fraction_of_remaining,
        random_state=SEED + 1,
    )

    for train_local, val_local in val_splitter.split(
        trainval_df, groups=trainval_groups
    ):
        train_idx = trainval_idx[train_local]
        val_idx = trainval_idx[val_local]

        score = calculate_score(
            df, train_idx, val_idx, test_idx
        )

        if score < best_score:
            best_score = score
            best_indices = (
                train_idx.copy(),
                val_idx.copy(),
                test_idx.copy(),
            )

if best_indices is None:
    raise RuntimeError("Could not create a dataset split.")

train_idx, val_idx, test_idx = best_indices


# ============================================================
# 4. VERIFY THAT PATIENT GROUPS DO NOT OVERLAP
# ============================================================

train_groups = set(
    df.iloc[train_idx]["patient_group_id"]
)
val_groups = set(
    df.iloc[val_idx]["patient_group_id"]
)
test_groups = set(
    df.iloc[test_idx]["patient_group_id"]
)

if train_groups & val_groups:
    raise RuntimeError("Patient leakage between train and validation.")

if train_groups & test_groups:
    raise RuntimeError("Patient leakage between train and test.")

if val_groups & test_groups:
    raise RuntimeError("Patient leakage between validation and test.")

if len(train_idx) + len(val_idx) + len(test_idx) != len(df):
    raise RuntimeError("Some images were omitted from the split.")

split_labels = pd.Series(index=df.index, dtype="object")
split_labels.iloc[train_idx] = "train"
split_labels.iloc[val_idx] = "val"
split_labels.iloc[test_idx] = "test"

df["split"] = split_labels

if df["split"].isna().any():
    raise RuntimeError("Some images do not have a split assignment.")


# ============================================================
# 5. PRINT AND SAVE THE SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

for split_name, indices in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx),
]:
    counts = get_split_counts(df, indices)

    print(f"\n{split_name}")
    print(f"  Images:      {counts['images']}")
    print(f"  Adenoma:     {counts['adenoma']}")
    print(f"  Hyperplastic:{counts['hyperplastic']}")
    print(f"  Patient groups: {counts['patients']}")

    percentage = counts["images"] / len(df) * 100
    print(f"  Dataset share: {percentage:.2f}%")

print("\nPatient-group overlap check:")
print(f"  Train vs validation: {len(train_groups & val_groups)}")
print(f"  Train vs test:       {len(train_groups & test_groups)}")
print(f"  Validation vs test:  {len(val_groups & test_groups)}")

print(f"\nSplit search score: {best_score:.6f}")


# ============================================================
# 6. COPY IMAGES INTO THE NEW FOLDERS
# ============================================================

# The source images are never moved or modified.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    for _, row in df.iterrows():
        split_name = row["split"]
        class_name = row["class_label"]

        source = ROOT / str(row["relative_path"])
        destination_folder = OUTPUT_DIR / split_name / class_name
        destination_folder.mkdir(parents=True, exist_ok=True)

        destination = destination_folder / str(row["image_id"])

        if destination.exists():
            raise FileExistsError(
                f"Destination image already exists: {destination}"
            )

        shutil.copy2(source, destination)

    # Save the private manifest with the new split assignments.
    df.to_csv(OUTPUT_MANIFEST, index=False)

except Exception:
    print(
        "\nAn error occurred during copying. "
        "Some output files may have been created."
    )
    print(
        "Inspect the output folder before trying again; "
        "the script will not overwrite a non-empty output folder."
    )
    raise


# ============================================================
# 7. FINAL VERIFICATION
# ============================================================

copied_count = sum(
    1
    for path in OUTPUT_DIR.rglob("*")
    if path.is_file()
)

if copied_count != len(df):
    raise RuntimeError(
        f"Expected {len(df)} copied images, found {copied_count}."
    )

print("\n" + "=" * 60)
print("DATASET V2 SPLIT COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Original images preserved: YES")
print(f"Images assigned: {len(df)}")
print(f"Images copied: {copied_count}")
print(f"Split folders: {OUTPUT_DIR}")
print(f"Private split manifest: {OUTPUT_MANIFEST}")
print("\nNext: review the counts above before training a model.")

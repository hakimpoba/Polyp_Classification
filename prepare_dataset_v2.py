
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

ORIGINAL_MANIFEST = ROOT / "patient_groups.csv"
NEW_MANIFEST = ROOT / "data_pdf_sorted_manifest.csv"
OUTPUT_MANIFEST = ROOT / "dataset_v2_manifest.csv"

CLASSES = {"adenoma", "hyperplastic"}
SPLITS = {"train", "val"}

# The user confirmed these four new images belong to
# the same patient across both histology classes.
OVERLAP_FILENAMES = {
    "AD017PDF107.png",
    "AD018PDF107.png",
    "HP001P107.png",
    "HP002P107.png",
}


def fail(message):
    raise SystemExit(f"\nERROR: {message}")


def main():
    # 1. Confirm source manifests exist.
    for path in (ORIGINAL_MANIFEST, NEW_MANIFEST):
        if not path.is_file():
            fail(f"Required manifest not found: {path.name}")

    old = pd.read_csv(ORIGINAL_MANIFEST, dtype=str).fillna("")
    new = pd.read_csv(NEW_MANIFEST, dtype=str).fillna("")

    required_old = {"filename", "class", "patient_id", "split"}
    required_new = {
        "image_id", "relative_path", "class_label",
        "patient_group_id", "split"
    }

    if not required_old.issubset(old.columns):
        fail(f"Original manifest needs columns: {sorted(required_old)}")

    if not required_new.issubset(new.columns):
        fail(f"New manifest needs columns: {sorted(required_new)}")

    if len(old) != 101 or len(new) != 29:
        fail(
            f"Expected 101 original and 29 new rows; "
            f"found {len(old)} and {len(new)}."
        )

    # 2. Validate original manifest and locate each image
    # strictly within its assigned data_split folder.
    old_rows = []
    errors = []

    for _, row in old.iterrows():
        filename = row["filename"].strip()
        class_name = row["class"].strip().lower()
        split = row["split"].strip().lower()
        patient_id = row["patient_id"].strip()

        if not filename or not patient_id:
            errors.append(f"Original row has blank filename/patient ID: {row.to_dict()}")
            continue

        if class_name not in CLASSES:
            errors.append(f"Unexpected original class for {filename}: {class_name}")
            continue

        if split not in SPLITS:
            errors.append(f"Unexpected original split for {filename}: {split}")
            continue

        image_path = (
            ROOT / "data_split" / split / class_name / filename
        )

        if not image_path.is_file():
            errors.append(
                f"Original image missing from assigned split: "
                f"{image_path.relative_to(ROOT).as_posix()}"
            )
            continue

        old_rows.append({
            "image_id": filename,
            "relative_path": image_path.relative_to(ROOT).as_posix(),
            "class_label": class_name,
            "patient_group_id": f"ORIGINAL_{patient_id}",
            "source_dataset": "original_101",
            "original_split": split,
            "modality": "unknown",
        })

    # 3. Validate new manifest and locate each image.
    new_rows = []

    for _, row in new.iterrows():
        image_id = row["image_id"].strip()
        relative_path = row["relative_path"].strip().replace("\\", "/")
        class_name = row["class_label"].strip().lower()
        group_id = row["patient_group_id"].strip()

        if not image_id or not relative_path or not group_id:
            errors.append(f"New manifest has a blank required field: {row.to_dict()}")
            continue

        if class_name not in CLASSES:
            errors.append(f"Unexpected new class for {image_id}: {class_name}")
            continue

        # Ensure relative paths stay within this project.
        image_path = (ROOT / relative_path).resolve()

        if not image_path.is_relative_to(ROOT.resolve()):
            errors.append(f"Unsafe path in new manifest: {relative_path}")
            continue

        if not image_path.is_file():
            errors.append(f"New image missing: {relative_path}")
            continue

        if image_id in OVERLAP_FILENAMES:
            group_id = "NEW_OVERLAP_107"

        new_rows.append({
            "image_id": image_id,
            "relative_path": image_path.relative_to(ROOT).as_posix(),
            "class_label": class_name,
            "patient_group_id": group_id,
            "source_dataset": "new_29",
            "original_split": "unassigned",
            "modality": "white-light",
        })

    if errors:
        print("\nValidation problems found:")
        for error in errors:
            print(f"- {error}")
        fail(
            f"Found {len(errors)} issue(s). "
            "No combined manifest was written. Fix the issues first."
        )

    # 4. Combine and validate the final manifest.
    combined = pd.DataFrame(old_rows + new_rows)

    if len(combined) != 130:
        fail(f"Expected 130 combined rows, found {len(combined)}.")

    if combined["relative_path"].duplicated().any():
        duplicates = combined.loc[
            combined["relative_path"].duplicated(keep=False),
            "relative_path"
        ].tolist()
        fail(f"Duplicate image paths found: {duplicates}")

    expected_overlap = combined[
        combined["image_id"].isin(OVERLAP_FILENAMES)
    ]

    if len(expected_overlap) != 4:
        fail(
            "Expected all four confirmed overlap images, "
            f"found {len(expected_overlap)}."
        )

    if expected_overlap["patient_group_id"].nunique() != 1:
        fail("The four overlap images were not assigned one patient group.")

    # 5. Save only after every check succeeds.
    combined.to_csv(OUTPUT_MANIFEST, index=False)

    print("\n" + "=" * 55)
    print("DATASET V2 MANIFEST CREATED SUCCESSFULLY")
    print("=" * 55)
    print(f"Original images verified: {len(old_rows)}")
    print(f"New images verified:      {len(new_rows)}")
    print(f"Combined images:          {len(combined)}")
    print(
        "Distinct patient groups: ",
        combined["patient_group_id"].nunique()
    )

    print("\nImages by class:")
    print(combined["class_label"].value_counts().to_string())

    print("\nImages by source:")
    print(combined["source_dataset"].value_counts().to_string())

    print("\nNew cross-class overlap group:")
    print(
        expected_overlap[
            ["image_id", "class_label", "patient_group_id"]
        ].to_string(index=False)
    )

    print(f"\nPrivate manifest: {OUTPUT_MANIFEST}")
    print("\nOriginal images and prior experiment results were not changed.")
    print("Original split assignments are recorded for reference only.")
    print("A fresh patient-level split has NOT yet been created.")


if __name__ == "__main__":
    main()

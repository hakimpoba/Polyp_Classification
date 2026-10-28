
from pathlib import Path
import re
import pandas as pd

# Project-relative path: no machine-specific Windows path needed.
DATA_DIR = Path(__file__).resolve().parent / "data_pdf_sorted"
OUTPUT_CSV = Path(__file__).resolve().parent / "data_pdf_sorted_manifest.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# AD004PDF002: adenoma, polyp 004, patient 002
ADENOMA_PATTERN = re.compile(r"^AD(\d+)PDF(\d+)$", re.IGNORECASE)

# HP002P106: hyperplastic, polyp 002, patient 106
HYPERPLASTIC_PATTERN = re.compile(r"^HP(\d+)P(\d+)$", re.IGNORECASE)


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    rows = []
    errors = []

    for class_name in ("adenoma", "hyperplastic"):
        class_dir = DATA_DIR / class_name

        if not class_dir.is_dir():
            errors.append(f"Missing class folder: {class_dir}")
            continue

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            filename = image_path.stem

            if class_name == "adenoma":
                match = ADENOMA_PATTERN.fullmatch(filename)

                if not match:
                    errors.append(
                        f"Unexpected adenoma filename: {image_path.name}"
                    )
                    continue

                polyp_number, patient_number = match.groups()
                patient_group_id = f"ADSYS_PDF{patient_number}"

            else:
                match = HYPERPLASTIC_PATTERN.fullmatch(filename)

                if not match:
                    errors.append(
                        f"Unexpected hyperplastic filename: {image_path.name}"
                    )
                    continue

                polyp_number, patient_number = match.groups()
                patient_group_id = f"HPSYS_P{patient_number}"

            rows.append({
                "image_id": image_path.name,
                "relative_path": image_path.relative_to(
                    DATA_DIR.parent
                ).as_posix(),
                "class_label": class_name,
                "polyp_number": polyp_number,
                "patient_number": patient_number,
                "patient_group_id": patient_group_id,
                "split": "unassigned",
            })

    if errors:
        print("\nFilename or folder problems found:")
        for error in errors:
            print(f"- {error}")

        raise SystemExit(
            "\nFix these issues before creating the manifest."
        )

    if not rows:
        raise SystemExit("No valid images found.")

    df = pd.DataFrame(rows)

    duplicate_names = df["image_id"].duplicated(keep=False)
    if duplicate_names.any():
        print("Warning: repeated image filenames:")
        print(df.loc[duplicate_names, "image_id"].to_string(index=False))
        raise SystemExit("Resolve repeated filenames before continuing.")

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nManifest created successfully.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Total images: {len(df)}")
    print("\nImages by class:")
    print(df["class_label"].value_counts().to_string())
    print("\nPatient groups by class:")
    print(
        df.groupby("class_label")["patient_group_id"]
        .nunique()
        .to_string()
    )

    print(
        "\nIMPORTANT: AD and HP patient-number systems are kept separate."
    )
    print(
        "This does not prove that patients cannot overlap across classes."
    )


if __name__ == "__main__":
    main()

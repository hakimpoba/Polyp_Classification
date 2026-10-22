
from pathlib import Path
from collections import Counter, defaultdict
import csv
import shutil
import sys

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

SOURCE_DIR = PROJECT_DIR / "data" / "train"
CSV_FILE = PROJECT_DIR / "patient_groups.csv"
OUTPUT_DIR = PROJECT_DIR / "data_split"

CLASSES = {"adenoma", "hyperplastic"}
SPLITS = {"train", "val"}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def stop(message):
    print(f"\nERROR: {message}")
    print("No images were copied by this script.")
    sys.exit(1)


# 1. Check the input paths
if not CSV_FILE.is_file():
    stop(f"CSV file not found: {CSV_FILE}")

if not SOURCE_DIR.is_dir():
    stop(f"Source folder not found: {SOURCE_DIR}")

# 2. Read and validate the CSV
with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    required_columns = {"filename", "class", "patient_id", "split"}
    if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
        stop(
            "CSV must contain these columns: "
            "filename, class, patient_id, split"
        )

    rows = list(reader)

if not rows:
    stop("The CSV contains no image records.")

seen = set()
patient_splits = defaultdict(set)
errors = []

for line_number, row in enumerate(rows, start=2):
    filename = (row.get("filename") or "").strip()
    class_name = (row.get("class") or "").strip().lower()
    patient_id = (row.get("patient_id") or "").strip()
    split = (row.get("split") or "").strip().lower()

    if not filename:
        errors.append(f"CSV line {line_number}: missing filename")
        continue

    # Filenames must be simple filenames, not paths.
    if Path(filename).name != filename:
        errors.append(f"CSV line {line_number}: invalid filename {filename}")
        continue

    if class_name not in CLASSES:
        errors.append(f"CSV line {line_number}: invalid class {class_name}")
        continue

    if not patient_id:
        errors.append(f"CSV line {line_number}: missing patient_id")
        continue

    if split not in SPLITS:
        errors.append(f"CSV line {line_number}: split must be train or val")
        continue

    key = (class_name, filename)
    if key in seen:
        errors.append(f"Duplicate CSV entry: {class_name}/{filename}")
        continue
    seen.add(key)

    patient_splits[patient_id].add(split)

    source = SOURCE_DIR / class_name / filename
    if not source.is_file():
        errors.append(f"Image not found: {source}")

# 3. Check that no patient appears in both sets
for patient_id, assigned_splits in patient_splits.items():
    if len(assigned_splits) > 1:
        errors.append(
            f"Patient {patient_id} appears in both train and val."
        )

# 4. Check the CSV against all images in the source folders
actual_images = set()

for class_name in CLASSES:
    class_dir = SOURCE_DIR / class_name

    if not class_dir.is_dir():
        errors.append(f"Missing class folder: {class_dir}")
        continue

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            actual_images.add((class_name, path.name))

missing_entries = actual_images - seen
extra_entries = seen - actual_images

for class_name, filename in sorted(missing_entries):
    errors.append(f"Image missing from CSV: {class_name}/{filename}")

for class_name, filename in sorted(extra_entries):
    errors.append(f"CSV entry not found in source: {class_name}/{filename}")

if errors:
    print("\nValidation failed:")
    for error in errors:
        print(" -", error)
    stop("Fix the CSV or folder issues, then run the script again.")

# 5. Avoid accidentally overwriting an existing split
if OUTPUT_DIR.exists():
    existing_files = [
        p for p in OUTPUT_DIR.rglob("*") if p.is_file()
    ]
    if existing_files:
        stop(
            "data_split already contains files. "
            "Inspect it before proceeding; this script will not overwrite them."
        )

# 6. Create folders and copy images
for split in SPLITS:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True, exist_ok=True
        )

for row in rows:
    filename = row["filename"].strip()
    class_name = row["class"].strip().lower()
    split = row["split"].strip().lower()

    source = SOURCE_DIR / class_name / filename
    destination = OUTPUT_DIR / split / class_name / filename

    shutil.copy2(source, destination)

# 7. Print the final counts
counts = Counter(
    (row["split"].strip().lower(), row["class"].strip().lower())
    for row in rows
)

print("\nDataset split completed successfully!")
print(f"Total images: {len(rows)}")
print(f"Unique patient groups: {len(patient_splits)}")

for split in ("train", "val"):
    total = sum(counts[(split, c)] for c in CLASSES)
    print(f"\n{split.upper()}: {total} images")
    for class_name in ("adenoma", "hyperplastic"):
        print(f"  {class_name}: {counts[(split, class_name)]}")

print(f"\nOutput folder: {OUTPUT_DIR}")
print("Original images were not moved or deleted.")

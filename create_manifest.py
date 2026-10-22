
from pathlib import Path
import csv

PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

DATA_DIR = PROJECT_DIR / "data" / "train"
OUTPUT_CSV = PROJECT_DIR / "patient_groups.csv"

CLASSES = ["adenoma", "hyperplastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

rows = []

for class_name in CLASSES:
    class_dir = DATA_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {class_dir}")

    for image_path in sorted(class_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({
                "filename": image_path.name,
                "class": class_name,
                "patient_id": "",
                "split": ""
            })

with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "class", "patient_id", "split"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Images found: {len(rows)}")
print(f"Adenoma: {sum(r['class'] == 'adenoma' for r in rows)}")
print(f"Hyperplastic: {sum(r['class'] == 'hyperplastic' for r in rows)}")
print(f"Template created: {OUTPUT_CSV}")
print("Next: fill in patient_id and split in the CSV.")

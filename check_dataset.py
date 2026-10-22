from pathlib import Path
from PIL import Image

# DATA_DIR = Path("hymenoptera_data")
DATA_DIR = Path("data")

splits = ["train", "val"]
classes = ["adenoma", "hyperplastic"]
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

total_valid = 0

for split in splits:
    print(f"\n{'=' * 35}")
    print(f"DATASET SPLIT: {split.upper()}")
    print("=" * 35)

    for class_name in classes:
        class_dir = DATA_DIR / split / class_name

        if not class_dir.exists():
            print(f"\nMISSING FOLDER: {class_dir}")
            continue

        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

        valid = 0
        invalid = []

        for path in images:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid += 1
            except Exception:
                invalid.append(path.name)

        print(f"\nClass: {class_name}")
        print(f"Images found: {len(images)}")
        print(f"Valid images: {valid}")
        print(f"Unreadable images: {len(invalid)}")

        for name in invalid:
            print(f"  Cannot read: {name}")

        total_valid += valid

print(f"\nTotal valid images across train and val: {total_valid}")
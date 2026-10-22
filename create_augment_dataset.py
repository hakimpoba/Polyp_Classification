
from pathlib import Path
import random
import shutil

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Project paths
PROJECT_DIR = Path(
    r"C:\Users\ASUS\Documents\DEGREE\Projects\Polyp_Classification"
)

SOURCE_DIR = PROJECT_DIR / "data_split"
OUTPUT_DIR = PROJECT_DIR / "data_augmented"

AUGMENTATIONS_PER_IMAGE = 3
SEED = 42

random.seed(SEED)

# Mild transformations suitable for endoscopic images
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(
        degrees=5,
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
    ),
    transforms.RandomAffine(
        degrees=0,
        scale=(0.95, 1.05),
        interpolation=InterpolationMode.BILINEAR
    )
])

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    train_dir = SOURCE_DIR / "train"
    val_dir = SOURCE_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train and val folders inside {SOURCE_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename or move it before creating a fresh dataset."
        )

    # Check both class folders exist.
    classes = sorted(
        p.name for p in train_dir.iterdir() if p.is_dir()
    )

    if not classes:
        raise RuntimeError("No class folders found in the training folder.")

    for split in ("train", "val"):
        for class_name in classes:
            if not (SOURCE_DIR / split / class_name).is_dir():
                raise FileNotFoundError(
                    f"Missing folder: {SOURCE_DIR / split / class_name}"
                )

    original_count = 0
    augmented_count = 0

    # Training: preserve originals and create augmented copies.
    for class_name in classes:
        source_class = train_dir / class_name
        output_class = OUTPUT_DIR / "train" / class_name
        output_class.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in source_class.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        for image_path in images:
            image = load_rgb(image_path)

            # Copy original without modifying it.
            original_destination = output_class / image_path.name
            shutil.copy2(image_path, original_destination)
            original_count += 1

            # Create separately named augmented images.
            for index in range(1, AUGMENTATIONS_PER_IMAGE + 1):
                augmented_image = augmentation(image)

                output_name = (
                    f"{image_path.stem}_aug{index:02d}.jpg"
                )

                augmented_image.save(
                    output_class / output_name,
                    quality=95
                )
                augmented_count += 1

        print(f"Processed training class: {class_name}")

    # Validation: copy originals only; never augment validation images.
    val_count = 0

    for class_name in classes:
        source_class = val_dir / class_name
        output_class = OUTPUT_DIR / "val" / class_name
        output_class.mkdir(parents=True, exist_ok=True)

        for image_path in source_class.iterdir():
            if (
                image_path.is_file()
                and image_path.suffix.lower() in VALID_EXTENSIONS
            ):
                shutil.copy2(image_path, output_class / image_path.name)
                val_count += 1

    print("\n" + "=" * 45)
    print("DATASET AUGMENTATION COMPLETE")
    print("=" * 45)
    print(f"Original training files: {original_count}")
    print(f"Augmented training files: {augmented_count}")
    print(f"Total training files: {original_count + augmented_count}")
    print(f"Unchanged validation files: {val_count}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()

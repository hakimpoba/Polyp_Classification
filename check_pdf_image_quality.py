
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import io

from PIL import Image, ImageFile
import imagehash

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_pdf_sorted"

REPORT_CSV = PROJECT_DIR / "data_pdf_image_quality_report.csv"
DUPLICATES_CSV = PROJECT_DIR / "data_pdf_duplicate_candidates.csv"

CLASSES = ("adenoma", "hyperplastic")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

# Smaller threshold = more similar images required.
# A pHash distance of 6 is a review threshold, not proof
# that two images are duplicates.
NEAR_DUPLICATE_THRESHOLD = 6

# Do not silently accept truncated/corrupted image files.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path):
    """Calculate a file-level SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_image(path):
    """Check whether an image opens and collect metadata."""
    result = {
        "file_name": path.name,
        "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
        "class_label": path.parent.name,
        "status": "OK",
        "width": "",
        "height": "",
        "format": "",
        "mode": "",
        "file_size_bytes": path.stat().st_size,
        "sha256": "",
        "pixel_hash": "",
        "phash": "",
        "error": "",
    }

    try:
        # Verify the file structure.
        with Image.open(path) as image:
            image.verify()

        # Reopen after verify and fully decode the image.
        with Image.open(path) as image:
            image.load()

            result["width"], result["height"] = image.size
            result["format"] = image.format or ""
            result["mode"] = image.mode

            # Normalized RGB pixels help identify identical
            # decoded images even if file encoding differs.
            rgb = image.convert("RGB")
            result["pixel_hash"] = hashlib.sha256(
                str(rgb.size).encode("utf-8") + rgb.tobytes()
            ).hexdigest()

            # Perceptual hash identifies visually similar images.
            result["phash"] = str(imagehash.phash(rgb))

        result["sha256"] = sha256_file(path)

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    image_paths = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    if not image_paths:
        raise SystemExit("No supported image files found.")

    print(f"Found {len(image_paths)} image files.")
    print("Checking image readability and metadata...\n")

    rows = []

    for index, path in enumerate(image_paths, start=1):
        row = inspect_image(path)
        rows.append(row)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{path.name}: {row['status']}"
        )

    # --------------------------------------------------
    # SAVE IMAGE-LEVEL REPORT
    # --------------------------------------------------

    report_fields = list(rows[0].keys())

    with REPORT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=report_fields)
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # FIND EXACT AND NEAR-DUPLICATE CANDIDATES
    # --------------------------------------------------

    valid_rows = [row for row in rows if row["status"] == "OK"]
    pairs = []
    seen_pairs = set()

    def add_pair(row_a, row_b, reason, distance=""):
        key = tuple(sorted((
            row_a["relative_path"],
            row_b["relative_path"]
        )))

        if key in seen_pairs:
            return

        seen_pairs.add(key)

        pairs.append({
            "file_a": row_a["relative_path"],
            "class_a": row_a["class_label"],
            "file_b": row_b["relative_path"],
            "class_b": row_b["class_label"],
            "reason": reason,
            "phash_distance": distance,
            "review_status": "Needs manual review",
        })

    # Byte-identical files.
    by_sha = defaultdict(list)
    for row in valid_rows:
        by_sha[row["sha256"]].append(row)

    for group in by_sha.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    add_pair(group[i], group[j], "Identical file bytes")

    # Identical decoded RGB pixels, even if file bytes differ.
    by_pixels = defaultdict(list)
    for row in valid_rows:
        by_pixels[row["pixel_hash"]].append(row)

    for group in by_pixels.values():
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    add_pair(
                        group[i], group[j],
                        "Identical decoded RGB pixels"
                    )

    # Perceptually similar images.
    # Pairwise comparison is suitable for this small dataset.
    for i in range(len(valid_rows)):
        for j in range(i + 1, len(valid_rows)):
            row_a = valid_rows[i]
            row_b = valid_rows[j]

            hash_a = imagehash.hex_to_hash(row_a["phash"])
            hash_b = imagehash.hex_to_hash(row_b["phash"])
            distance = hash_a - hash_b

            if distance <= NEAR_DUPLICATE_THRESHOLD:
                add_pair(
                    row_a,
                    row_b,
                    "Near-duplicate candidate",
                    distance
                )

    pair_fields = [
        "file_a", "class_a", "file_b", "class_b",
        "reason", "phash_distance", "review_status"
    ]

    with DUPLICATES_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pairs)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("IMAGE QUALITY CHECK COMPLETE")
    print("=" * 55)

    print(f"Total image files: {len(rows)}")
    print(f"Readable images: {len(valid_rows)}")
    print(f"Unreadable/error images: {len(rows) - len(valid_rows)}")

    print("\nImages by class:")
    for class_name in CLASSES:
        count = sum(
            row["class_label"] == class_name for row in rows
        )
        print(f"  {class_name}: {count}")

    print("\nImage dimensions:")
    dimensions = defaultdict(int)

    for row in valid_rows:
        dimensions[(row["width"], row["height"])] += 1

    for (width, height), count in sorted(dimensions.items()):
        print(f"  {width} x {height}: {count} image(s)")

    print(f"\nDuplicate candidate pairs: {len(pairs)}")

    if pairs:
        for pair in pairs:
            print(
                f"  {pair['reason']}: "
                f"{Path(pair['file_a']).name} <-> "
                f"{Path(pair['file_b']).name}"
            )

    print("\nReports saved:")
    print(f"  {REPORT_CSV}")
    print(f"  {DUPLICATES_CSV}")

    print("\nIMPORTANT:")
    print("- No original images were changed or deleted.")
    print("- Similarity matches require manual review.")
    print("- Similar images do not prove same-patient identity.")


if __name__ == "__main__":
    main()

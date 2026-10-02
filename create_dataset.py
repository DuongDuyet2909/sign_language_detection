"""Extract 21 (x, y) hand landmarks per image and save a training dataset."""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path: Path):
    """Read images through Python to support non-ASCII paths on Windows."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def create_dataset(data_dir: Path, output: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {data_dir}. Run collect_imgs.py first."
        )
    data, labels = [], []
    skipped = 0
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        for class_id in ("0", "1", "2"):
            folder = data_dir / class_id
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = read_image(path)
                if image is None:
                    print(f"Skipping unreadable image: {path}")
                    skipped += 1
                    continue
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    skipped += 1
                    continue
                # Preserve the original feature representation used by model.p.
                landmarks = result.multi_hand_landmarks[0].landmark
                data.append([(point.x, point.y) for point in landmarks])
                labels.append(class_id)
    counts = Counter(labels)
    if any(counts[class_id] == 0 for class_id in ("0", "1", "2")):
        raise ValueError(
            "Every class needs a detected hand. Existing dataset was not changed."
        )
    with output.open("wb") as file:
        pickle.dump({"data": data, "labels": labels}, file)
    print(f"Saved {len(data)} samples to {output}")
    print(f"Samples per class: {dict(sorted(counts.items()))}; skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data.pkl")
    args = parser.parse_args()
    create_dataset(args.data_dir, args.output)

"""Collect mirrored webcam images for the A, B and L gesture classes."""

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent / "data"
LABELS = {0: "A", 1: "B", 2: "L"}


def collect_images(camera: int = 0, samples: int = 100) -> None:
    """Append images to each class folder without replacing old captures."""
    if samples < 1:
        raise ValueError("The number of samples must be positive.")
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}.")
        for class_id, label in LABELS.items():
            folder = DATA_DIR / str(class_id)
            folder.mkdir(parents=True, exist_ok=True)
            existing = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
            start = max(existing, default=-1) + 1
            print(f"Class {class_id}: {label}. Press Q to start; Esc to exit.")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Cannot read a frame from the camera.")
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame, f"Show {label} | Q: start | Esc: exit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Collect gestures", frame)
                key = cv2.waitKey(25) & 0xFF
                if key == 27:
                    return
                if key in (ord("q"), ord("Q")):
                    break
            for index in range(start, start + samples):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Cannot read a frame from the camera.")
                frame = cv2.flip(frame, 1)
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    raise RuntimeError("Cannot encode the captured image as JPEG.")
                # Python file I/O supports Vietnamese paths on Windows.
                (folder / f"{index}.jpg").write_bytes(encoded.tobytes())
                cv2.imshow("Collect gestures", frame)
                if cv2.waitKey(25) & 0xFF == 27:
                    return
            print(f"Saved {samples} new images for {label}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--samples", type=int, default=100, help="New images per class")
    args = parser.parse_args()
    collect_images(args.camera, args.samples)

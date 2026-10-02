"""Recognize A, B and L gesture classes from a mirrored webcam feed."""

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LABELS = {0: "A", 1: "B", 2: "L"}


def run(camera: int, model_path: Path) -> None:
    # Only load trusted pickle files.
    with model_path.open("rb") as file:
        model = pickle.load(file)["model"]
    if getattr(model, "n_features_in_", None) != 42:
        raise ValueError("This demo requires a model trained on 42 landmark features.")

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}. Try --camera 1.")
        with mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Cannot read a frame from the camera.")
                # Match the mirrored images used during data collection.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    coordinates = np.asarray(
                        [(p.x, p.y) for p in hand.landmark], dtype=np.float32
                    )
                    prediction = int(model.predict(coordinates.reshape(1, 42))[0])
                    label = LABELS[prediction]
                    drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style(),
                    )
                    x1 = max(0, min(width - 1, int(coordinates[:, 0].min() * width) - 10))
                    y1 = max(0, min(height - 1, int(coordinates[:, 1].min() * height) - 10))
                    x2 = max(0, min(width - 1, int(coordinates[:, 0].max() * width) + 10))
                    y2 = max(0, min(height - 1, int(coordinates[:, 1].max() * height) + 10))
                    text_y = y1 - 10 if y1 >= 45 else min(height - 10, y1 + 40)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(
                        frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3, cv2.LINE_AA,
                    )
                cv2.imshow("Hand Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "model.p")
    args = parser.parse_args()
    run(args.camera, args.model)

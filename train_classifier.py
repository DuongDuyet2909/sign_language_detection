"""Train and evaluate a Random Forest on 42 hand-landmark features."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def train_classifier(data_path: Path, output: Path) -> None:
    # Only load trusted pickle files: deserialization can execute Python code.
    with data_path.open("rb") as file:
        dataset = pickle.load(file)
    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"])
    if data.size == 0 or labels.ndim != 1 or len(data) != len(labels):
        raise ValueError("The dataset must contain matching, nonempty data and labels.")
    data = data.reshape(len(data), -1)
    if data.shape[1] != 42 or not np.isfinite(data).all():
        raise ValueError("Each sample must contain exactly 42 finite coordinates.")
    if set(labels.astype(str)) != {"0", "1", "2"}:
        raise ValueError("Expected labels 0 (A), 1 (B), 2 (L).")

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Frames from one recording can inflate this score. Test independent sessions.")
    with output.open("wb") as file:
        pickle.dump({"model": model}, file)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data.pkl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "model.p")
    args = parser.parse_args()
    train_classifier(args.data, args.output)

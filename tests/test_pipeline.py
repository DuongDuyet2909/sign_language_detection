"""Pipeline regression checks. No physical camera or GUI is opened."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import collect_imgs
import create_dataset
import inference_classifier
import train_classifier

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_unicode_image_path_and_corrupt_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ảnh bàn tay.png"
            source = np.full((24, 32, 3), 127, dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", source)
            self.assertTrue(ok)
            path.write_bytes(encoded.tobytes())
            np.testing.assert_array_equal(create_dataset.read_image(path), source)
            path.write_bytes(b"")
            self.assertIsNone(create_dataset.read_image(path))
            path.write_bytes(b"not an image")
            self.assertIsNone(create_dataset.read_image(path))

    def test_empty_extraction_preserves_existing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("0", "1", "2"):
                (root / label).mkdir()
            output = root / "data.pkl"
            output.write_bytes(b"original dataset")
            with self.assertRaises(ValueError):
                create_dataset.create_dataset(root, output)
            self.assertEqual(output.read_bytes(), b"original dataset")

    def test_wrong_feature_count_preserves_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.pkl"
            with dataset.open("wb") as file:
                pickle.dump({"data": [[0.0] * 84] * 3, "labels": ["0", "1", "2"]}, file)
            output = root / "model.p"
            output.write_bytes(b"original model")
            with self.assertRaisesRegex(ValueError, "42"):
                train_classifier.train_classifier(dataset, output)
            self.assertEqual(output.read_bytes(), b"original model")

    def test_bundled_model_and_training_roundtrip(self):
        with (ROOT / "data.pkl").open("rb") as file:
            dataset = pickle.load(file)
        features = np.asarray(dataset["data"], dtype=np.float32).reshape(-1, 42)
        with (ROOT / "model.p").open("rb") as file:
            bundled = pickle.load(file)["model"]
        self.assertEqual(bundled.n_features_in_, 42)
        self.assertEqual(set(bundled.classes_), {"0", "1", "2"})
        self.assertTrue(set(bundled.predict(features)).issubset({"0", "1", "2"}))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.p"
            train_classifier.train_classifier(ROOT / "data.pkl", output)
            with output.open("rb") as file:
                trained = pickle.load(file)["model"]
            self.assertEqual(trained.n_features_in_, 42)
            self.assertEqual(trained.n_estimators, 100)
            self.assertEqual(len(trained.predict(features)), len(features))

    def test_capture_read_failure_releases_camera(self):
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collect_imgs, "DATA_DIR", Path(directory)), \
                 patch.object(cv2, "VideoCapture", return_value=camera), \
                 patch.object(cv2, "destroyAllWindows") as close_windows:
                with self.assertRaisesRegex(RuntimeError, "read a frame"):
                    collect_imgs.collect_images()
                camera.release.assert_called_once()
                close_windows.assert_called_once()

    def test_inference_unavailable_camera_is_released(self):
        camera = MagicMock()
        camera.isOpened.return_value = False
        with patch.object(cv2, "VideoCapture", return_value=camera), \
             patch.object(cv2, "destroyAllWindows") as close_windows:
            with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
                inference_classifier.run(0, ROOT / "model.p")
            camera.release.assert_called_once()
            close_windows.assert_called_once()


if __name__ == "__main__":
    unittest.main()

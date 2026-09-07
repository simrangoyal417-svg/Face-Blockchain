"""Local face embeddings using OpenCV's SFace model."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import FaceDetectionError, detect_raw_faces


class FaceEncodingError(ValueError):
    """Raised when a face embedding cannot be generated."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER_MODEL = PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def _load_models(
    detector_model: str | Path,
    recognizer_model: str | Path,
    image_size: tuple[int, int],
) -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    detector_path = Path(detector_model)
    recognizer_path = Path(recognizer_model)
    if not recognizer_path.is_file():
        raise FaceEncodingError(
            f"Face recognition model not found: {recognizer_path}. "
            "Download face_recognition_sface_2021dec.onnx into models/."
        )
    try:
        detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", image_size, 0.9, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    except cv2.error as error:
        raise FaceEncodingError(f"Could not load face models: {error}") from error
    return detector, recognizer


def encode_faces(
    image_path: str | Path,
    detector_model: str | Path = DEFAULT_DETECTOR_MODEL,
    recognizer_model: str | Path = DEFAULT_RECOGNIZER_MODEL,
) -> list[np.ndarray]:
    """Return one SFace embedding for every detected face in an image."""
    path = Path(image_path)
    try:
        image, raw_detections = detect_raw_faces(path, detector_model)
        _, recognizer = _load_models(
            detector_model, recognizer_model, (image.shape[1], image.shape[0])
        )
        embeddings = []
        for face in raw_detections:
            aligned = recognizer.alignCrop(image, face)
            f1 = recognizer.feature(aligned)
            aligned_flip = cv2.flip(aligned, 1)
            f2 = recognizer.feature(aligned_flip)
            f = f1 + f2
            norm = np.linalg.norm(f)
            if norm > 0:
                f = f / norm
            embeddings.append(f)
        return embeddings
    except FaceDetectionError as error:
        raise FaceEncodingError(str(error)) from error
    except cv2.error as error:
        raise FaceEncodingError(f"Could not encode a face in '{path}': {error}") from error


def encode_face(
    image_path: str | Path,
    face_index: int = 0,
    detector_model: str | Path = DEFAULT_DETECTOR_MODEL,
    recognizer_model: str | Path = DEFAULT_RECOGNIZER_MODEL,
) -> np.ndarray:
    """Return one embedding, selecting ``face_index`` for multi-face images."""
    embeddings = encode_faces(image_path, detector_model, recognizer_model)
    if face_index < 0 or face_index >= len(embeddings):
        raise FaceEncodingError(
            f"Face index {face_index} is out of range; image contains {len(embeddings)} face(s)."
        )
    return embeddings[face_index]
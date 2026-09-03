import numpy as np
import pytest

from face.detector import FaceDetectionError, detect_faces
from face.matcher import compare_faces


def test_same_embedding_matches() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.99, 0.01]]))
    assert result["is_match"] is True


def test_different_embedding_does_not_match() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert result["is_match"] is False


def test_threshold_is_configurable() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.8, 0.6]]), threshold=0.9)
    assert result["similarity"] == pytest.approx(0.8)
    assert result["is_match"] is False


def test_missing_image_has_clear_error() -> None:
    with pytest.raises(FaceDetectionError, match="Image not found"):
        detect_faces("does-not-exist.jpg")


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        compare_faces(np.array([[1.0]]), np.array([[1.0]]), threshold=1.1)
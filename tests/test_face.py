import numpy as np
import pytest

from face.detector import FaceDetectionError, detect_faces
from face import matcher
from face.encoder import FaceEncodingError
from face.matcher import compare_faces, find_best_match


def test_same_embedding_matches() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.99, 0.01]]))
    assert result["is_match"] is True


def test_different_embedding_does_not_match() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert result["is_match"] is False


def test_threshold_is_configurable() -> None:
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.8, 0.6]]), threshold=0.9)
    assert result["similarity"] == pytest.approx(0.8)
    assert result["threshold_percent"] == 90.0
    assert result["is_match"] is False


def test_threshold_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.95")
    result = compare_faces(np.array([[1.0, 0.0]]), np.array([[0.9, 0.4358899]]))
    assert result["threshold"] == pytest.approx(0.95)
    assert result["decision"] == "different_person"


def test_highest_score_is_not_confirmed_without_threshold(monkeypatch) -> None:
    embeddings = {
        "input.jpg": [np.array([[1.0, 0.0]])],
        "candidate.jpg": [np.array([[0.8, 0.6]])],
    }
    monkeypatch.setattr(matcher, "encode_faces", lambda path: embeddings[str(path)])

    result = find_best_match("input.jpg", ["candidate.jpg"], threshold=0.9)

    assert result["best_candidate"] == "candidate.jpg"
    assert result["best_match"] is None
    assert result["decision"] == "no_reliable_match"


def test_no_face_candidate_has_explicit_decision(monkeypatch) -> None:
    def encode(path):
        if str(path) == "input.jpg":
            return [np.array([[1.0, 0.0]])]
        raise FaceEncodingError(f"No face detected in '{path}'")

    monkeypatch.setattr(matcher, "encode_faces", encode)
    result = find_best_match("input.jpg", ["candidate.jpg"])

    assert result["best_match"] is None
    assert result["decision"] == "no_face_detected"
    assert result["candidates"][0]["decision"] == "no_face_detected"


def test_multiple_faces_uses_strongest_candidate_face(monkeypatch) -> None:
    embeddings = {
        "input.jpg": [np.array([[1.0, 0.0]])],
        "group.jpg": [
            np.array([[0.0, 1.0]]),
            np.array([[0.99, 0.01]]),
        ],
    }
    monkeypatch.setattr(matcher, "encode_faces", lambda path: embeddings[str(path)])

    result = find_best_match("input.jpg", [{"image_path": "group.jpg"}], threshold=0.9)

    candidate_result = result["candidates"][0]
    assert candidate_result["face_count"] == 2
    assert candidate_result["selected_face_index"] == 1
    assert candidate_result["is_match"] is True


def test_missing_image_has_clear_error() -> None:
    with pytest.raises(FaceDetectionError, match="Image not found"):
        detect_faces("does-not-exist.jpg")


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        compare_faces(np.array([[1.0]]), np.array([[1.0]]), threshold=1.1)